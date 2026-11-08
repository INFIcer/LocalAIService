"""TTS 适配器：IndexTTS-2.5（原生常驻引擎），保留 ComfyUI 后端兼容。"""

from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..constants import REQUIRED_MODEL_FILES
from ..core.audio import archive_engine_wav
from ..core.errors import EngineBackendError, InsufficientVramError
from ..core.text import engine_emotion, normalize_language
from ..core.vram import ensure_model_loaded, get_vram
from ..core.voices import resolve_voice
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.tts")


@dataclass
class TtsResult:
    """一次合成结果；router 只消费这个结构。"""

    wav: bytes
    path: Path
    seed: int
    language: str
    comfy_filename: str = ""


def _missing_files(directory: Path) -> list[str]:
    """只检查主权重；辅助模型由原生引擎自己检查。"""
    missing: list[str] = []
    for name in REQUIRED_MODEL_FILES:
        path = directory / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return missing


class TtsAdapter(BaseAdapter):
    name = "tts"
    title = "语音合成（TTS）"
    model_name = "IndexTTS-2.5"
    requires_gpu = True
    DEFAULT_ESTIMATE_MB = 6400
    # TTS 的加载增量会被 WDDM 缓存复用低估，但推理峰值需要保守下限。
    MIN_ESTIMATE_MB = 6200
    ENDPOINTS = ("/tts", "/v1/audio/speech", "/api/v1/tts/tasks")
    # 实测并行 2 无吞吐收益（4 条短句 8.53s -> 8.06s），且 engine/engine.py
    # 内部还有 _INFER_LOCK、随机种子是模块级状态，强制串行。
    CAN_INFER_IN_PARALLEL = False
    PARALLEL_NOTE = (
        "IndexTTS 引擎内部有 _INFER_LOCK 且种子/参考音频是模块级状态；"
        "实测并行 2 无收益（8.53s→8.06s），强制串行"
    )

    @property
    def loader_hint(self) -> str:
        return "IndexTTS-2.5（TTS，常驻）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("engine")
        return self._module

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def _engine_status(self) -> dict[str, Any]:
        try:
            module = self.get_module()
            status = module.model_status()
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": False,
                "detail": f"引擎不可用：{exc}",
                "missing": [],
                "loaded": False,
                "load_seconds": None,
                "model_dir": str(self.cfg.model_dir),
            }

        missing = status.get("missing") or []
        if status.get("ready"):
            detail = "独立引擎模型文件已就绪" + (
                "（模型已加载）" if status.get("loaded") else "（首次请求时加载）"
            )
        else:
            detail = f"独立引擎模型文件不完整，缺少：{', '.join(missing)}"
        return {
            "ready": bool(status.get("ready")),
            "detail": detail,
            "missing": missing,
            "loaded": bool(status.get("loaded")),
            "load_seconds": status.get("load_seconds"),
            "load_error": status.get("load_error"),
            "model_dir": status.get("model_dir", str(self.cfg.model_dir)),
            "device": status.get("device"),
            "use_bf16": status.get("use_bf16"),
        }

    def _comfy_status(self) -> dict[str, Any]:
        comfy_missing = _missing_files(self.cfg.comfy_model_dir)
        weights_missing = _missing_files(self.cfg.model_dir)
        if not comfy_missing:
            detail = "模型文件已就绪"
        elif not weights_missing:
            detail = (
                "权重已下载完成，但节点读取的目录里没有这些文件；"
                f"需要把 {self.cfg.model_dir} 映射/复制到 {self.cfg.comfy_model_dir}（见 docs/INSTALL.md）"
            )
        else:
            detail = f"模型权重尚未就绪，{self.cfg.model_dir} 缺少 {len(weights_missing)} 个文件"
        return {
            "ready": not comfy_missing,
            "detail": detail,
            "model_dir": str(self.cfg.comfy_model_dir),
            "weights_dir": str(self.cfg.model_dir),
            "missing": comfy_missing,
            "weights_missing": weights_missing,
            "loaded": False,
        }

    def status(self) -> dict[str, Any]:
        if self.cfg.tts_backend == "comfy":
            return self._comfy_status()
        status = self._engine_status()
        return {
            "ready": status["ready"],
            "detail": status["detail"],
            "model_dir": status["model_dir"],
            "weights_dir": str(self.cfg.model_dir),
            "missing": status["missing"],
            "weights_missing": status["missing"],
            "loaded": status["loaded"],
            "load_seconds": status["load_seconds"],
            "load_error": status["load_error"],
            "device": status.get("device"),
            "use_bf16": status.get("use_bf16"),
        }

    def unload(self) -> None:
        if self.cfg.tts_backend != "engine":
            return
        super().unload()

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    async def infer(
        self,
        *,
        text: str,
        voice_name: str,
        speed: float,
        emotion: Optional[str],
        seed: Optional[int],
        language: Optional[str],
        client: Any = None,
    ) -> TtsResult:
        """完整合成链路（engine / comfy 两个后端都从这里进）。"""
        if self.cfg.tts_backend == "comfy":
            from .tts_comfy import synthesize_comfy

            return await synthesize_comfy(
                client=client,
                text=text,
                voice_name=voice_name,
                speed=speed,
                emotion=emotion,
                seed=seed,
                language=language,
            )
        return await self._synthesize_engine(
            text=text,
            voice_name=voice_name,
            speed=speed,
            emotion=emotion,
            seed=seed,
            language=language,
        )

    async def _synthesize_engine(
        self,
        *,
        text: str,
        voice_name: str,
        speed: float,
        emotion: Optional[str],
        seed: Optional[int],
        language: Optional[str],
    ) -> TtsResult:
        voice = resolve_voice(voice_name)
        actual_seed = seed if seed is not None else secrets.randbelow(2**32)
        actual_language = normalize_language(language, text)
        emo_vector, emo_text = engine_emotion(emotion)

        module = self.get_module()
        self.mark_activity()
        if emo_text and not getattr(module, "USE_QWEN_EMO", False):
            raise EngineBackendError(
                "engine 后端当前没有加载 Qwen 情感模型，不能按情感文本合成；"
                "请改用 8 维情感向量（例如 0.8,0,0,0,0,0,0,0），"
                "或设置 INDEX_TTS_QWEN_EMO=1 后重启服务。",
                status_code=400,
            )

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cfg.output_dir / f".engine_tmp_{uuid.uuid4().hex}.wav"
        vram = get_vram()
        try:
            # 推理是同步阻塞的 GPU 调用，放到线程里跑，避免卡住事件循环。
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    # 加载前统一腾显存；已加载时只 touch，不会重复加载。
                    await ensure_model_loaded(self)
                    self.mark_activity()
                    meta = await asyncio.to_thread(
                        module.synthesize,
                        text,
                        str(voice.path),
                        str(tmp_path),
                        actual_language,
                        speed,
                        emo_vector=emo_vector,
                        emo_text=emo_text,
                        seed=actual_seed,
                    )
                    vram.touch(self.name)
                finally:
                    vram.end_use(self.name)
            path = archive_engine_wav(tmp_path, voice_name, actual_seed)
        except Exception as exc:  # noqa: BLE001
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if isinstance(exc, (InsufficientVramError, EngineBackendError)):
                raise
            raise EngineBackendError(
                f"独立引擎合成失败：{type(exc).__name__}: {exc}",
                status_code=503,
            ) from exc

        self.mark_activity()
        wav = path.read_bytes()
        logger.info(
            "合成完成（engine）：voice=%s seed=%s language=%s bytes=%d 音频=%.2fs 耗时=%.2fs -> %s",
            voice_name,
            actual_seed,
            actual_language,
            len(wav),
            float(meta.get("duration") or 0.0),
            float(meta.get("elapsed") or 0.0),
            path,
        )
        return TtsResult(
            wav=wav,
            path=path,
            seed=actual_seed,
            language=actual_language,
            comfy_filename="",
        )
