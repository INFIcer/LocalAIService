"""ASR 适配器：faster-whisper（第二识别引擎，原生时间戳）。

和 ``AsrAdapter``（SenseVoice）并列注册为能力 ``whisper``，独立挂进显存管家：
- 第一次请求才加载，空闲看门狗按 ``LOCALAI_WHISPER_IDLE_UNLOAD`` 卸载；
- 可与 TTS / SenseVoice 按 LRU 互相腾显存（``vram.ensure_capacity``）；
- 显存初值按档位给（tiny 150 / base 250 / small 600 / medium 1600 /
  large-v3 3500 MB），之后由显存管家按实测增量校准。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import get_config
from ..constants import WHISPER_ESTIMATE_MB
from ..core.errors import AsrBackendError, InsufficientVramError
from ..core.vram import ensure_model_loaded, get_vram
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.whisper")


class WhisperAdapter(BaseAdapter):
    """faster-whisper 适配器（能力名 ``whisper``）。"""

    name = "whisper"
    title = "语音识别（faster-whisper）"
    model_name = "faster-whisper"
    requires_gpu = True
    #: 初值按档位在 ``estimate_mb()`` 里算，这里只给一个保守兜底。
    DEFAULT_ESTIMATE_MB = 600
    #: 校准下限：tiny 实测也有一百多 MB，留 100 避免被校准成 0。
    MIN_ESTIMATE_MB = 100
    MAX_ESTIMATE_MB: float = 4000
    ENDPOINTS = ("/asr", "/asr/segments", "/asr/status")
    PARALLEL_NOTE = "CTranslate2 推理未实测可并发，强制串行（容量 1）"

    def __init__(self, cfg: Any = None) -> None:
        super().__init__(cfg or get_config())

    @property
    def loader_hint(self) -> str:
        return f"faster-whisper-{self.cfg.whisper_model}（ASR，原生时间戳）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("whisper_engine")
        return self._module

    # ------------------------------------------------------------------
    # 状态 / 显存估算
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        size = self.cfg.whisper_model
        try:
            module = self.get_module()
            status = module.model_status(size)
        except Exception as exc:  # noqa: BLE001 - health 不能让整个接口挂掉
            return {
                "ready": False,
                "detail": f"whisper 引擎不可用：{exc}",
                "missing": [],
                "loaded": False,
                "load_seconds": None,
                "model_dir": str(self.cfg.whisper_weights_dir(size)),
                "model": f"faster-whisper-{size}",
                "device": self.cfg.whisper_device,
                "compute_type": self.cfg.whisper_compute,
            }

        if status.get("ready"):
            detail = f"faster-whisper-{size} 权重已就绪" + (
                "（模型已加载）" if status.get("loaded") else "（首次请求时加载）"
            )
        else:
            detail = (
                f"faster-whisper-{size} 权重未下载：{status.get('model_dir')}"
                "（首次请求时从 ModelScope 下载，失败再退 HF 镜像）"
            )
        return {
            "ready": bool(status.get("ready")),
            "detail": detail,
            "missing": status.get("missing") or [],
            "loaded": bool(status.get("loaded")),
            "load_seconds": status.get("load_seconds"),
            "load_error": status.get("load_error"),
            "model_dir": status.get("model_dir"),
            "model": status.get("model"),
            "device": status.get("device"),
            "compute_type": status.get("compute_type"),
            "size": status.get("size"),
            "size_mb": status.get("size_mb"),
            "package": status.get("package"),
            "ctranslate2": status.get("ctranslate2"),
            "load_device": status.get("device"),
        }

    def estimate_mb(self) -> int:
        """按当前档位给显存初值（之后由管家校准）。"""
        return int(WHISPER_ESTIMATE_MB.get(self.cfg.whisper_model, self.DEFAULT_ESTIMATE_MB))

    def load(self) -> None:
        """同步加载当前档位（调用方放线程池）。"""
        self.get_module().ensure_loaded(
            self.cfg.whisper_model,
            self.cfg.whisper_device,
            self.cfg.whisper_compute,
        )

    def loaded(self) -> bool:
        module = self._module
        if module is None:
            return False
        try:
            return bool(module.is_loaded(self.cfg.whisper_model))
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    async def _run(self, audio_path: str, language: str, *, timestamps: bool) -> dict[str, Any]:
        """统一的加载 + 推理流程（显存腾挪 / 活动时间与 SenseVoice 一致）。"""
        module = self.get_module()
        vram = get_vram()
        self.mark_activity()
        try:
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    await ensure_model_loaded(self)
                    self.mark_activity()
                    result = await asyncio.to_thread(
                        module.transcribe,
                        str(audio_path),
                        self.cfg.whisper_model,
                        language,
                        timestamps,
                        True,
                    )
                    vram.touch(self.name)
                    self.mark_activity()
                finally:
                    vram.end_use(self.name)
        except (InsufficientVramError, AsrBackendError):
            raise
        except Exception as exc:  # noqa: BLE001 - 统一成 503 JSON
            raise AsrBackendError(
                f"faster-whisper 识别失败：{type(exc).__name__}: {exc}",
                detail=str(exc),
            ) from exc
        result["engine"] = self.name
        return result

    async def infer(
        self, audio_path: str, language: str = "auto", *, timestamps: bool = False
    ) -> dict[str, Any]:
        """文本识别（``timestamps=False`` 时不返回 segments，省一点序列化）。"""
        return await self._run(audio_path, language, timestamps=timestamps)

    async def infer_segments(self, audio_path: str, language: str = "auto") -> dict[str, Any]:
        """带时间戳的分段识别（whisper 用原生 segment 时间戳）。"""
        return await self._run(audio_path, language, timestamps=True)


__all__ = ["WhisperAdapter"]
