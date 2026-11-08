"""声纹适配器：campplus 192 维向量（登记 / 识别 / 未知聚类）。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..core.errors import InsufficientVramError, SpeakerBackendError
from ..core.vram import ensure_model_loaded, get_vram
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.speaker")


class SpeakerAdapter(BaseAdapter):
    name = "speaker"
    title = "声纹识别（campplus）"
    model_name = "campplus"
    requires_gpu = True
    DEFAULT_ESTIMATE_MB = 300
    ENDPOINTS = (
        "/speakers",
        "/speakers/status",
        "/speakers/unknown",
        "/speakers/enroll",
        "/speakers/identify",
        "/speakers/{name}",
    )

    @property
    def loader_hint(self) -> str:
        return "campplus（声纹，在已有 CUDA context 上增量很小）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("speaker_engine")
        return self._module

    def _threshold(self) -> float:
        try:
            return float(getattr(self.get_module(), "KNOWN_THRESHOLD", 0.65))
        except Exception:  # noqa: BLE001
            return 0.65

    def status(self) -> dict[str, Any]:
        threshold = 0.65
        try:
            module = self.get_module()
            status = module.model_status()
            threshold = float(getattr(module, "KNOWN_THRESHOLD", threshold))
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": False,
                "detail": f"声纹引擎不可用：{exc}",
                "missing": [],
                "loaded": False,
                "load_seconds": None,
                "model_dir": str(self.cfg.speaker_model_dir),
                "speakers_dir": str(self.cfg.speakers_dir),
                "threshold": threshold,
            }

        missing = status.get("missing") or []
        if status.get("ready"):
            detail = "声纹模型 campplus 已就绪" + (
                "（模型已加载）" if status.get("loaded") else "（首次请求时加载）"
            )
        else:
            detail = f"声纹模型文件不完整，缺少：{', '.join(missing)}"
        return {
            "ready": bool(status.get("ready")),
            "detail": detail,
            "missing": missing,
            "loaded": bool(status.get("loaded")),
            "load_seconds": status.get("load_seconds"),
            "load_error": status.get("load_error"),
            "model_dir": status.get("model_dir", str(self.cfg.speaker_model_dir)),
            "speakers_dir": str(self.cfg.speakers_dir),
            "threshold": threshold,
        }

    async def call(self, func_name: str, *args: Any) -> Any:
        """在声纹推理锁里跑一个原生引擎函数（同步 GPU 调用放线程池）。"""
        module = self.get_module()
        vram = get_vram()
        self.mark_activity()
        try:
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    await ensure_model_loaded(self)
                    self.mark_activity()
                    result = await asyncio.to_thread(getattr(module, func_name), *args)
                    vram.touch(self.name)
                    self.mark_activity()
                finally:
                    vram.end_use(self.name)
        except (InsufficientVramError, SpeakerBackendError):
            raise
        except ValueError as exc:  # 名字/参数不合法：算客户端错误
            raise SpeakerBackendError(
                f"声纹参数不合法：{exc}",
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:  # noqa: BLE001 - 统一成声纹错误
            raise SpeakerBackendError(
                f"声纹处理失败（{func_name}）：{type(exc).__name__}: {exc}",
                detail=str(exc),
            ) from exc
        return result

    async def identify_for(self, audio_path: str) -> dict[str, Any]:
        """算声纹并归到「已知某人」或「未知-N」；供 /asr 使用。"""
        result = await self.call("speaker_for", str(audio_path))
        return {
            "name": result.get("name"),
            "score": round(float(result.get("score") or 0.0), 4),
            "known": bool(result.get("known")),
        }
