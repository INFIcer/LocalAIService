"""ASR 适配器：SenseVoiceSmall（按需加载 / 空闲卸载）。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..core.errors import AsrBackendError, InsufficientVramError
from ..core.vram import ensure_model_loaded, get_vram
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.asr")


class AsrAdapter(BaseAdapter):
    name = "asr"
    title = "语音识别（ASR）"
    model_name = "SenseVoiceSmall"
    requires_gpu = True
    DEFAULT_ESTIMATE_MB = 1200
    # 多次加载后实测增量会被 CUDA 缓存复用拉低；保留 900MB 下限。
    MIN_ESTIMATE_MB = 900
    ENDPOINTS = ("/asr", "/api/v1/asr", "/asr/status")

    @property
    def loader_hint(self) -> str:
        return "SenseVoiceSmall（ASR）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("asr_engine")
        return self._module

    def status(self) -> dict[str, Any]:
        try:
            module = self.get_module()
            status = module.model_status()
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": False,
                "detail": f"ASR 引擎不可用：{exc}",
                "missing": [],
                "loaded": False,
                "load_seconds": None,
                "model_dir": str(self.cfg.asr_model_dir),
            }

        missing = status.get("missing") or []
        if status.get("ready"):
            detail = "SenseVoice 模型文件已就绪" + (
                "（模型已加载）" if status.get("loaded") else "（首次请求时加载）"
            )
        else:
            detail = f"SenseVoice 模型文件不完整，缺少：{', '.join(missing)}"
        return {
            "ready": bool(status.get("ready")),
            "detail": detail,
            "missing": missing,
            "loaded": bool(status.get("loaded")),
            "load_seconds": status.get("load_seconds"),
            "load_error": status.get("load_error"),
            "model_dir": status.get("model_dir", str(self.cfg.asr_model_dir)),
            "device": status.get("device"),
        }

    async def infer(self, audio_path: str, language: str = "auto") -> dict[str, Any]:
        """识别一段音频；加载/推理都在本能力自己的锁里。"""
        module = self.get_module()
        vram = get_vram()
        # 先记一笔活动时间：即使加载要 10 秒，看门狗也不会在这期间误判空闲。
        self.mark_activity()
        try:
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    await ensure_model_loaded(self)
                    self.mark_activity()
                    result = await asyncio.to_thread(module.transcribe, str(audio_path), language)
                    vram.touch(self.name)
                    self.mark_activity()
                finally:
                    vram.end_use(self.name)
        except (InsufficientVramError, AsrBackendError):
            raise
        except Exception as exc:  # noqa: BLE001 - 统一成 503 JSON
            raise AsrBackendError(
                f"SenseVoice 识别失败：{type(exc).__name__}: {exc}",
                detail=str(exc),
            ) from exc
        return result
