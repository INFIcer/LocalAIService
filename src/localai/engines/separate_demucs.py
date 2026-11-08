"""人声分离适配器：Demucs（htdemucs / htdemucs_6s）。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ..constants import SEPARATE_MODELS
from ..core.errors import InsufficientVramError, SeparateBackendError
from ..core.vram import ensure_model_loaded, get_vram
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.separate")


class SeparateAdapter(BaseAdapter):
    name = "separate"
    title = "人声分离（Demucs）"
    model_name = "Demucs htdemucs"
    requires_gpu = True
    DEFAULT_ESTIMATE_MB = 1000
    # WDDM 下加载增量可能只统计到 140MB，但推理峰值约 0.9GB。
    MIN_ESTIMATE_MB = 800
    ENDPOINTS = ("/separate", "/api/v1/separate", "/separate/status")

    @property
    def loader_hint(self) -> str:
        return "Demucs htdemucs（人声分离）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("separator_engine")
        return self._module

    def status(self, model_name: str = "htdemucs") -> dict[str, Any]:
        try:
            module = self.get_module()
            status = module.model_status(model_name)
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": False,
                "detail": f"人声分离引擎不可用：{exc}",
                "missing": [],
                "loaded": False,
                "model": "htdemucs",
                "loaded_model": None,
                "load_seconds": None,
                "weights_dir": None,
                "available": [],
            }

        return {
            "ready": bool(status.get("ready")),
            "detail": status.get("detail"),
            "missing": status.get("missing") or [],
            "loaded": bool(status.get("loaded")),
            "model": status.get("model") or model_name,
            "loaded_model": status.get("loaded_model"),
            "load_seconds": status.get("load_seconds"),
            "load_error": status.get("load_error"),
            "weights_dir": status.get("weights_dir"),
            "available": status.get("available") or [],
            "device": status.get("device"),
            "segment": status.get("segment"),
            "shifts": status.get("shifts"),
        }

    def available_models(self) -> list[str]:
        try:
            models = self.get_module().available_models()
        except Exception:  # noqa: BLE001
            return []
        return [name for name in SEPARATE_MODELS if name in models]

    async def infer(
        self,
        *,
        input_wav: str,
        out_dir: str,
        model_name: str,
        stems: list[str],
    ) -> dict[str, Any]:
        """分离并落盘；返回 ``{"result": ..., "released_models": [...]}``。"""
        module = self.get_module()
        # 权重不全就直接报错，不要为了一个注定失败的请求先卸掉别的模型。
        model_info = module.model_status(model_name)
        if not model_info.get("ready"):
            raise SeparateBackendError(
                model_info.get("detail") or f"Demucs 模型 {model_name} 未就绪",
                status_code=503,
            )

        vram = get_vram()
        released_models: list[str] = []
        try:
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    released_models = await ensure_model_loaded(
                        self,
                        loader=lambda: module.ensure_loaded(model_name),
                    )
                    self.mark_activity()
                    result = await asyncio.to_thread(
                        module.separate,
                        str(input_wav),
                        str(out_dir),
                        model_name,
                        stems,
                    )
                    vram.touch(self.name)
                    self.mark_activity()
                finally:
                    vram.end_use(self.name)
        except (InsufficientVramError, SeparateBackendError):
            raise
        except module.SeparatorError as exc:
            raise SeparateBackendError(f"Demucs 分离失败：{exc}", detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - 统一成 503 JSON
            raise SeparateBackendError(
                f"Demucs 分离失败：{type(exc).__name__}: {exc}",
                detail=str(exc),
            ) from exc
        return {"result": result, "released_models": released_models}
