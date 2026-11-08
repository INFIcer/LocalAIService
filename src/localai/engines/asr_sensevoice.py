"""ASR 适配器：SenseVoiceSmall（按需加载 / 空闲卸载 / 可控并发）。"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ..core.errors import AsrBackendError, InsufficientVramError
from ..core.vram import ensure_model_loaded, get_vram
from ..config import get_config
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
    #: 实际是否可并发由「本次配置的容量」决定：容量 1 时强制串行，
    #: 容量 2~4 时引擎内部信号量也放行同样数量。详见 docs/SERVICE_API.md。
    PARALLEL_NOTE = (
        "容量 1 默认串行；容量 >1 时 engine/asr_engine.py 的信号量同步放开，"
        "实测 2 路稳定且吞吐提升，默认仍保持 1"
    )

    def __init__(self, cfg: Any = None) -> None:
        """先对齐原生引擎的并发容量，再让基类构造服务层闸门。

        配置优先级（从高到低）：
        1. ``LOCALAI_ASR_PARALLEL``（服务层规范名，Config 会同时投影到调度器）；
        2. ``LOCALAI_JOBS_PARALLEL_ASR``（兼容旧名，只设它也能生效）。

        ``Config.from_env()`` 已经把两者归一成 ``cfg.jobs_parallel_asr``，这里
        再写回 ``LOCALAI_ASR_PARALLEL``，确保原生引擎 ``import`` 时看到的是
        同一个数字；适配器闸门也使用同一个值，避免两处配置打架。
        """
        cfg = cfg or get_config()
        requested = max(1, min(4, int(cfg.jobs_parallel_for("asr") or 1)))
        os.environ["LOCALAI_ASR_PARALLEL"] = str(requested)
        self._asr_requested_parallel = requested
        # 把原生引擎的闸门也换到同一容量。模块只 import 标准库，不加载模型；
        # 若模块已被提前 import，configure_parallel() 会在无推理时安全换挡。
        try:
            import_engine_module("asr_engine").configure_parallel(requested)
        except Exception as exc:  # noqa: BLE001 - 不因引擎 import 失败拖垮服务启动
            logger.warning("启动时对齐 ASR 并发容量失败（稍后首次推理会重试）：%s", exc)
        super().__init__(cfg)
        if self.parallel_capacity > 1:
            self.parallel_note = (
                f"容量 {self.parallel_capacity}：引擎信号量与调度器同值，"
                f"允许 {self.parallel_capacity} 路 SenseVoice 推理"
            )

    @property
    def CAN_INFER_IN_PARALLEL(self) -> bool:  # type: ignore[override]
        """容量 >1 才允许并发；容量 1 与旧版完全一致（串行）。"""
        return int(getattr(self, "_asr_requested_parallel", 1)) > 1

    @property
    def loader_hint(self) -> str:
        return "SenseVoiceSmall（ASR）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                module = import_engine_module("asr_engine")
                # 二次兜底：即使构造时 import 失败过，首次取模块也把容量对齐。
                module.configure_parallel(self._asr_requested_parallel)
                self._module = module
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
                "parallel": {
                    "capacity": self.parallel_capacity,
                    "in_use": self.infer_lock.in_use,
                    "peak": self.infer_lock.peak,
                    "env": "LOCALAI_ASR_PARALLEL",
                },
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
            "parallel": status.get("parallel")
            or {
                "capacity": self.parallel_capacity,
                "in_use": self.infer_lock.in_use,
                "peak": self.infer_lock.peak,
                "env": "LOCALAI_ASR_PARALLEL",
            },
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
