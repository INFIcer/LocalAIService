"""能力适配器基类：统一 status / load / infer / unload / estimate 五件事。

新增一种能力 = 新增一个子类 + 在 ``localai.engines.build_registry()`` 里注册。
``localai.api`` 里没有任何 if-else 认能力名。
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..config import Config, get_config
from ..core.concurrency import ConcurrencyGate

logger = logging.getLogger("localai.engines")


class BaseAdapter(ABC):
    """所有能力适配器的共同父类。"""

    #: 能力名（注册表键 / 显存管家键 / /capabilities 键）
    name: str = ""
    #: 中文名（日志、/capabilities 展示）
    title: str = ""
    #: 主要模型名（日志、/capabilities 展示）
    model_name: str = ""
    #: 是否需要显存（False 的能力不进显存管家、不挂空闲看门狗）
    requires_gpu: bool = True
    #: 首次估算 / 校准下限（MB），由显存管家使用
    DEFAULT_ESTIMATE_MB: float = 0
    MIN_ESTIMATE_MB: float = 0
    #: 对外暴露的 HTTP 路径（/capabilities 用来自发现）
    ENDPOINTS: tuple[str, ...] = ()

    #: 是否实测确认「同一个已加载模型可以同时跑多次推理」。
    #: 默认 False：没实测过的能力一律强制串行（容量 1），宁可不快也不能崩。
    CAN_INFER_IN_PARALLEL: bool = False
    #: 并发能力说明（写进日志 / /capabilities，解释为什么被限制成 1）
    PARALLEL_NOTE: str = "未实测可并发，强制串行"

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or get_config()
        requested = int(self.cfg.jobs_parallel_for(self.name))
        effective = requested if self.CAN_INFER_IN_PARALLEL else 1
        self.infer_lock = ConcurrencyGate(
            self.name or type(self).__name__, capacity=effective
        )
        #: 配置里要求的并行度（可能被能力限制否决）
        self.parallel_requested = requested
        #: 实际生效的并行度
        self.parallel_capacity = effective
        self.parallel_note = self.PARALLEL_NOTE
        if requested > 1 and effective == 1:
            logger.warning(
                "能力 %s 配置了并行度 %d，但该模型未确认可并发（%s），强制按 1 串行",
                self.name,
                requested,
                self.PARALLEL_NOTE,
            )
        self._module: Any = None
        self._module_lock = threading.Lock()
        self._last_activity: float = 0.0

    # ------------------------------------------------------------------
    # 原生模块惰性导入
    # ------------------------------------------------------------------
    @abstractmethod
    def get_module(self) -> Any:
        """惰性 import 并缓存原生引擎模块。"""

    def module_imported(self) -> bool:
        return self._module is not None

    # ------------------------------------------------------------------
    # 统一状态
    # ------------------------------------------------------------------
    @abstractmethod
    def status(self) -> dict[str, Any]:
        """返回 ready / loaded / detail / model_dir 等状态（不触发加载）。"""

    def ready(self) -> bool:
        try:
            return bool(self.status().get("ready"))
        except Exception:  # noqa: BLE001
            return False

    def loaded(self) -> bool:
        module = self._module
        if module is None:
            return False
        try:
            return bool(module.is_loaded())
        except Exception:  # noqa: BLE001
            return False

    def load(self) -> None:
        """同步加载（调用方放线程池）。"""
        self.get_module().ensure_loaded()

    def unload(self) -> None:
        """同步卸载（可重复调用）。"""
        self.get_module().unload()

    def estimate_mb(self) -> int:
        return int(self.DEFAULT_ESTIMATE_MB)

    # ------------------------------------------------------------------
    # 空闲计时
    # ------------------------------------------------------------------
    def mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    def reset_activity(self) -> None:
        self._last_activity = 0.0

    def idle_seconds(self) -> Optional[float]:
        if not self._last_activity:
            return None
        return max(0.0, time.monotonic() - self._last_activity)

    def idle_unload_seconds(self) -> int:
        """本能力的空闲卸载阈值（秒，0=关闭）。"""
        return {
            "tts": self.cfg.idle_unload,
            "asr": self.cfg.asr_idle_unload,
            "whisper": self.cfg.whisper_idle_unload,
            "speaker": self.cfg.speaker_idle_unload,
            "separate": self.cfg.separate_idle_unload,
            "translate": self.cfg.translate_idle_unload,
        }.get(self.name, 0)

    # ------------------------------------------------------------------
    # 自描述（/capabilities）
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        try:
            status = self.status()
        except Exception as exc:  # noqa: BLE001
            status = {
                "ready": False,
                "loaded": False,
                "detail": f"状态读取失败：{type(exc).__name__}: {exc}",
            }
        idle = self.idle_seconds()
        return {
            "name": self.name,
            "title": self.title,
            "ready": bool(status.get("ready")),
            "loaded": bool(status.get("loaded")),
            "requires_gpu": self.requires_gpu,
            "model": status.get("model") or self.model_name or None,
            "model_dir": status.get("model_dir") or status.get("weights_dir"),
            "idle_unload_seconds": self.idle_unload_seconds(),
            "idle_seconds": None if idle is None else round(idle, 1),
            "estimate_mb": self.estimate_mb() if self.requires_gpu else 0,
            "parallel": {
                "requested": self.parallel_requested,
                "capacity": self.parallel_capacity,
                "can_infer_in_parallel": self.CAN_INFER_IN_PARALLEL,
                "note": self.parallel_note,
                "running": self.infer_lock.in_use,
                "peak": self.infer_lock.peak,
            },
            "detail": status.get("detail"),
            "endpoints": list(self.ENDPOINTS),
        }
