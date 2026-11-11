"""统一显存管家（从 ``engine/vram_manager.py`` 迁移过来，逻辑不变）。

在 8GB 显卡上，让 TTS / ASR / 声纹 / 人声分离四个能力 **按需加载、按 LRU
卸载、按实测校准估算**，避免任何一个能力把整张卡顶死。

设计要点
--------
1. **真实显存优先**：能用 ``torch.cuda.mem_get_info()`` 就用它；拿不到时
   退回解析 ``nvidia-smi``，再拿不到才用估算值。
2. **LRU 腾地方**：``ensure_capacity(name)`` 只卸「其它已经加载、且当前没有
   请求在用」的模型，从最久没用过的开始卸；每卸一个都重新查一次真实空闲。
3. **绝不硬卸正在推理的模型**：每个能力注册时带一把 ``asyncio.Lock``；
   卸载动作必须拿到那把锁，推理期间锁被占着，管家会跳过它。
4. **估算自动校准**：加载前后各查一次空闲显存，差值就是本次真实增量；
   每次最多向实测值修正 ±20%，并对 TTS / Demucs / ASR 保留保守下限。
   结果持久化到 ``<HOME>/logs/vram_estimates.json``，重启后继续使用。
5. **线程/协程安全**：内部用 ``threading.RLock`` 保护状态，用
   ``asyncio.Lock`` 串行化「查显存 + 卸模型」这类会互相影响的动作。

本模块 **不会在 import 时加载模型，也不会主动初始化 CUDA context**。
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config, get_config
from .concurrency import exclusive
from .errors import InsufficientVramError, is_cuda_oom

logger = logging.getLogger("localai.vram")

# 8G 卡（RTX 3070）的标称总显存，仅用于「所有探测都失败」时的兜底估算。
FALLBACK_TOTAL_MB = 8192.0
# Windows 桌面/驱动的基础占用，兜底估算时减掉，宁可保守也不要盲目放行。
FALLBACK_BASE_MB = 800.0


def _async_lock(lock: Any) -> Any:
    """卸载/加载时需要的「独占」上下文。

    ``adapter.infer_lock`` 现在是容量可调的 :class:`ConcurrencyGate`：
    ``async with gate`` 只占 1 个槽位，容量 > 1 时并不独占；卸载必须用
    ``gate.exclusive()`` 等所有推理退出。旧式 ``asyncio.Lock`` 原样返回。
    """
    return exclusive(lock)


@dataclass
class ModelEntry:
    """一个可卸载能力的注册信息。"""

    name: str
    unload_fn: Callable[[], Any]
    estimate_mb: float
    loader_hint: str = ""
    is_loaded_fn: Optional[Callable[[], bool]] = None
    infer_lock: Any = None
    # 估算下限：个别大模型（TTS）的加载增量会被 WDDM 缓存复用低估，
    # 但推理峰值仍然很高，不能让校准把容量估算降到危险值。
    min_estimate_mb: float = 0.0
    # 估算上限：防止一次错误的实测把估算撑到无法加载（例如 WDDM 把别的进程占用算进来）。
    max_estimate_mb: float = 0.0

    # 运行状态
    last_used: Optional[float] = None          # time.time()，用于展示
    last_used_mono: Optional[float] = None     # time.monotonic()，用于 LRU
    active: int = 0                            # 正在使用这个能力的请求数
    measured_mb: Optional[float] = None        # 最近一次实测增量
    samples: int = 0

    def is_loaded(self) -> bool:
        if self.is_loaded_fn is None:
            return False
        try:
            return bool(self.is_loaded_fn())
        except Exception:  # noqa: BLE001 - 状态查询失败按未加载处理
            return False


class VramManager:
    """四个 GPU 能力的统一显存管家。"""

    def __init__(
        self,
        *,
        estimates_path: Path,
        margin_mb: int = 512,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.estimates_path = Path(estimates_path)
        self.margin_mb = max(0, int(margin_mb))
        self.log = logger_ or logger
        self._entries: dict[str, ModelEntry] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._op_lock = asyncio.Lock()
        self._health_cache: Optional[tuple[float, int, int]] = None
        self._saved: dict[str, dict[str, Any]] = {}
        self._load_saved_estimates()

    # ------------------------------------------------------------------
    # 注册 / 状态
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        unload_fn: Callable[[], Any],
        estimate_mb: float,
        loader_hint: str = "",
        *,
        is_loaded_fn: Optional[Callable[[], bool]] = None,
        infer_lock: Any = None,
        min_estimate_mb: float = 0.0,
        max_estimate_mb: float = 0.0,
    ) -> None:
        """注册一个可卸载能力。

        参数
        ----
        name:          能力名（tts / asr / speaker / separate）。
        unload_fn:     同步卸载函数，必须可重复调用。
        estimate_mb:   初始显存估算（MB）；持久化文件里有校准值时优先用校准值。
        loader_hint:   给 /vram 展示的说明。
        is_loaded_fn:  查询是否已加载的函数；不传则视为从未加载。
        infer_lock:    该能力的 asyncio.Lock；卸载时会先拿这把锁。
        min_estimate_mb: 校准下限。
        """
        if not name:
            raise ValueError("能力名不能为空")
        saved = self._saved.get(name) or {}
        estimate = float(saved.get("estimate_mb") or estimate_mb)
        if max_estimate_mb > 0:
            estimate = min(estimate, float(max_estimate_mb))
        entry = ModelEntry(
            name=name,
            unload_fn=unload_fn,
            estimate_mb=max(float(min_estimate_mb), estimate),
            loader_hint=loader_hint,
            is_loaded_fn=is_loaded_fn,
            infer_lock=infer_lock,
            min_estimate_mb=float(min_estimate_mb),
            max_estimate_mb=float(max_estimate_mb),
            measured_mb=(
                float(saved["last_measured_mb"])
                if saved.get("last_measured_mb") is not None
                else None
            ),
            samples=int(saved.get("samples") or 0),
        )
        with self._lock:
            self._entries[name] = entry
            if name not in self._order:
                self._order.append(name)
        self.log.info(
            "显存管家注册：%s，估算 %d MB%s",
            name,
            round(entry.estimate_mb),
            "（来自持久化校准）" if saved else "",
        )

    def estimate_mb(self, name: str) -> int:
        """返回某能力当前使用的估算值（MB，已按实测校准）。"""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(f"未注册的显存能力：{name}")
            return int(round(entry.estimate_mb))

    def estimates(self) -> dict[str, int]:
        """返回所有能力当前使用的估算值。"""
        with self._lock:
            return {
                name: int(round(entry.estimate_mb))
                for name, entry in self._entries.items()
            }

    def loaded_names(self) -> list[str]:
        with self._lock:
            entries = list(self._entries.values())
        return [entry.name for entry in entries if entry.is_loaded()]

    # ------------------------------------------------------------------
    # 使用中标记（防止卸载正在推理的模型）
    # ------------------------------------------------------------------

    def begin_use(self, name: str) -> None:
        """请求开始使用某能力：+1 活跃计数，并刷新 LRU 时间。"""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry.active += 1
            entry.last_used = time.time()
            entry.last_used_mono = time.monotonic()

    def end_use(self, name: str) -> None:
        """请求结束：-1 活跃计数，并刷新「刚用过」时间。"""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry.active = max(0, entry.active - 1)
            entry.last_used = time.time()
            entry.last_used_mono = time.monotonic()

    def touch(self, name: str) -> None:
        """标记「刚用过」，用于 LRU。"""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry.last_used = time.time()
            entry.last_used_mono = time.monotonic()

    # ------------------------------------------------------------------
    # 显存查询
    # ------------------------------------------------------------------

    def _read_memory_torch(self, *, initialize: bool) -> Optional[tuple[float, float]]:
        """尝试用 torch 读真实显存；initialize=False 时不主动初始化 CUDA。"""
        torch = sys.modules.get("torch")
        if torch is None:
            if not initialize:
                return None
            try:
                import torch as torch_module  # noqa: PLC0415 - 故意延迟

                torch = torch_module
            except Exception as exc:  # noqa: BLE001
                self.log.debug("导入 torch 失败，改走 nvidia-smi：%s", exc)
                return None
        try:
            if not torch.cuda.is_available():
                return None
            if not initialize and not torch.cuda.is_initialized():
                return None
            free, total = torch.cuda.mem_get_info()
            return float(free) / (1024 * 1024), float(total) / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("torch.cuda.mem_get_info 失败：%s", exc)
            return None

    def _read_memory_nvidia_smi(self) -> Optional[tuple[float, float]]:
        """解析 nvidia-smi；拿不到就返回 None。"""
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=creationflags,
            )
            first = next((line for line in output.splitlines() if line.strip()), "")
            free_text, total_text = [part.strip() for part in first.split(",")[:2]]
            return float(free_text), float(total_text)
        except Exception as exc:  # noqa: BLE001
            self.log.debug("nvidia-smi 查询失败：%s", exc)
            return None

    def _fallback_memory(self) -> tuple[float, float]:
        """所有探测都失败：按已加载模型的估算值倒推一个保守空闲值。"""
        with self._lock:
            used = sum(
                entry.estimate_mb
                for entry in self._entries.values()
                if entry.is_loaded()
            )
        free = max(0.0, FALLBACK_TOTAL_MB - FALLBACK_BASE_MB - used)
        return free, FALLBACK_TOTAL_MB

    def read_memory(self, *, initialize: bool = False) -> tuple[float, float]:
        """返回 (空闲 MB, 总显存 MB)。

        特例：视觉模型跑在 Vulkan 子进程里，Windows 上 CUDA 的
        ``mem_get_info`` 经常看不到这块占用，会把空闲报高；这种时候
        强制用 nvidia-smi 的真实数据。
        """
        with self._lock:
            vision_loaded = any(
                entry.name == "vision" and entry.is_loaded()
                for entry in self._entries.values()
            )
        if vision_loaded:
            smi = self._read_memory_nvidia_smi()
            if smi is not None:
                return smi
        result = self._read_memory_torch(initialize=initialize)
        if result is None:
            result = self._read_memory_nvidia_smi()
        if result is None:
            result = self._fallback_memory()
        return result

    def free_mb(self) -> int:
        """当前空闲显存（精确查询，可能初始化 CUDA）。"""
        free, _ = self.read_memory(initialize=True)
        return int(round(free))

    def health_memory(self) -> tuple[int, int]:
        """给 /health 用的轻量显存数字：1 秒缓存，且不主动初始化 CUDA。"""
        now = time.monotonic()
        with self._lock:
            if self._health_cache is not None and now - self._health_cache[0] < 1.0:
                return self._health_cache[1], self._health_cache[2]
        free, total = self.read_memory(initialize=False)
        result = (int(round(free)), int(round(total)))
        with self._lock:
            self._health_cache = (now, result[0], result[1])
        return result

    def release_cuda_cache(self) -> None:
        """加载失败后清一下 CUDA 缓存，尽量不把半截分配留在显存里。"""
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def torch_memory(self) -> dict[str, Optional[int]]:
        """给 /vram 的诊断字段：torch 已分配 / 已保留（不主动初始化 CUDA）。"""
        torch = sys.modules.get("torch")
        empty: dict[str, Optional[int]] = {
            "torch_allocated_mb": None,
            "torch_reserved_mb": None,
        }
        if torch is None:
            return empty
        try:
            if not torch.cuda.is_available() or not torch.cuda.is_initialized():
                return empty
            return {
                "torch_allocated_mb": int(round(torch.cuda.memory_allocated() / (1024 * 1024))),
                "torch_reserved_mb": int(round(torch.cuda.memory_reserved() / (1024 * 1024))),
            }
        except Exception:  # noqa: BLE001
            return empty

    # ------------------------------------------------------------------
    # 核心：加载前腾显存
    # ------------------------------------------------------------------

    async def ensure_capacity(
        self,
        name: str,
        need_mb: Optional[float] = None,
        *,
        margin_mb: Optional[int] = None,
    ) -> dict[str, Any]:
        """在加载 ``name`` 之前确保显存够用；不够就按 LRU 卸载其它模型。"""
        with self._lock:
            target = self._entries.get(name)
            if target is None:
                raise KeyError(f"未注册的显存能力：{name}")
            base_need = (
                float(need_mb) if need_mb is not None else float(target.estimate_mb)
            )
            margin = self.margin_mb if margin_mb is None else max(0, int(margin_mb))
        required = max(0.0, base_need + margin)

        async with self._op_lock:
            free_mb, total_mb = await asyncio.to_thread(self.read_memory, initialize=True)
            if free_mb >= required:
                return {
                    "ok": True,
                    "released": [],
                    "free_mb": int(round(free_mb)),
                    "total_mb": int(round(total_mb)),
                    "need_mb": int(math.ceil(required)),
                }

            # 候选：其它已加载、当前没有请求在用、最近没用过的排前面。
            with self._lock:
                candidates = [
                    entry for entry in self._entries.values() if entry.name != name
                ]
                candidates.sort(
                    key=lambda entry: (
                        entry.last_used_mono
                        if entry.last_used_mono is not None
                        else float("-inf")
                    )
                )

            released: list[str] = []
            for entry in candidates:
                with self._lock:
                    if entry.active > 0:
                        continue
                if not entry.is_loaded():
                    continue
                async with _async_lock(entry.infer_lock):
                    # 拿到推理锁后再检查一次：可能刚才有请求进来并把它标成 active。
                    with self._lock:
                        if entry.active > 0:
                            continue
                    if not entry.is_loaded():
                        continue
                    before_mb, _ = await asyncio.to_thread(self.read_memory, initialize=True)
                    try:
                        await asyncio.to_thread(entry.unload_fn)
                    except Exception as exc:  # noqa: BLE001 - 卸一个失败不影响继续
                        self.log.warning("腾显存：卸载 %s 失败：%s", entry.name, exc)
                        continue
                    after_mb, _ = await asyncio.to_thread(self.read_memory, initialize=True)
                    if entry.is_loaded():
                        self.log.warning(
                            "腾显存：调用 %s.unload() 后仍显示已加载，跳过", entry.name
                        )
                        continue
                    delta = max(0.0, before_mb - after_mb)
                    if delta < 1.0:
                        # 小模型（声纹）的增量可能被驱动统计吃掉，用估算值兜底展示。
                        delta = entry.estimate_mb
                    released.append(entry.name)
                    free_mb = after_mb
                    self.log.info(
                        "腾显存：卸掉 %s（释放约 %d MB），当前空闲 %d MB / 需要 %d MB",
                        entry.name,
                        round(delta),
                        round(after_mb),
                        round(required),
                    )
                if free_mb >= required:
                    break

            if free_mb >= required:
                return {
                    "ok": True,
                    "released": released,
                    "free_mb": int(round(free_mb)),
                    "total_mb": int(round(total_mb)),
                    "need_mb": int(math.ceil(required)),
                }

            shortfall = max(0.0, required - free_mb)
            shortfall_mb = int(math.ceil(shortfall))
            reason = (
                f"显存不足：空闲 {free_mb:.0f} MB，需要 {required:.0f} MB"
                f"（模型估算 {base_need:.0f} MB + 安全余量 {margin} MB），"
                f"已卸载其它模型 {('、'.join(released) if released else '无')}，"
                f"仍差 {shortfall_mb} MB。请关闭占用显卡的程序后重试。"
            )
            self.log.warning("腾显存失败：%s", reason)
            return {
                "ok": False,
                "released": released,
                "free_mb": int(round(free_mb)),
                "total_mb": int(round(total_mb)),
                "need_mb": int(math.ceil(required)),
                "shortfall_mb": shortfall_mb,
                "reason": reason,
            }

    # ------------------------------------------------------------------
    # 手动卸载
    # ------------------------------------------------------------------

    async def unload_models(
        self,
        names: Optional[list[str]] = None,
        *,
        unload_all: bool = False,
    ) -> dict[str, Any]:
        """手动卸载指定能力（或全部）；会等推理锁，不会卸到一半。"""
        with self._lock:
            if unload_all:
                targets = list(self._entries.values())
            else:
                targets = [
                    self._entries[name]
                    for name in (names or [])
                    if name in self._entries
                ]
        released: list[str] = []
        for entry in targets:
            if not entry.is_loaded():
                continue
            async with _async_lock(entry.infer_lock):
                if not entry.is_loaded():
                    continue
                try:
                    await asyncio.to_thread(entry.unload_fn)
                except Exception as exc:  # noqa: BLE001
                    self.log.warning("手动卸载 %s 失败：%s", entry.name, exc)
                    continue
                if not entry.is_loaded():
                    released.append(entry.name)
                    with self._lock:
                        entry.last_used = None
                        entry.last_used_mono = None
                    self.log.info("手动卸载：已卸掉 %s", entry.name)
        free_mb, total_mb = await asyncio.to_thread(self.read_memory, initialize=False)
        return {
            "ok": True,
            "released": released,
            "free_mb": int(round(free_mb)),
            "total_mb": int(round(total_mb)),
        }

    # ------------------------------------------------------------------
    # 诊断快照
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """返回当前显存与各能力状态，供 GET /vram 使用。"""
        free, total = self.read_memory(initialize=False)
        now_mono = time.monotonic()
        models: list[dict[str, Any]] = []
        with self._lock:
            entries = [self._entries[name] for name in self._order if name in self._entries]
            for entry in entries:
                loaded = entry.is_loaded()
                last_used_text: Optional[str] = None
                if entry.last_used is not None:
                    last_used_text = datetime.fromtimestamp(entry.last_used).astimezone().isoformat(
                        timespec="seconds"
                    )
                idle_seconds: Optional[float] = None
                if loaded and entry.last_used_mono is not None:
                    idle_seconds = round(max(0.0, now_mono - entry.last_used_mono), 1)
                models.append(
                    {
                        "name": entry.name,
                        "loaded": loaded,
                        "last_used": last_used_text,
                        "idle_seconds": idle_seconds,
                        "estimate_mb": int(round(entry.estimate_mb)),
                        "measured_mb": (
                            int(round(entry.measured_mb))
                            if entry.measured_mb is not None
                            else None
                        ),
                        "samples": entry.samples,
                        "active": entry.active,
                        "loader": entry.loader_hint,
                    }
                )
        return {
            "free_mb": int(round(free)),
            "total_mb": int(round(total)),
            "used_mb": int(round(max(0.0, total - free))),
            **self.torch_memory(),
            "models": models,
            "margin_mb": self.margin_mb,
        }

    # ------------------------------------------------------------------
    # 估算校准 / 持久化
    # ------------------------------------------------------------------

    def _load_saved_estimates(self) -> None:
        try:
            raw = self.estimates_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            models = data.get("models") if isinstance(data, dict) else None
            if isinstance(models, dict):
                self._saved = {
                    str(name): value
                    for name, value in models.items()
                    if isinstance(value, dict)
                }
                self.log.info(
                    "已读取显存估算校准：%s（%s）",
                    self.estimates_path,
                    "、".join(
                        f"{name}={round(float(value.get('estimate_mb') or 0))}MB"
                        for name, value in self._saved.items()
                    )
                    or "空",
                )
        except FileNotFoundError:
            self.log.info("暂无显存估算记录，使用初始估算：%s", self.estimates_path)
        except Exception as exc:  # noqa: BLE001 - 文件损坏时用初始值，不影响启动
            self.log.warning("读取显存估算记录失败（用初始值）：%s", exc)
            self._saved = {}

    def _save_estimates_locked(self) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "models": {
                name: {
                    "estimate_mb": int(round(entry.estimate_mb)),
                    "last_measured_mb": (
                        int(round(entry.measured_mb))
                        if entry.measured_mb is not None
                        else None
                    ),
                    "samples": entry.samples,
                    "loader": entry.loader_hint,
                }
                for name, entry in self._entries.items()
            },
        }
        try:
            self.estimates_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.estimates_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, self.estimates_path)
        except Exception as exc:  # noqa: BLE001 - 持久化失败不影响本次请求
            self.log.warning("保存显存估算记录失败：%s", exc)

    def record_load(
        self,
        name: str,
        before_free_mb: Optional[float],
        after_free_mb: Optional[float],
    ) -> Optional[int]:
        """加载成功后用「加载前后空闲显存差」校准估算值（每次最多 ±20%）。"""
        if before_free_mb is None or after_free_mb is None:
            return None
        measured = float(before_free_mb) - float(after_free_mb)
        return self.record_measured(name, measured)

    def record_measured(self, name: str, measured_mb: Optional[float]) -> Optional[int]:
        """用已知的真实增量校准估算。

        视觉模型跑在独立子进程里，CUDA 的 ``mem_get_info`` 统计不到，
        由适配器用 nvidia-smi 测出增量后调用这里。
        """
        if measured_mb is None:
            return None
        measured = float(measured_mb)
        # 小于 8MB 视为统计噪声（例如模型已经在别处加载过），不校准。
        if measured < 8:
            return None
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return None
            old = entry.estimate_mb
            low = old * 0.8
            high = old * 1.2
            new = min(high, max(low, measured))
            new = max(new, entry.min_estimate_mb)
            if entry.max_estimate_mb > 0:
                new = min(new, entry.max_estimate_mb)
            entry.estimate_mb = max(1.0, new)
            entry.measured_mb = measured
            entry.samples += 1
            self._save_estimates_locked()
            change = entry.estimate_mb - old
        self.log.info(
            "显存估算校准：%s %d MB -> %d MB（本次实测增量 %d MB，样本 %d）",
            name,
            round(old),
            round(entry.estimate_mb),
            round(measured),
            entry.samples,
        )
        if abs(change) >= 1:
            self.log.info("校准变化：%s %+.0f MB（限制在 ±20%% 内）", name, change)
        return int(round(entry.estimate_mb))


# ---------------------------------------------------------------------------
# 进程级单例 + 服务层辅助函数
# ---------------------------------------------------------------------------

_VRAM: Optional[VramManager] = None
_VRAM_LOCK = threading.Lock()
# 加载动作串行化：避免两个请求同时判断「显存够」然后一起加载。
_LOAD_LOCK = asyncio.Lock()


def get_vram(cfg: Optional[Config] = None) -> VramManager:
    """取进程级显存管家（第一次调用时创建）。"""
    global _VRAM
    if _VRAM is not None:
        return _VRAM
    with _VRAM_LOCK:
        if _VRAM is None:
            cfg = cfg or get_config()
            _VRAM = VramManager(
                estimates_path=cfg.vram_estimates_path,
                margin_mb=cfg.vram_margin_mb,
                logger_=logging.getLogger("localai.vram"),
            )
    return _VRAM


def register_adapters(registry: Any) -> VramManager:
    """把注册表里所有需要显存的能力登记到管家（只登记函数，不加载模型）。"""
    vram = get_vram()
    for adapter in registry.all():
        if not adapter.requires_gpu:
            continue
        vram.register(
            adapter.name,
            adapter.unload,
            adapter.estimate_mb(),
            adapter.loader_hint,
            is_loaded_fn=adapter.loaded,
            infer_lock=adapter.infer_lock,
            min_estimate_mb=adapter.MIN_ESTIMATE_MB,
        max_estimate_mb=getattr(adapter, "MAX_ESTIMATE_MB", 0),
        )
    logger.info(
        "显存管家就绪：margin=%dMB 估算文件=%s 初始估算=%s",
        vram.margin_mb,
        vram.estimates_path,
        vram.estimates(),
    )
    return vram


async def ensure_model_loaded(adapter: Any, loader: Optional[Callable[[], Any]] = None) -> list[str]:
    """加载前腾显存 -> 加载 -> 记录实测增量；返回本次被卸载的模型名。

    调用方必须已经持有该能力的推理锁，并调用 ``VRAM.begin_use(name)``，
    这样 ensure_capacity 不会卸掉正在请求的能力。

    单飞：``_LOAD_LOCK`` 内会二次检查 ``adapter.loaded()``，并发请求只会真正
    加载一次，后来者在锁上排队、拿到锁后直接返回。锁顺序固定为
    「能力的 infer_lock -> ``_LOAD_LOCK``」：``_LOAD_LOCK`` 内只会卸载
    ``ensure_capacity`` 判定为没在推理（active==0）的能力，所以不会反过来
    等一个正持有自己的 infer_lock 又在等 ``_LOAD_LOCK`` 的请求（vision 例外，
    但它已加载时请求在预检就返回，不会持闸等待）。
    """
    vram = get_vram()
    name = adapter.name
    if adapter.loaded():
        vram.touch(name)
        return []

    async with _LOAD_LOCK:
        # 等锁期间可能已经被别的请求加载好了。
        if adapter.loaded():
            vram.touch(name)
            return []

        # 视觉模型跑在独立 llama-server 里，Windows 上 CUDA 的 mem_get_info
        # 看不到 Vulkan 已经占用的显存，可能把空闲报高，导致 TTS 与 VLM
        # 同时挤在 8G 卡上（实际会触发 WDDM 换页）。所以在加载任何其它
        # GPU 模型前，先显式卸掉视觉模型；这是唯一一个跨进程的显存占用。
        extra_released: list[str] = []
        if name != "vision" and "vision" in vram.loaded_names():
            try:
                outcome = await vram.unload_models(["vision"])
                extra_released = list(outcome.get("released") or [])
                if extra_released:
                    logger.info(
                        "加载 %s 前先卸载视觉模型（跨进程显存，CUDA 统计不可靠）",
                        name,
                    )
            except Exception as exc:  # noqa: BLE001 - 卸不掉就继续走容量检查
                logger.warning("加载 %s 前卸载视觉模型失败：%s", name, exc)

        capacity = await vram.ensure_capacity(name)
        if not capacity.get("ok"):
            raise InsufficientVramError(capacity)

        released = extra_released + list(capacity.get("released") or [])
        before_free, _ = await asyncio.to_thread(lambda: vram.read_memory(initialize=True))
        load_call = loader or adapter.load
        try:
            await asyncio.to_thread(load_call)
        except Exception as exc:  # noqa: BLE001
            vram.release_cuda_cache()
            if is_cuda_oom(exc):
                after_free, _ = await asyncio.to_thread(
                    lambda: vram.read_memory(initialize=True)
                )
                need_mb = int(capacity.get("need_mb") or 0)
                shortfall = max(0, need_mb - int(round(after_free)))
                result = {
                    "ok": False,
                    "released": released,
                    "free_mb": int(round(after_free)),
                    "need_mb": need_mb,
                    "shortfall_mb": shortfall,
                    "reason": (
                        f"加载 {name} 时 CUDA 显存不足：空闲 {after_free:.0f} MB，"
                        f"需要 {need_mb} MB，仍差 {shortfall} MB；已卸载 "
                        f"{('、'.join(released) if released else '无')}。"
                    ),
                }
                logger.error("加载 %s 时显存不足：%s", name, exc)
                raise InsufficientVramError(result) from exc
            raise

        after_free, _ = await asyncio.to_thread(lambda: vram.read_memory(initialize=True))
        # 视觉模型跑在子进程里，torch 的差值不可信；估算由适配器用
        # nvidia-smi 实测后调用 record_measured 校准，这里不要重复记。
        if name != "vision":
            vram.record_load(name, before_free, after_free)
        vram.touch(name)
        logger.info(
            "模型加载完成：%s（加载前后空闲 %.0f -> %.0f MB，本次腾出 %s）",
            name,
            before_free,
            after_free,
            "、".join(released) or "无",
        )
        return released
