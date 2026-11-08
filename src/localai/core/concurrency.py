"""按能力可配置的并发闸门（替代单一 ``asyncio.Lock``）。

为什么需要它
------------
以前每个适配器只有一把 ``asyncio.Lock``：同一能力的推理被**串成一条**，
即使显存足够、模型本身也能并发，调度器开再多任务也跑不起来。这里把它换成
「容量可调的信号量」：

- 容量为 1 时语义与旧锁**完全一致**（默认，行为不变）；
- 容量 > 1 时允许最多 N 个任务同时进入 ``async with adapter.infer_lock``；
- 卸载 / 空闲回收这类「必须独占」的动作走 :meth:`ConcurrencyGate.exclusive`，
  它会等所有槽位空出来再占满整把闸门，语义等同旧锁。

为什么不用 ``asyncio.Semaphore``
--------------------------------
``asyncio.Semaphore`` 的容量只能在构造时指定，运行中没法「降级」。撞到
CUDA OOM 时我们需要把某个能力的并发立即从 N 降到 1，因此用
``asyncio.Condition`` + 计数器自己实现，容量随时可改。

线程模型
--------
本模块只在事件循环里使用（``asyncio.Condition`` 不允许跨事件循环复用）。
适配器随应用创建、随应用销毁，一个应用一个事件循环，所以是安全的。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger("localai.concurrency")


class _NullAsyncContext:
    """没有闸门时的空上下文（让调用方代码保持统一）。"""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class ConcurrencyGate:
    """容量可调的异步并发闸门。

    用法与 ``asyncio.Lock`` 相同::

        async with adapter.infer_lock:      # 占 1 个槽位
            ...

        async with adapter.infer_lock.exclusive():   # 独占全部槽位（卸载用）
            ...
    """

    def __init__(self, name: str, capacity: int = 1) -> None:
        self.name = name
        self._capacity = max(1, int(capacity))
        self._in_use = 0
        self._peak = 0
        self._cond = asyncio.Condition()
        #: 最近一次调整容量的原因（给 /jobs/stats 与日志看）
        self.reason = ""

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        """当前允许的最大并发数。"""
        return self._capacity

    @property
    def in_use(self) -> int:
        """当前已占用的槽位数。"""
        return self._in_use

    @property
    def peak(self) -> int:
        """进程启动以来观察到的最大并发数。"""
        return self._peak

    # ------------------------------------------------------------------
    # 容量调整
    # ------------------------------------------------------------------
    async def set_capacity(self, capacity: int, *, reason: str = "") -> int:
        """运行中调整容量（OOM 降级 / 人工调低）；返回调整后的容量。"""
        new_capacity = max(1, int(capacity))
        async with self._cond:
            old = self._capacity
            self._capacity = new_capacity
            if reason:
                self.reason = reason
            # 唤醒所有等待者，让它们按新容量重新判断（升容时立刻放行）。
            self._cond.notify_all()
        if old != new_capacity:
            logger.warning(
                "并发闸门调整：%s %d -> %d（%s）",
                self.name,
                old,
                new_capacity,
                reason or "未说明原因",
            )
        return new_capacity

    # ------------------------------------------------------------------
    # 普通获取 / 释放
    # ------------------------------------------------------------------
    async def acquire(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self._in_use < self._capacity)
            self._in_use += 1
            if self._in_use > self._peak:
                self._peak = self._in_use
            if self._in_use > 1:
                logger.debug(
                    "并发闸门：%s 当前 %d/%d 个推理同时进行",
                    self.name,
                    self._in_use,
                    self._capacity,
                )

    async def release(self) -> None:
        async with self._cond:
            self._in_use = max(0, self._in_use - 1)
            self._cond.notify_all()

    async def __aenter__(self) -> "ConcurrencyGate":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.release()
        return False

    # ------------------------------------------------------------------
    # 独占（卸载模型、空闲回收）
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator["ConcurrencyGate"]:
        """等所有推理退出后独占整把闸门。

        与旧 ``asyncio.Lock`` 的语义一致：拿到之后不会再有别的推理插进来；
        期间新的推理会排队等待。用于 ``vram`` 卸载与 ``idle`` 空闲回收。
        """
        async with self._cond:
            await self._cond.wait_for(lambda: self._in_use == 0)
            # 直接占满全部容量，阻塞期间所有新进入者。
            self._in_use = self._capacity
        try:
            yield self
        finally:
            async with self._cond:
                self._in_use = 0
                self._cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """给 /jobs/stats、/capabilities 的只读快照。"""
        return {
            "name": self.name,
            "capacity": self._capacity,
            "in_use": self._in_use,
            "peak": self._peak,
            "reason": self.reason,
        }


def exclusive(lock: Any) -> Any:
    """把「旧式 asyncio.Lock / 新式 ConcurrencyGate」统一成独占上下文。

    - ``ConcurrencyGate`` -> ``gate.exclusive()``（等所有槽位空出来）；
    - 其它（``asyncio.Lock`` 或 ``None``）-> 原样返回 / 空上下文。
    """
    if lock is None:
        return _NullAsyncContext()
    factory: Optional[Any] = getattr(lock, "exclusive", None)
    if callable(factory):
        return factory()
    return lock
