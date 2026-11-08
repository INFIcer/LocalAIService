"""任务调度器：严格三层排序 + 同类型在显存允许时并行。

排序（从高到低，**没有 aging**）
--------------------------------
1. **同类型优先**：``current_kind`` 仍有效、且该 kind 还有排队任务时，
   只在这个 kind 里取下一个；别的 kind 优先级再高也不切走。
2. **优先级**：任务提交时的 ``priority``（默认 0，越大越先）。
3. **提交时间**：``created_at`` 早的先；同毫秒用 rowid（入库顺序）兜底。

实现方式：``store.pick_next(current_kind)`` 先看同 kind，队列空了才看全局；
它返回的任务 kind 就是新的 ``current_kind``。**同类清空是唯一的切换出口**，
因此同类型持续涌入会把其它类型饿死 —— 这是明确接受的取舍（详见
``docs/SERVICE_API.md``）。

并行
----
- 同一个 ``current_kind`` 的任务可以同时跑，上限来自
  ``LOCALAI_JOBS_PARALLEL_TTS/ASR/SEPARATE``；不同 kind 永远串行，
  避免一张卡上叠两套模型。
- 新增一个并行任务前检查 ``vram.snapshot()``：
  ``free_mb - margin_mb >= LOCALAI_JOBS_PARALLEL_EXTRA_MB``，不够就只跑
  已经派发的那些，不新开。第一个任务不检查（它负责按需加载模型）。
- 适配器内部的 ``infer_lock`` 已换成容量可调的并发闸门；是否真的允许
  同模型并发由适配器的 ``CAN_INFER_IN_PARALLEL`` 决定（未实测的能力强制 1）。
- 撞 CUDA OOM：把该 kind 的并行**临时降到 1**、卸载其它模型、只重试这一条
  任务一次；再失败则置 failed，进程不受影响。

其它
----
- **超时**：到点先把任务置 failed，然后等底层阻塞调用真正退出再释放名额，
  避免同一张卡上出现两个推理（GPU 调用无法从线程里强杀）。
- **取消**：queued 直接 canceled；running 只打 ``cancel_requested``，
  执行器退出后置 canceled 且不写 result（不能强杀 GPU 调用）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Optional

from ..config import Config
from ..core.errors import (
    JobCanceledError,
    JobFailedError,
    JobTimeoutError,
    is_cuda_oom,
)
from .executors import JobContext, execute_job
from .store import TERMINAL_STATES, JobStore, NewJob

logger = logging.getLogger("localai.jobs.scheduler")

#: 记住多少个「原始异常对象」供同步接口重新抛出（仅内存，重启后失效）。
_MAX_REMEMBERED_ERRORS = 256

#: 参与同类并行的 kind（配置里能单独调并行度的三类）。
_PARALLEL_KINDS = ("tts", "asr", "separate")


class JobScheduler:
    """单 worker 调度循环（一个循环派发多个同 kind 任务）；随 lifespan 启停。"""

    def __init__(
        self,
        cfg: Config,
        registry: Any,
        store: JobStore,
        *,
        client: Any = None,
        vram: Any = None,
        executor: Optional[Callable[[dict[str, Any], JobContext], Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.store = store
        self.client = client
        self.vram = vram
        self.executor = executor or execute_job
        self.ctx = JobContext(cfg=cfg, registry=registry, client=client)

        self.current_kind: Optional[str] = None
        self.batch_streak = 0
        #: 正在执行的任务：job_id -> asyncio.Task（全部属于 current_kind）
        self._running: dict[str, asyncio.Task[Any]] = {}
        self._task: Optional[asyncio.Task[Any]] = None
        self._wake = asyncio.Event()
        self._slot = asyncio.Event()
        self._stopping = False
        self._waiters: dict[str, asyncio.Event] = {}
        self._cancel_requested: set[str] = set()
        self._errors: OrderedDict[str, BaseException] = OrderedDict()
        #: OOM 后被临时降级到 1 的 kind -> 原因
        self._degraded: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """恢复上次遗留的 running，然后启动调度循环。"""
        recovered = await self.store.recover_running(self.cfg.jobs_recover)
        if recovered.get("requeued") or recovered.get("failed"):
            logger.warning(
                "任务恢复完成：requeue=%d fail=%d（LOCALAI_JOBS_RECOVER=%s）",
                recovered.get("requeued", 0),
                recovered.get("failed", 0),
                self.cfg.jobs_recover,
            )
        counts = await self.store.health()
        logger.info(
            "任务调度器启动：queued=%d running=%d 排序=同类型>优先级>提交时间"
            "（无 aging） 总并发上限=%s 每类并行=%s 并行额外显存=%dMB timeout=%.0fs",
            counts["queued"],
            counts["running"],
            self.cfg.jobs_global_cap() or "不额外限制",
            self.parallel_map(),
            self.cfg.jobs_parallel_extra_mb,
            self.cfg.jobs_timeout,
        )
        for kind in _PARALLEL_KINDS:
            adapter = self.registry.find(kind)
            if adapter is not None and not getattr(
                adapter, "CAN_INFER_IN_PARALLEL", False
            ):
                logger.info(
                    "能力 %s 未确认可并发，同类并行固定为 1（%s）",
                    kind,
                    getattr(adapter, "PARALLEL_NOTE", "未实测"),
                )
        self._stopping = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker(), name="localai-job-worker")

    async def stop(self) -> None:
        """停止调度循环；正在跑的任务保留 running，交给下次启动的崩溃恢复。"""
        self._stopping = True
        self._wake.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # 并行时可能有多个执行任务；一并取消（DB 里保持 running，下次 requeue）。
        running = list(self._running.values())
        for item in running:
            item.cancel()
        if running:
            with contextlib.suppress(Exception):
                await asyncio.gather(*running, return_exceptions=True)
        logger.info(
            "任务调度器已停止（当前运行中：%s）",
            "、".join(self._running) if self._running else "-",
        )

    # ------------------------------------------------------------------
    # 提交 / 等待 / 取消
    # ------------------------------------------------------------------
    async def submit(
        self,
        kind: str,
        action: str,
        payload: dict[str, Any],
        *,
        priority: int = 0,
        client_ip: str = "",
        job_id: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> str:
        """写库并唤醒调度器；返回 job id。"""
        job_id = job_id or uuid.uuid4().hex
        await self.store.create(
            NewJob(
                id=job_id,
                kind=kind,
                action=action,
                payload=payload,
                priority=int(priority),
                client_ip=client_ip or "",
                meta=meta,
            )
        )
        self._wake.set()
        logger.info(
            "任务提交：id=%s kind=%s action=%s priority=%d client=%s",
            job_id,
            kind,
            action,
            int(priority),
            client_ip or "-",
        )
        return job_id

    async def wait(self, job_id: str, timeout: Optional[float] = None) -> dict[str, Any]:
        """等任务到终态；done 返回记录，failed/canceled 抛对应异常。

        同步旧接口用这个函数「提交后等结果」，因此失败时必须把调度器里保存的
        **原始异常对象**重新抛出，复用旧的 HTTP 错误处理器。
        """
        deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
        while True:
            job = await self.store.get(job_id)
            if job is None:
                raise KeyError(f"任务不存在：{job_id}")
            if job["state"] in TERMINAL_STATES:
                return self._outcome(job)
            event = self._waiters.get(job_id)
            if event is None:
                event = asyncio.Event()
                self._waiters[job_id] = event
            # 装上事件后再查一次，避免「装事件之前任务刚好结束」的丢唤醒。
            job = await self.store.get(job_id)
            if job is not None and job["state"] in TERMINAL_STATES:
                return self._outcome(job)
            remaining = None
            if deadline is not None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise JobTimeoutError(f"等待任务 {job_id} 超时")
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise JobTimeoutError(f"等待任务 {job_id} 超时") from exc

    def _outcome(self, job: dict[str, Any]) -> dict[str, Any]:
        state = job["state"]
        self._waiters.pop(job["id"], None)
        if state == "done":
            self._errors.pop(job["id"], None)
            return job
        if state == "canceled":
            raise JobCanceledError(job.get("error") or "任务已取消")
        error = self._errors.pop(job["id"], None)
        if error is not None:
            raise error
        raise JobFailedError(job.get("error") or "任务失败")

    async def cancel(self, job_id: str) -> Optional[dict[str, Any]]:
        """queued 直接取消；running 标记 cancel_requested，退出后不写结果。"""
        job = await self.store.get(job_id)
        if job is None:
            return None
        state = job["state"]
        if state == "queued":
            updated = await self.store.cancel_queued(job_id)
            await self._cleanup_upload(job_id)
            self._wake.set()
            logger.info("任务取消（排队中）：id=%s kind=%s", job_id, job["kind"])
            return updated
        if state == "running":
            self._cancel_requested.add(job_id)
            await self.store.set_cancel_requested(job_id)
            logger.info("任务取消（运行中，尽力而为）：id=%s kind=%s", job_id, job["kind"])
            return await self.store.get(job_id)
        return job

    # ------------------------------------------------------------------
    # 并行度 / 观测
    # ------------------------------------------------------------------
    def _parallel_limit(self, kind: str) -> int:
        """该类当前实际允许的并行度（配置 + 适配器能力 + OOM 降级）。"""
        if kind in self._degraded:
            return 1
        adapter = self.registry.find(kind)
        if adapter is None:
            return 1
        gate = getattr(adapter, "infer_lock", None)
        capacity = int(getattr(gate, "capacity", 1) or 1)
        return max(1, capacity)

    def parallel_map(self) -> dict[str, int]:
        """所有能力当前生效的并行度（给 /jobs/stats、/capabilities）。"""
        return {
            "tts": self._parallel_limit("tts"),
            "asr": self._parallel_limit("asr"),
            "separate": self._parallel_limit("separate"),
            "speaker": self._parallel_limit("speaker"),
            "media": self._parallel_limit("media"),
        }

    @property
    def running_job_id(self) -> Optional[str]:
        """兼容旧字段：并行时返回其中一个（完整列表见 stats）。"""
        return next(iter(self._running), None)

    async def health(self) -> dict[str, Any]:
        counts = await self.store.health()
        return {
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "current_kind": self.current_kind,
        }

    async def stats(self) -> dict[str, Any]:
        base = await self.store.stats()
        base["scheduler"] = {
            "current_kind": self.current_kind,
            "batch_streak": self.batch_streak,
            "loaded_model": self._loaded_model(),
            "running_job_id": self.running_job_id,
            "running_job_ids": list(self._running),
            "running": len(self._running),
            "max_parallel": self.cfg.jobs_max_parallel,
            "per_kind_parallel": self.parallel_map(),
            "asr_parallel": self._parallel_limit("asr"),
            "parallel_extra_mb": self.cfg.jobs_parallel_extra_mb,
            "parallel_degraded": dict(self._degraded),
            "concurrency": self._concurrency_snapshot(),
            "timeout_seconds": self.cfg.jobs_timeout,
            "recover_mode": self.cfg.jobs_recover,
            "paused": False,
        }
        return base

    def _concurrency_snapshot(self) -> dict[str, Any]:
        """每个能力的并发闸门状态（容量 / 当前占用 / 峰值）。"""
        snapshot: dict[str, Any] = {}
        for name in self.registry.names():
            adapter = self.registry.get(name)
            gate = getattr(adapter, "infer_lock", None)
            if hasattr(gate, "snapshot"):
                snapshot[name] = gate.snapshot()
        return snapshot

    def _loaded_model(self) -> Optional[str]:
        names: list[str] = []
        for adapter in self.registry.gpu_adapters():
            try:
                if adapter.loaded():
                    names.append(getattr(adapter, "model_name", "") or adapter.name)
            except Exception:  # noqa: BLE001 - 状态读取失败不影响统计
                continue
        return "+".join(names) if names else None

    def _model_of(self, kind: str) -> str:
        adapter = self.registry.find(kind)
        if adapter is None:
            return "ffmpeg" if kind == "media" else kind
        return getattr(adapter, "model_name", "") or kind

    def _vram_text(self) -> str:
        if self.vram is None:
            return "-"
        try:
            free, total = self.vram.health_memory()
            return f"{free}/{total}MB"
        except Exception:  # noqa: BLE001
            return "-"

    # ------------------------------------------------------------------
    # 调度循环
    # ------------------------------------------------------------------
    async def _worker(self) -> None:
        logger.debug("调度循环进入")
        while not self._stopping:
            try:
                # 能开多少开多少，直到达到并行上限 / 显存不够 / 没任务。
                while not self._stopping and await self._try_start_one():
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 调度异常不能让循环退出
                logger.error("调度器派发任务失败：%s", exc)
            if self._stopping:
                break
            await self._wait_for_work()
        logger.debug("调度循环退出")

    async def _wait_for_work(self) -> None:
        """等「新任务提交」或「有任务跑完」，最多 ``jobs_poll_interval``。"""
        wake = asyncio.create_task(self._wake.wait())
        slot = asyncio.create_task(self._slot.wait())
        try:
            await asyncio.wait(
                {wake, slot},
                timeout=self.cfg.jobs_poll_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for item in (wake, slot):
                item.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(wake, slot, return_exceptions=True)
        self._wake.clear()
        self._slot.clear()

    async def _try_start_one(self) -> bool:
        """尝试派发一个任务；派发了返回 True，派不动返回 False。"""
        if self._stopping:
            return False
        global_cap = self.cfg.jobs_global_cap()
        if global_cap is not None and len(self._running) >= global_cap:
            return False

        job = await self.store.pick_next(current_kind=self.current_kind)
        if job is None:
            return False
        kind = job["kind"]

        # 不同 kind 串行：当前批次还没跑完时，别把另一种模型拉上卡。
        if self._running and kind != self.current_kind:
            logger.debug(
                "调度等待：%s 任务 %s 要等 %s 批次收尾（不同 kind 不并行）",
                kind,
                job["id"],
                self.current_kind,
            )
            return False

        limit = self._parallel_limit(kind)
        running_same = len(self._running)  # 同一时刻只会有一个 kind 在跑
        if running_same >= limit:
            return False
        # 第一个任务负责加载模型，不做显存预扣；新增并行任务必须确认放得下。
        if running_same >= 1 and not self._vram_allows_extra(kind):
            return False

        if not await self.store.mark_running(job["id"]):
            # 在取出与认领之间被取消了，继续看下一个。
            return True
        await self._start_job(job)
        return True

    def _vram_allows_extra(self, kind: str) -> bool:
        """再开一个并行任务，显存够不够（模型必须已加载）。"""
        adapter = self.registry.find(kind)
        if adapter is None or not getattr(adapter, "requires_gpu", False):
            return False
        try:
            if not adapter.loaded():
                return False
        except Exception:  # noqa: BLE001
            return False
        if self.vram is None:
            return False
        try:
            snapshot = self.vram.snapshot()
        except Exception as exc:  # noqa: BLE001 - 查不到就当不够
            logger.debug("读取显存快照失败，暂不新增并行：%s", exc)
            return False
        free_mb = int(snapshot.get("free_mb") or 0)
        margin_mb = int(snapshot.get("margin_mb") or self.cfg.vram_margin_mb)
        extra_mb = int(self.cfg.jobs_parallel_extra_mb)
        allowed = free_mb - margin_mb >= extra_mb
        logger.debug(
            "并行显存检查：kind=%s 空闲=%dMB 余量=%dMB 每个新增需=%dMB -> %s",
            kind,
            free_mb,
            margin_mb,
            extra_mb,
            "放行" if allowed else "不新开",
        )
        return allowed

    async def _start_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        kind = job["kind"]
        if kind != self.current_kind:
            if self.current_kind is not None:
                queued_new = await self.store.queued_count(kind) + 1
                logger.info(
                    "调度切换：%s→%s（批次 %d/%d；同类队列已清空是唯一切换出口）",
                    self.current_kind,
                    kind,
                    self.batch_streak,
                    queued_new,
                )
            self.current_kind = kind
            self.batch_streak = 1
        else:
            self.batch_streak += 1

        task = asyncio.create_task(
            self._run_job(job), name=f"localai-job-{job_id[:8]}"
        )
        self._running[job_id] = task
        task.add_done_callback(lambda _t, jid=job_id: self._on_job_done(jid))

    def _on_job_done(self, job_id: str) -> None:
        self._running.pop(job_id, None)
        # 唤醒调度循环：名额空出来了，可以派发队列里的下一个。
        self._slot.set()

    async def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        kind = job["kind"]
        action = job["action"]
        wait_ms = 0.0
        try:
            created = datetime.fromisoformat(job["created_at"])
            wait_ms = max(0.0, (datetime.now().astimezone() - created).total_seconds() * 1000)
        except (TypeError, ValueError):
            pass
        model = self._model_of(kind)
        logger.info(
            "任务开始：id=%s kind=%s action=%s model=%s 等待=%.0fms 显存=%s 并发=%d/%d",
            job_id,
            kind,
            action,
            model,
            wait_ms,
            self._vram_text(),
            len(self._running),
            self._parallel_limit(kind),
        )

        started_perf = time.monotonic()
        error: Optional[BaseException] = None
        result: Optional[dict[str, Any]] = None
        try:
            attempt = 0
            parallel_before = self._parallel_limit(kind)
            while True:
                result, error = await self._execute_once(job)
                if error is None:
                    break
                # 并行推理撞 OOM：降级到 1、卸载其它模型、只重试这一条一次。
                if is_cuda_oom(error) and parallel_before > 1 and attempt == 0:
                    attempt += 1
                    await self._degrade_after_oom(kind, error)
                    parallel_before = 1
                    logger.warning(
                        "任务 %s 并行推理显存不足，已降级重试一次：%s", job_id, error
                    )
                    continue
                await self.store.fail(job_id, f"{type(error).__name__}: {error}")
                break
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - 兜底：不能把状态卡在 running
            error = exc
            with contextlib.suppress(Exception):
                await self.store.fail(job_id, f"{type(exc).__name__}: {exc}")

        if error is not None:
            self._remember_error(job_id, error)
            logger.error(
                "任务失败：id=%s kind=%s action=%s 等待=%.0fms 耗时=%.0fms 显存=%s 错误=%s: %s",
                job_id,
                kind,
                action,
                wait_ms,
                self._elapsed_ms(started_perf),
                self._vram_text(),
                type(error).__name__,
                error,
            )
        elif result is not None:
            if job_id in self._cancel_requested:
                await self.store.mark_canceled(job_id, "已取消（运行中尽力取消，结果已丢弃）")
                logger.info(
                    "任务取消完成：id=%s kind=%s 结果已丢弃 耗时=%.0fms",
                    job_id,
                    kind,
                    self._elapsed_ms(started_perf),
                )
            else:
                await self.store.finish(job_id, result)
                logger.info(
                    "任务完成：id=%s kind=%s action=%s model=%s 等待=%.0fms 耗时=%.0fms 显存=%s",
                    job_id,
                    kind,
                    action,
                    model,
                    wait_ms,
                    self._elapsed_ms(started_perf),
                    self._vram_text(),
                )
        self._cancel_requested.discard(job_id)
        self._notify_waiters(job_id)
        await self._cleanup_upload(job_id)

    async def _execute_once(
        self, job: dict[str, Any]
    ) -> tuple[Optional[dict[str, Any]], Optional[BaseException]]:
        """执行一次；返回 (结果, 异常)。超时在这里直接置 failed。"""
        job_id = job["id"]
        kind = job["kind"]
        exec_task = asyncio.create_task(self.executor(job, self.ctx))
        try:
            done, _pending = await asyncio.wait({exec_task}, timeout=self.cfg.jobs_timeout)
            if not done:
                message = f"任务超时（{self.cfg.jobs_timeout:.0f} 秒）"
                await self.store.fail(job_id, message)
                logger.error(
                    "任务超时：id=%s kind=%s 超过 %.0fs，已置 failed；等待底层调用退出",
                    job_id,
                    kind,
                    self.cfg.jobs_timeout,
                )
                # GPU 调用无法强杀：等它真正退出，避免下一个任务和它抢显存。
                with contextlib.suppress(BaseException):
                    await exec_task
                return None, JobTimeoutError(message)
            return exec_task.result(), None
        except asyncio.CancelledError:
            exec_task.cancel()
            with contextlib.suppress(BaseException):
                await exec_task
            raise
        except BaseException as exc:  # noqa: BLE001 - 记录到 DB，再由 waiter 重抛
            return None, exc

    async def _degrade_after_oom(self, kind: str, error: BaseException) -> None:
        """并行撞 OOM：该 kind 并行临时降为 1，卸载其它模型腾空间。"""
        reason = f"CUDA OOM 后降级：{type(error).__name__}: {error}"
        self._degraded[kind] = reason[:400]
        adapter = self.registry.find(kind)
        gate = getattr(adapter, "infer_lock", None) if adapter is not None else None
        if gate is not None and hasattr(gate, "set_capacity"):
            with contextlib.suppress(Exception):
                await gate.set_capacity(1, reason="并行推理 CUDA OOM，临时降为 1")
        if adapter is not None:
            with contextlib.suppress(Exception):
                adapter.parallel_capacity = 1
        released: list[str] = []
        if self.vram is not None:
            try:
                loaded = list(self.vram.loaded_names())
                others = [name for name in loaded if name != kind]
                if others:
                    outcome = await self.vram.unload_models(others)
                    released = list(outcome.get("released") or [])
            except Exception as exc:  # noqa: BLE001 - 卸载失败也要继续重试
                logger.warning("OOM 降级时卸载其它模型失败：%s", exc)
            with contextlib.suppress(Exception):
                self.vram.release_cuda_cache()
        logger.error(
            "显存不足降级：%s 同类并行临时降为 1，已卸载其它模型=%s，重试该任务一次（%s）",
            kind,
            "、".join(released) or "无",
            reason,
        )

    @staticmethod
    def _elapsed_ms(started_perf: float) -> float:
        """从 ``time.monotonic()`` 起点算执行耗时（毫秒）。"""
        return max(0.0, (time.monotonic() - started_perf) * 1000)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _remember_error(self, job_id: str, error: BaseException) -> None:
        self._errors[job_id] = error
        self._errors.move_to_end(job_id)
        while len(self._errors) > _MAX_REMEMBERED_ERRORS:
            self._errors.popitem(last=False)

    def _notify_waiters(self, job_id: str) -> None:
        event = self._waiters.get(job_id)
        if event is not None:
            event.set()

    async def _cleanup_upload(self, job_id: str) -> None:
        """任务结束后按配置删除 ``data/uploads/<job_id>/``。"""
        if self.cfg.jobs_keep_uploads:
            return
        target = self.cfg.jobs_upload_dir / job_id
        if not target.exists():
            return
        try:
            await asyncio.to_thread(shutil.rmtree, target, True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理上传目录失败：%s（%s）", target, exc)
