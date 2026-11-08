"""任务调度器：单 worker、同 kind 批处理、优先级 + 老化、超时 / 取消。

设计取舍
--------
- **单 worker + 单卡**：GPU 同一时刻只跑一个任务，逻辑简单、可控；
  ``LOCALAI_JOBS_MAX_PARALLEL`` 目前只作为预留字段（始终为 1）。
- **批处理**：当前 kind 还有排队任务时继续跑同一 kind，避免 TTS ↔ 分离
  反复加载/卸载模型；但别的 kind 的有效优先级**严格更高**时立刻切换，
  再加上 aging（每等 ``LOCALAI_JOBS_AGING_SECONDS`` 秒 +1 优先级），
  保证低优先级任务不会被永远饿死。
- **执行交给 adapter**：``adapter.infer()`` 内部已经处理 ``infer_lock``、
  ``vram.ensure_capacity``、加载/卸载；调度器只负责顺序与状态流转。
- **超时**：到点先把任务置 failed，然后等底层阻塞调用真正退出再跑下一个，
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
from ..core.errors import JobCanceledError, JobFailedError, JobTimeoutError
from .executors import JobContext, execute_job
from .store import TERMINAL_STATES, JobStore, NewJob

logger = logging.getLogger("localai.jobs.scheduler")

#: 记住多少个「原始异常对象」供同步接口重新抛出（仅内存，重启后失效）。
_MAX_REMEMBERED_ERRORS = 256


class JobScheduler:
    """单 worker 调度循环；随 FastAPI lifespan 启停。"""

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
        self.running_job_id: Optional[str] = None
        self._task: Optional[asyncio.Task[Any]] = None
        self._wake = asyncio.Event()
        self._stopping = False
        self._waiters: dict[str, asyncio.Event] = {}
        self._cancel_requested: set[str] = set()
        self._errors: OrderedDict[str, BaseException] = OrderedDict()

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
            "任务调度器启动：queued=%d running=%d max_parallel=%d aging=%ds timeout=%.0fs",
            counts["queued"],
            counts["running"],
            self.cfg.jobs_max_parallel,
            self.cfg.jobs_aging_seconds,
            self.cfg.jobs_timeout,
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
        logger.info("任务调度器已停止（当前 running=%s）", self.running_job_id or "-")

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
            logger.info("任务取消（排队中）：id=%s kind=%s", job_id, job["kind"])
            return updated
        if state == "running":
            self._cancel_requested.add(job_id)
            await self.store.set_cancel_requested(job_id)
            logger.info("任务取消（运行中，尽力而为）：id=%s kind=%s", job_id, job["kind"])
            return await self.store.get(job_id)
        return job

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
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
            "max_parallel": self.cfg.jobs_max_parallel,
            "aging_seconds": self.cfg.jobs_aging_seconds,
            "timeout_seconds": self.cfg.jobs_timeout,
            "recover_mode": self.cfg.jobs_recover,
            "paused": False,
        }
        return base

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
                job = await self.store.pick_next(
                    current_kind=self.current_kind,
                    aging_seconds=self.cfg.jobs_aging_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - 数据库异常不能让循环退出
                logger.error("调度器取任务失败：%s", exc)
                job = None
            if job is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.cfg.jobs_poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 单个任务异常不能带崩调度器
                logger.exception("调度任务时发生未预期错误：%s", exc)
        logger.debug("调度循环退出")

    async def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        kind = job["kind"]
        action = job["action"]
        if not await self.store.mark_running(job_id):
            # 在取出与认领之间被取消了。
            return

        self.running_job_id = job_id
        if kind != self.current_kind:
            if self.current_kind is not None:
                # 新批次大小 = 还在排队的同 kind 任务 + 当前刚领到的这一个。
                queued_new = await self.store.queued_count(kind) + 1
                logger.info(
                    "调度切换：%s→%s（批次 %d/%d）",
                    self.current_kind,
                    kind,
                    self.batch_streak,
                    queued_new,
                )
            self.current_kind = kind
            self.batch_streak = 1
        else:
            self.batch_streak += 1

        wait_ms = 0.0
        try:
            created = datetime.fromisoformat(job["created_at"])
            wait_ms = max(0.0, (datetime.now().astimezone() - created).total_seconds() * 1000)
        except (TypeError, ValueError):
            pass
        model = self._model_of(kind)
        logger.info(
            "任务开始：id=%s kind=%s action=%s model=%s 等待=%.0fms 显存=%s",
            job_id,
            kind,
            action,
            model,
            wait_ms,
            self._vram_text(),
        )

        exec_task = asyncio.create_task(self.executor(job, self.ctx))
        started_perf = asyncio.get_running_loop().time()
        error: Optional[BaseException] = None
        result: Optional[dict[str, Any]] = None
        try:
            done, _pending = await asyncio.wait({exec_task}, timeout=self.cfg.jobs_timeout)
            if not done:
                message = f"任务超时（{self.cfg.jobs_timeout:.0f} 秒）"
                await self.store.fail(job_id, message)
                error = JobTimeoutError(message)
                logger.error(
                    "任务超时：id=%s kind=%s 超过 %.0fs，已置 failed；等待底层调用退出",
                    job_id,
                    kind,
                    self.cfg.jobs_timeout,
                )
                # GPU 调用无法强杀：等它真正退出，避免下一个任务和它抢显存。
                with contextlib.suppress(BaseException):
                    await exec_task
            else:
                result = exec_task.result()
        except asyncio.CancelledError:
            exec_task.cancel()
            with contextlib.suppress(BaseException):
                await exec_task
            raise
        except BaseException as exc:  # noqa: BLE001 - 记录到 DB，再由 waiter 重抛
            error = exc
            await self.store.fail(job_id, f"{type(exc).__name__}: {exc}")
        finally:
            if exec_task is not None and not exec_task.done():
                exec_task.cancel()
                with contextlib.suppress(BaseException):
                    await exec_task
            self.running_job_id = None

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
