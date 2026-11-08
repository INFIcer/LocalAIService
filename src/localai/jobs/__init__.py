"""任务队列：持久化任务记录 + 批处理调度器 + 执行器。

对上层（``localai.api``）只暴露三样东西::

    JobStore        SQLite 任务库（<HOME>/data/jobs.db）
    JobScheduler    单 worker 调度循环
    execute_job     把一个任务交给对应能力适配器

所有任务状态都在数据库里，服务重启不丢；同步旧接口通过
``JobScheduler.submit()`` + ``wait()`` 复用同一套执行链路。
"""

from __future__ import annotations

from .executors import (
    DEFAULT_ACTION,
    KIND_ACTIONS,
    JobContext,
    execute_job,
    prepare_payload,
)
from .scheduler import JobScheduler
from .store import JOB_STATES, TERMINAL_STATES, JobStore, NewJob

__all__ = [
    "DEFAULT_ACTION",
    "JOB_STATES",
    "KIND_ACTIONS",
    "TERMINAL_STATES",
    "JobContext",
    "JobScheduler",
    "JobStore",
    "NewJob",
    "execute_job",
    "prepare_payload",
]
