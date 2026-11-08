"""同步接口复用任务队列的小工具。

旧接口（``/tts``、``/asr``、``/separate`` …）的语义是「请求即任务、等结果」。
为了让这些请求也留下持久化记录，路由内部改成::

    job_id = scheduler.submit(...)
    job = scheduler.wait(job_id, timeout=LOCALAI_JOB_TIMEOUT + 30)

成功时从 ``job["result"]`` 还原成旧的响应体；失败时 :meth:`JobScheduler.wait`
会把**原始异常**重新抛出，因此旧的 4xx/5xx 处理器仍然生效。
"""

from __future__ import annotations

import shutil
from typing import Any, Optional

from fastapi import Request

from ..config import Config
from ..jobs import JobScheduler
from .deps import get_cfg, get_scheduler

#: 等待任务完成时在任务超时之外额外留的余量（秒）。
WAIT_MARGIN_SECONDS = 30.0


def client_ip(request: Request) -> str:
    """记录调用方 IP（127.0.0.1:xxxxx -> 127.0.0.1）。"""
    try:
        return request.client.host if request.client else ""
    except Exception:  # noqa: BLE001
        return ""


def sync_wait_timeout(cfg: Config, kind: str = "") -> float:
    """同步接口等待上限：按任务类型取超时，再留一点余量。"""
    return float(cfg.job_timeout_for(kind)) + WAIT_MARGIN_SECONDS


async def submit_and_wait(
    request: Request,
    kind: str,
    action: str,
    payload: dict[str, Any],
    *,
    priority: int = 0,
    job_id: Optional[str] = None,
) -> dict[str, Any]:
    """提交一个任务并等它完成；返回 done 状态的 job 记录。"""
    scheduler: JobScheduler = get_scheduler(request)
    cfg = get_cfg(request)
    created_id = await scheduler.submit(
        kind,
        action,
        payload,
        priority=priority,
        client_ip=client_ip(request),
        job_id=job_id,
    )
    return await scheduler.wait(created_id, timeout=sync_wait_timeout(cfg, kind))


def cleanup_job_upload(cfg: Config, job_id: str) -> None:
    """参数校验失败时清掉刚落盘的上传目录。"""
    shutil.rmtree(cfg.jobs_upload_dir / job_id, ignore_errors=True)
