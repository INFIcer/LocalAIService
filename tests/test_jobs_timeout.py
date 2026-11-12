"""任务超时硬上限：置 failed 后立刻释放并发名额，不阻塞调度循环。

覆盖点
------
- 执行器卡住时，任务在 jobs_timeout 后置 failed，错误信息含「任务超时」；
- 第一个任务超时后，第二个任务能马上开始并完成（名额已释放）；
- 宽限期内取消不掉的执行任务被登记到 scheduler._stuck，结束后自动移除。
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient
from unittest import mock

from localai.api.app import create_app
from localai.config import Config
from localai.jobs.scheduler import JobScheduler


def _make_cfg(tmp_path: Path, **overrides: Any) -> Config:
    cfg = Config.from_env(home=str(tmp_path))
    defaults: dict[str, Any] = {
        "jobs_timeout": 0.2,
        "jobs_poll_interval": 0.03,
        "jobs_retention_days": 0,
        "output_retention_days": 0,
        "tmp_retention_hours": 0,
    }
    defaults.update(overrides)
    return dataclasses.replace(cfg, **defaults)


class SlowFirstExecutor:
    """第一次调用挂住 30 秒（可取消），之后立即返回。"""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, job: dict[str, Any], ctx: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            await asyncio.sleep(30)
        return {"ok": True}


class SlowCancelExecutor(SlowFirstExecutor):
    """第一次调用在被取消后还要 0.8 秒才退出，模拟卡在不可中断的底层调用里。"""

    async def __call__(self, job: dict[str, Any], ctx: Any) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0.8)
                raise
        return {"ok": True}


@contextmanager
def _client(
    tmp_path: Path, executor: Any, **overrides: Any
) -> Iterator[tuple[TestClient, Any]]:
    env_backup = dict(os.environ)
    cfg = _make_cfg(tmp_path, **overrides)
    with mock.patch.object(
        JobScheduler, "_TIMEOUT_GRACE_SECONDS", 0.1
    ), mock.patch("localai.jobs.scheduler.execute_job", executor):
        app = create_app(cfg)
        try:
            with TestClient(app) as client:
                yield client, app
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


def _submit_tts(client: TestClient, text: str) -> str:
    response = client.post("/jobs/tts", json={"text": text, "voice": "default"})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _wait_state(
    client: TestClient, job_id: str, states: set[str], timeout: float = 8.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("state") in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"任务 {job_id} 没在 {timeout}s 内进入 {states}")


def test_timeout_fails_job_and_frees_slot(tmp_path: Path) -> None:
    executor = SlowFirstExecutor()
    with _client(tmp_path, executor) as (client, app):
        first = _submit_tts(client, "超时任务")
        job = _wait_state(client, first, {"failed"}, timeout=5.0)
        assert "任务超时" in (job["error"] or "")

        scheduler = app.state.scheduler
        # 可取消的执行任务在宽限期内退出，不登记为 stuck。
        assert scheduler._stuck == {}

        second = _submit_tts(client, "下一个任务")
        _wait_state(client, second, {"done"}, timeout=3.0)
        assert executor.calls == 2


def test_stuck_call_tracked_and_cleaned_up(tmp_path: Path) -> None:
    executor = SlowCancelExecutor()
    with _client(tmp_path, executor) as (client, app):
        job_id = _submit_tts(client, "卡住的任务")
        failed = _wait_state(client, job_id, {"failed"}, timeout=5.0)
        assert "任务超时" in (failed["error"] or "")

        scheduler = app.state.scheduler
        # 宽限期 0.1s 后仍未退出 -> 登记；0.8s 后真正结束 -> 自动移除。
        registered = False
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if scheduler._stuck:
                registered = True
                break
            time.sleep(0.02)
        assert registered, "取消不掉的执行任务应被登记到 _stuck"

        deadline = time.monotonic() + 3.0
        while scheduler._stuck and time.monotonic() < deadline:
            time.sleep(0.02)
        assert scheduler._stuck == {}
