"""保留期清理（任务历史、outputs、临时目录）的测试。

覆盖点
------
- JobStore.prune 只删除超过保留期的终态记录，queued/running/新终态记录保留；
- 0 天表示不清理；
- run_retention 按 mtime 清理过期 outputs 与临时目录条目，新文件不受影响；
- retention_loop 可被 cancel 正常退出（生命周期成对）。
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from localai.config import Config
from localai.core import retention
from localai.jobs.store import JobStore, NewJob


def _make_cfg(tmp_path: Path, **overrides: Any) -> Config:
    cfg = Config.from_env(home=str(tmp_path))
    defaults: dict[str, Any] = {
        "jobs_retention_days": 0,
        "output_retention_days": 0,
        "tmp_retention_hours": 0,
    }
    defaults.update(overrides)
    return dataclasses.replace(cfg, **defaults)


async def _seed(cfg: Config) -> dict[str, str]:
    store = JobStore(cfg.jobs_db_path)
    await store.init()
    try:
        ids: dict[str, str] = {}

        async def create(state: str) -> str:
            job_id = f"job-{state}-{len(ids)}-{time.time_ns()}"
            await store.create(
                NewJob(id=job_id, kind="tts", action="tts.synthesize", payload={})
            )
            if state == "queued":
                pass
            elif state == "running":
                assert await store.mark_running(job_id)
            elif state == "done":
                assert await store.mark_running(job_id)
                assert await store.finish(job_id, {"path": "out.wav"})
            elif state == "failed":
                assert await store.mark_running(job_id)
                assert await store.fail(job_id, "boom")
            elif state == "canceled":
                assert await store.mark_running(job_id)
                assert await store.mark_canceled(job_id)
            else:
                raise AssertionError(state)
            return job_id

        for state in ("done", "failed", "canceled"):
            ids[f"old_{state}"] = await create(state)
        ids["queued"] = await create("queued")
        ids["running"] = await create("running")
        ids["new_done"] = await create("done")
        return ids
    finally:
        await store.close()


def _backdate(cfg: Config, job_id: str, when: datetime) -> None:
    stamp = when.isoformat(timespec="milliseconds")
    connection = sqlite3.connect(str(cfg.jobs_db_path))
    try:
        connection.execute(
            "UPDATE jobs SET created_at = ? WHERE id = ?", (stamp, job_id)
        )
        connection.execute(
            "UPDATE jobs SET started_at = ? WHERE id = ? AND started_at IS NOT NULL",
            (stamp, job_id),
        )
        connection.execute(
            "UPDATE jobs SET finished_at = ? WHERE id = ? AND finished_at IS NOT NULL",
            (stamp, job_id),
        )
        connection.commit()
    finally:
        connection.close()


def test_prune_removes_only_expired_terminal_jobs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    ids = asyncio.run(_seed(cfg))
    old = datetime.now().astimezone() - timedelta(days=40)
    new = datetime.now().astimezone()
    for key in ("old_done", "old_failed", "old_canceled", "queued", "running"):
        _backdate(cfg, ids[key], old)
    _backdate(cfg, ids["new_done"], new)

    async def _run() -> tuple[int, dict[str, Any], int]:
        store = JobStore(cfg.jobs_db_path)
        await store.init()
        try:
            deleted = await store.prune(30)
            rows = {key: await store.get(job_id) for key, job_id in ids.items()}
            # 0 = 不清理。
            zero = await store.prune(0)
            return deleted, rows, zero
        finally:
            await store.close()

    deleted, rows, zero = asyncio.run(_run())
    assert deleted == 3
    assert zero == 0
    for key in ("old_done", "old_failed", "old_canceled"):
        assert rows[key] is None, key
    assert rows["queued"]["state"] == "queued"
    assert rows["running"]["state"] == "running"
    assert rows["new_done"]["state"] == "done"


def test_run_retention_sweeps_old_outputs_and_temp(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, output_retention_days=30, tmp_retention_hours=24)
    store = JobStore(cfg.jobs_db_path)  # jobs_retention_days=0，不会碰数据库
    expired = time.time() - 40 * 86400

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old_output = output_dir / "old.wav"
    old_output.write_bytes(b"old")
    new_output = output_dir / "new.wav"
    new_output.write_bytes(b"new")
    os.utime(old_output, (expired, expired))

    frames = Path(cfg.frames_dir)
    frames.mkdir(parents=True, exist_ok=True)
    old_frame_dir = frames / "vision_old"
    old_frame_dir.mkdir()
    old_frame = old_frame_dir / "f0.jpg"
    old_frame.write_bytes(b"x")
    os.utime(old_frame, (expired, expired))
    os.utime(old_frame_dir, (expired, expired))
    new_frame_dir = frames / "vision_new"
    new_frame_dir.mkdir()

    result = asyncio.run(retention.run_retention(cfg, store))
    assert result["outputs"] == 1
    assert result["tmp"] >= 1
    assert not old_output.exists()
    assert new_output.exists()
    assert not old_frame_dir.exists()
    assert new_frame_dir.exists()


def test_retention_loop_can_be_cancelled(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    store = JobStore(cfg.jobs_db_path)

    async def _scenario() -> None:
        task = asyncio.create_task(
            retention.retention_loop(cfg, store, interval_seconds=60.0)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_scenario())
