"""任务队列测试：调度顺序、优先级、持久化恢复、取消、统计与旧接口兼容。

测试策略
--------
- 用 ``TestClient`` 真打 HTTP，但把调度器的执行器换成 ``FakeExecutor``：
  它按 kind 返回合法的结果结构并记录完成顺序，**不加载任何 GPU 模型**，
  因此测试快、稳定、可在无显卡环境跑。
- 每个用例用独立的 ``tmp_path`` 作为 ``LOCALAI_HOME``，任务库互不污染。
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.config import Config


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeExecutor:
    """记录「完成顺序」的假执行器；可选 gate 让第一个任务先卡住。"""

    def __init__(
        self,
        root: Path,
        *,
        delay: float = 0.1,
        gate: threading.Event | None = None,
    ) -> None:
        self.root = root
        self.delay = delay
        self.gate = gate
        self.calls: list[str] = []
        self.finished: list[str] = []
        self._counter = 0

    def _wav(self, kind: str) -> Path:
        self._counter += 1
        path = self.root / f"{kind}_{self._counter}.wav"
        sf.write(str(path), [0.0] * 800, 16000, subtype="PCM_16")
        return path

    async def __call__(self, job: dict[str, Any], ctx: Any) -> dict[str, Any]:
        import asyncio

        kind = job["kind"]
        self.calls.append(kind)
        gate = self.gate
        if gate is not None:
            # 只卡住第一个任务；to_thread 不阻塞事件循环，测试线程可以继续提交。
            self.gate = None
            await asyncio.to_thread(gate.wait, 10)
        if self.delay:
            await asyncio.sleep(self.delay)
        self.finished.append(kind)
        if kind == "tts":
            path = self._wav(kind)
            return {
                "path": str(path),
                "output_path": str(path),
                "seed": 12345,
                "language": "zh",
                "comfy_filename": "",
                "duration_seconds": 0.05,
                "model": "IndexTTS-2.5",
                "voice": "default",
                "text_chars": 2,
            }
        if kind == "asr":
            return {
                "ok": True,
                "text": "测试文本",
                "language": "zh",
                "emotion": None,
                "events": [],
                "duration": 0.05,
                "elapsed_ms": 3,
                "loaded": False,
                "speaker": None,
                "source_format": ".wav",
                "converted": False,
                "model": "SenseVoiceSmall",
            }
        if kind == "separate":
            out_dir = self.root / f"separate_{self._counter}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "vocals.wav"
            sf.write(str(path), [0.0] * 800, 44100, subtype="PCM_16")
            return {
                "ok": True,
                "model": "htdemucs",
                "sample_rate": 44100,
                "duration": 0.05,
                "elapsed": 0.01,
                "separation_elapsed": 0.01,
                "stems": {"vocals": str(path)},
                "dir": str(out_dir),
                "loaded": False,
                "released_models": [],
            }
        return {"ok": True}


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, **overrides: Any) -> Config:
    cfg = Config.from_env(home=str(tmp_path))
    cfg = dataclasses.replace(
        cfg,
        jobs_timeout=float(overrides.pop("jobs_timeout", 8.0)),
        jobs_poll_interval=0.03,
        jobs_aging_seconds=int(overrides.pop("jobs_aging_seconds", 60)),
        **overrides,
    )
    cfg.export_env()
    return cfg


@contextmanager
def _running_app(
    tmp_path: Path,
    executor: FakeExecutor,
    **cfg_overrides: Any,
) -> Iterator[tuple[TestClient, Any, Config]]:
    """起一个真实 FastAPI app（临时 HOME），并换上假执行器。"""
    env_backup = dict(os.environ)
    cfg = _make_cfg(tmp_path, **cfg_overrides)
    app = create_app(cfg)
    try:
        with TestClient(app) as client:
            app.state.scheduler.executor = executor
            yield client, app, cfg
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def _wait_state(
    client: TestClient,
    job_id: str,
    states: set[str],
    timeout: float = 8.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job.get("state") in states:
            return job
        time.sleep(0.02)
    raise AssertionError(f"任务 {job_id} 没在 {timeout}s 内进入 {states}：{job}")


def _wait_all_done(client: TestClient, job_ids: list[str], timeout: float = 12.0) -> None:
    for job_id in job_ids:
        _wait_state(client, job_id, {"done"}, timeout=timeout)


def _submit(client: TestClient, kind: str, payload: dict[str, Any], priority: int = 0) -> str:
    response = client.post(
        "/jobs",
        json={"kind": kind, "priority": priority, "payload": payload},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True and body["state"] == "queued"
    return body["id"]


def _empty_tts(text: str = "你好", voice: str = "default") -> dict[str, Any]:
    return {"text": text, "voice": voice, "speed": 1.0}


def _separate_payload(tmp_path: Path) -> dict[str, Any]:
    """通用 /jobs 的分离 payload：prepare_payload 要求文件真实存在。"""
    path = tmp_path / "sep_input.wav"
    if not path.exists():
        sf.write(str(path), [0.0] * 800, 16000, subtype="PCM_16")
    return {"file": str(path), "model": "htdemucs", "stems": ["vocals"]}


# ---------------------------------------------------------------------------
# a) 同 kind 批处理：4 TTS + 2 分离，执行顺序应为 tts×4 -> separate×2
# ---------------------------------------------------------------------------

def test_batching_runs_same_kind_first(tmp_path: Path) -> None:
    gate = threading.Event()
    executor = FakeExecutor(tmp_path, delay=0.08, gate=gate)
    with _running_app(tmp_path, executor) as (client, app, cfg):
        # 先提交一个 TTS 让它被领走并卡在 gate 上，保证后面 5 个都在队列里。
        first = _submit(client, "tts", _empty_tts())
        _wait_state(client, first, {"running"})

        # 打乱顺序提交：separate / tts / separate / tts / tts
        ids = [first]
        for kind in ("separate", "tts", "separate", "tts", "tts"):
            ids.append(
                _submit(
                    client,
                    kind,
                    _separate_payload(tmp_path) if kind == "separate" else _empty_tts(),
                )
            )
        gate.set()
        _wait_all_done(client, ids)

        assert executor.finished == ["tts", "tts", "tts", "tts", "separate", "separate"]

        # 从 jobs 表的 finished_at 再验证一次（持久化证据）。
        connection = sqlite3.connect(str(cfg.jobs_db_path))
        try:
            rows = connection.execute(
                "SELECT kind FROM jobs WHERE state = 'done' ORDER BY finished_at, rowid"
            ).fetchall()
        finally:
            connection.close()
        assert [row[0] for row in rows] == [
            "tts",
            "tts",
            "tts",
            "tts",
            "separate",
            "separate",
        ]

        stats = client.get("/jobs/stats").json()
        assert stats["by_kind"] == {"tts": 4, "separate": 2}
        assert stats["by_state"]["done"] == 6
        assert stats["scheduler"]["current_kind"] == "separate"
        assert stats["scheduler"]["batch_streak"] == 2


# ---------------------------------------------------------------------------
# b) 优先级：先排队的低优先级分离任务会让位给后来的高优先级 TTS
# ---------------------------------------------------------------------------

def test_priority_beats_batching_of_queued_jobs(tmp_path: Path) -> None:
    gate = threading.Event()
    executor = FakeExecutor(tmp_path, delay=0.05, gate=gate)
    with _running_app(tmp_path, executor, jobs_aging_seconds=600) as (client, app, cfg):
        # 第一个分离任务已经在跑（无法抢占，这是 GPU 调用的物理限制）。
        sep_payload = _separate_payload(tmp_path)
        running = _submit(client, "separate", sep_payload)
        _wait_state(client, running, {"running"})

        # 后续排队：两个低优先级分离 + 一个高优先级 TTS。
        low1 = _submit(client, "separate", sep_payload, priority=0)
        low2 = _submit(client, "separate", sep_payload, priority=0)
        high = _submit(client, "tts", _empty_tts(), priority=10)
        gate.set()
        _wait_all_done(client, [running, low1, low2, high])

        # 高优先级 TTS 必须排在两个「排队中的」低优先级分离之前。
        assert executor.finished == ["separate", "tts", "separate", "separate"]


# ---------------------------------------------------------------------------
# c) 持久化：服务重启后队列不丢；running 按 requeue 重新排队
# ---------------------------------------------------------------------------

def test_restart_requeues_running_and_keeps_queue(tmp_path: Path) -> None:
    slow = FakeExecutor(tmp_path, delay=30.0)
    cfg = _make_cfg(tmp_path)

    with _running_app(tmp_path, slow) as (client, app, _cfg):
        running_id = _submit(client, "tts", _empty_tts("第一个"))
        _wait_state(client, running_id, {"running"})
        queued_id = _submit(client, "separate", _separate_payload(tmp_path))
        queued_id2 = _submit(client, "tts", _empty_tts("第三个"))
        ids = [running_id, queued_id, queued_id2]

    # 进程退出后，running 记录还在，队列里的两条也没丢。
    connection = sqlite3.connect(str(cfg.jobs_db_path))
    try:
        states = dict(
            connection.execute(
                "SELECT id, state FROM jobs WHERE id IN (?, ?, ?)", ids
            ).fetchall()
        )
    finally:
        connection.close()
    assert states[running_id] == "running"
    assert states[queued_id] == "queued"
    assert states[queued_id2] == "queued"

    # 重新启动：默认 LOCALAI_JOBS_RECOVER=requeue，running 回队列并全部跑完。
    restarted = FakeExecutor(tmp_path, delay=0.01)
    with _running_app(tmp_path, restarted) as (client, app, _cfg):
        _wait_all_done(client, ids, timeout=15.0)
        running_job = client.get(f"/jobs/{running_id}").json()
        assert running_job["state"] == "done"
        assert running_job["attempts"] == 2  # 第一次被重启打断，第二次成功
        assert {job_id for job_id in ids} <= {job["id"] for job in client.get("/jobs").json()["jobs"]}


def test_restart_fail_mode_marks_running_failed(tmp_path: Path) -> None:
    slow = FakeExecutor(tmp_path, delay=30.0)
    cfg = _make_cfg(tmp_path)
    with _running_app(tmp_path, slow) as (client, app, _cfg):
        job_id = _submit(client, "tts", _empty_tts())
        _wait_state(client, job_id, {"running"})

    # 模拟「崩溃时任务还在 running」：进程退出后记录仍是 running。
    with _running_app(tmp_path, FakeExecutor(tmp_path), jobs_recover="fail") as (client, app, _cfg):
        job = client.get(f"/jobs/{job_id}").json()
        assert job["state"] == "failed"
        assert "服务重启中断" in (job["error"] or "")


# ---------------------------------------------------------------------------
# d) 取消：queued 立即取消；running 标记后不写结果
# ---------------------------------------------------------------------------

def test_cancel_queued_and_running(tmp_path: Path) -> None:
    gate = threading.Event()
    executor = FakeExecutor(tmp_path, delay=0.05, gate=gate)
    with _running_app(tmp_path, executor) as (client, app, cfg):
        running_id = _submit(client, "tts", _empty_tts("running"))
        _wait_state(client, running_id, {"running"})
        queued_id = _submit(client, "tts", _empty_tts("queued"))

        # queued：直接 canceled。
        response = client.post(f"/jobs/{queued_id}/cancel")
        assert response.status_code == 200
        assert response.json()["state"] == "canceled"
        assert client.get(f"/jobs/{queued_id}").json()["state"] == "canceled"

        # running：标记 cancel_requested；执行器退出后置 canceled 且 result 为空。
        response = client.post(f"/jobs/{running_id}/cancel")
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "running"
        assert body["cancel_requested"] is True

        gate.set()
        job = _wait_state(client, running_id, {"canceled", "done"}, timeout=8.0)
        assert job["state"] == "canceled"
        assert job["result"] is None
        assert queued_id not in [job["id"] for job in client.get("/jobs?state=done").json()["jobs"]]


# ---------------------------------------------------------------------------
# e) stats 字段 + 旧同步接口仍返回原内容
# ---------------------------------------------------------------------------

def test_stats_fields_and_sync_endpoints_use_queue(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.01)
    with _running_app(tmp_path, executor) as (client, app, cfg):
        # 旧 TTS 接口：仍返回 WAV 二进制。
        tts = client.post("/tts", json={"text": "你好", "voice": "default"})
        assert tts.status_code == 200, tts.text
        assert tts.headers["content-type"].startswith("audio/wav")
        assert tts.headers["X-Seed"] == "12345"
        assert tts.content[:4] == b"RIFF"

        wav_path = tmp_path / "input.wav"
        sf.write(str(wav_path), [0.0] * 800, 16000, subtype="PCM_16")
        wav_bytes = wav_path.read_bytes()

        # 旧 ASR 接口：仍返回原来的 JSON 结构。
        asr = client.post(
            "/asr",
            files={"file": ("input.wav", wav_bytes, "audio/wav")},
            data={"language": "auto", "speaker": "0"},
        )
        assert asr.status_code == 200, asr.text
        body = asr.json()
        assert body["ok"] is True and body["text"] == "测试文本"
        assert "elapsed_ms" in body and "converted" in body

        # 旧分离接口：仍返回原来的 JSON 结构。
        separate = client.post(
            "/separate",
            files={"file": ("input.wav", wav_bytes, "audio/wav")},
            data={"model": "htdemucs", "stems": "vocals"},
        )
        assert separate.status_code == 200, separate.text
        sep_body = separate.json()
        assert sep_body["ok"] is True
        assert sep_body["model"] == "htdemucs"
        assert "dir" in sep_body and "stems" in sep_body

        # 三个同步请求都留下了任务记录，而且都是 done。
        listed = client.get("/jobs?limit=20").json()
        assert listed["total"] == 3
        assert {job["state"] for job in listed["jobs"]} == {"done"}
        assert {job["kind"] for job in listed["jobs"]} == {"tts", "asr", "separate"}

        # /jobs/stats 字段齐全。
        stats = client.get("/jobs/stats").json()
        for key in (
            "queued",
            "running",
            "by_kind",
            "by_state",
            "avg_wait_ms",
            "avg_run_ms",
            "oldest_queued_seconds",
            "scheduler",
        ):
            assert key in stats, key
        for key in ("current_kind", "batch_streak", "loaded_model"):
            assert key in stats["scheduler"]
        assert stats["by_kind"] == {"tts": 1, "asr": 1, "separate": 1}
        assert stats["avg_run_ms"] >= 0

        # /health 也带上队列信息。
        health = client.get("/health").json()
        assert health["jobs"] == {"queued": 0, "running": 0, "current_kind": "separate"}
