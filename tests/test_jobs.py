"""任务队列测试：三层排序、同类并行、OOM 降级、持久化恢复、取消、统计与旧接口兼容。

测试策略
--------
- 用 ``TestClient`` 真打 HTTP，但把调度器的执行器换成 ``FakeExecutor``：
  它按 kind 返回合法的结果结构并记录完成顺序，**不加载任何 GPU 模型**，
  因此测试快、稳定、可在无显卡环境跑。
- 需要「所有任务都在队列里再开跑」的用例（a/c）先把任务**预写进 jobs 库**，
  再启动应用，避免和调度器的 0.2 秒轮询抢时序。
- 每个用例用独立的 ``tmp_path`` 作为 ``LOCALAI_HOME``，任务库互不污染。
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.config import Config
from localai.jobs.executors import DEFAULT_ACTION
from localai.jobs.store import JobStore, NewJob


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeExecutor:
    """记录完成顺序 / 并发峰值的假执行器。

    - ``gate``：只卡住第一个任务，方便在「批次进行中」提交新任务；
    - ``oom_first_call``：第一次被调用时抛 CUDA OOM，用于验证降级重试。
    """

    def __init__(
        self,
        root: Path,
        *,
        delay: float = 0.1,
        gate: threading.Event | None = None,
        oom_first_call: bool = False,
    ) -> None:
        self.root = root
        self.delay = delay
        self.gate = gate
        self.oom_first_call = oom_first_call
        self.calls: list[str] = []          # 被调用的 job id（含重试）
        self.finished: list[str] = []       # 完成顺序（kind）
        self.sequence: list[str] = []       # 完成顺序（job id）
        self.tags: list[str] = []           # 完成顺序（tts 文本 / kind）
        self.oom_raised: list[str] = []     # 抛过 OOM 的 job id
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self._counter = 0

    def _wav(self, kind: str) -> Path:
        self._counter += 1
        path = self.root / f"{kind}_{self._counter}.wav"
        sf.write(str(path), [0.0] * 800, 16000, subtype="PCM_16")
        return path

    async def __call__(self, job: dict[str, Any], ctx: Any) -> dict[str, Any]:
        job_id = job["id"]
        kind = job["kind"]
        self.calls.append(job_id)
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            gate = self.gate
            if gate is not None:
                # 只卡住第一个任务；to_thread 不阻塞事件循环，测试线程可以继续提交。
                self.gate = None
                await asyncio.to_thread(gate.wait, 10)
            if self.oom_first_call:
                self.oom_first_call = False
                self.oom_raised.append(job_id)
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            if self.delay:
                await asyncio.sleep(self.delay)
            self.finished.append(kind)
            self.sequence.append(job_id)
            self.tags.append(job.get("payload", {}).get("text") or kind)
            return self._result(kind)
        finally:
            with self._lock:
                self.active -= 1

    def _result(self, kind: str) -> dict[str, Any]:
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


class StubVram:
    """给调度器并行显存检查用的假管家。"""

    def __init__(self, free_mb: int, *, margin_mb: int = 512) -> None:
        self.free_mb = free_mb
        self.margin_mb = margin_mb

    def snapshot(self) -> dict[str, Any]:
        return {"free_mb": self.free_mb, "margin_mb": self.margin_mb}

    def loaded_names(self) -> list[str]:
        return []

    async def unload_models(self, names: list[str] | None = None, *, unload_all: bool = False):
        return {"ok": True, "released": list(names or [])}

    def release_cuda_cache(self) -> None:
        return None

    def health_memory(self) -> tuple[int, int]:
        return (self.free_mb, 8192)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path, **overrides: Any) -> Config:
    cfg = Config.from_env(home=str(tmp_path))
    cfg = dataclasses.replace(
        cfg,
        jobs_timeout=float(overrides.pop("jobs_timeout", 8.0)),
        jobs_poll_interval=0.03,
        **overrides,
    )
    cfg.export_env()
    return cfg


def _seed_jobs(cfg: Config, entries: list[tuple[str, int, dict[str, Any]]]) -> list[str]:
    """在应用启动前把任务直接写进 SQLite，保证调度器一次看到完整队列。"""

    async def _run() -> list[str]:
        store = JobStore(cfg.jobs_db_path)
        await store.init()
        ids: list[str] = []
        try:
            for kind, priority, payload in entries:
                job_id = uuid.uuid4().hex
                await store.create(
                    NewJob(
                        id=job_id,
                        kind=kind,
                        action=DEFAULT_ACTION[kind],
                        payload=payload,
                        priority=priority,
                    )
                )
                ids.append(job_id)
        finally:
            await store.close()
        return ids

    return asyncio.run(_run())


@contextmanager
def _running_app(
    tmp_path: Path,
    executor: FakeExecutor,
    *,
    seed: list[tuple[str, int, dict[str, Any]]] | None = None,
    **cfg_overrides: Any,
) -> Iterator[tuple[TestClient, Any, Config, list[str]]]:
    """起一个真实 FastAPI app（临时 HOME），执行器在调度器启动前就换好。"""
    env_backup = dict(os.environ)
    cfg = _make_cfg(tmp_path, **cfg_overrides)
    preload_ids = _seed_jobs(cfg, seed) if seed else []
    # 关键：在 lifespan 启动调度器之前打好补丁，避免真实执行器先领走预置任务。
    with mock.patch("localai.jobs.scheduler.execute_job", executor):
        app = create_app(cfg)
        try:
            with TestClient(app) as client:
                yield client, app, cfg, preload_ids
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


def _asr_payload(tmp_path: Path, name: str = "asr_input.wav") -> dict[str, Any]:
    path = tmp_path / name
    if not path.exists():
        sf.write(str(path), [0.0] * 800, 16000, subtype="PCM_16")
    return {"file": str(path), "language": "auto", "speaker": False}


def _separate_payload(tmp_path: Path) -> dict[str, Any]:
    """通用 /jobs 的分离 payload：prepare_payload 要求文件真实存在。"""
    path = tmp_path / "sep_input.wav"
    if not path.exists():
        sf.write(str(path), [0.0] * 800, 16000, subtype="PCM_16")
    return {"file": str(path), "model": "htdemucs", "stems": ["vocals"]}


def _force_parallel_ready(app: Any, free_mb: int) -> None:
    """让 ASR 看起来已加载、显存充足（FakeExecutor 不会真的加载模型）。"""
    asr = app.state.registry.get("asr")
    asr.loaded = lambda: True  # type: ignore[method-assign]
    app.state.scheduler.vram = StubVram(free_mb)


# ---------------------------------------------------------------------------
# a) 同类型优先：先提交 3 个低优先级 separate，再 3 个 priority=100 的 tts
#    第一个任务按优先级选出 tts -> 必须先把 3 个 tts 全跑完才轮到 separate
# ---------------------------------------------------------------------------

def test_kind_first_beats_higher_priority_other_kind(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.02)
    seed: list[tuple[str, int, dict[str, Any]]] = [
        ("separate", 0, _separate_payload(tmp_path)) for _ in range(3)
    ] + [("tts", 100, _empty_tts()) for _ in range(3)]
    with _running_app(tmp_path, executor, seed=seed) as (client, app, cfg, ids):
        _wait_all_done(client, ids)

        # 实际执行顺序：tts×3（同类优先）-> separate×3。
        assert executor.finished == ["tts", "tts", "tts", "separate", "separate", "separate"]
        assert executor.sequence == ids[3:] + ids[:3]

        stats = client.get("/jobs/stats").json()
        assert stats["scheduler"]["current_kind"] == "separate"
        assert stats["scheduler"]["batch_streak"] == 3
        assert "aging_seconds" not in stats["scheduler"]


# ---------------------------------------------------------------------------
# b) 同类中间插入：tts 批次跑到一半，priority=999 的 separate 也不能插队
# ---------------------------------------------------------------------------

def test_no_priority_jump_into_running_kind_batch(tmp_path: Path) -> None:
    gate = threading.Event()
    executor = FakeExecutor(tmp_path, delay=0.05, gate=gate)
    seed = [("tts", 0, _empty_tts(f"t{i}")) for i in range(3)]
    with _running_app(tmp_path, executor, seed=seed) as (client, app, cfg, ids):
        _wait_state(client, ids[0], {"running"})

        # 批次进行中插入一个优先级 999 的分离任务。
        intruder = _submit(client, "separate", _separate_payload(tmp_path), priority=999)
        time.sleep(0.25)
        # 它必须还在排队，不能抢在 3 个 tts 之前。
        assert client.get(f"/jobs/{intruder}").json()["state"] == "queued"

        gate.set()
        _wait_all_done(client, [*ids, intruder])
        assert executor.finished == ["tts", "tts", "tts", "separate"]


# ---------------------------------------------------------------------------
# c) 同类里按优先级：0/5/2 + 一个更早提交的同优先级 -> 5,2,0（更早）,0（更晚）
# ---------------------------------------------------------------------------

def test_same_kind_priority_then_fifo(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.02)
    seed = [
        ("tts", 0, _empty_tts("A_early_p0")),
        ("tts", 5, _empty_tts("B_p5")),
        ("tts", 2, _empty_tts("C_p2")),
        ("tts", 0, _empty_tts("D_late_p0")),
    ]
    with _running_app(tmp_path, executor, seed=seed) as (client, app, cfg, ids):
        _wait_all_done(client, ids)
        assert executor.tags == ["B_p5", "C_p2", "A_early_p0", "D_late_p0"]
        assert executor.sequence == [ids[1], ids[2], ids[0], ids[3]]


# ---------------------------------------------------------------------------
# d) 同类并行：ASR 并行度 2 时 running 峰值到 2；显存不够时回落到 1
# ---------------------------------------------------------------------------

def test_asr_parallel_reaches_two(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.3)
    from localai.engines.asr_sensevoice import AsrAdapter

    with mock.patch.object(AsrAdapter, "CAN_INFER_IN_PARALLEL", True):
        with _running_app(tmp_path, executor, jobs_parallel_asr=2) as (client, app, cfg, _seed):
            _force_parallel_ready(app, free_mb=6000)
            ids = [_submit(client, "asr", _asr_payload(tmp_path)) for _ in range(4)]

            peak = 0
            deadline = time.monotonic() + 12.0
            done = 0
            while time.monotonic() < deadline:
                stats = client.get("/jobs/stats").json()
                peak = max(peak, stats["scheduler"]["running"])
                if stats["done"] == 4:
                    break
                time.sleep(0.03)
            _wait_all_done(client, ids)

            assert executor.peak == 2, f"执行器并发峰值={executor.peak}"
            assert peak == 2, f"调度器 running 峰值={peak}"
            stats = client.get("/jobs/stats").json()
            assert stats["scheduler"]["per_kind_parallel"]["asr"] == 2


def test_parallel_falls_back_when_vram_low(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.2)
    from localai.engines.asr_sensevoice import AsrAdapter

    with mock.patch.object(AsrAdapter, "CAN_INFER_IN_PARALLEL", True):
        with _running_app(tmp_path, executor, jobs_parallel_asr=2) as (client, app, cfg, _seed):
            # 空闲 700MB - 余量 512MB = 188MB < 每个并行任务预留 600MB。
            _force_parallel_ready(app, free_mb=700)
            ids = [_submit(client, "asr", _asr_payload(tmp_path)) for _ in range(3)]
            _wait_all_done(client, ids)
            assert executor.peak == 1


def test_parallel_oom_degrades_to_serial_and_retries_once(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.05, oom_first_call=True)
    from localai.engines.asr_sensevoice import AsrAdapter

    with mock.patch.object(AsrAdapter, "CAN_INFER_IN_PARALLEL", True):
        with _running_app(tmp_path, executor, jobs_parallel_asr=2) as (client, app, cfg, _seed):
            _force_parallel_ready(app, free_mb=6000)
            ids = [_submit(client, "asr", _asr_payload(tmp_path)) for _ in range(4)]
            _wait_all_done(client, ids, timeout=15.0)

            # 撞过一次 OOM，降级到 1 后重试成功；进程仍在正常服务。
            assert len(executor.oom_raised) == 1
            assert executor.calls.count(executor.oom_raised[0]) == 2  # 原次 + 重试一次
            stats = client.get("/jobs/stats").json()
            assert stats["scheduler"]["per_kind_parallel"]["asr"] == 1
            assert "asr" in stats["scheduler"]["parallel_degraded"]
            assert client.get("/health").json()["ok"] is True


# ---------------------------------------------------------------------------
# e) 持久化：服务重启后队列不丢；running 按 requeue 重新排队
# ---------------------------------------------------------------------------

def test_restart_requeues_running_and_keeps_queue(tmp_path: Path) -> None:
    slow = FakeExecutor(tmp_path, delay=30.0)
    cfg = _make_cfg(tmp_path)

    with _running_app(tmp_path, slow) as (client, app, _cfg, _ids):
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
    with _running_app(tmp_path, restarted) as (client, app, _cfg, _ids):
        _wait_all_done(client, ids, timeout=15.0)
        running_job = client.get(f"/jobs/{running_id}").json()
        assert running_job["state"] == "done"
        assert running_job["attempts"] == 2  # 第一次被重启打断，第二次成功
        assert {job_id for job_id in ids} <= {job["id"] for job in client.get("/jobs").json()["jobs"]}


def test_restart_fail_mode_marks_running_failed(tmp_path: Path) -> None:
    slow = FakeExecutor(tmp_path, delay=30.0)
    cfg = _make_cfg(tmp_path)
    with _running_app(tmp_path, slow) as (client, app, _cfg, _ids):
        job_id = _submit(client, "tts", _empty_tts())
        _wait_state(client, job_id, {"running"})

    # 模拟「崩溃时任务还在 running」：进程退出后记录仍是 running。
    with _running_app(tmp_path, FakeExecutor(tmp_path), jobs_recover="fail") as (client, app, _cfg, _ids):
        job = client.get(f"/jobs/{job_id}").json()
        assert job["state"] == "failed"
        assert "服务重启中断" in (job["error"] or "")


# ---------------------------------------------------------------------------
# f) 取消：queued 立即取消；running 标记后不写结果
# ---------------------------------------------------------------------------

def test_cancel_queued_and_running(tmp_path: Path) -> None:
    gate = threading.Event()
    executor = FakeExecutor(tmp_path, delay=0.05, gate=gate)
    with _running_app(tmp_path, executor) as (client, app, cfg, _seed):
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
# g) stats 字段 + 旧同步接口仍返回原内容
# ---------------------------------------------------------------------------

def test_stats_fields_and_sync_endpoints_use_queue(tmp_path: Path) -> None:
    executor = FakeExecutor(tmp_path, delay=0.01)
    with _running_app(tmp_path, executor) as (client, app, cfg, _seed):
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

        # /jobs/stats 字段齐全；aging 已彻底移除，换成了 per_kind_parallel。
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
        for key in (
            "current_kind",
            "batch_streak",
            "loaded_model",
            "per_kind_parallel",
            "parallel_extra_mb",
            "parallel_degraded",
            "concurrency",
        ):
            assert key in stats["scheduler"], key
        assert "aging_seconds" not in stats["scheduler"]
        per_kind = stats["scheduler"]["per_kind_parallel"]
        # 翻译是新增能力；ASR 的实际并行度跟随配置（默认 2，可设 1）。
        assert set(per_kind) == {
            "tts",
            "asr",
            "separate",
            "speaker",
            "media",
            "translate",
            "download",
            "analyze",
        }
        assert per_kind["asr"] == cfg.asr_parallel_effective()
        assert per_kind["translate"] == 1
        assert stats["scheduler"]["parallel_extra_mb"] == 600
        assert stats["by_kind"] == {"tts": 1, "asr": 1, "separate": 1}
        assert stats["avg_run_ms"] >= 0

        # /capabilities 里的 jobs 也换成新字段。
        caps = client.get("/capabilities").json()
        assert caps["jobs"]["per_kind_parallel"]["asr"] == cfg.asr_parallel_effective()
        assert caps["jobs"]["per_kind_parallel"]["translate"] == 1
        assert "aging_seconds" not in caps["jobs"]

        # /health 也带上队列信息。
        health = client.get("/health").json()
        assert health["jobs"] == {"queued": 0, "running": 0, "current_kind": "separate"}
