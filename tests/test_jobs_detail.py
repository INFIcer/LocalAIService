"""任务详情（WebUI 双击弹窗）依赖的接口测试。

覆盖点
------
- ``GET /jobs/{id}`` 必须带齐 payload / result / error / started_at /
  finished_at / attempts / client_ip，文本字段保留换行原文；
- ``GET /jobs/{id}/result`` 对 ``result.path`` 统一返回文件（音频 / 图片 /
  其它按 MIME，inline 展示），不再只支持音频；
- 结果路径必须在 ``outputs/``、``data/downloads/``、``data/uploads/`` 之下，
  越界返回 403 而不是 500；文件已删除时回退到 JSON 结果；
- ``/ui`` 与 ``app.js`` 里确实带任务详情弹窗的选择器和函数名。

测试不加载任何模型：任务直接写进临时 ``LOCALAI_HOME`` 的 SQLite，状态设为
终态，调度器启动时不会重新领取。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.config import Config
from localai.jobs.store import JobStore, NewJob


def _make_cfg(tmp_path: Path) -> Config:
    cfg = Config.from_env(home=str(tmp_path))
    cfg = dataclasses.replace(cfg, jobs_timeout=5.0, jobs_poll_interval=0.03)
    cfg.export_env()
    return cfg


def _seed_job(
    tmp_path: Path,
    cfg: Config,
    *,
    kind: str = "tts",
    action: str = "tts.synthesize",
    payload: dict[str, Any] | None = None,
    state: str = "done",
    result: dict[str, Any] | None = None,
    error: str | None = None,
    attempts: int = 1,
    client_ip: str = "127.0.0.1",
) -> str:
    """直接写库造一条终态任务（绕过调度器，不触发任何模型）。"""

    job_id = uuid.uuid4().hex

    async def _create() -> None:
        store = JobStore(cfg.jobs_db_path)
        await store.init()
        try:
            await store.create(
                NewJob(
                    id=job_id,
                    kind=kind,
                    action=action,
                    payload=payload or {},
                    client_ip=client_ip,
                )
            )
        finally:
            await store.close()

    import asyncio

    asyncio.run(_create())

    created = "2026-01-02T03:04:05.000+08:00"
    started = None if state == "queued" else "2026-01-02T03:04:07.000+08:00"
    finished = None if state not in ("done", "failed", "canceled") else "2026-01-02T03:04:09.000+08:00"
    connection = sqlite3.connect(str(cfg.jobs_db_path))
    try:
        connection.execute(
            "UPDATE jobs SET state = ?, created_at = ?, started_at = ?,"
            " finished_at = ?, result = ?, error = ?, attempts = ? WHERE id = ?",
            (
                state,
                created,
                started,
                finished,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                error,
                attempts,
                job_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return job_id


@contextmanager
def _client(tmp_path: Path) -> Iterator[tuple[TestClient, Config]]:
    env_backup = dict(os.environ)
    cfg = _make_cfg(tmp_path)
    app = create_app(cfg)
    try:
        with TestClient(app) as client:
            yield client, cfg
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


# ---------------------------------------------------------------------------
# GET /jobs/{id}：字段补齐
# ---------------------------------------------------------------------------

def test_job_detail_has_payload_result_and_timestamps(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    text = "第一行文本\n第二行文本，保留换行。"
    result = {"path": "E:/LocalAIService/outputs/demo.wav", "duration_seconds": 1.25}
    job_id = _seed_job(
        tmp_path,
        cfg,
        payload={"text": text, "voice": "default", "speed": 1.0},
        result=result,
        attempts=2,
    )

    with _client(tmp_path) as (client, _cfg):
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()

        assert job["ok"] is True
        assert job["id"] == job_id
        assert job["kind"] == "tts"
        assert job["action"] == "tts.synthesize"
        assert job["state"] == "done"
        assert job["payload"]["text"] == text
        assert job["result"] == result
        assert job["error"] is None
        assert job["started_at"] and job["finished_at"]
        assert job["attempts"] == 2
        assert job["client_ip"] == "127.0.0.1"
        assert isinstance(job["meta"], dict)

        # 列表接口同样带 payload/result，前端详情接口失败时可以兜底。
        listed = client.get("/jobs?limit=10").json()["jobs"]
        assert listed[0]["payload"]["text"] == text


def test_failed_job_detail_keeps_full_error_text(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    error = (
        'voice "__definitely_missing__" 不存在\n'
        "报错位置: jobs/executors.py:123\n"
        "Traceback (most recent call last): ..."
    )
    job_id = _seed_job(
        tmp_path,
        cfg,
        state="failed",
        payload={"text": "测试", "voice": "__definitely_missing__"},
        error=error,
    )

    with _client(tmp_path) as (client, _cfg):
        job = client.get(f"/jobs/{job_id}").json()
        assert job["state"] == "failed"
        assert job["error"] == error
        assert "\n" in job["error"]  # 换行原样保留，弹窗里可完整看到


# ---------------------------------------------------------------------------
# GET /jobs/{id}/result：按 MIME 下发文件 + 目录白名单
# ---------------------------------------------------------------------------

def test_result_serves_audio_image_and_other_files(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    wav = Path(cfg.output_dir) / "demo.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"RIFF....WAVEfmt ")

    png = Path(cfg.downloads_dir) / "2026-01-02" / "frame.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)

    txt = Path(cfg.jobs_upload_dir) / "job1" / "notes.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("普通文件", encoding="utf-8")

    jobs = {
        "audio": _seed_job(tmp_path, cfg, result={"path": str(wav)}),
        "image": _seed_job(tmp_path, cfg, kind="vision", result={"path": str(png)}),
        "other": _seed_job(tmp_path, cfg, kind="download", result={"path": str(txt)}),
    }

    with _client(tmp_path) as (client, _cfg):
        audio = client.get(f"/jobs/{jobs['audio']}/result")
        assert audio.status_code == 200, audio.text
        assert audio.headers["content-type"].startswith("audio/")
        assert "inline" in audio.headers["content-disposition"]
        assert audio.content == wav.read_bytes()

        image = client.get(f"/jobs/{jobs['image']}/result")
        assert image.status_code == 200, image.text
        assert image.headers["content-type"] == "image/png"
        assert image.content == png.read_bytes()

        other = client.get(f"/jobs/{jobs['other']}/result")
        assert other.status_code == 200, other.text
        assert other.headers["content-type"].startswith("text/plain")
        assert other.content == txt.read_bytes()


def test_result_rejects_path_outside_allowed_dirs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    outside = tmp_path / "outside" / "secret.wav"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"RIFFsecret")
    job_id = _seed_job(tmp_path, cfg, result={"path": str(outside)})

    with _client(tmp_path) as (client, _cfg):
        response = client.get(f"/jobs/{job_id}/result")
        assert response.status_code == 403, response.text
        body = response.json()
        assert body["ok"] is False
        assert body["error"] == "result_path_not_allowed"
        assert "outputs/" in body["detail"]


def test_result_missing_file_falls_back_to_json(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    missing = Path(cfg.output_dir) / "not_there.wav"
    result = {"path": str(missing), "note": "文件已被清理"}
    job_id = _seed_job(tmp_path, cfg, result=result)

    with _client(tmp_path) as (client, _cfg):
        response = client.get(f"/jobs/{job_id}/result")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True and body["state"] == "done"
        assert body["result"] == result


def test_result_requires_done_and_known_job(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    queued = _seed_job(tmp_path, cfg, state="queued", result=None)

    with _client(tmp_path) as (client, _cfg):
        pending = client.get(f"/jobs/{queued}/result")
        assert pending.status_code == 409
        assert pending.json()["error"] == "job_not_done"

        unknown = client.get("/jobs/not-a-real-id")
        assert unknown.status_code == 404
        assert unknown.json()["detail"].startswith("任务不存在")


# ---------------------------------------------------------------------------
# 前端静态资源：详情弹窗的选择器 / 函数名确实存在
# ---------------------------------------------------------------------------

def test_webui_ships_job_detail_modal(tmp_path: Path) -> None:
    with _client(tmp_path) as (client, _cfg):
        page = client.get("/ui")
        assert page.status_code == 200
        assert "modal-job" in page.text
        assert "job-modal-body" in page.text

        script = client.get("/ui/app.js")
        assert script.status_code == 200
        for marker in (
            "openJobDetail",
            "closeJobDetail",
            "jobDetailOpen",       # 弹窗打开时暂停 3 秒自动刷新
            "dblclick",            # 双击任务行
            "data-copy-key",       # 一键复制 id / 路径 / 错误
            "job-modal-cancel",    # 弹窗内取消任务
            "/vision/frame?path=",  # 图片缩略图数据源
        ):
            assert marker in script.text, marker
