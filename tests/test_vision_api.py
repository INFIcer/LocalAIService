"""视觉接口测试：/vision/status、参数校验，以及标记 slow 的真实模型测试。

默认 ``pytest -q`` 会跳过 slow（见 pyproject 的 ``-m 'not slow'``），
需要真实模型时用 ``pytest -m slow tests/test_vision_api.py``。
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from localai.api.app import create_app
from localai.engines.vision_vlm import VisionAdapter


def _client() -> TestClient:
    return TestClient(create_app())


def test_vision_status_fields() -> None:
    with _client() as client:
        response = client.get("/vision/status")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        for key in (
            "ready",
            "loaded",
            "model",
            "model_dir",
            "runtime",
            "port",
            "vram_estimate_mb",
            "planner_default",
            "prompts",
            "endpoints",
        ):
            assert key in data, f"缺少字段 {key}"
        assert data["planner_default"] in ("hybrid", "agent")
        assert data["prompts"]["score"] is True
        assert data["prompts"]["agent"] is True
        assert data["prompts"]["plan"] is True


def test_describe_rejects_non_image() -> None:
    with _client() as client:
        response = client.post(
            "/vision/describe",
            files={"file": ("note.txt", b"this is not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "图片" in response.json()["detail"]


def test_frames_requires_video_or_job() -> None:
    with _client() as client:
        response = client.post("/vision/frames", json={"target_frames": 3})
        assert response.status_code == 400
        assert "video_path" in response.json()["detail"] or "job_id" in response.json()["detail"]


def test_frames_rejects_missing_file() -> None:
    with _client() as client:
        response = client.post(
            "/vision/frames",
            json={"video_path": "Z:/definitely/not/here.mp4", "target_frames": 3},
        )
        assert response.status_code == 400
        assert "不存在" in response.json()["detail"]


def test_scan_rejects_missing_paths() -> None:
    with _client() as client:
        response = client.post(
            "/vision/scan", json={"images": ["Z:/no/such/frame.jpg"], "task": "score"}
        )
        assert response.status_code == 400


def _blank_jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 180), (250, 250, 250)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_scan_multipart_multiple_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """multipart 多图要真的被当成 UploadFile（曾因类型判断踩过坑）。"""

    async def fake_describe_many(self, images, prompt, **_kwargs):  # noqa: ANN001
        return {"texts": [f"ok-{index}" for index in range(len(images))], "elapsed_ms": 1}

    monkeypatch.setattr(VisionAdapter, "describe_many", fake_describe_many)
    data = _blank_jpeg()
    with _client() as client:
        response = client.post(
            "/vision/scan",
            files=[
                ("file", ("a.jpg", data, "image/jpeg")),
                ("file", ("b.jpg", data, "image/jpeg")),
            ],
            data={"task": "score"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["count"] == 2
        assert [item["text"] for item in payload["results"]] == ["ok-0", "ok-1"]


@pytest.mark.slow
def test_describe_real_model_blank_frame() -> None:
    """真实模型：纯白图应被判 blank / usable=false（用评分提示词那套）。"""
    with _client() as client:
        response = client.post(
            "/vision/describe",
            files={"file": ("blank.jpg", _blank_jpeg(), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["ok"] is True
        assert data["text"]
        parsed = data.get("parsed") or {}
        assert parsed.get("usable") is False
        assert parsed.get("kind") in ("blank", "other", "transition")
