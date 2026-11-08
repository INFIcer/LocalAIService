"""视觉新接口测试：effort 状态、``/vision/frame`` 防穿越、``/vision/video`` 只回文字。

全部用假媒体 / 假 VLM，不加载模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import localai.api.routers.vision as vision_router
from localai.api.app import create_app
from localai.config import Config


class FakeMedia:
    def __init__(self, duration: float = 431.0) -> None:
        self.duration = duration
        self.clips: list[tuple[float, float]] = []
        self.windows: list[tuple[float, float, float]] = []

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "probe":
            return {"duration": self.duration, "video": {"codec": "h264"}}
        if name == "clip":
            start, end = float(args[1]), float(args[2])
            self.clips.append((start, end))
            target = Path(kwargs["out"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake-clip")
            return {
                "path": str(target),
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "size_mb": 0.1,
            }
        if name == "extract_frames_window":
            start, end = float(args[1]), float(args[2])
            fps = float(kwargs.get("fps") or 1.0)
            out_dir = Path(kwargs["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            self.windows.append((start, end, fps))
            frames = []
            index = 0
            while index / fps < end - start:
                point = start + index / fps
                path = out_dir / f"window_{index:04d}_{point:.3f}s.jpg"
                _jpeg(path)
                frames.append(
                    {"path": str(path), "time_sec": round(point, 3), "width": 640, "height": 360}
                )
                index += 1
                if index > 200:
                    break
            return frames
        if name == "extract_frames":
            out_dir = Path(args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            count = int(kwargs.get("count") or 1)
            frames = []
            for index in range(count):
                point = self.duration * (index + 0.5) / count
                path = out_dir / f"frame_{index:04d}_{point:.3f}s.jpg"
                _jpeg(path)
                frames.append(
                    {"path": str(path), "time_sec": round(point, 3), "width": 640, "height": 360}
                )
            return frames
        raise AssertionError(f"FakeMedia 不认识：{name}")


class FakeVision:
    def __init__(self) -> None:
        self.frames_calls = 0
        self.last_frame_count = 0

    async def describe_frames(self, images: Any, prompt: str, **kwargs: Any) -> dict[str, Any]:
        paths = list(images)
        self.frames_calls += 1
        self.last_frame_count = len(paths)
        self.last_labels = kwargs.get("labels")
        return {
            "text": json.dumps(
                {
                    "summary": "这一段在演示九款 TTS 模型的 RTF 速度对比，先放排行榜，再切到商用许可表格。",
                    "key_moments": [
                        {"time": 203.5, "what": "RTF 速度排行条形图"},
                        {"time": 241.0, "what": "商用许可三档表格"},
                        {"time": 9999.0, "what": "越界应被过滤"},
                    ],
                },
                ensure_ascii=False,
            ),
            "elapsed_ms": 1,
            "model": "fake-vlm",
        }


class FakeRegistry:
    def __init__(self, media: Any, vision: Any) -> None:
        self.media = media
        self.vision = vision

    def get(self, name: str) -> Any:
        if name == "media":
            return self.media
        if name == "vision":
            return self.vision
        raise KeyError(name)


def _jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (120, 80, 40)).save(path, format="JPEG")


def _cfg(tmp_path: Path) -> Config:
    cfg = Config.from_env(tmp_path)
    # localai.config 在 import 时会把默认 home 的配置写回环境变量，
    # 这里强制把 home / output_dir 指到 tmp_path，避免测试往仓库根目录写文件。
    cfg.home = tmp_path
    cfg.output_dir = tmp_path / "outputs"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.downloads_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture()
def patched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = _cfg(tmp_path)
    media = FakeMedia(duration=431.0)
    vision = FakeVision()
    registry = FakeRegistry(media, vision)
    monkeypatch.setattr(vision_router, "get_cfg", lambda _request: cfg)
    monkeypatch.setattr(vision_router, "get_registry", lambda _request: registry)
    return cfg, media, vision


# ---------------------------------------------------------------------------
# /vision/status 的 effort 预算
# ---------------------------------------------------------------------------

def test_status_exposes_effort_budgets() -> None:
    with TestClient(create_app()) as client:
        data = client.get("/vision/status").json()
    assert data["effort_default"] == "auto"
    assert data["effort_choices"] == ["auto", "low", "medium", "high"]
    budgets = data["effort_budgets"]
    assert budgets["auto"]["effort"] == "auto"
    assert budgets["low"]["clip_limit"] == 0
    assert budgets["medium"]["vlm_limit"] == 16
    assert budgets["high"]["clip_limit"] == 2
    assert data["prompts"]["video"] is True
    assert data["limits"]["video_frames_full"] == 64


# ---------------------------------------------------------------------------
# /vision/frame 路径穿越防护
# ---------------------------------------------------------------------------

def test_frame_endpoint_serves_only_allowed_dirs(patched) -> None:
    cfg, _media, _vision = patched
    good = cfg.output_dir / "frames" / "a.jpg"
    _jpeg(good)
    downloaded = cfg.downloads_dir / "b.jpg"
    _jpeg(downloaded)
    outside = cfg.output_dir.parent / "outside.jpg"
    _jpeg(outside)

    with TestClient(create_app()) as client:
        response = client.get("/vision/frame", params={"path": str(good)})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("image/jpeg")
        assert response.content[:2] == b"\xff\xd8"

        response = client.get("/vision/frame", params={"path": str(downloaded)})
        assert response.status_code == 200

        # 相对路径里的 .. 解析后已离开 outputs/ -> 400
        traversal = str(cfg.output_dir / "frames" / ".." / ".." / "outside.jpg")
        response = client.get("/vision/frame", params={"path": traversal})
        assert response.status_code == 400, response.text

        # 绝对路径外泄 -> 400
        response = client.get("/vision/frame", params={"path": str(outside)})
        assert response.status_code == 400

        response = client.get("/vision/frame", params={"path": "C:/Windows/win.ini"})
        assert response.status_code == 400

        response = client.get("/vision/frame", params={"path": ""})
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# /vision/video：只回文字、段级 clip、帧数上限
# ---------------------------------------------------------------------------

def test_video_segment_uses_clip_and_returns_text_only(patched, tmp_path: Path) -> None:
    cfg, media, vision = patched
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake-video")

    with TestClient(create_app()) as client:
        response = client.post(
            "/vision/video",
            json={
                "video_path": str(video),
                "start": 200,
                "end": 260,
                "fps": 1.0,
            },
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ok"] is True
    assert media.clips == [(200.0, 260.0)], "段级细看必须先 clip"
    assert data["frames_used"] <= 48
    assert data["window"] == {"start": 200.0, "end": 260.0, "seconds": 60.0}
    assert data["fps"] <= 1.0
    assert "排行" in data["summary"]
    assert [item["time"] for item in data["key_moments"]] == [203.5, 241.0]
    assert data["confidence"] == "normal"
    # 每帧都带时间标签，避免小模型只盯第一帧。
    assert vision.last_labels and vision.last_labels[0].startswith("第 1 帧 time=")
    # 只回文字：响应里不能出现图片数据。
    raw = response.text
    assert "base64" not in raw and "data:image" not in raw
    assert "images" not in data and "preview" not in data

    # 48 帧上限 + VLM 实际收到的帧数一致
    assert vision.last_frame_count == data["frames_used"]


def test_video_full_caps_at_64_frames(patched, tmp_path: Path) -> None:
    cfg, media, _vision = patched
    video = tmp_path / "long.mp4"
    video.write_bytes(b"fake-video")

    with TestClient(create_app()) as client:
        response = client.post(
            "/vision/video",
            json={"video_path": str(video), "fps": 1.0},
        )
    assert response.status_code == 200, response.text
    data = response.json()
    # 431 秒全片按 1fps 会超过画质上限 16 帧，必须自动降 fps。
    assert data["frames_used"] <= 16
    assert data["fps"] < 1.0
    assert "帧数上限 16" in data["note"]
    assert data["window"]["start"] == 0.0
    assert data["window"]["end"] == 431.0


def test_video_rejects_bad_window(patched, tmp_path: Path) -> None:
    _cfg, _media, _vision = patched
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake-video")
    with TestClient(create_app()) as client:
        response = client.post(
            "/vision/video",
            json={"video_path": str(video), "start": 30, "end": 20},
        )
    assert response.status_code == 400
    assert "end" in response.json()["detail"]
