"""``/media/scenes`` 场景边界检测测试（ffmpeg 合成视频，不加载任何模型）。

合成一段 4.5 秒、三色（黑 -> 白 -> 蓝）各 1.5 秒的视频，真实切换点在
1.5s / 3.0s；ffmpeg 的 ``pts_time`` 精度是帧级（25fps = 0.04s），所以断言
误差 < 0.2 秒。同时覆盖最小间隔去抖与 ``max_scenes`` 截断。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localai.api.app import app
from localai.config import get_config
from localai.engines._native import import_engine_module

_FFMPEG = get_config().ffmpeg
pytestmark = pytest.mark.skipif(
    _FFMPEG is None or not Path(_FFMPEG).is_file(), reason="本机没有可用的 ffmpeg"
)

#: 三色视频里真实的镜头切换时间点（秒）。
TRUE_BOUNDARIES = (1.5, 3.0)


def _make_tricolor_video(root: Path) -> Path:
    """黑 / 白 / 蓝各 1.5 秒 -> concat 成 4.5 秒视频（切换点稳定可测）。"""
    clips: list[Path] = []
    for color in ("black", "white", "blue"):
        clip = root / f"{color}.mp4"
        subprocess.run(
            [
                str(_FFMPEG),
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=320x240:d=1.5",
                "-r",
                "25",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ],
            check=True,
            capture_output=True,
        )
        clips.append(clip)

    list_file = root / "list.txt"
    list_file.write_text(
        "".join(f"file '{clip.as_posix()}'\n" for clip in clips), encoding="utf-8"
    )
    video = root / "tricolor.mp4"
    subprocess.run(
        [
            str(_FFMPEG),
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    assert video.is_file()
    return video


@pytest.fixture(scope="module")
def ffmpeg_tool():
    return import_engine_module("ffmpeg_tool")


@pytest.fixture(scope="module")
def tricolor_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_tricolor_video(tmp_path_factory.mktemp("scenes"))


def test_debounce_keeps_min_gap(ffmpeg_tool) -> None:
    points = [(1.0, 0.5), (1.2, 0.6), (1.4, 0.4), (2.5, 0.3)]
    kept = ffmpeg_tool._debounce_scenes(points, min_gap=1.0)
    assert [item[0] for item in kept] == [1.0, 2.5]
    # 间隔调小后都保留
    assert len(ffmpeg_tool._debounce_scenes(points, min_gap=0.0)) == 4


def test_detect_scenes_finds_boundaries(
    ffmpeg_tool, tricolor_video: Path, tmp_path: Path
) -> None:
    result = ffmpeg_tool.detect_scenes(
        str(tricolor_video), threshold=0.3, min_gap=1.0, save_dir=str(tmp_path)
    )
    assert result["count"] == 2
    times = [item["time"] for item in result["scenes"]]
    for found, truth in zip(times, TRUE_BOUNDARIES):
        assert abs(found - truth) <= 0.2, f"边界误差过大：{found} vs {truth}"
    assert all(item["score"] >= 0.3 for item in result["scenes"])

    json_path = Path(result["json_path"])
    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["count"] == 2
    assert len(payload["scenes"]) == 2


def test_detect_scenes_respects_max_scenes(ffmpeg_tool, tricolor_video: Path) -> None:
    result = ffmpeg_tool.detect_scenes(
        str(tricolor_video), threshold=0.3, min_gap=0.0, max_scenes=1
    )
    assert result["count"] == 1
    result_all = ffmpeg_tool.detect_scenes(
        str(tricolor_video), threshold=0.3, min_gap=0.0, max_scenes=200
    )
    assert result_all["count"] == 2


def test_media_scenes_endpoint() -> None:
    root = Path(tempfile.mkdtemp(prefix="localai_scenes_endpoint_"))
    video_path = _make_tricolor_video(root)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/media/scenes",
            files={"file": ("tricolor.mp4", video_path.read_bytes(), "video/mp4")},
            data={"threshold": "0.3", "min_gap": "1.0"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["count"] == 2
    assert body["json_path"] and Path(body["json_path"]).is_file()
    for item in body["scenes"]:
        assert 0 <= item["time"] <= (body["duration"] or 0) + 0.1
        assert "score" in item
