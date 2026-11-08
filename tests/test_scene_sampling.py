"""场景检测采样模式测试：mock ffmpeg 调用，不读真实视频。

验证：

- >15 分钟默认切成 8 个 60 秒窗口，每个窗口单独跑、时间点加窗口偏移；
- 单窗口失败只跳过该窗口并记 warning；
- mode="full" 强制整片检测。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from localai.engines._native import import_engine_module


@pytest.fixture(scope="module")
def ffmpeg_tool():
    return import_engine_module("ffmpeg_tool")


def _stdout(pts: float = 5.0, score: float = 0.9) -> str:
    return (
        f"frame=1 pts_time:{pts:.6f}\n"
        f"lavfi.scene_score={score:.6f}\n"
    )


def test_sampled_windows_and_offsets(
    ffmpeg_tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_probe(_path: str) -> dict[str, Any]:
        return {"duration": 3600.0, "video": {"codec": "h264"}}

    def fake_run(args, *, timeout=None, probe=False):  # noqa: ANN001
        calls.append({"args": list(args), "timeout": timeout})
        return SimpleNamespace(stdout=_stdout(), stderr="")

    monkeypatch.setattr(ffmpeg_tool, "probe", fake_probe)
    monkeypatch.setattr(ffmpeg_tool, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(ffmpeg_tool, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg_tool, "_run", fake_run)
    # 文件存在性校验是纯本地检查，用临时空文件即可。
    import tempfile
    from pathlib import Path

    video = Path(tempfile.mkdtemp(prefix="localai_scene_sample_")) / "movie.mkv"
    video.write_bytes(b"fake")

    result = ffmpeg_tool.detect_scenes(str(video), mode="auto")
    assert result["mode"] == "sampled"
    assert len(result["windows"]) == 8
    assert result["count"] == 8
    assert len(calls) == 8
    assert all(call["timeout"] == 45.0 for call in calls)
    for window, scene, call in zip(result["windows"], result["scenes"], calls):
        assert window["start"] >= 10.0
        assert window["end"] <= 3600.0 - 10.0 + 0.01
        assert window["end"] - window["start"] == pytest.approx(60.0, abs=0.02)
        assert scene["time"] == pytest.approx(window["start"] + 5.0, abs=0.01)
        assert "-ss" in call["args"] and "-t" in call["args"]
    # 窗口之间不重叠，覆盖片头片尾以外的整段时间。
    starts = [item["start"] for item in result["windows"]]
    assert starts == sorted(starts)


def test_sampled_window_timeout_is_skipped(
    ffmpeg_tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_probe(_path: str) -> dict[str, Any]:
        return {"duration": 2000.0, "video": {"codec": "h264"}}

    def fake_run(args, *, timeout=None, probe=False):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 3:
            raise ffmpeg_tool.MediaError("ffmpeg.exe 执行超时（>45 秒）")
        return SimpleNamespace(stdout=_stdout(pts=10.0), stderr="")

    monkeypatch.setattr(ffmpeg_tool, "probe", fake_probe)
    monkeypatch.setattr(ffmpeg_tool, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(ffmpeg_tool, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg_tool, "_run", fake_run)
    import tempfile
    from pathlib import Path

    video = Path(tempfile.mkdtemp(prefix="localai_scene_timeout_")) / "movie.mkv"
    video.write_bytes(b"fake")

    result = ffmpeg_tool.detect_scenes(str(video))
    assert result["mode"] == "sampled"
    assert result["count"] == 7
    assert len(result["warnings"]) == 1
    assert "已跳过" in result["warnings"][0]
    assert "error" in result["windows"][2]
    assert calls["count"] == 8


def test_4k_long_video_uses_keyframe_packets(
    ffmpeg_tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4K 超长片：8 个窗口只读关键帧包，不再全帧 decode。"""
    calls: list[list[str]] = []

    def fake_probe(_path: str) -> dict[str, Any]:
        return {
            "duration": 8520.0,
            "video": {"codec": "hevc", "width": 3840, "height": 2160},
        }

    def fake_run(args, *, timeout=None, probe=None):  # noqa: ANN001
        calls.append(list(args))
        interval = args[args.index("-read_intervals") + 1]
        start = float(interval.split("%")[0])
        # 窗口内两个关键帧；绝对时间戳。
        stdout = f"{start + 2:.3f},K__\n{start + 12:.3f},___\n{start + 22:.3f},K__\n"
        return SimpleNamespace(stdout=stdout, stderr="")

    monkeypatch.setattr(ffmpeg_tool, "probe", fake_probe)
    monkeypatch.setattr(ffmpeg_tool, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(ffmpeg_tool, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg_tool, "_run", fake_run)
    import tempfile
    from pathlib import Path

    video = Path(tempfile.mkdtemp(prefix="localai_scene_4k_")) / "movie.mkv"
    video.write_bytes(b"fake")

    result = ffmpeg_tool.detect_scenes(str(video))
    assert result["mode"] == "sampled"
    assert result["detector"] == "keyframe"
    assert len(calls) == 8
    assert all("-read_intervals" in call for call in calls)
    # 每个窗口 2 个关键帧，8 个窗口 16 个边界。
    assert result["count"] == 16
    assert all(scene["score"] == 1.0 for scene in result["scenes"])


def test_full_mode_runs_once(
    ffmpeg_tool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_probe(_path: str) -> dict[str, Any]:
        return {"duration": 3600.0, "video": {"codec": "h264"}}

    def fake_run(args, *, timeout=None, probe=False):  # noqa: ANN001
        calls.append({"args": list(args), "timeout": timeout})
        return SimpleNamespace(stdout=_stdout(pts=100.0), stderr="")

    monkeypatch.setattr(ffmpeg_tool, "probe", fake_probe)
    monkeypatch.setattr(ffmpeg_tool, "ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(ffmpeg_tool, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg_tool, "_run", fake_run)
    import tempfile
    from pathlib import Path

    video = Path(tempfile.mkdtemp(prefix="localai_scene_full_")) / "movie.mkv"
    video.write_bytes(b"fake")

    result = ffmpeg_tool.detect_scenes(str(video), mode="full", timeout=123.0)
    assert result["mode"] == "full"
    assert result["count"] == 1
    assert len(calls) == 1
    assert calls[0]["timeout"] == 123.0
    assert "-ss" not in calls[0]["args"]
