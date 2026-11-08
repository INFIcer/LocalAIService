"""字幕接口测试：列内嵌 / 外挂软字幕、抽 segments / srt、双语去重。

全部素材由 ffmpeg 现场生成（10 秒小视频），不加载模型、不读大文件。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.config import Config, get_config

_FFMPEG = get_config().ffmpeg
pytestmark = pytest.mark.skipif(
    _FFMPEG is None or not Path(_FFMPEG).is_file(), reason="本机没有可用的 ffmpeg"
)

_EMBEDDED_SRT = """1
00:00:00,500 --> 00:00:02,500
你好，世界

2
00:00:03,000 --> 00:00:05,500
第二句字幕

3
00:00:06,000 --> 00:00:08,000
第三句字幕
"""

_SIDECAR_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\i1}こんにちは\\N你好
Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,重复行\\N重复行
Dialogue: 0,0:00:07.00,0:00:08.50,Default,,0,0,0,,さようなら\\N再见
"""


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(
        [str(_FFMPEG), "-hide_banner", "-y", *args],
        check=True,
        capture_output=True,
    )


def _make_video_with_embedded(root: Path, *, with_subtitle: bool = True) -> Path:
    srt = root / "embedded.srt"
    srt.write_text(_EMBEDDED_SRT, encoding="utf-8")
    video = root / ("demo.mkv" if with_subtitle else "plain.mp4")
    args = [
        "-f", "lavfi",
        "-i", "color=c=navy:s=320x180:d=10:r=25",
    ]
    if with_subtitle:
        args += ["-i", str(srt), "-map", "0:v", "-map", "1:s", "-c:s", "srt"]
    args += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", "9",
        str(video),
    ]
    _ffmpeg(args)
    assert video.is_file()
    return video


@pytest.fixture(scope="module")
def sample(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("subtitles")
    video = _make_video_with_embedded(root)
    # 同名前缀外挂：双语 ASS（日文 + 中文）。
    (root / f"{video.stem}.scjp.ass").write_text(_SIDECAR_ASS, encoding="utf-8")
    plain = _make_video_with_embedded(root, with_subtitle=False)
    cfg = Config.from_env(root)
    cfg.home = root
    cfg.output_dir = root / "outputs"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(cfg)) as client:
        yield client, video, plain


def test_list_embedded_and_sidecar(sample) -> None:
    client, video, _plain = sample
    response = client.post("/media/subtitles", json={"video_path": str(video)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["has_soft_subtitle"] is True
    assert body["hint"] == "embedded"
    assert body["embedded"], body
    stream = body["embedded"][0]
    assert stream["codec"] == "subrip"
    assert stream["index"] == 1
    assert body["sidecar"]
    assert body["sidecar"][0]["format"] == "ass"
    assert body["sidecar"][0]["language_hint"] == "zh-ja"


def test_list_upload_multipart(sample) -> None:
    client, video, _plain = sample
    response = client.post(
        "/media/subtitles",
        files={"file": ("demo.mkv", video.read_bytes(), "video/x-matroska")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["uploaded"] is True
    assert body["embedded"][0]["codec"] == "subrip"


def test_list_hardcoded_or_none(sample) -> None:
    client, _video, plain = sample
    response = client.post("/media/subtitles", json={"video_path": str(plain)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["has_soft_subtitle"] is False
    assert body["embedded"] == []
    assert body["sidecar"] == []
    assert body["hint"] == "hardcoded_or_none"


def test_extract_embedded_segments(sample) -> None:
    client, video, _plain = sample
    listing = client.post("/media/subtitles", json={"video_path": str(video)}).json()
    index = listing["embedded"][0]["index"]
    response = client.post(
        "/media/subtitles/extract",
        json={"video_path": str(video), "index": index},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["source"] == "embedded"
    assert body["count"] == 3
    assert body["segments"][0]["text"] == "你好，世界"
    assert body["language"] == "und"


def test_extract_sidecar_bilingual_and_srt(sample) -> None:
    client, video, _plain = sample
    sidecar = video.parent / f"{video.stem}.scjp.ass"
    response = client.post(
        "/media/subtitles/extract",
        json={"sidecar": str(sidecar)},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "sidecar"
    assert body["language"] == "zh-ja"
    assert [item["text"] for item in body["segments"]] == ["你好", "重复行", "再见"]

    srt_response = client.post(
        "/media/subtitles/extract",
        json={"sidecar": str(sidecar), "format": "srt"},
    )
    assert srt_response.status_code == 200, srt_response.text
    saved = srt_response.json()
    path = Path(saved["path"])
    assert path.is_file() and path.parent.name == "subtitles"
    content = path.read_text(encoding="utf-8")
    assert "你好" in content and "こんにちは" not in content
    assert "重复行\n重复行" not in content


def test_extract_rejects_missing_index(sample) -> None:
    client, video, _plain = sample
    response = client.post(
        "/media/subtitles/extract",
        json={"video_path": str(video)},
    )
    assert response.status_code == 400
    assert "index" in response.json()["detail"]
