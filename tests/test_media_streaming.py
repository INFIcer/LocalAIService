"""媒体转码接口的流式返回与输出上限测试。

覆盖点
------
- /media/to_wav 与 /media/audio 成功时返回完整文件字节，响应结束后临时产物被清理；
- 产物超过 media_output_max_mb 时返回 413，文案含上限值与可操作建议；
- 超限时同样不残留半截产物。
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.api.routers import media as media_router
from localai.config import Config, get_config

_BASE = get_config()
_FFMPEG = _BASE.ffmpeg
pytestmark = pytest.mark.skipif(
    _FFMPEG is None or not Path(_FFMPEG).is_file(),
    reason="本机没有可用的 ffmpeg",
)


def _make_cfg(tmp_path: Path) -> Config:
    cfg = Config.from_env(home=str(tmp_path))
    return dataclasses.replace(
        cfg,
        ffmpeg=Path(_FFMPEG),
        ffprobe=Path(_BASE.ffprobe) if _BASE.ffprobe else None,
        media_output_max_mb=0,
        jobs_retention_days=0,
        output_retention_days=0,
        tmp_retention_hours=0,
    )


@contextmanager
def _client(cfg: Config) -> Iterator[TestClient]:
    env_backup = dict(os.environ)
    app = create_app(cfg)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        os.environ.clear()
        os.environ.update(env_backup)


def _temp_wavs() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("ffmpeg_wav16k_*.wav"))


def _make_tone(root: Path) -> Path:
    target = root / "tone.wav"
    subprocess.run(
        [
            str(_FFMPEG),
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


def _post_media(client: TestClient, url: str, source: Path, **kwargs):
    with source.open("rb") as handle:
        return client.post(
            url,
            files={"file": (source.name, handle, "audio/wav")},
            **kwargs,
        )


def test_to_wav_returns_file_and_cleans_temp(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    source = _make_tone(tmp_path)
    before = _temp_wavs()

    with _client(cfg) as client:
        response = _post_media(client, "/media/to_wav", source)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/")
    assert response.content[:4] == b"RIFF"
    assert len(response.content) > 1000
    # 响应发完后，ffmpeg 生成的临时 wav 应已被删除。
    assert _temp_wavs() == before


def test_to_wav_rejects_output_over_limit(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    source = _make_tone(tmp_path)
    monkeypatch.setattr(media_router, "_output_limit_bytes", lambda _cfg: 1024)
    before = _temp_wavs()

    with _client(cfg) as client:
        response = _post_media(client, "/media/to_wav", source)

    assert response.status_code == 413, response.text
    detail = response.json()["detail"]
    assert "上限" in detail
    assert "LOCALAI_MEDIA_OUTPUT_MAX_MB" in detail
    assert _temp_wavs() == before


def test_audio_extract_returns_file_and_removes_work_dir(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    source = _make_tone(tmp_path)

    with _client(cfg) as client:
        response = _post_media(client, "/media/audio", source, data={"format": "wav"})

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/")
    assert response.content[:4] == b"RIFF"
    assert list(Path(cfg.media_tmp_dir).glob("extract_*")) == []


def test_audio_extract_rejects_output_over_limit(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    source = _make_tone(tmp_path)
    monkeypatch.setattr(media_router, "_output_limit_bytes", lambda _cfg: 1024)

    with _client(cfg) as client:
        response = _post_media(client, "/media/audio", source, data={"format": "wav"})

    assert response.status_code == 413, response.text
    assert "上限" in response.json()["detail"]
    assert list(Path(cfg.media_tmp_dir).glob("extract_*")) == []
