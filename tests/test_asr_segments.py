"""``/asr/segments`` 与分段整理逻辑测试。

覆盖四件事（全部用假引擎，不加载真实模型）：
1. ``normalize_segments``：乱序 / 重叠 / 非法项 -> 升序、无重叠、``end > start``；
2. sensevoice 分支：``timestamp_source`` 必须是 ``vad``；
3. whisper 分支：``timestamp_source`` 是 ``native``，响应带 ``engine``；
4. 非法 ``engine`` 返回 400（不会落盘、不会加载模型）。
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localai.api.app import app
from localai.config import Config
from localai.core.segments import normalize_segments
from localai.engines import build_registry
from localai.engines.asr_fasterwhisper import WhisperAdapter
from localai.engines.asr_sensevoice import AsrAdapter
from localai.jobs.executors import JobContext, execute_job


@pytest.fixture(scope="module")
def client():
    """不进入 lifespan：分段路由只依赖 app.state.cfg/registry。

    启动完整 lifespan 会创建调度器/看门狗并把 ``LOCALAI_ASR_PARALLEL`` 写回
    环境变量，污染其它测试文件；这里保持进程环境不变。
    """
    return TestClient(app)


@pytest.fixture()
def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, [0.0] * 8000, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 纯函数：分段整理
# ---------------------------------------------------------------------------


def test_normalize_segments_sorts_and_removes_overlap() -> None:
    raw = [
        {"start": 2.0, "end": 3.0, "text": "第二段"},
        {"start": 0.5, "end": 1.8, "text": "第一段"},
        {"start": 1.5, "end": 2.5, "text": "重叠段"},
        {"start": 4.0, "end": 3.5, "text": "非法段"},
        {"start": 3.2, "end": 3.2, "text": "零长度"},
        {"start": 4.1, "end": 5.0, "text": "  "},
    ]
    segments = normalize_segments(raw)

    assert [item["start"] for item in segments] == sorted(item["start"] for item in segments)
    previous_end = -1.0
    for item in segments:
        assert item["end"] > item["start"]
        assert item["start"] >= previous_end - 1e-9, "分段之间不能重叠"
        previous_end = item["end"]
    # 非法 / 零长度被丢掉；重叠项被推到上一段结束之后。
    assert len(segments) == 4
    # 第三段原本 2.0~3.0，被上一段（推到 2.5 结束）压缩成 2.5~3.0。
    assert segments[2]["start"] == 2.5


def test_normalize_segments_clamps_to_duration() -> None:
    segments = normalize_segments(
        [{"start": 0.0, "end": 9.0, "text": "超出"}, {"start": -1.0, "end": 1.0, "text": "负数"}],
        duration=5.0,
    )
    assert segments[0]["start"] == 0.0
    assert segments[-1]["end"] == 5.0


# ---------------------------------------------------------------------------
# /asr/segments：两个引擎
# ---------------------------------------------------------------------------


def test_segments_sensevoice_uses_vad(
    client: TestClient, wav_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_infer_segments(self, audio_path: str, language: str = "auto"):
        return {
            "text": "你好世界",
            "language": "zh",
            "duration": 3.0,
            "segments": [
                {"start": 1.2, "end": 3.0, "text": "世界"},
                {"start": 0.0, "end": 1.2, "text": "你好"},
            ],
            "timestamp_source": "vad",
            "engine": "sensevoice",
            "elapsed": 0.1,
            "device": "cuda:0",
        }

    monkeypatch.setattr(AsrAdapter, "infer_segments", fake_infer_segments)
    response = client.post(
        "/asr/segments",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"engine": "sensevoice", "language": "zh"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["engine"] == "sensevoice"
    assert body["timestamp_source"] == "vad"
    assert [seg["start"] for seg in body["segments"]] == [0.0, 1.2]
    assert all(seg["end"] > seg["start"] for seg in body["segments"])


def test_segments_whisper_returns_native(
    client: TestClient, wav_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_infer_segments(self, audio_path: str, language: str = "auto"):
        return {
            "text": "hello world",
            "language": "en",
            "duration": 3.2,
            "segments": [
                {"start": 0.0, "end": 1.4, "text": " hello"},
                {"start": 1.4, "end": 3.2, "text": " world"},
            ],
            "timestamp_source": "native",
            "engine": "whisper",
            "elapsed": 0.2,
            "device": "cuda",
        }

    monkeypatch.setattr(WhisperAdapter, "infer_segments", fake_infer_segments)
    response = client.post(
        "/asr/segments",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"engine": "whisper"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "whisper"
    assert body["timestamp_source"] == "native"
    assert len(body["segments"]) == 2
    assert body["segments"][-1]["end"] == 3.2


def test_segments_auto_prefers_whisper(
    client: TestClient, wav_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []

    async def fake_whisper(self, audio_path: str, language: str = "auto"):
        called.append("whisper")
        return {
            "text": "x",
            "language": "zh",
            "duration": 1.0,
            "segments": [{"start": 0.0, "end": 1.0, "text": "x"}],
            "timestamp_source": "native",
            "engine": "whisper",
        }

    monkeypatch.setattr(WhisperAdapter, "infer_segments", fake_whisper)
    response = client.post(
        "/asr/segments",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"engine": "auto"},
    )
    assert response.status_code == 200, response.text
    assert called == ["whisper"]
    assert response.json()["engine"] == "whisper"


def test_segments_rejects_unknown_engine(client: TestClient, wav_bytes: bytes) -> None:
    response = client.post(
        "/asr/segments",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"engine": "__nope__"},
    )
    assert response.status_code == 400
    assert "engine" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /asr：engine 参数与响应字段（直接调执行器，不经过调度器 / 不加载模型）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("sensevoice", "sensevoice"), ("whisper", "whisper")],
)
def test_asr_engine_selection(
    tmp_path: Path,
    wav_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    expected: str,
) -> None:
    async def fake_sense(self, audio_path: str, language: str = "auto", **_: object):
        return {"text": "假识别", "language": "zh", "duration": 1.0, "timestamp_source": "vad"}

    async def fake_whisper(self, audio_path: str, language: str = "auto", **_: object):
        return {"text": "fake", "language": "zh", "duration": 1.0, "timestamp_source": "native"}

    monkeypatch.setattr(AsrAdapter, "infer", fake_sense)
    monkeypatch.setattr(WhisperAdapter, "infer", fake_whisper)

    wav_path = tmp_path / "a.wav"
    wav_path.write_bytes(wav_bytes)
    cfg = Config.from_env(home=str(tmp_path))
    registry = build_registry(cfg)
    job = {
        "kind": "asr",
        "action": "asr.transcribe",
        "payload": {
            "file": str(wav_path),
            "language": "zh",
            "engine": requested,
            "timestamps": False,
            "source_format": ".wav",
            "original_name": "a.wav",
        },
    }
    result = asyncio.run(execute_job(job, JobContext(cfg=cfg, registry=registry)))
    assert result["engine"] == expected
    assert result["ok"] is True
