"""服务层冒烟测试：不加载任何 GPU 模型，跑得快、可在 CI 跑。

覆盖：
- 包/配置/版本；
- 应用装配与全部旧接口路径存在（OpenAPI）；
- /health、/capabilities、/voices、/vram、/docs；
- 参数校验、错误码、显存卸载空请求；
- 媒体探测（有 ffmpeg 时）。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localai import API_VERSION, __version__
from localai.api.app import app
from localai.config import CONFIG, get_config
from localai.core.audio import audio_headers
from localai.core.voices import scan_voices

# 任务要求必须保持兼容的全部路径。
REQUIRED_PATHS = {
    "/health",
    "/voices",
    "/tts",
    "/v1/audio/speech",
    "/api/v1/voices",
    "/api/v1/tts/tasks",
    "/api/v1/upload",
    "/asr",
    "/api/v1/asr",
    "/asr/status",
    "/speakers",
    "/speakers/status",
    "/speakers/unknown",
    "/speakers/enroll",
    "/speakers/identify",
    "/speakers/{name}",
    "/separate",
    "/api/v1/separate",
    "/separate/status",
    "/media/probe",
    "/media/to_wav",
    "/media/frames",
    "/media/audio",
    "/vram",
    "/vram/unload",
    "/capabilities",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_version_and_api_version() -> None:
    assert __version__
    assert API_VERSION == "1"


def test_config_paths_are_derived_from_home() -> None:
    cfg = get_config()
    assert cfg.home == CONFIG.home
    assert cfg.voices_dir.parent == cfg.home or str(cfg.voices_dir).startswith(str(cfg.home))
    assert cfg.model_dir.name == "IndexTTS-2.5"
    assert cfg.ffmpeg is None or isinstance(cfg.ffmpeg, Path)


def test_all_legacy_paths_registered(client: TestClient) -> None:
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    paths = set(openapi.json().get("paths", {}))
    missing = REQUIRED_PATHS - paths
    assert not missing, f"缺少接口：{sorted(missing)}"


def test_health_shape(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    for key in (
        "ok",
        "backend",
        "model_ready",
        "engine_loaded",
        "voices",
        "asr",
        "speaker",
        "separate",
        "vram",
        "media",
    ):
        assert key in body
    assert body["backend"] in ("engine", "comfy")
    assert "free_mb" in body["vram"]
    # 新服务启动后必须仍是 0 模型（测试本身不会触发加载）。
    assert body["engine_loaded"] is False
    assert body["asr"]["loaded"] is False
    assert body["speaker"]["loaded"] is False
    assert body["separate"]["loaded"] is False


def test_capabilities(client: TestClient) -> None:
    response = client.get("/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["api_version"] == API_VERSION
    assert body["service"]["version"] == __version__
    assert body["listen"]["port"] == get_config().port
    assert "auth" in body and "required" in body["auth"]
    caps = body["capabilities"]
    assert {"tts", "asr", "speaker", "separate", "media"} <= set(caps)
    for name in ("tts", "asr", "speaker", "separate"):
        assert caps[name]["requires_gpu"] is True
        assert "ready" in caps[name] and "loaded" in caps[name]
        assert "endpoints" in caps[name]
    assert caps["media"]["requires_gpu"] is False
    assert caps["asr"]["parallel"]["capacity"] == get_config().asr_parallel
    assert body["jobs"]["asr_parallel"] == get_config().asr_parallel


def test_asr_status_exposes_parallel_config(client: TestClient) -> None:
    response = client.get("/asr/status")
    assert response.status_code == 200
    body = response.json()
    assert body["parallel"]["capacity"] == get_config().asr_parallel
    assert body["parallel"]["env"] == "LOCALAI_ASR_PARALLEL"


def test_voices(client: TestClient) -> None:
    response = client.get("/voices")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["voices"]) == len(scan_voices())


def test_vram_and_empty_unload(client: TestClient) -> None:
    response = client.get("/vram")
    assert response.status_code == 200
    body = response.json()
    assert {"free_mb", "total_mb", "models", "margin_mb"} <= set(body)
    names = {model["name"] for model in body["models"]}
    assert {"tts", "asr", "speaker", "separate"} <= names

    empty = client.post("/vram/unload", json={})
    assert empty.status_code == 200
    assert empty.json()["released"] == []

    bad = client.post("/vram/unload", json={"models": ["nope"]})
    assert bad.status_code == 400


def test_tts_rejects_unknown_voice_without_loading(client: TestClient) -> None:
    response = client.post(
        "/tts",
        json={"text": "你好", "voice": "__definitely_missing__"},
    )
    assert response.status_code == 404
    assert response.json()["error"] == "voice_not_found"
    # 失败发生在加载之前：显存里仍然没有 TTS。
    assert client.get("/vram").json()["models"][0]["loaded"] is False


def test_speech_rejects_non_wav_format(client: TestClient) -> None:
    response = client.post(
        "/v1/audio/speech",
        json={"input": "你好", "voice": "default", "response_format": "mp3"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_response_format"


def test_asr_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/asr",
        files={"file": ("x.txt", b"hello", "text/plain")},
        data={"language": "auto"},
    )
    assert response.status_code == 400


def test_speaker_upload_rejects_bad_extension(client: TestClient) -> None:
    response = client.post(
        "/speakers/identify",
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_docs_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_audio_headers_ascii_fallback(tmp_path: Path) -> None:
    headers = audio_headers(tmp_path / "中文_1.wav", 7)
    assert headers["X-Seed"] == "7"
    assert "Content-Disposition" in headers


@pytest.mark.skipif(get_config().ffmpeg is None, reason="本机没有 ffmpeg")
def test_media_probe_smoke(client: TestClient, tmp_path: Path) -> None:
    wav_path = tmp_path / "tiny.wav"
    sf.write(str(wav_path), [0.0] * 800, 16000, subtype="PCM_16")
    response = client.post(
        "/media/probe",
        files={"file": ("tiny.wav", wav_path.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["duration"] > 0


def test_package_imports_without_loading_models() -> None:
    """导入服务层不应把 torch 拉进进程（/health 才会按需用 nvidia-smi）。"""
    import sys

    assert "torch" not in sys.modules


def test_config_export_sets_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = get_config()
    cfg.export_env()
    import os

    assert os.environ["INDEX_TTS_MODEL_DIR"] == str(cfg.model_dir)
    assert os.environ["INDEX_TTS_ASR_IDLE_UNLOAD"] == str(cfg.asr_idle_unload)
    assert json.dumps({"ok": True})  # 确认 json 可用（导入健康）
