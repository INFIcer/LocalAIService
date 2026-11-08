"""本地模型探测 / 搜索路径 / 配置指向采用。

全部离线、用临时目录造假模型；不加载任何真实权重，也不写用户目录。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localai.admin.discovery import (
    DiscoveryError,
    clear_cache,
    match_directory,
    scan,
)
from localai.api.app import create_app
from localai.config import Config, config_file_path

_NATIVE_MODULES = (
    "engine",
    "asr_engine",
    "whisper_engine",
    "translate_engine",
    "speaker_engine",
    "separator_engine",
    "ffmpeg_tool",
    "vision_engine",
)


@pytest.fixture(autouse=True)
def _cleanup_modules():
    saved_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("LOCALAI_", "INDEX_TTS_"))
    }
    clear_cache()
    yield
    for name in _NATIVE_MODULES:
        sys.modules.pop(name, None)
    for key in list(os.environ):
        if key.startswith(("LOCALAI_", "INDEX_TTS_")) and key not in saved_env:
            os.environ.pop(key, None)
    os.environ.update(saved_env)
    clear_cache()


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("LOCALAI_", "INDEX_TTS_", "TTS_BACKEND")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    cfg = Config.from_env(tmp_path)
    cfg.token = ""
    app = create_app(cfg)
    with TestClient(app) as test_client:
        yield test_client, cfg, tmp_path


def _files(root: Path, names: tuple[str, ...], size: int = 16) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * size)
    return root


def _tts(root: Path) -> Path:
    return _files(
        root,
        ("gpt.pth", "s2mel.pth", "codec.pth", "config.yaml"),
    )


def _asr(root: Path) -> Path:
    return _files(
        root,
        (
            "model.pt",
            "tokens.json",
            "am.mvn",
            "config.yaml",
            "chn_jpn_yue_eng_ko_spectok.bpe.model",
        ),
    )


def _whisper(root: Path) -> Path:
    return _files(
        root,
        ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt"),
    )


def _nllb(root: Path) -> Path:
    return _files(
        root,
        (
            "config.json",
            "tokenizer.json",
            "sentencepiece.bpe.model",
            "pytorch_model.bin",
        ),
    )


def _demucs(root: Path) -> Path:
    return _files(root, ("955717e8-8726e21a.th", "htdemucs.yaml"))


def _junction(link: Path, target: Path) -> bool:
    """Windows 目录 junction；失败返回 False（测试跳过）。"""
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0 and link.exists()


# ---------------------------------------------------------------------------
# 搜索路径
# ---------------------------------------------------------------------------
def test_search_paths_default_empty_and_crud(client) -> None:
    http, cfg, home = client
    response = http.get("/admin/models/search-paths")
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "paths": [],
        "count": 0,
        "source": "config",
    }
    # 默认不预置任何路径
    assert cfg.model_search_paths == []
    assert not config_file_path(home).exists()

    missing = home / "does-not-exist"
    response = http.put("/admin/models/search-paths", json={"paths": [str(missing)]})
    assert response.status_code == 400
    assert "不存在" in response.json()["detail"]

    first = home / "models-a"
    second = home / "models-b"
    first.mkdir()
    second.mkdir()

    response = http.put("/admin/models/search-paths", json={"add": str(first)})
    assert response.status_code == 200, response.text
    assert response.json()["paths"] == [str(first)]
    # 再加一次同一路径 -> 去重
    response = http.put("/admin/models/search-paths", json={"add": str(first)})
    assert response.status_code == 200
    assert response.json()["count"] == 1

    response = http.put("/admin/models/search-paths", json={"add": str(second)})
    assert response.status_code == 200
    assert response.json()["count"] == 2

    response = http.put(
        "/admin/models/search-paths", json={"remove": str(first)}
    )
    assert response.status_code == 200
    assert response.json()["paths"] == [str(second)]

    saved = json.loads(config_file_path(home).read_text(encoding="utf-8"))
    assert saved["model_search_paths"] == [str(second)]

    response = http.put(
        "/admin/models/search-paths", json={"remove": str(first)}
    )
    assert response.status_code == 400

    response = http.put("/admin/models/search-paths", json={"paths": []})
    assert response.status_code == 200
    assert response.json()["paths"] == []
    saved = json.loads(config_file_path(home).read_text(encoding="utf-8"))
    assert "model_search_paths" not in saved


def test_search_paths_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(
        "LOCALAI_MODEL_SEARCH_PATHS", f"{first}{os.pathsep}{second}"
    )
    cfg = Config.from_env(tmp_path)
    assert cfg.model_search_paths == [str(first), str(second)]
    app = create_app(cfg)
    with TestClient(app) as http:
        body = http.get("/admin/models/search-paths").json()
        assert body["count"] == 2
        assert body["source"] == "env"


def test_config_file_path_overrides_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch)
    tts = tmp_path / "tts"
    translate = tmp_path / "translate"
    whisper = tmp_path / "whisper-small"
    for path in (tts, translate, whisper):
        path.mkdir()
    config_file_path(tmp_path).write_text(
        json.dumps(
            {
                "tts_model_dir": str(tts),
                "translate_model_dir": str(translate),
                "whisper_model_dirs": {"small": str(whisper)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg = Config.from_env(tmp_path)
    assert cfg.model_dir == tts
    assert cfg.translate_weight_dir == translate
    assert cfg.whisper_weights_dir("small") == whisper
    # 环境变量优先：显式设置后忽略 config.json
    monkeypatch.setenv("LOCALAI_WHISPER_MODEL_DIRS", json.dumps({"small": str(tts)}))
    cfg2 = Config.from_env(tmp_path)
    assert cfg2.whisper_weights_dir("small") == tts


def test_search_paths_max_twenty(client) -> None:
    http, _cfg, home = client
    paths = []
    for index in range(21):
        directory = home / f"m{index}"
        directory.mkdir()
        paths.append(str(directory))
    response = http.put("/admin/models/search-paths", json={"paths": paths})
    assert response.status_code == 400
    assert "20" in response.json()["detail"]


def test_search_paths_listed_in_settings_readonly(client) -> None:
    http, _cfg, _home = client
    body = http.get("/admin/settings").json()
    item = body["settings"]["model_search_paths"]
    assert item["type"] == "list"
    assert item["value"] == []
    assert item["readonly"] is True
    assert item["source"] == "default"


def test_scan_without_paths_returns_400(client) -> None:
    http, _cfg, _home = client
    response = http.post("/admin/models/scan", json={})
    assert response.status_code == 400
    assert "搜索路径" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 签名 / 扫描
# ---------------------------------------------------------------------------
def test_signature_matching_and_partial(tmp_path: Path) -> None:
    tts = _tts(tmp_path / "IndexTTS-2.5")
    asr = _asr(tmp_path / "SenseVoiceSmall")
    whisper = _whisper(tmp_path / "faster-whisper-small")
    nllb = _nllb(tmp_path / "nllb-200-distilled-600M")
    speaker = _files(tmp_path / "campplus", ("campplus_cn_common.bin", "config.yaml"))
    demucs = _demucs(tmp_path / "checkpoints")
    vad = _files(tmp_path / "fsmn-vad", ("model.pt", "config.yaml"))
    vision = _files(
        tmp_path / "vision",
        ("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-F16.gguf"),
    )

    expected = {
        "tts": (tts, "IndexTTS-2.5"),
        "asr": (asr, "SenseVoiceSmall"),
        "whisper": (whisper, "small"),
        "translate": (nllb, "nllb-200-distilled-600M"),
        "speaker": (speaker, "campplus"),
        "separate": (demucs, "htdemucs"),
        "vad": (vad, "fsmn-vad"),
        "vision": (vision, "Qwen2.5-VL-3B"),
    }
    for capability, (directory, model) in expected.items():
        hits = match_directory(directory, verify_sizes=False)
        own = [item for item in hits if item["capability"] == capability]
        assert own, f"{capability} 未识别：{hits}"
        assert own[0]["model"] == model
        assert own[0]["confidence"] == "exact"
        assert own[0]["missing"] == []

    # 缺文件 -> partial，并列出缺什么
    (tts / "codec.pth").unlink()
    hits = match_directory(tts, verify_sizes=False)
    assert hits and hits[0]["confidence"] == "partial"
    assert "codec.pth" in hits[0]["missing"]

    # 同名但大小明显不对 -> 大小校验降级 partial
    hits = match_directory(whisper, verify_sizes=True)
    assert hits[0]["confidence"] == "partial"
    assert "大小" in hits[0]["detail"]


def test_scan_depth_noise_and_loop(tmp_path: Path) -> None:
    nested = _whisper(tmp_path / "a" / "b" / "small")
    _whisper(tmp_path / "node_modules" / "small")
    _whisper(tmp_path / "__pycache__" / "small")
    _whisper(tmp_path / ".hidden" / "small")

    shallow = scan([str(tmp_path)], max_depth=2)
    assert shallow["stats"]["matches"] == 0
    deep = scan([str(tmp_path)], max_depth=3)
    assert deep["stats"]["matches"] == 1
    assert Path(deep["matches"][0]["path"]) == nested

    loop_root = tmp_path / "loop-root"
    (loop_root / "a").mkdir(parents=True)
    _whisper(loop_root / "a" / "small")
    if not _junction(loop_root / "a" / "back", loop_root):
        pytest.skip("当前环境无法创建目录 junction")
    started = time.perf_counter()
    result = scan([str(loop_root)], max_depth=12, timeout_sec=10)
    assert time.perf_counter() - started < 5
    assert len(result["matches"]) == 1
    assert not result["stats"]["truncated"]


def test_scan_api_background_and_adopt(client) -> None:
    http, cfg, home = client
    whisper = _whisper(home / "external" / "faster-whisper-small")
    tts = _tts(home / "external" / "IndexTTS-2.5")

    response = http.put(
        "/admin/models/search-paths", json={"paths": [str(home / "external")]}
    )
    assert response.status_code == 200, response.text

    response = http.post("/admin/models/scan", json={"max_depth": 4})
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    snapshot = {}
    for _ in range(100):
        status = http.get(f"/admin/models/scan/{task_id}")
        assert status.status_code == 200, status.text
        snapshot = status.json()
        if snapshot["status"] != "running":
            break
        time.sleep(0.03)
    assert snapshot["status"] == "done", snapshot
    assert snapshot["progress"]["matches"] == 2
    assert snapshot["result"]["stats"]["visited_dirs"] >= 2
    assert {item["capability"] for item in snapshot["result"]["matches"]} == {
        "tts",
        "whisper",
    }

    # adopt whisper：写按档映射，模型清单立即反映新路径
    response = http.post(
        "/admin/models/adopt",
        json={"capability": "whisper", "path": str(whisper), "name": "small"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["config_key"] == "whisper_model_dirs"
    assert body["requires_restart"] is False
    assert cfg.whisper_weights_dir("small") == whisper

    saved = json.loads(config_file_path(home).read_text(encoding="utf-8"))
    assert saved["whisper_model_dirs"]["small"] == str(whisper)

    catalog = http.get("/admin/models").json()["models"]
    item = next(
        row for row in catalog if row["capability"] == "whisper" and row["name"] == "small"
    )
    assert item["path"] == str(whisper)

    # 用户文件只读：内容仍在，未生成链接/副本
    assert (whisper / "model.bin").read_bytes() == b"x" * 16
    assert not (whisper / "model.bin").is_symlink()

    # adopt tts：只写 tts_model_dir
    response = http.post(
        "/admin/models/adopt", json={"capability": "tts", "path": str(tts)}
    )
    assert response.status_code == 200, response.text
    assert response.json()["config_key"] == "tts_model_dir"
    assert cfg.model_dir == tts
    saved = json.loads(config_file_path(home).read_text(encoding="utf-8"))
    assert saved["tts_model_dir"] == str(tts)
    catalog = http.get("/admin/models").json()["models"]
    tts_row = next(row for row in catalog if row["capability"] == "tts")
    assert tts_row["path"] == str(tts)

    # 无签名 -> 400；未修改文件
    bogus = home / "bogus"
    bogus.mkdir()
    response = http.post(
        "/admin/models/adopt", json={"capability": "tts", "path": str(bogus)}
    )
    assert response.status_code == 400
    assert "签名" in response.json()["detail"]


def test_adopt_rejects_incomplete_signature(client) -> None:
    http, cfg, home = client
    incomplete = _files(home / "IndexTTS-broken", ("gpt.pth", "s2mel.pth"))
    response = http.post(
        "/admin/models/adopt", json={"capability": "tts", "path": str(incomplete)}
    )
    assert response.status_code == 400
    assert "不完整" in response.json()["detail"]
    assert cfg.model_dir != incomplete
    assert not config_file_path(home).exists()


def test_normalize_search_path_errors() -> None:
    with pytest.raises(DiscoveryError):
        from localai.admin.discovery import normalize_search_paths

        normalize_search_paths(["does-not-exist"])
