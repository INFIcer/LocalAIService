"""管理界面与 ``/admin/*`` 接口测试。

这些测试不加载任何 GPU 模型、不联网、也不真的重启服务：
- 设置写入使用临时 ``LOCALAI_HOME``，不会污染仓库里的 config.json；
- 下载接口只测参数校验，避免真的下载几 GB 权重；
- 重启接口把 ``schedule_restart`` 换成假函数，只验证必须带 confirm。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.config import Config, config_file_path

#: 原生引擎模块在 import 时会把路径常量固化下来。管理测试用临时 HOME 起应用，
#: 如果不清理，后面的测试会看到临时目录里的路径（真实服务只起一个应用，没这个问题）。
_NATIVE_MODULES = (
    "engine",
    "asr_engine",
    "whisper_engine",
    "translate_engine",
    "speaker_engine",
    "separator_engine",
    "ffmpeg_tool",
)


@pytest.fixture(autouse=True)
def _cleanup_native_modules():
    yield
    for name in _NATIVE_MODULES:
        sys.modules.pop(name, None)


def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉可能从开发机启动脚本带进来的设置环境变量。"""
    import os

    prefixes = ("LOCALAI_", "INDEX_TTS_")
    for name in list(os.environ):
        if name.startswith(prefixes):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_settings_env(monkeypatch)
    cfg = Config.from_env(tmp_path)
    cfg.token = ""
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, cfg, tmp_path


def _item(client: TestClient, key: str) -> dict:
    payload = client.get("/admin/settings").json()
    assert payload["ok"] is True
    return payload["settings"][key]


# ---------------------------------------------------------------------------
# 设置
# ---------------------------------------------------------------------------
def test_settings_shape_and_defaults(admin_client) -> None:
    client, cfg, _tmp = admin_client
    response = client.get("/admin/settings")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["config_file"].endswith("config.json")
    groups = {group["group"] for group in body["groups"]}
    assert {"service", "vram", "idle", "parallel", "tts", "asr", "translate", "media"} <= groups

    for key in ("host", "port", "token", "vram_margin_mb", "idle_unload", "asr_parallel", "tts_backend"):
        item = body["settings"][key]
        for field in ("value", "default", "source", "requires_restart", "description"):
            assert field in item
    assert body["settings"]["token"]["value"] == ""
    assert body["settings"]["token"]["secret"] is True
    assert body["settings"]["port"]["source"] == "default"
    # 默认值不应被写进 config.json
    assert not config_file_path(cfg.home).exists()


def test_update_settings_rejects_unknown_key(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    response = client.put("/admin/settings", json={"values": {"not_a_key": 1}})
    assert response.status_code == 400
    assert "不认识" in response.json()["detail"]


def test_update_settings_rejects_out_of_range(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    response = client.put("/admin/settings", json={"port": 70000})
    assert response.status_code == 400
    assert "65535" in response.json()["detail"]

    response = client.put("/admin/settings", json={"values": {"asr_parallel": 9}})
    assert response.status_code == 400
    assert "4" in response.json()["detail"]


def test_update_settings_writes_only_changed_keys(admin_client) -> None:
    client, cfg, tmp = admin_client
    path = config_file_path(tmp)
    path.write_text(
        json.dumps({"vram_margin_mb": 640, "custom_unknown": "keep-me"}, ensure_ascii=False),
        encoding="utf-8",
    )

    response = client.put("/admin/settings", json={"values": {"idle_unload": 120}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["applied"] == ["idle_unload"]
    assert cfg.idle_unload == 120

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["idle_unload"] == 120
    # 未改动的键与不认识的键都要保留
    assert saved["vram_margin_mb"] == 640
    assert saved["custom_unknown"] == "keep-me"

    # 改回默认值时从文件里删除该键（仍然只写改动项）
    response = client.put("/admin/settings", json={"values": {"idle_unload": 900}})
    assert response.status_code == 200
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "idle_unload" not in saved
    assert saved["vram_margin_mb"] == 640


def test_config_file_is_loaded_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    path = config_file_path(tmp_path)
    path.write_text(
        json.dumps({"whisper_model": "base", "idle_unload": 60}, ensure_ascii=False),
        encoding="utf-8",
    )
    cfg = Config.from_env(tmp_path)
    assert cfg.whisper_model == "base"
    assert cfg.idle_unload == 60

    app = create_app(cfg)
    with TestClient(app) as client:
        item = _item(client, "idle_unload")
        assert item["source"] == "file"
        assert item["value"] == 60


def test_env_override_is_reported(admin_client, monkeypatch: pytest.MonkeyPatch) -> None:
    client, cfg, _tmp = admin_client
    # 模拟「启动时被环境变量覆盖」：真实进程里由 from_env 记录。
    cfg.env_overrides["vram_margin_mb"] = True
    monkeypatch.setenv("LOCALAI_VRAM_MARGIN_MB", "768")

    item = _item(client, "vram_margin_mb")
    assert item["source"] == "env"

    response = client.put("/admin/settings", json={"values": {"vram_margin_mb": 640}})
    assert response.status_code == 200
    body = response.json()
    assert "vram_margin_mb" in body["overridden_by_env"]
    assert body["warnings"], "环境变量覆盖时必须给出明确警告"


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def test_models_list_fields(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    response = client.get("/admin/models")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert isinstance(body["models"], list) and body["models"]

    capabilities = {item["capability"] for item in body["models"]}
    assert {"tts", "asr", "whisper", "translate", "speaker", "separate", "vad"} <= capabilities
    for item in body["models"]:
        for field in (
            "capability",
            "name",
            "path",
            "present",
            "size_mb",
            "expected_files",
            "missing_files",
            "active",
            "options",
        ):
            assert field in item, f"模型项缺少字段 {field}"
        assert isinstance(item["present"], bool)
        assert isinstance(item["active"], bool)
        assert isinstance(item["expected_files"], list)
        assert isinstance(item["missing_files"], list)

    whisper_sizes = {item["name"] for item in body["models"] if item["capability"] == "whisper"}
    assert {"tiny", "base", "small", "medium", "large-v3"} <= whisper_sizes


def test_activate_whisper_writes_config(admin_client) -> None:
    client, cfg, tmp = admin_client
    response = client.post(
        "/admin/models/activate", json={"capability": "whisper", "name": "base"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["requires_restart"] is False
    assert body["note"]
    assert cfg.whisper_model == "base"

    saved = json.loads(config_file_path(tmp).read_text(encoding="utf-8"))
    assert saved["whisper_model"] == "base"
    item = _item(client, "whisper_model")
    assert item["source"] == "file"
    assert item["value"] == "base"


def test_activate_rejects_unsupported_capability(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    response = client.post(
        "/admin/models/activate", json={"capability": "tts", "name": "engine"}
    )
    assert response.status_code == 400
    assert "不支持" in response.json()["detail"]


def test_download_parameter_validation(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    bad_cap = client.post(
        "/admin/models/download", json={"capability": "nope", "name": ""}
    )
    assert bad_cap.status_code == 400

    bad_size = client.post(
        "/admin/models/download", json={"capability": "whisper", "name": "huge"}
    )
    assert bad_size.status_code == 400
    assert "tiny" in bad_size.json()["detail"]

    missing_task = client.get("/admin/models/tasks/not-exist")
    assert missing_task.status_code == 404


# ---------------------------------------------------------------------------
# 系统 / 日志 / 重启
# ---------------------------------------------------------------------------
def test_system_and_logs(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    system = client.get("/admin/system")
    assert system.status_code == 200
    body = system.json()
    for key in ("service", "listen", "gpu", "disk", "autostart", "paths"):
        assert key in body
    assert body["service"]["version"]
    assert "loopback" in body["listen"]

    logs = client.get("/admin/logs", params={"lines": 5, "file": "service"})
    assert logs.status_code == 200
    log_body = logs.json()
    assert log_body["ok"] is True
    assert log_body["file"] == "service"
    assert isinstance(log_body["lines"], list)

    bad = client.get("/admin/logs", params={"file": "../secrets"})
    assert bad.status_code == 400


def test_restart_requires_confirm(admin_client, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _cfg, _tmp = admin_client
    response = client.post("/admin/restart", json={})
    assert response.status_code == 400
    assert "confirm" in response.json()["detail"].lower()

    called: list[bool] = []

    def fake_schedule(app, cfg, **kwargs):
        called.append(True)
        return {"ok": True, "detail": "fake"}

    import localai.api.routers.admin as admin_router

    monkeypatch.setattr(admin_router, "schedule_restart", fake_schedule)
    response = client.post("/admin/restart", json={"confirm": True})
    assert response.status_code == 202
    assert response.json()["ok"] is True
    assert called == [True]


# ---------------------------------------------------------------------------
# 静态页面 / 鉴权
# ---------------------------------------------------------------------------
def test_ui_page_served(admin_client) -> None:
    client, _cfg, _tmp = admin_client
    for path in ("/ui", "/ui/"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "text/html" in response.headers.get("content-type", "")
        assert "LocalAIService 控制台" in response.text
        assert "概览" in response.text
        assert "模型" in response.text
        assert "/ui/app.js" in response.text

    assert client.get("/ui/styles.css").status_code == 200
    assert client.get("/ui/app.js").status_code == 200


def test_admin_requires_token_but_ui_is_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    cfg = Config.from_env(tmp_path)
    cfg.token = "secret-token"
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ui").status_code == 200

        denied = client.get("/admin/settings")
        assert denied.status_code == 401
        assert denied.json()["error"] == "unauthorized"

        allowed = client.get(
            "/admin/settings", headers={"Authorization": "Bearer secret-token"}
        )
        assert allowed.status_code == 200

        # 写接口同样受保护
        denied_put = client.put("/admin/settings", json={"values": {"idle_unload": 60}})
        assert denied_put.status_code == 401
