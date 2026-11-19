"""托盘动作层测试：不建图标、不开浏览器、不拉真实服务。

只测能在普通进程里稳定复现的部分：菜单映射、URL、探测判定、
退出顺序、端口占用时的降级，以及 /admin/shutdown 的令牌门。
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.cli import build_parser
from localai.config import Config
from localai.tray import actions as tray


class FakeProc:
    def __init__(self, *, returncode: Optional[int] = None, timeouts: int = 0) -> None:
        self.returncode = returncode
        self.timeouts = timeouts
        self.calls: list[str] = []
        self.pid = 4242

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self.calls.append(f"wait:{timeout}")
        if self.timeouts > 0:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("fake", timeout or 0)
        return self.returncode or 0

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")


# ---------------------------------------------------------------------------
# 纯契约
# ---------------------------------------------------------------------------
def test_ui_url_and_tooltip_contract() -> None:
    assert tray.ui_url(7880) == "http://127.0.0.1:7880/ui"
    assert tray.ui_url(7899) == "http://127.0.0.1:7899/ui"
    for mode in ("starting", tray.MODE_ATTACHED, tray.MODE_CONFLICT, "failed", "ready"):
        text = tray.build_tooltip(7880, mode)
        assert "LocalAIService" in text
        assert "7880" in text
        assert len(text) <= 127


def test_menu_items_and_action_mapping() -> None:
    assert [action for action, _label in tray.MENU_ITEMS] == [
        tray.ACTION_OPEN,
        tray.ACTION_COPY,
        tray.ACTION_QUIT,
    ]
    assert [label for _action, label in tray.MENU_ITEMS] == ["打开 WebUI", "复制地址", "退出"]
    for action, label in tray.MENU_ITEMS:
        assert len(label) <= 10
        assert tray.ACTION_BY_MENU_ID[tray.MENU_IDS[action]] == action


def test_health_shape_detection() -> None:
    good = {"ok": True, "backend": "engine", "jobs": {}, "vision": {}, "asr": {}}
    assert tray.is_localai_health(good) is True
    # ok=false 只表示模型没就绪；服务本身仍是本服务，托盘仍然挂靠。
    assert tray.is_localai_health({**good, "ok": False}) is True
    assert tray.is_localai_health({"ok": True, "hello": "world"}) is False
    assert tray.is_localai_health([1, 2, 3]) is False
    assert tray.is_localai_health(None) is False


def test_shutdown_request_headers() -> None:
    request = tray.build_shutdown_request(7899, "abc123", "secret")
    assert request.url == "http://127.0.0.1:7899/admin/shutdown"
    headers = dict(request.headers)
    assert headers[tray.SHUTDOWN_HEADER] == "abc123"
    assert headers["Authorization"] == "Bearer secret"

    plain = dict(tray.build_shutdown_request(7880, "abc123").headers)
    assert "Authorization" not in plain


def test_port_free_uses_bind_not_connect() -> None:
    # 本机防火墙会让未监听端口的 connect 挂到超时，必须用 bind 判断空闲。
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        assert tray.port_free(port) is False
    finally:
        sock.close()
    assert tray.port_free(port) is True


def test_probe_treats_foreign_json_as_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tray, "port_free", lambda port, timeout=0.4: False)
    monkeypatch.setattr(tray, "http_get_json", lambda url, timeout=2.5: {"hello": "world"})
    assert tray.probe_service(7880) == tray.STATE_CONFLICT

    monkeypatch.setattr(
        tray,
        "http_get_json",
        lambda url, timeout=2.5: {"ok": True, "backend": "engine", "jobs": {}, "vision": {}, "asr": {}},
    )
    assert tray.probe_service(7880) == tray.STATE_RUNNING

    monkeypatch.setattr(tray, "port_free", lambda port, timeout=0.4: True)
    assert tray.probe_service(7880) == tray.STATE_FREE


# ---------------------------------------------------------------------------
# 退出顺序
# ---------------------------------------------------------------------------
def test_quit_requests_graceful_shutdown_first() -> None:
    proc = FakeProc()
    result = tray.stop_owned_process(proc, lambda: True)
    assert result.steps == ["shutdown-requested", "graceful-exit"]
    assert proc.calls == ["wait:20.0"]


def test_quit_falls_back_to_terminate_when_shutdown_refused() -> None:
    proc = FakeProc()
    result = tray.stop_owned_process(proc, lambda: False)
    assert result.steps == ["terminate", "terminated"]
    assert proc.calls[0] == "terminate"


def test_quit_kills_after_terminate_timeout() -> None:
    proc = FakeProc(timeouts=2)
    result = tray.stop_owned_process(proc, lambda: True)
    assert result.steps == [
        "shutdown-requested",
        "graceful-timeout",
        "terminate",
        "terminate-timeout",
        "kill",
        "killed",
    ]
    assert proc.calls[:2] == ["wait:20.0", "terminate"]


def test_quit_skips_already_exited_process() -> None:
    proc = FakeProc(returncode=0)
    result = tray.stop_owned_process(proc, lambda: True)
    assert result.steps == ["already-exited"]
    assert proc.calls == []


def test_quit_treats_shutdown_exception_as_refused() -> None:
    proc = FakeProc()

    def boom() -> bool:
        raise RuntimeError("connection refused")

    result = tray.stop_owned_process(proc, boom)
    assert result.steps == ["terminate", "terminated"]


# ---------------------------------------------------------------------------
# 控制器：拉起 / 挂靠 / 冲突
# ---------------------------------------------------------------------------
def _make_controller(probe_state: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    spawned: list[FakeProc] = []

    def fake_spawn(token: str) -> tray.SpawnedService:
        proc = FakeProc()
        spawned.append(proc)
        return tray.SpawnedService(proc=proc, command=["python"], token=token)

    controller = tray.TrayController(
        port=7899,
        home=tmp_path,
        probe=lambda port: probe_state,
        spawn=fake_spawn,
        shutdown_client=lambda: True,
    )
    return controller, spawned


def test_controller_starts_when_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, spawned = _make_controller(tray.STATE_FREE, monkeypatch, tmp_path)
    assert controller.start() == tray.MODE_OWNED
    assert len(spawned) == 1
    assert controller.proc is spawned[0]
    assert controller.shutdown_token


def test_controller_attaches_to_running_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, spawned = _make_controller(tray.STATE_RUNNING, monkeypatch, tmp_path)
    assert controller.start() == tray.MODE_ATTACHED
    assert spawned == []
    assert controller.proc is None
    assert controller.quit().steps == ["external-service"]


def test_controller_refuses_to_start_on_foreign_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, spawned = _make_controller(tray.STATE_CONFLICT, monkeypatch, tmp_path)
    assert controller.start() == tray.MODE_CONFLICT
    assert spawned == []
    assert controller.quit().steps == ["external-service"]


def test_controller_quit_uses_shutdown_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, spawned = _make_controller(tray.STATE_FREE, monkeypatch, tmp_path)
    controller.start()
    requested: list[bool] = []

    def request() -> bool:
        requested.append(True)
        return True

    controller._shutdown_override = request
    result = controller.quit()
    assert requested == [True]
    assert result.steps == ["shutdown-requested", "graceful-exit"]


def test_service_process_kills_whole_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)

    monkeypatch.setattr(tray.subprocess, "run", fake_run)
    proc = FakeProc()
    wrapped = tray.ServiceProcess(proc)  # type: ignore[arg-type]
    wrapped.terminate()
    wrapped.kill()
    assert commands == [
        ["taskkill", "/PID", "4242", "/T", "/F"],
        ["taskkill", "/PID", "4242", "/T", "/F"],
    ]
    assert proc.calls == []


def test_service_process_falls_back_when_taskkill_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        raise FileNotFoundError("taskkill")

    monkeypatch.setattr(tray.subprocess, "run", fake_run)
    proc = FakeProc()
    tray.ServiceProcess(proc).terminate()  # type: ignore[arg-type]
    assert proc.calls == ["terminate"]


def test_spawn_builds_expected_command_and_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(tray.subprocess, "Popen", fake_popen)
    controller = tray.TrayController(port=7899, home=tmp_path)
    spawned = controller._spawn("tok-123")

    assert captured["command"] == [
        controller.python_executable,
        "-m",
        "localai",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "7899",
    ]
    assert captured["env"]["LOCALAI_SHUTDOWN_TOKEN"] == "tok-123"
    assert captured["env"]["LOCALAI_PORT"] == "7899"
    assert captured["env"]["LOCALAI_HOST"] == "127.0.0.1"
    assert captured["cwd"] == str(tmp_path)
    assert spawned.token == "tok-123"
    assert (tmp_path / "logs" / "service.out.log").is_file()


# ---------------------------------------------------------------------------
# 动作层
# ---------------------------------------------------------------------------
class FakeController:
    port = 7880
    mode = tray.MODE_OWNED

    def __init__(self) -> None:
        self.quit_calls = 0

    def quit(self) -> tray.StopResult:
        self.quit_calls += 1
        return tray.StopResult(owned=True, steps=["test"])


def test_actions_open_copy_quit() -> None:
    opened: list[str] = []
    copied: list[str] = []
    controller = FakeController()
    actions = tray.TrayActions(
        controller,
        open_url=opened.append,
        copy_text=copied.append,
    )

    assert actions.perform(tray.ACTION_OPEN) == tray.ACTION_OPEN
    assert actions.perform(tray.ACTION_COPY) == tray.ACTION_COPY
    assert actions.perform(tray.ACTION_QUIT) == tray.ACTION_QUIT
    assert opened == ["http://127.0.0.1:7880/ui"]
    assert copied == ["http://127.0.0.1:7880/ui"]
    assert controller.quit_calls == 1

    with pytest.raises(ValueError):
        actions.perform("unknown")


def test_tray_cli_parser_wires_port() -> None:
    args = build_parser().parse_args(["tray", "--port", "7899"])
    assert args.port == 7899
    assert args.func.__name__ == "cmd_tray"


# ---------------------------------------------------------------------------
# /admin/shutdown 的令牌门
# ---------------------------------------------------------------------------
class FakeServer:
    def __init__(self) -> None:
        self.should_exit = False


def _client_with_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, token: str, cfg_token: str = ""):
    if token:
        monkeypatch.setenv("LOCALAI_SHUTDOWN_TOKEN", token)
    else:
        monkeypatch.delenv("LOCALAI_SHUTDOWN_TOKEN", raising=False)
    cfg = Config.from_env(tmp_path)
    cfg.token = cfg_token
    app = create_app(cfg)
    server = FakeServer()
    app.state.uvicorn_server = server
    return app, server


def test_shutdown_endpoint_requires_injected_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, server = _client_with_token(monkeypatch, tmp_path, token="tray-secret")
    with TestClient(app) as client:
        denied = client.post("/admin/shutdown")
        assert denied.status_code == 403
        assert server.should_exit is False

        wrong = client.post("/admin/shutdown", headers={tray.SHUTDOWN_HEADER: "nope"})
        assert wrong.status_code == 403

        ok = client.post("/admin/shutdown", headers={tray.SHUTDOWN_HEADER: "tray-secret"})
        assert ok.status_code == 200
        assert ok.json()["ok"] is True
        assert server.should_exit is True
        assert app.state.shutdown_requested is True


def test_shutdown_endpoint_rejects_manually_started_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, server = _client_with_token(monkeypatch, tmp_path, token="")
    with TestClient(app) as client:
        response = client.post("/admin/shutdown", headers={tray.SHUTDOWN_HEADER: "anything"})
    assert response.status_code == 403
    assert "托盘" in response.json()["detail"]
    assert server.should_exit is False


def test_shutdown_endpoint_respects_global_bearer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app, server = _client_with_token(monkeypatch, tmp_path, token="tray-secret", cfg_token="api-token")
    with TestClient(app) as client:
        no_auth = client.post(
            "/admin/shutdown", headers={tray.SHUTDOWN_HEADER: "tray-secret"}
        )
        assert no_auth.status_code == 401

        ok = client.post(
            "/admin/shutdown",
            headers={
                tray.SHUTDOWN_HEADER: "tray-secret",
                "Authorization": "Bearer api-token",
            },
        )
        assert ok.status_code == 200
        assert server.should_exit is True


def test_shutdown_route_is_registered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app, _server = _client_with_token(monkeypatch, tmp_path, token="tray-secret")
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert "/admin/shutdown" in paths


def test_tray_icon_asset_exists() -> None:
    icon = Path(tray.__file__).with_name("tray.ico")
    assert icon.is_file()
    header = icon.read_bytes()[:4]
    assert header[:2] == b"\x00\x00"
    assert header[2:4] == b"\x01\x00"
