"""托盘动作层测试：不建图标、不开浏览器、不拉真实服务。

只测能在普通进程里稳定复现的部分：菜单映射、URL、探测判定、
退出顺序、端口占用时的降级，以及 /admin/shutdown 的令牌门。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from localai.api.app import create_app
from localai.cli import build_parser
from localai.config import Config
from localai.core import runtime as token_file
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
        tray.ACTION_CLEAR_VRAM,
        tray.ACTION_QUIT,
    ]
    assert [label for _action, label in tray.MENU_ITEMS] == [
        "打开 WebUI",
        "复制地址",
        "清理显存",
        "退出",
    ]
    # 插入新项后 id 自动重排；退出仍是最后一项，菜单分隔线插在它前面。
    assert tray.MENU_IDS[tray.ACTION_CLEAR_VRAM] == 1003
    assert tray.MENU_IDS[tray.ACTION_QUIT] == 1004
    assert tray.ACTION_BY_MENU_ID[1003] == tray.ACTION_CLEAR_VRAM
    assert tray.ACTION_BY_MENU_ID[1004] == tray.ACTION_QUIT
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
def _make_controller(
    probe_state: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    **overrides,
):
    """控制器工厂：所有外部副作用（探测、等待、探测进程、杀进程）都可注入。"""
    spawned: list[FakeProc] = []

    def fake_spawn(token: str) -> tray.SpawnedService:
        proc = FakeProc()
        spawned.append(proc)
        return tray.SpawnedService(proc=proc, command=["python"], token=token)

    probe = overrides.pop("probe", lambda port: probe_state)
    controller = tray.TrayController(
        port=7899,
        home=tmp_path,
        probe=probe,
        # 探测与重启空窗的重试都用注入值，测试不碰真实网络。
        health_wait=overrides.pop("health_wait", lambda port: probe_state),
        spawn=fake_spawn,
        shutdown_client=overrides.pop("shutdown_client", lambda: True),
        find_listener=overrides.pop("find_listener", lambda port: None),
        kill_tree=overrides.pop("kill_tree", lambda pid: True),
        wait_stop=overrides.pop("wait_stop", lambda port: True),
        health=overrides.pop("health", lambda port: HEALTHY),
        token_reader=overrides.pop("token_reader", lambda home, port: "file-token"),
        **overrides,
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
    # 挂靠模式下退出同样要关掉服务：先走优雅请求。
    assert controller.quit().steps == ["shutdown-requested", "graceful-exit"]


def test_controller_refuses_to_start_on_foreign_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    controller, spawned = _make_controller(tray.STATE_CONFLICT, monkeypatch, tmp_path)
    assert controller.start() == tray.MODE_CONFLICT
    assert spawned == []
    assert controller.quit().steps == ["foreign-port"]


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


def test_shutdown_endpoint_accepts_generated_token_for_manually_started_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """手工启动的服务也生成令牌并写文件：挂靠模式的托盘靠它走优雅关停。"""
    app, server = _client_with_token(monkeypatch, tmp_path, token="")
    path = token_file.shutdown_token_path(tmp_path, app.state.cfg.port)
    with TestClient(app) as client:
        assert path.is_file(), "服务运行时必须把令牌写到运行目录"
        generated = path.read_text(encoding="utf-8").strip()
        assert generated == app.state.shutdown_token

        denied = client.post(
            "/admin/shutdown", headers={tray.SHUTDOWN_HEADER: "anything"}
        )
        assert denied.status_code == 403
        assert server.should_exit is False

        ok = client.post("/admin/shutdown", headers={tray.SHUTDOWN_HEADER: generated})
        assert ok.status_code == 200
        assert server.should_exit is True

    assert not path.exists(), "服务正常退出后要删掉令牌文件"


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


# ---------------------------------------------------------------------------
# 失败可见性、重复实例、首次启动提示、重启空窗
# ---------------------------------------------------------------------------
HEALTHY = {"ok": True, "backend": "engine", "jobs": {}, "vision": {}, "asr": {}}


def _tray_app():
    if os.name != "nt":
        pytest.skip("托盘入口只在 Windows 上可用")
    from localai.tray import app

    return app


def _winicon():
    if os.name != "nt":
        pytest.skip("通知区图标只在 Windows 上可用")
    from localai.tray import winicon

    return winicon


def _reset_tray_log(app) -> None:
    """_configure_logging 只在没有 handler 时挂新的；测试间要清掉上一个临时目录的句柄。"""
    for handler in list(app.log.handlers):
        app.log.removeHandler(handler)
        handler.close()


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _FakeConfig:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.port = 7880
        self.host = "127.0.0.1"
        self.token = ""


# -- A：任何启动失败都要写日志 + 弹框 ----------------------------------------
def test_fatal_error_logs_traceback_and_popup_is_one_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _tray_app()
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "_show_dialog", lambda title, text: dialogs.append((title, text)))

    try:
        raise RuntimeError("端口 99999 不可用")
    except RuntimeError as exc:
        app.report_fatal(tmp_path, "LocalAIService 托盘启动失败", exc)

    path = app.error_log_path(tmp_path)
    assert path.name == "tray-error.log"
    written = path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last)" in written
    assert "RuntimeError: 端口 99999 不可用" in written

    assert dialogs, "失败必须弹框，pythonw 下没有别的可见通道"
    title, message = dialogs[0]
    assert "失败" in title
    assert "RuntimeError: 端口 99999 不可用" in message
    assert str(path) in message
    # 弹框只给一句话，不贴整段 traceback。
    assert "Traceback" not in message


def test_main_reports_failure_before_run_can_handle_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _tray_app()
    _reset_tray_log(app)
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "_show_dialog", lambda title, text: dialogs.append((title, text)))
    monkeypatch.setattr(app, "_parse_args", lambda argv: argparse.Namespace(port=None, host=None))
    monkeypatch.setattr("localai.config.get_config", lambda: _FakeConfig(tmp_path))

    def boom(args):
        raise OSError("配置读不出来")

    monkeypatch.setattr(app, "_run", boom)

    assert app.main([]) == 1
    assert "OSError: 配置读不出来" in app.error_log_path(tmp_path).read_text(encoding="utf-8")
    assert dialogs and "配置读不出来" in dialogs[0][1]


def test_icon_add_failure_is_raised_not_silent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    winicon = _winicon()
    window = winicon.TrayWindow(
        object(), _FakeModeController(winicon.MODE_OWNED), icon_path=tmp_path / "missing.ico"
    )

    def no_icon() -> bool:
        return False

    monkeypatch.setattr(window, "_create_window", lambda: setattr(window, "_hwnd", 4321))
    monkeypatch.setattr(window, "_load_icon", lambda: None)
    monkeypatch.setattr(window, "_add_icon", no_icon)
    with pytest.raises(RuntimeError):
        window.run()


# -- B：第二次双击必须看得见 ------------------------------------------------
def test_second_launch_shows_hint_instead_of_exiting_silently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _tray_app()
    _reset_tray_log(app)
    dialogs: list[tuple[str, str]] = []
    woken: list[bool] = []
    monkeypatch.setattr(app, "_show_dialog", lambda title, text: dialogs.append((title, text)))
    monkeypatch.setattr(
        app, "_wake_existing_tray", lambda timeout=3.0: woken.append(True) or False
    )
    monkeypatch.setattr(app, "_acquire_instance_mutex", lambda: None)
    monkeypatch.setattr("localai.config.get_config", lambda: _FakeConfig(tmp_path))
    try:
        code = app._run(argparse.Namespace(port=None, host=None))
    finally:
        _reset_tray_log(app)

    assert code == app.EXIT_ALREADY_RUNNING != 0
    assert woken == [True], "第二次双击要顺手叫醒已有托盘打开页面"
    assert dialogs and "托盘已经在运行" in dialogs[0][1]
    assert "打开 WebUI" in dialogs[0][1]
    # 主人要求弹框只报「已在运行」，不要再教人去通知区找图标。
    assert "通知区" not in dialogs[0][1]
    assert "^" not in dialogs[0][1]
    log_text = (tmp_path / "logs" / "tray.log").read_text(encoding="utf-8")
    assert "托盘已在运行" in log_text


# -- C：图标注册成功后弹一次气球 --------------------------------------------
class _FakeModeController:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.port = 7880

    def health_state(self) -> str:
        return tray.STATE_RUNNING

    def process_exited(self) -> bool:
        return False


class _FakeShell32:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object]] = []

    def Shell_NotifyIconW(self, operation, data) -> bool:
        self.calls.append((operation, data._obj))
        return True


def test_startup_balloon_is_shown_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    winicon = _winicon()
    shell = _FakeShell32()
    monkeypatch.setattr(winicon, "_shell32", shell)
    window = winicon.TrayWindow(
        object(), _FakeModeController(winicon.MODE_OWNED), icon_path=tmp_path / "tray.ico"
    )
    window._hwnd = 4321

    window._announce_initial()
    window._announce_initial()
    _keep = window  # 防止 WNDPROC 回调被提前回收

    infos = [data for _op, data in shell.calls if data.uFlags & winicon.NIF_INFO]
    assert len(infos) == 1, "气球只在启动时弹一次"
    assert infos[0].szInfoTitle == "本地 AI 服务已启动"
    assert infos[0].szInfo == "127.0.0.1:7880"
    assert "通知区" not in infos[0].szInfo
    assert "^" not in infos[0].szInfo
    assert infos[0].dwInfoFlags == winicon.NIIF_INFO
    assert _keep._ballooned == {"started"}


# -- D：重启空窗里不拉起第二份 ----------------------------------------------
def test_restart_wait_attaches_as_soon_as_health_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_get(url: str, timeout: float = 2.5):
        calls.append(url)
        return HEALTHY if len(calls) >= 3 else None

    monkeypatch.setattr(tray, "http_get_json", fake_get)
    state = tray.wait_for_restarting_service(7880, interval=0.0, sleep=lambda seconds: None)
    assert state == tray.STATE_RUNNING
    assert len(calls) == 3
    assert calls[0].endswith("/health")


def test_restart_wait_gives_up_after_its_window(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds

    monkeypatch.setattr(tray, "http_get_json", lambda url, timeout=2.5: None)
    state = tray.wait_for_restarting_service(
        7880,
        wait_seconds=0.5,
        interval=0.25,
        monotonic=lambda: now[0],
        sleep=fake_sleep,
    )
    assert state == tray.STATE_FREE


def test_controller_attaches_when_health_appears_after_port_freed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 真实 probe：端口 bind 成功（服务正在重启，旧进程已退），随后 /health 回来。
    port = _free_port()
    monkeypatch.setattr(tray, "http_get_json", lambda url, timeout=2.5: HEALTHY)
    spawned: list[tray.SpawnedService] = []

    controller = tray.TrayController(
        port=port,
        home=tmp_path,
        spawn=lambda token: spawned.append(
            tray.SpawnedService(proc=FakeProc(), command=["python"], token=token)
        ),
        shutdown_client=lambda: True,
    )
    try:
        assert controller.start() == tray.MODE_ATTACHED
    finally:
        for handler in list(tray.log.handlers):
            tray.log.removeHandler(handler)
            handler.close()
    assert spawned == [], "重启空窗里不能拉起第二份服务"


def test_controller_spawns_after_restart_window_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[FakeProc] = []

    def fake_spawn(token: str) -> tray.SpawnedService:
        proc = FakeProc()
        spawned.append(proc)
        return tray.SpawnedService(proc=proc, command=["python"], token=token)

    controller = tray.TrayController(
        port=7899,
        home=tmp_path,
        probe=lambda port: tray.STATE_FREE,
        health_wait=lambda port: tray.STATE_FREE,
        spawn=fake_spawn,
        shutdown_client=lambda: True,
    )
    assert controller.start() == tray.MODE_OWNED
    assert len(spawned) == 1


# ---------------------------------------------------------------------------
# 清理显存：请求契约与结果文案
# ---------------------------------------------------------------------------
class _VramResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, _size: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


class _VramOpener:
    """假 opener：抓住托盘真正发出的 Request，不碰网络。"""

    def __init__(self, payload: object = None, *, status: int = 200, error=None) -> None:
        self.requests: list[tuple[object, float]] = []
        self._payload = payload
        self._status = status
        self._error = error

    def open(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self._error is not None:
            raise self._error
        return _VramResponse(self._payload, self._status)


def test_clear_vram_request_body_and_bearer_header() -> None:
    opener = _VramOpener({"ok": True, "released": ["speaker"], "skipped": []})
    result = tray.VramUnloadClient(7899, "secret", opener=opener).request()

    assert result == {"ok": True, "released": ["speaker"], "skipped": []}
    assert len(opener.requests) == 1
    request, timeout = opener.requests[0]
    assert request.full_url == "http://127.0.0.1:7899/vram/unload"
    assert request.get_method() == "POST"
    assert json.loads(request.data.decode("utf-8")) == {"all": True, "skip_busy": True}
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == tray.VRAM_UNLOAD_TIMEOUT == 5.0


def test_clear_vram_request_without_token_has_no_authorization() -> None:
    opener = _VramOpener({"ok": True, "released": [], "skipped": []})
    tray.VramUnloadClient(7880, opener=opener).request()

    request, _timeout = opener.requests[0]
    assert request.get_header("Authorization") is None


def test_clear_vram_client_returns_none_on_http_error() -> None:
    opener = _VramOpener({"ok": False}, status=500)
    assert tray.VramUnloadClient(7899, opener=opener).request() is None


def test_clear_vram_client_returns_none_on_exception() -> None:
    opener = _VramOpener(error=OSError("connection refused"))
    assert tray.VramUnloadClient(7899, opener=opener).request() is None


def _actions_with_unload(unload, notices: list[tuple[str, str, int]]) -> tray.TrayActions:
    actions = tray.TrayActions(
        FakeController(),
        open_url=lambda _url: None,
        copy_text=lambda _text: None,
        unload_client=unload,
    )
    # app.py 也是窗口建好后再接回调；这里覆盖 set_notify 这条路径。
    actions.set_notify(lambda title, text, flag: notices.append((title, text, flag)))
    return actions


def test_clear_vram_notice_uses_chinese_model_names() -> None:
    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(
        lambda: {"ok": True, "released": ["tts", "vision"], "skipped": []}, notices
    )

    assert actions.perform(tray.ACTION_CLEAR_VRAM) == tray.ACTION_CLEAR_VRAM
    assert notices == [("显存已清理", "已卸下 语音合成、识图", tray.NOTIFY_INFO)]


def test_clear_vram_notice_mentions_busy_models_and_dedupes_names() -> None:
    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(
        lambda: {
            "ok": True,
            "released": ["asr", "whisper"],
            "skipped": ["separate"],
        },
        notices,
    )

    actions.perform(tray.ACTION_CLEAR_VRAM)
    assert notices == [
        ("显存已清理", "已卸下 语音识别；正在用的 分轨 没动", tray.NOTIFY_INFO)
    ]


def test_clear_vram_notice_when_everything_is_busy() -> None:
    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(
        lambda: {"ok": True, "released": [], "skipped": ["tts"]}, notices
    )

    actions.perform(tray.ACTION_CLEAR_VRAM)
    assert notices == [
        (
            "暂时没腾出显存",
            "正在用的 语音合成 没动；等它跑完再试一次。",
            tray.NOTIFY_WARNING,
        )
    ]


def test_clear_vram_notice_when_nothing_is_loaded() -> None:
    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(
        lambda: {"ok": True, "released": [], "skipped": []}, notices
    )

    actions.perform(tray.ACTION_CLEAR_VRAM)
    assert notices == [("显存已清理", "现在没有模型占着显存。", tray.NOTIFY_INFO)]


def test_clear_vram_notice_falls_back_to_raw_model_name() -> None:
    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(
        lambda: {"ok": True, "released": ["mystery"], "skipped": []}, notices
    )

    actions.perform(tray.ACTION_CLEAR_VRAM)
    assert notices == [("显存已清理", "已卸下 mystery", tray.NOTIFY_INFO)]


def test_clear_vram_notice_reports_failure_on_none() -> None:
    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(lambda: None, notices)

    assert actions.perform(tray.ACTION_CLEAR_VRAM) == tray.ACTION_CLEAR_VRAM
    assert notices == [
        ("清理失败", "服务没响应，稍后再试。", tray.NOTIFY_ERROR)
    ]


def test_clear_vram_notice_reports_failure_on_exception() -> None:
    def boom():
        raise RuntimeError("timed out")

    notices: list[tuple[str, str, int]] = []
    actions = _actions_with_unload(boom, notices)

    assert actions.perform(tray.ACTION_CLEAR_VRAM) == tray.ACTION_CLEAR_VRAM
    assert notices == [
        ("清理失败", "服务没响应，稍后再试。", tray.NOTIFY_ERROR)
    ]


def test_clear_vram_balloon_is_not_deduped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    winicon = _winicon()
    shell = _FakeShell32()
    monkeypatch.setattr(winicon, "_shell32", shell)
    window = winicon.TrayWindow(
        object(), _FakeModeController(winicon.MODE_OWNED), icon_path=tmp_path / "tray.ico"
    )
    window._hwnd = 4321

    window._balloon(
        "clear-vram", "显存已清理", "已卸下 声纹", winicon.NIIF_INFO, once=False
    )
    window._balloon(
        "clear-vram", "显存已清理", "已卸下 声纹", winicon.NIIF_INFO, once=False
    )

    infos = [data for _op, data in shell.calls if data.uFlags & winicon.NIF_INFO]
    assert len(infos) == 2, "每次清理都要弹，不能被 _ballooned 去重"
    assert "clear-vram" not in window._ballooned


def test_notify_result_queues_then_shows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    winicon = _winicon()
    window = winicon.TrayWindow(
        object(), _FakeModeController(winicon.MODE_OWNED), icon_path=tmp_path / "tray.ico"
    )
    posted: list[int] = []
    balloons: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(window, "_post", lambda message: posted.append(message))
    monkeypatch.setattr(
        window, "_balloon", lambda *args, **kwargs: balloons.append((args, kwargs))
    )

    window.notify_result("显存已清理", "已卸下 声纹", winicon.NIIF_INFO)
    assert posted == [winicon.WM_TRAY_NOTIFY]

    window._show_notice()
    assert balloons == [
        (
            ("clear-vram", "显存已清理", "已卸下 声纹", winicon.NIIF_INFO),
            {"once": False},
        )
    ]


# ---------------------------------------------------------------------------
# 生命周期归属：退出托盘 = 服务退出
# ---------------------------------------------------------------------------
def test_quit_uses_persisted_token_for_attached_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """挂靠模式没有注入的令牌，读服务写下的令牌文件走优雅关停。"""
    calls: list[tuple[int, str, str]] = []

    class FakeClient:
        def __init__(self, port: int, token: str, bearer_token: str = "") -> None:
            calls.append((int(port), token, bearer_token))

        def request(self) -> bool:
            return True

    monkeypatch.setattr(tray, "ShutdownClient", FakeClient)
    controller, _spawned = _make_controller(
        tray.STATE_RUNNING,
        monkeypatch,
        tmp_path,
        shutdown_client=None,
        token_reader=lambda home, port: "file-token",
    )
    assert controller.start() == tray.MODE_ATTACHED

    result = controller.quit()

    assert calls == [(7899, "file-token", "")]
    assert result.steps == ["shutdown-requested", "graceful-exit"]
    assert result.owned is False


def test_quit_kills_attached_service_when_no_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    killed: list[int] = []
    controller, _spawned = _make_controller(
        tray.STATE_RUNNING,
        monkeypatch,
        tmp_path,
        token_reader=lambda home, port: "",
        find_listener=lambda port: 4321,
        kill_tree=lambda pid: killed.append(pid) or True,
    )
    assert controller.start() == tray.MODE_ATTACHED

    result = controller.quit()

    assert result.steps == ["no-token", "kill-tree", "tree-killed"]
    assert killed == [4321]


def test_quit_falls_back_to_kill_after_graceful_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    killed: list[int] = []
    controller, _spawned = _make_controller(
        tray.STATE_RUNNING,
        monkeypatch,
        tmp_path,
        token_reader=lambda home, port: "tok",
        wait_stop=lambda port: False,
        find_listener=lambda port: 4321,
        kill_tree=lambda pid: killed.append(pid) or True,
        health=lambda port: {**HEALTHY, "pid": 4321},
    )
    assert controller.start() == tray.MODE_ATTACHED

    result = controller.quit()

    # 顺序不可颠倒：先请求优雅退出，等满超时才强杀。
    assert result.steps == [
        "shutdown-requested",
        "graceful-timeout",
        "kill-tree",
        "tree-killed",
    ]
    assert killed == [4321]


def test_quit_leaves_foreign_port_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched: list[str] = []
    controller, _spawned = _make_controller(
        tray.STATE_CONFLICT,
        monkeypatch,
        tmp_path,
        find_listener=lambda port: touched.append("find") or 4321,
        kill_tree=lambda pid: touched.append("kill") or True,
    )
    assert controller.start() == tray.MODE_CONFLICT

    result = controller.quit()

    assert result.steps == ["foreign-port"]
    assert touched == [], "端口属于别的程序时既不能定位也不该强杀"


def test_quit_leaves_service_alone_when_health_shape_is_foreign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    killed: list[int] = []
    controller, _spawned = _make_controller(
        tray.STATE_RUNNING,
        monkeypatch,
        tmp_path,
        token_reader=lambda home, port: "",
        find_listener=lambda port: 4321,
        kill_tree=lambda pid: killed.append(pid) or True,
        health=lambda port: {"hello": "world"},
    )
    assert controller.start() == tray.MODE_ATTACHED

    result = controller.quit()

    assert result.steps == ["no-token", "foreign-port"]
    assert killed == []


def test_quit_aborts_when_health_pid_differs_from_listener(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    killed: list[int] = []
    controller, _spawned = _make_controller(
        tray.STATE_RUNNING,
        monkeypatch,
        tmp_path,
        token_reader=lambda home, port: "",
        find_listener=lambda port: 4321,
        kill_tree=lambda pid: killed.append(pid) or True,
        health=lambda port: {**HEALTHY, "pid": 999},
    )
    assert controller.start() == tray.MODE_ATTACHED

    result = controller.quit()

    assert result.steps == ["no-token", "foreign-port"]
    assert killed == [], "监听进程与应答 /health 的进程不是同一个时不能动手"


def test_quit_reports_no_listener_when_pid_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, _spawned = _make_controller(
        tray.STATE_RUNNING,
        monkeypatch,
        tmp_path,
        token_reader=lambda home, port: "",
        find_listener=lambda port: None,
    )
    assert controller.start() == tray.MODE_ATTACHED

    result = controller.quit()

    assert result.steps == ["no-token", "no-listener"]


def test_quit_stops_service_replaced_by_webui_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """托盘拉起的旧进程被重启脚本结束后，退出时按端口关掉新进程。"""
    killed: list[int] = []
    states = [tray.STATE_FREE]

    def probe(port: int) -> str:
        return states.pop(0) if len(states) > 1 else states[0]

    controller, spawned = _make_controller(
        tray.STATE_FREE,
        monkeypatch,
        tmp_path,
        probe=probe,
        # 重启脚本用 WMI 拉起新进程，旧令牌传不过去：新服务不认它。
        shutdown_client=lambda: False,
        find_listener=lambda port: 7777,
        kill_tree=lambda pid: killed.append(pid) or True,
        health=lambda port: {**HEALTHY, "pid": 7777},
    )
    assert controller.start() == tray.MODE_OWNED
    spawned[0].returncode = 0  # 旧服务已经退出
    states[:] = [tray.STATE_RUNNING]  # 新服务接管了同一个端口

    result = controller.quit()

    assert result.steps == ["shutdown-refused", "kill-tree", "tree-killed"]
    assert killed == [7777]


def test_quit_stops_own_process_before_giving_up_on_foreign_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller, spawned = _make_controller(tray.STATE_FREE, monkeypatch, tmp_path)
    assert controller.start() == tray.MODE_OWNED
    controller._probe = lambda port: tray.STATE_CONFLICT

    result = controller.quit()

    # 端口虽然被别的程序占了，自己拉起的进程仍必须收干净。
    assert result.steps == ["shutdown-requested", "graceful-exit", "foreign-port"]
    assert spawned[0].calls == ["wait:20.0"]


def test_quit_with_nothing_to_stop_reports_no_service(tmp_path: Path) -> None:
    controller = tray.TrayController(
        port=7899,
        home=tmp_path,
        probe=lambda port: tray.STATE_FREE,
        find_listener=lambda port: None,
        kill_tree=lambda pid: False,
        health=lambda port: None,
        token_reader=lambda home, port: "",
    )
    result = controller.quit()
    assert result.steps == ["no-service"]
    assert result.owned is False


def test_wait_for_service_stop_returns_true_when_service_stops() -> None:
    states = [tray.STATE_RUNNING, tray.STATE_RUNNING, tray.STATE_FREE]
    calls: list[int] = []

    def probe(port: int) -> str:
        calls.append(port)
        return states.pop(0)

    assert tray.wait_for_service_stop(7880, probe=probe, sleep=lambda seconds: None) is True
    assert calls == [7880, 7880, 7880]


def test_wait_for_service_stop_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(tray, "probe_service", lambda port: tray.STATE_RUNNING)

    def fake_sleep(seconds: float) -> None:
        now[0] += seconds

    assert (
        tray.wait_for_service_stop(
            7880,
            wait_seconds=0.5,
            interval=0.25,
            monotonic=lambda: now[0],
            sleep=fake_sleep,
        )
        is False
    )
    assert now[0] >= 0.5


# ---------------------------------------------------------------------------
# 令牌文件：服务写、托盘读，按端口区分
# ---------------------------------------------------------------------------
def test_shutdown_token_file_roundtrip_and_port_scope(tmp_path: Path) -> None:
    path = token_file.shutdown_token_path(tmp_path, 7899)
    assert path == tmp_path / "run" / "shutdown-7899.token"
    assert token_file.read_shutdown_token(tmp_path, 7899) == ""

    assert token_file.write_shutdown_token(tmp_path, 7899, "abc") == path
    assert token_file.read_shutdown_token(tmp_path, 7899) == "abc"
    token_file.write_shutdown_token(tmp_path, 7880, "other")
    assert token_file.read_shutdown_token(tmp_path, 7899) == "abc"
    assert token_file.read_shutdown_token(tmp_path, 7880) == "other"

    # 只删自己写的那份，别把后来者（重启后的新进程）写的删掉。
    token_file.remove_shutdown_token(tmp_path, 7899, "stale")
    assert path.is_file()
    token_file.remove_shutdown_token(tmp_path, 7899, "abc")
    assert not path.exists()


def test_write_shutdown_token_ignores_empty_token(tmp_path: Path) -> None:
    assert token_file.write_shutdown_token(tmp_path, 7899, "  ") is None
    assert not token_file.shutdown_token_path(tmp_path, 7899).exists()


# ---------------------------------------------------------------------------
# Windows 进程工具：按端口定位监听进程、taskkill 兜底
# ---------------------------------------------------------------------------
def _winproc():
    if os.name != "nt":
        pytest.skip("Win32 进程工具只在 Windows 上可用")
    from localai.tray import winproc

    return winproc


def _spawn_sleeper(seconds: int = 60) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({int(seconds)})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def test_find_listening_pid_returns_own_process() -> None:
    winproc = _winproc()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert winproc.find_listening_pid(port) == os.getpid()
    finally:
        sock.close()
    assert winproc.find_listening_pid(port) is None


def test_kill_process_tree_ends_a_real_process() -> None:
    winproc = _winproc()
    proc = _spawn_sleeper()
    try:
        assert winproc.kill_process_tree(proc.pid) is True
        proc.wait(timeout=10)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# 挂靠模式的启动气泡：退出托盘会关服务
# ---------------------------------------------------------------------------
def test_attached_balloon_says_quitting_tray_closes_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    winicon = _winicon()
    shell = _FakeShell32()
    monkeypatch.setattr(winicon, "_shell32", shell)
    window = winicon.TrayWindow(
        object(),
        _FakeModeController(winicon.MODE_ATTACHED),
        icon_path=tmp_path / "tray.ico",
    )
    window._hwnd = 4321

    window._announce_initial()
    _keep = window

    infos = [data for _op, data in shell.calls if data.uFlags & winicon.NIF_INFO]
    assert len(infos) == 1
    assert infos[0].szInfoTitle == "服务已在运行"
    assert infos[0].szInfo == "退出托盘会一并关闭服务。"
    assert "通知区" not in infos[0].szInfo
    assert "^" not in infos[0].szInfo
