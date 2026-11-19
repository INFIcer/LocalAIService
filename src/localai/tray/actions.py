"""托盘的动作层：URL、菜单契约、服务探测与进程收尾。

UI 层（winicon）只把点击事件翻译成这里的动作 id；本模块不创建窗口、不打开
浏览器，也不直接 import ctypes（Win32 细节在 winproc 里按需加载），因此可以在
普通测试里替换掉所有外部副作用。
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol

from ..core.runtime import SHUTDOWN_TOKEN_ENV, read_shutdown_token

log = logging.getLogger("localai.tray")

#: 与服务约定的退出头：令牌由服务自己持有，见 core.runtime。
SHUTDOWN_HEADER = "X-LocalAI-Shutdown-Token"
SHUTDOWN_PATH = "/admin/shutdown"

ACTION_OPEN = "open"
ACTION_COPY = "copy"
ACTION_CLEAR_VRAM = "clear_vram"
ACTION_QUIT = "quit"

#: 气泡图标级别；与 Shell_NotifyIcon 的 NIIF_* 同值，避免动作层依赖 ctypes。
NOTIFY_INFO = 1
NOTIFY_WARNING = 2
NOTIFY_ERROR = 3

#: 右键菜单：id、文案。文字保持短，给普通人看。
MENU_ITEMS: tuple[tuple[str, str], ...] = (
    (ACTION_OPEN, "打开 WebUI"),
    (ACTION_COPY, "复制地址"),
    (ACTION_CLEAR_VRAM, "清理显存"),
    (ACTION_QUIT, "退出"),
)

#: 菜单命令 id 只在一处定义，UI 与测试共用。
MENU_IDS: dict[str, int] = {
    action: 1001 + index for index, (action, _label) in enumerate(MENU_ITEMS)
}
ACTION_BY_MENU_ID: dict[int, str] = {menu_id: action for action, menu_id in MENU_IDS.items()}

STATE_FREE = "free"
STATE_RUNNING = "running"
STATE_CONFLICT = "conflict"

MODE_OWNED = "owned"
MODE_ATTACHED = "attached"
MODE_CONFLICT = "conflict"

#: 端口刚空出来时先看 /health 会不会回来：服务重启的空窗里 bind 是成功的，
#: 直接拉起会撞端口。窗口约 5 秒，与重启脚本的实际停顿时长同量级。
RESTART_WAIT_SECONDS = 5.0
RESTART_WAIT_INTERVAL = 0.25
RESTART_HEALTH_TIMEOUT = 0.5

#: 优雅退出最多等多久：服务要停调度器、取消看门狗、关任务库、停视觉子进程。
#: 与 stop_owned_process 的 graceful_timeout 取同一个数量级，超时后转强杀。
GRACEFUL_EXIT_SECONDS = 20.0
GRACEFUL_POLL_INTERVAL = 0.25


def ui_url(port: int) -> str:
    """浏览器打开的地址：托盘只在本机，固定走回环。"""
    return f"http://127.0.0.1:{int(port)}/ui"


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}/health"


def build_tooltip(port: int, mode: str) -> str:
    """通知区悬浮文字：必须能一眼看到服务名和端口。"""
    port = int(port)
    if mode == "starting":
        return f"LocalAIService · 启动中… · 127.0.0.1:{port}"
    if mode == MODE_ATTACHED:
        return f"LocalAIService · 外部运行 · 127.0.0.1:{port}"
    if mode == MODE_CONFLICT:
        return f"LocalAIService · 端口 {port} 被占用"
    if mode == "failed":
        return f"LocalAIService · 启动失败 · 127.0.0.1:{port}"
    return f"LocalAIService · 127.0.0.1:{port}"


def is_localai_health(payload: object) -> bool:
    """分辨「本地 AI 服务」和碰巧占用同一端口的其它程序。"""
    if not isinstance(payload, Mapping):
        return False
    if not isinstance(payload.get("ok"), bool):
        return False
    return {"backend", "jobs", "vision", "asr"} <= set(payload.keys())


def _direct_opener() -> urllib.request.OpenerDirector:
    # 回环请求不应走系统代理；显式清空代理表。
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_get_json(url: str, timeout: float = 2.5) -> object:
    try:
        with _direct_opener().open(url, timeout=timeout) as response:  # noqa: S310
            if not 200 <= int(response.status) < 300:
                return None
            return json.loads(response.read(64 * 1024).decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - 探测失败按「不是本服务」处理
        return None


def port_free(port: int) -> bool:
    """探测回环端口是否可绑定。

    不用 connect 判断：本机防火墙会让未监听端口的 connect 挂到超时，
    看起来像「有人监听」。bind 成功才是真正的空闲。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def probe_service(port: int, *, timeout: float = 2.5) -> str:
    """返回 free / running / conflict，用来决定「拉起、挂靠还是只提示」。"""
    if port_free(port):
        return STATE_FREE
    payload = http_get_json(health_url(port), timeout=timeout)
    return STATE_RUNNING if is_localai_health(payload) else STATE_CONFLICT


def wait_for_restarting_service(
    port: int,
    *,
    wait_seconds: float = RESTART_WAIT_SECONDS,
    interval: float = RESTART_WAIT_INTERVAL,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """端口刚空闲时先短轮询 /health：服务可能正在重启，等它自己回来。

    返回 STATE_RUNNING（服务已回来，应挂靠）或 STATE_FREE（等满仍没有，可拉起）。
    只认本服务形状的 /health：别的程序抢到端口不会让托盘误挂靠。
    """
    deadline = monotonic() + float(wait_seconds)
    while True:
        if is_localai_health(http_get_json(health_url(port), timeout=RESTART_HEALTH_TIMEOUT)):
            return STATE_RUNNING
        if monotonic() >= deadline:
            return STATE_FREE
        sleep(interval)


def wait_for_service_stop(
    port: int,
    *,
    wait_seconds: float = GRACEFUL_EXIT_SECONDS,
    interval: float = GRACEFUL_POLL_INTERVAL,
    probe: Optional[Callable[[int], str]] = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """等「本服务」不再应答：返回 True（已关）或 False（等满超时）。

    判定口径与 probe_service 一致：端口空出来、或者被换成了别的程序，都算本服务
    已经不在；仍在应答本服务形状的 /health 才算没关掉。
    """
    check = probe or probe_service  # 调用时再取模块属性，测试可以打桩
    deadline = monotonic() + float(wait_seconds)
    while True:
        if check(port) != STATE_RUNNING:
            return True
        if monotonic() >= deadline:
            return False
        sleep(interval)


@dataclass(frozen=True)
class ShutdownRequest:
    url: str
    headers: tuple[tuple[str, str], ...]


def build_shutdown_request(port: int, token: str, bearer_token: str = "") -> ShutdownRequest:
    headers = [(SHUTDOWN_HEADER, token)]
    if bearer_token:
        headers.append(("Authorization", f"Bearer {bearer_token}"))
    return ShutdownRequest(f"http://127.0.0.1:{int(port)}{SHUTDOWN_PATH}", tuple(headers))


class ShutdownClient:
    """向托盘拉起的服务发退出请求；服务用它自己收尾（关调度器、停子进程）。"""

    def __init__(
        self,
        port: int,
        token: str,
        bearer_token: str = "",
        *,
        timeout: float = 5.0,
        opener: Optional[urllib.request.OpenerDirector] = None,
    ) -> None:
        self._request = build_shutdown_request(port, token, bearer_token)
        self._timeout = timeout
        self._opener = opener or _direct_opener()

    def request(self) -> bool:
        request = urllib.request.Request(
            self._request.url,
            data=b"{}",
            method="POST",
            headers=dict(self._request.headers),
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:  # noqa: S310
                return 200 <= int(response.status) < 300
        except Exception as exc:  # noqa: BLE001 - 失败就走兜底 kill
            log.info("优雅退出请求失败：%s", exc)
            return False


VRAM_UNLOAD_PATH = "/vram/unload"
VRAM_UNLOAD_TIMEOUT = 5.0

#: 显存接口返回的能力名 -> 给普通人看的中文短名。
MODEL_LABELS: dict[str, str] = {
    "tts": "语音合成",
    "asr": "语音识别",
    "whisper": "语音识别",
    "speaker": "声纹",
    "separate": "分轨",
    "translate": "翻译",
    "vision": "识图",
}


@dataclass(frozen=True)
class VramUnloadRequest:
    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes


def build_vram_unload_request(port: int, bearer_token: str = "") -> VramUnloadRequest:
    """手动释放全部模型：忙的跳过，服务端不会等长推理。"""
    headers = [("Content-Type", "application/json")]
    if bearer_token:
        headers.append(("Authorization", f"Bearer {bearer_token}"))
    body = json.dumps({"all": True, "skip_busy": True}).encode("utf-8")
    return VramUnloadRequest(
        f"http://127.0.0.1:{int(port)}{VRAM_UNLOAD_PATH}", tuple(headers), body
    )


class VramUnloadClient:
    """让服务卸载全部已加载模型；失败返回 None，由调用方翻译成一句人话。"""

    def __init__(
        self,
        port: int,
        bearer_token: str = "",
        *,
        timeout: float = VRAM_UNLOAD_TIMEOUT,
        opener: Optional[urllib.request.OpenerDirector] = None,
    ) -> None:
        self._request = build_vram_unload_request(port, bearer_token)
        self._timeout = timeout
        self._opener = opener or _direct_opener()

    def request(self) -> Optional[dict[str, object]]:
        request = urllib.request.Request(
            self._request.url,
            data=self._request.body,
            method="POST",
            headers=dict(self._request.headers),
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:  # noqa: S310
                if not 200 <= int(response.status) < 300:
                    log.info("清理显存请求失败：HTTP %s", response.status)
                    return None
                payload = json.loads(
                    response.read(64 * 1024).decode("utf-8", errors="replace")
                )
        except Exception as exc:  # noqa: BLE001 - 服务没起来/超时都按失败提示
            log.info("清理显存请求失败：%s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        return payload


def _model_labels(names: object) -> list[str]:
    """把接口返回的能力名换成中文短名，保留顺序并去掉重复（asr/whisper 同名）。"""
    if not isinstance(names, (list, tuple)):
        return []
    labels: list[str] = []
    for name in names:
        label = MODEL_LABELS.get(str(name), str(name))
        if label and label not in labels:
            labels.append(label)
    return labels


def build_clear_vram_notice(
    result: Optional[Mapping[str, object]],
) -> tuple[str, str, int]:
    """把 /vram/unload 的返回值翻译成气泡标题、正文与图标级别。"""
    if result is None:
        return ("清理失败", "服务没响应，稍后再试。", NOTIFY_ERROR)
    released = _model_labels(result.get("released"))
    skipped = _model_labels(result.get("skipped"))
    if released:
        text = f"已卸下 {'、'.join(released)}"
        if skipped:
            text += f"；正在用的 {'、'.join(skipped)} 没动"
        return ("显存已清理", text, NOTIFY_INFO)
    if skipped:
        return (
            "暂时没腾出显存",
            f"正在用的 {'、'.join(skipped)} 没动；等它跑完再试一次。",
            NOTIFY_WARNING,
        )
    return ("显存已清理", "现在没有模型占着显存。", NOTIFY_INFO)


class ProcessLike(Protocol):
    def poll(self) -> Optional[int]: ...
    def wait(self, timeout: Optional[float] = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


class ServiceProcess:
    """Popen 包装：venv 里的 python(w).exe 是转发器，真正的服务在它的子进程里。

    直接 terminate 只结束转发器会留下服务孤儿，所以兜底用 taskkill /T 结束整棵树。
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self.pid = proc.pid

    def poll(self) -> Optional[int]:
        return self._proc.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        return self._proc.wait(timeout=timeout)

    def _kill_tree(self) -> None:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(self.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        except Exception:  # noqa: BLE001 - taskkill 不可用时退回结束转发器
            self._proc.terminate()

    def terminate(self) -> None:
        self._kill_tree()

    def kill(self) -> None:
        self._kill_tree()


@dataclass
class StopResult:
    owned: bool
    steps: list[str] = field(default_factory=list)
    exit_code: Optional[int] = None


def stop_owned_process(
    proc: Optional[ProcessLike],
    request_shutdown: Callable[[], bool],
    *,
    graceful_timeout: float = 20.0,
    terminate_timeout: float = 5.0,
    kill_timeout: float = 5.0,
) -> StopResult:
    """自己拉起的那份服务的退出顺序：请求自收尾 -> terminate -> kill。

    挂靠模式没有进程句柄，走 TrayController 的「按端口定位 + 优雅请求 + 强杀」
    路径；两条路径都先请求服务自己收尾，强杀只作兜底。
    """
    result = StopResult(owned=True)
    if proc is None or proc.poll() is not None:
        result.steps.append("already-exited")
        return result

    try:
        requested = bool(request_shutdown())
    except Exception:  # noqa: BLE001
        requested = False
    if requested:
        result.steps.append("shutdown-requested")
        try:
            result.exit_code = proc.wait(timeout=graceful_timeout)
            result.steps.append("graceful-exit")
            return result
        except subprocess.TimeoutExpired:
            result.steps.append("graceful-timeout")

    result.steps.append("terminate")
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001
        result.steps.append("terminate-error")
    try:
        result.exit_code = proc.wait(timeout=terminate_timeout)
        result.steps.append("terminated")
        return result
    except subprocess.TimeoutExpired:
        result.steps.append("terminate-timeout")

    result.steps.append("kill")
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        result.steps.append("kill-error")
    try:
        result.exit_code = proc.wait(timeout=kill_timeout)
        result.steps.append("killed")
    except subprocess.TimeoutExpired:
        result.steps.append("kill-timeout")
    return result


@dataclass
class SpawnedService:
    proc: ProcessLike
    command: list[str]
    token: str


def _default_find_listener(port: int) -> Optional[int]:
    """谁在监听这个端口；查不到时返回 None（调用方按「不动手」处理）。"""
    from .winproc import find_listening_pid  # 延迟加载：动作层不直接依赖 ctypes

    try:
        return find_listening_pid(port)
    except Exception:  # noqa: BLE001 - 查询失败只影响兜底强杀
        log.debug("查询监听进程失败", exc_info=True)
        return None


def _default_kill_tree(pid: int) -> bool:
    from .winproc import kill_process_tree

    try:
        return kill_process_tree(pid)
    except Exception:  # noqa: BLE001
        log.debug("结束进程树失败", exc_info=True)
        return False


def _default_health(port: int) -> object:
    return http_get_json(health_url(port))


class TrayController:
    """服务生命周期：探测端口、决定是否拉起、退出时停掉当前服务。

    托盘是服务的生命周期所有者：退出托盘 = 服务退出。服务可能是本托盘拉起的
    （owned），也可能是启动时已经在跑、托盘挂靠上去的（attached），还可能被
    WebUI 的「重启服务」换成了新进程；退出时一律按「当前监听该端口、且 /health
    形状是本服务」的进程处置，端口属于别的程序时不动手。
    """

    def __init__(
        self,
        *,
        port: int,
        home: Path,
        host: str = "127.0.0.1",
        bearer_token: str = "",
        python_executable: Optional[str] = None,
        probe: Callable[[int], str] = probe_service,
        shutdown_client: Optional[Callable[[], bool]] = None,
        spawn: Optional[Callable[[str], SpawnedService]] = None,
        health_wait: Optional[Callable[[int], str]] = None,
        find_listener: Optional[Callable[[int], Optional[int]]] = None,
        kill_tree: Optional[Callable[[int], bool]] = None,
        wait_stop: Optional[Callable[[int], bool]] = None,
        health: Optional[Callable[[int], object]] = None,
        token_reader: Callable[[Path, int], str] = read_shutdown_token,
        log_: Optional[logging.Logger] = None,
    ) -> None:
        self.port = int(port)
        self.home = Path(home)
        self.host = host
        self.bearer_token = bearer_token
        self.python_executable = python_executable or sys.executable
        self.proc: Optional[ProcessLike] = None
        self.mode = ""
        self.shutdown_token = ""
        self._probe = probe
        self._spawn_override = spawn
        self._shutdown_override = shutdown_client
        self._health_wait = health_wait or wait_for_restarting_service
        self._find_listener = find_listener or _default_find_listener
        self._kill_tree = kill_tree or _default_kill_tree
        self._wait_stop = wait_stop or wait_for_service_stop
        self._health = health or _default_health
        self._token_reader = token_reader
        self._log = log_ or log

    # -- 状态 ---------------------------------------------------------
    def start(self) -> str:
        state = self._probe(self.port)
        if state == STATE_FREE:
            # bind 成功不等于没人管：服务可能刚被重启流程结束、新进程还没起来。
            # 先短等 /health 回来，避免在重启空窗里拉起第二份。
            self._log.info(
                "端口 %s 空闲，先确认不是服务重启的空窗（最多 %s 秒）",
                self.port,
                RESTART_WAIT_SECONDS,
            )
            if self._health_wait(self.port) == STATE_RUNNING:
                state = STATE_RUNNING
        if state == STATE_RUNNING:
            self.mode = MODE_ATTACHED
            self._log.info("端口 %s 上已有服务，托盘只挂图标，不重复启动", self.port)
            return self.mode
        if state == STATE_CONFLICT:
            self.mode = MODE_CONFLICT
            self._log.warning("端口 %s 被其它程序占用，未启动服务", self.port)
            return self.mode

        self.shutdown_token = secrets.token_hex(16)
        spawned = (self._spawn_override or self._spawn)(self.shutdown_token)
        self.proc = spawned.proc
        self.mode = MODE_OWNED
        self._log.info("已启动服务：pid=%s", getattr(spawned.proc, "pid", "?"))
        return self.mode

    def health_state(self) -> str:
        return self._probe(self.port)

    def process_exited(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None

    def quit(self) -> StopResult:
        """退出托盘：先收自己拉起的进程，再按端口处置「当前服务」。

        两步都遵守「先请求自收尾、超时再强杀」；第二步覆盖挂靠模式，以及服务被
        WebUI 的「重启服务」换成新进程的情况。端口属于别的程序时第二步不动手。
        """
        result = StopResult(owned=self.mode == MODE_OWNED)
        self._stop_owned_process(result)
        self._stop_current_service(result)
        self._log.info("托盘退出：%s", " -> ".join(result.steps))
        return result

    # -- 内部 ---------------------------------------------------------
    def _stop_owned_process(self, result: StopResult) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        owned = stop_owned_process(self.proc, self._request_shutdown)
        result.steps.extend(owned.steps)
        result.exit_code = owned.exit_code

    def _stop_current_service(self, result: StopResult) -> None:
        """按端口再判定一次：优雅关停优先，拿不到令牌或服务不配合时才强杀。"""
        state = self._probe(self.port)
        if state == STATE_FREE:
            # 自己拉起的进程已经收干净；「端口空着」不必单独记一步。
            if not result.steps:
                result.steps.append("no-service")
            return
        if state == STATE_CONFLICT:
            # 端口上不是本服务：绝不碰它。
            result.steps.append("foreign-port")
            return

        token = self._shutdown_token()
        if not token:
            result.steps.append("no-token")
        elif self._request_shutdown_with(token):
            result.steps.append("shutdown-requested")
            if self._wait_stop(self.port):
                result.steps.append("graceful-exit")
                return
            result.steps.append("graceful-timeout")
        else:
            result.steps.append("shutdown-refused")

        pid = self._find_listener(self.port)
        if not pid:
            result.steps.append("no-listener")
            return
        if not self._is_our_service(pid):
            # 定位到进程后再确认一次身份：两步之间端口可能已经换了主人。
            result.steps.append("foreign-port")
            return
        result.steps.append("kill-tree")
        result.steps.append("tree-killed" if self._kill_tree(pid) else "tree-kill-failed")

    def _shutdown_token(self) -> str:
        """优雅退出令牌：自己的优先，否则读服务写下的令牌文件（挂靠模式）。"""
        if self.shutdown_token:
            return self.shutdown_token
        try:
            return str(self._token_reader(self.home, self.port) or "")
        except Exception:  # noqa: BLE001 - 读不到就当没有令牌
            return ""

    def _is_our_service(self, pid: int) -> bool:
        """确认 pid 就是应答 /health 的那个进程（形状判定 + 可选的 pid 对比）。"""
        try:
            payload = self._health(self.port)
        except Exception:  # noqa: BLE001
            return False
        if not is_localai_health(payload):
            return False
        reported = payload.get("pid") if isinstance(payload, Mapping) else None
        if reported is None:
            return True
        try:
            return int(reported) == int(pid)
        except (TypeError, ValueError):
            return False

    def _request_shutdown(self) -> bool:
        return self._request_shutdown_with(self._shutdown_token())

    def _request_shutdown_with(self, token: str) -> bool:
        if self._shutdown_override is not None:
            return bool(self._shutdown_override())
        if not token:
            return False
        return ShutdownClient(self.port, token, self.bearer_token).request()

    def _spawn(self, token: str) -> SpawnedService:
        logs_dir = self.home / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path = logs_dir / "service.out.log"
        err_path = logs_dir / "service.err.log"
        env = os.environ.copy()
        env["LOCALAI_SHUTDOWN_TOKEN"] = token
        # --port/--host 也要写回环境，否则服务内部的 cfg 会按旧值报告监听地址。
        env["LOCALAI_PORT"] = str(self.port)
        env["LOCALAI_HOST"] = self.host
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        command = [
            self.python_executable,
            "-m",
            "localai",
            "serve",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        self._log.info("拉起服务：%s", " ".join(command))
        with out_path.open("ab") as out, err_path.open("ab") as err:
            proc = subprocess.Popen(
                command,
                cwd=str(self.home),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                creationflags=creationflags,
                close_fds=True,
            )
        return SpawnedService(proc=ServiceProcess(proc), command=command, token=token)


class TrayActions:
    """菜单动作与 UI 的唯一接口；测试只替换 opener / clipper / controller。"""

    def __init__(
        self,
        controller: TrayController,
        *,
        open_url: Callable[[str], object],
        copy_text: Callable[[str], object],
        unload_client: Optional[Callable[[], Optional[Mapping[str, object]]]] = None,
        notify: Optional[Callable[[str, str, int], object]] = None,
        log_: Optional[logging.Logger] = None,
    ) -> None:
        self.controller = controller
        self._open_url = open_url
        self._copy_text = copy_text
        self._unload_client = unload_client
        self._notify = notify
        self._log = log_ or log

    def set_notify(self, notify: Callable[[str, str, int], object]) -> None:
        """UI 建好后接回调：清理显存是异步动作，结果要靠它弹气泡。"""
        self._notify = notify

    def _clear_vram(self) -> Optional[Mapping[str, object]]:
        if self._unload_client is not None:
            return self._unload_client()
        return VramUnloadClient(
            self.controller.port, getattr(self.controller, "bearer_token", "")
        ).request()

    def perform(self, action: str) -> str:
        if action == ACTION_OPEN:
            url = ui_url(self.controller.port)
            self._log.info("打开 WebUI：%s", url)
            self._open_url(url)
            return ACTION_OPEN
        if action == ACTION_COPY:
            url = ui_url(self.controller.port)
            self._copy_text(url)
            self._log.info("已复制地址：%s", url)
            return ACTION_COPY
        if action == ACTION_CLEAR_VRAM:
            try:
                result = self._clear_vram()
            except Exception:  # noqa: BLE001 - 请求态失败也要翻译成一句人话
                self._log.exception("清理显存请求出错")
                result = None
            title, text, flag = build_clear_vram_notice(result)
            self._log.info("清理显存：%s", text)
            if self._notify is not None:
                try:
                    self._notify(title, text, flag)
                except Exception:  # noqa: BLE001 - 气泡失败不影响动作结果
                    self._log.exception("清理显存结果通知失败")
            return ACTION_CLEAR_VRAM
        if action == ACTION_QUIT:
            self._log.info("收到退出请求")
            self.controller.quit()
            return ACTION_QUIT
        raise ValueError(f"未知托盘动作：{action!r}")


__all__ = [
    "ACTION_BY_MENU_ID",
    "ACTION_CLEAR_VRAM",
    "ACTION_COPY",
    "ACTION_OPEN",
    "ACTION_QUIT",
    "MENU_IDS",
    "MENU_ITEMS",
    "MODEL_LABELS",
    "NOTIFY_ERROR",
    "NOTIFY_INFO",
    "NOTIFY_WARNING",
    "MODE_ATTACHED",
    "MODE_CONFLICT",
    "MODE_OWNED",
    "STATE_CONFLICT",
    "STATE_FREE",
    "STATE_RUNNING",
    "GRACEFUL_EXIT_SECONDS",
    "ServiceProcess",
    "ShutdownClient",
    "ShutdownRequest",
    "SpawnedService",
    "StopResult",
    "TrayActions",
    "TrayController",
    "VramUnloadClient",
    "VramUnloadRequest",
    "build_clear_vram_notice",
    "build_shutdown_request",
    "build_tooltip",
    "build_vram_unload_request",
    "health_url",
    "is_localai_health",
    "port_free",
    "probe_service",
    "stop_owned_process",
    "ui_url",
    "wait_for_restarting_service",
    "wait_for_service_stop",
]
