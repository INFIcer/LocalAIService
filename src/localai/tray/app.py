"""托盘启动入口：python -m localai.tray / localai tray。

托盘是父进程：先探测端口，能拉起就拉起服务，已在跑就只挂图标；
退出时只收自己拉起的进程。日志写 <LOCALAI_HOME>/logs/tray.log。

pythonw 没有控制台，任何失败都必须留下两条痕迹：完整 traceback 写
<LOCALAI_HOME>/logs/tray-error.log，一句话中文提示用 MessageBoxW 弹出来。
第二次双击不会静默退出：叫醒已有托盘打开页面，再弹框说明图标在哪。
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import logging.handlers
import os
import sys
import time
import traceback
import webbrowser
from ctypes import wintypes
from pathlib import Path
from typing import Optional

log = logging.getLogger("localai.tray")

MUTEX_NAME = "Local\\LocalAIService-Tray"
ERROR_ALREADY_EXISTS = 183
ERROR_LOG_NAME = "tray-error.log"

#: 第二个实例的退出码：调用方（脚本、人）靠它区分「已有托盘」和「启动失败」。
EXIT_ALREADY_RUNNING = 3

def _kernel32():
    # 延迟加载：非 Windows 上只 import 这个模块时不应因为 WinDLL 失败。
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    library.CreateMutexW.restype = wintypes.HANDLE
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    library.CloseHandle.restype = wintypes.BOOL
    return library


def _configure_logging(home: Path) -> None:
    logs_dir = Path(home) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in log.handlers):
        handler = logging.handlers.RotatingFileHandler(
            logs_dir / "tray.log", maxBytes=512 * 1024, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)


def error_log_path(home: Path) -> Path:
    return Path(home) / "logs" / ERROR_LOG_NAME


def write_error_log(home: Path, title: str, exc: BaseException) -> Optional[Path]:
    """把完整 traceback 写进 logs/tray-error.log；失败时返回 None。"""
    path = error_log_path(home)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{stamp} {title}\n{body}", encoding="utf-8")
        return path
    except OSError:
        log.exception("写入错误日志失败：%s", path)
        return None


def _show_dialog(title: str, text: str) -> None:
    """失败提示的唯一可见通道；winicon 只在 Windows 上可导入，所以延迟 import。"""
    try:
        from .winicon import show_error

        show_error(title, text)
    except Exception:  # noqa: BLE001 - 连弹框都失败时只剩日志
        log.exception("弹出提示框失败")


def report_fatal(home: Path, title: str, exc: BaseException) -> None:
    """记录完整 traceback，再弹一句能照着做的短提示（不贴 traceback）。"""
    log.error("%s：%s: %s", title, type(exc).__name__, exc, exc_info=exc)
    path = write_error_log(home, title, exc)
    detail = f"{type(exc).__name__}: {exc}".strip()
    if len(detail) > 300:
        detail = detail[:300] + "…"
    detail = detail or "未知错误"
    where = f"\n\n详细错误：{path}" if path is not None else ""
    _show_dialog(title, f"{detail}{where}")


def _configured_home() -> Path:
    """取数据根目录；配置读不出来时退回环境变量与仓库根，日志仍要写得出去。"""
    try:
        from ..config import get_config

        return Path(get_config().home)
    except Exception:  # noqa: BLE001
        raw = os.environ.get("LOCALAI_HOME") or os.environ.get("INDEX_TTS_HOME")
        if raw:
            return Path(raw).expanduser()
        return Path(__file__).resolve().parents[3]


def _acquire_instance_mutex() -> Optional[int]:
    """同名互斥量保证只有一个托盘；返回句柄，已存在则返回 None。"""
    kernel32 = _kernel32()
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        return None
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return int(handle)


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="localai tray",
        description="Windows 通知区托盘：后台启动服务、打开 WebUI、退出时收掉自己拉起的服务",
    )
    parser.add_argument("--port", type=int, help="服务端口（默认 LOCALAI_PORT / 7880）")
    parser.add_argument("--host", help="监听地址（默认 LOCALAI_HOST / 127.0.0.1）")
    return parser.parse_args(argv)


def _wake_existing_tray(timeout: float = 3.0) -> bool:
    """叫醒已有托盘打开页面：第二次双击至少有一个可见结果。"""
    try:
        from .winicon import request_open_from_existing
    except Exception:  # noqa: BLE001 - 只影响「顺手打开页面」，不影响提示
        return False
    deadline = time.monotonic() + timeout
    while True:
        try:
            if request_open_from_existing():
                return True
        except Exception:  # noqa: BLE001
            log.debug("叫醒已有托盘失败", exc_info=True)
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def report_already_running(port: int) -> int:
    """第二个实例：不重复建图标，但必须让人看见「已经有一个在跑」。"""
    log.warning("托盘已在运行，本次双击只提示图标位置，退出码 %s", EXIT_ALREADY_RUNNING)
    opened = _wake_existing_tray()
    log.info("已叫醒已有托盘打开页面=%s", opened)
    _show_dialog(
        "LocalAIService 托盘",
        "托盘已经在运行，看任务栏右下角的通知区（图标可能收在 ^ 里）。\n\n"
        f"双击图标可打开 WebUI（端口 {int(port)}），右键图标可以退出。",
    )
    return EXIT_ALREADY_RUNNING


def _run(args: argparse.Namespace) -> int:
    from ..config import get_config

    cfg = get_config()
    port = int(args.port or cfg.port)
    host = str(args.host or cfg.host)
    _configure_logging(cfg.home)

    handle = _acquire_instance_mutex()
    if handle is None:
        return report_already_running(port)

    from .actions import MODE_OWNED, TrayActions, TrayController
    from .winicon import TrayWindow, set_clipboard_text

    controller: Optional[TrayController] = None
    try:
        controller = TrayController(
            port=port,
            home=cfg.home,
            host=host,
            bearer_token=cfg.token,
        )
        controller.start()
        if controller.mode == "conflict":
            log.error(
                "端口 %s 被其它程序占用，未启动服务；可先退出占用程序再双击启动",
                port,
            )
        actions = TrayActions(
            controller,
            open_url=webbrowser.open,
            copy_text=set_clipboard_text,
        )
        window = TrayWindow(
            actions,
            controller,
            icon_path=Path(__file__).with_name("tray.ico"),
        )
        code = window.run()
        if code != 0 and controller.mode == MODE_OWNED:
            # 图标没建起来也要把服务收掉，避免留下不可见的后台进程。
            controller.quit()
        if controller.mode == MODE_OWNED:
            log.info("托盘退出，服务进程已结束=%s", controller.process_exited())
        return code
    except Exception as exc:  # noqa: BLE001 - 任何失败都要留痕并弹框
        if controller is not None and controller.mode == MODE_OWNED:
            try:
                controller.quit()
            except Exception:  # noqa: BLE001 - 清理失败只记录
                log.exception("清理托盘拉起的服务时出错")
        report_fatal(cfg.home, "LocalAIService 托盘启动失败", exc)
        return 1
    finally:
        if handle:
            _kernel32().CloseHandle(handle)


def main(argv: Optional[list[str]] = None) -> int:
    if os.name != "nt":
        print("托盘只支持 Windows。", file=sys.stderr)
        return 2
    args = _parse_args(argv)
    try:
        return _run(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - pythonw 下没有控制台，日志 + 弹框兜底
        home = _configured_home()
        try:
            _configure_logging(home)
        except Exception:  # noqa: BLE001 - 日志目录都不可写时仍要弹框
            pass
        report_fatal(home, "LocalAIService 托盘启动失败", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
