"""托盘启动入口：python -m localai.tray / localai tray。

托盘是父进程：先探测端口，能拉起就拉起服务，已在跑就只挂图标；
退出时只收自己拉起的进程。日志写 <LOCALAI_HOME>/logs/tray.log。
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import logging.handlers
import os
import sys
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path
from typing import Optional

log = logging.getLogger("localai.tray")

MUTEX_NAME = "Local\\LocalAIService-Tray"
ERROR_ALREADY_EXISTS = 183

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


def _run(args: argparse.Namespace) -> int:
    from ..config import get_config

    cfg = get_config()
    port = int(args.port or cfg.port)
    host = str(args.host or cfg.host)
    _configure_logging(cfg.home)

    from .actions import TrayActions, TrayController
    from .winicon import (
        TrayWindow,
        request_open_from_existing,
        set_clipboard_text,
        show_error,
    )

    handle = _acquire_instance_mutex()
    if handle is None:
        log.info("托盘已经在运行，交给已有实例打开页面")
        for _ in range(30):
            if request_open_from_existing():
                return 0
            time.sleep(0.1)
        return 0

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
        if code != 0 and controller.mode == "owned":
            # 图标没建起来也要把服务收掉，避免留下不可见的后台进程。
            controller.quit()
        if controller.mode == "owned":
            log.info("托盘退出，服务进程已结束=%s", controller.process_exited())
        return code
    except Exception as exc:
        log.exception("托盘启动失败")
        if controller is not None and controller.mode == "owned":
            try:
                controller.quit()
            except Exception:  # noqa: BLE001 - 清理失败只记录
                log.exception("清理托盘拉起的服务时出错")
        show_error(
            "LocalAIService 托盘启动失败",
            f"{type(exc).__name__}: {exc}\n\n日志：{cfg.home}\\logs\\tray.log",
        )
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
    except Exception as exc:  # noqa: BLE001 - pythonw 下没有控制台，写日志兜底
        try:
            from ..config import get_config

            _configure_logging(get_config().home)
        except Exception:  # noqa: BLE001
            pass
        log.exception("托盘未捕获异常：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
