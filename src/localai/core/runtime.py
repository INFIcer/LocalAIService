"""托盘与服务之间的运行时约定：优雅退出令牌文件。

托盘拉起服务时把随机令牌注入子进程环境（``LOCALAI_SHUTDOWN_TOKEN``）；服务启动
后把令牌写进 ``<HOME>/run/shutdown-<port>.token``，正常退出时删掉。这样挂靠模式
（服务不是本托盘拉起，例如执行过 WebUI 的「重启服务」）也能读到令牌，走同一条
``POST /admin/shutdown`` 优雅关停路径，而不是只能强杀。

安全边界：令牌文件落在用户自己的数据根目录，权限与 ``config.json`` 同级（那里本来
就可能存 ``LOCALAI_TOKEN``）；同一用户下的进程本来就能直接结束服务进程，所以这个
文件不额外扩大能力面。``LOCALAI_TOKEN`` 若已配置，``/admin/*`` 仍要求 Bearer，
文件里的令牌不能绕过它。文件名按端口区分，同一 ``HOME`` 下的多个实例互不覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: 托盘注入服务环境的退出令牌变量名（服务自己也读它）。
SHUTDOWN_TOKEN_ENV = "LOCALAI_SHUTDOWN_TOKEN"

#: 令牌文件所在目录与文件名模板；``*.token`` 已在 .gitignore 中。
TOKEN_DIR_NAME = "run"
TOKEN_FILE_TEMPLATE = "shutdown-{port}.token"


def shutdown_token_path(home: Path, port: int) -> Path:
    """某端口对应的令牌文件；服务与托盘都按这个路径读写。"""
    return Path(home) / TOKEN_DIR_NAME / TOKEN_FILE_TEMPLATE.format(port=int(port))


def write_shutdown_token(home: Path, port: int, token: str) -> Optional[Path]:
    """原子写入令牌文件；目录不可写时返回 None（托盘退回强杀路径）。"""
    token = str(token or "").strip()
    if not token:
        return None
    path = shutdown_token_path(home, port)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(token + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def read_shutdown_token(home: Path, port: int) -> str:
    """读令牌；文件不存在或不可读时返回空串（调用方据此跳过优雅关停）。"""
    try:
        return shutdown_token_path(home, port).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def remove_shutdown_token(home: Path, port: int, token: str) -> None:
    """删除令牌文件，仅当内容仍是自己写的那份。

    服务可能被 WebUI 重启换成了新进程：新进程先启动并写了自己的令牌，旧进程
    随后退出时不能把新文件删掉。
    """
    token = str(token or "").strip()
    if not token:
        return
    if read_shutdown_token(home, port) != token:
        return
    try:
        shutdown_token_path(home, port).unlink(missing_ok=True)
    except OSError:
        pass


__all__ = [
    "SHUTDOWN_TOKEN_ENV",
    "read_shutdown_token",
    "remove_shutdown_token",
    "shutdown_token_path",
    "write_shutdown_token",
]
