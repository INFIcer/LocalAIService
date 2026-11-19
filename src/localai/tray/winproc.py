"""Windows 进程工具（ctypes + taskkill，零第三方依赖）。

托盘只依赖这里的几个语义，不直接接触 Win32：

- ``find_listening_pid(port)``：读系统 TCP 监听表，回答「谁占着这个端口」。用
  ``GetExtendedTcpTable`` 而不是解析 ``netstat -ano``：后者输出随系统语言变化，
  列位置也不稳定；
- ``kill_process_tree(pid)``：``taskkill /T /F``，只作为优雅关停失败后的兜底；
- 作业对象：托盘自己拉起的服务连同它的子孙进程（视觉引擎等）一起挂进
  ``kill-on-close`` 作业。托盘进程无论正常退出还是被任务管理器强杀/崩溃，内核
  都会在最后一个作业句柄关闭时结束整棵树。挂靠模式的外部服务没有子进程句柄，
  不在此列。
"""

from __future__ import annotations

import ctypes
import socket
import subprocess
from ctypes import wintypes
from typing import Iterable, Optional

AF_INET = 2
AF_INET6 = 23
TCP_TABLE_OWNER_PID_LISTENER = 3
ERROR_INSUFFICIENT_BUFFER = 122
_BUFFER_START = 16 * 1024
_BUFFER_MAX = 4 * 1024 * 1024
_BUFFER_TRIES = 4

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class MIB_TCPROW_OWNER_PID(ctypes.Structure):
    _fields_ = [
        ("dwState", wintypes.DWORD),
        ("dwLocalAddr", wintypes.DWORD),
        ("dwLocalPort", wintypes.DWORD),
        ("dwRemoteAddr", wintypes.DWORD),
        ("dwRemotePort", wintypes.DWORD),
        ("dwOwningPid", wintypes.DWORD),
    ]


class MIB_TCP6ROW_OWNER_PID(ctypes.Structure):
    _fields_ = [
        ("ucLocalAddr", ctypes.c_ubyte * 16),
        ("dwLocalScopeId", wintypes.DWORD),
        ("dwLocalPort", wintypes.DWORD),
        ("ucRemoteAddr", ctypes.c_ubyte * 16),
        ("dwRemoteScopeId", wintypes.DWORD),
        ("dwRemotePort", wintypes.DWORD),
        ("dwState", wintypes.DWORD),
        ("dwOwningPid", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _iphlpapi():
    library = ctypes.WinDLL("iphlpapi", use_last_error=True)
    library.GetExtendedTcpTable.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.ULONG),
        wintypes.BOOL,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.ULONG,
    ]
    library.GetExtendedTcpTable.restype = wintypes.DWORD
    return library


def _kernel32():
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    library.CreateJobObjectW.restype = wintypes.HANDLE
    library.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    library.SetInformationJobObject.restype = wintypes.BOOL
    library.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    library.AssignProcessToJobObject.restype = wintypes.BOOL
    library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    library.OpenProcess.restype = wintypes.HANDLE
    library.CloseHandle.argtypes = [wintypes.HANDLE]
    library.CloseHandle.restype = wintypes.BOOL
    return library


def _listener_rows(address_family: int, row_type: type) -> Iterable[object]:
    """读一张监听表，返回记录对象；系统调用失败时返回空列表。

    表结构是「一个 DWORD 记录数 + 紧凑排列的记录数组」，长度随进程数变化，
    所以先用一个猜测的缓冲区，装不下再按系统回写的长度重试。
    """
    table = _iphlpapi()
    size = wintypes.ULONG(_BUFFER_START)
    payload = b""
    for _attempt in range(_BUFFER_TRIES):
        buffer = ctypes.create_string_buffer(max(size.value, _BUFFER_START))
        result = table.GetExtendedTcpTable(
            buffer,
            ctypes.byref(size),
            False,
            address_family,
            TCP_TABLE_OWNER_PID_LISTENER,
            0,
        )
        if result == 0:
            payload = buffer.raw[: size.value]
            break
        if result != ERROR_INSUFFICIENT_BUFFER or size.value > _BUFFER_MAX:
            return []
    else:
        return []

    if len(payload) < ctypes.sizeof(wintypes.DWORD):
        return []
    count = int.from_bytes(payload[:4], "little")
    offset = ctypes.sizeof(wintypes.DWORD)
    array = row_type * count
    if count <= 0 or len(payload) < offset + ctypes.sizeof(array):
        return []
    return list(array.from_buffer_copy(payload[offset:]))


def find_listening_pid(port: int) -> Optional[int]:
    """返回监听该 TCP 端口的进程 pid；端口没被监听时返回 None。

    IPv4 与 IPv6 的监听表都查：服务监听 127.0.0.1 时落在 IPv4 表，监听 ::/::1
    时落在 IPv6 表。端口号在表里是网络字节序，读出来要转回主机序。
    """
    wanted = int(port)
    for address_family, row_type in (
        (AF_INET, MIB_TCPROW_OWNER_PID),
        (AF_INET6, MIB_TCP6ROW_OWNER_PID),
    ):
        for row in _listener_rows(address_family, row_type):
            if socket.ntohs(int(row.dwLocalPort) & 0xFFFF) != wanted:
                continue
            pid = int(row.dwOwningPid)
            if pid > 0:
                return pid
    return None


def kill_process_tree(pid: int) -> bool:
    """结束进程及其全部子孙；taskkill 不可用或结束失败时返回 False。"""
    pid = int(pid)
    if pid <= 0:
        return False
    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def create_kill_on_close_job() -> Optional[int]:
    """创建 kill-on-close 作业对象；返回句柄，失败返回 None。"""
    kernel32 = _kernel32()
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        handle,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    )
    if not ok:
        kernel32.CloseHandle(handle)
        return None
    return int(handle)


def assign_process_to_job(job_handle: int, pid: int) -> bool:
    """把进程挂进作业对象；进程已退出或没有权限时返回 False。"""
    if not job_handle or int(pid) <= 0:
        return False
    kernel32 = _kernel32()
    process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(pid))
    if not process:
        return False
    try:
        return bool(kernel32.AssignProcessToJobObject(int(job_handle), process))
    finally:
        kernel32.CloseHandle(process)


def close_job(job_handle: Optional[int]) -> None:
    """关闭作业句柄；最后一个句柄关闭时内核结束作业里的全部进程。"""
    if not job_handle:
        return
    try:
        _kernel32().CloseHandle(int(job_handle))
    except OSError:
        pass


__all__ = [
    "JOBOBJECT_EXTENDED_LIMIT_INFORMATION",
    "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
    "close_job",
    "assign_process_to_job",
    "create_kill_on_close_job",
    "find_listening_pid",
    "kill_process_tree",
]
