"""Windows 通知区图标（零依赖，ctypes 直调 Win32）。

只做 UI：建隐藏窗口、Shell_NotifyIcon、右键菜单、双击与悬浮提示。
动作本身在 actions 层；这里不探测端口、不拉进程。
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections import deque
from ctypes import wintypes
from pathlib import Path
from typing import Optional

from .actions import (
    ACTION_BY_MENU_ID,
    ACTION_CLEAR_VRAM,
    ACTION_OPEN,
    ACTION_QUIT,
    NOTIFY_ERROR,
    MENU_IDS,
    MENU_ITEMS,
    MODE_ATTACHED,
    MODE_CONFLICT,
    MODE_OWNED,
    STATE_CONFLICT,
    STATE_RUNNING,
    build_tooltip,
)

log = logging.getLogger("localai.tray")

CLASS_NAME = "LocalAIServiceTrayWindow"
WINDOW_TITLE = "LocalAIService"

WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
WM_TIMER = 0x0113
WM_NULL = 0x0000
WM_LBUTTONDBLCLK = 0x0203
WM_RBUTTONUP = 0x0205
WM_CONTEXTMENU = 0x007B
WM_APP = 0x8000
WM_TRAY = WM_APP + 1
WM_TRAY_READY = WM_APP + 2
WM_TRAY_EXIT = WM_APP + 3
WM_TRAY_FAILED = WM_APP + 4
WM_TRAY_NOTIFY = WM_APP + 5

CS_DBLCLKS = 0x0008
NIM_ADD = 0
NIM_MODIFY = 1
NIM_DELETE = 2
NIF_MESSAGE = 0x01
NIF_ICON = 0x02
NIF_TIP = 0x04
NIF_INFO = 0x10
NIIF_INFO = 0x01
NIIF_WARNING = 0x02
NIIF_ERROR = 0x03
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_DEFAULTSIZE = 0x00000040
MF_STRING = 0x00000000
MF_SEPARATOR = 0x00000800
TPM_LEFTALIGN = 0x0000
TPM_RIGHTBUTTON = 0x0002
IDC_ARROW = 32512
IDI_APPLICATION = 32512
GMEM_MOVEABLE = 0x0002
CF_UNICODETEXT = 13

TIMER_ID = 1
POLL_MS = 2000
READY_TIMEOUT_SECONDS = 90.0


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT),
        ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON),
        ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD),
        ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256),
        ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64),
        ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", GUID),
        ("hBalloonIcon", wintypes.HICON),
    ]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_shell32 = ctypes.WinDLL("shell32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
_user32.RegisterClassW.restype = wintypes.ATOM
_user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p,
]
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.DefWindowProcW.restype = ctypes.c_ssize_t
_user32.DestroyWindow.argtypes = [wintypes.HWND]
_user32.DestroyWindow.restype = wintypes.BOOL
_user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_user32.GetMessageW.restype = wintypes.BOOL
_user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
_user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
_user32.DispatchMessageW.restype = ctypes.c_ssize_t
_user32.PostQuitMessage.argtypes = [ctypes.c_int]
_user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT]
_user32.LoadImageW.restype = wintypes.HANDLE
_user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
_user32.LoadIconW.restype = wintypes.HICON
_user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
_user32.LoadCursorW.restype = wintypes.HANDLE
_user32.DestroyIcon.argtypes = [wintypes.HICON]
_user32.CreatePopupMenu.restype = wintypes.HMENU
_user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_size_t, wintypes.LPCWSTR]
_user32.SetMenuDefaultItem.argtypes = [wintypes.HMENU, wintypes.UINT, wintypes.UINT]
_user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, ctypes.c_void_p]
_user32.DestroyMenu.argtypes = [wintypes.HMENU]
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
_user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
_user32.SetTimer.restype = ctypes.c_size_t
_user32.KillTimer.argtypes = [wintypes.HWND, ctypes.c_size_t]
_user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.FindWindowW.restype = wintypes.HWND
_user32.MessageBoxW.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
_user32.MessageBoxW.restype = ctypes.c_int
_user32.OpenClipboard.argtypes = [wintypes.HWND]
_user32.EmptyClipboard.restype = wintypes.BOOL
_user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
_user32.SetClipboardData.restype = wintypes.HANDLE
_user32.CloseClipboard.restype = wintypes.BOOL
_shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
_shell32.Shell_NotifyIconW.restype = wintypes.BOOL
_kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
_kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
_kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalLock.restype = ctypes.c_void_p
_kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
_kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE


def _make_int_resource(value: int) -> ctypes.c_void_p:
    return ctypes.cast(ctypes.c_void_p(value), wintypes.LPCWSTR)


def show_error(title: str, text: str) -> None:
    """托盘无法启动时的最后提示（pythonw 没有控制台）。"""
    _user32.MessageBoxW(None, text, title, 0x00000010)


def set_clipboard_text(text: str) -> bool:
    """把文本放进剪贴板；剪贴板被别的程序占用时短暂重试。"""
    for _ in range(5):
        if _user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        log.warning("剪贴板打开失败，地址未复制")
        return False
    try:
        _user32.EmptyClipboard()
        size = (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
        handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        pointer = _kernel32.GlobalLock(handle)
        if not pointer:
            _kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, ctypes.create_unicode_buffer(text), size)
        _kernel32.GlobalUnlock(handle)
        if not _user32.SetClipboardData(CF_UNICODETEXT, handle):
            _kernel32.GlobalFree(handle)
            return False
        return True
    finally:
        _user32.CloseClipboard()


def find_tray_window() -> int:
    return int(_user32.FindWindowW(CLASS_NAME, None) or 0)


def request_open_from_existing() -> bool:
    """第二个实例不重复建图标，叫醒已在运行的托盘打开页面。"""
    hwnd = find_tray_window()
    if not hwnd:
        return False
    return bool(_user32.PostMessageW(hwnd, WM_COMMAND, MENU_IDS[ACTION_OPEN], 0))


class TrayWindow:
    def __init__(self, actions, controller, icon_path: Path, *, poll_ms: int = POLL_MS) -> None:
        self._actions = actions
        self._controller = controller
        self._icon_path = Path(icon_path)
        self._poll_ms = int(poll_ms)
        self._hwnd = 0
        self._hicon = 0
        self._wndproc = WNDPROC(self._on_message)
        self._mode = "starting" if controller.mode == MODE_OWNED else controller.mode
        self._destroyed = False
        self._quitting = False
        self._ballooned: set[str] = set()
        self._notices: deque[tuple[str, str, int]] = deque()
        self._ready_thread: Optional[threading.Thread] = None
        self._clear_thread: Optional[threading.Thread] = None
        self._stop_ready = threading.Event()

    # -- 生命周期 -----------------------------------------------------
    def run(self) -> int:
        self._create_window()
        self._load_icon()
        if not self._add_icon():
            # 图标没建起来时窗口循环不会启动，必须让调用方看到失败（pythonw 无控制台）。
            self._destroy()
            raise RuntimeError("Shell_NotifyIcon(NIM_ADD) 失败：通知区图标未创建")
        self._update_tooltip(force=True)
        self._announce_initial()
        if self._controller.mode == MODE_OWNED:
            self._start_ready_watch()
        _user32.SetTimer(self._hwnd, TIMER_ID, self._poll_ms, None)
        log.info(
            "托盘已就绪：窗口=%s 端口=%s 模式=%s",
            self._hwnd,
            self._controller.port,
            self._controller.mode,
        )
        return self._message_loop()

    def _message_loop(self) -> int:
        message = MSG()
        while True:
            result = _user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if result == 0 or result == -1:
                break
            _user32.TranslateMessage(ctypes.byref(message))
            _user32.DispatchMessageW(ctypes.byref(message))
        return 0

    def _create_window(self) -> None:
        instance = _kernel32.GetModuleHandleW(None)
        window_class = WNDCLASSW()
        window_class.style = CS_DBLCLKS
        window_class.lpfnWndProc = ctypes.cast(self._wndproc, ctypes.c_void_p)
        window_class.hInstance = instance
        window_class.hCursor = _user32.LoadCursorW(None, _make_int_resource(IDC_ARROW))
        window_class.lpszClassName = CLASS_NAME
        atom = _user32.RegisterClassW(ctypes.byref(window_class))
        if not atom:
            error = ctypes.get_last_error()
            if error not in (0, 1410):
                log.warning("注册托盘窗口类失败：WinError %s", error)
        self._hwnd = int(
            _user32.CreateWindowExW(
                0, CLASS_NAME, WINDOW_TITLE, 0,
                0, 0, 0, 0, None, None, instance, None,
            )
            or 0
        )
        if not self._hwnd:
            raise OSError(f"创建托盘窗口失败：WinError {ctypes.get_last_error()}")

    def _load_icon(self) -> None:
        if self._icon_path.is_file():
            self._hicon = int(
                _user32.LoadImageW(
                    None, str(self._icon_path), IMAGE_ICON, 0, 0,
                    LR_LOADFROMFILE | LR_DEFAULTSIZE,
                )
                or 0
            )
        if not self._hicon:
            log.warning("托盘图标文件不可用，退回系统图标：%s", self._icon_path)
            self._hicon = int(_user32.LoadIconW(None, _make_int_resource(IDI_APPLICATION)) or 0)

    def _notify_data(self, flags: int) -> NOTIFYICONDATAW:
        data = NOTIFYICONDATAW()
        data.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        data.hWnd = self._hwnd
        data.uID = 1
        data.uFlags = flags
        return data

    def _add_icon(self) -> bool:
        data = self._notify_data(NIF_MESSAGE | NIF_ICON | NIF_TIP)
        data.uCallbackMessage = WM_TRAY
        data.hIcon = self._hicon
        data.szTip = build_tooltip(self._controller.port, self._mode)
        if not _shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(data)):
            log.error("Shell_NotifyIcon(NIM_ADD) 失败，通知区图标未创建")
            return False
        return True

    def _destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self._stop_ready.set()
        if self._hwnd:
            _user32.KillTimer(self._hwnd, TIMER_ID)
            data = self._notify_data(0)
            _shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(data))
            _user32.DestroyWindow(self._hwnd)
        if self._hicon:
            _user32.DestroyIcon(self._hicon)
            self._hicon = 0
        _user32.PostQuitMessage(0)

    # -- 状态 ---------------------------------------------------------
    def _announce_initial(self) -> None:
        """图标刚注册时弹一次气球，只报启动结果。"""
        mode = self._controller.mode
        if mode == MODE_OWNED:
            self._balloon(
                "started",
                "本地 AI 服务已启动",
                f"127.0.0.1:{self._controller.port}",
                NIIF_INFO,
            )
        elif mode == MODE_ATTACHED:
            self._balloon(
                "attached",
                "服务已在运行",
                "托盘只负责打开页面；退出托盘不会关闭它。",
                NIIF_INFO,
            )
        elif mode == MODE_CONFLICT:
            self._balloon(
                "conflict",
                "端口被占用",
                f"127.0.0.1:{self._controller.port} 被其它程序占用，没有启动新服务。",
                NIIF_WARNING,
            )

    def _start_ready_watch(self) -> None:
        self._ready_thread = threading.Thread(
            target=self._wait_until_ready, name="localai-tray-ready", daemon=True
        )
        self._ready_thread.start()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while not self._stop_ready.is_set() and time.monotonic() < deadline:
            if self._controller.process_exited():
                self._post(WM_TRAY_FAILED)
                return
            state = self._controller.health_state()
            if state == STATE_RUNNING:
                self._post(WM_TRAY_READY)
                return
            if state == STATE_CONFLICT:
                self._post(WM_TRAY_FAILED)
                return
            self._stop_ready.wait(0.5)
        if not self._stop_ready.is_set():
            self._post(WM_TRAY_FAILED)

    def _post(self, message: int) -> None:
        if self._hwnd and not self._destroyed:
            _user32.PostMessageW(self._hwnd, message, 0, 0)

    def _tick(self) -> None:
        if self._controller.mode == MODE_OWNED and self._controller.process_exited():
            self._set_mode("failed")

    def _set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._update_tooltip(force=True)
        if mode == "failed":
            self._balloon(
                "failed",
                "服务启动失败",
                "查看 logs/service.err.log；可在托盘菜单退出后重试。",
                NIIF_ERROR,
            )

    def _update_tooltip(self, *, force: bool = False) -> None:
        if not self._hwnd:
            return
        data = self._notify_data(NIF_TIP)
        data.szTip = build_tooltip(self._controller.port, self._mode)
        _shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(data))
        if force:
            log.debug("tooltip -> %s", data.szTip)

    def _balloon(
        self, key: str, title: str, text: str, flag: int, *, once: bool = True
    ) -> None:
        """弹气球；once=False 用于每次都要汇报的动作结果（例如清理显存）。"""
        if not self._hwnd:
            return
        if once:
            if key in self._ballooned:
                return
            self._ballooned.add(key)
        data = self._notify_data(NIF_INFO)
        data.szInfoTitle = title
        data.szInfo = text
        data.dwInfoFlags = flag
        _shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(data))

    # -- 交互 ---------------------------------------------------------
    def _show_menu(self) -> None:
        menu = _user32.CreatePopupMenu()
        if not menu:
            return
        try:
            for action, label in MENU_ITEMS:
                if action == ACTION_QUIT:
                    _user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
                _user32.AppendMenuW(menu, MF_STRING, MENU_IDS[action], label)
            _user32.SetMenuDefaultItem(menu, MENU_IDS[ACTION_OPEN], 0)
            point = POINT()
            _user32.GetCursorPos(ctypes.byref(point))
            _user32.SetForegroundWindow(self._hwnd)
            _user32.TrackPopupMenu(
                menu,
                TPM_LEFTALIGN | TPM_RIGHTBUTTON,
                point.x,
                point.y,
                0,
                self._hwnd,
                None,
            )
            _user32.PostMessageW(self._hwnd, WM_NULL, 0, 0)
        finally:
            _user32.DestroyMenu(menu)

    def _handle(self, action: str) -> None:
        if self._quitting:
            return
        if action == ACTION_QUIT:
            self._begin_quit()
            return
        if action == ACTION_CLEAR_VRAM:
            self._start_clear_vram()
            return
        try:
            self._actions.perform(action)
        except Exception:
            log.exception("托盘动作失败：%s", action)

    # -- 异步结果 -----------------------------------------------------
    def notify_result(self, title: str, text: str, flag: int) -> None:
        """后台线程把结果交给窗口线程：队列 + PostMessage，不能直接碰 Shell_NotifyIcon。"""
        if self._destroyed:
            return
        self._notices.append((title, text, flag))
        self._post(WM_TRAY_NOTIFY)

    def _show_notice(self) -> None:
        try:
            title, text, flag = self._notices.popleft()
        except IndexError:
            return
        self._balloon("clear-vram", title, text, flag, once=False)

    def _start_clear_vram(self) -> None:
        """清理显存是网络请求，最多几秒；放后台线程，窗口线程继续响应菜单与消息。"""
        thread = threading.Thread(
            target=self._clear_vram_worker, name="localai-tray-vram", daemon=True
        )
        self._clear_thread = thread
        thread.start()

    def _clear_vram_worker(self) -> None:
        try:
            self._actions.perform(ACTION_CLEAR_VRAM)
        except Exception:
            log.exception("清理显存失败")
            self.notify_result("清理失败", "清理显存时出错，稍后再试。", NOTIFY_ERROR)

    def _begin_quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._stop_ready.set()
        data = self._notify_data(NIF_TIP)
        data.szTip = "LocalAIService · 正在退出…"
        _shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(data))
        threading.Thread(target=self._quit_worker, name="localai-tray-quit", daemon=True).start()

    def _quit_worker(self) -> None:
        try:
            self._actions.perform(ACTION_QUIT)
        except Exception:
            log.exception("退出托盘时出错")
        finally:
            self._post(WM_TRAY_EXIT)

    def _on_message(self, hwnd, message, wparam, lparam):
        try:
            if message == WM_TRAY:
                event = int(lparam) & 0xFFFF
                if event == WM_LBUTTONDBLCLK:
                    self._handle(ACTION_OPEN)
                elif event in (WM_RBUTTONUP, WM_CONTEXTMENU):
                    self._show_menu()
                return 0
            if message == WM_COMMAND:
                action = ACTION_BY_MENU_ID.get(int(wparam) & 0xFFFF)
                if action:
                    self._handle(action)
                return 0
            if message == WM_TIMER:
                self._tick()
                return 0
            if message == WM_TRAY_READY:
                self._set_mode("ready")
                return 0
            if message == WM_TRAY_FAILED:
                self._set_mode("failed")
                return 0
            if message == WM_TRAY_NOTIFY:
                self._show_notice()
                return 0
            if message == WM_TRAY_EXIT:
                self._destroy()
                return 0
            if message == WM_CLOSE:
                self._begin_quit()
                return 0
            if message == WM_DESTROY:
                return 0
        except Exception:
            log.exception("托盘窗口消息处理失败：0x%04X", int(message))
        return _user32.DefWindowProcW(hwnd, message, wparam, lparam)


__all__ = [
    "CLASS_NAME",
    "TrayWindow",
    "find_tray_window",
    "request_open_from_existing",
    "set_clipboard_text",
    "show_error",
]
