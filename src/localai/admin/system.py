"""系统信息、开机自启、日志与重启。

所有 Windows 操作都走 ``powershell -NoProfile -NonInteractive``，不弹 UAC、
不弹窗口；失败时把 PowerShell 的 stderr 原样带回接口，方便排错。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .. import API_VERSION, __version__
from ..config import Config

logger = logging.getLogger("localai.admin.system")

TASK_NAME = "LocalAIService-Service"

#: 允许查看的日志文件（键 -> 文件名），白名单避免路径穿越。
LOG_FILES = {
    "service": "service.out.log",
    "out": "service.out.log",
    "err": "service.err.log",
    "service.err": "service.err.log",
}

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
# 注意：不要用 DETACHED_PROCESS 启动 powershell.exe。实测在 Windows 上
# PowerShell 会直接退出而不执行 -Command / -File，导致「重启按钮没反应」。
# 用 CREATE_NO_WINDOW + CREATE_NEW_PROCESS_GROUP 已经足够让脚本独立运行，
# 父进程（服务）被结束后子进程仍然继续。


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def service_root(cfg: Config) -> Path:
    """启动脚本所在目录：优先包位置推导，其次 LOCALAI_HOME。"""
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "start_ai_service.ps1").is_file():
        return candidate
    return Path(cfg.home)


def _dir_size_mb(directory: Path) -> float:
    total = 0
    try:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return round(total / (1024 * 1024), 1)


def _run_powershell(script: str, timeout: float = 30.0) -> tuple[int, str, str]:
    """执行一段 PowerShell，返回 (returncode, stdout, stderr)。"""
    if os.name != "nt":
        return 1, "", "当前系统不是 Windows，不支持该操作"
    command = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
        "$ProgressPreference='SilentlyContinue';" + script
    )
    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_CREATE_NO_WINDOW,
        )
    except FileNotFoundError:
        return 1, "", "找不到 powershell.exe"
    except subprocess.TimeoutExpired:
        return 1, "", f"PowerShell 执行超时（{timeout:.0f} 秒）"
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


def _ps_quote(value: str) -> str:
    """PowerShell 单引号字符串转义。"""
    return "'" + str(value).replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# 计划任务（开机自启）
# ---------------------------------------------------------------------------
_STATUS_PS = (
    f"$name = {_ps_quote(TASK_NAME)};"
    "$t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue;"
    "if ($null -eq $t) { '{\"exists\":false}' } else {"
    "$i = Get-ScheduledTaskInfo -TaskName $name -ErrorAction SilentlyContinue;"
    "$a = $null; if ($t.Actions.Count -gt 0) { $a = $t.Actions[0] };"
    "[pscustomobject]@{"
    "exists=$true;"
    "name=[string]$t.TaskName;"
    "state=[string]$t.State;"
    "enabled=([string]$t.State -ne 'Disabled');"
    "execute=if ($a) {[string]$a.Execute} else {''};"
    "arguments=if ($a) {[string]$a.Arguments} else {''};"
    "working_directory=if ($a) {[string]$a.WorkingDirectory} else {''};"
    "last_run_time=if ($i -and $i.LastRunTime) {[string]$i.LastRunTime} else {''};"
    "last_task_result=if ($i) {[int]$i.LastTaskResult} else {0};"
    "} | ConvertTo-Json -Compress }"
)


_SCHED_CACHE: dict[str, Any] = {"at": 0.0, "info": None}
_SCHED_CACHE_SECONDS = 30.0


def scheduled_task_status(*, force: bool = False) -> dict[str, Any]:
    """读取 ``LocalAIService-Service`` 的状态；Windows 之外返回 supported=false。

    PowerShell 查询一次要 1 秒多，系统页每刷一次都查太慢，所以结果缓存 30 秒；
    启用 / 禁用后调用方传 ``force=True`` 立即重查。
    """
    now = time.monotonic()
    cached = _SCHED_CACHE.get("info")
    if (
        not force
        and isinstance(cached, dict)
        and now - float(_SCHED_CACHE.get("at") or 0.0) < _SCHED_CACHE_SECONDS
    ):
        return dict(cached)
    info = _scheduled_task_status_uncached()
    _SCHED_CACHE["at"] = now
    _SCHED_CACHE["info"] = dict(info)
    return info


def _scheduled_task_status_uncached() -> dict[str, Any]:
    """真正去问 Windows 计划任务；结果由 ``scheduled_task_status`` 缓存。"""
    if os.name != "nt":
        return {
            "supported": False,
            "exists": False,
            "name": TASK_NAME,
            "state": "unsupported",
            "enabled": False,
            "detail": "当前系统不是 Windows，开机自启使用系统自带服务管理",
        }
    code, out, err = _run_powershell(_STATUS_PS)
    if code != 0 and not out:
        return {
            "supported": True,
            "exists": False,
            "name": TASK_NAME,
            "state": "unknown",
            "enabled": False,
            "error": err or f"PowerShell 退出码 {code}",
        }
    try:
        data = json.loads(out.splitlines()[-1] if out else "{}")
    except ValueError:
        return {
            "supported": True,
            "exists": False,
            "name": TASK_NAME,
            "state": "unknown",
            "enabled": False,
            "error": f"无法解析计划任务状态：{out or err}",
        }
    if not data.get("exists"):
        return {
            "supported": True,
            "exists": False,
            "name": TASK_NAME,
            "state": "missing",
            "enabled": False,
            "detail": "计划任务不存在，可在此创建",
        }
    return {
        "supported": True,
        "exists": True,
        "name": data.get("name") or TASK_NAME,
        "state": data.get("state") or "unknown",
        "enabled": bool(data.get("enabled")),
        "execute": data.get("execute") or "",
        "arguments": data.get("arguments") or "",
        "working_directory": data.get("working_directory") or "",
        "last_run_time": data.get("last_run_time") or "",
        "last_task_result": data.get("last_task_result"),
        "detail": "已启用" if data.get("enabled") else "已禁用",
    }


def set_autostart(cfg: Config, enabled: bool) -> dict[str, Any]:
    """启用 / 禁用计划任务；不存在且 enabled=true 时创建。

    创建任务不带 ``-RunLevel Highest``，避免触发 UAC；当前用户登录时启动即可。
    """
    if os.name != "nt":
        raise NotImplementedError("开机自启管理只支持 Windows")
    root = service_root(cfg)
    script = root / "start_ai_service.ps1"
    if enabled and not script.is_file():
        raise FileNotFoundError(f"找不到启动脚本：{script}")

    name = _ps_quote(TASK_NAME)
    if enabled:
        ps = (
            f"$name = {name};"
            "$t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue;"
            "if ($null -ne $t) {"
            "Enable-ScheduledTask -TaskName $name | Out-Null;"
            "} else {"
            f"$script = {_ps_quote(str(script))};"
            f"$wd = {_ps_quote(str(root))};"
            "$arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File \"' + $script + '\"';"
            "$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $wd;"
            "$trigger = New-ScheduledTaskTrigger -AtLogOn;"
            "Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger "
            "-Description '本地 AI 服务（TTS/ASR/翻译/声纹/人声分离）登录时自动启动' -Force | Out-Null;"
            "};"
            "'{\"ok\":true}'"
        )
    else:
        ps = (
            f"$name = {name};"
            "$t = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue;"
            "if ($null -ne $t) { Disable-ScheduledTask -TaskName $name | Out-Null };"
            "'{\"ok\":true}'"
        )

    code, out, err = _run_powershell(ps, timeout=45.0)
    if code != 0:
        raise RuntimeError(
            "计划任务操作失败："
            + (err or out or f"PowerShell 退出码 {code}")
            + "。如果提示权限不足，请用管理员 PowerShell 手动执行，或检查任务名是否被组策略锁定。"
        )
    status = scheduled_task_status(force=True)
    status["ok"] = True
    status["requested_enabled"] = bool(enabled)
    return status


# ---------------------------------------------------------------------------
# 系统信息
# ---------------------------------------------------------------------------
_GPU_CACHE: dict[str, Any] = {"at": 0.0, "info": None}
_GPU_CACHE_SECONDS = 3.0


def _gpu_info(vram: Any) -> dict[str, Any]:
    # nvidia-smi 在这台机器上要 1.5 秒左右，状态页和插件的指标胶囊都在轮询，
    # 不加缓存会把接口拖慢到插件超时（显示显存 0）。GPU 信息 3 秒缓存足够新。
    now = time.monotonic()
    cached = _GPU_CACHE.get("info")
    if cached is not None and now - float(_GPU_CACHE.get("at") or 0.0) < _GPU_CACHE_SECONDS:
        return dict(cached)

    info = _read_gpu_info(vram)
    _GPU_CACHE["at"] = now
    _GPU_CACHE["info"] = dict(info)
    return info


def _read_gpu_info(vram: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": None,
        "driver": None,
        "free_mb": None,
        "total_mb": None,
        "used_mb": None,
    }
    try:
        free_mb, total_mb = vram.health_memory()
        info["free_mb"] = free_mb
        info["total_mb"] = total_mb
        if total_mb:
            info["used_mb"] = max(0, int(total_mb) - int(free_mb))
    except Exception:  # noqa: BLE001 - 没有 GPU 也要能看系统页
        pass

    import shutil

    smi = shutil.which("nvidia-smi")
    if not smi:
        return info
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=name,driver_version,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            creationflags=_CREATE_NO_WINDOW,
        )
        line = next((row for row in proc.stdout.splitlines() if row.strip()), "")
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            info["name"] = parts[0]
            info["driver"] = parts[1]
            info["free_mb"] = int(float(parts[2]))
            info["total_mb"] = int(float(parts[3]))
            info["used_mb"] = max(0, info["total_mb"] - info["free_mb"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-smi 查询失败：%s", exc)
    return info


def system_info(
    cfg: Config,
    *,
    vram: Any,
    registry: Any,
    started_at: Optional[str],
    jobs_stats: Optional[dict[str, Any]] = None,
    pid: Optional[int] = None,
) -> dict[str, Any]:
    """``GET /admin/system`` 的完整响应。"""
    host = str(cfg.host)
    loopback = host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")
    uptime = None
    if started_at:
        try:
            started = datetime.fromisoformat(started_at)
            uptime = round(max(0.0, (datetime.now().astimezone() - started).total_seconds()), 1)
        except (TypeError, ValueError):
            pass

    models_dir = cfg.home / "models"
    disk = {
        "models_mb": _dir_size_mb(models_dir),
        "voices_mb": _dir_size_mb(cfg.voices_dir),
        "outputs_mb": _dir_size_mb(cfg.output_dir),
        "logs_mb": _dir_size_mb(cfg.logs_dir),
        "data_mb": _dir_size_mb(cfg.data_dir),
    }
    disk["total_mb"] = round(sum(disk.values()), 1)

    loaded = []
    try:
        for adapter in registry.all():
            if adapter.loaded():
                loaded.append(adapter.name)
    except Exception:  # noqa: BLE001
        pass

    # 下载引擎状态（yt-dlp 版本 / 限额）；读取失败不影响系统页。
    try:
        from ..engines._native import import_engine_module

        dl_status = import_engine_module("download_engine").status_info()
        download_block = {
            "yt_dlp_version": (dl_status.get("yt_dlp") or {}).get("version"),
            "yt_dlp_path": (dl_status.get("yt_dlp") or {}).get("path"),
            "aria2c": (dl_status.get("aria2c") or {}).get("available"),
            "download_dir": str(cfg.downloads_dir),
            "max_mb": cfg.download_max_mb,
            "sync_max_mb": cfg.download_sync_max_mb,
            "timeout_seconds": cfg.download_timeout,
        }
    except Exception as exc:  # noqa: BLE001
        download_block = {"yt_dlp_version": None, "error": f"{type(exc).__name__}: {exc}"}

    # 视觉模型状态（模型页 / 系统页都能看到版本与加载状态）。
    try:
        vision_status = registry.get("vision").status()
        vision_block = {
            "model": vision_status.get("model"),
            "runtime": vision_status.get("runtime"),
            "runtime_version": vision_status.get("runtime_version"),
            "ready": bool(vision_status.get("ready")),
            "loaded": bool(vision_status.get("loaded")),
            "port": vision_status.get("port"),
            "model_size_mb": vision_status.get("model_size_mb"),
            "mmproj_size_mb": vision_status.get("mmproj_size_mb"),
            "planner": cfg.vision_planner,
            "idle_unload_seconds": cfg.vision_idle_unload,
        }
    except Exception as exc:  # noqa: BLE001 - 系统页不能因此挂掉
        vision_block = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "service": {
            "name": "localai-service",
            "version": __version__,
            "api_version": API_VERSION,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pid": pid or os.getpid(),
            "started_at": started_at,
            "uptime_seconds": uptime,
        },
        "listen": {
            "host": host,
            "port": cfg.port,
            "base_url": cfg.base_url,
            "loopback": loopback,
            "warning": None
            if loopback
            else "服务监听在非回环地址，局域网内其它机器可能访问管理接口；请务必设置 LOCALAI_TOKEN。",
        },
        "auth_required": bool(cfg.token),
        "gpu": _gpu_info(vram),
        "disk": disk,
        "models_loaded": loaded,
        "vision": vision_block,
        "download": download_block,
        "jobs": jobs_stats or {},
        "autostart": scheduled_task_status(),
        "paths": {
            "home": str(cfg.home),
            "models": str(models_dir),
            "voices": str(cfg.voices_dir),
            "outputs": str(cfg.output_dir),
            "logs": str(cfg.logs_dir),
            "config_file": str(cfg.home / "config.json"),
        },
    }


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def _looks_like_utf16(data: bytes, zero_side: int, sample: int = 4096) -> bool:
    """无 BOM 时判断字节流是否像 UTF-16（``zero_side`` 为 0 的字节位置）。

    UTF-16 的 ASCII / 中文文本在 ``zero_side`` 那一侧有大量 0 字节；日志空洞里
    的 NUL 是整片连续的，两侧都会是 0，所以这里要求「一侧几乎全 0、另一侧几乎
    不 0」，普通 UTF-8 文本和空洞都不满足。
    """
    chunk = data[:sample]
    if len(chunk) < 8:
        return False
    side_total = side_zeros = other_total = other_zeros = 0
    for index, byte in enumerate(chunk):
        if index % 2 == zero_side:
            side_total += 1
            if byte == 0:
                side_zeros += 1
        else:
            other_total += 1
            if byte == 0:
                other_zeros += 1
    if side_total == 0 or other_total == 0:
        return False
    return side_zeros >= side_total * 0.6 and other_zeros <= other_total * 0.2


def _decode_log(data: bytes) -> tuple[str, str]:
    """自动探测日志编码，返回 (文本, 编码名)。

    启动脚本用 Windows PowerShell 的 ``Out-File -Encoding UTF8`` 写带 BOM 的
    UTF-8；Python 自己的日志是 UTF-8 无 BOM；历史文件里也出现过 UTF-16。
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace"), "utf-8-sig"
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16-le", errors="replace"), "utf-16-le"
    if data.startswith(b"\xfe\xff"):
        return data.decode("utf-16-be", errors="replace"), "utf-16-be"
    # 无 BOM 的 UTF-16 判定不能只看 \x00 占比：日志被截断或稀疏写会留下
    # 成片的 NUL 空洞（尾部窗口正好落进空洞里时整段都会被误判成 UTF-16，
    # 解出来全是 \x00 乱码）。真正的 UTF-16 ASCII 文本里 0 字节是隔字节出现的，
    # 所以先看「每两个字节里固定一侧为 0」这个结构，再决定是否按 UTF-16 解。
    if data and data.count(b"\x00") * 4 > len(data):
        # 偶数位是 0 → Big-Endian；奇数位是 0 → Little-Endian（ASCII 区特征）。
        if _looks_like_utf16(data, 0):
            return data.decode("utf-16-be", errors="replace"), "utf-16-be"
        if _looks_like_utf16(data, 1):
            return data.decode("utf-16-le", errors="replace"), "utf-16-le"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # 尾部窗口可能刚好切在一个多字节字符中间，丢掉开头 1~3 个字节再试。
        for skip in range(1, 4):
            if len(data) <= skip:
                break
            try:
                text = data[skip:].decode("utf-8")
            except UnicodeDecodeError:
                continue
            return text.replace("\x00", ""), "utf-8"
    else:
        # 空洞留在文本里会让前端显示成一串空白/方块，直接丢掉 NUL。
        return text.replace("\x00", ""), "utf-8"
    try:
        return data.decode("gbk"), "gbk"
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace"), "latin-1"


def read_log_tail(cfg: Config, file_key: str = "service", lines: int = 200) -> dict[str, Any]:
    """读取 ``logs/`` 下某个日志的尾部若干行。"""
    key = str(file_key or "service").strip().lower()
    name = LOG_FILES.get(key)
    if not name:
        raise ValueError(
            f"不支持的日志文件 {file_key!r}；可选：{', '.join(sorted(LOG_FILES))}"
        )
    path = Path(cfg.logs_dir) / name
    if not path.is_file():
        return {
            "ok": True,
            "file": key,
            "path": str(path),
            "exists": False,
            "encoding": None,
            "lines": [],
            "text": "",
            "detail": "日志文件还不存在",
        }

    # 只读尾部，避免把几百 MB 的日志整个读进内存。
    max_lines = max(1, min(5000, int(lines)))
    want_bytes = max(64 * 1024, min(4 * 1024 * 1024, max_lines * 800))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > want_bytes:
                handle.seek(size - want_bytes)
            data = handle.read()
    except OSError as exc:
        raise RuntimeError(f"读取日志失败：{path}（{exc}）") from exc

    text, encoding = _decode_log(data)
    all_lines = text.splitlines()
    tail = all_lines[-max_lines:]
    return {
        "ok": True,
        "file": key,
        "filename": name,
        "path": str(path),
        "exists": True,
        "encoding": encoding,
        "size_bytes": size,
        "total_lines_read": len(all_lines),
        "lines": tail,
        "text": "\n".join(tail),
    }


# ---------------------------------------------------------------------------
# 重启
# ---------------------------------------------------------------------------
def schedule_restart(app: Any, cfg: Config, *, delay: float = 1.5) -> dict[str, Any]:
    """安排一次异步重启：延迟后用脱离进程执行 ``tools/restart_service.ps1``。

    必须在返回 HTTP 响应之后才真正杀进程，否则调用方只会看到连接被重置。
    """
    if os.name != "nt":
        raise NotImplementedError("自动重启目前只支持 Windows")

    root = service_root(cfg)
    helper = root / "tools" / "restart_service.ps1"
    if not helper.is_file():
        raise FileNotFoundError(f"找不到重启脚本：{helper}")

    async def _later() -> None:
        await asyncio.sleep(delay)
        try:
            subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-WindowStyle",
                    "Hidden",
                    "-File",
                    str(helper),
                    "-Port",
                    str(cfg.port),
                    "-StartScript",
                    str(root / "start_ai_service.ps1"),
                ],
                cwd=str(root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW | _NEW_GROUP,
                close_fds=True,
            )
            logger.warning("已安排重启：端口=%s 启动脚本=%s", cfg.port, root / "start_ai_service.ps1")
        except Exception:  # noqa: BLE001
            logger.exception("启动重启脚本失败")

    task = asyncio.create_task(_later())
    app.state.restart_task = task
    return {
        "ok": True,
        "detail": f"服务将在约 {delay:.0f} 秒后重启（先释放端口 {cfg.port}，再执行 start_ai_service.ps1）",
        "port": cfg.port,
        "script": str(root / "start_ai_service.ps1"),
    }


__all__ = [
    "LOG_FILES",
    "TASK_NAME",
    "read_log_tail",
    "schedule_restart",
    "scheduled_task_status",
    "service_root",
    "set_autostart",
    "system_info",
]
