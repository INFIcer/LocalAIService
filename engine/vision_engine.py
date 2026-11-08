"""本地小视觉模型（VLM）引擎：Qwen2.5-VL-3B GGUF + llama.cpp。

设计取舍
--------
- **不把 torch/transformers 拉进常驻服务**：视觉能力用 llama.cpp 的
  ``llama-server.exe`` 跑一个 **只监听 127.0.0.1** 的独立进程，服务层用
  httpx 调它的 OpenAI 兼容接口。这样 8G 卡上 VLM 与 TTS 的显存冲突完全
  交给服务层已有的显存管家（LRU 互卸）解决，也不会把 transformers 的
  权重常驻在服务进程里。
- **按需启动**：服务启动时 0 模型；第一次视觉请求才拉起 llama-server。
- **可恢复**：进程崩溃/被杀后，下一次请求会自动重新拉起；启动前会清理
  上次遗留的 ``runtime.json``（只杀 PID 对应且进程名确实是 llama-server
  的进程，避免误伤）。
- **模型与二进制都在 ``LOCALAI_HOME/models/vision/``**，日志写 ``logs/``。

对外接口（服务层适配器只依赖这些）::

    model_status() / status()      只读文件与进程状态，不加载模型
    is_loaded()                    进程是否还活着
    ensure_loaded(timeout=...)     需要时拉起 llama-server 并等 /health
    describe(path, prompt, ...)    单图 -> 文本
    describe_text(prompt, ...)     纯文本（规划器）-> 文本
    describe_many(paths, ...)      多图批量（复用同一进程，逐张推理）
    unload()                       停掉本引擎拉起的进程

模型选择：Qwen2.5-VL-3B-Instruct **Q4_K_M**（1.93GB）+ mmproj-F16
（1.34GB）。中文 OCR 可用，显存约 3.2~3.5GB，实测单帧 1~3 秒。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("localai.vision.engine")

#: 与下载脚本/文档保持一致的固定版本（llama.cpp 的构建号）。
LLAMA_BUILD = "b12202"
#: llama.cpp Vulkan 构建：体积小（约 37MB）、不依赖 CUDA runtime，
#: 在 RTX 3070 上单帧 1~3 秒，满足「≤4 秒/帧」。
LLAMA_ASSET = f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip"
LLAMA_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{LLAMA_ASSET}"
)
#: GitHub 直连在部分网络会重置，按顺序尝试镜像；只影响下载，不影响运行。
LLAMA_MIRRORS = (
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
    "https://ghfast.top/",
)

#: ModelScope 优先，hf-mirror 兜底（与项目其它模型一致）。
MODEL_REPO = "unsloth/Qwen2.5-VL-3B-Instruct-GGUF"
MODEL_FILES = {
    "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf": (
        f"https://www.modelscope.cn/api/v1/models/{MODEL_REPO}/repo"
        "?Revision=master&FilePath=Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        f"https://hf-mirror.com/{MODEL_REPO}/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    ),
    "mmproj-F16.gguf": (
        f"https://www.modelscope.cn/api/v1/models/{MODEL_REPO}/repo"
        "?Revision=master&FilePath=mmproj-F16.gguf",
        f"https://hf-mirror.com/{MODEL_REPO}/resolve/main/mmproj-F16.gguf",
    ),
    # 视觉投影层：先下 F16，再用 llama-quantize 压到 Q8_0（显存从 3.9GB 降到 3.4GB）。

}

#: 默认显存估算（权重 3.27GB + 上下文/计算缓冲）。服务层显存管家会按实测校准。
ESTIMATE_MB = 3450

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_IS_WINDOWS = os.name == "nt"


class VisionError(RuntimeError):
    """视觉引擎不可用 / 推理失败（适配器会转成 503）。"""


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def _home() -> Path:
    raw = os.environ.get("LOCALAI_HOME") or os.environ.get("INDEX_TTS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return str(value).strip() if value is not None and str(value).strip() else default


def vision_dir() -> Path:
    raw = _env("LOCALAI_VISION_DIR")
    return Path(raw).expanduser() if raw else _home() / "models" / "vision"


def _logs_dir() -> Path:
    raw = _env("LOCALAI_LOGS_DIR")
    return Path(raw).expanduser() if raw else _home() / "logs"


def _server_exe() -> Path:
    return vision_dir() / _env("LOCALAI_VISION_SERVER", "llama-server.exe")


def _model_path() -> Path:
    return vision_dir() / _env(
        "LOCALAI_VISION_MODEL", "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    )


def _mmproj_path() -> Path:
    return vision_dir() / _env("LOCALAI_VISION_MMPROJ", "mmproj-Q8_0.gguf")


def _port() -> int:
    try:
        return max(1, min(65535, int(_env("LOCALAI_VISION_PORT", "7891"))))
    except ValueError:
        return 7891


def _ctx_size() -> int:
    try:
        return max(2048, min(32768, int(_env("LOCALAI_VISION_CTX", "4096"))))
    except ValueError:
        return 4096


def _gpu_layers() -> int:
    try:
        return max(0, min(999, int(_env("LOCALAI_VISION_GPU_LAYERS", "99"))))
    except ValueError:
        return 99


def _timeout() -> float:
    try:
        return max(10.0, float(_env("LOCALAI_VISION_TIMEOUT", "120")))
    except ValueError:
        return 120.0


def _runtime_file() -> Path:
    return vision_dir() / "runtime.json"


def _log_path() -> Path:
    return _logs_dir() / "vision_server.log"


def _version_file() -> Path:
    return vision_dir() / "BUILD.txt"


# ---------------------------------------------------------------------------
# 进程生命周期
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_process: Optional[subprocess.Popen[Any]] = None
_log_handle: Optional[Any] = None
_started_at: Optional[float] = None
_load_seconds: Optional[float] = None
_last_error: Optional[str] = None
_external = False  # 端口上已有别人拉起的 llama-server（重启遗留）
_last_call_ms: Optional[float] = None
_calls = 0
#: 最近一次加载的整卡显存增量（nvidia-smi used 差值，MB）。CUDA 看不到
#: Vulkan 子进程的占用，所以适配器用这个值回填显存管家的估算。
_vram_delta_mb: Optional[int] = None


def gpu_memory_used_mb() -> Optional[int]:
    """整卡已用显存（MB，nvidia-smi）；拿不到返回 None。"""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        first = next((line for line in (proc.stdout or "").splitlines() if line.strip()), "")
        return int(float(first.strip()))
    except Exception:  # noqa: BLE001
        return None


def vram_delta_mb() -> Optional[int]:
    """最近一次加载测到的显存增量（MB）。"""
    return _vram_delta_mb


def _port_open(port: Optional[int] = None) -> bool:
    """TCP 探活；不产生模型调用。"""
    probe = port or _port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.4)
    try:
        return sock.connect_ex(("127.0.0.1", probe)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _http_get(
    path: str, *, timeout: float = 2.0, port: Optional[int] = None
) -> tuple[int, str]:
    """极简 HTTP GET（只用于 /health、/props，避免额外依赖）。"""
    target = f"http://127.0.0.1:{port or _port()}{path}"
    try:
        with urllib.request.urlopen(target, timeout=timeout) as response:  # noqa: S310
            return int(response.status), response.read(8192).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        return int(exc.code), exc.read(4096).decode("utf-8", errors="replace") if exc.fp else ""
    except Exception as exc:  # noqa: BLE001 - 探活失败按不可用处理
        return 0, f"{type(exc).__name__}: {exc}"


def _healthy(port: Optional[int] = None) -> bool:
    """llama-server 的 /health：加载中返回 503，就绪返回 200。"""
    code, body = _http_get("/health", port=port)
    if code != 200:
        return False
    try:
        data = json.loads(body)
    except ValueError:
        return True
    status = str(data.get("status") or "").lower()
    return status in ("", "ok", "ready")


def _pid_is_llama_server(pid: int) -> bool:
    """确认 PID 现在确实是 llama-server.exe，避免 PID 复用误杀。"""
    if not _IS_WINDOWS:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        return "llama-server" in (proc.stdout or "").lower()
    except Exception:  # noqa: BLE001
        return False


def _kill_pid(pid: int) -> None:
    if not _pid_is_llama_server(pid):
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            capture_output=True,
            timeout=10,
            creationflags=_CREATE_NO_WINDOW,
        )
        logger.info("已清理遗留的 llama-server 进程：pid=%s", pid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("清理遗留 llama-server（pid=%s）失败：%s", pid, exc)


def _cleanup_stale_runtime() -> None:
    """服务重启后清理上次留下的 llama-server（只认记录过、且名字匹配的 PID）。"""
    path = _runtime_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    pid = int(data.get("pid") or 0)
    if pid > 0:
        _kill_pid(pid)
        # 等端口真正释放，避免刚杀完就启动时误判「端口被占」。
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _port_open():
            time.sleep(0.2)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_runtime(pid: int) -> None:
    payload = {
        "pid": int(pid),
        "port": _port(),
        "model": str(_model_path()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "llama_build": LLAMA_BUILD,
    }
    try:
        _runtime_file().parent.mkdir(parents=True, exist_ok=True)
        _runtime_file().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("写入 vision runtime.json 失败：%s", exc)


def _start() -> None:
    """拉起 llama-server 并等到 /health 就绪（调用方持有 ``_lock``）。"""
    global _process, _log_handle, _started_at, _load_seconds, _external, _last_error
    global _vram_delta_mb
    exe = _server_exe()
    model = _model_path()
    mmproj = _mmproj_path()
    missing = [str(path) for path in (exe, model, mmproj) if not path.is_file()]
    if missing:
        raise VisionError(
            "视觉模型文件不完整，缺少：" + "、".join(missing) + "；"
            "可调用 /admin/models/download（capability=vision）或按 docs/SERVICE_SETUP.md 放置"
        )

    _cleanup_stale_runtime()
    if _port_open():
        # 端口被占：等几秒（可能是刚杀掉的旧进程还在退出），可用就复用。
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _healthy():
                logger.warning(
                    "视觉端口 %s 已有可用的 llama-server，直接复用（不由本进程管理）",
                    _port(),
                )
                _external = True
                _started_at = time.time()
                _load_seconds = 0.0
                return
            if not _port_open():
                break
            time.sleep(0.25)
        if _port_open() and not _healthy():
            raise VisionError(
                f"视觉端口 {_port()} 已被其它程序占用，且不是可用的 llama-server；"
                "请设置 LOCALAI_VISION_PORT 换一个端口"
            )

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_handle = log_path.open("a", encoding="utf-8", errors="replace")
    _log_handle.write(
        f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动 llama-server "
        f"({LLAMA_BUILD}, port={_port()}, ngl={_gpu_layers()}, ctx={_ctx_size()}) =====\n"
    )
    _log_handle.flush()

    args = [
        str(exe),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        str(_port()),
        "--alias",
        "qwen2.5-vl-3b",
        "-ngl",
        str(_gpu_layers()),
        "-c",
        str(_ctx_size()),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "-t",
        str(max(2, min(8, (os.cpu_count() or 4) - 2))),
        "--parallel",
        "1",
        "--jinja",
        "--no-webui",
    ]
    started = time.perf_counter()
    before_used = gpu_memory_used_mb()
    logger.info("启动视觉引擎：%s", " ".join(args))
    try:
        _process = subprocess.Popen(
            args,
            cwd=str(vision_dir()),
            stdin=subprocess.DEVNULL,
            stdout=_log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_CREATE_NO_WINDOW,
            close_fds=True,
        )
    except Exception as exc:  # noqa: BLE001
        _last_error = f"{type(exc).__name__}: {exc}"
        raise VisionError(f"无法启动 llama-server：{_last_error}") from exc

    deadline = time.monotonic() + max(15.0, _timeout())
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            code = _process.returncode
            _last_error = f"llama-server 启动后立即退出（exit={code}），日志见 {log_path}"
            raise VisionError(_last_error)
        if _healthy():
            _started_at = time.time()
            _load_seconds = round(time.perf_counter() - started, 2)
            _external = False
            _write_runtime(_process.pid or 0)
            logger.info(
                "视觉引擎就绪：pid=%s 加载耗时=%.1fs 日志=%s",
                _process.pid,
                _load_seconds,
                log_path,
            )
            _warmup()
            after_used = gpu_memory_used_mb()
            if before_used is not None and after_used is not None:
                _vram_delta_mb = max(0, int(after_used) - int(before_used))
                logger.info("视觉引擎显存增量：%d MB（nvidia-smi）", _vram_delta_mb)
            return
        time.sleep(0.25)

    _last_error = f"等待 llama-server 就绪超时（{_timeout():.0f}s），日志见 {log_path}"
    stop()
    raise VisionError(_last_error)


def _warmup() -> None:
    """用一张 32x32 小图预热视觉编码器。

    Vulkan 后端第一次跑视觉图会现场编译 shader（约 8~10 秒）；预热后
    真实调用稳定在 1~3 秒/帧。预热失败不影响后续请求（只记日志）。
    """
    global _calls, _last_call_ms
    if not _healthy():
        return
    try:
        from PIL import Image  # noqa: PLC0415

        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (128, 128, 128)).save(buffer, format="JPEG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        started = time.perf_counter()
        _chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "只回复 OK"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
                        },
                    ],
                }
            ],
            max_tokens=4,
        )
        # 预热不计入调用统计。
        _calls = max(0, _calls - 1)
        _last_call_ms = None
        logger.info("视觉引擎预热完成：%.2fs", time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - 预热失败也要能正常服务
        logger.warning("视觉引擎预热失败（不影响后续请求）：%s", exc)


def ensure_loaded(*, timeout: Optional[float] = None) -> None:
    """确保视觉引擎可用；已就绪时零成本，崩溃后自动重拉。"""
    global _last_error, _process
    with _lock:
        if _process is not None and _process.poll() is None:
            return
        if _external and _healthy():
            return
        if _process is not None:
            logger.warning(
                "检测到 llama-server 已退出（exit=%s），重新拉起", _process.returncode
            )
            _close_log()
            _process = None
        _last_error = None
        _start()
        if timeout is not None and _load_seconds is not None:
            logger.debug("视觉引擎加载耗时 %.2fs（调用方超时 %.1fs）", _load_seconds, timeout)


def stop() -> None:
    """停止本进程拉起的 llama-server；可重复调用。"""
    global _process, _external, _started_at
    with _lock:
        proc = _process
        _process = None
        _external = False
        _started_at = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    logger.warning("强制结束 llama-server 失败（pid=%s）", getattr(proc, "pid", None))
            logger.info("视觉引擎已停止（pid=%s）", getattr(proc, "pid", None))
        _close_log()
        try:
            _runtime_file().unlink(missing_ok=True)
        except OSError:
            pass


def _close_log() -> None:
    global _log_handle
    if _log_handle is not None:
        try:
            _log_handle.flush()
            _log_handle.close()
        except Exception:  # noqa: BLE001
            pass
        _log_handle = None


def unload() -> None:
    stop()


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

def _size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 1) if path.is_file() else 0.0
    except OSError:
        return 0.0


def _runtime_version() -> Optional[str]:
    try:
        text = _version_file().read_text(encoding="utf-8").strip()
        return text or LLAMA_BUILD
    except OSError:
        return None


def is_loaded() -> bool:
    """进程是否存活（不发起 HTTP，vram 管家会高频调用）。"""
    proc = _process
    if proc is not None:
        return proc.poll() is None
    if _external:
        return _port_open()
    return False


def status() -> dict[str, Any]:
    """只读状态：不加载模型、不启动进程。"""
    exe, model, mmproj = _server_exe(), _model_path(), _mmproj_path()
    missing = [
        path.name for path in (exe, model, mmproj) if not path.is_file()
    ]
    loaded = is_loaded()
    ready = not missing
    if ready:
        detail = (
            f"Qwen2.5-VL-3B-Instruct Q4_K_M 已就绪"
            + (f"（llama-server 已加载，pid={getattr(_process, 'pid', '-')}）" if loaded else "（首次视觉请求时加载）")
        )
    else:
        detail = (
            "视觉模型未安装，缺少："
            + "、".join(missing)
            + "；可调用 /admin/models/download（capability=vision）下载"
        )
    return {
        "ready": ready,
        "loaded": loaded,
        "model": "Qwen2.5-VL-3B-Instruct-Q4_K_M",
        "model_dir": str(vision_dir()),
        "model_path": str(model),
        "model_size_mb": _size_mb(model),
        "mmproj_path": str(mmproj),
        "mmproj_size_mb": _size_mb(mmproj),
        "server_path": str(exe),
        "server_size_mb": _size_mb(exe),
        "runtime": "llama.cpp",
        "runtime_version": _runtime_version() or LLAMA_BUILD,
        "port": _port(),
        "pid": getattr(_process, "pid", None),
        "external_process": bool(_external),
        "started_at": (
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(_started_at))
            if _started_at
            else None
        ),
        "load_seconds": _load_seconds,
        "last_error": _last_error,
        "calls": _calls,
        "last_call_ms": _last_call_ms,
        "ctx": _ctx_size(),
        "gpu_layers": _gpu_layers(),
        "idle_unload_seconds": int(_env("LOCALAI_VISION_IDLE_UNLOAD", "900") or 900),
        "log_path": str(_log_path()),
        "estimate_mb": ESTIMATE_MB,
        "vram_delta_mb": _vram_delta_mb,
        "detail": detail,
        "missing": missing,
    }


def model_status() -> dict[str, Any]:
    """适配器入口别名。"""
    return status()


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def _image_data_url(path: str | Path, *, max_side: int = 1024) -> str:
    """把图片读成 data URL；顺手缩到最长边 1024，控制视觉 token 数。"""
    target = Path(path)
    if not target.is_file():
        raise VisionError(f"图片不存在：{target}")
    try:
        from PIL import Image  # noqa: PLC0415 - 只有真正识图时才需要
    except Exception as exc:  # noqa: BLE001
        raise VisionError(f"缺少 Pillow，无法读取图片：{exc}") from exc

    try:
        with Image.open(target) as image:
            image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest > max_side:
                scale = max_side / float(longest)
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=88, optimize=True)
    except Exception as exc:  # noqa: BLE001
        raise VisionError(f"读取图片失败：{target}（{type(exc).__name__}: {exc}）") from exc

    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _chat(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
) -> str:
    """调 llama-server 的 /v1/chat/completions，返回正文（同步，放线程池）。"""
    global _last_call_ms, _calls
    body = json.dumps(
        {
            "model": "qwen2.5-vl-3b",
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{_port()}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=float(timeout or _timeout())
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise VisionError(
            f"视觉推理失败（{type(exc).__name__}: {exc}）；"
            f"llama-server 日志见 {_log_path()}"
        ) from exc
    _last_call_ms = round((time.perf_counter() - started) * 1000, 1)
    _calls += 1
    try:
        data = json.loads(raw)
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # 少数版本返回 content 数组
            content = "".join(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
        text = str(content or "").strip()
    except (ValueError, IndexError, AttributeError) as exc:
        raise VisionError(f"llama-server 返回无法解析：{raw[:300]}") from exc
    if not text:
        raise VisionError(f"llama-server 返回空内容：{raw[:300]}")
    return text


def describe(
    image_path: str | Path,
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """单图理解：返回模型原始文本（通常是 JSON）。"""
    ensure_loaded()
    image_url = _image_data_url(image_path)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )
    return _chat(messages, max_tokens=max_tokens, temperature=temperature)


def describe_text(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """纯文本推理（关键帧规划器用），不传图片。"""
    ensure_loaded()
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return _chat(messages, max_tokens=max_tokens, temperature=temperature)


def describe_many(
    images: Iterable[str | Path],
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> list[str]:
    """多图批量：只加载一次模型，逐张请求以保证「一张图一段结果」一一对应。

    llama.cpp 的 OpenAI 接口一次请求可以塞多张图，但多图会共享一段回答，
    无法稳定拆分；批量的收益主要在「不重复加载模型 / 不重复起进程」，
    这里用单进程串行实现，和 /vision/scan 的语义一致。
    """
    ensure_loaded()
    results: list[str] = []
    for image in images:
        results.append(
            describe(
                image,
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 安装 / 下载（管理界面与 doctor 复用）
# ---------------------------------------------------------------------------

def _download(url: str, target: Path, *, progress: Optional[Callable[[str], None]] = None) -> None:
    """带断点续传的下载（先写 .part，完成后原子替换）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "localai-service/vision"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        # 服务器不支持 Range 时会返回 200，此时必须从头写，不能追加。
        if existing and int(getattr(response, "status", 200)) != 206:
            existing = 0
            mode = "wb"
        total = int(response.headers.get("Content-Length") or 0) + existing
        done = existing
        last_report = 0.0
        with part.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if progress is not None and now - last_report >= 2.0:
                    last_report = now
                    progress(
                        f"{target.name}: {done / 1024**2:.1f}/"
                        f"{(total / 1024**2 if total else 0):.1f} MB"
                    )
    if total and part.stat().st_size < total:
        raise VisionError(f"下载不完整：{target.name}（{part.stat().st_size}/{total} 字节）")
    part.replace(target)


def _first_success(urls: tuple[str, ...] | list[str], target: Path, **kwargs: Any) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            _download(url, target, **kwargs)
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            logger.warning("下载失败，换下一个源：%s", errors[-1])
    raise VisionError("下载失败：" + "；".join(errors))


def ensure_assets(*, progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """补齐 ``models/vision/`` 下的二进制与 GGUF；已存在的跳过。

    返回 ``{ok, downloaded, skipped, detail}``。供管理界面后台下载任务调用。
    """
    target_dir = vision_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    skipped: list[str] = []

    def report(text: str) -> None:
        logger.info("%s", text)
        if progress is not None:
            try:
                progress(text)
            except Exception:  # noqa: BLE001
                pass

    # 1) llama-server
    if not _server_exe().is_file():
        zip_path = target_dir / LLAMA_ASSET
        if not zip_path.is_file():
            report(f"下载 llama.cpp {LLAMA_BUILD}（Vulkan 构建）")
            _first_success([prefix + LLAMA_URL for prefix in LLAMA_MIRRORS], zip_path, progress=progress)
        report("解压 llama-server")
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if not name:
                    continue
                extracted = archive.extract(member, target_dir)
                source = Path(extracted)
                final = target_dir / name
                if source != final:
                    shutil.move(str(source), str(final))
        _version_file().write_text(
            f"llama.cpp {LLAMA_BUILD}（Windows Vulkan x64）\n", encoding="utf-8"
        )
        downloaded.append(LLAMA_ASSET)
    else:
        skipped.append(_server_exe().name)

    # 2) Qwen2.5-VL GGUF + mmproj
    for name, urls in MODEL_FILES.items():
        target = target_dir / name
        if target.is_file() and target.stat().st_size > 0:
            skipped.append(name)
            continue
        report(f"下载 {name}")
        _first_success(list(urls), target, progress=progress)
        downloaded.append(name)

    # 3) 视觉投影层：默认用 Q8_0（显存比 F16 少约 500MB）。
    default_mmproj = target_dir / "mmproj-Q8_0.gguf"
    if not _mmproj_path().is_file() and _mmproj_path().name == "mmproj-Q8_0.gguf":
        f16 = target_dir / "mmproj-F16.gguf"
        quantize = target_dir / "llama-quantize.exe"
        if f16.is_file() and quantize.is_file():
            report("量化视觉投影层 mmproj-F16 -> Q8_0")
            proc = subprocess.run(
                [str(quantize), str(f16), str(default_mmproj), "Q8_0"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
                cwd=str(target_dir),
                creationflags=_CREATE_NO_WINDOW,
            )
            if proc.returncode != 0 or not default_mmproj.is_file():
                logger.warning(
                    "mmproj 量化失败（exit=%s），将回退使用 F16：%s",
                    proc.returncode,
                    (proc.stderr or proc.stdout or "")[-500:],
                )
            else:
                downloaded.append(default_mmproj.name)
        elif f16.is_file():
            logger.warning(
                "找不到 llama-quantize.exe，视觉投影层回退使用 F16（显存约多 500MB）"
            )

    missing = [
        path.name
        for path in (_server_exe(), _model_path(), _mmproj_path())
        if not path.is_file()
    ]
    if missing and _mmproj_path().name == "mmproj-Q8_0.gguf":
        # 量化不可用时明确提示用户可以显式改回 F16。
        missing.append("（可设 LOCALAI_VISION_MMPROJ=mmproj-F16.gguf 回退）")
    return {
        "ok": not missing,
        "downloaded": downloaded,
        "skipped": skipped,
        "missing": missing,
        "dir": str(target_dir),
        "detail": "视觉模型已就绪" if not missing else "仍缺少：" + "、".join(missing),
    }


def status_info() -> dict[str, Any]:
    """给 CLI / doctor 的轻量状态（不用起服务）。"""
    return status()


__all__ = [
    "ESTIMATE_MB",
    "LLAMA_BUILD",
    "VisionError",
    "describe",
    "describe_many",
    "describe_text",
    "ensure_assets",
    "ensure_loaded",
    "is_loaded",
    "model_status",
    "status",
    "status_info",
    "stop",
    "unload",
    "vision_dir",
    "vram_delta_mb",
]
