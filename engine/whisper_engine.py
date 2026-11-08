"""
faster-whisper 语音识别引擎（第二个 ASR 引擎，原生时间戳）
=========================================================

这个文件是「不依赖 ComfyUI / 不依赖 HTTP」的本地 faster-whisper 入口，供
``E:\\LocalAIService\\src\\localai\\engines\\asr_fasterwhisper.py`` 调用。
和 SenseVoice 并列：能力名 ``whisper``，可以单独加载 / 卸载，互不影响。

设计要点
--------
1. **懒加载**：import 本模块不碰 torch / faster-whisper，也不初始化 CUDA；
   第一次 ``transcribe()`` 或 ``ensure_loaded()`` 才真正构造 ``WhisperModel``。
   服务启动仍然 0 模型。
2. **权重目录统一**：``LOCALAI_HOME/models/whisper/<size>/``（可用
   ``LOCALAI_WHISPER_DIR`` 覆盖根目录）。faster-whisper 的
   ``WhisperModel(..., download_root=...)`` 会自己拼缓存目录，这里改成
   **先自己把权重落到固定目录，再把该目录传给 WhisperModel**，这样：
   - 权重不散落到用户目录，换机器只需搬 ``models/``；
   - ModelScope / HF 镜像 / 本地 HF 缓存三种来源都能落到同一个目录。
3. **下载顺序**（先探测，不通再退）：
   a. 目录里已有完整权重 -> 直接用；
   b. 本机 HuggingFace 缓存里已有 ``models--Systran--faster-whisper-<size>``
      -> 复制过来（不联网）；
   c. ModelScope 的 ``Systran/faster-whisper-<size>``（国内可用）；
   d. HF 镜像 ``HF_ENDPOINT=https://hf-mirror.com``（直连 HF 已知不通）。
4. **8G 显存适配**：默认 ``cuda`` + ``int8_float16``；CPU 回退 ``int8``。
   CUDA 初始化或显存失败时按开关自动退 CPU（``LOCALAI_WHISPER_ALLOW_CPU_FALLBACK``，
   默认 1）。
5. **可卸载**：``unload()`` 释放模型引用 + ``gc`` + ``torch.cuda.empty_cache()``。
6. **时间戳**：faster-whisper 原生按句返回 ``start/end``（秒），不需要 VAD 二次
   切分；``vad_filter=True`` 只用来丢掉静音，不改变 segment 的绝对时间。

对外主要函数
------------
- ``model_status(size=None)``：权重是否就绪、是否已加载（不触发下载/加载）。
- ``is_loaded(size=None)``：当前是否已加载（可指定 size）。
- ``ensure_loaded(size, device, compute_type)``：加载并返回常驻 WhisperModel。
- ``transcribe(path, size="small", language="auto", timestamps=True, vad=True)``：
  返回 ``{text, language, duration, segments, words?, ...}``。
- ``unload()``：释放模型和显存；重复调用安全。
- ``ensure_weights(size)``：只准备权重（给 doctor / 下载脚本用）。

命令行自检（不启动 HTTP 服务）
-----------------------------
    E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe E:\\LocalAIService\\engine\\whisper_engine.py ^
        --audio E:\\LocalAIService\\outputs\\pick_female-A.wav --size tiny
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

ENGINE_DIR = Path(__file__).resolve().parent
# 服务根目录（engine 的上一级）；改名或走 junction 都会解析到真实目录。
SERVICE_ROOT = ENGINE_DIR.parent

#: 支持的四档 + large-v3（和 faster-whisper 的官方 short name 一致）。
SIZES = ("tiny", "base", "small", "medium", "large-v3")

#: 权重根目录：默认 ``<HOME>/models/whisper``。
WEIGHTS_ROOT = Path(
    os.environ.get("LOCALAI_WHISPER_DIR", str(SERVICE_ROOT / "models" / "whisper"))
)

#: faster-whisper 在 HF / ModelScope 上的仓库名模板。
REPO_TEMPLATE = "Systran/faster-whisper-{size}"

#: 判断权重目录是否完整的文件（large-v3 也带 vocabulary.txt）。
REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")

#: 环境变量默认值（服务层 config.py 也会注入同名变量）。
DEFAULT_SIZE = (os.environ.get("LOCALAI_WHISPER_MODEL", "small") or "small").strip().lower()
DEFAULT_DEVICE = (os.environ.get("LOCALAI_WHISPER_DEVICE", "cuda") or "cuda").strip()
DEFAULT_COMPUTE = (
    os.environ.get("LOCALAI_WHISPER_COMPUTE", "int8_float16") or "int8_float16"
).strip()

#: CPU 上不支持 float16，统一换成 int8。
CPU_COMPUTE = "int8"

#: 中文识别提示词：faster-whisper 对中文容易输出繁体（「會議記錄」），给一句
#: 普通话提示后实测 CER 从 16.6% 降到 0.9%。只在 ``language=auto/zh`` 时使用，
#: 显式指定其它语言时不加；设成空字符串可关闭。
DEFAULT_PROMPT = (
    os.environ.get("LOCALAI_WHISPER_PROMPT", "以下是普通话的句子。") or ""
).strip()

#: CUDA 失败时是否回退 CPU（默认允许，显卡被占满时服务仍可用）。
ALLOW_CPU_FALLBACK = os.environ.get(
    "LOCALAI_WHISPER_ALLOW_CPU_FALLBACK", "1"
).strip().lower() not in ("0", "false", "no", "off")

#: 下载超时（秒）交给 huggingface_hub / modelscope 自己的重试逻辑；这里只控制
#: 单个大文件的分块大小之外不再额外限制。

logger = logging.getLogger("faster-whisper")

#: 模型加载 / 推理 / 卸载的统一互斥锁。
#: faster-whisper 的 CTranslate2 推理默认不保证同一模型多线程并发安全；
#: 服务层适配器容量固定为 1，这里再用 RLock 兜一层（可重入，transcribe 内部
#: 调 ensure_loaded 不会自锁）。
_ENGINE_LOCK = threading.RLock()

_model: Any = None
_loaded_size: Optional[str] = None
_loaded_device: Optional[str] = None
_loaded_compute: Optional[str] = None
_load_seconds: Optional[float] = None
_load_error: Optional[str] = None
_weights_source: Optional[str] = None


class WhisperError(RuntimeError):
    """whisper 侧可预期错误（缺依赖、下载失败、加载/识别失败）。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def normalize_size(size: Optional[str]) -> str:
    """把 size 归一成支持的四档之一；不认识时回落到默认值并记日志。"""
    value = (size or DEFAULT_SIZE or "small").strip().lower()
    if value in SIZES:
        return value
    logger.warning("不支持的 whisper size=%r，回落到 %s", size, DEFAULT_SIZE)
    return DEFAULT_SIZE if DEFAULT_SIZE in SIZES else "small"


def weights_dir(size: Optional[str] = None) -> Path:
    """某档模型的权重目录：``<WEIGHTS_ROOT>/<size>/``。"""
    return WEIGHTS_ROOT / normalize_size(size)


def _missing_files(target: Path) -> list[str]:
    """返回权重目录里缺失或为空的文件。"""
    missing: list[str] = []
    for name in REQUIRED_FILES:
        path = target / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return missing


def _dir_size_mb(target: Path) -> float:
    total = 0
    try:
        for item in target.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return round(total / (1024 * 1024), 1)


def _package_version(name: str) -> Optional[str]:
    """取包版本（不 import，避免副作用）。"""
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:  # noqa: BLE001 - 未安装 / 元数据缺失都按没有处理
        return None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 权重准备（本地 -> HF 本地缓存 -> ModelScope -> HF 镜像）
# ---------------------------------------------------------------------------


def _copy_from_hf_cache(size: str, target: Path) -> Optional[str]:
    """本机 HF 缓存里有就直接复制，不联网。

    兼容两种缓存布局：
    - ``models--Systran--faster-whisper-<size>/snapshots/<rev>/``
    - 直接就是文件目录（用户手动放置）
    """
    repo_dir_name = f"models--Systran--faster-whisper-{size}"
    candidates: list[Path] = []
    env_home = (os.environ.get("HF_HOME") or "").strip()
    hub_cache = (os.environ.get("HF_HUB_CACHE") or "").strip()
    roots: list[Path] = []
    if hub_cache:
        roots.append(Path(hub_cache))
    if env_home:
        roots.append(Path(env_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        repo_dir = root / repo_dir_name
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            candidates.extend(sorted(snapshots.iterdir(), reverse=True))
        elif repo_dir.is_dir():
            candidates.append(repo_dir)

    for candidate in candidates:
        if _missing_files(candidate):
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_FILES:
            source = candidate / name
            if not source.is_file():
                continue
            try:
                shutil.copy2(source, target / name)
            except OSError as exc:
                logger.warning("从 HF 缓存复制 %s 失败：%s", name, exc)
                return None
        if not _missing_files(target):
            logger.info("whisper %s 权重来自本机 HF 缓存：%s", size, candidate)
            return "hf-cache"
    return None


def _download_modelscope(size: str, target: Path) -> str:
    """从 ModelScope 下载（国内可达）；仓库名与 HF 一致。"""
    if not _has_module("modelscope"):
        raise WhisperError("未安装 modelscope，无法从 ModelScope 下载")
    from modelscope import snapshot_download  # noqa: PLC0415 - 故意延迟

    repo = REPO_TEMPLATE.format(size=size)
    logger.info("从 ModelScope 下载 whisper-%s：%s -> %s", size, repo, target)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(target))
    return "modelscope"


def _download_hf_mirror(size: str, target: Path) -> str:
    """从 HF 镜像（hf-mirror.com）下载；直连 HF 已知不通。"""
    if not _has_module("huggingface_hub"):
        raise WhisperError("未安装 huggingface_hub，无法从 HF 镜像下载")

    endpoint = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as hub_constants  # noqa: PLC0415

        # 模块已经 import 过时，改环境变量不会生效，这里同步覆盖。
        hub_constants.ENDPOINT = endpoint
    except Exception:  # noqa: BLE001 - 改不了常量就只靠环境变量
        pass

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    repo = REPO_TEMPLATE.format(size=size)
    logger.info("从 HF 镜像下载 whisper-%s：%s -> %s", size, repo, target)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(target), max_workers=4)
    return "hf-mirror"


def ensure_weights(size: Optional[str] = None) -> dict[str, Any]:
    """确保某档权重在 ``models/whisper/<size>/``；返回状态字典。

    只准备权重，不创建 WhisperModel。下载失败抛 :class:`WhisperError`，
    错误信息里列出每种来源的失败原因。
    """
    global _weights_source

    size = normalize_size(size)
    target = weights_dir(size)
    missing = _missing_files(target)
    if not missing:
        _weights_source = "local"
        return {
            "ready": True,
            "size": size,
            "model_dir": str(target),
            "missing": [],
            "source": "local",
            "size_mb": _dir_size_mb(target),
        }

    logger.info("whisper %s 权重不完整（缺 %s），开始准备：%s", size, ", ".join(missing), target)
    reasons: list[str] = []
    source: Optional[str] = None

    # 1) 本机 HF 缓存
    try:
        source = _copy_from_hf_cache(size, target)
    except Exception as exc:  # noqa: BLE001 - 缓存复制失败不算致命
        reasons.append(f"HF 本地缓存：{type(exc).__name__}: {exc}")

    # 2) ModelScope
    if source is None:
        try:
            source = _download_modelscope(size, target)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"ModelScope：{type(exc).__name__}: {exc}")
            logger.warning("ModelScope 下载 whisper-%s 失败：%s", size, exc)

    # 3) HF 镜像
    if source is None:
        try:
            source = _download_hf_mirror(size, target)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"HF 镜像：{type(exc).__name__}: {exc}")
            logger.warning("HF 镜像下载 whisper-%s 失败：%s", size, exc)

    missing = _missing_files(target)
    if missing or source is None:
        detail = "；".join(reasons) or "未知原因"
        raise WhisperError(
            f"whisper-{size} 权重准备失败（目录 {target} 仍缺 {', '.join(missing) or '文件'}）：{detail}"
        )

    _weights_source = source
    logger.info("whisper %s 权重就绪：%s（来源 %s，%.1f MB）", size, target, source, _dir_size_mb(target))
    return {
        "ready": True,
        "size": size,
        "model_dir": str(target),
        "missing": [],
        "source": source,
        "size_mb": _dir_size_mb(target),
    }


# ---------------------------------------------------------------------------
# 状态 / 加载 / 卸载
# ---------------------------------------------------------------------------


def model_status(size: Optional[str] = None) -> dict[str, Any]:
    """给 /health、/asr/status 用的模型状态；不触发下载或加载。"""
    size = normalize_size(size)
    target = weights_dir(size)
    missing = _missing_files(target)
    return {
        "ready": not missing,
        "size": size,
        "model": f"faster-whisper-{size}",
        "model_dir": str(target),
        "missing": missing,
        "size_mb": _dir_size_mb(target) if target.is_dir() else 0.0,
        "weights_source": _weights_source,
        "loaded": _model is not None,
        "loaded_size": _loaded_size,
        "load_seconds": round(_load_seconds, 2) if _load_seconds is not None else None,
        "load_error": _load_error,
        "device": _loaded_device or DEFAULT_DEVICE,
        "compute_type": _loaded_compute or DEFAULT_COMPUTE,
        "package": _package_version("faster-whisper"),
        "ctranslate2": _package_version("ctranslate2"),
        "supported_sizes": list(SIZES),
        "default_size": DEFAULT_SIZE,
    }


def is_loaded(size: Optional[str] = None) -> bool:
    """当前是否已加载；给了 size 时要求加载的正是这一档。"""
    if _model is None:
        return False
    if size is None:
        return True
    return _loaded_size == normalize_size(size)


def _log_vram(tag: str) -> None:
    """打印显存占用；失败不影响识别。"""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "[显存] %s：torch 已分配 %.2f GB / 已保留 %.2f GB；系统空闲 %.2f / %.2f GB",
            tag, allocated, reserved, free / (1024 ** 3), total / (1024 ** 3),
        )
    except Exception:  # noqa: BLE001 - 显存日志失败不影响识别
        pass


def _free_cuda() -> None:
    """回收 Python 对象并归还 CUDA 缓存；失败不影响主流程。"""
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


_DLL_PREPARED = False


def _prepare_cuda_dlls() -> None:
    """把 pip 装的 CUDA 运行库目录加进 DLL 搜索路径（Windows 必需）。

    CTranslate2 的 CUDA 版本链接的是 CUDA 12 的 ``cublas64_12.dll`` /
    ``cudnn64_9.dll``，而本机 torch 是 cu130 自带的 CUDA 13；两者互不覆盖。
    这里优先用 ``nvidia-*-cu12`` pip 包里的 DLL 目录（engine 的 requirements
    里已固定），加到 ``os.add_dll_directory`` 与 ``PATH``，让 ctranslate2 能
    加载到正确版本。找不到时保持原样，交给 ctranslate2 自己报错。
    """
    global _DLL_PREPARED

    if _DLL_PREPARED or os.name != "nt":
        return
    _DLL_PREPARED = True

    candidates: list[Path] = []
    for base in (Path(sys.prefix) / "Lib" / "site-packages", Path(sys.prefix) / "lib" / "site-packages"):
        nvidia_root = base / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for package in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
            for sub in ("bin", "lib"):
                directory = nvidia_root / package / sub
                if directory.is_dir():
                    candidates.append(directory)
    # 没装 pip 版 CUDA 时，退而用系统 CUDA Toolkit（如果存在）。
    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        for version_dir in sorted(toolkit.glob("v12*"), reverse=True):
            bin_dir = version_dir / "bin"
            if bin_dir.is_dir():
                candidates.append(bin_dir)
                break

    if not candidates:
        return
    for directory in candidates:
        try:
            os.add_dll_directory(str(directory))
        except (OSError, AttributeError):
            pass
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        [str(directory) for directory in candidates] + [existing]
    )
    logger.debug("CUDA DLL 目录已加入搜索路径：%s", [str(item) for item in candidates])


def _cuda_available() -> bool:
    """用 ctranslate2 判断 CUDA 是否可用（它才是真正跑推理的后端）。"""
    _prepare_cuda_dlls()
    try:
        import ctranslate2  # noqa: PLC0415

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("ctranslate2 检查 CUDA 失败：%s", exc)
        return False


def _resolve_device(requested: str) -> str:
    """解析设备；cuda 不可用时按开关回退 CPU。"""
    device = (requested or DEFAULT_DEVICE or "cuda").strip()
    if device.startswith("cuda") and not _cuda_available():
        if ALLOW_CPU_FALLBACK:
            logger.warning("请求的设备 %s 不可用（CUDA 未就绪），回退 CPU", device)
            return "cpu"
        raise WhisperError(f"请求的设备 {device} 不可用，且没有开启 CPU 回退")
    return device


def _resolve_compute(device: str, requested: str) -> str:
    """CPU 上把 float16 系列换成 int8；其它按原样。"""
    compute = (requested or DEFAULT_COMPUTE or "int8_float16").strip()
    if device == "cpu" and ("float16" in compute or compute in ("float32", "auto")):
        return CPU_COMPUTE
    return compute


def _build_model(size: str, device: str, compute_type: str):
    """真正构造 WhisperModel（调用方持有 _ENGINE_LOCK）。"""
    if not _has_module("faster_whisper"):
        raise WhisperError(
            "未安装 faster-whisper；请用 engine/.venv/Scripts/python.exe -m pip install faster-whisper==1.2.1"
        )
    _prepare_cuda_dlls()
    info = ensure_weights(size)
    model_dir = info["model_dir"]

    from faster_whisper import WhisperModel  # noqa: PLC0415 - 故意延迟

    logger.info(
        "开始加载 faster-whisper-%s：model_dir=%s device=%s compute_type=%s",
        size, model_dir, device, compute_type,
    )
    return WhisperModel(model_dir, device=device, compute_type=compute_type)


def ensure_loaded(
    size: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
):
    """加载并返回常驻 WhisperModel；失败抛 :class:`WhisperError`。

    ``size`` 与当前已加载的不一致时，先卸载旧的再加载新的（8G 卡上不可能
    同时挂两档 whisper）。
    """
    global _model, _loaded_size, _loaded_device, _loaded_compute
    global _load_seconds, _load_error

    size = normalize_size(size)
    requested_device = (device or DEFAULT_DEVICE or "cuda").strip()
    requested_compute = (compute_type or DEFAULT_COMPUTE or "int8_float16").strip()

    with _ENGINE_LOCK:
        # 只要档位一致就直接复用：不重复探测设备，也避免「上次回退到 CPU
        # 后每个请求都重新尝试 CUDA」这种反复卸载/加载。
        if _model is not None and _loaded_size == size:
            return _model

        if _model is not None:
            logger.info(
                "whisper 已加载 %s，本次需要 %s，先卸载旧模型", _loaded_size, size
            )
            _unload_locked()

        started = time.perf_counter()
        target_device = _resolve_device(requested_device)
        target_compute = _resolve_compute(target_device, requested_compute)
        try:
            try:
                _model = _build_model(size, target_device, target_compute)
                _loaded_size = size
                _loaded_device = target_device
                _loaded_compute = target_compute
            except Exception as first_exc:  # noqa: BLE001
                # CUDA 侧失败（显存不足 / cuDNN 缺失）时，按开关再试一次 CPU。
                can_fallback = (
                    ALLOW_CPU_FALLBACK
                    and target_device != "cpu"
                    and isinstance(first_exc, Exception)
                )
                if not can_fallback:
                    raise
                logger.warning(
                    "CUDA 加载 faster-whisper-%s 失败（%s），回退 CPU 重试",
                    size, first_exc,
                )
                _free_cuda()
                _model = _build_model(size, "cpu", CPU_COMPUTE)
                _loaded_size = size
                _loaded_device = "cpu"
                _loaded_compute = CPU_COMPUTE
        except Exception as exc:  # noqa: BLE001 - 原样带给服务端/用户
            _load_error = f"{type(exc).__name__}: {exc}"
            _loaded_size = None
            _loaded_device = None
            _loaded_compute = None
            _model = None
            logger.exception("faster-whisper 加载失败")
            raise WhisperError(f"faster-whisper 加载失败：{_load_error}") from exc

        _load_seconds = time.perf_counter() - started
        _load_error = None
        logger.info(
            "faster-whisper-%s 加载完成，耗时 %.2f 秒（device=%s compute_type=%s）",
            size, _load_seconds, _loaded_device, _loaded_compute,
        )
        _log_vram("whisper 加载完成")
        return _model


def _unload_locked() -> None:
    """在持锁状态下释放模型（内部使用）。"""
    global _model, _loaded_size, _loaded_device, _loaded_compute
    global _load_seconds, _load_error

    model = _model
    _model = None
    _loaded_size = None
    _loaded_device = None
    _loaded_compute = None
    _load_seconds = None
    _load_error = None
    if model is not None:
        # WhisperModel 内部持有 ctranslate2 模型；显式断引用后再 gc。
        try:
            del model
        except Exception:  # noqa: BLE001
            pass
    _free_cuda()


def unload() -> None:
    """释放模型和显存；重复调用安全。"""
    with _ENGINE_LOCK:
        if _model is None:
            _free_cuda()
            return
        _unload_locked()
    logger.info("faster-whisper 已卸载，显存已释放")


# ---------------------------------------------------------------------------
# 识别
# ---------------------------------------------------------------------------


def _audio_duration(path: Path) -> Optional[float]:
    """读音频时长（秒）；读不出来返回 None。"""
    try:
        import soundfile as sf  # noqa: PLC0415

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return None


def transcribe(
    audio_path: str,
    size: Optional[str] = None,
    language: str = "auto",
    timestamps: bool = True,
    vad: bool = True,
    *,
    word_timestamps: bool = False,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> dict[str, Any]:
    """识别一段音频，返回全文 + 原生分段（秒）。

    参数
    ----
    audio_path: 音频/视频文件路径（faster-whisper 用 av 解码，常见容器都支持）。
    size:       ``tiny|base|small|medium|large-v3``；不传用 ``LOCALAI_WHISPER_MODEL``。
    language:   ``auto``（默认）/ ``zh`` / ``en`` / ``ja`` / ``ko`` / ``yue`` 等；
                传 ``auto`` 时由 whisper 自己检测。
    timestamps: 是否返回 ``segments``（默认 True）。
    vad:        是否开 ``vad_filter``（默认 True，只丢静音，不改绝对时间）。
    word_timestamps: 是否返回逐词时间戳（默认 False，慢一点）。

    返回
    ----
    dict::

        {
          "text": "全文",
          "language": "zh",
          "duration": 12.3,
          "segments": [{"start": 0.0, "end": 2.4, "text": "..."}],
          "words": [{"start":..., "end":..., "word":...}],  # 可选
          "timestamp_source": "native",
          "elapsed": 0.8, "device": "cuda", "compute_type": "int8_float16",
          "size": "small", "model_dir": "..."
        }

    错误统一抛 :class:`WhisperError`。
    """
    path = Path(audio_path)
    if not path.is_file():
        raise WhisperError(f"音频文件不存在：{path}")

    lang = (language or "auto").strip().lower() or "auto"
    language_arg = None if lang == "auto" else lang
    # 中文（含自动检测）时给一句普通话提示；显式别的语言不加。
    initial_prompt = DEFAULT_PROMPT if (DEFAULT_PROMPT and lang in ("auto", "zh")) else None

    with _ENGINE_LOCK:
        model = ensure_loaded(size, device, compute_type)
        started = time.perf_counter()
        try:
            _log_vram("whisper 推理前")
            segments_iter, info = model.transcribe(
                str(path),
                language=language_arg,
                beam_size=5,
                vad_filter=bool(vad),
                word_timestamps=bool(word_timestamps),
                initial_prompt=initial_prompt,
            )
            segments: list[dict[str, Any]] = []
            words: list[dict[str, Any]] = []
            texts: list[str] = []
            for segment in segments_iter:
                text = str(getattr(segment, "text", "") or "").strip()
                start = round(float(getattr(segment, "start", 0.0) or 0.0), 3)
                end = round(float(getattr(segment, "end", 0.0) or 0.0), 3)
                if end <= start:
                    continue
                if timestamps:
                    segments.append({"start": start, "end": end, "text": text})
                texts.append(text)
                if word_timestamps:
                    for word in getattr(segment, "words", None) or []:
                        words.append(
                            {
                                "start": round(float(getattr(word, "start", 0.0) or 0.0), 3),
                                "end": round(float(getattr(word, "end", 0.0) or 0.0), 3),
                                "word": str(getattr(word, "word", "") or ""),
                                "probability": round(
                                    float(getattr(word, "probability", 0.0) or 0.0), 4
                                ),
                            }
                        )
        except Exception as exc:  # noqa: BLE001 - 统一包成 WhisperError
            logger.exception("faster-whisper 识别失败")
            raise WhisperError(f"faster-whisper 识别失败：{type(exc).__name__}: {exc}") from exc

        elapsed = time.perf_counter() - started
        _log_vram("whisper 推理后")

    duration = _audio_duration(path)
    detected_language = getattr(info, "language", None) or lang
    language_probability = getattr(info, "language_probability", None)
    text = "".join(texts).strip()

    result: dict[str, Any] = {
        "text": text,
        "language": str(detected_language),
        "language_probability": (
            round(float(language_probability), 4)
            if language_probability is not None
            else None
        ),
        "duration": round(duration, 3) if duration is not None else None,
        "segments": segments,
        "timestamp_source": "native",
        "elapsed": round(elapsed, 3),
        "device": _loaded_device,
        "compute_type": _loaded_compute,
        "size": _loaded_size,
        "model_dir": str(weights_dir(_loaded_size or size)),
        "initial_prompt": initial_prompt,
    }
    if word_timestamps:
        result["words"] = words
    logger.info(
        "whisper 识别完成：%.2f 秒音频 / 耗时 %.2f 秒 / 语言=%s / %d 段 -> %s",
        duration or 0.0, elapsed, result["language"], len(segments), text[:80],
    )
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="faster-whisper 独立引擎命令行自检")
    parser.add_argument(
        "--audio",
        default=str(SERVICE_ROOT / "outputs" / "pick_female-A.wav"),
        help="要识别的音频",
    )
    parser.add_argument("--size", default=DEFAULT_SIZE, choices=list(SIZES))
    parser.add_argument("--language", default="auto", help="auto/zh/en/ja/ko/yue...")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--compute", default=DEFAULT_COMPUTE)
    parser.add_argument("--words", action="store_true", help="输出逐词时间戳")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("模型状态：", json.dumps(model_status(args.size), ensure_ascii=False, indent=2))
    started = time.perf_counter()
    result = transcribe(
        args.audio,
        size=args.size,
        language=args.language,
        device=args.device,
        compute_type=args.compute,
        word_timestamps=args.words,
    )
    print("总耗时（含首次加载）：%.2f 秒" % (time.perf_counter() - started))
    print("识别结果：", json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
