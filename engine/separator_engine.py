"""
Demucs 人声 / 伴奏分离引擎
==========================

这个文件是「不依赖 ComfyUI / 不依赖 HTTP」的本地音轨分离入口，供
``E:\\LocalAIService\\svc\\service.py`` 直接调用。底层用 Meta 的 Demucs
（``demucs 4.1.0`` 的 ``demucs.api.Separator``），模型是 ``htdemucs``。

设计要点
--------
1. **懒加载**：import 本模块、启动服务都不会碰 torch / 模型；第一次调用
   ``separate()``（或 ``ensure_loaded()``）才把权重读进显存。这样常驻服务
   仍然是 0 模型。
2. **可卸载**：``unload()`` 会把 Separator、模型引用、CUDA 缓存一起清掉；
   服务侧的空闲看门狗会在长时间没人用分离时调用它。下次请求再自动加载。
3. **权重落在项目里**：默认从
   ``E:\\LocalAIService\\models\\torch\\hub\\checkpoints`` 读取 ``.th`` 权重，
   不联网、不写用户目录。可用 ``INDEX_TTS_DEMUCS_REPO`` 覆盖。
4. **8G 显存适配**：默认 ``device=cuda:0``、``segment=7`` 秒、``shifts=1``；
   htdemucs 实测峰值约 0.6GB，3070 8G 很宽裕。显存不够或 CUDA 初始化失败时
   可设 ``INDEX_TTS_SEPARATE_ALLOW_CPU_FALLBACK=0`` 关掉 CPU 回退。
5. **输入统一**：调用方应先用 ffmpeg 转成 44.1kHz / 立体声 wav 再传进来；
   本模块也会在读取时按 Demucs 模型的采样率自动重采样（sphn/ffmpeg）。
6. **输出**：每个 stem 一个 wav（PCM16，44.1kHz，立体声），文件名固定为
   ``vocals.wav`` / ``drums.wav`` / ``bass.wav`` / ``other.wav``（6s 模型还
   有 ``guitar.wav`` / ``piano.wav``），方便下游直接拼路径。

支持的模型
----------
- ``htdemucs``：4 轨（vocals / drums / bass / other），默认；
- ``htdemucs_ft``：4 轨微调版（4 个模型 bag，显存/耗时约 4 倍，需额外权重）；
- ``htdemucs_6s``：6 轨（多 guitar / piano），需额外权重。

对外主要函数
------------
- ``model_status(model_name="htdemucs")``：权重是否齐全、是否已加载（不触发加载）。
- ``ensure_loaded(model_name="htdemucs")``：加载并返回常驻的 Separator。
- ``separate(input_wav, out_dir, model="htdemucs", stems=None)``：分离并落盘。
- ``unload()``：释放模型和显存。
- ``is_loaded()``：当前是否已加载。

命令行自检（不启动 HTTP 服务）
-----------------------------
    E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe E:\\LocalAIService\\engine\\separator_engine.py ^
        --audio D:\\dsh_ws\\_voice_src\\source_44k_stereo.wav ^
        --out E:\\LocalAIService\\outputs\\separate_probe
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

# 必须在 import torch 之前设置：和 TTS / ASR 引擎保持一致，减少 8G 卡上的碎片化。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ENGINE_DIR = Path(__file__).resolve().parent
# 服务根目录（engine 的上一级）；改名或走 junction 都会解析到真实目录。
SERVICE_ROOT = ENGINE_DIR.parent

# TORCH_HOME 指到项目里：即使某些代码路径走 torch.hub，权重也不会落到用户目录。
TORCH_HOME = Path(
    os.environ.get("INDEX_TTS_TORCH_HOME", str(SERVICE_ROOT / "models" / "torch"))
)
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

# Demucs 本地权重仓库：放 .th 权重和对应的 bag yaml（htdemucs.yaml 等）。
WEIGHTS_DIR = Path(
    os.environ.get("INDEX_TTS_DEMUCS_REPO", str(TORCH_HOME / "hub" / "checkpoints"))
)

# 设备：默认 cuda:0；排错时可设 INDEX_TTS_SEPARATE_DEVICE=cpu。
DEVICE = os.environ.get("INDEX_TTS_SEPARATE_DEVICE", "cuda:0").strip() or "cuda:0"

# CUDA 加载失败时是否回退 CPU（默认允许，保证显卡被占满时服务不彻底不可用）。
ALLOW_CPU_FALLBACK = os.environ.get(
    "INDEX_TTS_SEPARATE_ALLOW_CPU_FALLBACK", "1"
).strip().lower() not in ("0", "false", "no", "off")

# 推理参数：segment 越小越省显存，shifts 越大越慢但略稳。
SEGMENT = int(os.environ.get("INDEX_TTS_SEPARATE_SEGMENT", "7"))
SHIFTS = int(os.environ.get("INDEX_TTS_SEPARATE_SHIFTS", "1"))
OVERLAP = float(os.environ.get("INDEX_TTS_SEPARATE_OVERLAP", "0.25"))
JOBS = int(os.environ.get("INDEX_TTS_SEPARATE_JOBS", "0"))

DEFAULT_MODEL = "htdemucs"

# 每个 bag 需要的文件（bag yaml + 各子模型 .th）。
# 注意：htdemucs_ft 是 4 个模型的 bag，缺一个都算未就绪。
MODEL_FILES: dict[str, list[str]] = {
    "htdemucs": ["htdemucs.yaml", "955717e8-8726e21a.th"],
    "htdemucs_ft": [
        "htdemucs_ft.yaml",
        "f7e0c4bc-ba3fe64a.th",
        "d12395a8-e57c48e6.th",
        "92cfc3b6-ef3bcb9c.th",
        "04573f0d-f3cf25b2.th",
    ],
    "htdemucs_6s": ["htdemucs_6s.yaml", "5c90dfd2-34c22ccb.th"],
}

# 每个模型能输出的 stem 名（顺序和 Demucs 的 sources 一致）。
MODEL_SOURCES: dict[str, list[str]] = {
    "htdemucs": ["drums", "bass", "other", "vocals"],
    "htdemucs_ft": ["drums", "bass", "other", "vocals"],
    "htdemucs_6s": ["drums", "bass", "other", "vocals", "guitar", "piano"],
}

logger = logging.getLogger("demucs-separator")

# 加载锁与推理锁分开：
# - 加载锁保证只构造一次 Separator；
# - 推理锁保证同一时刻只有一个请求在跑 GPU（换模型时也不会和推理撞车）。
_LOAD_LOCK = threading.RLock()
_INFER_LOCK = threading.RLock()

_SEPARATOR: Any = None
_MODEL_NAME: Optional[str] = None
_LOAD_SECONDS: Optional[float] = None
_LOAD_ERROR: Optional[str] = None
_LOADED_DEVICE: Optional[str] = None


class SeparatorError(RuntimeError):
    """分离侧可预期错误（缺权重、模型名不合法、推理失败）。"""


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------


def _missing_files(model_name: str) -> list[str]:
    """返回某个模型缺少的权重文件（模型名不认识时返回空列表）。"""
    files = MODEL_FILES.get(model_name)
    if files is None:
        return []
    return [name for name in files if not (WEIGHTS_DIR / name).is_file()]


def available_models() -> list[str]:
    """本机权重齐全、可以直接用的模型名。"""
    return [name for name in MODEL_FILES if not _missing_files(name)]


def model_status(model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    """检查 Demucs 权重是否齐全；只读文件系统，不 import torch、不加载模型。"""
    name = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    missing = _missing_files(name)
    known = name in MODEL_FILES
    ready = known and not missing
    if not known:
        detail = f"不认识的 Demucs 模型 {name}；可选：{', '.join(MODEL_FILES)}"
    elif ready:
        if is_loaded():
            detail = f"Demucs 权重已就绪（{name}），当前已加载 {_MODEL_NAME}"
        else:
            detail = f"Demucs 权重已就绪（{name}）（首次请求时加载）"
    else:
        detail = f"Demucs 权重不完整（{name}），缺少：{', '.join(missing)}"
    return {
        "ready": ready,
        "loaded": is_loaded(),
        "model": name,
        "loaded_model": _MODEL_NAME,
        "weights_dir": str(WEIGHTS_DIR),
        "missing": missing,
        "available": available_models(),
        "device": _LOADED_DEVICE or DEVICE,
        "segment": SEGMENT,
        "shifts": SHIFTS,
        "load_seconds": _LOAD_SECONDS,
        "load_error": _LOAD_ERROR,
        "detail": detail,
        "sources": MODEL_SOURCES.get(name, []),
    }


def is_loaded() -> bool:
    """当前是否已经加载了某个 Demucs 模型。"""
    return _SEPARATOR is not None


def loaded_model() -> Optional[str]:
    """当前已加载的模型名；没加载返回 None。"""
    return _MODEL_NAME


# ---------------------------------------------------------------------------
# 加载 / 卸载
# ---------------------------------------------------------------------------


def _resolve_device() -> str:
    """把配置里的 device 落到实际可用的设备上（CUDA 不可用时可回退 CPU）。"""
    import torch  # 延迟 import：没装 torch 也能 import 本模块

    if DEVICE.startswith("cuda"):
        if torch.cuda.is_available():
            return DEVICE
        if not ALLOW_CPU_FALLBACK:
            raise SeparatorError("CUDA 不可用，且 INDEX_TTS_SEPARATE_ALLOW_CPU_FALLBACK=0")
        logger.warning("CUDA 不可用，Demucs 回退到 CPU（会慢很多）")
        return "cpu"
    return DEVICE


def ensure_loaded(model_name: str = DEFAULT_MODEL) -> Any:
    """加载 Demucs 模型（线程安全，重复调用只加载一次）。

    换模型时会先卸载旧模型，保证 8G 卡上同一时刻只有一个 Demucs 常驻。
    """
    global _SEPARATOR, _MODEL_NAME, _LOAD_SECONDS, _LOAD_ERROR, _LOADED_DEVICE

    name = (model_name or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if name not in MODEL_FILES:
        raise SeparatorError(
            f"不支持的 Demucs 模型 {name!r}；可选：{', '.join(MODEL_FILES)}"
        )

    if _SEPARATOR is not None and _MODEL_NAME == name:
        return _SEPARATOR

    # 先确认目标模型权重齐全，再动旧模型：避免「请求一个没下全的模型，
    # 结果把当前能用的模型也卸了」。
    status = model_status(name)
    if not status["ready"]:
        raise SeparatorError(status["detail"])

    with _LOAD_LOCK:
        if _SEPARATOR is not None and _MODEL_NAME == name:
            return _SEPARATOR
        if _SEPARATOR is not None:
            # 请求了另一个模型：先释放旧的，避免两份权重同时占显存。
            _unload_locked()

        started = time.perf_counter()
        try:
            device = _resolve_device()
            # 延迟 import：服务启动和 import 本模块都不会加载 torch / demucs。
            from demucs.api import Separator  # noqa: PLC0415

            logger.info(
                "加载 Demucs %s：repo=%s device=%s segment=%s shifts=%s",
                name, WEIGHTS_DIR, device, SEGMENT, SHIFTS,
            )
            separator = Separator(
                model=name,
                repo=WEIGHTS_DIR,
                device=device,
                shifts=SHIFTS,
                overlap=OVERLAP,
                split=True,
                segment=SEGMENT,
                jobs=JOBS,
                progress=False,
            )
            # 明确把权重放到目标设备：这样 loaded=true 就代表显存里确实有模型。
            separator._model.to(device)  # noqa: SLF001 - demucs.api 没有公开的 to()
            separator._model.eval()  # noqa: SLF001

            _SEPARATOR = separator
            _MODEL_NAME = name
            _LOADED_DEVICE = device
            _LOAD_SECONDS = round(time.perf_counter() - started, 2)
            _LOAD_ERROR = None
            logger.info("Demucs %s 加载完成：%.2fs", name, _LOAD_SECONDS)
            return separator
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            _SEPARATOR = None
            _MODEL_NAME = None
            _LOADED_DEVICE = None
            _LOAD_SECONDS = None
            logger.error("Demucs 加载失败：%s", _LOAD_ERROR)
            raise SeparatorError(f"Demucs 模型加载失败：{_LOAD_ERROR}") from exc


def _unload_locked() -> None:
    """真正的释放逻辑；调用方必须持有 _LOAD_LOCK。"""
    global _SEPARATOR, _MODEL_NAME, _LOAD_SECONDS, _LOADED_DEVICE

    if _SEPARATOR is None:
        return
    _SEPARATOR = None
    _MODEL_NAME = None
    _LOAD_SECONDS = None
    _LOADED_DEVICE = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def unload() -> None:
    """释放 Demucs 模型和显存；没加载时是空操作。"""
    with _LOAD_LOCK:
        if _SEPARATOR is None:
            return
        _unload_locked()
    logger.info("Demucs 模型已卸载")


# ---------------------------------------------------------------------------
# 分离
# ---------------------------------------------------------------------------


def _normalize_stems(model_name: str, stems: Any) -> list[str]:
    """把 "vocals,drums" 或列表统一成合法、去重、保序的 stem 名列表。"""
    sources = MODEL_SOURCES[model_name]
    if stems is None:
        return list(sources)
    if isinstance(stems, str):
        requested = [part.strip().lower() for part in stems.split(",") if part.strip()]
    else:
        requested = [str(part).strip().lower() for part in stems if str(part).strip()]
    if not requested:
        raise SeparatorError("stems 不能为空")
    unknown = [name for name in requested if name not in sources]
    if unknown:
        raise SeparatorError(
            f"模型 {model_name} 没有这些 stem：{', '.join(unknown)}；"
            f"可用：{', '.join(sources)}"
        )
    # 去重但保留调用方给的顺序
    ordered: list[str] = []
    for name in requested:
        if name not in ordered:
            ordered.append(name)
    return ordered


def separate(
    input_wav: str | Path,
    out_dir: str | Path,
    model: str = DEFAULT_MODEL,
    stems: Any = None,
) -> dict[str, Any]:
    """把一段音频分离成多个 stem，落盘到 ``out_dir``。

    参数
    ----
    input_wav : 44.1kHz / 立体声 wav 最稳（其它格式也能读，Demucs 会自行重采样）
    out_dir   : 输出目录（会自动创建）
    model     : ``htdemucs`` / ``htdemucs_ft`` / ``htdemucs_6s``
    stems     : ``"vocals,drums"`` 或列表；None = 该模型的全部 stem

    返回::

        {
          "model": "htdemucs",
          "sample_rate": 44100,
          "duration": 9.752,
          "elapsed": 1.23,
          "dir": "E:\\...\\outputs\\separate\\20261108_...",
          "stems": {"vocals": "E:\\...\\vocals.wav", ...}
        }
    """
    name = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    requested = _normalize_stems(name, stems)
    source = Path(input_wav)
    if not source.is_file():
        raise SeparatorError(f"找不到输入音频：{source}")
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    separator = ensure_loaded(name)
    started = time.perf_counter()
    try:
        with _INFER_LOCK:
            origin, separated = separator.separate_audio_file(source)
            # 先记下时长，尽量在张量还在时完成所有 GPU -> CPU 的搬运。
            duration = float(origin.shape[-1]) / float(separator.samplerate)
            saved: dict[str, str] = {}
            for stem in requested:
                tensor = separated[stem]
                array = tensor.detach().cpu().numpy()
                # Demucs 输出是 (channels, samples)，soundfile 要 (samples, channels)。
                if array.ndim == 2:
                    array = array.T
                else:
                    array = array.reshape(-1, 1)
                path = target_dir / f"{stem}.wav"
                import soundfile as sf  # 延迟 import，也很便宜

                sf.write(str(path), array, int(separator.samplerate), subtype="PCM_16")
                saved[stem] = str(path.resolve())
        elapsed = round(time.perf_counter() - started, 2)
    except SeparatorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SeparatorError(f"Demucs 分离失败：{type(exc).__name__}: {exc}") from exc

    logger.info(
        "Demucs 分离完成：%s -> %s（%s），时长 %.2fs，耗时 %.2fs",
        source.name, target_dir, ", ".join(saved), duration, elapsed,
    )
    return {
        "model": name,
        "sample_rate": int(separator.samplerate),
        "duration": round(duration, 3),
        "elapsed": elapsed,
        "dir": str(target_dir.resolve()),
        "stems": saved,
    }


# ---------------------------------------------------------------------------
# 命令行自检
# ---------------------------------------------------------------------------


def _main() -> int:
    parser = argparse.ArgumentParser(description="Demucs 人声/伴奏分离自检")
    parser.add_argument("--audio", help="输入音频")
    parser.add_argument("--out", help="输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="htdemucs / htdemucs_ft / htdemucs_6s")
    parser.add_argument("--stems", default="", help="逗号分隔，默认全部")
    parser.add_argument("--status", action="store_true", help="只打印模型状态")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.status:
        print(json.dumps(model_status(args.model), ensure_ascii=False, indent=2))
        return 0
    if not args.audio or not args.out:
        parser.error("--audio 和 --out 必填（除非只用 --status）")

    stems = args.stems or None
    result = separate(args.audio, args.out, model=args.model, stems=stems)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
