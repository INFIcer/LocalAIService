"""
SenseVoiceSmall 语音识别引擎
============================

这个文件是「不依赖 ComfyUI / 不依赖 HTTP」的本地 ASR 入口，供
``E:\\LocalAIService\\svc\\service.py`` 直接调用。模型权重来自 ModelScope
``iic/SenseVoiceSmall``，已下载到 ``E:\\LocalAIService\\models\\SenseVoiceSmall``。

设计要点
--------
1. **懒加载**：进程启动、import 本模块都不会碰模型；第一次调用
   ``transcribe()`` 才真正把 SenseVoice 加载进显存。这样服务可以只挂监听，
   不占显存。
2. **可卸载**：``unload()`` 会把模型引用、CUDA 缓存一起清掉；服务侧的空闲
   看门狗会在长时间没人用 ASR 时调用它。下次请求再自动加载。
3. **8G 显存适配**：SenseVoiceSmall 是 CTC 小模型，fp32 大约占 1GB 左右；
   默认 ``device="cuda:0"``，显存不够或 CUDA 初始化失败时可以回退 CPU
   （``INDEX_TTS_ASR_ALLOW_CPU_FALLBACK=1``，默认开启）。
4. **富标签解析**：SenseVoice 的输出形如
   ``<|zh|><|HAPPY|><|Speech|><|withitn|>主人，我是大肥鱼。``，
   本模块把语言 / 情感 / 事件标签解析成结构化字段，正文去掉全部标签。
   解析逻辑是纯函数 ``parse_rich_tags()``，可以单独测试。
5. **离线**：``AutoModel(model=<本地目录>)`` 直接读本地权重，不联网。

对外主要函数
------------
- ``model_status()``：模型文件是否齐全、是否已加载（不触发加载）。
- ``ensure_loaded()``：加载并返回常驻的 AutoModel 实例。
- ``transcribe(audio_path, language="auto")``：识别一段音频，返回
  ``{text, language, emotion, events, duration, ...}``。
- ``unload()``：释放模型和显存。
- ``is_loaded()``：当前是否已加载。
- ``parse_rich_tags(raw)``：纯函数，解析 SenseVoice 富标签。

命令行自检（不启动 HTTP 服务）
------------------------------
    E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe E:\\LocalAIService\\engine\\asr_engine.py ^
        --audio E:\\LocalAIService\\outputs\\pick_female-A.wav
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# 必须在 import torch 之前设置：和 TTS 引擎保持一致，减少 8G 卡上的碎片化。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ENGINE_DIR = Path(__file__).resolve().parent
# 服务根目录（engine 的上一级）；改名或走 junction 都会解析到真实目录。
SERVICE_ROOT = ENGINE_DIR.parent

# 模型目录：可用 INDEX_TTS_ASR_MODEL_DIR 覆盖。
MODEL_DIR = Path(
    os.environ.get("INDEX_TTS_ASR_MODEL_DIR", str(SERVICE_ROOT / "models" / "SenseVoiceSmall"))
)

# SenseVoiceSmall 运行必需的文件（缺任何一个都算未就绪）。
# 说明：tokens.json 给 tiktoken 词表；bpe.model 给 SentencePiece 分词；
# am.mvn 是 fbank 的均值方差归一化；config.yaml 描述网络结构。
REQUIRED_FILES = [
    "model.pt",
    "config.yaml",
    "configuration.json",
    "am.mvn",
    "tokens.json",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
]

# 设备：默认 cuda:0；排错时可设 INDEX_TTS_ASR_DEVICE=cpu。
DEVICE = os.environ.get("INDEX_TTS_ASR_DEVICE", "cuda:0").strip() or "cuda:0"

# CUDA 加载失败时是否回退 CPU（默认允许，保证服务不会因为显卡被占满而彻底不可用）。
ALLOW_CPU_FALLBACK = os.environ.get(
    "INDEX_TTS_ASR_ALLOW_CPU_FALLBACK", "1"
).strip().lower() not in ("0", "false", "no", "off")

# 单次推理参数（环境变量可覆盖；一般不用改）。
BATCH_SIZE_S = int(os.environ.get("INDEX_TTS_ASR_BATCH_SIZE_S", "60"))
MERGE_LENGTH_S = int(os.environ.get("INDEX_TTS_ASR_MERGE_LENGTH_S", "15"))
USE_ITN = os.environ.get("INDEX_TTS_ASR_USE_ITN", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

logger = logging.getLogger("sensevoice-asr")

# 加载锁与推理锁分开：
# - 加载锁保证只构造一次 AutoModel；
# - 推理锁保证同一时刻只有一个请求在用模型（CTC 推理虽然无状态，但显存峰值可控）。
_LOAD_LOCK = threading.RLock()
_INFER_LOCK = threading.RLock()

_model: Any = None
_load_seconds: Optional[float] = None
_load_error: Optional[str] = None
_loaded_device: Optional[str] = None


class AsrError(RuntimeError):
    """ASR 侧可预期错误（缺文件、加载失败、识别失败）。"""


# ---------------------------------------------------------------------------
# 富标签解析（纯函数，方便单独测试）
# ---------------------------------------------------------------------------

# `<|xxx|>` 形式的标签。
_TAG_RE = re.compile(r"<\|([^|]+)\|>")

# SenseVoice 语言标签（来自 tokens.json）。
_LANGUAGE_TAGS = frozenset(
    """
    zh en zh/en en/zh yue minnan wuyu dialect ja de es ru ko fr pt tr pl ca nl ar
    sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk
    te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn
    yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as
    tt haw ln ha ba jw su
    """.split()
)

# 情感标签（SenseVoice 支持 8 类 + 未知）。
_EMOTION_TAGS = frozenset(
    {
        "HAPPY",
        "SAD",
        "ANGRY",
        "NEUTRAL",
        "FEARFUL",
        "DISGUSTED",
        "SURPRISED",
        "OTHER",
        "EMO_UNKNOWN",
    }
)

# 音频事件标签（去掉开头的 "/" 后归一）。ASR/AED/SER 是任务标记，不算事件。
_EVENT_TAGS = frozenset(
    {
        "Speech",
        "BGM",
        "Laughter",
        "Applause",
        "Cry",
        "Sneeze",
        "Breath",
        "Cough",
        "Sing",
        "Speech_Noise",
        "GBG",
        "Event_UNK",
    }
)

_TEXTNORM_TAGS = frozenset({"withitn", "woitn"})


def parse_rich_tags(raw: str) -> dict[str, Any]:
    """
    解析 SenseVoice 的富标签输出。

    输入示例::

        <|zh|><|HAPPY|><|Speech|><|withitn|>主人，我是大肥鱼。

    返回::

        {
            "text": "主人，我是大肥鱼。",
            "language": "zh",
            "emotion": "HAPPY",
            "events": ["Speech"],
            "textnorm": "withitn",
            "tags": ["zh", "HAPPY", "Speech", "withitn"],
        }

    规则：
    - 语言标签取第一个命中的（如 ``zh`` / ``en`` / ``yue``）；
    - 情感标签取第一个命中的（如 ``HAPPY``）；
    - 事件标签可以有多个，``/Speech`` 这类闭合标签会归一成 ``Speech``；
    - ``withitn`` / ``woitn`` 只记录到 ``textnorm``，不算事件；
    - 所有 ``<|...|>`` 都从正文里去掉；未知标签保留在 ``tags`` 里便于排错。
    """
    raw = raw or ""
    tags: list[str] = []
    language: Optional[str] = None
    emotion: Optional[str] = None
    textnorm: Optional[str] = None
    events: list[str] = []

    for match in _TAG_RE.finditer(raw):
        tag = match.group(1).strip()
        if not tag:
            continue
        tags.append(tag)

        lowered = tag.lower()
        if language is None and lowered in _LANGUAGE_TAGS:
            language = lowered
            continue
        if lowered == "nospeech":
            # 没有语音：语言位给 nospeech，正文通常是空的。
            if language is None:
                language = "nospeech"
            continue
        if tag in _EMOTION_TAGS or tag.upper() in _EMOTION_TAGS:
            if emotion is None:
                emotion = tag.upper()
            continue
        if tag in _TEXTNORM_TAGS:
            textnorm = tag
            continue
        # ``/Speech`` -> ``Speech``；事件可能连续出现多个。
        event = tag[1:] if tag.startswith("/") else tag
        if event in _EVENT_TAGS and event not in events:
            events.append(event)
            continue
        # 其它未知标签只记录，不下结论。

    # 去掉所有标签，再清理首尾和多余空白。
    text = _TAG_RE.sub("", raw).strip()
    text = re.sub(r"[ \t]+", " ", text)

    return {
        "text": text,
        "language": language,
        "emotion": emotion,
        "events": events,
        "textnorm": textnorm,
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# 模型状态 / 加载 / 卸载
# ---------------------------------------------------------------------------


def _missing_files() -> list[str]:
    """返回缺失的模型文件列表（只检查文件存在且非空）。"""
    missing: list[str] = []
    for name in REQUIRED_FILES:
        path = MODEL_DIR / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return missing


def model_status() -> dict[str, Any]:
    """给 /health、/asr/status 用的模型状态；不会触发模型加载。"""
    missing = _missing_files()
    return {
        "ready": not missing,
        "model_dir": str(MODEL_DIR),
        "missing": missing,
        "loaded": _model is not None,
        "load_seconds": round(_load_seconds, 2) if _load_seconds is not None else None,
        "load_error": _load_error,
        "device": _loaded_device or DEVICE,
    }


def is_loaded() -> bool:
    """当前是否已加载 SenseVoice。"""
    return _model is not None


def _log_vram(tag: str) -> None:
    """打印显存占用，方便在 8G 卡上观察峰值。"""
    try:
        import torch

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


def _resolve_device(requested: str) -> str:
    """把配置里的设备解析成实际可用的设备；cuda 不可用时按开关回退 CPU。"""
    import torch

    device = (requested or "cuda:0").strip()
    if device.startswith("cuda") and not torch.cuda.is_available():
        if ALLOW_CPU_FALLBACK:
            logger.warning("请求的设备 %s 不可用（CUDA 未就绪），回退 CPU", device)
            return "cpu"
        raise AsrError(f"请求的设备 {device} 不可用，且没有开启 CPU 回退")
    return device


def _build_model(device: str):
    """真正构造 funasr AutoModel；调用方需持有 _LOAD_LOCK。"""
    missing = _missing_files()
    if missing:
        raise AsrError(
            f"ASR 模型目录 {MODEL_DIR} 缺少 {len(missing)} 个文件：{', '.join(missing)}"
        )

    # 延迟 import：服务启动时不碰 funasr，保证「只挂监听」。
    from funasr import AutoModel

    logger.info("开始加载 SenseVoiceSmall：model_dir=%s device=%s", MODEL_DIR, device)
    model = AutoModel(
        model=str(MODEL_DIR),
        trust_remote_code=True,
        # 关掉联网版本检查；模型完全从本地目录读。
        disable_update=True,
        device=device,
        disable_pbar=True,
        disable_log=True,
    )
    return model


def ensure_loaded():
    """加载并返回常驻的 AutoModel 实例；失败会抛出 AsrError。"""
    global _model, _load_seconds, _load_error, _loaded_device

    if _model is not None:
        return _model

    with _LOAD_LOCK:
        if _model is not None:
            return _model

        started = time.perf_counter()
        device = _resolve_device(DEVICE)
        try:
            try:
                _model = _build_model(device)
                _loaded_device = device
            except Exception as first_exc:  # noqa: BLE001
                # CUDA 侧失败（常见是显存不足）时，按开关再试一次 CPU。
                can_fallback = (
                    ALLOW_CPU_FALLBACK
                    and device != "cpu"
                    and "CUDA out of memory" in str(first_exc)
                )
                if not can_fallback:
                    raise
                logger.warning(
                    "CUDA 加载 SenseVoice 失败（%s），回退 CPU 重试", first_exc
                )
                _free_cuda()
                _model = _build_model("cpu")
                _loaded_device = "cpu"

        except Exception as exc:  # noqa: BLE001 - 原样带给服务端/用户
            _load_error = f"{type(exc).__name__}: {exc}"
            _loaded_device = None
            logger.exception("SenseVoice 加载失败")
            raise AsrError(f"SenseVoice 加载失败：{_load_error}") from exc

        _load_seconds = time.perf_counter() - started
        _load_error = None
        logger.info(
            "SenseVoice 加载完成，耗时 %.2f 秒（device=%s）",
            _load_seconds, _loaded_device,
        )
        _log_vram("ASR 加载完成")
        return _model


def _free_cuda() -> None:
    """回收 Python 对象并归还 CUDA 缓存；失败不影响主流程。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def unload() -> None:
    """释放 SenseVoice 模型和显存；重复调用安全。"""
    global _model, _load_seconds, _load_error, _loaded_device

    with _LOAD_LOCK:
        if _model is None:
            return
        # 先断引用，再 gc；AutoModel 内部还持有 model / tokenizer / frontend。
        try:
            inner = getattr(_model, "model", None)
            if inner is not None:
                try:
                    inner.to("cpu")
                except Exception:  # noqa: BLE001 - 某些模块没有 to()
                    pass
        except Exception:  # noqa: BLE001
            pass
        _model = None
        _load_seconds = None
        _load_error = None
        _loaded_device = None

    _free_cuda()
    logger.info("SenseVoice 已卸载，显存已释放")


# ---------------------------------------------------------------------------
# 识别
# ---------------------------------------------------------------------------


def _audio_duration(path: Path) -> Optional[float]:
    """读音频时长（秒）；读不出来返回 None。"""
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return None


def transcribe(audio_path: str, language: str = "auto") -> dict[str, Any]:
    """
    识别一段音频。

    参数
    ----
    audio_path: 音频文件路径（wav/mp3/flac 等，torchaudio 能读即可）。
    language:   ``auto``（默认）/ ``zh`` / ``en`` / ``ja`` / ``ko`` / ``yue`` 等；
                传 ``auto`` 让 SenseVoice 自己判断。

    返回
    ----
    dict：
        text      去掉富标签后的正文
        language  识别出的语言（如 zh / en）
        emotion   情感标签（如 HAPPY / NEUTRAL）
        events    音频事件列表（如 ["Speech"]）
        duration  音频时长（秒，读不出来为 None）
        raw       SenseVoice 原始输出（含标签，排错用）
        elapsed   推理耗时（秒）
    出错时抛出 ``AsrError``。
    """
    path = Path(audio_path)
    if not path.is_file():
        raise AsrError(f"音频文件不存在：{path}")

    language = (language or "auto").strip().lower() or "auto"
    model = ensure_loaded()

    started = time.perf_counter()
    try:
        with _INFER_LOCK:
            _log_vram("ASR 推理前")
            results = model.generate(
                input=str(path),
                cache={},
                language=language,
                use_itn=USE_ITN,
                batch_size_s=BATCH_SIZE_S,
                merge_vad=True,
                merge_length_s=MERGE_LENGTH_S,
            )
            _log_vram("ASR 推理后")
    except Exception as exc:  # noqa: BLE001 - 统一包成 AsrError
        logger.exception("SenseVoice 识别失败")
        raise AsrError(f"SenseVoice 识别失败：{type(exc).__name__}: {exc}") from exc

    elapsed = time.perf_counter() - started

    if not results:
        raise AsrError("SenseVoice 没有返回识别结果")

    first = results[0]
    raw = str(first.get("text") or "") if isinstance(first, dict) else str(first)
    parsed = parse_rich_tags(raw)
    duration = _audio_duration(path)

    result = {
        "text": parsed["text"],
        "language": parsed["language"] or language,
        "emotion": parsed["emotion"],
        "events": parsed["events"],
        "duration": round(duration, 3) if duration is not None else None,
        "raw": raw,
        "textnorm": parsed["textnorm"],
        "tags": parsed["tags"],
        "elapsed": round(elapsed, 3),
        "device": _loaded_device,
    }
    logger.info(
        "识别完成：%.2f 秒音频 / 耗时 %.2f 秒 / 语言=%s 情感=%s 事件=%s -> %s",
        duration or 0.0, elapsed, result["language"], result["emotion"],
        ",".join(result["events"]) or "-", result["text"][:80],
    )
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="SenseVoiceSmall 独立引擎命令行自检")
    parser.add_argument(
        "--audio", default=str(SERVICE_ROOT / "outputs" / "pick_female-A.wav"), help="要识别的音频"
    )
    parser.add_argument("--language", default="auto", help="auto/zh/en/ja/ko/yue...")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("模型状态：", model_status())
    started = time.perf_counter()
    result = transcribe(args.audio, language=args.language)
    print("总耗时（含首次加载）：%.2f 秒" % (time.perf_counter() - started))
    print("识别结果：", result)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
