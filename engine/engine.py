"""
IndexTTS-2.5 独立推理引擎
=========================

这个文件是「不依赖 ComfyUI」的 IndexTTS-2.5 合成入口，供
``E:\\LocalAIService\\svc\\service.py`` 以 ``TTS_BACKEND=engine`` 方式直接调用。

设计要点
--------
1. **复制而不是软链**：推理源码来自 ComfyUI 自定义节点
   ``custom_nodes\\index-tts-2.5\\indextts``，已原样复制到
   ``E:\\LocalAIService\\engine\\indextts_vendor\\``。这样不会改动 ComfyUI 目录，
   也不会被节点的升级/删除影响。
2. **常驻单例**：第一次调用 ``synthesize()`` 时才加载模型，之后所有请求复用
   同一个 ``IndexTTS2`` 实例。加载过程有线程锁保护，避免并发重复加载。
3. **8G 显存适配**：使用 bf16（RTX 3070 支持）+ 上游自带的 ``low_vram`` 逻辑
   （显存 < 10GB 时自动把长文本按标点切块）。不启用 bigvgan 的 CUDA 自定义
   kernel / DeepSpeed / torch.compile，避免预编译扩展和 ABI 问题。
4. **离线**：辅助模型（w2v-bert-2.0 / campplus / bigvgan / semantic_codec）
   已经放在 ``<模型目录>\\hf_cache\\``，加载时不会联网。

对外主要函数
------------
- ``model_status()``：模型文件是否齐全、引擎是否已加载。
- ``synthesize(text, ref_audio_path, out_wav_path, language='zh', speed=1.0, ...)``：
  合成一段文本并写成 WAV，返回耗时、采样率、音频时长等元数据。
- ``ensure_loaded()``：显式加载模型（服务启动预热或排错时可用）。

命令行自检（不启动 HTTP 服务）
------------------------------
    E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe E:\\LocalAIService\\engine\\engine.py ^
        --text "主人，我是大肥鱼。" --voice E:\\LocalAIService\\voices\\default.mp3 ^
        --out E:\\LocalAIService\\outputs\\engine_selftest.wav
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# 必须在 import torch 之前设置：减少 8G 显存下的碎片化。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 引擎自身目录 / vendored 源码目录
ENGINE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ENGINE_DIR / "indextts_vendor"
# 服务根目录（engine 的上一级）。用 resolve() 是为了改名/走 junction 都能解析到真实目录。
SERVICE_ROOT = ENGINE_DIR.parent

# 模型目录：可用 INDEX_TTS_MODEL_DIR 覆盖；默认是服务根目录下的这份权重。
MODEL_DIR = Path(
    os.environ.get("INDEX_TTS_MODEL_DIR", str(SERVICE_ROOT / "models" / "TTS" / "IndexTTS-2.5"))
)

# 主权重（与节点 model_loader.REQUIRED_FILES 保持一致）
REQUIRED_FILES = [
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
]

# 辅助模型：缺任何一个都会触发上游的 ensure_models_available() 联网下载，
# 本机要求「不重下模型」，所以 /health 会一起检查。
REQUIRED_AUX = [
    r"hf_cache\w2v-bert-2.0\config.json",
    r"hf_cache\w2v-bert-2.0\model.safetensors",
    r"hf_cache\w2v-bert-2.0\preprocessor_config.json",
    r"hf_cache\campplus_cn_common.bin",
    r"hf_cache\bigvgan\config.json",
    r"hf_cache\bigvgan\bigvgan_generator.pt",
]

# 运行参数（环境变量可覆盖）
USE_BF16 = os.environ.get("INDEX_TTS_BF16", "1").strip().lower() not in ("0", "false", "no", "off")
USE_QWEN_EMO = os.environ.get("INDEX_TTS_QWEN_EMO", "0").strip().lower() in ("1", "true", "yes", "on")
# 设备：默认交给上游自动探测（有 CUDA 就是 cuda:0）；排错时可设成 cpu。
DEVICE = os.environ.get("INDEX_TTS_DEVICE", "").strip() or None
# 采样参数：和 ComfyUI 节点默认值对齐。
NUM_BEAMS = int(os.environ.get("INDEX_TTS_NUM_BEAMS", "3"))
MAX_MEL_TOKENS = int(os.environ.get("INDEX_TTS_MAX_MEL_TOKENS", "1500"))
MAX_TEXT_TOKENS = int(os.environ.get("INDEX_TTS_MAX_TEXT_TOKENS", "120"))
INTERVAL_SILENCE = int(os.environ.get("INDEX_TTS_INTERVAL_SILENCE", "200"))

logger = logging.getLogger("indextts-engine")

# 加载锁与推理锁分开：
# - 加载锁保证只构造一次 IndexTTS2；
# - 推理锁保证同一时刻只有一个请求在用 GPU（上游实例内部有缓存状态，不能并发）。
_LOAD_LOCK = threading.RLock()
_INFER_LOCK = threading.RLock()

_tts: Any = None
_load_seconds: Optional[float] = None
_load_error: Optional[str] = None


class EngineError(RuntimeError):
    """引擎侧可预期错误（缺文件、加载失败、推理失败）。"""


def _missing_files() -> list[str]:
    """返回缺失的主权重/辅助模型文件列表。"""
    missing: list[str] = []
    for name in REQUIRED_FILES + REQUIRED_AUX:
        path = MODEL_DIR / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return missing


def model_status() -> dict[str, Any]:
    """给 /health 用的模型状态；不触发模型加载。"""
    missing = _missing_files()
    return {
        "ready": not missing,
        "model_dir": str(MODEL_DIR),
        "missing": missing,
        "loaded": _tts is not None,
        "load_seconds": round(_load_seconds, 2) if _load_seconds is not None else None,
        "load_error": _load_error,
        "device": DEVICE or "auto",
        "use_bf16": USE_BF16,
        "use_qwen_emo": USE_QWEN_EMO,
    }


def is_loaded() -> bool:
    return _tts is not None


def _import_indextts():
    """把 vendored 源码放到 sys.path 最前，再 import 上游 IndexTTS2。"""
    vendor = str(VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    elif sys.path.index(vendor) != 0:
        sys.path.remove(vendor)
        sys.path.insert(0, vendor)

    # 清掉可能由其它路径导入的同名包，避免混用不同版本。
    for key in list(sys.modules):
        if key == "indextts" or key.startswith("indextts."):
            del sys.modules[key]
    gc.collect()

    from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

    return IndexTTS2


def _build_tts():
    """真正构造 IndexTTS2；调用方需持有 _LOAD_LOCK。"""
    missing = _missing_files()
    if missing:
        raise EngineError(
            f"模型目录 {MODEL_DIR} 缺少 {len(missing)} 个文件：{', '.join(missing)}"
        )

    import torch

    IndexTTS2 = _import_indextts()
    logger.info(
        "开始加载 IndexTTS-2.5：model_dir=%s device=%s bf16=%s qwen_emo=%s",
        MODEL_DIR, DEVICE or "auto", USE_BF16, USE_QWEN_EMO,
    )
    tts = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        # None → 上游自动选择 cuda:0 / cpu
        device=DEVICE,
        use_bf16=USE_BF16,
        # 保守配置：不启用自定义 CUDA kernel / DeepSpeed / accel / compile
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        use_qwen_emo=USE_QWEN_EMO,
    )
    # 上游在 8G 卡上会自动把 low_vram 置 True；这里只记录日志。
    if getattr(tts, "low_vram", False):
        logger.info("上游已启用 low_vram：长文本会按标点切块，降低显存峰值")
    return tts


def ensure_loaded():
    """加载并返回常驻的 IndexTTS2 实例；失败会抛出 EngineError。"""
    global _tts, _load_seconds, _load_error

    if _tts is not None:
        return _tts

    with _LOAD_LOCK:
        if _tts is not None:
            return _tts
        started = time.perf_counter()
        try:
            _tts = _build_tts()
        except Exception as exc:  # noqa: BLE001 - 原样带给服务端/用户
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("IndexTTS 引擎加载失败")
            raise EngineError(f"IndexTTS 引擎加载失败：{_load_error}") from exc

        _load_seconds = time.perf_counter() - started
        _load_error = None
        logger.info("IndexTTS 引擎加载完成，耗时 %.2f 秒", _load_seconds)
        _log_vram("加载完成")
        return _tts


def _apply_seed(seed: int) -> None:
    """固定随机种子；0 表示不固定（与上游节点语义一致）。"""
    if not seed:
        return
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    except Exception:  # noqa: BLE001 - 显存日志失败不影响合成
        pass


def synthesize(
    text: str,
    ref_audio_path: str,
    out_wav_path: str,
    language: str = "zh",
    speed: float = 1.0,
    *,
    emo_vector: Optional[list[float]] = None,
    emo_text: Optional[str] = None,
    emo_alpha: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """
    合成一段文本，结果写成 16bit PCM WAV。

    参数
    ----
    text:            要合成的文本（不能为空）。
    ref_audio_path:  参考音色音频路径（wav/mp3/flac 等，librosa 能读即可）。
    out_wav_path:    输出 WAV 路径；父目录会自动创建。
    language:        zh / en / ja / es / ar。
    speed:           0.5~2.0；内部换算成上游的 duration_factor = 1/speed。
    emo_vector:      8 维情感向量（可选）；顺序：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然。
    emo_text:        情感文本（可选，需要 INDEX_TTS_QWEN_EMO=1 才会加载 Qwen 情感模型）。
    emo_alpha:       情感强度，默认 1.0。
    seed:            随机种子；0 表示不固定。

    返回
    ----
    dict：包含 elapsed / load_seconds / duration / sample_rate / bytes / out_path 等。
    """
    global _tts

    text = (text or "").strip()
    if not text:
        raise EngineError("text 不能为空")

    ref = Path(ref_audio_path)
    if not ref.is_file():
        raise EngineError(f"参考音频不存在：{ref}")

    out = Path(out_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            out.unlink()
        except OSError:
            pass

    speed = max(0.5, min(2.0, float(speed or 1.0)))
    duration_factor = max(0.5, min(2.0, 1.0 / speed))

    if emo_text and not USE_QWEN_EMO:
        raise EngineError(
            "当前引擎没有加载 Qwen 情感模型，不能使用 emotion 文本；"
            "请改用 8 维情感向量，或设置环境变量 INDEX_TTS_QWEN_EMO=1 后重启服务。"
        )
    if emo_vector is not None:
        if len(emo_vector) != 8:
            raise EngineError("emo_vector 必须是 8 个 0~1 的数字")
        emo_vector = [max(0.0, min(1.0, float(v))) for v in emo_vector]

    tts = ensure_loaded()
    _apply_seed(int(seed or 0))

    started = time.perf_counter()
    with _INFER_LOCK:
        _log_vram("推理前")
        try:
            # 注意：这里用非流式 infer()，上游会自己写 WAV。
            # low_vram 模式下长文本会被上游拆段并在内部拼接。
            result = tts.infer(
                spk_audio_prompt=str(ref),
                text=text,
                output_path=str(out),
                lang=language,
                emo_audio_prompt=None,
                emo_alpha=float(emo_alpha),
                emo_vector=emo_vector,
                use_emo_text=bool(emo_text),
                emo_text=emo_text,
                use_random=False,
                interval_silence=INTERVAL_SILENCE,
                verbose=False,
                max_text_tokens_per_segment=MAX_TEXT_TOKENS,
                stream_return=False,
                duration_factor=duration_factor,
                text_normalization=True,
                do_sample=True,
                temperature=0.8,
                top_p=0.8,
                top_k=30,
                repetition_penalty=10.0,
                length_penalty=0.0,
                num_beams=NUM_BEAMS,
                max_mel_tokens=MAX_MEL_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - 统一包成 EngineError
            logger.exception("IndexTTS 推理失败")
            raise EngineError(f"IndexTTS 推理失败：{type(exc).__name__}: {exc}") from exc

    elapsed = time.perf_counter() - started

    if not out.is_file() or out.stat().st_size <= 0:
        raise EngineError(f"推理结束但没有生成音频文件：{out}（上游返回 {result!r}）")

    # 用 soundfile 读真实时长/采样率，服务端和验收脚本都以它为准。
    try:
        import soundfile as sf

        info = sf.info(str(out))
        duration = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
    except Exception as exc:  # noqa: BLE001
        raise EngineError(f"输出 WAV 无法读取：{exc}") from exc

    # 8G 卡上这一步很关键：推理过程中 PyTorch 会预留一些临时显存，
    # 不及时归还的话，第二次请求只剩不到 0.5GB 余量，容易在 CUDA 内核里崩掉。
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - 回收失败不影响结果
        pass

    _log_vram("推理后")
    meta = {
        "out_path": str(out),
        "bytes": out.stat().st_size,
        "duration": round(duration, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "elapsed": round(elapsed, 3),
        "load_seconds": round(_load_seconds, 2) if _load_seconds is not None else None,
        "rtf": round(elapsed / duration, 3) if duration > 0 else None,
        "duration_factor": duration_factor,
        "language": language,
        "seed": seed,
    }
    logger.info(
        "合成完成：%.2f 秒音频 / 耗时 %.2f 秒 / %.2f MB -> %s",
        duration, elapsed, meta["bytes"] / (1024 * 1024), out,
    )
    return meta


def unload() -> None:
    """释放模型（排错/换参数时用；服务正常运行时不需要）。"""
    global _tts, _load_seconds, _load_error

    with _LOAD_LOCK:
        if _tts is None:
            return
        for attr in (
            "gpt", "semantic_model", "semantic_codec", "s2mel",
            "campplus_model", "bigvgan", "qwen_emo", "emo_matrix", "spk_matrix",
        ):
            try:
                setattr(_tts, attr, None)
            except Exception:  # noqa: BLE001
                pass
        _tts = None
        _load_seconds = None
        _load_error = None

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    logger.info("IndexTTS 引擎已卸载")


def _main() -> int:
    parser = argparse.ArgumentParser(description="IndexTTS-2.5 独立引擎命令行自检")
    parser.add_argument("--text", default="主人，我是大肥鱼，这条声音是我自己挑的。")
    parser.add_argument("--voice", default=str(SERVICE_ROOT / "voices" / "default.mp3"))
    parser.add_argument("--out", default=str(SERVICE_ROOT / "outputs" / "engine_selftest.wav"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("模型状态：", model_status())
    started = time.perf_counter()
    meta = synthesize(
        text=args.text,
        ref_audio_path=args.voice,
        out_wav_path=args.out,
        language=args.language,
        speed=args.speed,
    )
    print("总耗时（含首次加载）：%.2f 秒" % (time.perf_counter() - started))
    print("结果：", meta)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
