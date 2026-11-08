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
6. **可控并发**：推理闸门容量由 ``LOCALAI_ASR_PARALLEL`` 决定（默认 1）。
   容量 1 时与旧版的一把 ``RLock`` 完全等价：同一时刻只有一个推理；容量
   2~4 时最多允许对应数量的 SenseVoice 推理同时进入。模型加载 / 卸载属于
   「结构性操作」，仍然独占：加载只在模型为空时发生（此时不可能有推理），
   卸载则先把全部推理槽位收干，再释放模型和 CUDA 缓存。

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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

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

# ---------------------------------------------------------------------------
# 分段转写用的 VAD（fsmn-vad）
# ---------------------------------------------------------------------------
#: VAD 权重目录（放在 LOCALAI_HOME/models 下，不散到用户目录）。
VAD_MODEL_DIR = Path(
    os.environ.get(
        "LOCALAI_ASR_VAD_DIR",
        str(SERVICE_ROOT / "models" / "vad" / "fsmn-vad"),
    )
)
#: ModelScope 仓库 id（本机 funasr 能直接 AutoModel(model="fsmn-vad")，这里
#: 先把权重落到固定目录，再按本地目录加载，避免运行期散落到用户缓存）。
VAD_MODEL_ID = os.environ.get(
    "LOCALAI_ASR_VAD_MODEL", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
)
#: 单个 VAD 段最长毫秒数；超过会被切分（默认 30 秒）。
VAD_MAX_SEGMENT_MS = int(os.environ.get("LOCALAI_ASR_VAD_MAX_SEGMENT_MS", "30000"))
#: 句尾静音容忍（毫秒）：超过这么久没声音就断句。默认 500ms。
#: 注意：不显式传这个参数时 funasr 会走「动态静音」策略，容易把几十秒的
#: 连续语音并成一段；显式传 500ms 后实测 46 秒里 12 句能切成 12 段。
VAD_MAX_END_SILENCE_MS = int(
    os.environ.get("LOCALAI_ASR_VAD_MAX_END_SILENCE_MS", "500")
)
#: VAD 权重完整性检查：这两个文件在就绪（其余文件由 funasr 读取）。
VAD_REQUIRED_FILES = ("model.pt", "config.yaml")


logger = logging.getLogger("sensevoice-asr")


def _read_parallel_capacity() -> int:
    """读取推理并发容量：默认 1，限制在 1~4。

    1 = 完全沿用旧版串行行为；2~4 = 允许同一时刻有 N 个 SenseVoice 推理。
    非法值静默回落到 1，避免一个笔误把服务变成不可用或开得过大。
    """
    raw = os.environ.get("LOCALAI_ASR_PARALLEL", "1")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("LOCALAI_ASR_PARALLEL=%r 不是整数，按 1（串行）处理", raw)
        return 1
    return max(1, min(4, value))


def _read_unload_timeout() -> float:
    """读取卸载等待超时；非法值回落到 30 秒。"""
    raw = os.environ.get("LOCALAI_ASR_UNLOAD_TIMEOUT", "30")
    try:
        return max(1.0, float(str(raw).strip() or "30"))
    except (TypeError, ValueError):
        logger.warning("LOCALAI_ASR_UNLOAD_TIMEOUT=%r 不是数字，按 30 秒处理", raw)
        return 30.0


#: 本进程实际生效的推理并发容量（导入时读取一次，服务层会在 import 前注入）。
ASR_PARALLEL = _read_parallel_capacity()

#: 卸载时等待所有推理退出的最长时间（秒）；超时只记日志、不强行拆模型。
ASR_UNLOAD_TIMEOUT = _read_unload_timeout()


class _InferenceDrainTimeout(RuntimeError):
    """卸载时等待推理退出的时间超时。"""


class _InferenceGate:
    """可配置容量的推理闸门，用法类似 ``threading.BoundedSemaphore``。

    为什么要自己包一层
    ------------------
    - 推理路径只做 ``with gate.slot():``，内部只拿信号量，**不嵌套任何独占锁**，
      因此不会出现「推理等独占锁、卸载等推理」的死锁；
    - 卸载走 ``with gate.exclusive():``：按容量逐个收干所有槽位。收干期间新的
      推理会阻塞在信号量上，等卸载结束才继续，保证「卸载时绝不会有推理在跑」；
    - 统计 ``in_use / peak`` 给 ``/asr/status``、``/capabilities`` 看。

    容量 1 时，``slot()`` 的行为与旧版的 ``RLock`` 完全一致：同一时刻只有一个
    线程能进入临界区。与 ``RLock`` 的唯一区别是不支持同线程重入，而本模块的
    推理路径没有重入，所以语义不受影响。
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._semaphore = threading.BoundedSemaphore(self._capacity)
        self._state_lock = threading.Lock()
        self._in_use = 0
        self._peak = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        with self._state_lock:
            return self._in_use

    @property
    def peak(self) -> int:
        with self._state_lock:
            return self._peak

    @contextmanager
    def slot(self) -> Iterator[None]:
        """占一个推理槽位；退出时**一定**归还，不会把计数搞乱。"""
        self._semaphore.acquire()
        with self._state_lock:
            self._in_use += 1
            if self._in_use > self._peak:
                self._peak = self._in_use
        try:
            yield
        finally:
            with self._state_lock:
                self._in_use = max(0, self._in_use - 1)
            self._semaphore.release()

    @contextmanager
    def exclusive(self, timeout: float = ASR_UNLOAD_TIMEOUT) -> Iterator[None]:
        """收干全部槽位后独占；用于卸载模型。

        超时抛 :class:`_InferenceDrainTimeout`。已经收上来的槽位会在 ``finally``
        里按实际数量归还，**不会多 release**（BoundedSemaphore 会因此报错）。
        """
        capacity = self._capacity
        acquired = 0
        deadline = time.monotonic() + max(0.0, float(timeout))
        try:
            while acquired < capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._semaphore.acquire(timeout=remaining):
                    raise _InferenceDrainTimeout(
                        f"等待推理退出超过 {timeout:.1f} 秒（已收 {acquired}/{capacity} 个槽位）"
                    )
                acquired += 1
            yield
        finally:
            for _ in range(acquired):
                self._semaphore.release()

    def snapshot(self) -> dict[str, Any]:
        """只读快照（给状态接口 / 日志）。"""
        with self._state_lock:
            return {
                "capacity": self._capacity,
                "in_use": self._in_use,
                "peak": self._peak,
            }


# 加载锁与推理闸门分开：
# - 加载锁（_LOAD_LOCK）是结构性操作的互斥锁：同一时刻只构造 / 释放一次模型；
# - 推理闸门（_INFER_GATE）限制并发推理数，容量 1 时就是旧版的串行锁。
# 获取顺序固定为「先推理槽位，后加载锁」；卸载也先收干推理槽位再拿加载锁，
# 两者顺序一致，因此不会互相等待成环。
_LOAD_LOCK = threading.RLock()
_INFER_GATE = _InferenceGate(ASR_PARALLEL)


def configure_parallel(capacity: int) -> int:
    """服务层启动时把引擎闸门对齐到配置容量；返回最终容量。

    正常启动时本模块还没被 import，直接按环境变量构造即可；但如果模块已经被
    import（例如测试或嵌入调用），这里允许在**没有推理在跑**时安全换挡。
    有推理正在跑时拒绝换挡并返回当前容量，避免把正在等待的调用丢掉。
    """
    global ASR_PARALLEL, _INFER_GATE

    wanted = max(1, min(4, int(capacity)))
    with _LOAD_LOCK:
        active = _INFER_GATE.in_use
        if active > 0:
            logger.warning(
                "拒绝调整 ASR 并发容量 %d -> %d：当前仍有 %d 个推理在跑",
                _INFER_GATE.capacity,
                wanted,
                active,
            )
            return _INFER_GATE.capacity
        if wanted != _INFER_GATE.capacity:
            logger.info(
                "ASR 并发容量调整：%d -> %d（无推理在跑，安全换挡）",
                _INFER_GATE.capacity,
                wanted,
            )
            ASR_PARALLEL = wanted
            _INFER_GATE = _InferenceGate(wanted)
    os.environ["LOCALAI_ASR_PARALLEL"] = str(wanted)
    return wanted


_model: Any = None
_load_seconds: Optional[float] = None
_load_error: Optional[str] = None
_loaded_device: Optional[str] = None

#: 分段转写用的 VAD 模型（懒加载；与 SenseVoice 一起在 unload() 里释放）。
_vad_model: Any = None
_vad_load_error: Optional[str] = None


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
        "parallel": parallel_status(),
        "vad": vad_model_status(),
    }


def vad_model_status() -> dict[str, Any]:
    """VAD 模型状态（只查文件与内存标志，不触发下载/加载）。"""
    missing: list[str] = []
    for name in VAD_REQUIRED_FILES:
        path = VAD_MODEL_DIR / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return {
        "ready": not missing,
        "model": "fsmn-vad",
        "model_id": VAD_MODEL_ID,
        "model_dir": str(VAD_MODEL_DIR),
        "missing": missing,
        "loaded": _vad_model is not None,
        "load_error": _vad_load_error,
        "max_segment_ms": VAD_MAX_SEGMENT_MS,
        "max_end_silence_ms": VAD_MAX_END_SILENCE_MS,
    }


def parallel_status() -> dict[str, Any]:
    """推理并发配置与实时占用（给 /asr/status、/capabilities 用）。"""
    snapshot = _INFER_GATE.snapshot()
    snapshot.update(
        {
            "env": "LOCALAI_ASR_PARALLEL",
            "unload_timeout_seconds": ASR_UNLOAD_TIMEOUT,
        }
    )
    return snapshot


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
    """加载并返回常驻的 AutoModel 实例；失败会抛出 AsrError。

    结构性互斥：整段构造过程都在 ``_LOAD_LOCK`` 里，且只有 ``_model is None``
    时才会真正构造。模型为空时不可能有推理在跑；一旦构造完成，等待同一把锁的
    其它调用会直接复用新模型。因此「加载时不能有并发推理」由模型生命周期
    本身 + 加载锁共同保证。
    """
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
    """释放 SenseVoice 模型和显存；重复调用安全。

    卸载是结构性操作，必须独占：
    1. 先收干推理闸门的全部槽位，期间新的推理会排队等待；
    2. 再拿 ``_LOAD_LOCK``，释放模型引用并清 CUDA 缓存；
    3. 最后按实际收到的槽位数归还信号量。

    如果等待超过 ``LOCALAI_ASR_UNLOAD_TIMEOUT``（默认 30 秒）仍有推理在跑，
    只记错误日志并放弃本次卸载，模型保持现状——**绝不把正在推理的模型拆掉**，
    也不会把信号量计数搞乱（下一次空闲回收还可以再试）。
    """
    global _model, _load_seconds, _load_error, _loaded_device
    global _vad_model, _vad_load_error

    try:
        with _INFER_GATE.exclusive(timeout=ASR_UNLOAD_TIMEOUT):
            with _LOAD_LOCK:
                if _model is None and _vad_model is None:
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
                # VAD 只是分段转写的辅助模型，跟着一起释放。
                _vad_model = None
                _vad_load_error = None

            # 在独占窗口内清缓存：避免新推理在 gc / empty_cache 期间进来。
            _free_cuda()
    except _InferenceDrainTimeout as exc:
        logger.error(
            "SenseVoice 卸载已放弃：仍有推理在跑，%s；模型保持加载，稍后空闲回收再试",
            exc,
        )
        return

    logger.info("SenseVoice 已卸载，显存已释放")


# ---------------------------------------------------------------------------
# 分段转写（SenseVoice 没有原生时间戳 -> 用 fsmn-vad 切句后逐段识别）
# ---------------------------------------------------------------------------


def _vad_missing_files() -> list[str]:
    missing: list[str] = []
    for name in VAD_REQUIRED_FILES:
        path = VAD_MODEL_DIR / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    return missing


def _ensure_vad_weights() -> None:
    """准备 fsmn-vad 权重；优先 ModelScope，落到 ``models/vad/fsmn-vad``。"""
    if not _vad_missing_files():
        return
    logger.info("准备 fsmn-vad 权重：%s -> %s", VAD_MODEL_ID, VAD_MODEL_DIR)
    VAD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope import snapshot_download  # noqa: PLC0415 - 故意延迟

        snapshot_download(VAD_MODEL_ID, local_dir=str(VAD_MODEL_DIR))
    except Exception as exc:  # noqa: BLE001
        raise AsrError(
            f"fsmn-vad 权重准备失败（目录 {VAD_MODEL_DIR}）：{type(exc).__name__}: {exc}"
        ) from exc
    missing = _vad_missing_files()
    if missing:
        raise AsrError(f"fsmn-vad 权重不完整，仍缺：{', '.join(missing)}")


def ensure_vad_loaded():
    """懒加载 fsmn-vad；失败抛 AsrError。"""
    global _vad_model, _vad_load_error

    if _vad_model is not None:
        return _vad_model

    with _LOAD_LOCK:
        if _vad_model is not None:
            return _vad_model
        _ensure_vad_weights()
        try:
            from funasr import AutoModel  # noqa: PLC0415 - 故意延迟

            device = _resolve_device(DEVICE)
            logger.info("开始加载 fsmn-vad：dir=%s device=%s", VAD_MODEL_DIR, device)
            _vad_model = AutoModel(
                model=str(VAD_MODEL_DIR),
                trust_remote_code=True,
                disable_update=True,
                device=device,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as exc:  # noqa: BLE001
            _vad_load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("fsmn-vad 加载失败")
            raise AsrError(f"fsmn-vad 加载失败：{_vad_load_error}") from exc
        _vad_load_error = None
        logger.info("fsmn-vad 加载完成")
    return _vad_model


def _parse_vad_value(values: Any, duration: Optional[float]) -> list[tuple[float, float]]:
    """把 funasr VAD 输出归一成 ``[(start_sec, end_sec), ...]``（升序、无重叠）。

    fsmn-vad 的 ``value`` 形如 ``[[0, 2320], [2830, 5250]]``（毫秒）；流式模式
    下可能出现 ``-1`` 表示「还没结束 / 从 0 开始」，这里统一兜底成 0 或音频末尾。
    """
    pairs: list[tuple[float, float]] = []
    for item in values or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            start_ms = float(item[0])
            end_ms = float(item[1])
        except (TypeError, ValueError):
            continue
        start = 0.0 if start_ms < 0 else start_ms / 1000.0
        end = (duration if duration is not None else start) if end_ms < 0 else end_ms / 1000.0
        if end <= start:
            continue
        if duration is not None:
            end = min(end, duration)
            if end <= start:
                continue
        if pairs and start < pairs[-1][1]:
            start = pairs[-1][1]
        if end <= start:
            continue
        pairs.append((round(start, 3), round(end, 3)))
    return pairs


def vad_boundaries(audio_path: str | Path) -> list[tuple[float, float]]:
    """用 fsmn-vad 取语音段落边界（秒）。"""
    path = Path(audio_path)
    if not path.is_file():
        raise AsrError(f"音频文件不存在：{path}")
    vad = ensure_vad_loaded()
    duration = _audio_duration(path)
    try:
        results = vad.generate(
            input=str(path),
            cache={},
            max_single_segment_time=VAD_MAX_SEGMENT_MS,
            max_end_silence_time=VAD_MAX_END_SILENCE_MS,
        )
    except Exception as exc:  # noqa: BLE001
        raise AsrError(f"fsmn-vad 切分失败：{type(exc).__name__}: {exc}") from exc
    values: Any = []
    if results:
        first = results[0]
        if isinstance(first, dict):
            values = first.get("value") or []
        elif isinstance(first, (list, tuple)):
            values = first
    return _parse_vad_value(values, duration)


def _ensure_16k_wav(path: Path) -> tuple[Path, Optional[Path]]:
    """确保是 16k 单声道 wav；返回 ``(可用路径, 需要清理的临时路径)``。"""
    try:
        import soundfile as sf  # noqa: PLC0415

        info = sf.info(str(path))
        if int(info.samplerate) == 16000:
            return path, None
    except Exception:  # noqa: BLE001 - 读不出信息就交给 ffmpeg 兜底
        pass
    try:
        import ffmpeg_tool  # noqa: PLC0415 - engine 目录下的薄封装

        converted = ffmpeg_tool.to_wav16k(str(path))
        return Path(converted["path"]), Path(converted["path"])
    except Exception as exc:  # noqa: BLE001
        raise AsrError(
            f"分段转写需要 16k wav，但转换失败：{type(exc).__name__}: {exc}"
        ) from exc


def transcribe_segments(audio_path: str, language: str = "auto") -> dict[str, Any]:
    """SenseVoice 的分段转写：VAD 切句 + 逐段识别。

    SenseVoice 本身没有时间戳，直接用 VAD 的语音段边界当 ``start/end``：
    段与段之间的静音**不会被并入任何一段**（间隙如实保留），因此
    ``timestamp_source`` 标记为 ``"vad"``。whisper 引擎走原生时间戳，
    标记为 ``"native"``。

    返回::

        {"text": 全文, "language": "zh", "duration": 12.3,
         "segments": [{"start": 0.0, "end": 2.4, "text": "..."}],
         "timestamp_source": "vad", "elapsed": 1.2, "device": "cuda:0"}

    只要有一句识别失败就整体抛 :class:`AsrError`（不返回半截结果）。
    """
    path = Path(audio_path)
    if not path.is_file():
        raise AsrError(f"音频文件不存在：{path}")

    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    started = time.perf_counter()
    work_path, cleanup_path = _ensure_16k_wav(path)
    duration = _audio_duration(work_path)

    boundaries = vad_boundaries(work_path)
    if not boundaries and duration:
        # VAD 一段都没检测到：视为整段静音，返回空分段而不是报错。
        logger.info("VAD 没有检测到语音段：%s（时长 %.2f 秒）", path.name, duration)

    segments: list[dict[str, Any]] = []
    texts: list[str] = []
    detected_language: Optional[str] = None
    temp_dir: Optional[Path] = None

    try:
        if boundaries:
            import soundfile as sf  # noqa: PLC0415

            data, sample_rate = sf.read(str(work_path), dtype="float32", always_2d=True)
            temp_dir = Path(tempfile.mkdtemp(prefix="localai_asr_seg_", dir=None))
            for index, (start, end) in enumerate(boundaries):
                start_index = max(0, int(round(start * sample_rate)))
                end_index = min(len(data), int(round(end * sample_rate)))
                if end_index <= start_index:
                    continue
                clip_path = temp_dir / f"seg_{index:04d}.wav"
                sf.write(str(clip_path), data[start_index:end_index], sample_rate, subtype="PCM_16")
                try:
                    result = transcribe(str(clip_path), language=language)
                except Exception as exc:  # noqa: BLE001 - 统一成整段失败
                    raise AsrError(
                        f"第 {index + 1} 段（{start:.2f}~{end:.2f}s）识别失败："
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                text = str(result.get("text") or "").strip()
                if detected_language is None and result.get("language") not in (None, "nospeech"):
                    detected_language = result.get("language")
                segments.append({"start": start, "end": end, "text": text})
                if text:
                    texts.append(text)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if cleanup_path is not None:
            try:
                cleanup_path.unlink(missing_ok=True)
            except OSError:
                pass

    # 中日韩直接拼，其它语言用空格分隔。
    lang = detected_language or (language if language != "auto" else "zh")
    separator = "" if str(lang).split("-")[0] in ("zh", "yue", "ja", "ko") else " "
    elapsed = time.perf_counter() - started
    return {
        "text": separator.join(texts),
        "language": lang,
        "duration": round(duration, 3) if duration is not None else None,
        "segments": segments,
        "timestamp_source": "vad",
        "elapsed": round(elapsed, 3),
        "device": _loaded_device,
        "vad": {
            "model": "fsmn-vad",
            "model_dir": str(VAD_MODEL_DIR),
            "segments": len(segments),
        },
    }


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

    # 推理路径只拿信号量槽位，不嵌套任何独占锁：
    # - 容量 1：与旧版 _INFER_LOCK 完全一致，同一时刻只有一个推理；
    # - 容量 >1：最多 N 个推理同时进入；
    # - 模型加载放在槽位内：模型为空时槽位持有者负责加载，其它槽位会等
    #   ensure_loaded() 里的加载锁；加载完成前不会有任何真正推理。
    with _INFER_GATE.slot():
        model = ensure_loaded()
        started = time.perf_counter()
        try:
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
