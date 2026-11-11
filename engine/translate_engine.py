"""
NLLB-200-distilled-600M 轻量翻译引擎
=====================================

这个文件是「不依赖 ComfyUI / 不依赖 HTTP」的本地 NLLB 翻译入口，供
``E:\\LocalAIService\\src\\localai\\engines\\translate_nllb.py`` 调用。
能力名 ``translate``，和 TTS / ASR 一样：**懒加载 + 空闲卸载 + 显存 LRU**。

设计要点
--------
1. **懒加载**：import 本模块不碰 torch / transformers，也不初始化 CUDA；
   第一次 ``translate()`` 或 ``ensure_loaded()`` 才加载权重。服务启动 0 模型。
2. **权重目录统一**：``LOCALAI_HOME/models/nllb/nllb-200-distilled-600M/``
   （可用 ``LOCALAI_TRANSLATE_DIR`` 覆盖根目录）。下载顺序（先探测，不通再退）：
   a. 目录里已有完整权重 -> 直接用；
   b. ModelScope：先用 HTTP API 列文件，探测
      ``AI-ModelScope/nllb-200-distilled-600M``、``facebook/nllb-200-distilled-600M``
      哪个仓库存在，再按文件下载（国内可达）；
   c. HF 镜像 ``HF_ENDPOINT=https://hf-mirror.com``（直连 HF 已知不通）；
   d. 本机 HuggingFace 缓存 ``models--facebook--nllb-200-distilled-600M``（不联网）。
   每次尝试都写 ``logs/nllb_install.log``，失败原因也写进去。
3. **语言码**：对外用 NLLB / FLORES-200 的码（``zho_Hans`` / ``eng_Latn`` ...），
   入口额外支持 ``zh`` / ``en`` / ``ja`` 这类常见别名（见 ``LANGUAGE_ALIASES``）。
4. **长文本**：先按段落（换行）切，再按句子（中英标点都认）切，超长句再按
   次级标点 / 字符切；每块不超过 ``max_len`` token；翻译后按原段落与换行结构
   拼回（段落不丢、句子不漏）。
5. **fp16**：GPU 上默认 fp16（3070 支持），CPU 回退 fp32；
   ``torch.inference_mode()`` 跑生成；``unload()`` 释放引用 + ``gc`` + 清缓存。
6. **批量**：``translate()`` 内部按 ``batch_size`` 分批推理，500 条短句一次调用
   即可；大批量请走 ``/translate/batch`` 或任务队列。

对外主要函数
------------
- ``model_status()``：权重是否就绪、是否已加载、设备/精度（不触发下载/加载）。
- ``is_loaded()`` / ``ensure_loaded()`` / ``unload()``：加载与卸载。
- ``normalize_lang(code)``：别名 -> NLLB 码；未知抛 :class:`LangError`。
- ``detect_language(text)``：``langdetect`` 自动判断，返回 NLLB 码。
- ``split_plan(text, max_len, count_tokens)``：切分计划（纯函数，可单测）。
- ``render_plan(plan, translations, joiner)``：按原结构拼回（纯函数，可单测）。
- ``translate(texts, src_lang, tgt_lang, ...)``：批量翻译，返回与输入等长的列表。
- ``translate_documents(texts, ...)``：同上，额外返回每条切了多少块。

命令行自检（不启动 HTTP 服务，第一次会下载权重）::

    E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe E:\\LocalAIService\\engine\\translate_engine.py ^
        --text "今天天气不错，我们出去走走吧。" --source zh --target en
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

ENGINE_DIR = Path(__file__).resolve().parent
# 服务根目录（engine 的上一级）；改名或走 junction 都会解析到真实目录。
SERVICE_ROOT = ENGINE_DIR.parent

#: 模型标识（同时也是目录名）。
MODEL_ID = "nllb-200-distilled-600M"

#: 权重根目录：默认 ``<HOME>/models/nllb``。
MODELS_ROOT = Path(
    os.environ.get("LOCALAI_TRANSLATE_DIR", str(SERVICE_ROOT / "models" / "nllb"))
)

#: 具体模型目录：adopt 的精确目录优先，否则 ``<MODELS_ROOT>/nllb-200-distilled-600M/``。
_EXACT_MODEL_DIR = (os.environ.get("LOCALAI_TRANSLATE_MODEL_DIR") or "").strip()
MODEL_DIR = Path(_EXACT_MODEL_DIR) if _EXACT_MODEL_DIR else (MODELS_ROOT / MODEL_ID)

#: 安装日志：``<HOME>/logs/nllb_install.log``（每次尝试都追加）。
LOGS_DIR = Path(os.environ.get("LOCALAI_LOGS_DIR", str(SERVICE_ROOT / "logs")))
INSTALL_LOG = LOGS_DIR / "nllb_install.log"

#: ModelScope 候选仓库（按顺序探测；用 HTTP API 列文件，谁的清单齐全用谁）。
MODELSCOPE_REPOS = (
    "AI-ModelScope/nllb-200-distilled-600M",
    "facebook/nllb-200-distilled-600M",
)

#: HF 仓库名（镜像下载用）。
HF_REPO = "facebook/nllb-200-distilled-600M"

#: 必需的小文件。
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

#: 权重文件二选一（safetensors 优先，ModelScope 镜像是 pytorch_model.bin）。
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

#: 下载时也要拉的额外文件（有就拉，没有不影响）。
OPTIONAL_FILES = (".gitattributes", "README.md")

#: GPU 上是否用 fp16（3070 支持；设 0 可强制 fp32）。
USE_FP16 = (os.environ.get("LOCALAI_TRANSLATE_FP16", "1") or "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

#: 翻译结果最多生成多少个新 token（相对本批输入 token 的倍数，另加 32 余量，夹在 64~1024）。
MAX_NEW_TOKENS_FACTOR = 2.0

logger = logging.getLogger("nllb-translate")

#: 加载 / 推理 / 卸载的统一互斥锁（同一模型不并发生成）。
_ENGINE_LOCK = threading.RLock()

_model: Any = None
_tokenizer: Any = None
_loaded_device: Optional[str] = None
_loaded_dtype: Optional[str] = None
_load_seconds: Optional[float] = None
_load_error: Optional[str] = None
_weights_source: Optional[str] = None
_install_lock = threading.Lock()
_install_logger: Optional[logging.Logger] = None


class TranslateError(RuntimeError):
    """引擎侧可预期错误（缺依赖、下载失败、加载/推理失败）。"""


class LangError(TranslateError):
    """语言码不合法 / 无法检测；``supported`` 里是可用语言。"""

    def __init__(self, message: str, *, supported: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.supported = supported or []


# ---------------------------------------------------------------------------
# 语言码表（NLLB / FLORES-200）
# ---------------------------------------------------------------------------

#: NLLB 语言码 -> 中文名。只收录常用语种（NLLB 官方 200 种，按需再加）。
NLLB_LANGUAGES: dict[str, str] = {
    "zho_Hans": "简体中文",
    "zho_Hant": "繁体中文",
    "yue_Hant": "粤语",
    "eng_Latn": "英语",
    "jpn_Jpan": "日语",
    "kor_Hang": "韩语",
    "fra_Latn": "法语",
    "deu_Latn": "德语",
    "spa_Latn": "西班牙语",
    "rus_Cyrl": "俄语",
    "ara_Arab": "阿拉伯语",
    "por_Latn": "葡萄牙语",
    "ita_Latn": "意大利语",
    "nld_Latn": "荷兰语",
    "pol_Latn": "波兰语",
    "tur_Latn": "土耳其语",
    "ukr_Cyrl": "乌克兰语",
    "vie_Latn": "越南语",
    "tha_Thai": "泰语",
    "ind_Latn": "印尼语",
    "msa_Latn": "马来语",
    "hin_Deva": "印地语",
    "ben_Beng": "孟加拉语",
    "tam_Taml": "泰米尔语",
    "tel_Telu": "泰卢固语",
    "urd_Arab": "乌尔都语",
    "fas_Arab": "波斯语",
    "heb_Hebr": "希伯来语",
    "swe_Latn": "瑞典语",
    "dan_Latn": "丹麦语",
    "nor_Latn": "挪威语",
    "fin_Latn": "芬兰语",
    "ces_Latn": "捷克语",
    "ron_Latn": "罗马尼亚语",
    "hun_Latn": "匈牙利语",
    "ell_Grek": "希腊语",
    "bul_Cyrl": "保加利亚语",
    "srp_Cyrl": "塞尔维亚语",
    "swh_Latn": "斯瓦希里语",
}

#: 入口别名（小写）-> NLLB 码。
LANGUAGE_ALIASES: dict[str, str] = {
    # 中文
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh_cn": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh_hans": "zho_Hans",
    "chs": "zho_Hans",
    "chinese": "zho_Hans",
    "cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh_tw": "zho_Hant",
    "zh-hant": "zho_Hant",
    "zh_hant": "zho_Hant",
    "cht": "zho_Hant",
    "traditional-chinese": "zho_Hant",
    "yue": "yue_Hant",
    "cantonese": "yue_Hant",
    # 英语
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "english": "eng_Latn",
    # 日韩
    "ja": "jpn_Jpan",
    "jp": "jpn_Jpan",
    "japanese": "jpn_Jpan",
    "ko": "kor_Hang",
    "kr": "kor_Hang",
    "korean": "kor_Hang",
    # 欧洲主要语言
    "fr": "fra_Latn",
    "french": "fra_Latn",
    "de": "deu_Latn",
    "german": "deu_Latn",
    "es": "spa_Latn",
    "spanish": "spa_Latn",
    "ru": "rus_Cyrl",
    "russian": "rus_Cyrl",
    "pt": "por_Latn",
    "portuguese": "por_Latn",
    "it": "ita_Latn",
    "italian": "ita_Latn",
    "nl": "nld_Latn",
    "dutch": "nld_Latn",
    "pl": "pol_Latn",
    "polish": "pol_Latn",
    "tr": "tur_Latn",
    "turkish": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ukrainian": "ukr_Cyrl",
    "sv": "swe_Latn",
    "swedish": "swe_Latn",
    "da": "dan_Latn",
    "danish": "dan_Latn",
    "no": "nor_Latn",
    "norwegian": "nor_Latn",
    "fi": "fin_Latn",
    "finnish": "fin_Latn",
    "cs": "ces_Latn",
    "czech": "ces_Latn",
    "ro": "ron_Latn",
    "romanian": "ron_Latn",
    "hu": "hun_Latn",
    "hungarian": "hun_Latn",
    "el": "ell_Grek",
    "greek": "ell_Grek",
    "bg": "bul_Cyrl",
    "bulgarian": "bul_Cyrl",
    "sr": "srp_Cyrl",
    "serbian": "srp_Cyrl",
    # 亚洲其它
    "vi": "vie_Latn",
    "vietnamese": "vie_Latn",
    "th": "tha_Thai",
    "thai": "tha_Thai",
    "id": "ind_Latn",
    "indonesian": "ind_Latn",
    "ms": "msa_Latn",
    "malay": "msa_Latn",
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    "bn": "ben_Beng",
    "bengali": "ben_Beng",
    "ta": "tam_Taml",
    "tamil": "tam_Taml",
    "te": "tel_Telu",
    "telugu": "tel_Telu",
    "ur": "urd_Arab",
    "urdu": "urd_Arab",
    # 中东 / 非洲
    "ar": "ara_Arab",
    "arabic": "ara_Arab",
    "fa": "fas_Arab",
    "persian": "fas_Arab",
    "he": "heb_Hebr",
    "hebrew": "heb_Hebr",
    "sw": "swh_Latn",
    "swahili": "swh_Latn",
}

#: ``langdetect`` 返回的 ISO639-1（可能带地区）-> NLLB 码。
#: auto 只覆盖这张表里的常见语种，表外的检测结果会明确报错让调用方显式给 source。
ISO_TO_NLLB: dict[str, str] = {
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "zh": "zho_Hans",
    "yue": "yue_Hant",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ar": "ara_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "msa_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "fa": "fas_Arab",
    "he": "heb_Hebr",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nor_Latn",
    "fi": "fin_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "bg": "bul_Cyrl",
    "sr": "srp_Cyrl",
    "sw": "swh_Latn",
}

#: 目标语言不需要空格分隔句子（中文/日文/粤语）。
_NO_SPACE_TARGETS = {"zho_Hans", "zho_Hant", "yue_Hant", "jpn_Jpan"}


def supported_codes() -> list[str]:
    """全部可用的 NLLB 语言码（排序后）。"""
    return sorted(NLLB_LANGUAGES)


def supported_aliases() -> list[str]:
    """全部常见别名（排序后，文档/错误信息用）。"""
    return sorted(LANGUAGE_ALIASES)


def language_table() -> list[dict[str, str]]:
    """给 /translate/status、/capabilities 用的语言表。"""
    return [
        {"code": code, "name": name}
        for code, name in sorted(NLLB_LANGUAGES.items())
    ]


def normalize_lang(value: Optional[str], *, allow_auto: bool = False) -> str:
    """把语言别名归一成 NLLB 码；不认识时抛 :class:`LangError`。

    ``allow_auto=True`` 时，``auto`` 会原样返回 ``"auto"``（由调用方再做检测）。
    """
    raw = (value or "").strip()
    if not raw:
        raise LangError(
            "语言不能为空；可用 NLLB 码（如 zho_Hans/eng_Latn）或常见别名（zh/en/ja）",
            supported=supported_codes(),
        )
    if raw.lower() == "auto":
        if allow_auto:
            return "auto"
        raise LangError("这里不接受 auto，请显式指定语言", supported=supported_codes())
    if raw in NLLB_LANGUAGES:
        return raw
    lowered = raw.lower()
    if lowered in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[lowered]
    # 也接受 zho-hans / eng-latn 这类写法
    parts = re.split(r"[-_]", lowered)
    if len(parts) == 2:
        candidate = f"{parts[0]}_{parts[1].capitalize()}"
        if candidate in NLLB_LANGUAGES:
            return candidate
    raise LangError(
        f"不支持的语言 {value!r}；可用 NLLB 码：{', '.join(supported_codes())}；"
        f"或常见别名：{', '.join(supported_aliases())}",
        supported=supported_codes(),
    )


def _detect_raw(text: str) -> list[tuple[str, float]]:
    """调用 langdetect，返回 [(ISO 码, 概率)]；失败抛 :class:`LangError`。"""
    sample = (text or "").strip()
    if not sample:
        raise LangError("文本为空，无法自动检测语言；请显式指定 source")
    try:
        from langdetect import DetectorFactory, LangDetectException, detect_langs
    except Exception as exc:  # noqa: BLE001 - 依赖没装
        raise LangError(
            "未安装语言检测库 langdetect，无法使用 source=auto；"
            "请 pip install langdetect 或显式指定 source",
            supported=supported_codes(),
        ) from exc
    # 固定种子，保证同一段文本每次检测结果一致。
    DetectorFactory.seed = 0
    try:
        results = detect_langs(sample[:4000])
    except LangDetectException as exc:
        raise LangError(
            f"语言检测失败（文本太短或几乎没有有效字符）：{exc}；请显式指定 source"
        ) from exc
    return [(str(item.lang).lower(), float(item.prob)) for item in results]


def detect_language(text: str) -> str:
    """自动检测语言并返回 NLLB 码。

    只覆盖 :data:`ISO_TO_NLLB` 里的常见语种；检测到表外语种时抛
    :class:`LangError`，提示调用方显式给 ``source``。
    """
    results = _detect_raw(text)
    for iso, _prob in results:
        code = ISO_TO_NLLB.get(iso)
        if code is not None:
            return code
    detected = ", ".join(f"{iso}({prob:.0%})" for iso, prob in results) or "未知"
    raise LangError(
        f"检测到的语言不在支持列表里（{detected}）；"
        "auto 只覆盖常见语种，请显式指定 source。"
        f"可用 NLLB 码：{', '.join(supported_codes())}",
        supported=supported_codes(),
    )


# ---------------------------------------------------------------------------
# 长文本切分 / 拼回（纯函数，不依赖模型）
# ---------------------------------------------------------------------------

#: 句子结束标点（中英都认）。
_SENTENCE_END = set("。！？!?；;…")
#: 紧跟结束标点的收尾符号（引号/括号），归到同一句。
_CLOSERS = set("”’\"'」』）)]】》〉")
#: 超长句再切的次级标点。
_CLAUSE_END = set("，,、：:—–-")


@dataclass
class Chunk:
    """一个待翻译块。"""

    text: str
    #: True = 这是同一个句子被硬切开后的续块，拼回时不要加空格。
    continuation: bool = False


@dataclass
class Block:
    """段结构：``kind='text'`` 是一个段落，``kind='sep'`` 是段落之间的换行。"""

    kind: str
    text: str = ""
    chunks: list[Chunk] = field(default_factory=list)


def _split_sentences(paragraph: str) -> list[tuple[str, str]]:
    """把一段文本切成 ``[(句子, 句子后面的空白), ...]``，不丢任何字符。"""
    sentences: list[tuple[str, str]] = []
    start = 0
    index = 0
    length = len(paragraph)
    while index < length:
        char = paragraph[index]
        boundary = False
        end = index
        if char in _SENTENCE_END:
            end = index + 1
            while end < length and paragraph[end] in _CLOSERS:
                end += 1
            boundary = True
        elif char == ".":
            # 英文句号：只在后面是空白/结尾时切，避免切开 3.14 / e.g.
            end = index + 1
            while end < length and paragraph[end] in _CLOSERS:
                end += 1
            if end >= length or paragraph[end].isspace():
                boundary = True
        if boundary:
            gap = ""
            cursor = end
            while cursor < length and paragraph[cursor].isspace():
                gap += paragraph[cursor]
                cursor += 1
            sentences.append((paragraph[start:end], gap))
            start = cursor
            index = cursor
            continue
        index += 1
    if start < length:
        sentences.append((paragraph[start:], ""))
    return [(sentence, gap) for sentence, gap in sentences if sentence]


def _count_default(text: str) -> int:
    """没有 tokenizer 时的保守估算：按字符数算 token 上界。"""
    return len(text)


def _hard_split(
    text: str, max_len: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """把超长文本按「不超过 max_len token」硬切成若干块（尽量在字符边界）。"""
    if max_len <= 0:
        return [text]
    pieces: list[str] = []
    rest = text
    while rest:
        if count_tokens(rest) <= max_len:
            pieces.append(rest)
            break
        # 二分找最长合法前缀；计数器对长度单调，足够用。
        low, high = 1, len(rest)
        best = 1
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(rest[:middle]) <= max_len:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        # 尽量在空白/标点处断开，读起来更自然。
        cut = best
        window_start = max(1, best - 24)
        for position in range(best, window_start - 1, -1):
            if rest[position - 1].isspace() or rest[position - 1] in "，,、：:。.!?！？;；":
                cut = position
                break
        cut = max(1, min(cut, len(rest)))
        pieces.append(rest[:cut])
        rest = rest[cut:]
    return pieces


def _split_oversized(
    sentence: str, max_len: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """超长句：先按次级标点（逗号/冒号/顿号）切，再对仍超长的块硬切。"""
    if count_tokens(sentence) <= max_len:
        return [sentence]
    clauses: list[str] = []
    buffer = ""
    for char in sentence:
        buffer += char
        if char in _CLAUSE_END:
            clauses.append(buffer)
            buffer = ""
    if buffer:
        clauses.append(buffer)

    pieces: list[str] = []
    current = ""
    for clause in clauses:
        if count_tokens(clause) > max_len:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(clause, max_len, count_tokens))
            continue
        if current and count_tokens(current + clause) > max_len:
            pieces.append(current)
            current = clause
        else:
            current += clause
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece]


def split_plan(
    text: str,
    max_len: int = 400,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> list[Block]:
    """把一段文本切成可翻译的块，同时保留段落与换行结构。

    返回 :class:`Block` 列表：
    - ``kind='sep'``：段落之间的换行（原样保留，不翻译）；
    - ``kind='text'``：一个段落，内部 ``chunks`` 依次对应翻译结果。

    拼回用 :func:`render_plan`。``count_tokens`` 不传时按字符数保守估算；
    真正加载模型后会传入 tokenizer 计数。
    """
    counter = count_tokens or _count_default
    max_len = max(1, int(max_len))
    blocks: list[Block] = []

    for piece in re.split(r"(\n+)", text or ""):
        if not piece:
            continue
        if piece.strip("\r\n") == "" and "\n" in piece:
            # 连续换行：并到上一个分隔块里，保持原始换行数。
            if blocks and blocks[-1].kind == "sep":
                blocks[-1].text += piece
            else:
                blocks.append(Block(kind="sep", text=piece))
            continue

        chunks: list[Chunk] = []
        current = ""
        # 上一个句子后面的原始空白：并句时补回去，免得把 "Hello. World."
        # 拼成 "Hello.World." 再喂给模型。
        pending_gap = ""
        for sentence, gap in _split_sentences(piece):
            parts = _split_oversized(sentence, max_len, counter)
            for part_index, part in enumerate(parts):
                tail = gap if part_index == len(parts) - 1 else ""
                if not current:
                    current = part
                    pending_gap = tail
                    continue
                if counter(current + pending_gap + part) <= max_len:
                    current += pending_gap + part
                    pending_gap = tail
                    continue
                chunks.append(Chunk(text=current))
                current = part
                pending_gap = tail
        if current:
            chunks.append(Chunk(text=current))
        if chunks:
            blocks.append(Block(kind="text", chunks=chunks))
    return blocks


def plan_chunks(plan: Sequence[Block]) -> list[str]:
    """展开成按顺序排列的待翻译块文本列表。"""
    return [chunk.text for block in plan if block.kind == "text" for chunk in block.chunks]


def render_plan(
    plan: Sequence[Block],
    translations: Sequence[str],
    *,
    joiner: str = " ",
) -> str:
    """把译文按原结构拼回。

    ``translations`` 必须与 :func:`plan_chunks` 一一对应、条数相同。
    段落之间的换行原样保留；同一段内被切开的块用 ``joiner`` 连接
    （中文/日文目标建议传空字符串，避免句间多出空格）。
    """
    expected = sum(len(block.chunks) for block in plan if block.kind == "text")
    if len(translations) != expected:
        raise TranslateError(
            f"拼回失败：译文 {len(translations)} 条，切分块 {expected} 块，数量不一致"
        )
    output: list[str] = []
    cursor = 0
    for block in plan:
        if block.kind == "sep":
            output.append(block.text)
            continue
        for chunk_index, _chunk in enumerate(block.chunks):
            text = str(translations[cursor]).strip()
            cursor += 1
            if chunk_index == 0:
                output.append(text)
            elif text:
                output.append(joiner + text)
    return "".join(output)


def joiner_for(target_lang: str) -> str:
    """目标语言拼回时的块间连接符（中文/日文不加空格）。"""
    return "" if target_lang in _NO_SPACE_TARGETS else " "


# ---------------------------------------------------------------------------
# 权重准备（本地 -> ModelScope -> HF 镜像 -> 本机 HF 缓存）
# ---------------------------------------------------------------------------


def _install_log() -> logging.Logger:
    """带文件输出的安装日志（logs/nllb_install.log）。"""
    global _install_logger
    if _install_logger is not None:
        return _install_logger
    with _install_lock:
        if _install_logger is not None:
            return _install_logger
        target = logging.getLogger("nllb-install")
        target.setLevel(logging.INFO)
        target.propagate = True
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(INSTALL_LOG, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            target.addHandler(handler)
        except OSError as exc:
            logger.warning("无法写安装日志 %s：%s", INSTALL_LOG, exc)
        _install_logger = target
    return target


def _missing_files(target: Path = MODEL_DIR) -> list[str]:
    """返回权重目录里缺失或为空的必需文件。"""
    missing: list[str] = []
    for name in MODEL_FILES:
        path = target / name
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                missing.append(name)
        except OSError:
            missing.append(name)
    if not any((target / name).is_file() and (target / name).stat().st_size > 0 for name in WEIGHT_FILES):
        missing.append(f"{WEIGHT_FILES[0]}(或 {WEIGHT_FILES[1]})")
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
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _http_json(url: str, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "localai/translate"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _modelscope_file_list(repo: str) -> list[dict[str, Any]]:
    """用 ModelScope HTTP API 列出仓库文件（不下载）。"""
    url = (
        f"https://modelscope.cn/api/v1/models/{repo}/repo/files"
        "?Revision=master&Recursive=true"
    )
    payload = _http_json(url)
    files = ((payload or {}).get("Data") or {}).get("Files") or []
    if not isinstance(files, list) or not files:
        raise TranslateError(f"ModelScope 仓库 {repo} 返回的文件列表为空")
    return [item for item in files if isinstance(item, dict) and item.get("Path")]


def _download_file(url: str, target: Path, expected_size: int = 0) -> None:
    """下载单个文件到 target（先写 .part，校验大小后原子替换）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    log = _install_log()
    request = urllib.request.Request(url, headers={"User-Agent": "localai/translate"})
    started = time.perf_counter()
    downloaded = 0
    last_report = 0
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as sink:
        total = int(response.headers.get("Content-Length") or expected_size or 0)
        log.info(
            "下载 %s（预计 %.1f MB）", target.name, total / (1024 * 1024) if total else 0
        )
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            sink.write(block)
            downloaded += len(block)
            if total and downloaded - last_report >= total // 10:
                last_report = downloaded
                log.info(
                    "  %s 进度 %.0f%%（%.1f/%.1f MB）",
                    target.name,
                    downloaded * 100 / total,
                    downloaded / (1024 * 1024),
                    total / (1024 * 1024),
                )
    if expected_size and part.stat().st_size != expected_size:
        raise TranslateError(
            f"{target.name} 下载大小不符：期望 {expected_size} 字节，实际 {part.stat().st_size}"
        )
    final_size = part.stat().st_size
    os.replace(part, target)
    log.info(
        "下载完成 %s（%.1f MB，%.1f 秒）",
        target.name,
        final_size / (1024 * 1024),
        time.perf_counter() - started,
    )


def _download_from_modelscope(target: Path) -> str:
    """探测 ModelScope 候选仓库并下载；返回实际使用的仓库名。"""
    log = _install_log()
    reasons: list[str] = []
    for repo in MODELSCOPE_REPOS:
        log.info("尝试 ModelScope 仓库：%s", repo)
        try:
            files = _modelscope_file_list(repo)
        except Exception as exc:  # noqa: BLE001 - 探测失败就换下一个
            reason = f"{type(exc).__name__}: {exc}"
            reasons.append(f"{repo}: {reason}")
            log.warning("ModelScope 仓库 %s 不可用：%s", repo, reason)
            continue

        available = {str(item["Path"]): item for item in files}
        has_weight = [name for name in WEIGHT_FILES if name in available]
        missing = [name for name in MODEL_FILES if name not in available]
        if not has_weight or missing:
            reason = f"缺少文件：{', '.join(missing + ([] if has_weight else list(WEIGHT_FILES)))}"
            reasons.append(f"{repo}: {reason}")
            log.warning("ModelScope 仓库 %s 清单不完整：%s", repo, reason)
            continue

        log.info(
            "ModelScope 仓库 %s 可用：%d 个文件，权重 %s",
            repo,
            len(available),
            has_weight[0],
        )
        target.mkdir(parents=True, exist_ok=True)
        # 先下小文件再下大权重：小文件失败可以立刻换源，不浪费带宽。
        ordered = [name for name in MODEL_FILES if name in available]
        ordered += [name for name in has_weight]
        ordered += [name for name in OPTIONAL_FILES if name in available]
        try:
            for name in ordered:
                item = available[name]
                url = f"https://modelscope.cn/models/{repo}/resolve/master/{name}"
                _download_file(
                    url,
                    target / name,
                    expected_size=int(item.get("Size") or 0),
                )
        except Exception as exc:  # noqa: BLE001 - 下载中断同样换源
            reason = f"{type(exc).__name__}: {exc}"
            reasons.append(f"{repo}: {reason}")
            log.error("从 ModelScope %s 下载失败：%s", repo, reason)
            continue
        if not _missing_files(target):
            log.info("ModelScope 下载成功：%s -> %s", repo, target)
            return f"modelscope:{repo}"
        reasons.append(f"{repo}: 下载后仍缺 {', '.join(_missing_files(target))}")
    raise TranslateError("；".join(reasons) or "ModelScope 没有可用仓库")


def _download_from_hf_mirror(target: Path) -> str:
    """从 HF 镜像（hf-mirror.com）下载；直连 HF 已知不通。"""
    if not _has_module("huggingface_hub"):
        raise TranslateError("未安装 huggingface_hub，无法从 HF 镜像下载")
    endpoint = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as hub_constants  # noqa: PLC0415

        hub_constants.ENDPOINT = endpoint
    except Exception:  # noqa: BLE001 - 改不了常量就只靠环境变量
        pass
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    log = _install_log()
    log.info("尝试 HF 镜像：%s（仓库 %s）", endpoint, HF_REPO)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(target),
        max_workers=4,
        allow_patterns=list(MODEL_FILES) + list(WEIGHT_FILES),
    )
    return f"hf-mirror:{endpoint}"


def _copy_from_hf_cache(target: Path) -> Optional[str]:
    """本机 HF 缓存里有就复制，不联网。"""
    repo_dir_name = "models--facebook--nllb-200-distilled-600M"
    roots: list[Path] = []
    hub_cache = (os.environ.get("HF_HUB_CACHE") or "").strip()
    env_home = (os.environ.get("HF_HOME") or "").strip()
    if hub_cache:
        roots.append(Path(hub_cache))
    if env_home:
        roots.append(Path(env_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    candidates: list[Path] = []
    for root in roots:
        repo_dir = root / repo_dir_name
        snapshots = repo_dir / "snapshots"
        try:
            if snapshots.is_dir():
                candidates.extend(sorted(snapshots.iterdir(), reverse=True))
            elif repo_dir.is_dir():
                candidates.append(repo_dir)
        except OSError:
            continue

    log = _install_log()
    for candidate in candidates:
        if _missing_files(candidate):
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in list(MODEL_FILES):
            source = candidate / name
            if source.is_file():
                shutil.copy2(source, target / name)
        for name in WEIGHT_FILES:
            source = candidate / name
            if source.is_file():
                shutil.copy2(source, target / name)
                break
        if not _missing_files(target):
            log.info("使用本机 HF 缓存：%s", candidate)
            return "hf-cache"
    return None


def ensure_weights() -> dict[str, Any]:
    """确保权重在 ``<MODEL_DIR>``；返回状态字典。

    只准备权重，不加载模型。失败抛 :class:`TranslateError`，错误信息里
    列出每种来源的失败原因；全过程写 ``logs/nllb_install.log``。
    """
    global _weights_source

    log = _install_log()
    missing = _missing_files(MODEL_DIR)
    if not missing:
        _weights_source = _weights_source or "local"
        return {
            "ready": True,
            "model_dir": str(MODEL_DIR),
            "missing": [],
            "source": "local",
            "size_mb": _dir_size_mb(MODEL_DIR),
        }

    log.info("=" * 60)
    log.info(
        "NLLB 权重准备开始：目录=%s 缺失=%s", MODEL_DIR, ", ".join(missing)
    )
    logger.info("NLLB 权重不完整（缺 %s），开始下载：%s", ", ".join(missing), MODEL_DIR)
    reasons: list[str] = []
    source: Optional[str] = None

    # 1) ModelScope（国内可达，优先）
    try:
        source = _download_from_modelscope(MODEL_DIR)
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        reasons.append(f"ModelScope：{reason}")
        log.warning("ModelScope 全部候选失败：%s", reason)

    # 2) HF 镜像
    if source is None:
        try:
            source = _download_from_hf_mirror(MODEL_DIR)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            reasons.append(f"HF 镜像：{reason}")
            log.warning("HF 镜像下载失败：%s", reason)

    # 3) 本机 HF 缓存
    if source is None:
        try:
            source = _copy_from_hf_cache(MODEL_DIR)
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            reasons.append(f"HF 本地缓存：{reason}")
            log.warning("HF 本地缓存复制失败：%s", reason)

    missing = _missing_files(MODEL_DIR)
    if missing or source is None:
        detail = "；".join(reasons) or "未知原因"
        log.error("NLLB 权重准备失败：仍缺 %s；%s", ", ".join(missing), detail)
        raise TranslateError(
            f"NLLB 权重准备失败（目录 {MODEL_DIR} 仍缺 {', '.join(missing) or '文件'}）：{detail}"
        )

    _weights_source = source
    log.info(
        "NLLB 权重就绪：%s（来源 %s，%.1f MB）",
        MODEL_DIR,
        source,
        _dir_size_mb(MODEL_DIR),
    )
    return {
        "ready": True,
        "model_dir": str(MODEL_DIR),
        "missing": [],
        "source": source,
        "size_mb": _dir_size_mb(MODEL_DIR),
    }


# ---------------------------------------------------------------------------
# 状态 / 加载 / 卸载
# ---------------------------------------------------------------------------


def model_status() -> dict[str, Any]:
    """给 /health、/translate/status、/capabilities 用的状态；不触发下载或加载。"""
    missing = _missing_files(MODEL_DIR)
    return {
        "ready": not missing,
        "model": MODEL_ID,
        "model_dir": str(MODEL_DIR),
        "missing": missing,
        "size_mb": _dir_size_mb(MODEL_DIR) if MODEL_DIR.is_dir() else 0.0,
        "weights_source": _weights_source,
        "loaded": _model is not None,
        "device": _loaded_device,
        "dtype": _loaded_dtype,
        "fp16": USE_FP16,
        "load_seconds": round(_load_seconds, 2) if _load_seconds is not None else None,
        "load_error": _load_error,
        "package": _package_version("transformers"),
        "languages": len(NLLB_LANGUAGES),
        "install_log": str(INSTALL_LOG),
    }


def is_loaded() -> bool:
    """当前是否已加载模型。"""
    return _model is not None


def _log_vram(tag: str) -> None:
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "[显存] %s：torch 已分配 %.2f GB / 已保留 %.2f GB；系统空闲 %.2f / %.2f GB",
            tag,
            allocated,
            reserved,
            free / (1024 ** 3),
            total / (1024 ** 3),
        )
    except Exception:  # noqa: BLE001 - 显存日志失败不影响翻译
        pass


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _resolve_device() -> str:
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch 不可用，翻译回退 CPU：%s", exc)
        return "cpu"


def ensure_loaded() -> tuple[Any, Any]:
    """加载并返回 ``(model, tokenizer)``；已加载时直接复用。"""
    global _model, _tokenizer, _loaded_device, _loaded_dtype
    global _load_seconds, _load_error

    with _ENGINE_LOCK:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer

        if not _has_module("transformers"):
            raise TranslateError(
                "未安装 transformers；请用 engine/.venv/Scripts/python.exe "
                "-m pip install transformers==4.49.0"
            )
        info = ensure_weights()
        device = _resolve_device()
        if device == "cuda" and not USE_FP16:
            dtype_name = "float32"
        elif device == "cuda":
            dtype_name = "float16"
        else:
            dtype_name = "float32"

        started = time.perf_counter()
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

            dtype = torch.float16 if dtype_name == "float16" else torch.float32
            logger.info(
                "开始加载 NLLB：model_dir=%s device=%s dtype=%s",
                info["model_dir"],
                device,
                dtype_name,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                info["model_dir"], src_lang="eng_Latn", local_files_only=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                info["model_dir"],
                torch_dtype=dtype,
                local_files_only=True,
            )
            model.to(device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - 统一包成 TranslateError
            _load_error = f"{type(exc).__name__}: {exc}"
            _model = None
            _tokenizer = None
            _loaded_device = None
            _loaded_dtype = None
            logger.exception("NLLB 加载失败")
            _free_cuda()
            raise TranslateError(f"NLLB 加载失败：{_load_error}") from exc

        _model = model
        _tokenizer = tokenizer
        _loaded_device = device
        _loaded_dtype = dtype_name
        _load_seconds = time.perf_counter() - started
        _load_error = None
        logger.info(
            "NLLB 加载完成，耗时 %.2f 秒（device=%s dtype=%s）",
            _load_seconds,
            device,
            dtype_name,
        )
        _log_vram("NLLB 加载完成")
        return _model, _tokenizer


def unload() -> None:
    """释放模型和显存；重复调用安全。"""
    global _model, _tokenizer, _loaded_device, _loaded_dtype
    global _load_seconds, _load_error

    with _ENGINE_LOCK:
        model = _model
        _model = None
        _tokenizer = None
        _loaded_device = None
        _loaded_dtype = None
        _load_seconds = None
        _load_error = None
        if model is not None:
            try:
                del model
            except Exception:  # noqa: BLE001
                pass
        _free_cuda()
    logger.info("NLLB 已卸载，显存已释放")


# ---------------------------------------------------------------------------
# 翻译
# ---------------------------------------------------------------------------


def _token_counter(tokenizer: Any) -> Callable[[str], int]:
    """用真 tokenizer 计数的回调（每次调用都取一遍，保证准确）。"""

    def count(text: str) -> int:
        try:
            return len(tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:  # noqa: BLE001 - 个别空串/怪字符退化为字符数
            return len(text)

    return count


def _translate_chunks(
    chunks: Sequence[str],
    src_lang: str,
    tgt_lang: str,
    *,
    batch_size: int,
    num_beams: int,
    max_len: int,
) -> list[str]:
    """真正跑模型：把待翻译块分批生成。调用方必须持有 ``_ENGINE_LOCK``。"""
    model, tokenizer = ensure_loaded()
    import torch  # noqa: PLC0415

    device = _loaded_device or "cpu"
    tokenizer.src_lang = src_lang
    try:
        target_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    except Exception as exc:  # noqa: BLE001
        raise LangError(f"目标语言码 {tgt_lang!r} 不被分词器接受：{exc}") from exc
    if target_token_id is None or int(target_token_id) < 0:
        raise LangError(
            f"目标语言码 {tgt_lang!r} 不被 NLLB 分词器接受；"
            f"可用：{', '.join(supported_codes())}"
        )

    batch_size = max(1, min(64, int(batch_size)))
    num_beams = max(1, min(8, int(num_beams)))
    results: list[str] = []
    use_cuda = device.startswith("cuda")

    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        # 生成长度按本批真实输入长度算，而不是固定 800：固定值会让短句也按最长
        # 序列预留 KV cache、多跑解码步，冷启动和显存都吃亏。2 倍 + 32 余量，
        # 夹在 64~1024，英文输出 token 数通常与中文输入同量级，质量不受影响。
        in_tokens = int(encoded["input_ids"].shape[1])
        max_new_tokens = int(
            max(64, min(1024, int(in_tokens * MAX_NEW_TOKENS_FACTOR) + 32))
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=int(target_token_id),
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=num_beams > 1,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(str(text).strip() for text in decoded)
        # Windows/WDDM 下 torch 的 reserved 不会主动归还，批与批之间不回收就会
        # 随形状变化一路滚大（曾到过 16868MB）。每批结束释放中间张量并清缓存。
        del encoded, generated
        if use_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - 回收失败不该让翻译失败
                pass

    if len(results) != len(chunks):
        raise TranslateError(
            f"推理结果条数不符：输入 {len(chunks)} 块，输出 {len(results)} 条"
        )
    return results


def _resolve_languages(
    texts: Sequence[str], src_lang: str, tgt_lang: str
) -> tuple[str, str]:
    """解析源/目标语言；``source=auto`` 时用第一条非空文本检测（不加载模型）。"""
    source = normalize_lang(src_lang, allow_auto=True)
    target = normalize_lang(tgt_lang)
    if source == "auto":
        sample = next((str(text) for text in texts if str(text or "").strip()), "")
        source = detect_language(sample)
    return source, target


def translate(
    texts: Sequence[str],
    src_lang: str = "auto",
    tgt_lang: str = "eng_Latn",
    max_len: int = 400,
    batch_size: int = 8,
    num_beams: int = 2,
) -> list[str]:
    """批量翻译，返回与 ``texts`` 等长的译文列表。

    - ``src_lang`` 支持 NLLB 码、常见别名，或 ``auto``（自动检测第一条非空文本）；
    - 每条文本内部按段落/句子切分，各自不超过 ``max_len`` token；
    - 返回顺序与输入一致，空字符串原样返回。
    """
    source, target = _resolve_languages(texts, src_lang, tgt_lang)
    if source == target:
        return [str(text or "") for text in texts]

    with _ENGINE_LOCK:
        _model_ref, tokenizer = ensure_loaded()
        counter = _token_counter(tokenizer)

        plans: list[list[Block]] = []
        pending: list[str] = []
        for text in texts:
            plan = split_plan(str(text or ""), max_len=max_len, count_tokens=counter)
            plans.append(plan)
            pending.extend(plan_chunks(plan))

        joiner = joiner_for(target)
        translated_chunks = (
            _translate_chunks(
                pending,
                source,
                target,
                batch_size=batch_size,
                num_beams=num_beams,
                max_len=max_len,
            )
            if pending
            else []
        )

        rendered: list[str] = []
        cursor = 0
        for plan in plans:
            count = sum(len(block.chunks) for block in plan if block.kind == "text")
            piece = translated_chunks[cursor : cursor + count]
            cursor += count
            rendered.append(render_plan(plan, piece, joiner=joiner))
        return rendered


def translate_documents(
    texts: Sequence[str],
    src_lang: str = "auto",
    tgt_lang: str = "eng_Latn",
    max_len: int = 400,
    batch_size: int = 8,
    num_beams: int = 2,
) -> list[dict[str, Any]]:
    """同 :func:`translate`，但返回 ``[{"text": 译文, "chunks": 块数}]``。"""
    source, target = _resolve_languages(texts, src_lang, tgt_lang)
    if source == target:
        return [
            {"text": str(text or ""), "chunks": 1, "source": source, "target": target}
            for text in texts
        ]

    with _ENGINE_LOCK:
        _model_ref, tokenizer = ensure_loaded()
        counter = _token_counter(tokenizer)

        plans = [
            split_plan(str(text or ""), max_len=max_len, count_tokens=counter)
            for text in texts
        ]
        pending = [chunk for plan in plans for chunk in plan_chunks(plan)]
        joiner = joiner_for(target)
        translated_chunks = (
            _translate_chunks(
                pending,
                source,
                target,
                batch_size=batch_size,
                num_beams=num_beams,
                max_len=max_len,
            )
            if pending
            else []
        )
        documents: list[dict[str, Any]] = []
        cursor = 0
        for plan in plans:
            count = sum(len(block.chunks) for block in plan if block.kind == "text")
            piece = translated_chunks[cursor : cursor + count]
            cursor += count
            documents.append(
                {
                    "text": render_plan(plan, piece, joiner=joiner),
                    "chunks": count,
                    "source": source,
                    "target": target,
                }
            )
        return documents


# ---------------------------------------------------------------------------
# 命令行自检
# ---------------------------------------------------------------------------


def _main() -> int:
    parser = argparse.ArgumentParser(description="NLLB-200 独立翻译引擎命令行自检")
    parser.add_argument("--text", default="今天天气不错，我们出去走走吧。")
    parser.add_argument("--source", default="auto", help="auto / zh / zho_Hans / en ...")
    parser.add_argument("--target", default="en", help="en / eng_Latn / ja ...")
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beams", type=int, default=2)
    parser.add_argument("--download-only", action="store_true", help="只准备权重，不加载")
    parser.add_argument("--unload", action="store_true", help="翻译后立即卸载")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("模型状态：", json.dumps(model_status(), ensure_ascii=False, indent=2))
    if args.download_only:
        print("权重准备：", json.dumps(ensure_weights(), ensure_ascii=False, indent=2))
        return 0
    started = time.perf_counter()
    docs = translate_documents(
        [args.text],
        args.source,
        args.target,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_beams=args.beams,
    )
    elapsed = time.perf_counter() - started
    print("耗时（含首次加载）：%.2f 秒" % elapsed)
    print("结果：", json.dumps(docs, ensure_ascii=False, indent=2))
    if args.unload:
        unload()
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
