"""文本相关：语言判断、情感参数解析、语速换算。"""

from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import HTTPException

from ..constants import SUPPORTED_LANGUAGES

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def detect_language(text: str) -> str:
    if _CJK_RE.search(text):
        return "zh"
    if _KANA_RE.search(text):
        return "ja"
    if _ARABIC_RE.search(text):
        return "ar"
    return "en"


def normalize_language(language: Optional[str], text: str) -> str:
    if language is None or not language.strip():
        return detect_language(text)
    value = language.strip().lower()
    if value not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的语言 {language!r}，可选：{', '.join(SUPPORTED_LANGUAGES)}",
        )
    return value


def duration_factor_for(speed: float) -> float:
    """API 的 speed 越大越快；节点的 duration_factor 越大越慢，两者互为倒数。"""
    factor = 1.0 / speed
    return max(0.5, min(2.0, round(factor, 3)))


def parse_emotion(value: str) -> dict[str, Any]:
    """emotion 两种写法：

    - 8 个 0~1 的数字（逗号/空格分隔）-> ``emo_vector``；
    - 其它非空文本 -> ``emo_text``（需要 Qwen 情感模型）。
    """
    text = value.strip()
    parts = [p for p in re.split(r"[\s,，]+", text) if p]
    if len(parts) == 8:
        try:
            numbers = [float(p) for p in parts]
        except ValueError:
            numbers = []
        if numbers and all(0.0 <= n <= 1.0 for n in numbers):
            return {"emo_vector": ",".join(f"{n:g}" for n in numbers)}
    return {"emo_text": text, "use_emo_text": True}


def engine_emotion(emotion: Optional[str]) -> tuple[Optional[list[float]], Optional[str]]:
    """把 API 的 emotion 字段拆成引擎要的 (8 维向量, 情感文本)。"""
    if not emotion or not emotion.strip():
        return None, None
    parsed = parse_emotion(emotion)
    if "emo_vector" in parsed:
        values = [float(v) for v in str(parsed["emo_vector"]).split(",") if v.strip()]
        if len(values) == 8:
            return values, None
    return None, parsed.get("emo_text") or emotion.strip()
