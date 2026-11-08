"""翻译接口的公共参数校验（同步接口与任务队列共用）。

只做「与服务层协议相关」的检查，不 import 引擎、不加载模型：
- 单条文本上限 :data:`TRANSLATE_MAX_TEXT`（5000 字）；
- 单批条数上限 :data:`TRANSLATE_BATCH_MAX`（500 条）；
- 单批总字符上限 :data:`TRANSLATE_BATCH_CHARS`（200k）。

超限统一抛 :class:`ValueError`，路由转成 HTTP 400（带中文说明）。
"""

from __future__ import annotations

from typing import Any, Sequence

from ..constants import (
    TRANSLATE_BATCH_CHARS,
    TRANSLATE_BATCH_MAX,
    TRANSLATE_ENGINES,
    TRANSLATE_MAX_TEXT,
)


def normalize_engine(value: Any) -> str:
    """归一翻译引擎名；目前只支持 ``nllb``（空值按默认）。"""
    name = str(value or "").strip().lower() or "nllb"
    if name in ("default", "auto"):
        return "nllb"
    if name not in TRANSLATE_ENGINES:
        raise ValueError(
            f"不支持的翻译引擎 {value!r}；可选：{', '.join(TRANSLATE_ENGINES)}"
        )
    return name


def normalize_texts(texts: Any, *, single: bool = False) -> list[str]:
    """校验并归一待翻译文本列表；返回 ``list[str]``。"""
    if single:
        if not isinstance(texts, str):
            raise ValueError("text 必须是字符串")
        values: list[Any] = [texts]
    else:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise ValueError("texts 必须是字符串数组")
        values = list(texts)

    if not values:
        raise ValueError("texts 不能为空")
    if len(values) > TRANSLATE_BATCH_MAX:
        raise ValueError(
            f"texts 条数超限：{len(values)} 条，单批最多 {TRANSLATE_BATCH_MAX} 条；"
            "更大规模请用 POST /jobs/translate 进任务队列"
        )

    normalized: list[str] = []
    total_chars = 0
    for index, item in enumerate(values):
        if not isinstance(item, str):
            raise ValueError(f"texts[{index}] 必须是字符串，收到 {type(item).__name__}")
        if single and not item.strip():
            raise ValueError("text 不能为空")
        if len(item) > TRANSLATE_MAX_TEXT:
            raise ValueError(
                f"texts[{index}] 太长：{len(item)} 字，单条上限 {TRANSLATE_MAX_TEXT} 字；"
                "长文请拆成多条或走 /jobs/translate"
            )
        total_chars += len(item)
        normalized.append(item)

    if total_chars > TRANSLATE_BATCH_CHARS:
        raise ValueError(
            f"texts 总量超限：{total_chars} 字，单批最多 {TRANSLATE_BATCH_CHARS} 字；"
            "更大规模请用 POST /jobs/translate 进任务队列"
        )
    return normalized
