"""分段转写结果的统一整理。

两个引擎给分段的方式不同：

- **whisper**：原生 segment 时间戳（``timestamp_source="native"``）；
- **sensevoice**：没有原生时间戳，用 ``fsmn-vad`` 切句后逐段识别，
  ``start/end`` 就是 VAD 的语音段边界（``timestamp_source="vad"``）。

无论来源，对外都必须满足：**按时间升序、互不重叠、``end > start``**。
这里把「引擎原始输出 -> 稳定契约」的整理逻辑收敛成一个纯函数，路由和测试
共用，避免每个接口各写一份。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

#: 时间精度（秒）；契约里统一保留 3 位小数。
PRECISION = 3


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_segments(
    raw: Optional[Iterable[dict[str, Any]]],
    *,
    duration: Optional[float] = None,
) -> list[dict[str, Any]]:
    """把任意来源的分段整理成升序、无重叠、``end > start`` 的列表。

    规则（按顺序执行，保证结果稳定）：

    1. 丢掉缺 ``end`` / ``end <= start`` 的项；
    2. ``start`` 夹到 ``[0, duration]``，``end`` 夹到 ``[start, duration]``；
    3. 按 ``start`` 升序排序（``start`` 相同按 ``end``）；
    4. 若与上一段重叠，把本段 ``start`` 推到上一段 ``end``；推完仍不合法就丢弃；
    5. 统一四舍五入到 3 位小数；四舍五入后又 ``end <= start`` 的补 0.001 秒。

    不修改入参，返回新的 list。
    """
    items: list[tuple[float, float, str]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        start = _as_float(item.get("start"), 0.0)
        end = _as_float(item.get("end"), 0.0)
        text = str(item.get("text") or "").strip()
        if end <= start:
            continue
        start = max(0.0, start)
        if duration is not None:
            end = min(end, float(duration))
        if end <= start:
            continue
        items.append((start, end, text))

    items.sort(key=lambda entry: (entry[0], entry[1]))

    result: list[dict[str, Any]] = []
    last_end = 0.0
    for start, end, text in items:
        if result and start < last_end:
            start = last_end
        if end <= start:
            continue
        start_r = round(start, PRECISION)
        end_r = round(end, PRECISION)
        if end_r <= start_r:
            end_r = round(start_r + 10 ** (-PRECISION), PRECISION)
        result.append({"start": start_r, "end": end_r, "text": text})
        last_end = end_r
    return result


def segments_duration(segments: Iterable[dict[str, Any]]) -> Optional[float]:
    """分段里最后一段的结束时间（给缺少 duration 的引擎兜底）。"""
    last: Optional[float] = None
    for item in segments:
        end = _as_float(item.get("end"), 0.0)
        if last is None or end > last:
            last = end
    return round(last, PRECISION) if last is not None else None
