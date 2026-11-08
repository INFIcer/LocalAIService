"""站点字幕解析：SRT / WebVTT -> 带时间戳的 segments。

下载引擎会尽量把站点字幕转成 srt（拿不到就保留 vtt）。链接分析优先用站点
字幕，因此这里提供两个纯函数：

- :func:`parse_subtitle_file`：读一个 srt/vtt，返回 ``[{start,end,text}]``；
- :func:`pick_subtitle`：从多个字幕文件里按语言优先级挑一个（中文优先）。

不依赖 ffmpeg，也不联网，便于单元测试。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

#: 中文优先、英文其次。
LANGUAGE_PRIORITY = ("zh", "chs", "chi", "en", "eng")

#: 时间轴行：``00:00:01,000 --> 00:00:03,500``（vtt 用点号）。
_TIME_LINE = re.compile(
    r"^\s*((?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)

#: 字幕里的样式标签 / 内联时间标签。
_TAG = re.compile(r"<[^>]+>")
_CUE_ID = re.compile(r"^\d+$")


def _parse_timestamp(value: str) -> float:
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return 0.0
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0.0, numbers[0], numbers[1]
    else:
        return 0.0
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def parse_subtitle_text(text: str) -> list[dict[str, float | str]]:
    """解析 SRT / WebVTT 文本；坏块跳过，结果按开始时间升序。"""
    cleaned = str(text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = cleaned.split("\n")
    segments: list[dict[str, float | str]] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        match = _TIME_LINE.match(line)
        if not match:
            index += 1
            continue
        start = _parse_timestamp(match.group(1))
        end = _parse_timestamp(match.group(2))
        index += 1
        body: list[str] = []
        while index < total and lines[index].strip():
            body.append(lines[index].strip())
            index += 1
        # 跳过块之间的空行 / 序号行。
        while index < total and (not lines[index].strip() or _CUE_ID.match(lines[index].strip())):
            index += 1
        content = _TAG.sub("", " ".join(body)).strip()
        content = re.sub(r"\s+", " ", content)
        if content and end > start:
            segments.append({"start": round(start, 3), "end": round(end, 3), "text": content})
    segments.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return segments


def parse_subtitle_file(path: str | Path) -> list[dict[str, float | str]]:
    """读取一个字幕文件；不存在 / 编码异常 / 格式不对时返回空列表。"""
    target = Path(path)
    if not target.is_file():
        return []
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return parse_subtitle_text(target.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
        except OSError:
            return []
    return []


def pick_subtitle(paths: Iterable[str | Path]) -> Optional[Path]:
    """按语言优先级挑一个字幕文件；都不匹配时取第一个。"""
    files = [Path(item) for item in paths if item]
    if not files:
        return None
    for keyword in LANGUAGE_PRIORITY:
        for path in files:
            if keyword in path.name.lower():
                return path
    return files[0]


def subtitle_language(path: str | Path) -> Optional[str]:
    """从文件名里猜语言（``xxx.zh-CN.srt`` -> ``zh-CN``）。"""
    name = Path(path).name.lower()
    match = re.search(r"\.([a-z]{2}(?:-[a-z]{2})?)\.(?:srt|vtt|ass)$", name)
    if match:
        return match.group(1)
    for keyword in LANGUAGE_PRIORITY:
        if keyword in name:
            return keyword
    return None
