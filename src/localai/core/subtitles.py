"""字幕解析：SRT / WebVTT / ASS -> 带时间戳的 segments。

下载引擎会尽量把站点字幕转成 srt（拿不到就保留 vtt），媒体接口还会直接
解析外挂 ``.ass`` / ``.ssa``。这里提供纯函数，不依赖 ffmpeg / 网络，方便单测：

- :func:`parse_subtitle_file`：读一个字幕文件，返回 ``[{start,end,text}]``；
- :func:`parse_subtitle_text` / :func:`parse_ass_text`：解析文本；
- :func:`pick_subtitle`：从多个字幕文件里按语言优先级挑一个（中文优先）；
- :func:`find_sidecar_subtitles`：找视频同名的外挂字幕；
- :func:`parse_multilingual`：双语字幕只保留中文、去掉重复行。

双语（ASS 常见：一行日文、一行中文；或同一时间点两条独立事件）的处理规则：

1. 一个事件里有多个候选行时，优先「不含假名且含汉字」的行（通常是中文），
   同样优先时取靠下的一行（常见排版是中文在下）；
2. 时间轴完全相同的多条事件合并，只留语言分最高的一条；
3. 同一事件里完全重复的行去掉；相邻且文本相同的事件只保留一条。
"""

from __future__ import annotations

import glob as _glob
import html
import re
from pathlib import Path
from typing import Any, Iterable, Optional

#: 中文优先、英文其次。
LANGUAGE_PRIORITY = ("zh", "chs", "chi", "sc", "tc", "en", "eng")

#: 时间轴行：``00:00:01,000 --> 00:00:03,500``（vtt 用点号）。
_TIME_LINE = re.compile(
    r"^\s*((?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"((?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)

#: 字幕里的样式标签 / 内联时间标签。
_TAG = re.compile(r"<[^>]+>")
_CUE_ID = re.compile(r"^\d+$")
_ASS_TAG = re.compile(r"\{[^}]*\}")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")

#: 支持的外挂字幕扩展名（按优先级）。
SIDECAR_EXTENSIONS = (".ass", ".ssa", ".srt", ".vtt")

#: 文件名里的语言标记 -> 统一提示。``scjp`` / ``tcjp`` 是「简日 / 繁日双语」。
_LANGUAGE_HINTS = {
    "scjp": "zh-ja",
    "tcjp": "zh-ja",
    "sc": "zh",
    "chs": "zh",
    "cn": "zh",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "tc": "zh-Hant",
    "cht": "zh-Hant",
    "zh-tw": "zh-Hant",
    "zh-hk": "zh-Hant",
    "jp": "ja",
    "ja": "ja",
    "en": "en",
    "eng": "en",
}
_LANGUAGE_TOKEN = re.compile(
    r"(?i)(?:^|[.\-_\[\(])(scjp|tcjp|zh-hans|zh-hant|zh-cn|zh-tw|zh-hk|chs|cht|sc|tc|zh|cn|jp|ja|en|eng)(?:$|[.\-_\]\)])"
)


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


def language_hint(path: str | Path) -> Optional[str]:
    """从文件名猜语言（``xxx.scjp.ass`` -> ``zh-ja``）。

    先看扩展名前的最后一段（最可靠），再退回全文关键词；都猜不到返回 None。
    """
    name = Path(path).name.lower()
    stem = Path(name).stem
    # 优先匹配最后一个点号分段，例如 ``[Dungeon Meshi][01].scjp.ass``。
    tail = stem.rsplit(".", 1)[-1] if "." in stem else stem
    for token in (tail, stem):
        value = _LANGUAGE_HINTS.get(token)
        if value:
            return value
    match = _LANGUAGE_TOKEN.search(stem)
    if match:
        return _LANGUAGE_HINTS.get(match.group(1).lower())
    return None


def _score_line(line: str) -> tuple[int, int]:
    """给候选行打语言分：中文得分高，假名扣分；同分时靠后的行优先。"""
    cjk = len(_CJK.findall(line))
    kana = len(_KANA.findall(line))
    score = cjk * 3 - kana * 5
    # 中文标点是很强的信号。
    if re.search(r"[，。！？；：、“”‘’（）《》]", line):
        score += 2
    return score, cjk


def _clean_line(line: str) -> str:
    text = _ASS_TAG.sub("", str(line or ""))
    text = text.replace("\\N", "\n").replace("\\n", "\n")
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


def _best_text(text: str) -> str:
    """多行事件只留一行：优先中文，其次不含假名，最后取末行。"""
    # 先清样式 / 把 ``\N`` 换成真实换行，再逐行判断；顺序反了会把两行
    # 当成一行（曾导致双语 ASS 把日文也一起返回）。
    normalized = _clean_line(str(text or ""))
    lines: list[str] = []
    for raw in normalized.replace("\r", "\n").split("\n"):
        line = _clean_line(raw)
        if line and line not in lines:
            lines.append(line)
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    # 没有假名 = 不是日文；多行中文/英文是同一句的换行，直接拼起来，
    # 不能因为「只留一行」把后半句吞掉。
    if not any(_KANA.search(line) for line in lines):
        return " ".join(lines)
    # 有日文时优先「不含假名且含汉字」的中文行；同分取靠后（中文常在下）。
    chinese = [
        (index, _score_line(line))
        for index, line in enumerate(lines)
        if not _KANA.search(line) and _CJK.search(line)
    ]
    if chinese:
        best_index, _score = max(
            chinese, key=lambda item: (item[1][0], item[1][1], item[0])
        )
        return lines[best_index]
    # 全是日文（或没有中文候选）：保留全部行，避免丢内容。
    return " ".join(lines)


def _dedupe_cues(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并同一时间点的多条事件、去掉相邻重复文本。"""
    if not cues:
        return []
    ordered = sorted(cues, key=lambda item: (float(item["start"]), float(item["end"])))
    merged: list[dict[str, Any]] = []
    for cue in ordered:
        start = round(float(cue["start"]), 3)
        end = round(float(cue["end"]), 3)
        text = str(cue.get("text") or "").strip()
        if not text or end <= start:
            continue
        # 时间轴完全相同的多条事件：保留语言分最高（中文优先）的一条。
        if merged and abs(merged[-1]["start"] - start) < 0.02 and abs(
            merged[-1]["end"] - end
        ) < 0.02:
            previous = merged[-1]
            if _score_line(text) > _score_line(str(previous["text"])):
                previous["text"] = text
            continue
        # 相邻且文本相同：只留第一条（常见于 ASS 逐帧重复事件）。
        if (
            merged
            and text == str(merged[-1]["text"])
            and start <= float(merged[-1]["end"]) + 0.08
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), end)
            continue
        merged.append({"start": start, "end": end, "text": text})
    return merged


def _finalize(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for cue in cues:
        text = _best_text(str(cue.get("text") or ""))
        if text:
            cleaned.append(
                {
                    "start": round(float(cue["start"]), 3),
                    "end": round(float(cue["end"]), 3),
                    "text": text,
                }
            )
    return _dedupe_cues(cleaned)


def parse_subtitle_text(text: str) -> list[dict[str, float | str]]:
    """解析 SRT / WebVTT 文本；坏块跳过，结果按开始时间升序。

    事件正文保留换行，交给 :func:`_finalize` 做双语取舍与去重。
    """
    cleaned = str(text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = cleaned.split("\n")
    cues: list[dict[str, Any]] = []
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
        if body:
            cues.append({"start": start, "end": end, "text": "\n".join(body)})
    return _finalize(cues)  # type: ignore[return-value]


def parse_ass_text(text: str) -> list[dict[str, float | str]]:
    """解析 ASS / SSA 的 ``[Events]`` 段（只处理 Dialogue）。"""
    cleaned = str(text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = cleaned.split("\n")
    in_events = False
    columns: list[str] = []
    cues: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "[events]":
            in_events = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_events:
                break
            continue
        if not in_events:
            continue
        if stripped.lower().startswith("format:"):
            columns = [part.strip().lower() for part in stripped.split(":", 1)[1].split(",")]
            continue
        if not stripped.lower().startswith("dialogue:"):
            continue
        payload = stripped.split(":", 1)[1].lstrip()
        if not columns:
            columns = [
                "layer",
                "start",
                "end",
                "style",
                "name",
                "marginl",
                "marginr",
                "marginv",
                "effect",
                "text",
            ]
        # Text 是最后一列，里面可能还有逗号；只切 len(columns)-1 次。
        values = payload.split(",", len(columns) - 1)
        if len(values) < len(columns):
            values += [""] * (len(columns) - len(values))
        row = dict(zip(columns, values))
        try:
            start = _parse_ass_timestamp(row.get("start", ""))
            end = _parse_ass_timestamp(row.get("end", ""))
        except (TypeError, ValueError):
            continue
        body = str(row.get("text") or "")
        if body:
            cues.append({"start": start, "end": end, "text": body})
    return _finalize(cues)  # type: ignore[return-value]


def _parse_ass_timestamp(value: str) -> float:
    """ASS 时间轴：``0:00:01.23``（百分秒，只有两位小数）。"""
    text = str(value or "").strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) != 3:
        return _parse_timestamp(text)
    try:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return 0.0
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def parse_subtitle_file(path: str | Path) -> list[dict[str, float | str]]:
    """读取一个字幕文件；不存在 / 编码异常 / 格式不对时返回空列表。"""
    target = Path(path)
    if not target.is_file():
        return []
    parser = parse_ass_text if target.suffix.lower() in (".ass", ".ssa") else parse_subtitle_text
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return parser(target.read_text(encoding=encoding))
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
    match = re.search(r"\.([a-z]{2}(?:-[a-z]{2})?)\.(?:srt|vtt|ass|ssa)$", name)
    if match:
        return match.group(1)
    for keyword in LANGUAGE_PRIORITY:
        if keyword in name:
            return keyword
    return None


def _stem_glob(video: Path) -> str:
    """转义视频名后拼 ``*``，避免 ``[DBD-Raws]`` 被 glob 当成字符类。"""
    return _glob.escape(str(video.stem)) + "*"


def find_sidecar_subtitles(video: str | Path) -> list[dict[str, str]]:
    """找视频同目录、同名前缀的外挂字幕。

    例如 ``[DBD-Raws][Dungeon Meshi][01]...mkv`` 旁边的
    ``....scjp.ass`` / ``....tcjp.ass``。返回
    ``[{"path","format","language_hint"}]``，按文件名排序。
    """
    target = Path(video)
    if not target.parent.is_dir():
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in sorted(target.parent.glob(_stem_glob(target))):
        if not item.is_file() or item.resolve() == target.resolve():
            continue
        suffix = item.suffix.lower()
        if suffix not in SIDECAR_EXTENSIONS:
            continue
        key = str(item.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "path": str(item),
                "format": suffix.lstrip("."),
                "language_hint": language_hint(item) or "",
            }
        )
    return found


def is_chinese_language(value: Optional[str]) -> bool:
    """语言码 / 标题是否像中文（chi / zho / zh / chs / 简 / 繁）。"""
    text = str(value or "").strip().lower()
    if not text:
        return False
    return text.startswith(("zh", "chi", "zho", "chs", "cht", "sc", "tc")) or any(
        word in text for word in ("chinese", "简体", "繁体", "中文", "简中", "繁中")
    )


def rank_soft_subtitle_sources(listing: dict[str, Any]) -> list[dict[str, Any]]:
    """把 ``/media/subtitles`` 的结果排成优先候选（中文、非 forced、内嵌优先）。

    返回 ``[{"source":"embedded","index":39}, {"source":"sidecar","path":"..."}]``；
    调用方逐个尝试抽取，第一个成功的就是该用哪个字幕源。
    """
    embedded: list[dict[str, Any]] = []
    for item in listing.get("embedded") or []:
        language = str(item.get("language") or "")
        title = str(item.get("title") or "")
        chinese = is_chinese_language(language) or is_chinese_language(title)
        simplified = any(
            word in title.lower() for word in ("simplified", "简体", "简中", "sc")
        )
        score = (
            0 if chinese else 2,
            1 if item.get("forced") else 0,
            0 if simplified else 1,
            0 if item.get("default") else 1,
            int(item.get("index") or 0),
        )
        embedded.append(
            {"source": "embedded", "index": int(item.get("index")), "score": score}
        )
    embedded.sort(key=lambda row: row["score"])

    sidecar: list[dict[str, Any]] = []
    for item in listing.get("sidecar") or []:
        hint = str(item.get("language_hint") or "")
        chinese = is_chinese_language(hint)
        simplified = hint.lower().startswith("zh") and "hant" not in hint.lower()
        score = (
            0 if chinese else 2,
            0 if simplified else 1,
            str(item.get("path") or ""),
        )
        sidecar.append(
            {
                "source": "sidecar",
                "path": str(item.get("path")),
                "score": score,
            }
        )
    sidecar.sort(key=lambda row: row["score"])
    # 内嵌优先于外挂：内嵌通常和片源同版本，时间轴最稳。
    return embedded + sidecar


def format_srt(segments: Iterable[dict[str, Any]]) -> str:
    """把 segments 写成标准 SRT 文本。"""
    blocks: list[str] = []
    for index, item in enumerate(segments, start=1):
        start = _srt_timestamp(float(item.get("start") or 0.0))
        end = _srt_timestamp(float(item.get("end") or 0.0))
        text = str(item.get("text") or "").strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(milliseconds, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


__all__ = [
    "LANGUAGE_PRIORITY",
    "SIDECAR_EXTENSIONS",
    "find_sidecar_subtitles",
    "format_srt",
    "is_chinese_language",
    "language_hint",
    "parse_ass_text",
    "parse_subtitle_file",
    "parse_subtitle_text",
    "pick_subtitle",
    "rank_soft_subtitle_sources",
    "subtitle_language",
]
