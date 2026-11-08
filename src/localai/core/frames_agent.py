"""关键帧挑选代理：按字幕 + 场景 + VLM 评分循环抽帧，凑够 N 张有效帧。

为什么需要它
------------
均匀抽帧（0% / 50% / 100%）在讲解类视频里经常抽到章节页、转场和空镜。
本模块把「字幕线索 + 场景边界 + 画面信息量」合在一起：

1. **规划候选点**：字幕关键词位（对比/数据/排行/步骤/结论…）、场景切换点、
   均匀补充点；``agent`` planner 会额外让 VLM 用
   ``docs/prompts/vlm_frame_plan.md`` 规划一批。
2. **先做便宜的预筛**：抽 8~16 张，感知哈希去重 + 亮度/方差判断近纯色，
   空白帧直接判 ``blank``，不浪费 VLM 调用。
3. **再让 VLM 判信息量**：``hybrid`` 用单帧评分提示词逐张打分；
   ``agent`` 用多轮代理协议（extract / final）边看边补抽。
4. **失败降级**：VLM 不可用或全部超时，退回「方差 + 场景点」的确定性结果，
   响应里带 ``degraded=true``，进程不崩。

对外只有一个入口 :meth:`FramesAgent.run`，HTTP 路由和任务执行器共用。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from ..config import Config, get_config

logger = logging.getLogger("localai.frames_agent")

#: 字幕里值得抽帧的关键词（和 vlm_frame_plan.md 的规则保持一致）。
KEYWORDS = (
    "对比",
    "数据",
    "排行",
    "排名",
    "步骤",
    "参数",
    "结论",
    "表格",
    "代码",
    "演示",
    "总结",
    "测试",
    "效果",
    "结果",
    "价格",
    "许可",
    "速度",
    "性能",
    "教程",
    "方法",
)

#: 默认抽帧宽度：768px 足够看清幻灯片正文，又能把视觉 token 控制在 ~1000。
FRAME_WIDTH = 768
#: 感知哈希阈值默认值（汉明距离），配置项可覆盖。
DEFAULT_PHASH_DISTANCE = 6
#: 首轮候选数量范围。
MIN_INITIAL = 8
MAX_INITIAL = 16
#: 补抽每轮数量范围。
MIN_SUPPLEMENT = 4
MAX_SUPPLEMENT = 8

#: 分析力度可选值；默认 ``auto``（按时长自动定档，旧调用不传 effort 也走它）。
#: ``low|medium|high`` 仍然可以显式指定，预算语义保持不变。
EFFORT_CHOICES = ("auto", "low", "medium", "high")
EFFORT_DEFAULT = "auto"

#: auto 档位阈值（秒）：≤10 分钟 = low，10~40 分钟 = medium，>40 分钟 = high。
AUTO_LOW_MAX_SECONDS = 10 * 60
AUTO_MEDIUM_MAX_SECONDS = 40 * 60
#: 超长片覆盖分段：每约 5 分钟一段，夹在 8~24 段；142 分钟 -> 24 段。
AUTO_LONG_SEGMENT_SECONDS = 300
AUTO_MIN_LONG_SEGMENTS = 8
AUTO_MAX_LONG_SEGMENTS = 24

#: 无人声空档判定阈值（秒）：相邻字幕间隔 ≥ 这个值就算「没人说话」。
SILENCE_MIN_GAP = 5.0

#: effort -> 预算。数字越小/越大都有明确含义，测试会逐项断言。
EFFORT_BUDGETS: dict[str, dict[str, Any]] = {
    "low": {
        "candidate_factor": 2,
        "min_candidates": 6,
        "max_rounds": 2,
        "vlm_limit": 6,
        "clip_limit": 0,
        "clip_fps": 1.0,
        "fixed_segment_midpoint": True,
        "note": "只做全片分段覆盖（每段中点 1 张），不细看",
    },
    "medium": {
        "candidate_factor": 4,
        "min_candidates": 6,
        "max_rounds": 4,
        "vlm_limit": 16,
        "clip_limit": 1,
        "clip_fps": 1.0,
        "fixed_segment_midpoint": False,
        "note": "覆盖 + 字幕关键词 + 无人声段",
    },
    "high": {
        "candidate_factor": 6,
        "min_candidates": 6,
        "max_rounds": 6,
        "vlm_limit": 32,
        "clip_limit": 2,
        "clip_fps": 2.0,
        "fixed_segment_midpoint": False,
        "note": "在 medium 基础上允许裁剪细看可疑段",
    },
}

#: 单次 extract_clip 允许的最长窗口（秒），和提示词里的约定一致。
MAX_CLIP_SECONDS = 60.0
#: extract_clip 允许的 fps 范围。
CLIP_MIN_FPS = 0.5
CLIP_MAX_FPS = 3.0


# ---------------------------------------------------------------------------
# 小工具：JSON / 提示词 / 图片
# ---------------------------------------------------------------------------

def parse_json_loose(text: str) -> Optional[dict[str, Any]]:
    """尽量从模型输出里解析出 JSON 对象。

    VLM 经常不听话地包一层 markdown 围栏，或者在前后加一句解释；这里：
    去掉 ```json 围栏 -> 找第一个 ``{`` 到最后一个 ``}`` -> 容忍尾逗号。
    """
    if not text:
        return None
    raw = str(text).strip()
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = raw[start : end + 1]
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            data = json.loads(attempt)
        except (ValueError, TypeError):
            continue
        return data if isinstance(data, dict) else None
    return None


def load_prompt(cfg: Config, name: str) -> str:
    """从 ``<HOME>/docs/prompts/<name>`` 读提示词，去掉文件头的说明块。"""
    candidates = [
        Path(cfg.home) / "docs" / "prompts" / name,
        Path(__file__).resolve().parents[3] / "docs" / "prompts" / name,
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # 文件开头是「用途/输出严格 JSON」的说明，真正的系统提示词在 --- 之后。
        if "---" in text:
            text = text.split("---", 1)[1]
        return text.strip()
    logger.warning("提示词文件不存在：%s（按空提示词继续）", name)
    return ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _analyze_image(path: str | Path) -> dict[str, Any]:
    """亮度 / 方差 / 64 位 dHash；失败时返回 ``ok=False``（交给 VLM 兜底）。"""
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"缺少 Pillow/numpy：{exc}", "blank": False}

    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            small = gray.resize((64, 64), Image.Resampling.BILINEAR)
            array = np.asarray(small, dtype=np.float32)
            brightness = float(array.mean())
            std = float(array.std())
            hash_image = gray.resize((9, 8), Image.Resampling.LANCZOS)
            pixels = np.asarray(hash_image, dtype=np.int16)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"读图失败：{type(exc).__name__}: {exc}", "blank": False}

    # dHash：横向相邻像素比较，得到 64 位整数。
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    phash = 0
    for bit in bits:
        phash = (phash << 1) | int(bool(bit))

    # 近纯色/低方差：画面几乎没有结构，直接判 blank。
    blank = std < 6.0 or (brightness < 12.0 and std < 12.0) or (
        brightness > 246.0 and std < 8.0
    )
    return {
        "ok": True,
        "brightness": round(brightness, 2),
        "std": round(std, 2),
        "phash": phash,
        "blank": bool(blank),
    }


def _hamming(left: int, right: int) -> int:
    return int(left ^ right).bit_count()


def _load_segments_from_file(path: Path) -> list[dict[str, Any]]:
    """复用 ``core.subtitles`` 的解析器，读站点字幕（srt/vtt）。"""
    try:
        from ..core.subtitles import parse_subtitle_file  # noqa: PLC0415

        return parse_subtitle_file(path)
    except Exception as exc:  # noqa: BLE001 - 字幕只是线索，解析失败不影响挑帧
        logger.warning("解析字幕失败：%s（%s）", path, exc)
        return []


def _normalize_segments(raw: Any) -> list[dict[str, Any]]:
    """把调用方给的字幕归一成 ``[{start,end,text}]``，容错字符串/字典。"""
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        path = Path(str(raw))
        return _load_segments_from_file(path) if path.is_file() else []
    segments: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start") if item.get("start") is not None else 0.0)
            end = float(item.get("end") if item.get("end") is not None else start)
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or "").strip()
        if end < start:
            start, end = end, start
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    segments.sort(key=lambda item: item["start"])
    return segments


def _subtitle_nearby(segments: Sequence[dict[str, Any]], time_sec: float) -> str:
    """取时间点附近 ±4 秒的字幕，最多拼 3 句（控制 prompt 长度）。"""
    picked: list[str] = []
    for item in segments:
        if item["end"] < time_sec - 4 or item["start"] > time_sec + 4:
            continue
        text = str(item.get("text") or "").strip()
        if text:
            picked.append(text)
        if len(picked) >= 3:
            break
    return " ".join(picked)[:200]


def normalize_effort(value: Any) -> str:
    """把调用方给的 effort 归一成 ``auto|low|medium|high``，非法值回落 auto。"""
    text = str(value or "").strip().lower()
    return text if text in EFFORT_CHOICES else EFFORT_DEFAULT


def auto_effort_profile(duration: Optional[float]) -> str:
    """auto 档位：按时长落到 ``low|medium|high``（无时长按 medium 处理）。"""
    try:
        seconds = float(duration or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= AUTO_LOW_MAX_SECONDS:
        return "low"
    if seconds <= AUTO_MEDIUM_MAX_SECONDS:
        return "medium"
    return "high"


def coverage_segment_count(
    target_frames: int, effort: Any = "low", duration: Optional[float] = None
) -> int:
    """覆盖分段数 K。

    - 显式 ``low/medium/high``：沿用旧规则 ``K = max(target×2, 6)``；
    - ``auto``：≤10 分钟 ``max(target×2,6)``；10~40 分钟 ``max(target×2,8)``；
      >40 分钟 ``clamp(ceil(duration/300), 8, 24)``（142 分钟 -> 24 段）。
    """
    target = max(1, int(target_frames or 3))
    name = normalize_effort(effort)
    if name != "auto":
        return max(target * 2, 6)
    profile = auto_effort_profile(duration)
    if profile == "low":
        return max(target * 2, 6)
    if profile == "medium":
        return max(target * 2, 8)
    segments = int(-(-float(duration or 0) // AUTO_LONG_SEGMENT_SECONDS))
    return max(AUTO_MIN_LONG_SEGMENTS, min(AUTO_MAX_LONG_SEGMENTS, segments))


def effort_budget(
    target_frames: int,
    effort: Any = EFFORT_DEFAULT,
    duration: Optional[float] = None,
) -> dict[str, Any]:
    """按目标帧数 + effort（+ 时长）算出这一轮的所有预算。

    返回字段（都会被写进响应的 ``stats.budget``，方便调用方核对）：

    - ``effort``：调用方请求值（可能是 auto）；
    - ``profile``：auto 实际落到的档位（low/medium/high）；
    - ``coverage_segments``：覆盖分段数 K；候选数至少补齐 K，保证每段有候选；
    - ``candidates``：候选点数量 = ``target × candidate_factor``（至少 6 / K）；
    - ``max_rounds``：最多抽几轮（含首轮）；
    - ``vlm_limit``：VLM 调用次数硬上限（含规划 / 评分 / 代理决策）；
    - ``clip_limit``：``extract_clip`` 次数上限（low=0、medium=1、high=2）。
    """
    name = normalize_effort(effort)
    profile = auto_effort_profile(duration) if name == "auto" else name
    preset = EFFORT_BUDGETS[profile]
    target = max(1, int(target_frames or 3))
    segments = coverage_segment_count(target, name, duration)
    candidates = max(
        int(preset["min_candidates"]), target * int(preset["candidate_factor"])
    )
    # 覆盖是硬约束：候选池必须至少能覆盖每一段，否则首轮就会漏段。
    candidates = max(candidates, segments)
    return {
        "effort": name,
        "profile": profile,
        "coverage_segments": segments,
        "candidates": candidates,
        "max_rounds": int(preset["max_rounds"]),
        "vlm_limit": int(preset["vlm_limit"]),
        "clip_limit": int(preset["clip_limit"]),
        "clip_fps": float(preset["clip_fps"]),
        "fixed_segment_midpoint": bool(preset["fixed_segment_midpoint"]),
        "note": preset["note"],
    }


def segment_index(time_sec: float, duration: float, count: int) -> int:
    """时间点落在第几段（0-based）；时长非法时返回 0。"""
    if not duration or duration <= 0 or count <= 0:
        return 0
    index = int(float(time_sec) / float(duration) * int(count))
    return max(0, min(int(count) - 1, index))


def silence_gaps(
    segments: Sequence[dict[str, Any]],
    duration: float,
    *,
    min_gap: float = SILENCE_MIN_GAP,
) -> list[dict[str, float]]:
    """从字幕算「没有人声」的空档（含片头片尾）。

    规则：相邻字幕之间的间隔 ≥ ``min_gap``（默认 5 秒）就算空档；
    第一条字幕之前、最后一条字幕之后也各自算（只要够长）。
    没有字幕时整段视频就是一个人声空档（但太短的返回空列表）。
    """
    if not duration or duration <= 0:
        return []
    gaps: list[dict[str, float]] = []
    ordered = [
        item
        for item in segments
        if float(item.get("end", 0)) > float(item.get("start", 0))
    ]
    if not ordered:
        if duration >= min_gap:
            gaps.append({"start": 0.0, "end": round(float(duration), 3)})
        return gaps

    first_start = max(0.0, float(ordered[0]["start"]))
    if first_start >= min_gap:
        gaps.append({"start": 0.0, "end": round(first_start, 3)})
    for previous, current in zip(ordered, ordered[1:]):
        start = float(previous["end"])
        end = float(current["start"])
        if end - start >= min_gap:
            gaps.append(
                {"start": round(max(0.0, start), 3), "end": round(min(duration, end), 3)}
            )
    last_end = min(float(duration), float(ordered[-1]["end"]))
    if duration - last_end >= min_gap:
        gaps.append({"start": round(last_end, 3), "end": round(float(duration), 3)})
    # 去掉零长度/越界的（字幕本身可能有点乱）。
    return [gap for gap in gaps if gap["end"] - gap["start"] >= min_gap]


def gap_midpoints(
    gaps: Sequence[dict[str, Any]], duration: float
) -> list[dict[str, Any]]:
    """把每个空档换算成「中点候选」，带上起止秒数方便写提示词。"""
    points: list[dict[str, Any]] = []
    for gap in gaps:
        try:
            start = max(0.0, float(gap.get("start")))
            end = min(float(duration), float(gap.get("end")))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        middle = (start + end) / 2
        if middle <= 0.2 or middle >= duration - 0.2:
            continue
        points.append(
            {
                "time": round(middle, 3),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return points


def _summary_confidence(summary: str) -> str:
    """判断 summary 是否「太短 / 太模糊」，供调用方决定要不要自己核对原图。"""
    text = str(summary or "").strip()
    normalized = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    vague = ("大概", "可能", "也许", "或许", "似乎", "应该是", "不确定", "看起来像")
    if len(normalized) < 6 or any(word in text for word in vague):
        return "low"
    return "normal"


def summary_confidence(summary: str) -> str:
    """公开版本：``/vision/video`` 等外部模块复用同一套模糊词判定。"""
    return _summary_confidence(summary)


# ---------------------------------------------------------------------------
# 确定性规划
# ---------------------------------------------------------------------------

def plan_candidates(
    duration: Optional[float],
    segments: Sequence[dict[str, Any]],
    scene_times: Sequence[float],
    *,
    target_frames: int,
    limit: int = 40,
    effort: Any = EFFORT_DEFAULT,
    count: Optional[int] = None,
    gaps: Optional[Sequence[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """按「无人声空档 > 字幕关键词 > 场景切换 > 段覆盖/均匀补充」生成候选。

    硬性保证（对应 ``vlm_frame_plan.md`` 的分层覆盖）：

    1. 全片等分成 ``K = max(target×2, 6)`` 段，**每段至少 1 个候选**；
    2. 每个无人声空档的中点都进候选（除非与已有候选 < 2 秒）；
    3. 返回顺序按段轮转（round-robin），避免「第一批全在开头」；
    4. ``low`` 档每段只取中点，不做关键词加密。

    返回 ``[{time, priority, reason, segment}]``，priority 越小越优先。
    """
    if not duration or duration <= 0:
        return []
    total = max(1, min(40, int(count if count is not None else limit)))
    effort_name = normalize_effort(effort)
    budget = effort_budget(target_frames, effort_name, duration)
    k = coverage_segment_count(target_frames, effort_name, duration)
    gaps = list(gaps) if gaps is not None else silence_gaps(segments, duration)
    # 短视频放宽最小间隔，否则凑不出足够候选。
    min_gap = 3.0 if duration >= 30 else max(1.0, duration / max(4, target_frames * 4))

    picked: list[dict[str, Any]] = []
    seen: list[float] = []

    def push(
        priority: int,
        point: float,
        reason: str,
        *,
        locked: bool = False,
    ) -> None:
        try:
            value = float(point)
        except (TypeError, ValueError):
            return
        value = _clamp(value, 0.5, max(0.6, duration - 0.2))
        # 空档中点是「必查项」，只避让 2 秒内的重复；其余按最小间隔去重。
        gap_limit = 2.0 if locked else min_gap
        if any(abs(value - other) < gap_limit for other in seen):
            return
        seen.append(value)
        picked.append(
            {
                "time": round(value, 3),
                "priority": int(priority),
                "reason": reason,
                "segment": segment_index(value, duration, k),
                "locked": bool(locked),
            }
        )

    # 1) 无人声空档中点：locked，后续裁剪也不会优先丢掉。
    gap_points = gap_midpoints(gaps, duration)
    for item in gap_points:
        push(
            1,
            item["time"],
            f"无人声空档中点（{item['start']:.1f}~{item['end']:.1f}s）",
            locked=True,
        )

    # low：每段只取中点，不加密。
    if budget["fixed_segment_midpoint"]:
        for index in range(k):
            middle = duration * (index + 0.5) / k
            push(2, middle, f"第 {index + 1}/{k} 段中点（low 覆盖）")

    # 2) 字幕关键词（priority 1）
    if not budget["fixed_segment_midpoint"]:
        for item in segments:
            text = str(item.get("text") or "")
            hit = next((word for word in KEYWORDS if word in text), None)
            if hit is None:
                continue
            point = (float(item["start"]) + float(item["end"])) / 2
            push(1, point, f"字幕出现「{hit}」")

        # 3) 场景切换点（priority 2）
        for value in scene_times:
            try:
                point = float(value) + 0.3
            except (TypeError, ValueError):
                continue
            push(2, point, "场景切换点")

        # 4) 均匀补充（priority 3）
        count_uniform = max(target_frames * 4, 8)
        step = duration / count_uniform
        for index in range(count_uniform):
            push(3, step * (index + 0.5), "均匀补充")

    # 5) 段覆盖兜底：某段一个候选都没有时，补该段中点。
    covered = {item["segment"] for item in picked}
    for index in range(k):
        if index in covered:
            continue
        middle = duration * (index + 0.5) / k
        push(2, middle, f"补第 {index + 1}/{k} 段覆盖（该段此前无候选）")
        covered.add(index)

    # 6) 控制总数：locked（空档）与「每段至少一个」是硬约束，优先保留；
    #    预算只约束额外加密点，绝不因为预算把某一段或空档挤掉。
    if len(picked) > total:
        locked_items = [item for item in picked if item.get("locked")]
        rest = [item for item in picked if not item.get("locked")]
        per_segment: dict[int, dict[str, Any]] = {}
        for item in sorted(rest, key=lambda row: (row["priority"], row["time"])):
            per_segment.setdefault(int(item["segment"]), item)
        keep_ids = {id(item) for item in locked_items}
        keep_ids.update(id(item) for item in per_segment.values())
        base_count = len(keep_ids)
        target_count = max(total, base_count)
        for item in sorted(rest, key=lambda row: (row["priority"], row["time"])):
            if len(keep_ids) >= target_count:
                break
            keep_ids.add(id(item))
        picked = [item for item in picked if id(item) in keep_ids][:40]

    # 7) 段轮转排序：第一批 N 个尽量每段一个，禁止沿时间顺序挤在开头。
    buckets: dict[int, list[dict[str, Any]]] = {index: [] for index in range(k)}
    for item in picked:
        buckets.setdefault(int(item["segment"]), []).append(item)
    for index in buckets:
        buckets[index].sort(key=lambda row: (row["priority"], row["time"]))
    ordered: list[dict[str, Any]] = []
    while len(ordered) < len(picked):
        added = False
        for index in range(k):
            bucket = buckets.get(index) or []
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added = True
            if len(ordered) >= len(picked):
                break
        if not added:
            break
    return [
        {
            "time": item["time"],
            "priority": item["priority"],
            "reason": item["reason"],
            "segment": item["segment"],
        }
        for item in ordered
    ]


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class FramesAgent:
    """一次关键帧挑选的完整流程（无状态，可并发实例化）。"""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        registry: Any = None,
        *,
        enable_asr: bool = True,
    ) -> None:
        self.cfg = cfg or get_config()
        self.registry = registry
        self.enable_asr = enable_asr

    # ---- 依赖 ---------------------------------------------------------
    def _media(self) -> Any:
        if self.registry is None:
            raise RuntimeError("FramesAgent 需要 registry（media 能力）")
        return self.registry.get("media")

    def _vision(self) -> Any:
        if self.registry is None:
            raise RuntimeError("FramesAgent 需要 registry（vision 能力）")
        return self.registry.get("vision")

    def _planner(self, planner: Optional[str]) -> str:
        value = (planner or self.cfg.vision_planner or "hybrid").strip().lower()
        return value if value in ("hybrid", "agent") else "hybrid"

    # ---- effort 预算 / 覆盖统计 ---------------------------------------
    @staticmethod
    def _vlm_left(stats: dict[str, Any]) -> int:
        """本次任务还剩多少次 VLM 调用；0 表示预算已用尽。"""
        limit = int(stats.get("vlm_limit") or 0)
        if limit <= 0:
            return 999
        return max(0, limit - int(stats.get("vlm_calls") or 0))

    @staticmethod
    def _clip_left(stats: dict[str, Any]) -> int:
        limit = int(stats.get("clip_limit") or 0)
        return max(0, limit - int(stats.get("clip_calls") or 0))

    @staticmethod
    def _mark_seen(
        segments_seen: Sequence[dict[str, Any]],
        batch: Sequence[dict[str, Any]],
        duration: float,
        count: int,
    ) -> None:
        """更新「每段抽了几次」：时间段、次数都记，供提示词和覆盖判断用。"""
        for item in batch:
            try:
                index = segment_index(float(item["time"]), duration, count)
            except (TypeError, ValueError, KeyError):
                continue
            if not 0 <= index < len(segments_seen):
                continue
            entry = segments_seen[index]
            times = entry.setdefault("times", [])
            times.append(round(float(item["time"]), 3))
            if len(times) > 12:
                del times[:-12]
            entry["extracted"] = int(entry.get("extracted") or 0) + 1

    @staticmethod
    def _interleave_by_segment(
        pool: Sequence[dict[str, Any]], duration: float, target: int, k: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """把候选按段轮转排序，保证前 K 个候选分布在 K 个不同段。"""
        k = int(k or coverage_segment_count(target))
        buckets: dict[int, list[dict[str, Any]]] = {index: [] for index in range(k)}
        for item in pool:
            index = item.get("segment")
            if index is None:
                index = segment_index(float(item["time"]), duration, k)
            buckets.setdefault(int(index), []).append(dict(item))
        for index in buckets:
            buckets[index].sort(
                key=lambda row: (int(row.get("priority") or 3), float(row["time"]))
            )
        ordered: list[dict[str, Any]] = []
        total = len(pool)
        while len(ordered) < total:
            added = False
            for index in range(k):
                bucket = buckets.get(index) or []
                if not bucket:
                    continue
                ordered.append(bucket.pop(0))
                added = True
                if len(ordered) >= total:
                    break
            if not added:
                break
        return ordered

    def _ensure_coverage(
        self,
        pool: Sequence[dict[str, Any]],
        duration: float,
        target: int,
        gaps: Sequence[dict[str, Any]],
        k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """VLM 规划的候选可能漏段/漏空档：这里补齐后再做段轮转排序。"""
        items = [dict(item) for item in pool]
        k = int(k or coverage_segment_count(target))
        points = [float(item["time"]) for item in items]

        def add(priority: int, point: float, reason: str) -> None:
            if any(abs(point - value) < 2.0 for value in points):
                return
            value = _clamp(point, 0.5, max(0.6, duration - 0.2))
            items.append(
                {
                    "time": round(value, 3),
                    "priority": priority,
                    "reason": reason,
                    "segment": segment_index(value, duration, k),
                }
            )
            points.append(value)

        for gap in gap_midpoints(gaps, duration):
            add(1, gap["time"], "无人声空档中点（程序补齐）")
        covered = {int(item.get("segment", -1)) for item in items}
        for index in range(k):
            if index in covered:
                continue
            add(2, duration * (index + 0.5) / k, f"补第 {index + 1}/{k} 段覆盖")
        return self._interleave_by_segment(items, duration, target, k)

    def _supplement_times(
        self,
        pool: Sequence[dict[str, Any]],
        duration: float,
        target: int,
        used: Sequence[float],
        results: Sequence[dict[str, Any]],
        count: int,
        *,
        gaps: Sequence[dict[str, Any]] = (),
        k: Optional[int] = None,
    ) -> list[float]:
        """补抽时间点：优先没被最终选中的段，其次才是普通候选。"""
        k = int(k or coverage_segment_count(target))
        picked_segments = {
            segment_index(float(item["time"]), duration, k) for item in results
        }
        candidates = [
            item
            for item in pool
            if all(abs(float(item["time"]) - value) >= 2.0 for value in used)
        ]
        uncovered = [
            item
            for item in candidates
            if segment_index(float(item["time"]), duration, k) not in picked_segments
        ]
        ordered = uncovered + candidates
        times = self._take_times(ordered, used, count)

        # 候选池不够时，直接按「未覆盖段」的中点补。
        if len(times) < count:
            for index in range(k):
                if index in picked_segments:
                    continue
                middle = round(duration * (index + 0.5) / k, 3)
                if any(abs(middle - value) < 2.0 for value in used):
                    continue
                if any(abs(middle - value) < 2.0 for value in times):
                    continue
                times.append(middle)
                if len(times) >= count:
                    break
        return times

    @staticmethod
    def _uncovered_segments(segments_seen: Sequence[dict[str, Any]]) -> list[int]:
        """还没抽过任何一帧的分段（按 index 升序）。"""
        return [
            int(item["index"])
            for item in segments_seen
            if int(item.get("extracted") or 0) <= 0
        ]

    @staticmethod
    def _spread_targets(k: int, target: int) -> list[int]:
        """把 target 个最终帧均匀摊到 k 段上（含首尾段）。"""
        if k <= 0:
            return []
        if target <= 1:
            return [k // 2]
        count = min(int(target), int(k))
        return [
            int(round(position * (k - 1) / (count - 1)))
            for position in range(count)
        ]

    @staticmethod
    def _segment_alternative(
        index: int,
        duration: float,
        k: int,
        used: Sequence[float],
        *,
        attempt: int = 0,
        prefer_middle: bool = True,
    ) -> Optional[float]:
        """在某一段里找一个没抽过的时间点；用于覆盖 / 空白重抽。"""
        if k <= 0 or duration <= 0:
            return None
        span = float(duration) / k
        start = index * span
        fractions = [0.5, 0.3, 0.7, 0.18, 0.82, 0.42, 0.58, 0.1, 0.9]
        if not prefer_middle:
            fractions = fractions[1:]
        offset = max(0, int(attempt))
        for position in range(offset, len(fractions) + offset):
            fraction = fractions[position % len(fractions)]
            point = round(start + span * fraction, 3)
            if point <= 0.2 or point >= duration - 0.2:
                continue
            if any(abs(point - value) < 2.0 for value in used):
                continue
            return point
        return None

    def _coverage_times(
        self,
        pool: Sequence[dict[str, Any]],
        duration: float,
        target: int,
        used: Sequence[float],
        results: Sequence[dict[str, Any]],
        count: int,
        k: int,
        segments_seen: Sequence[dict[str, Any]],
    ) -> list[float]:
        """下一轮抽帧时间点（代码强制的覆盖规则）。

        规则 1：只要还有段没抽过，**本轮只从这些段里取**，且每段最多取 1 个；
        规则 2：全部段都抽过之后，才允许补抽同段的第二个时间点；
        规则 3：优先选还没有可用帧的段，避免一直补开头。
        """
        wanted = max(1, int(count))
        uncovered = self._uncovered_segments(segments_seen)
        picked: list[float] = []
        if uncovered:
            for index in uncovered:
                candidates = sorted(
                    (
                        item
                        for item in pool
                        if segment_index(float(item["time"]), duration, k) == index
                    ),
                    key=lambda row: (int(row.get("priority") or 3), float(row["time"])),
                )
                point: Optional[float] = None
                for item in candidates:
                    value = round(float(item["time"]), 3)
                    if any(abs(value - other) < 2.0 for other in used):
                        continue
                    if any(abs(value - other) < 2.0 for other in picked):
                        continue
                    point = value
                    break
                if point is None:
                    point = self._segment_alternative(
                        index, duration, k, list(used) + picked
                    )
                if point is not None:
                    picked.append(point)
                if len(picked) >= wanted:
                    break
            return picked

        # 所有段都覆盖过了：优先补「还没有可用帧」的段，再按候选池顺序补。
        picked_segments = {
            segment_index(float(item["time"]), duration, k) for item in results
        }
        ordered = sorted(
            (
                item
                for item in pool
                if all(abs(float(item["time"]) - value) >= 2.0 for value in used)
            ),
            key=lambda row: (
                0
                if segment_index(float(row["time"]), duration, k) not in picked_segments
                else 1,
                int(row.get("priority") or 3),
                float(row["time"]),
            ),
        )
        picked = self._take_times(ordered, used, wanted)
        if len(picked) < wanted:
            for index in range(k):
                if index in picked_segments:
                    continue
                point = self._segment_alternative(
                    index, duration, k, list(used) + picked
                )
                if point is None:
                    continue
                picked.append(point)
                if len(picked) >= wanted:
                    break
        return picked

    @staticmethod
    def _dense_subtitle_window(
        segments: Sequence[dict[str, Any]],
        duration: float,
        target: int,
        results: Sequence[dict[str, Any]],
        k: Optional[int] = None,
    ) -> Optional[tuple[float, float]]:
        """找一个「字幕最密、但还没有可用帧」的段，返回 30 秒左右的窗口。"""
        k = int(k or coverage_segment_count(target))
        picked = {segment_index(float(item["time"]), duration, k) for item in results}
        density: dict[int, int] = {index: 0 for index in range(k)}
        for item in segments:
            middle = (float(item["start"]) + float(item["end"])) / 2
            density[segment_index(middle, duration, k)] += len(
                str(item.get("text") or "")
            )
        candidates = [
            (score, index)
            for index, score in density.items()
            if index not in picked and score > 0
        ]
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _score, index = candidates[0]
        center = duration * (index + 0.5) / k
        span = min(30.0, duration, duration / k * 2)
        start = _clamp(center - span / 2, 0.0, max(0.0, duration - span))
        end = min(duration, start + span)
        if end - start < 3.0:
            return None
        return round(start, 3), round(end, 3)

    async def _hybrid_clip_probe(
        self,
        *,
        results: list[dict[str, Any]],
        media: Any,
        path: Path,
        out_dir: Path,
        duration: float,
        target: int,
        segments: Sequence[dict[str, Any]],
        used_times: list[float],
        history: list[dict[str, Any]],
        stats: dict[str, Any],
        deferred: list[str],
        segments_seen: Sequence[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """hybrid 在 high 档主动 clip 一次「字幕密集但还没看清」的段。"""
        # 只有显式 effort=high 才主动 clip；auto 的 high 只复用覆盖策略，
        # 避免超长片一上来就重编码 30 秒 4K 片段。
        if str(stats.get("effort")) != "high" or self._clip_left(stats) <= 0:
            return results
        # 覆盖轮没完成前不 clip：规则要求本轮抽帧只能来自未覆盖段。
        if self._uncovered_segments(segments_seen):
            return results
        if self._vlm_left(stats) <= 0:
            return results
        window = self._dense_subtitle_window(segments, duration, target, results, k)
        if window is None:
            return results
        start, end = window
        batch = await self._extract_clip(
            media,
            path,
            out_dir,
            start,
            end,
            float((stats.get("budget") or {}).get("clip_fps") or 1.0),
            stats,
            deferred,
        )
        if not batch:
            return results
        self._mark_seen(segments_seen, batch, duration, k)
        used_times.extend(item["time"] for item in batch)
        scored = await self._score_batch(batch, segments, duration, stats, deferred)
        history.extend(scored)
        return self._accepted(history, target, duration, k=k)

    @staticmethod
    def _usable_frame(item: dict[str, Any]) -> bool:
        return bool(
            item.get("usable")
            and not item.get("prefiltered")
            and int(item.get("score") or 0) >= 3
        )

    async def _retry_discarded_segments(
        self,
        *,
        media: Any,
        path: Path,
        out_dir: Path,
        duration: float,
        target: int,
        used_times: list[float],
        history: list[dict[str, Any]],
        segments: Sequence[dict[str, Any]],
        stats: dict[str, Any],
        deferred: list[str],
        segments_seen: Sequence[dict[str, Any]],
        k: int,
        check_cancel: Callable[[], None],
    ) -> list[int]:
        """某段抽回来的全是空白/重复：同段换时间点重抽，最多 3 次。

        仍拿不到可用帧就把段号记进 ``skipped_segments``。必须在覆盖轮
        （所有段都至少抽过 1 次）之后调用，否则会违反「先覆盖再重抽」。
        """
        skipped: list[int] = []
        usable_segments = {
            segment_index(float(item["time"]), duration, k)
            for item in history
            if self._usable_frame(item)
        }
        for index in range(k):
            entry = segments_seen[index] if index < len(segments_seen) else {}
            if int(entry.get("extracted") or 0) <= 0:
                continue
            if index in usable_segments:
                continue
            for attempt in range(3):
                check_cancel()
                if self._vlm_left(stats) <= 0:
                    break
                point = self._segment_alternative(
                    index, duration, k, used_times, attempt=attempt
                )
                if point is None:
                    break
                batch = await self._extract_and_screen(
                    media, path, out_dir, [point], stats, deferred
                )
                if not batch:
                    continue
                used_times.extend(item["time"] for item in batch)
                self._mark_seen(segments_seen, batch, duration, k)
                scored = await self._score_batch(
                    batch, segments, duration, stats, deferred
                )
                history.extend(scored)
                if any(self._usable_frame(item) for item in scored):
                    usable_segments.add(index)
                    break
            if index not in usable_segments:
                skipped.append(index)
        return skipped

    async def _enforce_coverage(
        self,
        *,
        results: list[dict[str, Any]],
        media: Any,
        path: Path,
        out_dir: Path,
        duration: float,
        target: int,
        used_times: list[float],
        history: list[dict[str, Any]],
        segments: Sequence[dict[str, Any]],
        gaps: Sequence[dict[str, Any]],
        stats: dict[str, Any],
        deferred: list[str],
        segments_seen: Sequence[dict[str, Any]],
        k: int,
        check_cancel: Callable[[], None],
    ) -> list[dict[str, Any]]:
        """最终结果的覆盖硬规则（代码强制，不依赖 VLM 听话）。

        1. ``target <= K`` 时同一段最多贡献 1 帧；
        2. 第一帧与最后一帧至少相距 1/3 时长（3 帧铺满全片，而不是挤在开头）。
        达不到就按「均匀摊到 K 段」的目标段补抽，最多 2 轮。
        """
        if not results or target <= 1:
            return results
        for _attempt in range(2):
            check_cancel()
            picked_segments = {
                segment_index(float(item["time"]), duration, k) for item in results
            }
            unique_ok = target > k or len(picked_segments) >= min(target, k)
            span = max(float(item["time"]) for item in results) - min(
                float(item["time"]) for item in results
            )
            span_ok = span >= duration / 3.0 - 0.05
            if unique_ok and span_ok:
                break
            if self._vlm_left(stats) <= 0:
                break
            wanted = self._spread_targets(k, target)
            missing = [index for index in wanted if index not in picked_segments]
            if not missing and not span_ok:
                # 极端情况：目标段都选中了但跨度还不够（例如都挤在中段），
                # 强行在首尾段各补一个点。
                missing = [index for index in (0, k - 1) if index not in picked_segments]
            if not missing:
                break
            times: list[float] = []
            for index in missing:
                point = self._segment_alternative(
                    index, duration, k, list(used_times) + times
                )
                if point is not None:
                    times.append(point)
            if not times:
                break
            batch = await self._extract_and_screen(
                media, path, out_dir, times, stats, deferred
            )
            if not batch:
                break
            used_times.extend(item["time"] for item in batch)
            self._mark_seen(segments_seen, batch, duration, k)
            scored = await self._score_batch(
                batch, segments, duration, stats, deferred
            )
            history.extend(scored)
            results = self._accepted(history, target, duration, k=k)
        return results

    # ---- 主流程 -------------------------------------------------------
    async def run(
        self,
        video_path: str | Path,
        *,
        target_frames: int = 3,
        planner: Optional[str] = None,
        effort: Any = EFFORT_DEFAULT,
        max_rounds: Optional[int] = None,
        subtitles: Any = None,
        scene_times: Optional[Sequence[float]] = None,
        subtitles_from: Optional[str] = None,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        is_canceled: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        target = max(1, min(10, int(target_frames or 3)))
        effort_name = normalize_effort(effort)
        mode = self._planner(planner)
        path = Path(video_path)
        if not path.is_file():
            raise ValueError(f"视频不存在：{path}")
        media = self._media()
        out_dir = (
            Path(self.cfg.frames_dir)
            / f"vision_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        deferred: list[str] = []

        def report(**payload: Any) -> None:
            if on_progress is None:
                return
            try:
                on_progress(payload)
            except Exception:  # noqa: BLE001 - 进度失败不影响挑帧
                pass

        def check_cancel() -> None:
            if is_canceled is not None and is_canceled():
                from ..core.errors import JobCanceledError  # noqa: PLC0415

                raise JobCanceledError("关键帧挑选任务已取消")

        report(stage="probe", percent=2, status="running")
        info = await media.call("probe", str(path))
        duration = info.get("duration")
        if not duration or float(duration) <= 0:
            raise ValueError(f"读不出视频时长，无法挑帧：{path}")
        duration = round(float(duration), 3)

        # effort=auto 必须拿到时长后才能定档；预算 / 分段数都依赖它。
        budget = effort_budget(target, effort_name, duration)
        profile = str(budget["profile"])
        rounds_limit = int(max_rounds) if max_rounds else budget["max_rounds"]
        rounds_limit = max(1, min(12, min(rounds_limit, int(budget["max_rounds"]))))
        k_segments = int(budget["coverage_segments"])
        stats: dict[str, Any] = {
            "candidates": 0,
            "vlm_calls": 0,
            "blank_skipped": 0,
            "duplicates_skipped": 0,
            "elapsed_ms": 0,
            "planner": mode,
            "effort": effort_name,
            "effort_profile": profile,
            "coverage_segments": k_segments,
            "max_rounds": rounds_limit,
            "subtitles": 0,
            "clip_calls": 0,
            "clip_frames": 0,
            "planned_candidates": 0,
            "silence_gaps": 0,
            "vlm_limit": int(budget["vlm_limit"]),
            "clip_limit": int(budget["clip_limit"]),
            "budget": budget,
        }

        # 1) 字幕：auto -> 服务端软字幕优先；否则调用方数组 -> 站点字幕 -> ASR
        report(stage="subtitles", percent=5, status="running")
        segments: list[dict[str, Any]] = []
        if isinstance(subtitles, str) and subtitles.strip().lower() == "auto":
            segments, subtitles_from = await self._auto_subtitles(path, report, deferred)
        elif subtitles is not None:
            segments = _normalize_segments(subtitles)
            subtitles_from = subtitles_from or ("provided" if segments else "none")
        else:
            picked = self._sidecar_subtitle(path)
            if picked is not None:
                segments = _load_segments_from_file(picked)
                subtitles_from = "site"
            elif self.enable_asr:
                segments, subtitle_error = await self._asr_segments(path, report)
                if subtitle_error:
                    deferred.append(subtitle_error)
                subtitles_from = "asr" if segments else "none"
            else:
                subtitles_from = "none"
        stats["subtitles"] = len(segments)

        # 1.5) 无人声空档：从字幕算，gap 中点必须进候选（代码强制，不靠模型自觉）。
        gaps = silence_gaps(segments, duration)
        stats["silence_gaps"] = len(gaps)
        segments_seen: list[dict[str, Any]] = [
            {"index": index, "times": [], "extracted": 0, "picked": 0}
            for index in range(k_segments)
        ]

        # 2) 场景边界（有就复用，没有才跑 ffmpeg）
        report(stage="scenes", percent=12, status="running")
        scenes = list(scene_times or [])
        if not scenes:
            try:
                scene_result = await media.call(
                    "detect_scenes", str(path), save_dir=str(self.cfg.scenes_dir)
                )
                scenes = [
                    float(item.get("time") or 0)
                    for item in (scene_result.get("scenes") or [])
                ]
            except Exception as exc:  # noqa: BLE001 - 场景只是线索
                logger.warning("场景检测失败（继续挑帧）：%s", exc)
                deferred.append(f"场景检测失败：{type(exc).__name__}: {exc}")

        used_times: list[float] = []
        history: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []

        # 3) 预筛一批候选（按 effort 预算决定数量；顺序是段轮转，天然分散）
        report(stage="candidates", percent=18, status="running")
        pool = plan_candidates(
            duration,
            segments,
            scenes,
            target_frames=target,
            limit=40,
            effort=effort_name,
            count=int(budget["candidates"]),
            gaps=gaps,
        )
        if mode == "agent" and self._vlm_left(stats) > 0:
            planned = await self._vlm_plan(duration, target, segments, scenes, stats, gaps)
            if planned:
                pool = self._merge_pools(planned, pool)
                pool = self._ensure_coverage(pool, duration, target, gaps, k_segments)
        stats["planned_candidates"] = len(pool)
        initial_count = min(int(budget["candidates"]), max(4, len(pool)))
        if mode == "agent":
            # 给规划 / 代理决策留出至少 2 次调用，别首轮就把预算烧光。
            initial_count = min(
                initial_count, max(3, int(budget["candidates"]) - 2)
            )
        # 覆盖轮：只从「还没抽过」的段取，每段 1 个；绝不先补第二张。
        initial_times = self._coverage_times(
            pool,
            duration,
            target,
            used_times,
            results,
            initial_count,
            k_segments,
            segments_seen,
        )
        batch = await self._extract_and_screen(
            media, path, out_dir, initial_times, stats, deferred
        )
        used_times.extend(item["time"] for item in batch)
        self._mark_seen(segments_seen, batch, duration, k_segments)
        check_cancel()

        # 4) 首轮评分
        report(stage="scoring", percent=25, status="running")
        scored = await self._score_batch(batch, segments, duration, stats, deferred)
        history.extend(scored)
        results = self._accepted(scored, target, duration, k=k_segments)
        rounds_used = 1

        # 4.5) high 档 hybrid 主动细看一段「字幕密集但还没看清」的段（clip 预算内）。
        if mode == "hybrid" and profile == "high":
            results = await self._hybrid_clip_probe(
                results=results,
                media=media,
                path=path,
                out_dir=out_dir,
                duration=duration,
                target=target,
                segments=segments,
                used_times=used_times,
                history=history,
                stats=stats,
                deferred=deferred,
                segments_seen=segments_seen,
                k=k_segments,
            )

        # 5) 循环补抽，直到够数或轮数用尽
        if mode == "agent":
            results, rounds_used, _agent_extra = await self._agent_loop(
                media=media,
                path=path,
                out_dir=out_dir,
                duration=duration,
                target=target,
                max_rounds=rounds_limit,
                segments=segments,
                scenes=scenes,
                used_times=used_times,
                history=history,
                results=results,
                rounds_used=rounds_used,
                stats=stats,
                deferred=deferred,
                report=report,
                check_cancel=check_cancel,
                gaps=gaps,
                segments_seen=segments_seen,
                k=k_segments,
            )
        else:
            while len(results) < target and rounds_used < rounds_limit:
                check_cancel()
                report(
                    stage="supplement",
                    percent=min(90, 30 + rounds_used * 10),
                    status="running",
                    round=rounds_used + 1,
                    picked=len(results),
                )
                need = min(MAX_SUPPLEMENT, max(MIN_SUPPLEMENT, target))
                # 未覆盖段优先；只有全部段都抽过才允许补同段第二张。
                times = self._coverage_times(
                    pool,
                    duration,
                    target,
                    used_times,
                    results,
                    need,
                    k_segments,
                    segments_seen,
                )
                if not times:
                    times = self._uniform_unseen(duration, used_times, need)
                if not times:
                    break
                batch = await self._extract_and_screen(
                    media, path, out_dir, times, stats, deferred
                )
                if not batch:
                    # 没有新画面可抽（或全被预筛掉），换一批均匀点再试一次。
                    times = self._uniform_unseen(duration, used_times, need, step=7.0)
                    batch = await self._extract_and_screen(
                        media, path, out_dir, times, stats, deferred
                    )
                used_times.extend(item["time"] for item in batch)
                self._mark_seen(segments_seen, batch, duration, k_segments)
                scored = await self._score_batch(
                    batch, segments, duration, stats, deferred
                )
                history.extend(scored)
                results = self._accepted(history, target, duration, k=k_segments)
                rounds_used += 1

        # 5.1) 覆盖收尾：target 已经够数也可能还有段没抽过；硬规则要求先铺满。
        while self._uncovered_segments(segments_seen) and self._vlm_left(stats) > 0:
            check_cancel()
            uncovered = self._uncovered_segments(segments_seen)
            times = self._coverage_times(
                pool,
                duration,
                target,
                used_times,
                results,
                len(uncovered),
                k_segments,
                segments_seen,
            )
            if not times:
                break
            batch = await self._extract_and_screen(
                media, path, out_dir, times, stats, deferred
            )
            if not batch:
                break
            used_times.extend(item["time"] for item in batch)
            self._mark_seen(segments_seen, batch, duration, k_segments)
            scored = await self._score_batch(
                batch, segments, duration, stats, deferred
            )
            history.extend(scored)
            results = self._accepted(history, target, duration, k=k_segments)
            rounds_used += 1

        # 5.2) 空白 / 重复段：同段换点重抽，最多 3 次；仍失败记 skipped_segments。
        skipped_segments = await self._retry_discarded_segments(
            media=media,
            path=path,
            out_dir=out_dir,
            duration=duration,
            target=target,
            used_times=used_times,
            history=history,
            segments=segments,
            stats=stats,
            deferred=deferred,
            segments_seen=segments_seen,
            k=k_segments,
            check_cancel=check_cancel,
        )

        # 5.5) 覆盖均衡强制：最终帧必须分散且首尾跨度 ≥ 1/3 时长。
        results = await self._enforce_coverage(
            results=results,
            media=media,
            path=path,
            out_dir=out_dir,
            duration=duration,
            target=target,
            used_times=used_times,
            history=history,
            segments=segments,
            gaps=gaps,
            stats=stats,
            deferred=deferred,
            segments_seen=segments_seen,
            k=k_segments,
            check_cancel=check_cancel,
        )

        # 6) 去重 + 降级 + 输出
        results = self._dedup_results(results, target)
        degraded = False
        degraded_reason: Optional[str] = None
        if not results:
            fallback = self._fallback_frames(history, target)
            if fallback:
                results = fallback
                degraded = True
                degraded_reason = "VLM 未产出有效帧，已回退到「方差 + 场景点」确定性结果"
        if results and all(item.get("degraded") for item in results):
            degraded = True
            degraded_reason = degraded_reason or "VLM 不可用，已回退到确定性结果"
        if deferred and not results:
            degraded_reason = "；".join(deferred[-3:])

        # 覆盖统计：最终每段选了几张（写进响应，方便核对「没有全挤在前 1/3」）。
        for item in results:
            index = segment_index(float(item["time"]), duration, k_segments)
            if 0 <= index < len(segments_seen):
                segments_seen[index]["picked"] += 1

        uncovered_segments = self._uncovered_segments(segments_seen)
        stats["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        stats["skipped_segments"] = skipped_segments
        stats["uncovered_segments"] = uncovered_segments
        if len(results) < target:
            stats["shortfall"] = target - len(results)
        report(stage="done", percent=100, status="done")
        payload = {
            "ok": True,
            "video": str(path),
            "duration": duration,
            "target": target,
            "effort": effort_name,
            "effort_profile": profile,
            "coverage_segments": k_segments,
            "budget": budget,
            "rounds_used": rounds_used,
            "frames": results,
            "silence_gaps": gaps,
            "segments_seen": [
                {
                    "index": item["index"],
                    "extracted": item["extracted"],
                    "picked": item["picked"],
                }
                for item in segments_seen
            ],
            "skipped_segments": skipped_segments,
            "uncovered_segments": uncovered_segments,
            "stats": stats,
        }
        if subtitles_from:
            payload["subtitles_from"] = subtitles_from
        if degraded:
            payload["degraded"] = True
            payload["degraded_reason"] = degraded_reason or "VLM 不可用"
        if deferred:
            payload["warnings"] = deferred[-5:]
        if len(results) < target:
            payload["detail"] = f"只凑到 {len(results)}/{target} 张有效帧"
        logger.info(
            "关键帧挑选完成：%s planner=%s 目标=%d 实得=%d 轮数=%d VLM调用=%d "
            "候选=%d 空白=%d 重复=%d 耗时=%dms%s",
            path.name,
            mode,
            target,
            len(results),
            rounds_used,
            stats["vlm_calls"],
            stats["candidates"],
            stats["blank_skipped"],
            stats["duplicates_skipped"],
            stats["elapsed_ms"],
            "（降级）" if degraded else "",
        )
        return payload

    # ---- 字幕 ---------------------------------------------------------
    @staticmethod
    def _subtitle_candidates(listing: dict[str, Any]) -> list[dict[str, Any]]:
        """把 /media/subtitles 的列表排成「优先中文、非 forced」的候选顺序。"""
        from ..core.subtitles import rank_soft_subtitle_sources  # noqa: PLC0415

        return rank_soft_subtitle_sources(listing)

    async def _auto_subtitles(
        self,
        path: Path,
        report: Callable[..., None],
        deferred: list[str],
    ) -> tuple[list[dict[str, Any]], str]:
        """``subtitles="auto"``：先走 /media/subtitles，有软字幕就不跑 ASR。"""
        report(stage="subtitles", percent=5, status="running")
        listing: dict[str, Any] = {}
        try:
            listing = await self._media().call("list_subtitles", str(path))
        except Exception as exc:  # noqa: BLE001 - 探测失败退回 ASR / 站点字幕
            logger.warning("字幕探测失败（尝试 ASR）：%s", exc)
            deferred.append(f"字幕探测失败：{type(exc).__name__}: {exc}")

        for candidate in self._subtitle_candidates(listing):
            try:
                if candidate["source"] == "embedded":
                    result = await self._media().call(
                        "extract_subtitle_segments",
                        str(path),
                        index=int(candidate["index"]),
                    )
                else:
                    result = await self._media().call(
                        "extract_subtitle_segments",
                        None,
                        sidecar=str(candidate["path"]),
                    )
                segments = _normalize_segments(result.get("segments"))
                if segments:
                    report(
                        stage="subtitles",
                        percent=7,
                        status="running",
                        source=candidate["source"],
                        count=len(segments),
                    )
                    return segments, candidate["source"]
            except Exception as exc:  # noqa: BLE001 - 某条字幕坏了就换下一条
                logger.warning(
                    "字幕抽取失败（%s index/path=%s）：%s",
                    candidate["source"],
                    candidate.get("index") or candidate.get("path"),
                    exc,
                )
                deferred.append(f"字幕抽取失败：{type(exc).__name__}: {exc}")

        if self.enable_asr:
            segments, subtitle_error = await self._asr_segments(path, report)
            if subtitle_error:
                deferred.append(subtitle_error)
            return segments, ("asr" if segments else "none")
        return [], "none"

    def _sidecar_subtitle(self, video: Path) -> Optional[Path]:
        """同目录的站点字幕：``<视频名>*.srt`` / ``*.vtt``。"""
        try:
            from ..core.subtitles import pick_subtitle  # noqa: PLC0415

            # 文件名里常有 ``[BV1xx]`` / ``[]``，直接拼 glob 会被当成字符类，
            # 必须用 glob.escape 转义，否则同目录字幕永远找不到。
            import glob as _glob  # noqa: PLC0415

            pattern = _glob.escape(str(video.stem)) + "*"
            candidates = [
                str(item)
                for item in video.parent.glob(pattern)
                if item.suffix.lower() in (".srt", ".vtt")
            ]
            picked = pick_subtitle(candidates)
            return Path(picked) if picked else None
        except Exception:  # noqa: BLE001
            return None

    async def _asr_segments(
        self, path: Path, report: Callable[..., None]
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """没有字幕时用 /asr/segments 同款链路；失败只记 warning。"""
        try:
            from ..core.asr_engines import adapter_for, normalize_engine  # noqa: PLC0415
            from ..core.segments import normalize_segments  # noqa: PLC0415

            engine = normalize_engine("auto", self.cfg, need_timestamps=True)
            media = self._media()
            report(stage="asr", percent=8, status="running")
            wav_path: Optional[Path] = None
            try:
                if path.suffix.lower() != ".wav":
                    converted = await media.call("to_wav16k", str(path))
                    wav_path = Path(converted["path"])
                adapter = adapter_for(self.registry, engine)
                # 必须调适配器的 infer_segments；原生 whisper_engine 模块没有
                # 这个函数，直接 get_module().infer_segments 会 AttributeError。
                result = await adapter.infer_segments(str(wav_path or path), "auto")
            finally:
                if wav_path is not None:
                    wav_path.unlink(missing_ok=True)
            segments = normalize_segments(
                result.get("segments"), duration=result.get("duration")
            )
            return _normalize_segments(segments), None
        except Exception as exc:  # noqa: BLE001 - ASR 失败不影响场景+均匀抽帧
            logger.warning("没有字幕且 ASR 分段失败（继续挑帧）：%s", exc)
            return [], f"ASR 分段失败：{type(exc).__name__}: {exc}"

    # ---- 抽帧 / 预筛 --------------------------------------------------
    async def _extract_and_screen(
        self,
        media: Any,
        video: Path,
        out_dir: Path,
        times: Sequence[float],
        stats: dict[str, Any],
        deferred: list[str],
    ) -> list[dict[str, Any]]:
        """抽一批帧并做亮度/方差 + 感知哈希预筛。"""
        points = sorted({round(float(item), 3) for item in times})
        if not points:
            return []
        try:
            frames = await media.call(
                "extract_frames",
                str(video),
                str(out_dir),
                times=points,
                width=FRAME_WIDTH,
            )
        except Exception as exc:  # noqa: BLE001 - 抽帧失败由上层降级
            logger.warning("抽帧失败：%s", exc)
            deferred.append(f"抽帧失败：{type(exc).__name__}: {exc}")
            return []

        return self._screen_frames(frames, stats)

    def _screen_frames(
        self, frames: Sequence[dict[str, Any]], stats: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """对抽回来的帧做亮度/方差 + 感知哈希预筛（抽帧来源无关）。"""
        kept: list[dict[str, Any]] = []
        threshold = int(self.cfg.vision_phash_distance or DEFAULT_PHASH_DISTANCE)
        for frame in frames:
            item = {
                "time": round(float(frame.get("time_sec") or 0.0), 3),
                "path": str(frame.get("path")),
            }
            stats["candidates"] += 1
            info = _analyze_image(item["path"])
            item["image"] = info
            if info.get("blank"):
                stats["blank_skipped"] += 1
                item.update(
                    {
                        "usable": False,
                        "kind": "blank",
                        "score": 0,
                        "ocr_text": "",
                        "summary": "纯色/低方差画面，预筛判为空白",
                        "why": "亮度或方差过低，跳过 VLM",
                        "prefiltered": "blank",
                    }
                )
                kept.append(item)
                continue
            duplicate = None
            for other in kept:
                if other.get("prefiltered") == "blank":
                    continue
                other_hash = (other.get("image") or {}).get("phash")
                if other_hash is not None and info.get("phash") is not None:
                    if _hamming(other_hash, info["phash"]) <= threshold:
                        duplicate = other
                        break
            if duplicate is not None:
                stats["duplicates_skipped"] += 1
                item.update(
                    {
                        "usable": False,
                        "kind": "duplicate",
                        "score": 0,
                        "ocr_text": "",
                        "summary": f"与 {duplicate['time']:.1f}s 构图几乎相同",
                        "why": "感知哈希去重",
                        "prefiltered": "duplicate",
                        "duplicate_of": duplicate["time"],
                    }
                )
                kept.append(item)
                continue
            kept.append(item)
        return kept

    async def _extract_clip(
        self,
        media: Any,
        video: Path,
        out_dir: Path,
        start: float,
        end: float,
        fps: float,
        stats: dict[str, Any],
        deferred: list[str],
    ) -> list[dict[str, Any]]:
        """``extract_clip`` 工具：在 ``[start,end]`` 内按 fps 连续抽帧并预筛。

        预算受 ``stats["clip_limit"]`` 限制；窗口超过 60 秒或次数用尽时
        不调用 ffmpeg，直接把原因写进 warnings（由上层决定是否改成普通抽帧）。
        """
        if self._clip_left(stats) <= 0:
            deferred.append("clip 次数已达 effort 上限，跳过")
            return []
        try:
            start_value = float(start)
            end_value = float(end)
            rate = float(fps)
        except (TypeError, ValueError):
            deferred.append(f"extract_clip 参数不是数字：start={start!r} end={end!r} fps={fps!r}")
            return []
        if end_value <= start_value or start_value < 0:
            deferred.append(f"extract_clip 时间窗口非法：{start_value}~{end_value}")
            return []
        if end_value - start_value > MAX_CLIP_SECONDS:
            deferred.append(
                f"extract_clip 窗口太长（{end_value - start_value:.1f}s > {MAX_CLIP_SECONDS:.0f}s）"
            )
            return []
        rate = _clamp(rate or 1.0, CLIP_MIN_FPS, CLIP_MAX_FPS)
        clip_dir = out_dir / "clips"
        try:
            frames = await media.call(
                "extract_frames_window",
                str(video),
                start_value,
                end_value,
                fps=rate,
                width=FRAME_WIDTH,
                out_dir=str(clip_dir),
            )
        except Exception as exc:  # noqa: BLE001 - 窗口抽帧失败不致命
            logger.warning("extract_clip 失败：%s", exc)
            deferred.append(f"extract_clip 失败：{type(exc).__name__}: {exc}")
            return []
        stats["clip_calls"] = int(stats.get("clip_calls") or 0) + 1
        batch = self._screen_frames(frames, stats)
        stats["clip_frames"] = int(stats.get("clip_frames") or 0) + len(batch)
        return batch

    # ---- VLM 调用 -----------------------------------------------------
    async def _vlm_plan(
        self,
        duration: float,
        target: int,
        segments: Sequence[dict[str, Any]],
        scenes: Sequence[float],
        stats: dict[str, Any],
        gaps: Sequence[dict[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """用 vlm_frame_plan.md 让模型规划一批候选时间点（纯文本）。"""
        if self._vlm_left(stats) <= 0:
            return []
        try:
            vision = self._vision()
            prompt = load_prompt(self.cfg, "vlm_frame_plan.md")
            user = json.dumps(
                {
                    "duration_sec": duration,
                    "target_frames": target,
                    "effort": stats.get("effort") or EFFORT_DEFAULT,
                    "subtitles": _trim_subtitles(segments, limit=24, text_limit=60),
                    "silence_gaps": [
                        {"start": item["start"], "end": item["end"]}
                        for item in list(gaps)[:12]
                    ],
                    "scene_times": [round(float(item), 2) for item in list(scenes)[:30]],
                },
                ensure_ascii=False,
            )
            outcome = await vision.describe_text(
                user, system_prompt=prompt, max_tokens=800
            )
            stats["vlm_calls"] += 1
            data = parse_json_loose(outcome.get("text") or "")
            if not data:
                return []
            planned: list[dict[str, Any]] = []
            for index, item in enumerate(data.get("candidates") or []):
                try:
                    point = float(item.get("time"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if not 0 <= point <= duration:
                    continue
                try:
                    priority = int(item.get("priority") or 2)
                except (TypeError, ValueError):
                    priority = 2
                planned.append(
                    {
                        "time": round(point, 3),
                        "priority": max(1, min(3, priority)),
                        "reason": str(item.get("reason") or "VLM 规划")[:60],
                        "segment": segment_index(
                            point, duration, coverage_segment_count(target)
                        ),
                    }
                )
                if len(planned) >= 40:
                    break
            return planned
        except Exception as exc:  # noqa: BLE001 - 规划失败用确定性候选
            logger.warning("VLM 规划失败（改用确定性候选）：%s", exc)
            return []

    async def _score_batch(
        self,
        batch: Sequence[dict[str, Any]],
        segments: Sequence[dict[str, Any]],
        duration: float,
        stats: dict[str, Any],
        deferred: list[str],
    ) -> list[dict[str, Any]]:
        """对一批帧逐张打分；预筛已判 blank/重复的不再调 VLM。"""
        scored: list[dict[str, Any]] = []
        failures = 0
        for item in batch:
            if item.get("prefiltered"):
                scored.append(item)
                continue
            if self._vlm_left(stats) <= 0:
                # effort 预算用尽：剩下的帧不再调 VLM，按「未评分」交给降级逻辑。
                item.update(
                    {
                        "usable": False,
                        "kind": "other",
                        "score": 0,
                        "ocr_text": "",
                        "summary": "VLM 调用预算已用尽，本帧未评分",
                        "why": f"effort={stats.get('effort')} 的 VLM 上限为 {stats.get('vlm_limit')}",
                        "budget_exhausted": True,
                    }
                )
                scored.append(item)
                continue
            if failures >= 2:
                # 连续失败通常意味着模型/进程整体不可用，别再逐张浪费超时。
                item.update(
                    {
                        "usable": False,
                        "kind": "other",
                        "score": 0,
                        "ocr_text": "",
                        "summary": "VLM 不可用，跳过评分",
                        "why": "连续两次调用失败，提前终止本批评分",
                        "degraded": True,
                    }
                )
                scored.append(item)
                continue
            try:
                result = await self._score_frame(item, segments, duration)
                stats["vlm_calls"] += 1
                item.update(result)
                failures = 0
            except Exception as exc:  # noqa: BLE001 - 单帧失败按无效处理
                failures += 1
                logger.warning("单帧 VLM 评分失败（%.1fs）：%s", item["time"], exc)
                deferred.append(f"VLM 评分失败：{type(exc).__name__}: {exc}")
                item.update(
                    {
                        "usable": False,
                        "kind": "other",
                        "score": 0,
                        "ocr_text": "",
                        "summary": "VLM 评分失败",
                        "why": f"{type(exc).__name__}: {exc}"[:60],
                        "degraded": True,
                    }
                )
            scored.append(item)
        return scored

    async def _score_frame(
        self,
        item: dict[str, Any],
        segments: Sequence[dict[str, Any]],
        duration: float,
    ) -> dict[str, Any]:
        vision = self._vision()
        prompt = load_prompt(self.cfg, "vlm_frame_score.md")
        nearby = _subtitle_nearby(segments, item["time"])
        user = json.dumps(
            {
                "time": item["time"],
                "duration_sec": duration,
                "subtitle_nearby": nearby,
                # 无人声段提示：明确告诉模型「没字幕不代表没内容」，
                # 和 vlm_frame_score.md 的判定规则第 4 条一致。
                "silence_gap": not nearby,
                "gap_note": (
                    "该时间点位于无人声空档：画面可能在演示操作/展示成品/放图表，"
                    "无人声不等于没内容，请按真实画面判定"
                    if not nearby
                    else ""
                ),
            },
            ensure_ascii=False,
        )
        outcome = await vision.describe_image(
            item["path"], user, system_prompt=prompt, max_tokens=400
        )
        data = parse_json_loose(outcome.get("text") or "")
        if not data:
            raise ValueError(f"评分结果不是 JSON：{(outcome.get('text') or '')[:120]}")
        return _normalize_score(data, item["time"])

    async def _agent_loop(
        self,
        *,
        media: Any,
        path: Path,
        out_dir: Path,
        duration: float,
        target: int,
        max_rounds: int,
        segments: Sequence[dict[str, Any]],
        scenes: Sequence[float],
        used_times: list[float],
        history: list[dict[str, Any]],
        results: list[dict[str, Any]],
        rounds_used: int,
        stats: dict[str, Any],
        deferred: list[str],
        report: Callable[..., None],
        check_cancel: Callable[[], None],
        gaps: Sequence[dict[str, Any]] = (),
        segments_seen: Sequence[dict[str, Any]] = (),
        k: Optional[int] = None,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """多轮代理：模型输出 extract / extract_clip / final，程序执行后回灌画面描述。"""
        vision = self._vision()
        prompt = load_prompt(self.cfg, "vlm_frame_agent.md")
        effort_name = str(stats.get("effort") or EFFORT_DEFAULT)
        budget = stats.get("budget") or effort_budget(
            target, effort_name, duration
        )
        k = int(k or budget.get("coverage_segments") or coverage_segment_count(target))
        extra = 0
        # 即使首轮候选已经够数，也至少让代理做一次 final 决策：
        # 这样 agent planner 真的走了协议，而不是悄悄退化成 hybrid。
        first_decision = True
        last_round: Optional[dict[str, Any]] = {
            "requested_times": [item["time"] for item in history[:8]],
            "frames": [_agent_frame_brief(item) for item in history[:8]],
        }
        while rounds_used < max_rounds and (len(results) < target or first_decision):
            first_decision = False
            check_cancel()
            if self._vlm_left(stats) <= 0:
                deferred.append("VLM 调用预算已用尽，代理提前结束")
                break
            report(
                stage="agent",
                percent=min(90, 30 + rounds_used * 10),
                status="running",
                round=rounds_used + 1,
            )
            user = json.dumps(
                {
                    "duration_sec": duration,
                    "effort": effort_name,
                    "effort_note": budget.get("note"),
                    "subtitles": _trim_subtitles(segments, limit=30, text_limit=60),
                    "silence_gaps": [
                        {"start": item["start"], "end": item["end"]}
                        for item in list(gaps)[:12]
                    ],
                    "segments_seen": [
                        {
                            "index": item["index"],
                            "times": (item.get("times") or [])[-6:],
                            "picked": int(item.get("picked") or 0),
                        }
                        for item in segments_seen
                    ],
                    "target_frames": target,
                    "max_rounds": max_rounds,
                    "round": rounds_used + 1,
                    "clip_budget": {
                        "limit": int(budget.get("clip_limit") or 0),
                        "used": int(stats.get("clip_calls") or 0),
                    },
                    "already_picked": [
                        {
                            "time": item["time"],
                            "usable": bool(item.get("usable")),
                            "kind": item.get("kind"),
                            "summary": item.get("summary"),
                        }
                        for item in results
                    ],
                    "last_round": last_round,
                },
                ensure_ascii=False,
            )
            try:
                outcome = await vision.describe_text(
                    user, system_prompt=prompt, max_tokens=700
                )
                stats["vlm_calls"] += 1
            except Exception as exc:  # noqa: BLE001 - 代理失败就走降级
                deferred.append(f"VLM agent 调用失败：{type(exc).__name__}: {exc}")
                break
            data = parse_json_loose(outcome.get("text") or "")
            if not data:
                deferred.append("VLM agent 返回无法解析的 JSON")
                break
            action = str(data.get("action") or "").strip().lower()

            if action == "extract_clip":
                # 代理协议扩展：{action:"extract_clip", start, end, fps}
                clip_start = data.get("start")
                clip_end = data.get("end")
                clip_fps = data.get("fps", budget.get("clip_fps") or 1.0)
                uncovered = self._uncovered_segments(segments_seen)
                if uncovered:
                    # 覆盖轮未完成：忽略 clip 请求，只从没抽过的段取 1 个点。
                    deferred.append("覆盖轮未完成，agent 的 extract_clip 已改为未覆盖段抽帧")
                    times = self._coverage_times(
                        plan_candidates(
                            duration,
                            segments,
                            scenes,
                            target_frames=target,
                            effort=effort_name,
                            count=int(budget.get("candidates") or 12),
                            gaps=gaps,
                        ),
                        duration,
                        target,
                        used_times,
                        results,
                        min(4, len(uncovered)),
                        k,
                        segments_seen,
                    )
                    batch = await self._extract_and_screen(
                        media, path, out_dir, times, stats, deferred
                    )
                    action_label = "coverage_extract"
                else:
                    batch = await self._extract_clip(
                        media,
                        path,
                        out_dir,
                        float(clip_start) if clip_start is not None else -1.0,
                        float(clip_end) if clip_end is not None else -1.0,
                        float(clip_fps or 1.0),
                        stats,
                        deferred,
                    )
                    action_label = "extract_clip"
                    if not batch:
                        # clip 不可用（次数用尽/窗口非法/ffmpeg 失败）：退回中点抽帧。
                        fallback = []
                        try:
                            middle = (float(clip_start) + float(clip_end)) / 2
                            fallback = [round(middle, 3)]
                        except (TypeError, ValueError):
                            fallback = []
                        if fallback:
                            batch = await self._extract_and_screen(
                                media, path, out_dir, fallback, stats, deferred
                            )
                if not batch:
                    break
                self._mark_seen(segments_seen, batch, duration, k)
                used_times.extend(item["time"] for item in batch)
                scored = await self._score_batch(
                    batch, segments, duration, stats, deferred
                )
                history.extend(scored)
                results = self._accepted(history, target, duration, k=k)
                rounds_used += 1
                extra += 1
                last_round = {
                    "action": action_label,
                    "window": [clip_start, clip_end],
                    "requested_times": [item["time"] for item in batch],
                    "frames": [
                        _agent_frame_brief(item)
                        for item in scored
                        if not item.get("prefiltered")
                    ],
                }
                continue

            if action == "final":
                wanted = data.get("frames") or []
                chosen: list[dict[str, Any]] = []
                for entry in wanted:
                    try:
                        point = float(entry.get("time"))
                    except (TypeError, ValueError, AttributeError):
                        continue
                    match = _nearest_scored(history, point)
                    if match is None or not match.get("usable"):
                        # final 里点名但还没抽过的位置：覆盖轮没完成时先补未覆盖段。
                        if not 0 <= point <= duration:
                            continue
                        uncovered = self._uncovered_segments(segments_seen)
                        times = (
                            self._coverage_times(
                                plan_candidates(
                                    duration,
                                    segments,
                                    scenes,
                                    target_frames=target,
                                    effort=effort_name,
                                    count=int(budget.get("candidates") or 12),
                                    gaps=gaps,
                                ),
                                duration,
                                target,
                                used_times,
                                results,
                                1,
                                k,
                                segments_seen,
                            )
                            if uncovered
                            else [point]
                        )
                        batch = await self._extract_and_screen(
                            media, path, out_dir, times, stats, deferred
                        )
                        used_times.extend(item["time"] for item in batch)
                        self._mark_seen(segments_seen, batch, duration, k)
                        scored = await self._score_batch(
                            batch, segments, duration, stats, deferred
                        )
                        history.extend(scored)
                        match = _nearest_scored(scored, point) or _nearest_scored(
                            history, point
                        )
                    if match is not None and match.get("usable"):
                        picked = dict(match)
                        picked["why"] = str(entry.get("why") or picked.get("why") or "")[:80]
                        chosen.append(picked)
                if chosen:
                    results = self._accepted(chosen + results, target, duration, k=k)
                break

            # 继续抽：extract。覆盖轮未完成时忽略模型时间点，强制走未覆盖段。
            if self._uncovered_segments(segments_seen):
                deferred.append("覆盖轮未完成，agent 的 extract 已改为未覆盖段抽帧")
                times = self._coverage_times(
                    plan_candidates(
                        duration,
                        segments,
                        scenes,
                        target_frames=target,
                        effort=effort_name,
                        count=int(budget.get("candidates") or 12),
                        gaps=gaps,
                    ),
                    duration,
                    target,
                    used_times,
                    results,
                    min(4, len(self._uncovered_segments(segments_seen))),
                    k,
                    segments_seen,
                )
            else:
                times = _valid_extract_times(
                    data.get("times"),
                    duration=duration,
                    used=used_times,
                    limit=4,
                )
            if not times:
                deferred.append("VLM agent 的 extract 时间点非法，改用确定性补抽")
                times = self._supplement_times(
                    plan_candidates(
                        duration,
                        segments,
                        scenes,
                        target_frames=target,
                        effort=effort_name,
                        count=int(budget.get("candidates") or 12),
                        gaps=gaps,
                    ),
                    duration,
                    target,
                    used_times,
                    results,
                    min(4, max(2, target)),
                    gaps=gaps,
                    k=k,
                )
            if not times:
                break
            batch = await self._extract_and_screen(
                media, path, out_dir, times, stats, deferred
            )
            used_times.extend(item["time"] for item in batch)
            self._mark_seen(segments_seen, batch, duration, k)
            scored = await self._score_batch(
                batch, segments, duration, stats, deferred
            )
            history.extend(scored)
            results = self._accepted(history, target, duration, k=k)
            rounds_used += 1
            extra += 1
            last_round = {
                "requested_times": times,
                "frames": [_agent_frame_brief(item) for item in scored],
            }
        return results, rounds_used, extra

    # ---- 选择 / 去重 / 降级 -------------------------------------------
    @staticmethod
    def _merge_pools(
        first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for item in list(first) + list(second):
            point = float(item["time"])
            if any(abs(point - other["time"]) < 2.0 for other in merged):
                continue
            merged.append(item)
        merged.sort(key=lambda item: (int(item.get("priority") or 3), item["time"]))
        return merged[:40]

    @staticmethod
    def _take_times(
        pool: Sequence[dict[str, Any]], used: Sequence[float], count: int
    ) -> list[float]:
        picked: list[float] = []
        for item in pool:
            point = round(float(item["time"]), 3)
            if any(abs(point - value) < 2.0 for value in used):
                continue
            if any(abs(point - value) < 2.0 for value in picked):
                continue
            picked.append(point)
            if len(picked) >= count:
                break
        return picked

    @staticmethod
    def _uniform_unseen(
        duration: float,
        used: Sequence[float],
        count: int,
        *,
        step: float = 4.0,
    ) -> list[float]:
        """按固定步长扫一遍还没抽过的位置（补抽兜底）。"""
        picked: list[float] = []
        point = min(1.0, duration / 4)
        while point < duration - 0.2 and len(picked) < count:
            if all(abs(point - value) >= 2.0 for value in used) and all(
                abs(point - value) >= 2.0 for value in picked
            ):
                picked.append(round(point, 3))
            point += step
        return picked

    def _accepted(
        self,
        history: Sequence[dict[str, Any]],
        target: int,
        duration: Optional[float] = None,
        *,
        k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """按「usable 且 score>=3」筛，优先不同段，再按分数取前 N。

        硬规则：``target <= K`` 时同一段最多贡献 1 帧（不靠 VLM 听话）；
        段数多于目标帧时按段均匀取，保证 3 帧是「全片 0/中间/结尾」而不是
        「最早的三段」。只有 ``target > K`` 时才允许同段补第二张。
        """
        threshold = int(self.cfg.vision_phash_distance or DEFAULT_PHASH_DISTANCE)
        valid = [
            item
            for item in history
            if item.get("usable")
            and int(item.get("score") or 0) >= 3
            and not item.get("prefiltered")
        ]
        valid.sort(key=lambda item: (-int(item.get("score") or 0), item["time"]))
        k = int(k or coverage_segment_count(target))
        chosen: list[dict[str, Any]] = []
        chosen_segments: set[int] = set()
        strict_segments = target <= k

        def take(item: dict[str, Any]) -> bool:
            index = (
                segment_index(float(item["time"]), duration, k) if duration else None
            )
            if strict_segments and index is not None and index in chosen_segments:
                return False
            if any(_same_frame(item, other, threshold) for other in chosen):
                return False
            chosen.append(item)
            if index is not None:
                chosen_segments.add(index)
            return True

        if duration:
            # 第一轮：每段只留分数最高的一张；段数多于目标帧时按段均匀取。
            per_segment: dict[int, dict[str, Any]] = {}
            for item in valid:
                index = segment_index(float(item["time"]), duration, k)
                per_segment.setdefault(index, item)
            available = sorted(per_segment)
            if target > 1 and len(available) > target:
                spread = [
                    round(position * (len(available) - 1) / (target - 1))
                    for position in range(target)
                ]
                for position in spread:
                    take(per_segment[available[int(position)]])
            else:
                for index in available:
                    if take(per_segment[index]) and len(chosen) >= target:
                        break
        if len(chosen) < target and not strict_segments:
            for item in valid:
                if item in chosen:
                    continue
                if take(item) and len(chosen) >= target:
                    break
        chosen.sort(key=lambda item: item["time"])
        return [_public_frame(item) for item in chosen]

    def _dedup_results(
        self, results: Sequence[dict[str, Any]], target: int
    ) -> list[dict[str, Any]]:
        threshold = int(self.cfg.vision_phash_distance or DEFAULT_PHASH_DISTANCE)
        kept: list[dict[str, Any]] = []
        for item in results:
            info = _analyze_image(item["path"]) if Path(item["path"]).is_file() else {}
            record = dict(item)
            record["_phash"] = info.get("phash")
            if any(_same_frame(record, other, threshold) for other in kept):
                continue
            kept.append(record)
        for item in kept:
            item.pop("_phash", None)
        kept.sort(key=lambda item: item["time"])
        return kept[:target]

    @staticmethod
    def _fallback_frames(
        history: Sequence[dict[str, Any]], target: int
    ) -> list[dict[str, Any]]:
        """VLM 全线失败时的确定性结果：非空白、非重复，按画面方差排序。"""
        candidates = [
            item
            for item in history
            if not item.get("prefiltered")
            and (item.get("image") or {}).get("std") is not None
        ]
        candidates.sort(
            key=lambda item: -float((item.get("image") or {}).get("std") or 0)
        )
        chosen = candidates[:target]
        chosen.sort(key=lambda item: item["time"])
        frames: list[dict[str, Any]] = []
        for item in chosen:
            frames.append(
                {
                    "time": item["time"],
                    "path": item["path"],
                    "score": 3,
                    "kind": "scene",
                    "ocr_text": "",
                    "summary": "确定性回退帧（方差较高，未调用 VLM）",
                    "why": "VLM 不可用，按画面方差 + 场景点回退",
                    "confidence": "low",
                    "degraded": True,
                }
            )
        return frames


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _norm_text(value: Any) -> str:
    """去掉空白/标点，只留文字，用于「同一张幻灯片」的容错比较。"""
    text = str(value or "")
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def _text_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    from difflib import SequenceMatcher  # noqa: PLC0415

    return SequenceMatcher(None, left, right).ratio()


def _same_frame(
    item: dict[str, Any], other: dict[str, Any], hash_threshold: int
) -> bool:
    """判断两帧是否「同一张幻灯片 / 几乎同一构图」。

    三条独立信号，命中任意一条就算重复：
    1. 感知哈希汉明距离 <= 阈值；
    2. OCR 文本（去空白标点后）相似度 >= 0.9 且长度足够；
    3. summary 完全相同（同一幻灯片被 VLM 描述成同一句话）。
    """
    left_hash = (item.get("image") or {}).get("phash") or item.get("_phash")
    right_hash = (other.get("image") or {}).get("phash") or other.get("_phash")
    if left_hash is not None and right_hash is not None:
        if _hamming(int(left_hash), int(right_hash)) <= hash_threshold:
            return True
    left_ocr = _norm_text(item.get("ocr_text"))
    right_ocr = _norm_text(other.get("ocr_text"))
    if len(left_ocr) >= 8 and len(right_ocr) >= 8:
        if left_ocr == right_ocr or _text_ratio(left_ocr, right_ocr) >= 0.9:
            return True
    left_summary = _norm_text(item.get("summary"))
    right_summary = _norm_text(other.get("summary"))
    if len(left_summary) >= 8 and left_summary == right_summary:
        return True
    return False


def _normalize_score(data: dict[str, Any], time_sec: float) -> dict[str, Any]:
    """把 VLM 评分 JSON 归一到固定字段，缺字段给安全默认值。"""
    try:
        score = int(float(data.get("score") or 0))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(5, score))
    usable = bool(data.get("usable", score >= 3))
    kind = str(data.get("kind") or "other").strip().lower() or "other"
    if kind not in (
        "slide",
        "chart",
        "code",
        "table",
        "demo",
        "face",
        "scene",
        "map",
        "blank",
        "transition",
        "other",
    ):
        kind = "other"
    return {
        "time": round(float(time_sec), 3),
        "usable": usable,
        "kind": kind,
        "score": score,
        "ocr_text": str(data.get("ocr_text") or "")[:1000],
        "summary": str(data.get("summary") or "")[:120],
        "why": str(data.get("why") or "")[:120],
        # 摘要太短或带「大概/可能」等模糊词 -> low，调用方可以自己再看原图。
        "confidence": _summary_confidence(str(data.get("summary") or "")),
    }


def _public_frame(item: dict[str, Any]) -> dict[str, Any]:
    summary = item.get("summary") or ""
    return {
        "time": round(float(item["time"]), 3),
        "path": item["path"],
        "score": int(item.get("score") or 0),
        "kind": item.get("kind") or "other",
        "ocr_text": item.get("ocr_text") or "",
        "summary": summary,
        "why": item.get("why") or "",
        "confidence": item.get("confidence") or _summary_confidence(summary),
    }


def _agent_frame_brief(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": item["time"],
        "usable": bool(item.get("usable")),
        "kind": item.get("kind"),
        "score": item.get("score"),
        "ocr": (item.get("ocr_text") or "")[:200],
        "summary": (item.get("summary") or "")[:60],
    }


def _trim_subtitles(
    segments: Sequence[dict[str, Any]],
    *,
    limit: int = 100,
    text_limit: int = 120,
) -> list[dict[str, Any]]:
    """控制 prompt 体积；超长时均匀抽样但保留首尾。"""
    items = list(segments)
    if len(items) <= limit:
        return [
            {"start": item["start"], "end": item["end"], "text": item["text"][:text_limit]}
            for item in items
        ]
    step = len(items) / float(limit)
    picked = [items[min(len(items) - 1, int(index * step))] for index in range(limit)]
    return [
        {"start": item["start"], "end": item["end"], "text": item["text"][:text_limit]}
        for item in picked
    ]


def _valid_extract_times(
    raw: Any,
    *,
    duration: float,
    used: Sequence[float],
    limit: int = 4,
) -> list[float]:
    if not isinstance(raw, (list, tuple)):
        return []
    picked: list[float] = []
    for value in raw:
        try:
            point = float(value)
        except (TypeError, ValueError):
            continue
        point = _clamp(point, 0.0, max(0.0, duration - 0.2))
        if any(abs(point - item) < 2.0 for item in used):
            continue
        if any(abs(point - item) < 2.0 for item in picked):
            continue
        picked.append(round(point, 3))
        if len(picked) >= limit:
            break
    return picked


def _nearest_scored(
    history: Sequence[dict[str, Any]], point: float
) -> Optional[dict[str, Any]]:
    best: Optional[dict[str, Any]] = None
    best_distance = 999.0
    for item in history:
        distance = abs(float(item["time"]) - point)
        if distance < best_distance:
            best = item
            best_distance = distance
    return best if best is not None and best_distance <= 3.0 else None


__all__ = [
    "FramesAgent",
    "KEYWORDS",
    "load_prompt",
    "parse_json_loose",
    "plan_candidates",
]
