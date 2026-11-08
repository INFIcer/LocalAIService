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

import asyncio
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
) -> list[dict[str, Any]]:
    """按「字幕关键词 > 场景切换 > 均匀补充」生成候选点。

    返回 ``[{time, priority, reason}]``，priority 越小越优先。
    """
    if not duration or duration <= 0:
        return []
    total = max(1, min(40, int(limit)))
    # 短视频放宽最小间隔，否则凑不出足够候选。
    min_gap = 3.0 if duration >= 30 else max(1.0, duration / max(4, target_frames * 4))

    scored: list[tuple[int, float, str]] = []
    for item in segments:
        text = str(item.get("text") or "")
        hit = next((word for word in KEYWORDS if word in text), None)
        if hit is None:
            continue
        point = (float(item["start"]) + float(item["end"])) / 2
        scored.append((1, _clamp(point, 0.5, duration - 0.2), f"字幕出现「{hit}」"))

    for value in scene_times:
        try:
            point = float(value)
        except (TypeError, ValueError):
            continue
        scored.append((2, _clamp(point + 0.3, 0.5, duration - 0.2), "场景切换点"))

    count_uniform = max(target_frames * 4, 8)
    step = duration / count_uniform
    for index in range(count_uniform):
        point = step * (index + 0.5)
        scored.append((3, _clamp(point, 0.4, duration - 0.2), "均匀补充"))

    scored.sort(key=lambda item: (item[0], item[1]))
    result: list[dict[str, Any]] = []
    for priority, point, reason in scored:
        if any(abs(point - item["time"]) < min_gap for item in result):
            continue
        result.append({"time": round(point, 3), "priority": priority, "reason": reason})
        if len(result) >= total:
            break
    return result


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

    # ---- 主流程 -------------------------------------------------------
    async def run(
        self,
        video_path: str | Path,
        *,
        target_frames: int = 3,
        planner: Optional[str] = None,
        max_rounds: Optional[int] = None,
        subtitles: Any = None,
        scene_times: Optional[Sequence[float]] = None,
        subtitles_from: Optional[str] = None,
        on_progress: Optional[Callable[[dict[str, Any]], None]] = None,
        is_canceled: Optional[Callable[[], bool]] = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        target = max(1, min(10, int(target_frames or 3)))
        rounds_limit = max(1, min(12, int(max_rounds or 6)))
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
        stats: dict[str, Any] = {
            "candidates": 0,
            "vlm_calls": 0,
            "blank_skipped": 0,
            "duplicates_skipped": 0,
            "elapsed_ms": 0,
            "planner": mode,
            "max_rounds": rounds_limit,
            "subtitles": 0,
        }
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

        # 1) 字幕：调用方给 -> 同目录站点字幕 -> ASR 分段
        report(stage="subtitles", percent=5, status="running")
        segments: list[dict[str, Any]] = []
        if subtitles is not None:
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

        # 3) 预筛一批候选（8~16 张）
        report(stage="candidates", percent=18, status="running")
        pool = plan_candidates(
            duration, segments, scenes, target_frames=target, limit=40
        )
        if mode == "agent":
            planned = await self._vlm_plan(duration, target, segments, scenes, stats)
            if planned:
                pool = self._merge_pools(planned, pool)
        initial_count = min(
            MAX_INITIAL, max(MIN_INITIAL, target * 4), max(4, len(pool))
        )
        initial_times = self._take_times(pool, used_times, initial_count)
        batch = await self._extract_and_screen(
            media, path, out_dir, initial_times, stats, deferred
        )
        used_times.extend(item["time"] for item in batch)
        check_cancel()

        # 4) 首轮评分
        report(stage="scoring", percent=25, status="running")
        scored = await self._score_batch(batch, segments, duration, stats, deferred)
        history.extend(scored)
        results = self._accepted(scored, target)
        rounds_used = 1

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
                times = self._take_times(pool, used_times, need)
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
                scored = await self._score_batch(
                    batch, segments, duration, stats, deferred
                )
                history.extend(scored)
                results = self._accepted(history, target)
                rounds_used += 1

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

        stats["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        stats["vlm_calls"] = stats["vlm_calls"]
        if len(results) < target:
            stats["shortfall"] = target - len(results)
        report(stage="done", percent=100, status="done")
        payload = {
            "ok": True,
            "video": str(path),
            "duration": duration,
            "target": target,
            "rounds_used": rounds_used,
            "frames": results,
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
                result = await asyncio.to_thread(
                    adapter.get_module().infer_segments, str(wav_path or path), "auto"
                )
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

    # ---- VLM 调用 -----------------------------------------------------
    async def _vlm_plan(
        self,
        duration: float,
        target: int,
        segments: Sequence[dict[str, Any]],
        scenes: Sequence[float],
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """用 vlm_frame_plan.md 让模型规划一批候选时间点（纯文本）。"""
        try:
            vision = self._vision()
            prompt = load_prompt(self.cfg, "vlm_frame_plan.md")
            user = json.dumps(
                {
                    "duration_sec": duration,
                    "target_frames": target,
                    "subtitles": _trim_subtitles(segments, limit=24, text_limit=60),
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
        user = json.dumps(
            {
                "time": item["time"],
                "duration_sec": duration,
                "subtitle_nearby": _subtitle_nearby(segments, item["time"]),
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
    ) -> tuple[list[dict[str, Any]], int, int]:
        """多轮代理：模型输出 extract/final，程序执行后把画面描述回灌。"""
        vision = self._vision()
        prompt = load_prompt(self.cfg, "vlm_frame_agent.md")
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
            report(
                stage="agent",
                percent=min(90, 30 + rounds_used * 10),
                status="running",
                round=rounds_used + 1,
            )
            user = json.dumps(
                {
                    "duration_sec": duration,
                    "subtitles": _trim_subtitles(segments, limit=30, text_limit=60),
                    "target_frames": target,
                    "max_rounds": max_rounds,
                    "round": rounds_used + 1,
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
                        # final 里点名但还没抽过的位置，补抽一次再评分。
                        if not 0 <= point <= duration:
                            continue
                        batch = await self._extract_and_screen(
                            media, path, out_dir, [point], stats, deferred
                        )
                        used_times.extend(item["time"] for item in batch)
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
                    results = self._accepted(chosen + results, target)
                break

            # 继续抽：extract
            times = _valid_extract_times(
                data.get("times"),
                duration=duration,
                used=used_times,
                limit=4,
            )
            if not times:
                deferred.append("VLM agent 的 extract 时间点非法，改用确定性补抽")
                times = self._take_times(
                    plan_candidates(duration, segments, scenes, target_frames=target),
                    used_times,
                    min(4, max(2, target)),
                )
            if not times:
                break
            batch = await self._extract_and_screen(
                media, path, out_dir, times, stats, deferred
            )
            used_times.extend(item["time"] for item in batch)
            scored = await self._score_batch(
                batch, segments, duration, stats, deferred
            )
            history.extend(scored)
            results = self._accepted(history, target)
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

    def _accepted(self, history: Sequence[dict[str, Any]], target: int) -> list[dict[str, Any]]:
        """按「usable 且 score>=3」筛，按分数降序取前 N（同一幻灯片只留一张）。"""
        threshold = int(self.cfg.vision_phash_distance or DEFAULT_PHASH_DISTANCE)
        valid = [
            item
            for item in history
            if item.get("usable")
            and int(item.get("score") or 0) >= 3
            and not item.get("prefiltered")
        ]
        valid.sort(key=lambda item: (-int(item.get("score") or 0), item["time"]))
        chosen: list[dict[str, Any]] = []
        for item in valid:
            if any(_same_frame(item, other, threshold) for other in chosen):
                continue
            chosen.append(item)
            if len(chosen) >= target:
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
    }


def _public_frame(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": round(float(item["time"]), 3),
        "path": item["path"],
        "score": int(item.get("score") or 0),
        "kind": item.get("kind") or "other",
        "ocr_text": item.get("ocr_text") or "",
        "summary": item.get("summary") or "",
        "why": item.get("why") or "",
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
