"""视觉 / 关键帧挑选路由。

接口
----
- ``POST /vision/describe``：单张图片理解（multipart），默认用单帧评分提示词
  （``docs/prompts/vlm_frame_score.md``），返回模型原始文本（通常是 JSON）。
- ``POST /vision/scan``：多图批量；multipart 上传或 JSON 路径列表都行，
  ``task=score|caption|ocr`` 决定提示词。
- ``POST /vision/frames``：按字幕 + 场景 + VLM 循环抽帧，直到凑够 N 张。
- ``GET  /vision/status``：模型、进程、显存估算、默认 planner。
- ``POST /jobs/vision``：同样的挑帧逻辑走任务队列（kind=vision）。

提示词全部从 ``<LOCALAI_HOME>/docs/prompts/`` 读取，改提示词不用改代码。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.datastructures import UploadFile as StarletteUploadFile

from ...core.frames_agent import (
    EFFORT_CHOICES,
    EFFORT_DEFAULT,
    FramesAgent,
    effort_budget,
    load_prompt,
    parse_json_loose,
    summary_confidence,
)
from ...core.vram import get_vram
from ..deps import get_cfg, get_registry
from ..schemas import VisionFramesRequest, VisionVideoRequest

logger = logging.getLogger("localai.api.vision")

router = APIRouter(tags=["vision"])

#: 常见图片文件头；只接受真正的图片，避免把任意文件喂给 VLM。
_IMAGE_MAGIC = (
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)

_CAPTION_PROMPT = (
    "你是视频画面描述器。用一句话（≤40 字）说明画面内容，并给出 kind "
    "（slide/chart/code/table/demo/face/scene/map/blank/transition/other）。"
    '只输出 JSON：{"summary":"...","kind":"...","ocr_text":"..."}'
)
_OCR_PROMPT = "提取这张图片里的所有文字，尽量保留原样；没有文字就输出空字符串。只输出文字本身，不要解释。"


def _sniff_image(data: bytes) -> Optional[str]:
    for magic, suffix in _IMAGE_MAGIC:
        if data.startswith(magic):
            return suffix
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


async def _save_image_upload(cfg: Any, upload: UploadFile, tag: str) -> Path:
    """把上传图片落盘到 media_tmp（请求结束会删），并校验确实是图片。"""
    from ..uploads import save_media_upload  # noqa: PLC0415

    path = await save_media_upload(cfg, upload, tag)
    try:
        with path.open("rb") as handle:
            head = handle.read(32)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"读取上传图片失败：{exc}") from exc
    suffix = _sniff_image(head)
    if suffix is None:
        path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="只接受图片文件（JPEG / PNG / WebP / GIF / BMP）",
        )
    return path


def _default_score_user(time_sec: Optional[float] = None) -> str:
    """单图评分的用户消息（和 vlm_frame_score.md 的输入格式一致）。"""
    import json

    return json.dumps(
        {
            "time": time_sec,
            "duration_sec": None,
            "subtitle_nearby": "",
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 单图 / 多图
# ---------------------------------------------------------------------------
@router.post("/vision/describe", summary="单图理解（默认单帧评分提示词）")
async def vision_describe_endpoint(
    request: Request,
    file: UploadFile = File(..., description="图片文件（JPEG/PNG/WebP/GIF/BMP）"),
    prompt: Optional[str] = Form(None, description="可选：覆盖用户消息；系统提示词仍用评分提示词"),
    time: Optional[float] = Form(None, description="可选：该帧的时间点（秒），只用于提示词"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    vision = get_registry(request).get("vision")
    path = await _save_image_upload(cfg, file, "vision_describe")
    try:
        system_prompt = load_prompt(cfg, "vlm_frame_score.md")
        user = (prompt or "").strip() or _default_score_user(time)
        outcome = await vision.describe_image(
            str(path), user, system_prompt=system_prompt, max_tokens=400
        )
    finally:
        path.unlink(missing_ok=True)
    text = outcome.get("text") or ""
    logger.info(
        "视觉单图：%s -> %d 字，耗时=%sms",
        Path(file.filename or "").name,
        len(text),
        outcome.get("elapsed_ms"),
    )
    return {
        "ok": True,
        "text": text,
        "parsed": parse_json_loose(text),
        "elapsed_ms": outcome.get("elapsed_ms"),
        "model": outcome.get("model"),
        "prompt_file": "docs/prompts/vlm_frame_score.md",
    }


@router.post("/vision/scan", summary="多图批量理解（score / caption / ocr）")
async def vision_scan_endpoint(request: Request) -> dict[str, Any]:
    """multipart 上传（字段名随意，可多张）或 JSON ``{"images":[...]}``。

    上传的图片在请求结束即删除；JSON 路径列表只读调用方本机文件。
    """
    cfg = get_cfg(request)
    vision = get_registry(request).get("vision")
    content_type = (request.headers.get("content-type") or "").lower()
    task = "score"
    images: list[Path] = []
    temp_paths: list[Path] = []

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        task = str(form.get("task") or "score").strip().lower()
        uploads = [
            value
            for _key, value in form.multi_items()
            if isinstance(value, StarletteUploadFile)
        ]
        if not uploads:
            raise HTTPException(status_code=400, detail="没有收到图片（multipart 字段 file）")
        for upload in uploads:
            path = await _save_image_upload(cfg, upload, "vision_scan")
            images.append(path)
            temp_paths.append(path)
    else:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail="请求体必须是 multipart 图片或 JSON（images / paths / image）",
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON 请求体必须是对象")
        task = str(body.get("task") or "score").strip().lower()
        raw = body.get("images") or body.get("paths") or body.get("image") or []
        if isinstance(raw, str):
            raw = [raw]
        for item in raw:
            path = Path(str(item))
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"图片不存在：{path}")
            images.append(path)

    if task not in ("score", "caption", "ocr"):
        raise HTTPException(status_code=400, detail="task 只支持 score / caption / ocr")
    if not images:
        raise HTTPException(status_code=400, detail="images 不能为空")
    if len(images) > 16:
        raise HTTPException(status_code=400, detail="一次最多 16 张")

    if task == "score":
        system_prompt = load_prompt(cfg, "vlm_frame_score.md")
        user = _default_score_user(None)
    elif task == "caption":
        system_prompt = _CAPTION_PROMPT
        user = "描述这张图片。"
    else:
        system_prompt = _OCR_PROMPT
        user = "提取文字。"

    try:
        outcome = await vision.describe_many(
            [str(path) for path in images],
            user,
            system_prompt=system_prompt,
            max_tokens=500,
        )
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
    texts = outcome.get("texts") or []
    results = [
        {
            "index": index,
            "path": str(images[index]) if index < len(images) else None,
            "text": text,
            "parsed": parse_json_loose(text) if task == "score" else None,
        }
        for index, text in enumerate(texts)
    ]
    return {
        "ok": True,
        "task": task,
        "count": len(results),
        "results": results,
        "elapsed_ms": outcome.get("elapsed_ms"),
        "model": outcome.get("model"),
    }


# ---------------------------------------------------------------------------
# 关键帧挑选
# ---------------------------------------------------------------------------
async def _resolve_video_path(payload: VisionFramesRequest, request: Request) -> Path:
    """``video_path`` 或 ``job_id`` 二选一，解析成真实文件路径。"""
    raw = (payload.video_path or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"视频不存在：{path}")
        return path
    job_id = (payload.job_id or "").strip()
    if not job_id:
        raise HTTPException(
            status_code=400, detail="必须给 video_path（服务端本地路径）或 job_id"
        )
    store = getattr(request.app.state, "jobs_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="任务库未就绪，无法按 job_id 取视频")
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{job_id}")
    result = job.get("result") or {}
    raw_path = result.get("path") or result.get("video") or result.get("output_path")
    if not raw_path or not Path(str(raw_path)).is_file():
        raise HTTPException(
            status_code=400,
            detail=f"任务 {job_id} 的结果里没有可用的视频文件",
        )
    return Path(str(raw_path))


@router.post("/vision/frames", summary="按字幕+时长循环抽帧，凑够 N 张有效关键帧")
async def vision_frames_endpoint(
    payload: VisionFramesRequest, request: Request
) -> dict[str, Any]:
    cfg = get_cfg(request)
    registry = get_registry(request)
    path = await _resolve_video_path(payload, request)
    agent = FramesAgent(cfg, registry)
    try:
        return await agent.run(
            path,
            target_frames=payload.target_frames,
            planner=payload.planner,
            effort=payload.effort,
            max_rounds=payload.max_rounds,
            subtitles=payload.subtitles,
            scene_times=payload.scene_times,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# 按需取帧图片 / 段级看视频（只返回文字）
# ---------------------------------------------------------------------------

#: 全片一次最多看多少帧（接口硬上限，防止有人误传 100 帧）。
VIDEO_MAX_FRAMES = 64
#: 单段一次最多看多少帧（接口硬上限）。
VIDEO_MAX_SEGMENT_FRAMES = 48
#: 画质优先的单次帧数：小模型在 16 帧以上会明显忽略后半段，实测 16 帧最稳。
VIDEO_QUALITY_FRAMES = 16
#: /vision/video 抽帧宽度：为多帧同请求留出视觉 token 余量。
VIDEO_FRAME_WIDTH = 640
#: 多帧同请求时每张缩略图最长边（448 能看清大标题，又不爆 4096 上下文）。
VIDEO_IMAGE_MAX_SIDE = 448


def _frame_dirs(cfg: Any) -> list[Path]:
    """``/vision/frame`` 允许读取的根目录：服务输出目录 + 下载目录。"""
    dirs = []
    for base in (Path(cfg.output_dir), Path(cfg.downloads_dir)):
        try:
            dirs.append(base.resolve())
        except OSError:
            continue
    return dirs


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_frame_path(cfg: Any, raw: str) -> Path:
    """校验帧路径必须在 ``outputs/`` 或 ``data/downloads/`` 下，防目录穿越。

    解析 ``..`` 和符号链接之后再比对根目录；``../../Windows/win.ini``、
    ``C:\\Windows\\...`` 这类绝对路径外泄都会在这里被 400 掉。
    """
    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="必须给 path（帧的绝对路径）")
    try:
        resolved = Path(text).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"路径不合法：{exc}") from exc
    dirs = _frame_dirs(cfg)
    if not any(_is_within(resolved, base) for base in dirs):
        raise HTTPException(
            status_code=400,
            detail=(
                "只允许读取服务输出目录（outputs/）或下载目录（data/downloads/）"
                f"下的帧：{resolved}"
            ),
        )
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"帧文件不存在：{resolved}")
    if resolved.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(
            status_code=400, detail=f"只支持 jpg/png 帧，收到 {resolved.suffix or '(无扩展名)'}"
        )
    return resolved


@router.get("/vision/frame", summary="按需取一张帧图片（仅限 outputs / downloads）")
async def vision_frame_endpoint(request: Request, path: str) -> Response:
    """返回帧的二进制（image/jpeg）；响应里不再塞 base64。

    路径必须落在 ``LOCALAI_HOME/outputs/`` 或 ``LOCALAI_HOME/data/downloads/``
    之下；其它位置（含 ``../../``、绝对路径外泄）一律 400。
    """
    cfg = get_cfg(request)
    resolved = _safe_frame_path(cfg, path)
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"读取帧失败：{exc}") from exc
    media_type = "image/png" if resolved.suffix.lower() == ".png" else "image/jpeg"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Cache-Control": "no-store", "X-Frame-Time": ""},
    )


def _video_window(
    payload: VisionVideoRequest, duration: float
) -> tuple[float, float, bool]:
    """归一 ``start/end``：都没给=全片；给一个就补另一个；非法窗口 400。"""
    if payload.start is None and payload.end is None:
        return 0.0, round(float(duration), 3), False
    start = 0.0 if payload.start is None else float(payload.start)
    end = round(float(duration), 3) if payload.end is None else float(payload.end)
    if start < 0:
        raise HTTPException(status_code=400, detail=f"start 不能是负数：{start}")
    if end <= start:
        raise HTTPException(status_code=400, detail=f"end 必须大于 start：{start}~{end}")
    if end > float(duration) + 0.05:
        raise HTTPException(
            status_code=400,
            detail=f"end={end:.3f}s 超出视频时长 {float(duration):.3f}s",
        )
    return round(start, 3), round(min(end, float(duration)), 3), True


def _video_prompt(
    payload: VisionVideoRequest,
    times: Sequence[float],
    start: float,
    end: float,
    fps: float,
) -> str:
    return json.dumps(
        {
            "window": {"start": start, "end": end},
            "fps": round(fps, 3),
            "frame_times": [round(float(item), 3) for item in times],
            "question": (payload.question or "").strip() or None,
        },
        ensure_ascii=False,
    )


def _image_max_side(count: int) -> int:
    """多帧同请求时的缩略图尺寸：统一 448，画质与上下文取平衡。"""
    return VIDEO_IMAGE_MAX_SIDE


@router.post("/vision/video", summary="段级细看：多帧一次性看视频，只返回文字")
async def vision_video_endpoint(
    payload: VisionVideoRequest, request: Request
) -> dict[str, Any]:
    """``start/end`` 都不给看全片；给了就用 ``clip`` 只切这一段。

    帧数上限：全片 64、单段 48；超了就自动降 fps 并在响应 ``note`` 里说明。
    **响应只有文字**：``summary`` + ``key_moments``，不返回图片 / base64。
    """
    started = time.perf_counter()
    cfg = get_cfg(request)
    registry = get_registry(request)
    media = registry.get("media")
    vision = registry.get("vision")
    path = Path(payload.video_path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"视频不存在：{path}")

    info = await media.call("probe", str(path))
    duration = info.get("duration")
    if not duration or float(duration) <= 0:
        raise HTTPException(status_code=400, detail=f"读不出视频时长：{path}")
    duration = round(float(duration), 3)
    if not info.get("video"):
        raise HTTPException(status_code=400, detail=f"这个文件没有视频流：{path}")

    start, end, is_segment = _video_window(payload, duration)
    span = round(end - start, 3)
    if span <= 0:
        raise HTTPException(status_code=400, detail="时间窗口为空")
    hard_cap = VIDEO_MAX_SEGMENT_FRAMES if is_segment else VIDEO_MAX_FRAMES
    # 小模型实测超过 16 帧就会明显忽略后半段：画质优先，把有效帧数再压到 16。
    cap = min(hard_cap, VIDEO_QUALITY_FRAMES)
    requested_fps = max(0.5, min(3.0, float(payload.fps or 1.0)))
    effective_fps = min(requested_fps, cap / span)
    note = ""
    if effective_fps < requested_fps - 0.01:
        note = (
            f"帧数上限 {cap}（接口硬上限 {hard_cap}，为保证小模型看清画面取 {VIDEO_QUALITY_FRAMES}），"
            f"fps 已从 {requested_fps:g} 降到 {effective_fps:.3f}（窗口 {span:.1f}s）"
        )

    work_dir = cfg.frames_dir / f"video_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_path: Optional[Path] = None
    try:
        if is_segment:
            # 按需求「只切该段」：先 clip 成独立 mp4，再从 clip 里抽帧。
            clip_dir = Path(cfg.output_dir) / "clips" / uuid.uuid4().hex[:8]
            clip_dir.mkdir(parents=True, exist_ok=True)
            clip_result = await media.call(
                "clip", str(path), start, end, out=str(clip_dir / "segment.mp4")
            )
            clip_path = Path(clip_result["path"])
            source = clip_path
            source_duration = float(clip_result.get("duration") or span)
        else:
            source = path
            source_duration = duration

        if effective_fps >= 0.5:
            frames = await media.call(
                "extract_frames_window",
                str(source),
                0.0 if is_segment else start,
                source_duration if is_segment else end,
                fps=effective_fps,
                width=VIDEO_FRAME_WIDTH,
                out_dir=str(work_dir),
            )
            if is_segment:
                for frame in frames:
                    frame["time_sec"] = round(start + float(frame["time_sec"]), 3)
        else:
            # 窗口太长（例如 431 秒全片看 64 帧），fps 会低于 0.5：
            # 用 count 模式做等效均匀抽帧，再把时间点换算到绝对时间轴。
            frames = await media.call(
                "extract_frames",
                str(source),
                str(work_dir),
                count=cap,
                width=VIDEO_FRAME_WIDTH,
            )
            if is_segment:
                for frame in frames:
                    frame["time_sec"] = round(start + float(frame["time_sec"]), 3)

        # 硬上限：无论 ffmpeg 实际吐了多少张，交给 VLM 的绝不超过 cap。
        frames = frames[:cap]
        if not frames:
            raise HTTPException(status_code=400, detail="没有抽到任何帧，无法看视频")
        times = [round(float(item.get("time_sec") or 0.0), 3) for item in frames]
        system_prompt = load_prompt(cfg, "vlm_video_watch.md")
        user = _video_prompt(payload, times, start, end, effective_fps)
        if not hasattr(vision, "describe_frames"):
            raise HTTPException(
                status_code=503, detail="当前视觉适配器不支持多帧一次性理解"
            )
        labels = [
            f"第 {index + 1} 帧 time={times[index]}s" for index in range(len(frames))
        ]
        outcome = await vision.describe_frames(
            [str(item["path"]) for item in frames],
            user,
            system_prompt=system_prompt,
            max_tokens=600,
            max_side=_image_max_side(len(frames)),
            labels=labels,
        )
    finally:
        if clip_path is not None:
            try:
                clip_path.unlink(missing_ok=True)
                clip_path.parent.rmdir()
            except OSError:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)

    text = str(outcome.get("text") or "")
    data = parse_json_loose(text)
    if isinstance(data, dict):
        summary = str(data.get("summary") or "").strip()
        raw_moments = data.get("key_moments") or []
    else:
        summary = text.strip()
        raw_moments = []
    if not summary:
        summary = text.strip()
    key_moments: list[dict[str, Any]] = []
    for item in raw_moments if isinstance(raw_moments, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            moment = float(item.get("time"))
        except (TypeError, ValueError):
            continue
        if moment < start - 0.5 or moment > end + 0.5:
            continue
        what = str(item.get("what") or "").strip()
        if not what:
            continue
        key_moments.append({"time": round(moment, 3), "what": what[:120]})
    key_moments.sort(key=lambda row: row["time"])

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "段级看视频：%s 窗口=%.1f~%.1fs fps=%.3f 帧=%d 耗时=%dms",
        path.name,
        start,
        end,
        effective_fps,
        len(frames),
        elapsed_ms,
    )
    return {
        "ok": True,
        "duration": duration,
        "window": {"start": start, "end": end, "seconds": span},
        "fps": round(effective_fps, 3),
        "fps_requested": requested_fps,
        "frames_used": len(frames),
        "summary": summary,
        "key_moments": key_moments,
        "confidence": summary_confidence(summary),
        "elapsed_ms": elapsed_ms,
        "note": note,
        "model": outcome.get("model"),
    }


@router.get("/vision/status", summary="视觉模型状态（不加载模型）")
async def vision_status_endpoint(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    adapter = get_registry(request).get("vision")
    status = adapter.status()
    try:
        estimates = get_vram().estimates()
    except Exception:  # noqa: BLE001 - 状态页不能因此挂掉
        estimates = {}
    prompts_dir = Path(cfg.home) / "docs" / "prompts"
    return {
        "ok": True,
        **status,
        "planner_default": cfg.vision_planner,
        "planners": ["hybrid", "agent"],
        "effort_default": EFFORT_DEFAULT,
        "effort_choices": list(EFFORT_CHOICES),
        "effort_budgets": {
            "low": effort_budget(3, "low"),
            "medium": effort_budget(3, "medium"),
            "high": effort_budget(3, "high"),
        },
        "effort_budget_note": (
            "候选数 = target × factor（至少 6）；max_rounds/vlm_limit/clip_limit "
            "为该 effort 的硬上限；target 不同时候选数按比例变化"
        ),
        "vram_estimate_mb": estimates.get("vision", adapter.estimate_mb()),
        "idle_unload_seconds": cfg.vision_idle_unload,
        "phash_distance": cfg.vision_phash_distance,
        "prompts": {
            "dir": str(prompts_dir),
            "score": (prompts_dir / "vlm_frame_score.md").is_file(),
            "agent": (prompts_dir / "vlm_frame_agent.md").is_file(),
            "plan": (prompts_dir / "vlm_frame_plan.md").is_file(),
            "video": (prompts_dir / "vlm_video_watch.md").is_file(),
        },
        "endpoints": list(adapter.ENDPOINTS),
        "limits": {
            "target_frames": [1, 10],
            "max_rounds": [1, 12],
            "video_frames_full": VIDEO_MAX_FRAMES,
            "video_frames_segment": VIDEO_MAX_SEGMENT_FRAMES,
            "video_quality_frames": VIDEO_QUALITY_FRAMES,
        },
    }
