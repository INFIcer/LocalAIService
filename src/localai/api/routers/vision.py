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

import asyncio
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from ...core.frames_agent import FramesAgent, load_prompt, parse_json_loose
from ...core.vram import get_vram
from ..deps import get_cfg, get_registry
from ..schemas import VisionFramesRequest

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
            max_rounds=payload.max_rounds,
            subtitles=payload.subtitles,
            scene_times=payload.scene_times,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        "vram_estimate_mb": estimates.get("vision", adapter.estimate_mb()),
        "idle_unload_seconds": cfg.vision_idle_unload,
        "phash_distance": cfg.vision_phash_distance,
        "prompts": {
            "dir": str(prompts_dir),
            "score": (prompts_dir / "vlm_frame_score.md").is_file(),
            "agent": (prompts_dir / "vlm_frame_agent.md").is_file(),
            "plan": (prompts_dir / "vlm_frame_plan.md").is_file(),
        },
        "endpoints": list(adapter.ENDPOINTS),
        "limits": {"target_frames": [1, 10], "max_rounds": [1, 12]},
    }
