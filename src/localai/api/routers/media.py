"""媒体路由：ffmpeg 探测 / 转 16k wav / 抽帧 / 抽音轨 / 场景检测。

契约（全部 multipart，文件字段名统一叫 ``file``）：

- ``POST /media/probe``  -> ``{ok, duration, format, audio, video, ...}``
- ``POST /media/to_wav`` -> audio/wav 二进制（16k 单声道 PCM16）
- ``POST /media/frames`` -> ``{ok, count, dir, frames[], preview_b64}``
- ``POST /media/audio``  -> audio/wav 或 audio/mpeg 二进制
- ``POST /media/scenes`` -> ``{ok, count, scenes[{time,score}], json_path}``

临时文件处理完立即删除；抽帧结果留在 ``<OUTPUT>/frames/<时间戳>/``；
场景检测结果（可选）落在 ``<OUTPUT>/scenes/<时间戳>.json``。
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ..deps import get_cfg, get_registry
from ..uploads import save_media_upload

logger = logging.getLogger("localai.api.media")

router = APIRouter(tags=["media"])


@router.post("/media/probe", summary="读取媒体信息")
async def media_probe_endpoint(
    request: Request,
    file: UploadFile = File(..., description="要探测的音频或视频文件"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    tmp_path = await save_media_upload(cfg, file, "probe")
    try:
        info = await media.call("probe", str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info(
        "媒体探测：%s 时长=%ss 视频=%s 音频=%s",
        Path(file.filename or "").name,
        info.get("duration"),
        bool(info.get("video")),
        bool(info.get("audio")),
    )
    return {"ok": True, **info}


@router.post("/media/to_wav", summary="任意音视频 -> 16k 单声道 WAV")
async def media_to_wav_endpoint(
    request: Request,
    file: UploadFile = File(..., description="任意音频或视频文件"),
) -> Response:
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    tmp_path = await save_media_upload(cfg, file, "to_wav")
    converted: Optional[dict[str, Any]] = None
    try:
        converted = await media.call("to_wav16k", str(tmp_path))
        wav_bytes = Path(converted["path"]).read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
        if converted is not None:
            Path(converted["path"]).unlink(missing_ok=True)

    stem = Path(file.filename or "audio").stem or "audio"
    download_name = f"{stem}_16k.wav"
    headers = {
        "X-Duration": str(converted["duration"]),
        "X-Sample-Rate": "16000",
        "X-Channels": "1",
        # 中文文件名用 RFC 5987 编码，同时给一个纯 ASCII 兜底名
        "Content-Disposition": (
            'attachment; filename="audio_16k.wav"; '
            f"filename*=UTF-8''{quote(download_name)}"
        ),
    }
    logger.info(
        "媒体转 wav：%s -> %s 字节，时长=%ss",
        Path(file.filename or "").name,
        len(wav_bytes),
        converted["duration"],
    )
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@router.post("/media/frames", summary="按数量/时间点/fps 抽帧")
async def media_frames_endpoint(
    request: Request,
    file: UploadFile = File(..., description="要抽帧的视频文件"),
    count: Optional[int] = Form(None, description="均匀抽 N 张（与 fps/times 互斥）"),
    fps: Optional[float] = Form(None, description="每秒抽 N 张（与 count/times 互斥）"),
    times: Optional[str] = Form(None, description="按时间点抽，逗号分隔的秒数，如 1,2.5,4"),
    width: Optional[int] = Form(None, description="输出宽度（高度按比例，自动取偶数）"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    given = [value is not None for value in (count, fps, times)]
    if sum(given) > 1:
        raise HTTPException(status_code=400, detail="count / fps / times 只能给一个")
    if width is not None and width <= 0:
        raise HTTPException(status_code=400, detail="width 必须是正整数")

    points: Optional[list[float]] = None
    if times is not None:
        try:
            points = [float(part) for part in times.split(",") if part.strip()]
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"times 格式不对，应为逗号分隔的秒数，如 1,2.5,4：{exc}",
            ) from exc
        if not points:
            raise HTTPException(status_code=400, detail="times 不能为空")

    tmp_path = await save_media_upload(cfg, file, "frames")
    # 每次请求一个独立目录，文件名里带时间点，调用方拿路径就能直接用
    out_dir = cfg.frames_dir / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    started = time.perf_counter()
    try:
        frames = await media.call(
            "extract_frames",
            str(tmp_path),
            str(out_dir),
            count=count,
            times=points,
            fps=fps,
            width=width,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    preview_b64: Optional[str] = None
    preview_mime: Optional[str] = None
    if frames:
        first_path = Path(frames[0]["path"])
        preview_b64 = base64.b64encode(first_path.read_bytes()).decode("ascii")
        preview_mime = "image/png" if first_path.suffix.lower() == ".png" else "image/jpeg"

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "抽帧完成：%s -> %d 张，目录=%s，耗时=%dms",
        Path(file.filename or "").name,
        len(frames),
        out_dir,
        elapsed_ms,
    )
    return {
        "ok": True,
        "count": len(frames),
        "dir": str(out_dir),
        "elapsed_ms": elapsed_ms,
        "frames": frames,
        "preview_b64": preview_b64,
        "preview_mime": preview_mime,
    }


@router.post("/media/audio", summary="只抽音轨，返回 wav/mp3")
async def media_audio_endpoint(
    request: Request,
    file: UploadFile = File(..., description="要抽音轨的音频或视频文件"),
    format: str = Form("wav", description="wav（保留原采样率）或 mp3"),
) -> Response:
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    audio_format = (format or "wav").strip().lower()
    if audio_format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format 只支持 wav 或 mp3")

    tmp_path = await save_media_upload(cfg, file, "audio")
    result: Optional[dict[str, Any]] = None
    out_dir = cfg.media_tmp_dir / f"extract_{uuid.uuid4().hex}"
    try:
        result = await media.call("extract_audio", str(tmp_path), str(out_dir), fmt=audio_format)
        audio_bytes = Path(result["path"]).read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
        if result is not None:
            Path(result["path"]).unlink(missing_ok=True)
        try:
            out_dir.rmdir()
        except OSError:
            pass

    media_type = "audio/wav" if audio_format == "wav" else "audio/mpeg"
    stem = Path(file.filename or "audio").stem or "audio"
    download_name = f"{stem}.{audio_format}"
    headers = {
        "X-Duration": str(result["duration"]),
        "X-Sample-Rate": str(result.get("sample_rate")),
        "X-Channels": str(result.get("channels")),
        "Content-Disposition": (
            f'attachment; filename="audio.{audio_format}"; '
            f"filename*=UTF-8''{quote(download_name)}"
        ),
    }
    logger.info(
        "抽音轨：%s -> %s（%s 字节，%sHz/%s 声道）",
        Path(file.filename or "").name,
        audio_format,
        len(audio_bytes),
        result.get("sample_rate"),
        result.get("channels"),
    )
    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@router.post("/media/scenes", summary="场景/镜头边界检测（ffmpeg scene 分数）")
async def media_scenes_endpoint(
    request: Request,
    file: UploadFile = File(..., description="要检测的视频文件"),
    threshold: float = Form(0.3, description="场景分数阈值 0~1，越大越严格"),
    max_scenes: int = Form(200, description="最多返回多少个边界"),
    min_gap: float = Form(1.0, description="最小间隔秒数（去抖）"),
) -> dict[str, Any]:
    """返回 ``{ok, count, scenes:[{time,score}], duration, json_path}``。

    结果同时落盘到 ``<OUTPUT>/scenes/<时间戳>.json``，方便二次处理。
    """
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    started = time.perf_counter()
    tmp_path = await save_media_upload(cfg, file, "scenes")
    try:
        result = await media.call(
            "detect_scenes",
            str(tmp_path),
            threshold=threshold,
            max_scenes=max_scenes,
            min_gap=min_gap,
            save_dir=str(cfg.scenes_dir),
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "场景检测：%s -> %d 个边界（阈值 %.2f，最小间隔 %.2fs，耗时 %dms）",
        Path(file.filename or "").name,
        result.get("count", 0),
        threshold,
        min_gap,
        elapsed_ms,
    )
    return {
        "ok": True,
        "count": result.get("count", 0),
        "scenes": result.get("scenes") or [],
        "duration": result.get("duration"),
        "threshold": result.get("threshold"),
        "max_scenes": result.get("max_scenes"),
        "min_gap": result.get("min_gap"),
        "json_path": result.get("json_path"),
        "elapsed_ms": elapsed_ms,
    }
