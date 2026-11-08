"""媒体路由：ffmpeg 探测 / 转 16k wav / 抽帧 / 抽音轨 / 场景检测 / 字幕。

契约：

- ``POST /media/probe``  -> ``{ok, duration, format, audio, video, ...}``（multipart）
- ``POST /media/to_wav`` -> audio/wav 二进制（16k 单声道 PCM16，multipart）
- ``POST /media/frames`` -> ``{ok, count, dir, frames[], preview_b64}``（multipart）
- ``POST /media/audio``  -> audio/wav 或 audio/mpeg 二进制（multipart）
- ``POST /media/scenes`` -> ``{ok, count, scenes[{time,score}], mode, windows, warnings}``
- ``POST /media/subtitles`` -> 内嵌 / 外挂软字幕列表（JSON 路径或 multipart 上传）
- ``POST /media/subtitles/extract`` -> 抽成一个字幕源（segments 或 srt）

临时文件处理完立即删除；抽帧结果留在 ``<OUTPUT>/frames/<时间戳>/``；
场景检测结果（可选）落在 ``<OUTPUT>/scenes/<时间戳>.json``；
抽取的 SRT 落在 ``<OUTPUT>/subtitles/<时间戳>.srt``。
"""

from __future__ import annotations

import asyncio
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

from ...jobs import prepare_payload
from ..deps import get_cfg, get_registry, get_scheduler
from ..job_helpers import client_ip, submit_and_wait
from ..schemas import MediaAnalyzeRequest, MediaDownloadRequest
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
async def media_scenes_endpoint(request: Request) -> dict[str, Any]:
    """返回 ``{ok, count, scenes:[{time,score}], mode, windows, warnings}``。

    multipart 上传字段 ``file``，或 JSON ``{"video_path": "..."}``（服务端本地
    路径，便于直接测超大文件，不用上传几十 GB）。时长 > 15 分钟默认走采样
    模式（8 个 60 秒窗口），避免整片跑满超时；结果同时落盘到
    ``<OUTPUT>/scenes/<时间戳>.json``。
    """
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    started = time.perf_counter()

    content_type = (request.headers.get("content-type") or "").lower()
    upload = None
    raw_path = ""
    threshold = 0.3
    max_scenes = 200
    min_gap = 1.0
    mode = "auto"
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        raw_path = str(form.get("video_path") or "").strip()
        try:
            threshold = float(form.get("threshold") or 0.3)
            max_scenes = int(form.get("max_scenes") or 200)
            min_gap = float(form.get("min_gap") or 1.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"参数不是数字：{exc}") from exc
        mode = str(form.get("mode") or "auto")
    else:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail="请求体必须是 multipart（file）或 JSON（video_path）",
            ) from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON 请求体必须是对象")
        raw_path = str(body.get("video_path") or "").strip()
        try:
            threshold = float(body.get("threshold", 0.3))
            max_scenes = int(body.get("max_scenes", 200))
            min_gap = float(body.get("min_gap", 1.0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"参数不是数字：{exc}") from exc
        mode = str(body.get("mode") or "auto")

    tmp_path: Optional[Path] = None
    if upload is not None and hasattr(upload, "read"):
        tmp_path = await save_media_upload(cfg, upload, "scenes")  # type: ignore[arg-type]
        target = tmp_path
    elif raw_path:
        target = Path(raw_path)
        if not target.is_file():
            raise HTTPException(status_code=400, detail=f"视频不存在：{target}")
    else:
        raise HTTPException(status_code=400, detail="必须上传 file 或给 video_path")
    file_name = Path(getattr(upload, "filename", "") or target.name).name
    try:
        result = await media.call(
            "detect_scenes",
            str(target),
            threshold=threshold,
            max_scenes=max_scenes,
            min_gap=min_gap,
            mode=mode,
            save_dir=str(cfg.scenes_dir),
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "场景检测：%s -> %d 个边界（mode=%s，阈值 %.2f，耗时 %dms）",
        file_name,
        result.get("count", 0),
        result.get("mode"),
        threshold,
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
        "mode": result.get("mode"),
        "detector": result.get("detector"),
        "windows": result.get("windows") or [],
        "warnings": result.get("warnings") or [],
        "json_path": result.get("json_path"),
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# 字幕：列软字幕 / 抽字幕
# ---------------------------------------------------------------------------


async def _subtitle_request_body(
    request: Request, cfg: Any, tag: str
) -> tuple[Optional[str], Optional[Path]]:
    """兼容 JSON 与 multipart：返回 ``(video_path, 临时上传文件)``。

    multipart 的 ``file`` 会落盘到 media_tmp，调用方负责删除；JSON 只接受
    服务端本地路径，不在这里复制大文件。
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        raw_path = str(form.get("video_path") or "").strip()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            if not raw_path:
                raise HTTPException(
                    status_code=400, detail="multipart 请求需要 file 或 video_path"
                )
            return raw_path, None
        tmp_path = await save_media_upload(cfg, upload, tag)  # type: ignore[arg-type]
        return raw_path or None, tmp_path
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail="请求体必须是 JSON（video_path）或 multipart（file）",
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON 请求体必须是对象")
    raw_path = str(body.get("video_path") or "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="必须给 video_path 或上传 file")
    return raw_path, None


@router.post("/media/subtitles", summary="列出内嵌 / 外挂软字幕")
async def media_subtitles_endpoint(request: Request) -> dict[str, Any]:
    """探测一个视频有没有软字幕。

    - 内嵌：ffprobe ``-select_streams s``，返回 codec / language / title / default；
    - 外挂：同目录同名前缀的 ``.ass/.ssa/.srt/.vtt``（含 ``.scjp`` / ``.tcjp`` 等后缀）；
    - 两者都没有时 ``has_soft_subtitle=false``、``hint="hardcoded_or_none"``，
      调用方据此决定用 OCR 还是 ASR（Baha / ANi 硬字幕 MP4 就是这一类）。

    JSON：``{"video_path": "..."}``；multipart：字段 ``file``（或 ``video_path``）。
    """
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    raw_path, tmp_path = await _subtitle_request_body(request, cfg, "subtitles")
    target = tmp_path or Path(str(raw_path))
    if not target.is_file():
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"视频不存在：{target}")
    try:
        result = await media.call("list_subtitles", str(target))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    logger.info(
        "字幕探测：%s -> 内嵌=%d 外挂=%d hint=%s",
        target.name,
        len(result.get("embedded") or []),
        len(result.get("sidecar") or []),
        result.get("hint"),
    )
    if tmp_path is not None:
        # 上传文件已被删除，响应用原始文件名而不是临时路径。
        result["video"] = Path(str(raw_path or target.name)).name
        result["uploaded"] = True
    return result


@router.post("/media/subtitles/extract", summary="抽取一个字幕源（segments / srt）")
async def media_subtitles_extract_endpoint(request: Request) -> dict[str, Any]:
    """抽内嵌字幕流（``video_path`` + ``index``）或外挂字幕（``sidecar``）。

    请求体：``{"video_path":"...","index":2}`` 或 ``{"sidecar":"...ass"}``，
    可选 ``format="segments"|"srt"``。默认返回 segments；``format=srt`` 时
    另存到 ``<OUTPUT>/subtitles/<时间戳>.srt`` 并在响应里给 ``path``。
    """
    cfg = get_cfg(request)
    media = get_registry(request).get("media")
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON 请求体必须是对象")

    raw_video = str(body.get("video_path") or "").strip()
    raw_sidecar = str(body.get("sidecar") or "").strip()
    if bool(raw_video) == bool(raw_sidecar):
        raise HTTPException(
            status_code=400,
            detail="video_path（配 index）与 sidecar 必须二选一",
        )
    output_format = str(body.get("format") or "segments").strip().lower()
    if output_format not in ("segments", "srt"):
        raise HTTPException(status_code=400, detail="format 只支持 segments / srt")

    index: Optional[int] = None
    if raw_video:
        if body.get("index") is None:
            raise HTTPException(status_code=400, detail="抽内嵌字幕必须给 index")
        try:
            index = int(body.get("index"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"index 必须是整数：{exc}") from exc
        target = Path(raw_video)
        if not target.is_file():
            raise HTTPException(status_code=400, detail=f"视频不存在：{target}")
    else:
        target = Path(raw_sidecar)
        if not target.is_file():
            raise HTTPException(status_code=400, detail=f"字幕文件不存在：{target}")

    result = await media.call(
        "extract_subtitle_segments",
        str(target) if raw_video else None,
        index=index,
        sidecar=None if raw_video else str(target),
        fmt=output_format,
        out_dir=str(cfg.output_dir / "subtitles"),
    )
    logger.info(
        "字幕抽取：%s source=%s 段数=%d 格式=%s",
        target.name,
        result.get("source"),
        result.get("count"),
        output_format,
    )
    result["format"] = output_format
    return result


# ---------------------------------------------------------------------------
# 下载 / 链接分析
# ---------------------------------------------------------------------------


def _download_http_error(module: Any, exc: BaseException) -> HTTPException:
    """把下载引擎异常翻译成 HTTP 状态码。"""
    text = str(exc)
    if isinstance(exc, module.DownloadBlocked):
        return HTTPException(status_code=400, detail=text)
    if isinstance(exc, module.DownloadCancelled):
        return HTTPException(status_code=409, detail=text)
    if isinstance(exc, module.DownloadTimeout):
        return HTTPException(status_code=504, detail=text)
    if "上限" in text:
        return HTTPException(status_code=413, detail=text)
    return HTTPException(status_code=502, detail=text)


@router.post("/media/download", summary="提交异步下载任务（yt-dlp + 直链兜底）")
async def media_download_endpoint(
    payload: MediaDownloadRequest, request: Request
) -> dict[str, Any]:
    """提交后立刻返回 ``{ok, job_id, state}``；进度看 ``GET /jobs/{id}``。"""
    cfg = get_cfg(request)
    scheduler = get_scheduler(request)
    try:
        action, normalized = prepare_payload(
            "download", "download.fetch", payload.model_dump(), cfg
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = await scheduler.submit(
        "download",
        action,
        normalized,
        priority=payload.priority,
        client_ip=client_ip(request),
    )
    return {"ok": True, "job_id": job_id, "id": job_id, "state": "queued"}


@router.post("/media/download/sync", summary="同步下载小文件（≤ LOCALAI_DOWNLOAD_SYNC_MAX_MB）")
async def media_download_sync_endpoint(
    payload: MediaDownloadRequest, request: Request
) -> dict[str, Any]:
    """小文件直接下完再返回；超过同步上限要求改用 ``/media/download``。

    这里不经过任务队列（调用方要的是「立刻拿到路径」），但仍在线程池里跑，
    不会卡住事件循环。
    """
    cfg = get_cfg(request)
    adapter = get_registry(request).get("download")
    module = adapter.get_module()
    out_dir = cfg.downloads_dir / datetime.now().strftime("%Y-%m-%d")
    limit = int(cfg.download_sync_max_mb)
    try:
        result = await asyncio.to_thread(
            module.download_any,
            payload.url,
            str(out_dir),
            quality=payload.quality,
            subtitles=payload.subtitles,
            cookies=payload.cookies,
            timeout=cfg.download_timeout,
            max_mb=limit,
        )
    except Exception as exc:  # noqa: BLE001 - 统一翻译成 HTTP 错误
        raise _download_http_error(module, exc) from exc
    logger.info(
        "同步下载完成：%s（%.2fMB，字幕 %d）",
        result.get("path"),
        result.get("size_mb") or 0,
        len(result.get("subtitles") or []),
    )
    return {
        "ok": True,
        "path": result.get("path"),
        "title": result.get("title"),
        "duration": result.get("duration"),
        "site": result.get("site"),
        "subtitles": result.get("subtitles") or [],
        "size_mb": result.get("size_mb"),
        "quality": result.get("quality"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "sync_limit_mb": limit,
    }


@router.get("/media/download/status", summary="下载引擎状态与限额")
async def media_download_status_endpoint(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    adapter = get_registry(request).get("download")
    status = adapter.status()
    counts = {"queued": 0, "running": 0}
    scheduler = getattr(request.app.state, "scheduler", None)
    store = getattr(request.app.state, "jobs_store", None)
    if store is not None:
        for state in ("queued", "running"):
            _jobs, total = await store.list_jobs(state=state, kind="download", limit=1)
            counts[state] = int(total)
        _jobs, total = await store.list_jobs(state="running", kind="analyze", limit=1)
        counts["analyze_running"] = int(total)
    return {
        "ok": True,
        "yt_dlp": {
            "installed": bool(status.get("ready")),
            "version": status.get("version"),
            "path": status.get("path"),
            "python": status.get("python"),
        },
        "aria2c": status.get("aria2c"),
        "download_dir": status.get("download_dir") or str(cfg.downloads_dir),
        "today_dir": status.get("today_dir"),
        "jobs": counts,
        "limits": {
            "max_mb": cfg.download_max_mb,
            "sync_max_mb": cfg.download_sync_max_mb,
            "timeout_seconds": cfg.download_timeout,
            "qualities": status.get("limits", {}).get("qualities")
            or ["720p", "best", "audio"],
        },
        "ffmpeg": status.get("ffmpeg"),
        "cookies": status.get("cookies"),
        "ssrf": status.get("ssrf"),
        "scheduler": "running" if scheduler is not None else "not_started",
    }


@router.post("/media/analyze", summary="链接分析：下载 -> 字幕/ASR -> 场景边界")
async def media_analyze_endpoint(
    payload: MediaAnalyzeRequest, request: Request
) -> dict[str, Any]:
    """同步等待分析结果（内部走 analyze 任务队列，事件循环不被阻塞）。

    返回 ``{ok,title,duration,path,subtitles_from,segments[],scenes[],frames[]}``。
    站点有字幕时优先用站点字幕（``subtitles_from="site"``），否则跑 ASR。
    """
    cfg = get_cfg(request)
    try:
        action, normalized = prepare_payload(
            "analyze", "analyze.url", payload.model_dump(), cfg
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = await submit_and_wait(
        request,
        "analyze",
        action,
        normalized,
        priority=0,
    )
    return job.get("result") or {}
