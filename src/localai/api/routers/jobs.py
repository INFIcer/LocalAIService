"""任务队列 HTTP 接口：提交 / 查询 / 取消 / 统计 / 取结果。

与旧接口的关系
--------------
- 旧同步接口语义完全不变（TTS 仍返回 WAV，ASR/分离仍返回原来的 JSON），
  只是内部改成「提交任务 -> 等完成」；
- 这里新增的是异步用法：提交后立刻返回 ``id``，调用方自己轮询
  ``GET /jobs/{id}`` 或直接取 ``GET /jobs/{id}/result``。

路由顺序刻意把 ``/jobs/stats`` 等静态路径放在 ``/jobs/{job_id}`` 之前。
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ...constants import (
    ASR_LANGUAGES,
    MEDIA_EXTENSIONS,
    MEDIA_MAX_BYTES,
    SEPARATE_MODELS,
    SEPARATE_STEMS,
)
from ...jobs import JOB_STATES, prepare_payload
from ..deps import get_cfg, get_jobs_store, get_scheduler
from ..job_helpers import cleanup_job_upload, client_ip
from ..schemas import JobCreateRequest, JobTtsRequest
from ..uploads import save_job_upload

logger = logging.getLogger("localai.api.jobs")

router = APIRouter(tags=["jobs"])

_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac"}


def _public(job: dict[str, Any], *, with_ok: bool = False) -> dict[str, Any]:
    item = dict(job)
    item["result"] = job.get("result")
    if with_ok:
        item["ok"] = True
    return item


def _not_found(job_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"任务不存在：{job_id}")


def _queued_response(job_id: str) -> dict[str, Any]:
    return {"ok": True, "id": job_id, "state": "queued"}


# ---------------------------------------------------------------------------
# 提交
# ---------------------------------------------------------------------------

@router.post("/jobs", summary="提交一个异步任务（JSON）")
async def create_job(payload: JobCreateRequest, request: Request) -> dict[str, Any]:
    scheduler = get_scheduler(request)
    cfg = get_cfg(request)
    try:
        action, normalized = prepare_payload(payload.kind, payload.action, payload.payload, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = await scheduler.submit(
        payload.kind.strip().lower(),
        action,
        normalized,
        priority=payload.priority,
        client_ip=client_ip(request),
        meta=payload.meta,
    )
    return _queued_response(job_id)


@router.post("/jobs/tts", summary="提交 TTS 任务（JSON），立刻返回任务 id")
async def create_tts_job(payload: JobTtsRequest, request: Request) -> dict[str, Any]:
    scheduler = get_scheduler(request)
    cfg = get_cfg(request)
    data = payload.model_dump(exclude={"priority"}, exclude_none=False)
    try:
        action, normalized = prepare_payload("tts", "tts.synthesize", data, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = await scheduler.submit(
        "tts",
        action,
        normalized,
        priority=payload.priority,
        client_ip=client_ip(request),
    )
    return _queued_response(job_id)


@router.post("/jobs/asr", summary="提交 ASR 任务（multipart 上传）")
async def create_asr_job(
    request: Request,
    file: UploadFile = File(..., description="音频或视频文件"),
    language: str = Form("auto", description="auto/zh/en/ja/ko/yue..."),
    speaker: str = Form("1", description="1=识别声纹，0=跳过"),
    engine: str = Form("", description="sensevoice / whisper / auto；空=按配置默认"),
    timestamps: str = Form("0", description="1=需要时间戳（engine=auto 时选 whisper）"),
    priority: int = Form(0, description="优先级，越大越先"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    scheduler = get_scheduler(request)
    raw_name = Path(file.filename or "audio.wav").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件类型 {suffix or '(无扩展名)'}；"
                f"支持：{', '.join(sorted(MEDIA_EXTENSIONS))}"
            ),
        )
    lang = (language or "auto").strip().lower() or "auto"
    if lang not in ASR_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 language={language!r}，可选：{', '.join(ASR_LANGUAGES)}",
        )
    job_id = uuid.uuid4().hex
    tmp_path = await save_job_upload(
        cfg,
        job_id,
        file,
        max_bytes=MEDIA_MAX_BYTES,
        too_large_status=400,
    )
    try:
        action, normalized = prepare_payload(
            "asr",
            "asr.transcribe",
            {
                "file": str(tmp_path),
                "language": lang,
                "speaker": speaker,
                "engine": engine,
                "timestamps": timestamps,
                "original_name": raw_name,
            },
            cfg,
        )
    except ValueError as exc:
        cleanup_job_upload(cfg, job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await scheduler.submit(
        "asr",
        action,
        normalized,
        priority=priority,
        client_ip=client_ip(request),
        job_id=job_id,
    )
    return _queued_response(job_id)


@router.post("/jobs/separate", summary="提交人声分离任务（multipart 上传）")
async def create_separate_job(
    request: Request,
    file: UploadFile = File(..., description="音频或视频文件"),
    model: str = Form("htdemucs"),
    stems: str = Form("vocals,drums,bass,other"),
    priority: int = Form(0),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    scheduler = get_scheduler(request)
    model_name = (model or "htdemucs").strip().lower() or "htdemucs"
    if model_name not in SEPARATE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 model={model!r}，可选：{', '.join(SEPARATE_MODELS)}",
        )
    stem_names = [part.strip().lower() for part in (stems or "").split(",") if part.strip()]
    if not stem_names:
        raise HTTPException(status_code=400, detail="stems 不能为空")
    unknown = [name for name in stem_names if name not in SEPARATE_STEMS[model_name]]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"模型 {model_name} 没有这些 stem：{', '.join(unknown)}",
        )
    raw_name = Path(file.filename or "upload.bin").name
    job_id = uuid.uuid4().hex
    tmp_path = await save_job_upload(cfg, job_id, file)
    try:
        action, normalized = prepare_payload(
            "separate",
            "separate.split",
            {
                "file": str(tmp_path),
                "model": model_name,
                "stems": stem_names,
                "original_name": raw_name,
            },
            cfg,
        )
    except ValueError as exc:
        cleanup_job_upload(cfg, job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await scheduler.submit(
        "separate",
        action,
        normalized,
        priority=priority,
        client_ip=client_ip(request),
        job_id=job_id,
    )
    return _queued_response(job_id)


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

@router.get("/jobs", summary="任务列表（分页，创建时间倒序）")
async def list_jobs(
    request: Request,
    state: Optional[str] = Query(None, description="queued/running/done/failed/canceled"),
    kind: Optional[str] = Query(None, description="tts/asr/speaker/separate/media"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    if state and state not in JOB_STATES:
        raise HTTPException(status_code=400, detail=f"state 可选：{', '.join(JOB_STATES)}")
    store = get_jobs_store(request)
    jobs, total = await store.list_jobs(state=state, kind=kind, limit=limit, offset=offset)
    return {
        "ok": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "jobs": [_public(job) for job in jobs],
    }


@router.get("/jobs/stats", summary="队列统计与调度器状态")
async def job_stats(request: Request) -> dict[str, Any]:
    scheduler = get_scheduler(request)
    stats = await scheduler.stats()
    return {"ok": True, **stats}


@router.get("/jobs/{job_id}", summary="任务详情（含 result/error）")
async def job_detail(job_id: str, request: Request) -> dict[str, Any]:
    store = get_jobs_store(request)
    job = await store.get(job_id)
    if job is None:
        raise _not_found(job_id)
    return _public(job, with_ok=True)


@router.post("/jobs/{job_id}/cancel", summary="取消任务（queued 立即；running 尽力）")
async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    scheduler = get_scheduler(request)
    store = get_jobs_store(request)
    before = await store.get(job_id)
    if before is None:
        raise _not_found(job_id)
    if before["state"] in ("done", "failed", "canceled"):
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "id": job_id,
                "state": before["state"],
                "error": "not_cancelable",
                "detail": "任务已经结束，不能取消",
            },
        )
    job = await scheduler.cancel(job_id)
    if job is None:
        raise _not_found(job_id)
    return {
        "ok": True,
        "id": job_id,
        "state": job["state"],
        "cancel_requested": bool((job.get("meta") or {}).get("cancel_requested")),
    }


@router.get("/jobs/{job_id}/result", summary="取任务结果；音频直接返回文件")
async def job_result(job_id: str, request: Request):
    store = get_jobs_store(request)
    job = await store.get(job_id)
    if job is None:
        raise _not_found(job_id)
    if job["state"] != "done":
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "id": job_id,
                "state": job["state"],
                "error": "job_not_done",
                "detail": job.get("error") or "任务尚未完成",
            },
        )
    result = job.get("result") or {}
    raw_path = result.get("path") or result.get("output_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES:
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return FileResponse(
                path,
                media_type=media_type,
                filename=path.name,
                headers={"Cache-Control": "no-store"},
            )
    return JSONResponse({"ok": True, "id": job_id, "state": "done", "result": result})
