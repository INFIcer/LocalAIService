"""ASR 路由：``/asr``、``/api/v1/asr``、``/asr/status``。

契约：multipart ``file=<音频/视频>``、``language=auto|zh|en|ja|ko|yue``、
``?speaker=0`` 关闭声纹。非 WAV 一律先由 ffmpeg 归一到 16k 单声道 WAV。
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ...constants import (
    ASR_LANGUAGES,
    MEDIA_EXTENSIONS,
    MEDIA_MAX_BYTES,
)
from ...jobs import prepare_payload
from ..deps import get_cfg, get_registry
from ..job_helpers import cleanup_job_upload, submit_and_wait
from ..uploads import save_job_upload

logger = logging.getLogger("localai.api.asr")

router = APIRouter(tags=["asr"])

_SPEAKER_OFF_VALUES = {"0", "false", "off", "no", "none", ""}


def speaker_enabled(value: str) -> bool:
    """把 ``?speaker=`` 的值转成 bool（默认开启）。"""
    return str(value or "").strip().lower() not in _SPEAKER_OFF_VALUES


@router.get("/asr/status", summary="ASR 引擎状态")
async def asr_status_endpoint(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    adapter = get_registry(request).get("asr")
    status = adapter.status()
    idle = adapter.idle_seconds()
    return {
        "ready": bool(status["ready"]),
        "loaded": bool(status["loaded"]),
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": cfg.asr_idle_unload,
        "parallel": status.get("parallel"),
        "detail": status["detail"],
    }


async def _handle_asr_request(
    request: Request,
    file: UploadFile,
    language: str,
    use_speaker: bool = True,
) -> JSONResponse:
    """旧同步入口：校验 -> 落盘 -> 提交队列 -> 等完成 -> 原样返回结果。

    上传文件落到 ``data/uploads/<job_id>/``，任务结束后由调度器按配置清理；
    处理链路（ffmpeg 转 16k、识别、可选声纹）与旧版完全一致。
    """
    cfg = get_cfg(request)
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
        action, payload = prepare_payload(
            "asr",
            "asr.transcribe",
            {
                "file": str(tmp_path),
                "language": lang,
                "speaker": use_speaker,
                "original_name": raw_name,
            },
            cfg,
        )
    except ValueError as exc:
        cleanup_job_upload(cfg, job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = await submit_and_wait(request, "asr", action, payload, job_id=job_id)
    return JSONResponse(content=job.get("result") or {})


@router.post("/asr", summary="语音识别（可选声纹）")
async def asr(
    request: Request,
    file: UploadFile = File(..., description="要识别的音频文件"),
    language: str = Form("auto", description="auto/zh/en/ja/ko/yue..."),
    speaker: str = "1",
) -> JSONResponse:
    return await _handle_asr_request(request, file, language, speaker_enabled(speaker))


@router.post("/api/v1/asr", summary="`/asr` 的别名")
async def api_v1_asr(
    request: Request,
    file: UploadFile = File(..., description="要识别的音频文件"),
    language: str = Form("auto", description="auto/zh/en/ja/ko/yue..."),
    speaker: str = "1",
) -> JSONResponse:
    return await _handle_asr_request(request, file, language, speaker_enabled(speaker))
