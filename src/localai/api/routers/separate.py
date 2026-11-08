"""人声分离路由：/separate、/api/v1/separate、/separate/status。

处理链路：上传 -> ffmpeg 转 44.1k 立体声 -> Demucs 分离 -> 每个 stem 落盘。
输出目录：``<OUTPUT>/separate/<时间戳>_<6位>/<stem>.wav``
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ...constants import SEPARATE_MODELS, SEPARATE_STEMS
from ...jobs import prepare_payload
from ..deps import get_cfg
from ..job_helpers import cleanup_job_upload, submit_and_wait
from ..uploads import save_job_upload

logger = logging.getLogger("localai.api.separate")

router = APIRouter(tags=["separate"])


@router.get("/separate/status", summary="人声分离引擎状态")
async def separate_status_endpoint(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    adapter = get_registry(request).get("separate")
    status = adapter.status()
    idle = adapter.idle_seconds()
    return {
        "ready": bool(status["ready"]),
        "loaded": bool(status["loaded"]),
        "model": status.get("loaded_model") or status.get("model"),
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": cfg.separate_idle_unload,
        "available": status.get("available") or [],
        "weights_dir": status.get("weights_dir"),
        "device": status.get("device"),
        "detail": status["detail"],
    }


async def _handle_separate_request(
    request: Request,
    file: UploadFile,
    model: str,
    stems: str,
) -> JSONResponse:
    """旧同步入口：校验 -> 落盘 -> 提交队列 -> 等完成 -> 原样返回结果。"""
    cfg = get_cfg(request)
    model_name = (model or "htdemucs").strip().lower() or "htdemucs"
    if model_name not in SEPARATE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 model={model!r}，可选：{', '.join(SEPARATE_MODELS)}",
        )

    raw_stems = (stems or "vocals,drums,bass,other").strip() or "vocals,drums,bass,other"
    stem_names: list[str] = []
    for part in raw_stems.split(","):
        name = part.strip().lower()
        if name and name not in stem_names:
            stem_names.append(name)
    if not stem_names:
        raise HTTPException(status_code=400, detail="stems 不能为空")
    allowed = SEPARATE_STEMS[model_name]
    unknown = [name for name in stem_names if name not in allowed]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"模型 {model_name} 没有这些 stem：{', '.join(unknown)}；"
                f"可用：{', '.join(allowed)}"
            ),
        )

    raw_name = Path(file.filename or "upload.bin").name
    job_id = uuid.uuid4().hex
    tmp_path = await save_job_upload(cfg, job_id, file)
    try:
        action, payload = prepare_payload(
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

    job = await submit_and_wait(request, "separate", action, payload, job_id=job_id)
    return JSONResponse(content=job.get("result") or {})


@router.post("/separate", summary="人声分离（返回 stem 路径）")
async def separate_endpoint(
    request: Request,
    file: UploadFile = File(..., description="要分离的音频或视频文件"),
    model: str = Form("htdemucs", description="htdemucs / htdemucs_ft / htdemucs_6s"),
    stems: str = Form("vocals,drums,bass,other", description="逗号分隔的 stem 名"),
) -> JSONResponse:
    return await _handle_separate_request(request, file, model, stems)


@router.post("/api/v1/separate", summary="`/separate` 的别名")
async def api_v1_separate(
    request: Request,
    file: UploadFile = File(..., description="要分离的音频或视频文件"),
    model: str = Form("htdemucs", description="htdemucs / htdemucs_ft / htdemucs_6s"),
    stems: str = Form("vocals,drums,bass,other", description="逗号分隔的 stem 名"),
) -> JSONResponse:
    return await _handle_separate_request(request, file, model, stems)
