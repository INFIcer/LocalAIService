"""人声分离路由：/separate、/api/v1/separate、/separate/status。

处理链路：上传 -> ffmpeg 转 44.1k 立体声 -> Demucs 分离 -> 每个 stem 落盘。
输出目录：``<OUTPUT>/separate/<时间戳>_<6位>/<stem>.wav``
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ...constants import SEPARATE_MODELS, SEPARATE_STEMS
from ...core.errors import InsufficientVramError, SeparateBackendError
from ..deps import get_cfg, get_registry
from ..uploads import save_media_upload

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
    cfg = get_cfg(request)
    registry = get_registry(request)
    started = time.perf_counter()
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

    tmp_path = await save_media_upload(cfg, file, "separate")
    converted_path = cfg.separate_tmp_dir / f"separate_{uuid.uuid4().hex}.wav"
    out_dir = cfg.separate_dir / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    media = registry.get("media")
    try:
        # 先统一成 Demucs 训练时的 44.1kHz 立体声，避免容器/采样率差异。
        converted = await media.call("to_wav44k", str(tmp_path), str(converted_path))
        outcome = await registry.get("separate").infer(
            input_wav=str(converted_path),
            out_dir=str(out_dir),
            model_name=model_name,
            stems=stem_names,
        )
        result = outcome["result"]
        released_models = outcome["released_models"]
    except (InsufficientVramError, SeparateBackendError):
        raise
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一成 503 JSON
        raise SeparateBackendError(
            f"Demucs 分离失败：{type(exc).__name__}: {exc}",
            detail=str(exc),
        ) from exc
    finally:
        for temp_file in (tmp_path, converted_path):
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                logger.warning("人声分离临时文件删除失败：%s", temp_file)

    elapsed = round(time.perf_counter() - started, 2)
    logger.info(
        "人声分离完成：%s model=%s stems=%s 时长=%ss 耗时=%.2fs 输出=%s released=%s",
        Path(file.filename or "").name,
        result.get("model"),
        ",".join(stem_names),
        result.get("duration"),
        elapsed,
        result.get("dir"),
        ",".join(released_models) or "-",
    )
    return JSONResponse(
        content={
            "ok": True,
            "model": result.get("model", model_name),
            "sample_rate": result.get("sample_rate", 44100),
            "duration": result.get("duration", converted.get("duration")),
            "elapsed": elapsed,
            "separation_elapsed": result.get("elapsed"),
            "stems": result.get("stems") or {},
            "dir": result.get("dir") or str(out_dir),
            "loaded": registry.get("separate").loaded(),
            "released_models": released_models,
        }
    )


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
