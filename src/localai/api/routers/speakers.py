"""声纹路由：/speakers*（campplus 192 维向量，纯分析）。

这里只输出向量；「这个向量是谁」由调用方的记忆层判断，服务端不保存名字、
样本或阈值。上传的原始音频只落临时目录，算完立即删除。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile

from ...core.errors import SpeakerBackendError
from ..deps import get_cfg, get_registry
from ..uploads import save_speaker_upload, unlink_quiet

logger = logging.getLogger("localai.api.speakers")

router = APIRouter(tags=["speaker"])


@router.get("/speakers/status", summary="声纹引擎状态")
async def speakers_status_endpoint(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    adapter = get_registry(request).get("speaker")
    status = adapter.status()
    idle = adapter.idle_seconds()
    return {
        "ready": bool(status["ready"]),
        "loaded": bool(status["loaded"]),
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": cfg.speaker_idle_unload,
        "model": status.get("model"),
        "dim": status.get("dim"),
        "model_dir": status.get("model_dir"),
        "detail": status["detail"],
    }


@router.post("/speakers/embed", summary="算一段音频的声纹向量（192 维，L2 归一化）")
async def embed_speaker(
    request: Request,
    file: UploadFile = File(..., description="要算声纹向量的音频（3~10 秒清晰语音最佳）"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    started = time.perf_counter()
    tmp_path = await save_speaker_upload(cfg, file, "embed")
    try:
        result = await get_registry(request).get("speaker").embed_for(str(tmp_path))
    except SpeakerBackendError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SpeakerBackendError(f"声纹向量计算失败：{type(exc).__name__}: {exc}") from exc
    finally:
        unlink_quiet(tmp_path)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info("声纹向量：dim=%s 耗时=%sms", result["dim"], elapsed_ms)
    return {
        "ok": True,
        "embedding": result["embedding"],
        "dim": result["dim"],
        "model": result["model"],
        "elapsed_ms": elapsed_ms,
    }
