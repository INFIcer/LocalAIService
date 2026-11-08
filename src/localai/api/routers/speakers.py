"""声纹路由：/speakers*（campplus 192 维向量）。

隐私：只把向量和名字写到 ``<HOME>/speakers/*.json``；
上传的原始音频只落临时目录，算完立即删除。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ...core.errors import SpeakerBackendError
from ..deps import get_cfg, get_registry
from ..uploads import save_speaker_upload

logger = logging.getLogger("localai.api.speakers")

router = APIRouter(tags=["speaker"])


@router.get("/speakers", summary="列出已登记的声纹")
async def list_speakers_endpoint(request: Request) -> dict[str, Any]:
    adapter = get_registry(request).get("speaker")
    try:
        speakers = await asyncio.to_thread(adapter.get_module().list_speakers)
    except Exception as exc:  # noqa: BLE001
        raise SpeakerBackendError(f"读取声纹列表失败：{type(exc).__name__}: {exc}") from exc
    return {"ok": True, "speakers": speakers, "count": len(speakers)}


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
        "threshold": status.get("threshold"),
        "model_dir": status.get("model_dir"),
        "speakers_dir": status.get("speakers_dir"),
        "detail": status["detail"],
    }


@router.get("/speakers/unknown", summary="未知声纹聚类")
async def unknown_speakers_endpoint(request: Request) -> dict[str, Any]:
    adapter = get_registry(request).get("speaker")
    try:
        clusters = await asyncio.to_thread(adapter.get_module().unknown_clusters)
    except Exception as exc:  # noqa: BLE001
        raise SpeakerBackendError(f"读取未知声纹聚类失败：{type(exc).__name__}: {exc}") from exc
    return {"ok": True, "unknown": clusters, "count": len(clusters)}


@router.post("/speakers/enroll", summary="登记声纹（同名累加样本）")
async def enroll_speaker(
    request: Request,
    file: UploadFile = File(..., description="这个人的一段清晰语音（3~10 秒最佳）"),
    name: str = Form(..., description="给这个人起的名字，例如「主人」"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    speaker_name = (name or "").strip()
    if not speaker_name:
        raise HTTPException(status_code=400, detail="name 不能为空")
    tmp_path = await save_speaker_upload(cfg, file, "enroll")
    try:
        result = await get_registry(request).get("speaker").call(
            "enroll", speaker_name, str(tmp_path)
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("声纹临时文件删除失败：%s", tmp_path)
    logger.info("声纹登记完成：%s（累计 %s 条样本）", result.get("name"), result.get("samples"))
    return {
        "ok": True,
        "name": result.get("name"),
        "samples": result.get("samples"),
        "created_at": result.get("created_at"),
    }


@router.post("/speakers/identify", summary="判断这段音频是谁")
async def identify_speaker(
    request: Request,
    file: UploadFile = File(..., description="要判断是谁的音频"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    tmp_path = await save_speaker_upload(cfg, file, "identify")
    try:
        result = await get_registry(request).get("speaker").call("identify", str(tmp_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("声纹临时文件删除失败：%s", tmp_path)
    # identify() 的返回值里带 embedding（numpy），不能直接进 JSON，剔掉。
    matches = [
        {"name": m.get("name"), "score": float(m.get("score") or 0.0)}
        for m in (result.get("matches") or [])
    ]
    return {
        "ok": True,
        "matches": matches,
        "best": result.get("best"),
        "score": float(result.get("score") or 0.0),
        "threshold": float(result.get("threshold") or 0.0),
        "known": bool(result.get("known")),
    }


@router.delete("/speakers/{name}", summary="删除一个已登记的声纹")
async def delete_speaker_endpoint(name: str, request: Request) -> dict[str, Any]:
    adapter = get_registry(request).get("speaker")
    try:
        deleted = await asyncio.to_thread(adapter.get_module().delete_speaker, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise SpeakerBackendError(f"删除声纹失败：{type(exc).__name__}: {exc}") from exc
    logger.info("删除声纹 %s：%s", name, "成功" if deleted else "不存在")
    return {"ok": True, "deleted": bool(deleted)}
