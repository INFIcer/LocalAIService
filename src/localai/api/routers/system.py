"""系统级接口：健康检查、能力发现、显存诊断、音色列表、ComfyUI 状态。"""

from __future__ import annotations

import asyncio
import platform
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ... import API_VERSION, __version__
from ...core.voices import scan_voices
from ..deps import get_cfg, get_client, get_registry, get_vram_manager
from ..schemas import VramUnloadRequest

router = APIRouter(tags=["system"])


@router.get("/health", summary="健康检查（不会加载模型）")
async def health(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    registry = get_registry(request)
    vram = get_vram_manager(request)
    media = registry.get("media")

    # ASR / 声纹状态只读文件/内存标志，不会触发加载。
    # 显存数字走 1 秒缓存 + nvidia-smi，避免 /health 触发 CUDA context。
    vram_free, vram_total = vram.health_memory()
    vram_health = {"free_mb": vram_free, "total_mb": vram_total}

    asr = registry.get("asr").status()
    asr_health = {
        "ready": bool(asr["ready"]),
        "loaded": bool(asr["loaded"]),
        "idle_unload_seconds": cfg.asr_idle_unload,
    }
    spk = registry.get("speaker").status()
    speaker_health = {
        "ready": bool(spk["ready"]),
        "loaded": bool(spk["loaded"]),
        "idle_unload_seconds": cfg.speaker_idle_unload,
    }
    sep = registry.get("separate").status()
    separate_health = {
        "ready": bool(sep["ready"]),
        "loaded": bool(sep["loaded"]),
        "idle_unload_seconds": cfg.separate_idle_unload,
        "model": sep.get("loaded_model") or sep.get("model") or "htdemucs",
    }

    # 任务队列：只读数据库计数，不触发任何模型。
    scheduler = getattr(request.app.state, "scheduler", None)
    jobs_health = (
        await scheduler.health()
        if scheduler is not None
        else {"queued": 0, "running": 0, "current_kind": None}
    )

    if cfg.tts_backend == "engine":
        tts = registry.get("tts")
        status = tts.status()
        voices = scan_voices()
        idle = tts.idle_seconds()
        return {
            "ok": bool(status["ready"]),
            "backend": "engine",
            "model_ready": bool(status["ready"]),
            "engine_loaded": bool(status.get("loaded")),
            "voices": len(voices),
            "model_detail": status["detail"],
            "model_dir": status["model_dir"],
            "engine_dir": str(cfg.engine_dir),
            "idle_unload_seconds": cfg.idle_unload,
            "idle_seconds": None if idle is None else round(idle, 1),
            "asr": asr_health,
            "speaker": speaker_health,
            "separate": separate_health,
            "vram": vram_health,
            "media": media.health(),
            "comfy": "disabled",
            "jobs": jobs_health,
        }

    # TTS_BACKEND=comfy：health 的 ok 表示「ComfyUI 可达」。
    from ...engines.tts_comfy import comfy_up

    client = get_client(request)
    up = await comfy_up(client, cfg)
    status = registry.get("tts").status()
    voices = scan_voices()
    return {
        "ok": up,
        "backend": "comfy",
        "comfy": "up" if up else "down",
        "model_ready": status["ready"],
        "voices": len(voices),
        "model_detail": status["detail"],
        "comfy_url": cfg.comfy_base,
        "asr": asr_health,
        "speaker": speaker_health,
        "separate": separate_health,
        "vram": vram_health,
        "media": media.health(),
        "jobs": jobs_health,
    }


@router.get("/capabilities", summary="能力发现（任意程序用它自发现）")
async def capabilities(request: Request) -> dict[str, Any]:
    """返回能力清单、版本、监听地址、显存与耗时配置。

    这是给「第二层」（DSH 插件等调用方）做自发现用的稳定入口：
    插件只需要知道一个 base_url，就能从这里拿到所有能力与端点。
    """
    cfg = get_cfg(request)
    registry = get_registry(request)
    vram = get_vram_manager(request)
    free_mb, total_mb = vram.health_memory()

    caps = {adapter.name: adapter.describe() for adapter in registry.all()}
    snapshot = await asyncio.to_thread(vram.snapshot)
    models = [
        {
            "name": item.get("name"),
            "loaded": item.get("loaded"),
            "estimate_mb": item.get("estimate_mb"),
        }
        for item in snapshot.get("models", [])
    ]
    return {
        "ok": True,
        "api_version": API_VERSION,
        "service": {
            "name": "localai-service",
            "version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "started_at": getattr(request.app.state, "started_at", None),
        },
        "listen": {
            "host": cfg.host,
            "port": cfg.port,
            "base_url": cfg.base_url,
        },
        "backend": cfg.tts_backend,
        "auth": {"required": bool(cfg.token), "scheme": "bearer"},
        "gpu": {
            "available": bool(total_mb),
            "free_mb": free_mb,
            "total_mb": total_mb,
            "margin_mb": cfg.vram_margin_mb,
        },
        "vram": {
            "free_mb": free_mb,
            "total_mb": total_mb,
            "margin_mb": cfg.vram_margin_mb,
            "models": models,
        },
        "capabilities": caps,
        "jobs": {
            "enabled": True,
            "sync_endpoints_use_queue": True,
            "max_parallel": cfg.jobs_max_parallel,
            "aging_seconds": cfg.jobs_aging_seconds,
            "timeout_seconds": cfg.jobs_timeout,
            "recover": cfg.jobs_recover,
        },
        "concurrency": {
            "upload_max_mb": 200,
            "audio_upload_max_mb": 50,
            "tts_max_text_chars": 5000,
            "job_timeout_seconds": cfg.job_timeout,
        },
        "defaults": {
            "idle_unload_seconds": {
                "tts": cfg.idle_unload,
                "asr": cfg.asr_idle_unload,
                "speaker": cfg.speaker_idle_unload,
                "separate": cfg.separate_idle_unload,
            },
        },
        "files": {
            "voices_dir": str(cfg.voices_dir),
            "output_dir": str(cfg.output_dir),
            "speakers_dir": str(cfg.speakers_dir),
            "logs_dir": str(cfg.logs_dir),
        },
        "links": {"docs": "/docs", "openapi": "/openapi.json"},
    }


@router.get("/vram", summary="显存诊断与各能力加载状态")
async def vram_status(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    vram = get_vram_manager(request)
    snapshot = await asyncio.to_thread(vram.snapshot)
    snapshot["estimates"] = vram.estimates()
    snapshot["margin_mb"] = cfg.vram_margin_mb
    return snapshot


@router.post("/vram/unload", summary="手动卸载指定/全部模型")
async def vram_unload(payload: VramUnloadRequest, request: Request) -> dict[str, Any]:
    registry = get_registry(request)
    vram = get_vram_manager(request)
    if not payload.all and not payload.models:
        return {"ok": True, "released": [], "detail": "没有指定要卸载的模型"}
    known = {adapter.name for adapter in registry.gpu_adapters()}
    unknown = [name for name in payload.models if name not in known]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"不认识的模型名：{', '.join(unknown)}；可选 {'/'.join(sorted(known))}",
        )
    result = await vram.unload_models(
        None if payload.all else payload.models,
        unload_all=payload.all,
    )
    return result


@router.get("/voices", summary="列出可用音色")
async def list_voices(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    voices = scan_voices()
    return {
        "voices": [voice.to_dict() for voice in voices],
        "count": len(voices),
        "directory": str(cfg.voices_dir),
    }


@router.get("/comfy/status", summary="ComfyUI 队列状态（仅 comfy 后端有意义）")
async def comfy_status(request: Request) -> JSONResponse:
    cfg = get_cfg(request)
    client = get_client(request)
    try:
        response = await client.get(f"{cfg.comfy_base}/queue", timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=502,
            content={"ok": False, "comfy": "down", "error": str(exc)},
        )
    if response.status_code != 200:
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "comfy": "down",
                "error": f"ComfyUI /queue 返回 HTTP {response.status_code}",
                "body": response.text,
            },
        )
    try:
        queue = response.json()
    except ValueError:
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "comfy": "down",
                "error": "ComfyUI /queue 返回的不是 JSON",
                "body": response.text,
            },
        )
    running = queue.get("queue_running") or []
    pending = queue.get("queue_pending") or []
    return JSONResponse(
        content={
            "ok": True,
            "comfy": "up",
            "running": len(running),
            "pending": len(pending),
            "queue_running": running,
            "queue_pending": pending,
        }
    )
