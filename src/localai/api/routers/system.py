"""系统级接口：健康检查、能力发现、显存诊断、音色列表、ComfyUI 状态。"""

from __future__ import annotations

import asyncio
import platform
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ... import API_VERSION, __version__
from ...constants import (
    AUDIO_MAX_BYTES,
    MEDIA_MAX_BYTES,
    TRANSLATE_BATCH_CHARS,
    TRANSLATE_BATCH_MAX,
    TRANSLATE_MAX_TEXT,
    TTS_MAX_TEXT,
)
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
    # 各能力的空闲卸载阈值统一来自 config.idle_unload_map()。
    idle = cfg.idle_unload_map()

    # ASR / 声纹状态只读文件/内存标志，不会触发加载。
    # 显存数字走 1 秒缓存 + nvidia-smi，避免 /health 触发 CUDA context。
    vram_free, vram_total = vram.health_memory()
    vram_health = {"free_mb": vram_free, "total_mb": vram_total}

    asr = registry.get("asr").status()
    asr_health = {
        "ready": bool(asr["ready"]),
        "loaded": bool(asr["loaded"]),
        "engine": cfg.asr_engine,
        "idle_unload_seconds": idle["asr"],
        "vad": asr.get("vad"),
    }
    whisper = registry.get("whisper").status()
    whisper_health = {
        "ready": bool(whisper.get("ready")),
        "loaded": bool(whisper.get("loaded")),
        "model": whisper.get("model") or f"faster-whisper-{cfg.whisper_model}",
        "idle_unload_seconds": idle["whisper"],
    }
    translate_status = registry.get("translate").status()
    translate_health = {
        "ready": bool(translate_status.get("ready")),
        "loaded": bool(translate_status.get("loaded")),
        "model": translate_status.get("model") or "nllb-200-distilled-600M",
        "idle_unload_seconds": idle["translate"],
        "device": translate_status.get("device"),
        "dtype": translate_status.get("dtype"),
    }
    spk = registry.get("speaker").status()
    speaker_health = {
        "ready": bool(spk["ready"]),
        "loaded": bool(spk["loaded"]),
        "idle_unload_seconds": idle["speaker"],
    }
    sep = registry.get("separate").status()
    separate_health = {
        "ready": bool(sep["ready"]),
        "loaded": bool(sep["loaded"]),
        "idle_unload_seconds": idle["separate"],
        "model": sep.get("loaded_model") or sep.get("model") or "htdemucs",
    }
    vision_status = registry.get("vision").status()
    vision_health = {
        "ready": bool(vision_status.get("ready")),
        "loaded": bool(vision_status.get("loaded")),
        "model": vision_status.get("model"),
        "runtime": vision_status.get("runtime"),
        "runtime_version": vision_status.get("runtime_version"),
        "port": vision_status.get("port"),
        "idle_unload_seconds": idle["vision"],
        "model_size_mb": vision_status.get("model_size_mb"),
        "mmproj_size_mb": vision_status.get("mmproj_size_mb"),
        "detail": vision_status.get("detail"),
    }
    download_health = registry.get("download").status()
    download_health = {
        "ready": bool(download_health.get("ready")),
        "version": download_health.get("version"),
        "path": download_health.get("path"),
        "aria2c": bool((download_health.get("aria2c") or {}).get("available")),
        "download_dir": str(cfg.downloads_dir),
        "limits": {
            "max_mb": cfg.download_max_mb,
            "sync_max_mb": cfg.download_sync_max_mb,
            "timeout_seconds": cfg.download_timeout,
        },
    }

    # 任务队列：只读数据库计数，不触发任何模型。
    scheduler = getattr(request.app.state, "scheduler", None)
    jobs_health = (
        await scheduler.health()
        if scheduler is not None
        else {"queued": 0, "running": 0, "current_kind": None}
    )

    # 与后端无关的部分只写一次；两种 backend 都给出同一组字段。
    common: dict[str, Any] = {
        "asr": asr_health,
        "whisper": whisper_health,
        "translate": translate_health,
        "speaker": speaker_health,
        "separate": separate_health,
        "download": download_health,
        "vision": vision_health,
        "vram": vram_health,
        "media": media.health(),
        "jobs": jobs_health,
    }
    tts_status = registry.get("tts").status()
    voices = scan_voices()

    if cfg.tts_backend == "engine":
        tts_idle = registry.get("tts").idle_seconds()
        return {
            "ok": bool(tts_status["ready"]),
            "backend": "engine",
            "model_ready": bool(tts_status["ready"]),
            "engine_loaded": bool(tts_status.get("loaded")),
            "voices": len(voices),
            "model_detail": tts_status["detail"],
            "model_dir": tts_status.get("model_dir"),
            "engine_dir": str(cfg.engine_dir),
            "idle_unload_seconds": idle["tts"],
            "idle_seconds": None if tts_idle is None else round(tts_idle, 1),
            "comfy": "disabled",
            **common,
        }

    # TTS_BACKEND=comfy：health 的 ok 表示「ComfyUI 可达」。
    from ...engines.tts_comfy import comfy_up

    client = get_client(request)
    up = await comfy_up(client, cfg)
    return {
        "ok": up,
        "backend": "comfy",
        "comfy": "up" if up else "down",
        "comfy_url": cfg.comfy_base,
        "model_ready": bool(tts_status["ready"]),
        "voices": len(voices),
        "model_detail": tts_status["detail"],
        "model_dir": tts_status.get("model_dir") or str(cfg.comfy_model_dir),
        "engine_dir": str(cfg.engine_dir),
        "idle_unload_seconds": idle["tts"],
        "idle_seconds": None,
        **common,
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
    idle = cfg.idle_unload_map()
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
        "asr": {
            "engine": cfg.asr_engine,
            "engines": ["sensevoice", "whisper"],
            "whisper": {
                "model": cfg.whisper_model,
                "device": cfg.whisper_device,
                "compute_type": cfg.whisper_compute,
                "weights_dir": str(cfg.whisper_weights_dir(cfg.whisper_model)),
                "idle_unload_seconds": idle["whisper"],
            },
        },
        "auth": {"required": bool(cfg.token), "scheme": "bearer"},
        "download": {
            "engine": "yt-dlp",
            **registry.get("download").status(),
            "limits": {
                "max_mb": cfg.download_max_mb,
                "sync_max_mb": cfg.download_sync_max_mb,
                "timeout_seconds": cfg.download_timeout,
            },
            "endpoints": [
                "/media/download",
                "/media/download/sync",
                "/media/download/status",
                "/media/analyze",
                "/jobs/analyze",
            ],
            "ssrf": "只允许 http/https；私有/回环/链路本地地址与 file:// 一律拒绝",
            "cookies": "B站自动读取本机凭证；其它站点需显式传 Netscape cookie 文件",
        },
        "vision": {
            **registry.get("vision").status(),
            "planner_default": cfg.vision_planner,
            "planners": ["hybrid", "agent"],
            "idle_unload_seconds": idle["vision"],
            "phash_distance": cfg.vision_phash_distance,
            "prompts_dir": str(Path(cfg.home) / "docs" / "prompts"),
            "endpoints": [
                "/vision/describe",
                "/vision/scan",
                "/vision/frames",
                "/vision/status",
                "/jobs/vision",
            ],
            "limits": {"target_frames": [1, 10], "max_rounds": [1, 12]},
            "note": "llama.cpp 子进程；启动 0 模型，首次请求才加载；与 TTS 按 LRU 互卸",
        },
        "translate": {
            "engine": "nllb",
            "model": "nllb-200-distilled-600M",
            "weights_dir": str(cfg.translate_weight_dir),
            "idle_unload_seconds": idle["translate"],
            "fp16": cfg.translate_fp16,
            "num_beams": cfg.translate_num_beams,
            "batch_size": cfg.translate_batch_size,
            "max_text_chars": TRANSLATE_MAX_TEXT,
            "batch_max_items": TRANSLATE_BATCH_MAX,
            "batch_max_chars": TRANSLATE_BATCH_CHARS,
            "languages": registry.get("translate").languages(),
            "endpoints": ["/translate", "/translate/batch", "/translate/status", "/jobs/translate"],
            "note": "source=auto 只覆盖常见语种；检测失败会要求显式给 source",
        },
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
            "order": "kind_first > priority > created_at（无 aging）",
            "max_parallel": cfg.jobs_max_parallel,
            "per_kind_parallel": cfg.jobs_parallel_map(),
            "asr_parallel": cfg.asr_parallel_effective(),
            "parallel_extra_mb": cfg.jobs_parallel_extra_mb,
            "timeout_seconds": cfg.jobs_timeout,
            "recover": cfg.jobs_recover,
        },
        "concurrency": {
            "upload_max_mb": MEDIA_MAX_BYTES // (1024 * 1024),
            "audio_upload_max_mb": AUDIO_MAX_BYTES // (1024 * 1024),
            "tts_max_text_chars": TTS_MAX_TEXT,
            "translate_max_text_chars": TRANSLATE_MAX_TEXT,
            "translate_batch_max_items": TRANSLATE_BATCH_MAX,
            "translate_batch_max_chars": TRANSLATE_BATCH_CHARS,
            "job_timeout_seconds": cfg.job_timeout,
        },
        "defaults": {"idle_unload_seconds": idle},
        "files": {
            "voices_dir": str(cfg.voices_dir),
            "output_dir": str(cfg.output_dir),
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
