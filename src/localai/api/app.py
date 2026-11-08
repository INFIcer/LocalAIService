"""FastAPI 装配：一个进程、一个端口提供全部能力。

分层约定
--------
- 本模块只做「装配」：配置、注册表、显存管家、异常映射、中间件、路由挂载；
- 具体能力在 ``localai.engines``，具体接口在 ``localai.api.routers``；
- 启动时 **0 模型**：只注册卸载函数与看门狗，不加载任何权重。
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import Config, ensure_directories, get_config
from ..core.errors import (
    AsrBackendError,
    ComfyError,
    EngineBackendError,
    InsufficientVramError,
    SeparateBackendError,
    SpeakerBackendError,
    VoiceNotFound,
)
from ..core.idle import IdleWatchdog
from ..core.vram import register_adapters
from ..engines import build_registry
from .routers import asr, media, separate, speakers, system, tts

logger = logging.getLogger("localai.api")

# 这些路径不要求鉴权：健康检查、能力发现、OpenAPI 文档。
_AUTH_EXEMPT = {"/health", "/capabilities", "/docs", "/redoc", "/openapi.json"}


def _http_client(cfg: Config) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.job_timeout + 30.0, connect=5.0, write=30.0),
        follow_redirects=True,
    )


def create_app(cfg: Optional[Config] = None) -> FastAPI:
    """构造 FastAPI 应用；测试里可以传自定义 Config。"""
    cfg = cfg or get_config()
    cfg.export_env()

    registry = build_registry(cfg)
    vram = register_adapters(registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ensure_directories(cfg)
        app.state.client = _http_client(cfg)
        app.state.idle_tasks = []

        logger.info(
            "本地 AI 服务启动：backend=%s 音色目录=%s 输出目录=%s 监听=%s",
            cfg.tts_backend,
            cfg.voices_dir,
            cfg.output_dir,
            cfg.base_url,
        )

        tts_adapter = registry.get("tts")
        tts_status = tts_adapter.status()
        logger.info("引擎模型状态：%s", tts_status["detail"])
        if cfg.tts_backend == "engine":
            logger.info("引擎目录：%s（首次 /tts 请求时加载模型）", cfg.engine_dir)
        else:
            logger.info("ComfyUI=%s 模型状态：%s", cfg.comfy_base, tts_status["detail"])

        asr_adapter = registry.get("asr")
        logger.info("ASR 模型状态：%s", asr_adapter.status()["detail"])
        logger.info("ASR 模型目录：%s（首次 /asr 请求时加载）", cfg.asr_model_dir)

        media_adapter = registry.get("media")
        media_health = media_adapter.health()
        if media_health.get("ffmpeg") == "ok":
            logger.info("媒体工具就绪：ffmpeg=%s", media_health.get("path"))
        else:
            logger.warning(
                "媒体工具不可用：ffmpeg 未找到（期望 %s）。/media/* 与视频转写会失败；"
                "可设置环境变量 LOCALAI_FFMPEG 指向 ffmpeg.exe。",
                media_health.get("expected"),
            )

        logger.info("声纹模型状态：%s", registry.get("speaker").status()["detail"])
        logger.info("声纹模型目录：%s（首次声纹请求时加载）", cfg.speaker_model_dir)

        sep_status = registry.get("separate").status()
        logger.info("人声分离模型状态：%s", sep_status["detail"])
        logger.info("Demucs 权重目录：%s（首次 /separate 请求时加载）", sep_status.get("weights_dir"))

        # 每个 GPU 能力一个独立看门狗；TTS 只在 engine 后端下启用。
        for adapter in registry.gpu_adapters():
            if adapter.name == "tts" and cfg.tts_backend != "engine":
                continue
            seconds = adapter.idle_unload_seconds()
            if seconds <= 0:
                continue
            watchdog = IdleWatchdog(adapter, seconds)
            task = asyncio.create_task(watchdog.run())
            app.state.idle_tasks.append(task)

        try:
            yield
        finally:
            for task in getattr(app.state, "idle_tasks", []):
                task.cancel()
            for task in getattr(app.state, "idle_tasks", []):
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await app.state.client.aclose()

    app = FastAPI(
        title="本地 AI 服务管理器（TTS + ASR + 声纹 + 媒体 + 人声分离）",
        description=(
            "一个端口提供 TTS（IndexTTS-2.5）、ASR（SenseVoiceSmall）、声纹（campplus）、"
            "ffmpeg 媒体工具与 Demucs 人声分离；模型懒加载 + 空闲卸载。\n\n"
            "能力发现：`GET /capabilities`；调用方文档：`docs/SERVICE_API.md`。"
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.state.cfg = cfg
    app.state.registry = registry
    app.state.vram = vram
    app.state.started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # 异常 -> JSON（字段与旧版保持一致）
    # ------------------------------------------------------------------
    @app.exception_handler(ComfyError)
    async def _comfy_error_handler(request: Request, exc: ComfyError) -> JSONResponse:
        logger.error("ComfyUI 错误：%s | %s", exc.message, exc.comfy_error)
        if request.url.path.startswith("/v1/"):
            content: dict[str, Any] = {
                "error": {
                    "message": exc.message,
                    "type": "upstream_error",
                    "code": "comfy_error",
                    "comfy_error": exc.comfy_error,
                    "comfy_messages": exc.messages,
                    "prompt_id": exc.prompt_id,
                }
            }
        else:
            content = {
                "ok": False,
                "error": "comfy_error",
                "detail": exc.message,
                "comfy_error": exc.comfy_error,
                "comfy_messages": exc.messages,
                "prompt_id": exc.prompt_id,
            }
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(VoiceNotFound)
    async def _voice_not_found_handler(request: Request, exc: VoiceNotFound) -> JSONResponse:
        if request.url.path.startswith("/v1/"):
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "code": "voice_not_found",
                    }
                },
            )
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "voice_not_found", "detail": str(exc)},
        )

    @app.exception_handler(EngineBackendError)
    async def _engine_error_handler(request: Request, exc: EngineBackendError) -> JSONResponse:
        logger.error("独立引擎错误：%s | %s", exc.message, exc.detail or "")
        if request.url.path.startswith("/v1/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": exc.message,
                        "type": "upstream_error",
                        "code": "engine_error",
                    }
                },
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "ok": False,
                "error": "engine_error",
                "detail": exc.message,
                "backend": "engine",
            },
        )

    @app.exception_handler(AsrBackendError)
    async def _asr_error_handler(request: Request, exc: AsrBackendError) -> JSONResponse:
        logger.error("ASR 引擎错误：%s | %s", exc.message, exc.detail or "")
        if request.url.path.startswith("/v1/"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": exc.message,
                        "type": "upstream_error",
                        "code": "asr_error",
                    }
                },
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": "asr_error", "detail": exc.message},
        )

    @app.exception_handler(SpeakerBackendError)
    async def _speaker_error_handler(request: Request, exc: SpeakerBackendError) -> JSONResponse:
        logger.error("声纹引擎错误：%s | %s", exc.message, exc.detail or "")
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": "speaker_error", "detail": exc.message},
        )

    @app.exception_handler(SeparateBackendError)
    async def _separate_error_handler(request: Request, exc: SeparateBackendError) -> JSONResponse:
        logger.error("人声分离引擎错误：%s | %s", exc.message, exc.detail or "")
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": "separate_error", "detail": exc.message},
        )

    @app.exception_handler(InsufficientVramError)
    async def _insufficient_vram_handler(
        request: Request, exc: InsufficientVramError
    ) -> JSONResponse:
        """显存不足统一返回 507；字段固定，调用方可以据此提示/重试。"""
        result = exc.result or {}
        logger.error(
            "显存不足：%s | need=%s free=%s released=%s",
            exc.message,
            result.get("need_mb"),
            result.get("free_mb"),
            result.get("released"),
        )
        return JSONResponse(
            status_code=507,
            content={
                "ok": False,
                "error": "insufficient_vram",
                "detail": exc.message,
                "need_mb": result.get("need_mb"),
                "free_mb": result.get("free_mb"),
                "released": result.get("released") or [],
                "shortfall_mb": result.get("shortfall_mb"),
            },
        )

    # ------------------------------------------------------------------
    # 可选鉴权（LOCALAI_TOKEN 非空时启用；默认关闭）
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next: Any) -> Any:
        if not cfg.token or request.url.path in _AUTH_EXEMPT:
            return await call_next(request)
        provided = ""
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided:
            provided = request.headers.get("x-localai-token", "").strip()
        if not provided or not hmac.compare_digest(provided, cfg.token):
            return JSONResponse(
                status_code=401,
                content={
                    "ok": False,
                    "error": "unauthorized",
                    "detail": "需要 Bearer 令牌（服务端设置了 LOCALAI_TOKEN）",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    app.include_router(system.router)
    app.include_router(tts.router)
    app.include_router(asr.router)
    app.include_router(speakers.router)
    app.include_router(separate.router)
    app.include_router(media.router)

    return app


# uvicorn / 旧入口统一使用这个实例。
app = create_app()
