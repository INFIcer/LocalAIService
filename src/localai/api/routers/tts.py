"""TTS 路由：/tts、/v1/audio/speech，以及 DSH 语音插件兼容接口。

兼容接口（``@memef1f1y/dsh-plugin-tts`` 的 index-tts2 provider）：

- ``GET  /api/v1/voices``                  -> ``{"voices": ["default", ...]}``
- ``POST /api/v1/tts/tasks`` ``{text,...}`` -> audio/wav 二进制
- ``POST /api/v1/upload`` (multipart file)  -> 存进音色目录
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ...constants import AUDIO_EXTENSIONS, AUDIO_MAX_BYTES
from ...core.audio import audio_headers
from ...core.errors import EngineBackendError, VoiceNotFound
from ...core.voices import pick_voice_name, scan_voices
from ..deps import get_cfg, get_client, get_registry
from ..schemas import CompatTaskRequest, SpeechRequest, TTSRequest

logger = logging.getLogger("localai.api.tts")

router = APIRouter(tags=["tts"])


def _audio_response(result: Any) -> Response:
    headers = audio_headers(result.path, result.seed)
    headers["X-Language"] = result.language
    if result.comfy_filename:
        headers["X-Comfy-Filename"] = result.comfy_filename
    return Response(content=result.wav, media_type="audio/wav", headers=headers)


@router.post("/tts", summary="语音合成，返回 WAV 二进制")
async def tts(payload: TTSRequest, request: Request) -> Response:
    adapter = get_registry(request).get("tts")
    result = await adapter.infer(
        client=get_client(request),
        text=payload.text,
        voice_name=payload.voice,
        speed=payload.speed,
        emotion=payload.emotion,
        seed=payload.seed,
        language=payload.language,
    )
    return _audio_response(result)


@router.post("/v1/audio/speech", summary="OpenAI 兼容语音合成")
async def openai_speech(payload: SpeechRequest, request: Request) -> Response:
    response_format = (payload.response_format or "").strip().lower()
    if response_format != "wav":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"只支持 response_format='wav'，收到 {payload.response_format!r}；"
                    "本服务统一返回 16bit PCM WAV。",
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": "unsupported_response_format",
                }
            },
        )

    adapter = get_registry(request).get("tts")
    result = await adapter.infer(
        client=get_client(request),
        text=payload.input,
        voice_name=payload.voice,
        speed=payload.speed,
        emotion=payload.emotion,
        seed=payload.seed,
        language=payload.language,
    )
    return _audio_response(result)


@router.get("/api/v1/voices", summary="DSH 插件兼容：音色名列表")
async def api_v1_voices() -> dict[str, Any]:
    voices = scan_voices()
    names = [v.name for v in voices]
    return {"voices": names, "count": len(names)}


@router.post("/api/v1/tts/tasks", summary="DSH 插件兼容：合成任务，返回 WAV")
async def api_v1_tts_tasks(payload: CompatTaskRequest, request: Request) -> Response:
    text = (payload.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"detail": "text 不能为空"})
    voice_name = pick_voice_name(payload.prompt_audio)
    speed = payload.speed if payload.speed and 0.5 <= payload.speed <= 2.0 else 1.0
    adapter = get_registry(request).get("tts")
    try:
        result = await adapter.infer(
            client=get_client(request),
            text=text,
            voice_name=voice_name,
            speed=speed,
            emotion=None,
            seed=payload.seed,
            language=payload.language,
        )
    except VoiceNotFound as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except EngineBackendError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
    headers = audio_headers(result.path, result.seed)
    return Response(content=result.wav, media_type="audio/wav", headers=headers)


@router.post("/api/v1/upload", summary="DSH 插件兼容：上传音色到音色目录")
async def api_v1_upload(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    cfg = get_cfg(request)
    raw_name = Path(file.filename or "upload.wav").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"只支持这些音频格式：{', '.join(sorted(AUDIO_EXTENSIONS))}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > AUDIO_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"音频太大（上限 {AUDIO_MAX_BYTES // (1024 * 1024)}MB）",
        )
    cfg.voices_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.voices_dir / raw_name
    target.write_bytes(data)
    logger.info("上传音色：%s（%d 字节）", target, len(data))
    return {"ok": True, "name": target.stem, "filename": target.name, "size": len(data)}
