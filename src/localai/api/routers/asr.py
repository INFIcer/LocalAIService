"""ASR 路由：``/asr``、``/api/v1/asr``、``/asr/status``。

契约：multipart ``file=<音频/视频>``、``language=auto|zh|en|ja|ko|yue``、
``?speaker=0`` 关闭声纹。非 WAV 一律先由 ffmpeg 归一到 16k 单声道 WAV。
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ...constants import (
    ASR_LANGUAGES,
    MEDIA_EXTENSIONS,
    MEDIA_MAX_BYTES,
    MEDIA_VIDEO_EXTENSIONS,
)
from ...core.errors import AsrBackendError, InsufficientVramError
from ..deps import get_cfg, get_registry

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
        "detail": status["detail"],
    }


async def _handle_asr_request(
    request: Request,
    file: UploadFile,
    language: str,
    use_speaker: bool = True,
) -> JSONResponse:
    """保存上传文件 ->（必要时转 16k wav）-> 识别 -> 删临时文件；两个入口共用。"""
    cfg = get_cfg(request)
    registry = get_registry(request)
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

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > MEDIA_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大（上限 {MEDIA_MAX_BYTES // (1024 * 1024)}MB）",
        )

    cfg.asr_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.asr_tmp_dir / f"asr_{uuid.uuid4().hex}{suffix}"
    wav_path: Optional[Path] = None
    converted = False
    started = time.perf_counter()
    speaker_info: Optional[dict[str, Any]] = None
    speaker_error: Optional[str] = None
    media = registry.get("media")
    asr = registry.get("asr")
    try:
        tmp_path.write_bytes(data)

        # 非 wav 一律先归一成 16k 单声道 wav：视频、m4a、ogg 都能吃。
        # 音频容器转换失败时回退直接识别（旧行为）；视频没有回退余地。
        if suffix != ".wav":
            try:
                conversion = await media.call("to_wav16k", str(tmp_path))
                wav_path = Path(conversion["path"])
                converted = True
            except HTTPException:
                if suffix in MEDIA_VIDEO_EXTENSIONS:
                    raise
                logger.warning("ffmpeg 转换失败，回退直接识别原始文件：%s", raw_name)
        input_path = wav_path or tmp_path

        result = await asr.infer(str(input_path), lang)

        # 声纹：默认开启；?speaker=0 时整段跳过，连 campplus 都不加载。
        # 声纹失败不影响识别结果，只把错误写进 speaker_error。
        if use_speaker:
            try:
                speaker_info = await registry.get("speaker").identify_for(input_path)
            except Exception as exc:  # noqa: BLE001 - 降级：文本照常返回
                speaker_error = f"{type(exc).__name__}: {exc}"
                logger.warning("声纹识别失败（不影响转写）：%s", speaker_error)
    except (InsufficientVramError, AsrBackendError):
        raise
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一成 503 JSON
        raise AsrBackendError(
            f"SenseVoice 识别失败：{type(exc).__name__}: {exc}",
            detail=str(exc),
        ) from exc
    finally:
        for temp_file in (tmp_path, wav_path):
            if temp_file is None:
                continue
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                logger.warning("ASR 临时文件删除失败：%s", temp_file)

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "ASR 完成：%s（%d 字节，转16k=%s）language=%s 耗时=%dms loaded=%s speaker=%s",
        raw_name,
        len(data),
        converted,
        result.get("language"),
        elapsed_ms,
        asr.loaded(),
        (speaker_info or {}).get("name"),
    )
    payload: dict[str, Any] = {
        "ok": True,
        "text": result.get("text") or "",
        "language": result.get("language") or lang,
        "emotion": result.get("emotion"),
        "events": result.get("events") or [],
        "duration": result.get("duration"),
        "elapsed_ms": elapsed_ms,
        "loaded": asr.loaded(),
        "speaker": speaker_info,
        "source_format": suffix,
        "converted": converted,
    }
    if speaker_error:
        payload["speaker_error"] = speaker_error
    return JSONResponse(content=payload)


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
