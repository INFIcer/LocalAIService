"""ASR 路由：``/asr``、``/api/v1/asr``、``/asr/status``、``/asr/segments``。

契约：multipart ``file=<音频/视频>``、``language=auto|zh|en|ja|ko|yue``、
``engine=sensevoice|whisper|auto`` 选择识别引擎；声纹不在这里，见 ``/speakers/embed``。
非 WAV 一律先由 ffmpeg 归一到 16k 单声道 WAV。

``/asr/segments`` 是**新增**的带时间戳分段接口：
- whisper 用原生 segment 时间戳（``timestamp_source="native"``）；
- sensevoice 没有原生时间戳，用 ``fsmn-vad`` 切句（``timestamp_source="vad"``）。
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
from ...core.asr_engines import (
    EngineChoiceError,
    adapter_for,
    normalize_engine,
)
from ...core.segments import normalize_segments, segments_duration
from ...jobs import prepare_payload
from ..deps import get_cfg, get_registry
from ..job_helpers import cleanup_job_upload, submit_and_wait
from ..uploads import save_job_upload, save_media_upload, unlink_quiet

logger = logging.getLogger("localai.api.asr")

router = APIRouter(tags=["asr"])

def _true_flag(value: Any) -> bool:
    """把 ``timestamps=1`` 这类开关转成 bool（默认关闭）。"""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on", "y")


def _validate_language(language: str) -> str:
    lang = (language or "auto").strip().lower() or "auto"
    if lang not in ASR_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 language={language!r}，可选：{', '.join(ASR_LANGUAGES)}",
        )
    return lang


def _validate_suffix(raw_name: str) -> str:
    suffix = Path(raw_name).suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不支持的文件类型 {suffix or '(无扩展名)'}；"
                f"支持：{', '.join(sorted(MEDIA_EXTENSIONS))}"
            ),
        )
    return suffix


@router.get("/asr/status", summary="ASR 引擎状态（含 whisper 配置与加载状态）")
async def asr_status_endpoint(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    registry = get_registry(request)
    adapter = registry.get("asr")
    status = adapter.status()
    idle = adapter.idle_seconds()

    whisper = registry.get("whisper")
    whisper_status = whisper.status()
    whisper_idle = whisper.idle_seconds()

    engines = {
        "sensevoice": {
            "adapter": "asr",
            "model": "SenseVoiceSmall",
            "ready": bool(status["ready"]),
            "loaded": bool(status["loaded"]),
            "load_seconds": status.get("load_seconds"),
            "model_dir": status.get("model_dir"),
            "device": status.get("device"),
            "idle_unload_seconds": cfg.asr_idle_unload,
            "detail": status["detail"],
            "vad": status.get("vad"),
        },
        "whisper": {
            "adapter": "whisper",
            "model": whisper_status.get("model") or f"faster-whisper-{cfg.whisper_model}",
            "ready": bool(whisper_status.get("ready")),
            "loaded": bool(whisper_status.get("loaded")),
            "load_seconds": whisper_status.get("load_seconds"),
            "load_error": whisper_status.get("load_error"),
            "model_dir": whisper_status.get("model_dir"),
            "device": whisper_status.get("device") or cfg.whisper_device,
            "compute_type": whisper_status.get("compute_type") or cfg.whisper_compute,
            "size": cfg.whisper_model,
            "size_mb": whisper_status.get("size_mb"),
            "idle_unload_seconds": cfg.whisper_idle_unload,
            "detail": whisper_status["detail"],
        },
    }
    return {
        "ready": bool(status["ready"]),
        "loaded": bool(status["loaded"]),
        "engine": cfg.asr_engine,
        "engines": engines,
        "whisper": {**engines["whisper"], "idle_seconds": None if whisper_idle is None else round(whisper_idle, 1)},
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": cfg.asr_idle_unload,
        "parallel": status.get("parallel"),
        "detail": status["detail"],
    }


async def _handle_asr_request(
    request: Request,
    file: UploadFile,
    language: str,
    engine: str = "",
    timestamps: bool = False,
) -> JSONResponse:
    """同步入口：校验 -> 落盘 -> 提交队列 -> 等完成 -> 原样返回结果。

    ``engine`` 为空表示用配置默认；``auto`` 在需要时间戳（``timestamps=1``）时
    选 whisper。真正要时间戳分段请用 ``POST /asr/segments``。
    """
    cfg = get_cfg(request)
    raw_name = Path(file.filename or "audio.wav").name
    _validate_suffix(raw_name)
    lang = _validate_language(language)
    try:
        # 只是提前校验，非法值不落到磁盘上；真正的解析在执行器里（同一套规则）。
        normalize_engine(engine, cfg, need_timestamps=timestamps)
    except EngineChoiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
                "engine": engine,
                "timestamps": "1" if timestamps else "0",
                "original_name": raw_name,
            },
            cfg,
        )
    except ValueError as exc:
        cleanup_job_upload(cfg, job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = await submit_and_wait(request, "asr", action, payload, job_id=job_id)
    return JSONResponse(content=job.get("result") or {})


@router.post("/asr", summary="语音识别（可选引擎）")
async def asr(
    request: Request,
    file: UploadFile = File(..., description="要识别的音频文件"),
    language: str = Form("auto", description="auto/zh/en/ja/ko/yue..."),
    engine: str = Form("", description="sensevoice / whisper / auto；空=按配置默认"),
    timestamps: str = Form("0", description="1=需要时间戳（engine=auto 时选 whisper）"),
) -> JSONResponse:
    return await _handle_asr_request(
        request,
        file,
        language,
        engine=engine,
        timestamps=_true_flag(timestamps),
    )


@router.post("/api/v1/asr", summary="`/asr` 的别名")
async def api_v1_asr(
    request: Request,
    file: UploadFile = File(..., description="要识别的音频文件"),
    language: str = Form("auto", description="auto/zh/en/ja/ko/yue..."),
    engine: str = Form("", description="sensevoice / whisper / auto；空=按配置默认"),
    timestamps: str = Form("0", description="1=需要时间戳（engine=auto 时选 whisper）"),
) -> JSONResponse:
    return await _handle_asr_request(
        request,
        file,
        language,
        engine=engine,
        timestamps=_true_flag(timestamps),
    )


@router.post("/asr/segments", summary="带时间戳的分段转写")
async def asr_segments(
    request: Request,
    file: UploadFile = File(..., description="音频或视频文件（自动转 16k wav）"),
    language: str = Form("auto", description="auto/zh/en/ja/ko/yue..."),
    engine: str = Form("", description="sensevoice / whisper / auto；空=按配置默认"),
) -> dict[str, Any]:
    """返回 ``{ok, engine, language, duration, segments[], text, timestamp_source}``。

    - 分段保证升序、无重叠、``end > start``（由 :func:`normalize_segments` 整理）；
    - ``engine=auto`` 时固定选 whisper（本接口就是要时间戳），这样 SenseVoice
      也能通过 VAD 给出时间戳，而 whisper 给的是原生时间戳。
    """
    cfg = get_cfg(request)
    registry = get_registry(request)
    lang = _validate_language(language)
    try:
        chosen = normalize_engine(engine, cfg, need_timestamps=True)
    except EngineChoiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_name = Path(file.filename or "audio.wav").name
    suffix = _validate_suffix(raw_name)

    media = registry.get("media")
    tmp_path = await save_media_upload(cfg, file, "segments")
    wav_path: Path | None = None
    converted = False
    try:
        if suffix != ".wav":
            try:
                conversion = await media.call("to_wav16k", str(tmp_path))
                wav_path = Path(conversion["path"])
                converted = True
            except HTTPException:
                from ...constants import MEDIA_VIDEO_EXTENSIONS

                if suffix in MEDIA_VIDEO_EXTENSIONS:
                    raise
                logger.warning("ffmpeg 转换失败，回退直接识别原始文件：%s", raw_name)
        input_path = wav_path or tmp_path
        adapter = adapter_for(registry, chosen)
        result = await adapter.infer_segments(str(input_path), lang)
    finally:
        unlink_quiet(tmp_path)
        unlink_quiet(wav_path)

    duration = result.get("duration")
    segments = normalize_segments(result.get("segments"), duration=duration)
    if duration is None:
        duration = segments_duration(segments)
    timestamp_source = result.get("timestamp_source") or (
        "native" if chosen == "whisper" else "vad"
    )
    logger.info(
        "分段转写：%s 引擎=%s 时长=%s 段数=%d 来源=%s",
        raw_name, chosen, duration, len(segments), timestamp_source,
    )
    return {
        "ok": True,
        "engine": result.get("engine") or chosen,
        "language": result.get("language") or lang,
        "duration": duration,
        "segments": segments,
        "text": result.get("text") or "",
        "timestamp_source": timestamp_source,
        "elapsed": result.get("elapsed"),
        "device": result.get("device"),
        "converted": converted,
        "source_format": suffix,
    }
