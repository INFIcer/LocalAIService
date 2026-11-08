"""任务执行器：把「一个 job 的 payload」变成「一次能力调用 + 可序列化结果」。

调度器只负责顺序、状态与超时；真正的推理逻辑放在这里，路由（同步/异步）
共用同一份实现，保证「旧接口走队列后语义不变」。

执行器契约
----------
``execute_job(job, ctx) -> dict``

- ``job`` 是 :class:`localai.jobs.store.JobStore` 返回的字典（含 payload）；
- ``ctx`` 是 :class:`JobContext`（配置 / 注册表 / httpx 客户端）；
- 返回值必须是 JSON 可序列化的 dict，会原样写进 ``jobs.result``；
- 失败直接抛异常：调度器会记 ``error``，同步接口会把原异常重新抛给调用方，
  从而复用已有的 4xx/5xx 处理器。
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import soundfile as sf
from fastapi import HTTPException

from ..config import Config
from ..constants import (
    ASR_ENGINE_CHOICES,
    ASR_LANGUAGES,
    MEDIA_EXTENSIONS,
    SEPARATE_MODELS,
    SEPARATE_STEMS,
    TTS_MAX_TEXT,
)
from ..core.asr_engines import adapter_for, normalize_engine
from ..core.errors import (
    AsrBackendError,
    InsufficientVramError,
    SeparateBackendError,
    TranslateBackendError,
)
from ..core.translate import normalize_engine as normalize_translate_engine
from ..core.translate import normalize_texts as normalize_translate_texts

logger = logging.getLogger("localai.jobs.executors")

# ---------------------------------------------------------------------------
# kind / action 白名单
# ---------------------------------------------------------------------------

#: 每种 kind 默认的 action（POST /jobs 不写 action 时使用）。
DEFAULT_ACTION = {
    "tts": "tts.synthesize",
    "asr": "asr.transcribe",
    "speaker": "speaker.identify",
    "separate": "separate.split",
    "media": "media.probe",
    "translate": "translate.text",
    "download": "download.fetch",
    "analyze": "analyze.url",
}

#: 每种 kind 允许的 action。
KIND_ACTIONS = {
    "tts": {"tts.synthesize"},
    "asr": {"asr.transcribe"},
    "speaker": {"speaker.enroll", "speaker.identify", "speaker_for"},
    "separate": {"separate.split"},
    "media": {"media.probe", "media.to_wav", "media.frames", "media.audio"},
    "translate": {"translate.text"},
    "download": {"download.fetch"},
    "analyze": {"analyze.url"},
}

_SPEAKER_OFF_VALUES = {"0", "false", "off", "no", "none", ""}


def speaker_enabled(value: Any) -> bool:
    """``?speaker=`` / payload.speaker 转 bool（默认开启）。"""
    if isinstance(value, bool):
        return value
    return str(value if value is not None else "").strip().lower() not in _SPEAKER_OFF_VALUES


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def _flag(value: Any) -> bool:
    """把 ``timestamps`` 这类开关值转成 bool（默认关闭）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in _TRUE_VALUES


@dataclass
class JobContext:
    """执行任务需要的运行时依赖（避免执行器 import FastAPI 的 request）。"""

    cfg: Config
    registry: Any
    client: Any = None
    #: 进度上报（下载 / 分析任务用）；由调度器为每个任务注入，写进 jobs.meta。
    progress: Optional[Callable[[dict[str, Any]], None]] = None
    #: 取消查询；返回 True 时执行器应在下一个检查点退出。
    is_canceled: Optional[Callable[[], bool]] = None

    def report(self, payload: dict[str, Any]) -> None:
        if self.progress is None:
            return
        try:
            self.progress(payload)
        except Exception:  # noqa: BLE001 - 进度失败不能影响任务
            logger.debug("任务进度上报失败", exc_info=True)

    def canceled(self) -> bool:
        try:
            return bool(self.is_canceled and self.is_canceled())
        except Exception:  # noqa: BLE001
            return False


# ---------------------------------------------------------------------------
# payload 校验 / 归一化
# ---------------------------------------------------------------------------

def _text(value: Any) -> str:
    return str(value or "").strip()


def _require_file(payload: dict[str, Any], cfg: Config, *, allowed=MEDIA_EXTENSIONS) -> Path:
    raw = _text(payload.get("file") or payload.get("path") or payload.get("input"))
    if not raw:
        raise ValueError("payload.file 不能为空（服务端本地路径）")
    path = Path(raw)
    if not path.is_file():
        raise ValueError(f"payload.file 指向的文件不存在：{path}")
    suffix = path.suffix.lower()
    if allowed and suffix not in allowed:
        raise ValueError(
            f"不支持的文件类型 {suffix or '(无扩展名)'}；支持：{', '.join(sorted(allowed))}"
        )
    return path


def prepare_payload(
    kind: str,
    action: Optional[str],
    payload: Optional[dict[str, Any]],
    cfg: Config,
) -> tuple[str, dict[str, Any]]:
    """校验并补全 payload；返回 ``(action, 归一化后的 payload)``。

    校验失败抛 :class:`ValueError`，路由转成 HTTP 400 —— 这类错误不会入队，
    与旧接口「参数错误不加载模型」的行为一致。
    """
    kind = _text(kind).lower()
    if kind not in KIND_ACTIONS:
        raise ValueError(f"不支持的 kind={kind!r}；可选：{', '.join(KIND_ACTIONS)}")
    act = _text(action) or DEFAULT_ACTION[kind]
    if act not in KIND_ACTIONS[kind]:
        raise ValueError(
            f"kind={kind} 不支持 action={act!r}；可选：{', '.join(sorted(KIND_ACTIONS[kind]))}"
        )
    data = dict(payload or {})

    if kind == "tts":
        text = _text(data.get("text"))
        if not text:
            raise ValueError("TTS 任务缺少 text")
        if len(text) > TTS_MAX_TEXT:
            raise ValueError(f"text 太长（上限 {TTS_MAX_TEXT} 字）")
        speed = data.get("speed", 1.0)
        try:
            speed = float(speed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"speed 必须是数字：{speed!r}") from exc
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed 必须在 0.5~2.0 之间")
        seed = data.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"seed 必须是整数：{seed!r}") from exc
            if not 0 <= seed <= 4294967295:
                raise ValueError("seed 必须在 0~4294967295 之间")
        return act, {
            "text": text,
            "voice": _text(data.get("voice")) or "default",
            "speed": speed,
            "emotion": data.get("emotion") or None,
            "seed": seed,
            "language": _text(data.get("language")) or None,
        }

    if kind == "asr":
        file_path = _require_file(data, cfg)
        language = (_text(data.get("language")) or "auto").lower()
        if language not in ASR_LANGUAGES:
            raise ValueError(
                f"不支持的 language={language!r}，可选：{', '.join(ASR_LANGUAGES)}"
            )
        raw_engine = (_text(data.get("engine")) or "").lower()
        if raw_engine in ("", "default"):
            engine = ""
        elif raw_engine in ASR_ENGINE_CHOICES:
            engine = raw_engine
        else:
            raise ValueError(
                f"不支持的 engine={raw_engine!r}，可选：{', '.join(ASR_ENGINE_CHOICES)}"
            )
        return act, {
            "file": str(file_path),
            "language": language,
            "speaker": speaker_enabled(data.get("speaker", True)),
            # 空字符串 = 用配置默认（LOCALAI_ASR_ENGINE）；auto 由执行器按需解析。
            "engine": engine,
            "timestamps": _flag(data.get("timestamps", False)),
            "source_format": file_path.suffix.lower(),
            "original_name": _text(data.get("original_name")) or file_path.name,
        }

    if kind == "translate":
        if data.get("texts") is not None:
            texts = normalize_translate_texts(data.get("texts"))
        elif data.get("text") is not None:
            texts = normalize_translate_texts(data.get("text"), single=True)
        else:
            raise ValueError("翻译任务缺少 texts（或 text）")
        engine = normalize_translate_engine(data.get("engine"))
        source = _text(data.get("source")) or "auto"
        target = _text(data.get("target")) or "en"
        try:
            max_len = max(32, min(1024, int(data.get("max_len") or cfg.translate_max_len)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_len 必须是整数：{data.get('max_len')!r}") from exc
        try:
            batch_size = max(
                1,
                min(64, int(data.get("batch_size") or cfg.translate_batch_size)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"batch_size 必须是整数：{data.get('batch_size')!r}") from exc
        return act, {
            "texts": texts,
            "source": source,
            "target": target,
            "engine": engine,
            "max_len": max_len,
            "batch_size": batch_size,
        }

    if kind == "separate":
        file_path = _require_file(data, cfg)
        model = (_text(data.get("model")) or "htdemucs").lower()
        if model not in SEPARATE_MODELS:
            raise ValueError(
                f"不支持的 model={model!r}，可选：{', '.join(SEPARATE_MODELS)}"
            )
        raw_stems = data.get("stems")
        if isinstance(raw_stems, str):
            raw_stems = raw_stems.split(",")
        if not raw_stems:
            raw_stems = ["vocals", "drums", "bass", "other"]
        stems: list[str] = []
        for part in raw_stems:
            name = _text(part).lower()
            if name and name not in stems:
                stems.append(name)
        if not stems:
            raise ValueError("stems 不能为空")
        unknown = [name for name in stems if name not in SEPARATE_STEMS[model]]
        if unknown:
            raise ValueError(
                f"模型 {model} 没有这些 stem：{', '.join(unknown)}；"
                f"可用：{', '.join(SEPARATE_STEMS[model])}"
            )
        return act, {
            "file": str(file_path),
            "model": model,
            "stems": stems,
            "original_name": _text(data.get("original_name")) or file_path.name,
        }

    if kind == "speaker":
        file_path = _require_file(data, cfg)
        normalized: dict[str, Any] = {
            "file": str(file_path),
            "original_name": _text(data.get("original_name")) or file_path.name,
        }
        if act == "speaker.enroll":
            name = _text(data.get("name"))
            if not name:
                raise ValueError("声纹登记缺少 name")
            normalized["name"] = name
        return act, normalized

    if kind in ("download", "analyze"):
        from ..engines._native import import_engine_module

        engine_module = import_engine_module("download_engine")
        url = _text(data.get("url") or data.get("link"))
        if not url:
            raise ValueError("payload.url 不能为空（要下载的 http/https 链接）")
        try:
            # 这里只做 scheme / 主机名预校验，不做 DNS：参数错误不入队。
            engine_module.validate_url(url, resolve=False)
        except engine_module.DownloadError as exc:
            raise ValueError(str(exc)) from exc
        quality = (_text(data.get("quality")) or "720p").lower()
        if quality not in engine_module.QUALITY_CHOICES:
            raise ValueError(
                f"不支持的 quality={quality!r}；可选：{', '.join(engine_module.QUALITY_CHOICES)}"
            )
        cookies = _text(data.get("cookies")) or None
        normalized = {
            "url": url,
            "quality": quality,
            "subtitles": _flag(data.get("subtitles", True)),
            "cookies": cookies,
        }
        if kind == "download":
            return act, normalized

        raw_engine = (_text(data.get("engine")) or "auto").lower()
        if raw_engine in ("", "default"):
            raw_engine = "auto"
        if raw_engine not in ("auto", *ASR_ENGINE_CHOICES):
            raise ValueError(
                f"不支持的 engine={raw_engine!r}，可选：auto, {', '.join(ASR_ENGINE_CHOICES)}"
            )
        try:
            max_frames = int(data.get("max_frames") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_frames 必须是整数：{data.get('max_frames')!r}") from exc
        if not 0 <= max_frames <= 200:
            raise ValueError("max_frames 必须在 0~200 之间（0=不抽帧）")
        normalized.update({"engine": raw_engine, "max_frames": max_frames})
        return act, normalized

    # media
    file_path = _require_file(data, cfg)
    normalized = {
        "file": str(file_path),
        "original_name": _text(data.get("original_name")) or file_path.name,
    }
    if act == "media.frames":
        count = data.get("count")
        fps = data.get("fps")
        times = data.get("times")
        given = [value is not None for value in (count, fps, times)]
        if sum(given) > 1:
            raise ValueError("count / fps / times 只能给一个")
        normalized.update(
            {
                "count": int(count) if count is not None else None,
                "fps": float(fps) if fps is not None else None,
                "times": times,
                "width": int(data["width"]) if data.get("width") is not None else None,
            }
        )
    elif act == "media.audio":
        audio_format = (_text(data.get("format")) or "wav").lower()
        if audio_format not in ("wav", "mp3"):
            raise ValueError("format 只支持 wav 或 mp3")
        normalized["format"] = audio_format
    return act, normalized


# ---------------------------------------------------------------------------
# 各能力执行
# ---------------------------------------------------------------------------

def _wav_duration(path: Path) -> Optional[float]:
    try:
        return round(float(sf.info(str(path)).duration), 3)
    except Exception:  # noqa: BLE001 - 时长只是附加信息
        return None


async def _run_tts(payload: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..engines.tts_indextts import TtsResult  # 仅用于类型说明

    adapter = ctx.registry.get("tts")
    result: TtsResult = await adapter.infer(
        client=ctx.client,
        text=payload["text"],
        voice_name=payload["voice"],
        speed=payload["speed"],
        emotion=payload["emotion"],
        seed=payload["seed"],
        language=payload["language"],
    )
    path = Path(result.path)
    return {
        "path": str(path),
        "output_path": str(path),
        "seed": result.seed,
        "language": result.language,
        "comfy_filename": result.comfy_filename or "",
        "bytes": len(result.wav),
        "duration_seconds": _wav_duration(path),
        "model": adapter.model_name,
        "voice": payload["voice"],
        "text_chars": len(payload["text"]),
    }


async def _run_asr(payload: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """与旧 ``/asr`` 相同的处理链路：转 16k -> 识别 ->（可选）声纹。

    ``payload.engine`` 为空 = 用配置默认（``LOCALAI_ASR_ENGINE``）；``auto`` 时
    需要时间戳就选 whisper，否则用配置默认。响应固定带 ``engine`` 字段。
    """
    cfg = ctx.cfg
    registry = ctx.registry
    tmp_path = Path(payload["file"])
    suffix = payload["source_format"]
    lang = payload["language"]
    use_speaker = bool(payload["speaker"])
    engine = normalize_engine(
        payload.get("engine"),
        cfg,
        need_timestamps=bool(payload.get("timestamps")),
    )
    wav_path: Optional[Path] = None
    converted = False
    started = time.perf_counter()
    speaker_info: Optional[dict[str, Any]] = None
    speaker_error: Optional[str] = None
    media = registry.get("media")
    asr = adapter_for(registry, engine)
    try:
        # 非 wav 一律先归一成 16k 单声道 wav；音频失败回退原文件，视频不回退。
        if suffix != ".wav":
            try:
                conversion = await media.call("to_wav16k", str(tmp_path))
                wav_path = Path(conversion["path"])
                converted = True
            except HTTPException:
                from ..constants import MEDIA_VIDEO_EXTENSIONS

                if suffix in MEDIA_VIDEO_EXTENSIONS:
                    raise
                logger.warning("ffmpeg 转换失败，回退直接识别原始文件：%s", payload["original_name"])
        input_path = wav_path or tmp_path
        result = await asr.infer(str(input_path), lang)

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
    except Exception as exc:  # noqa: BLE001 - 统一成 503
        raise AsrBackendError(
            f"{engine} 识别失败：{type(exc).__name__}: {exc}",
            detail=str(exc),
        ) from exc
    finally:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("ASR 临时文件删除失败：%s", wav_path)

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    data: dict[str, Any] = {
        "ok": True,
        "engine": engine,
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
        data["speaker_error"] = speaker_error
    return data


async def _run_separate(payload: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """与旧 ``/separate`` 完全相同的处理链路：转 44.1k -> Demucs -> 落盘。"""
    from datetime import datetime

    cfg = ctx.cfg
    registry = ctx.registry
    started = time.perf_counter()
    tmp_path = Path(payload["file"])
    converted_path = cfg.separate_tmp_dir / f"separate_{uuid.uuid4().hex}.wav"
    out_dir = cfg.separate_dir / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    media = registry.get("media")
    try:
        converted = await media.call("to_wav44k", str(tmp_path), str(converted_path))
        outcome = await registry.get("separate").infer(
            input_wav=str(converted_path),
            out_dir=str(out_dir),
            model_name=payload["model"],
            stems=payload["stems"],
        )
        result = outcome["result"]
        released_models = outcome["released_models"]
    except (InsufficientVramError, SeparateBackendError):
        raise
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一成 503
        raise SeparateBackendError(
            f"Demucs 分离失败：{type(exc).__name__}: {exc}",
            detail=str(exc),
        ) from exc
    finally:
        try:
            converted_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("人声分离临时文件删除失败：%s", converted_path)

    elapsed = round(time.perf_counter() - started, 2)
    return {
        "ok": True,
        "model": result.get("model", payload["model"]),
        "sample_rate": result.get("sample_rate", 44100),
        "duration": result.get("duration", converted.get("duration")),
        "elapsed": elapsed,
        "separation_elapsed": result.get("elapsed"),
        "stems": result.get("stems") or {},
        "dir": result.get("dir") or str(out_dir),
        "loaded": registry.get("separate").loaded(),
        "released_models": released_models,
    }


async def _run_translate(payload: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """翻译任务：转发给 translate 适配器（懒加载 / 显存 LRU / 空闲卸载都由它管）。"""
    adapter = ctx.registry.get("translate")
    try:
        outcome = await adapter.infer(
            payload["texts"],
            source=payload.get("source") or "auto",
            target=payload.get("target") or "en",
            batch_size=payload.get("batch_size"),
            max_len=int(payload.get("max_len") or ctx.cfg.translate_max_len),
        )
    except (InsufficientVramError, TranslateBackendError):
        raise
    except Exception as exc:  # noqa: BLE001 - 统一成 503
        raise TranslateBackendError(
            f"NLLB 翻译失败：{type(exc).__name__}: {exc}",
            detail=str(exc),
        ) from exc

    docs = outcome.get("docs") or []
    return {
        "ok": True,
        "engine": outcome.get("engine") or "nllb",
        "model": outcome.get("model"),
        "source": outcome.get("source"),
        "target": outcome.get("target"),
        "texts": [doc.get("text") or "" for doc in docs],
        "count": len(docs),
        "chunks": outcome.get("chunks") or 0,
        "elapsed_ms": outcome.get("elapsed_ms"),
        "loaded": bool(outcome.get("loaded")),
    }


async def _run_speaker(
    action: str, payload: dict[str, Any], ctx: JobContext
) -> dict[str, Any]:
    adapter = ctx.registry.get("speaker")
    if action == "speaker.enroll":
        result = await adapter.call("enroll", payload["name"], payload["file"])
        return {
            "ok": True,
            "name": result.get("name"),
            "samples": result.get("samples"),
            "created_at": result.get("created_at"),
        }
    if action == "speaker_for":
        result = await adapter.identify_for(payload["file"])
        return {"ok": True, **result}
    result = await adapter.call("identify", payload["file"])
    matches = [
        {"name": item.get("name"), "score": float(item.get("score") or 0.0)}
        for item in (result.get("matches") or [])
    ]
    return {
        "ok": True,
        "matches": matches,
        "best": result.get("best"),
        "score": float(result.get("score") or 0.0),
        "threshold": float(result.get("threshold") or 0.0),
        "known": bool(result.get("known")),
    }


async def _run_media(
    action: str, payload: dict[str, Any], ctx: JobContext
) -> dict[str, Any]:
    media = ctx.registry.get("media")
    path = payload["file"]
    if action == "media.probe":
        info = await media.call("probe", path)
        return {"ok": True, **info}
    if action == "media.to_wav":
        converted = await media.call("to_wav16k", path)
        return {"ok": True, **converted}
    if action == "media.frames":
        from datetime import datetime

        cfg = ctx.cfg
        points: Optional[list[float]] = None
        if payload.get("times") is not None:
            points = [float(part) for part in str(payload["times"]).split(",") if part.strip()]
        out_dir = cfg.frames_dir / (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        frames = await media.call(
            "extract_frames",
            path,
            str(out_dir),
            count=payload.get("count"),
            times=points,
            fps=payload.get("fps"),
            width=payload.get("width"),
        )
        return {
            "ok": True,
            "count": len(frames),
            "dir": str(out_dir),
            "frames": frames,
        }
    out_dir = ctx.cfg.media_tmp_dir / f"extract_{uuid.uuid4().hex}"
    result = await media.call("extract_audio", path, str(out_dir), fmt=payload["format"])
    return {"ok": True, **result}


# ---------------------------------------------------------------------------
# 下载 / 链接分析
# ---------------------------------------------------------------------------


def _download_out_dir(cfg: Config) -> Path:
    from datetime import datetime

    return cfg.downloads_dir / datetime.now().strftime("%Y-%m-%d")


def _frame_times(scenes: list[dict[str, Any]], duration: Optional[float], count: int) -> list[float]:
    """均匀取点 + 场景吸附：每个点靠近镜头边界（2 秒内）时挪到边界之后。

    这样抽出来的帧既能覆盖整段视频，又尽量落在「新镜头刚开始」的位置。
    """
    if count <= 0 or not duration or duration <= 0:
        return []
    step = float(duration) / count
    points: list[float] = []
    boundary = [
        float(item.get("time") or 0.0)
        for item in scenes
        if item.get("time") is not None
    ]
    for index in range(count):
        point = step * (index + 0.5)
        nearest = min(boundary, key=lambda value: abs(value - point)) if boundary else None
        if nearest is not None and abs(nearest - point) <= min(2.0, step / 2):
            point = nearest + 0.05
        points.append(round(min(max(0.0, point), max(0.0, duration - 0.05)), 3))
    # 去重后仍保持升序；重复点只留一个。
    return sorted(set(points))


async def _run_download(payload: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """下载任务：交给 download 适配器，进度实时写进任务的 ``meta.progress``。"""
    adapter = ctx.registry.get("download")
    started = time.perf_counter()
    result = await adapter.call(
        "download_any",
        payload["url"],
        str(_download_out_dir(ctx.cfg)),
        quality=payload["quality"],
        subtitles=bool(payload["subtitles"]),
        cookies=payload.get("cookies"),
        progress_cb=ctx.report,
        timeout=ctx.cfg.download_timeout,
        max_mb=ctx.cfg.download_max_mb,
        should_cancel=ctx.is_canceled,
    )
    result["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return result


async def _transcribe_segments(
    path: Path, engine: str, ctx: JobContext
) -> dict[str, Any]:
    """把下载到的媒体转 16k wav，再走带时间戳的分段识别。"""
    from ..core.segments import normalize_segments
    from ..core.asr_engines import normalize_engine

    chosen = normalize_engine(engine, ctx.cfg, need_timestamps=True)
    media = ctx.registry.get("media")
    wav_path: Optional[Path] = None
    try:
        if path.suffix.lower() != ".wav":
            converted = await media.call("to_wav16k", str(path))
            wav_path = Path(converted["path"])
        adapter = adapter_for(ctx.registry, chosen)
        result = await adapter.infer_segments(str(wav_path or path), "auto")
    finally:
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("分析任务临时 wav 删除失败：%s", wav_path)
    duration = result.get("duration")
    segments = normalize_segments(result.get("segments"), duration=duration)
    return {
        "engine": result.get("engine") or chosen,
        "language": result.get("language") or "auto",
        "duration": duration,
        "segments": segments,
        "text": result.get("text") or "",
        "timestamp_source": result.get("timestamp_source")
        or ("native" if chosen == "whisper" else "vad"),
    }


def _site_segments(paths: list[str]) -> tuple[Optional[Path], list[dict[str, Any]]]:
    """从站点字幕里挑一个并解析；没有可用字幕时返回 ``(None, [])``。"""
    from ..core.subtitles import parse_subtitle_file, pick_subtitle

    chosen = pick_subtitle(paths)
    if chosen is None:
        return None, []
    return chosen, parse_subtitle_file(chosen)


async def _run_analyze(payload: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """链接分析编排：下载 -> （站点字幕 | ASR）-> 场景边界 -> 可选抽帧。"""
    from datetime import datetime

    started = time.perf_counter()
    cfg = ctx.cfg
    media = ctx.registry.get("media")

    ctx.report({"stage": "download", "percent": 0, "status": "starting"})
    downloaded = await _run_download(payload, ctx)
    path = Path(downloaded["path"])
    duration = downloaded.get("duration")
    ctx.report({"stage": "probe", "percent": 5, "status": "running"})

    subtitles_from = "asr"
    subtitle_path: Optional[Path] = None
    segments: list[dict[str, Any]] = []
    if downloaded.get("subtitles"):
        subtitle_path, segments = _site_segments(list(downloaded["subtitles"]))
        if segments:
            subtitles_from = "site"
    asr_info: dict[str, Any] = {}
    if not segments:
        ctx.report({"stage": "asr", "percent": 15, "status": "running"})
        asr_info = await _transcribe_segments(path, payload.get("engine") or "auto", ctx)
        segments = asr_info["segments"]
        if duration is None:
            duration = asr_info.get("duration")

    ctx.report({"stage": "scenes", "percent": 70, "status": "running"})
    scene_result = await media.call(
        "detect_scenes",
        str(path),
        save_dir=str(cfg.scenes_dir),
    )
    scenes = scene_result.get("scenes") or []
    if duration is None:
        duration = scene_result.get("duration")

    frames: list[dict[str, Any]] = []
    max_frames = int(payload.get("max_frames") or 0)
    if max_frames > 0:
        ctx.report({"stage": "frames", "percent": 85, "status": "running"})
        times = _frame_times(scenes, duration, max_frames)
        if times:
            out_dir = cfg.frames_dir / (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            )
            frames = await media.call(
                "extract_frames", str(path), str(out_dir), times=times
            )

    elapsed = round(time.perf_counter() - started, 1)
    result = {
        "ok": True,
        "title": downloaded.get("title") or path.stem,
        "duration": None if duration is None else round(float(duration), 3),
        "path": str(path),
        "site": downloaded.get("site"),
        "subtitles_from": subtitles_from,
        "subtitle_path": str(subtitle_path) if subtitle_path else None,
        "segments": segments,
        "scenes": scenes,
        "frames": frames,
        "scenes_json": scene_result.get("json_path"),
        "asr": {
            "engine": asr_info.get("engine"),
            "language": asr_info.get("language"),
            "timestamp_source": asr_info.get("timestamp_source"),
        }
        if asr_info
        else None,
        "download": {
            "size_mb": downloaded.get("size_mb"),
            "quality": downloaded.get("quality"),
            "elapsed_seconds": downloaded.get("elapsed_seconds"),
        },
        "elapsed_seconds": elapsed,
    }
    ctx.report({"stage": "done", "percent": 100, "status": "done"})
    logger.info(
        "链接分析完成：%s 字幕来源=%s 段数=%d 场景=%d 帧=%d 耗时=%.1fs",
        path,
        subtitles_from,
        len(segments),
        len(scenes),
        len(frames),
        elapsed,
    )
    return result


async def execute_job(job: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """按 ``kind/action`` 分派到具体能力；返回 JSON 可序列化结果。"""
    kind = job["kind"]
    action = job["action"]
    payload = job.get("payload") or {}
    if kind == "tts":
        return await _run_tts(payload, ctx)
    if kind == "asr":
        return await _run_asr(payload, ctx)
    if kind == "separate":
        return await _run_separate(payload, ctx)
    if kind == "translate":
        return await _run_translate(payload, ctx)
    if kind == "speaker":
        return await _run_speaker(action, payload, ctx)
    if kind == "media":
        return await _run_media(action, payload, ctx)
    if kind == "download":
        return await _run_download(payload, ctx)
    if kind == "analyze":
        return await _run_analyze(payload, ctx)
    raise ValueError(f"未知任务类型：{kind}")
