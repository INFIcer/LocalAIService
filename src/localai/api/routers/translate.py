"""翻译路由：``/translate``、``/translate/batch``、``/translate/status``、``/jobs/translate``。

契约
----
- 同步接口（``/translate``、``/translate/batch``）直接调用 translate 适配器，
  不写任务库；适合单条或中等批量（500 条以内）。
- ``/jobs/translate`` 把超大批量提交到任务队列（``kind=translate``），
  立即返回 job id，结果用 ``GET /jobs/{id}/result`` 取。
- 参数超限（text 5000 字 / 500 条 / 20 万字）统一返回 400 + 中文说明。
- 语言码既支持 NLLB 码（``zho_Hans``），也支持常见别名（``zh`` / ``en``）；
  ``source=auto`` 用 langdetect 自动判断，检测失败会要求显式给 source。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ...core.translate import normalize_engine, normalize_texts
from ...jobs import prepare_payload
from ..deps import get_cfg, get_registry, get_scheduler
from ..job_helpers import client_ip
from ..schemas import JobTranslateRequest, TranslateBatchRequest, TranslateRequest

logger = logging.getLogger("localai.api.translate")

router = APIRouter(tags=["translate"])


def _adapter(request: Request) -> Any:
    return get_registry(request).get("translate")


def _resolve_languages(
    request: Request, source: str, target: str, sample: str
) -> tuple[str, str]:
    """把 source/target 归一成 NLLB 码；``auto`` 时立刻检测，失败给 400。"""
    adapter = _adapter(request)
    try:
        source_code = adapter.normalize_language(source, allow_auto=True)
        target_code = adapter.normalize_language(target)
        if source_code == "auto":
            source_code = adapter.detect_language(sample)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - 语言错误统一按 400 返回
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return source_code, target_code


def _validate(payload_texts: Any, *, single: bool, engine: Any) -> tuple[list[str], str]:
    try:
        texts = normalize_texts(payload_texts, single=single)
        engine_name = normalize_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return texts, engine_name


@router.post("/translate", summary="翻译单条文本（懒加载，长文本自动切分）")
async def translate(payload: TranslateRequest, request: Request) -> dict[str, Any]:
    texts, engine_name = _validate(payload.text, single=True, engine=payload.engine)
    source_code, target_code = _resolve_languages(
        request, payload.source, payload.target, texts[0]
    )
    adapter = _adapter(request)
    outcome = await adapter.infer(
        texts,
        source=source_code,
        target=target_code,
        batch_size=payload.batch_size,
        max_len=payload.max_len,
    )
    docs = outcome.get("docs") or [{}]
    return {
        "ok": True,
        "engine": engine_name,
        "source": outcome.get("source") or source_code,
        "target": outcome.get("target") or target_code,
        "text": docs[0].get("text") or "",
        "elapsed_ms": outcome.get("elapsed_ms"),
        "chunks": int(docs[0].get("chunks") or 0),
    }


@router.post("/translate/batch", summary="批量翻译（内部批量推理，最多 500 条）")
async def translate_batch(
    payload: TranslateBatchRequest, request: Request
) -> dict[str, Any]:
    texts, engine_name = _validate(payload.texts, single=False, engine=payload.engine)
    sample = next((text for text in texts if text.strip()), "")
    source_code, target_code = _resolve_languages(
        request, payload.source, payload.target, sample
    )
    adapter = _adapter(request)
    outcome = await adapter.infer(
        texts,
        source=source_code,
        target=target_code,
        batch_size=payload.batch_size,
        max_len=payload.max_len,
    )
    docs = outcome.get("docs") or []
    return {
        "ok": True,
        "engine": engine_name,
        "source": outcome.get("source") or source_code,
        "target": outcome.get("target") or target_code,
        "texts": [doc.get("text") or "" for doc in docs],
        "count": len(docs),
        "elapsed_ms": outcome.get("elapsed_ms"),
        "chunks": outcome.get("chunks") or 0,
    }


@router.get("/translate/status", summary="翻译引擎状态（不加载模型）")
async def translate_status(request: Request) -> dict[str, Any]:
    adapter = _adapter(request)
    status = adapter.status()
    idle = adapter.idle_seconds()
    return {
        "ok": True,
        "ready": bool(status.get("ready")),
        "loaded": bool(status.get("loaded")),
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": adapter.idle_unload_seconds(),
        "model": status.get("model"),
        "model_dir": status.get("model_dir"),
        "device": status.get("device"),
        "dtype": status.get("dtype"),
        "fp16": status.get("fp16"),
        "size_mb": status.get("size_mb"),
        "weights_source": status.get("weights_source"),
        "missing": status.get("missing") or [],
        "load_seconds": status.get("load_seconds"),
        "load_error": status.get("load_error"),
        "install_log": status.get("install_log"),
        "languages": adapter.languages(),
        "detail": status.get("detail"),
    }


@router.post("/jobs/translate", summary="提交翻译任务（超大批量），立刻返回任务 id")
async def create_translate_job(
    payload: JobTranslateRequest, request: Request
) -> dict[str, Any]:
    texts, engine_name = _validate(payload.texts, single=False, engine=payload.engine)
    sample = next((text for text in texts if text.strip()), "")
    source_code, target_code = _resolve_languages(
        request, payload.source, payload.target, sample
    )
    cfg = get_cfg(request)
    scheduler = get_scheduler(request)
    try:
        action, normalized = prepare_payload(
            "translate",
            "translate.text",
            {
                "texts": texts,
                "source": source_code,
                "target": target_code,
                "engine": engine_name,
                "max_len": payload.max_len,
                "batch_size": payload.batch_size,
            },
            cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = await scheduler.submit(
        "translate",
        action,
        normalized,
        priority=payload.priority,
        client_ip=client_ip(request),
    )
    return {
        "ok": True,
        "id": job_id,
        "state": "queued",
        "count": len(texts),
        "source": source_code,
        "target": target_code,
    }
