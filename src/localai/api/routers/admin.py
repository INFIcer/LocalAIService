"""管理接口：``/admin/*``。

安全约定
--------
- 全部接口都走全局 ``LOCALAI_TOKEN`` 中间件；``/admin/*`` 不例外；
- 写操作必须显式带请求体，重启额外要求 ``confirm: true``；
- 令牌永远不回显；模型下载同一时刻只允许一个任务。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...admin.discovery import (
    DiscoveryError,
    ModelScanner,
    ScanBusy,
    adopt_model,
    search_path_snapshot,
    update_search_paths,
)
from ...admin.models import (
    DownloadBusy,
    DownloadError,
    ModelDownloader,
    model_catalog,
)
from ...admin.settings import SettingsError, settings_snapshot, update_settings
from ...admin.system import (
    read_log_tail,
    schedule_restart,
    scheduled_task_status,
    set_autostart,
    system_info,
)
from ...core.idle import IdleWatchdog
from ..deps import get_cfg, get_registry, get_scheduler, get_vram_manager

logger = logging.getLogger("localai.api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# 请求体
# ---------------------------------------------------------------------------
class SettingsUpdateRequest(BaseModel):
    """设置保存请求。

    推荐 ``{"values": {...}}``；为了 curl 方便也接受顶层扁平键值对
    （``{"port": 7880}``），两者会合并，``values`` 优先。
    """

    values: dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class ModelDownloadRequest(BaseModel):
    capability: str = Field(..., description="tts / asr / whisper / translate / speaker / separate / vad")
    name: str = Field("", description="档位或模型名；whisper 必填（tiny/base/small/medium/large-v3）")


class ModelActivateRequest(BaseModel):
    capability: str = Field(..., description="whisper / asr / translate")
    name: str = Field("", description="要切换到的档位或引擎名")


class SearchPathsUpdateRequest(BaseModel):
    """整体替换 ``paths``，或按 ``add`` / ``remove`` 增删，二者只能选一种。"""

    paths: Optional[list[str]] = Field(None, description="新的完整搜索路径列表")
    add: Optional[str | list[str]] = Field(None, description="要追加的一个或多个目录")
    remove: Optional[str | list[str]] = Field(None, description="要移除的一个或多个目录")


class ModelScanRequest(BaseModel):
    paths: Optional[list[str]] = Field(None, description="省略则使用已保存的搜索路径")
    max_depth: int = Field(6, ge=0, le=12, description="递归深度，默认 6")


class ModelAdoptRequest(BaseModel):
    capability: str = Field(..., description="tts/asr/whisper/translate/speaker/separate/vision/vad")
    path: str = Field(..., description="已有模型所在目录（只读校验后写配置）")
    name: str = Field("", description="whisper 档位；其它能力可留空")


class AutostartRequest(BaseModel):
    enabled: bool = Field(..., description="true=启用开机自启，false=禁用")


class RestartRequest(BaseModel):
    confirm: bool = Field(False, description="必须显式传 true 才会重启，防误触")


# ---------------------------------------------------------------------------
# 设置
# ---------------------------------------------------------------------------
@router.get("/settings", summary="当前生效配置（按分组，带来源与是否需要重启）")
async def get_settings(request: Request) -> dict[str, Any]:
    return settings_snapshot(get_cfg(request))


def _watchdog_table(request: Request) -> list[dict[str, Any]]:
    app = request.app
    table = getattr(app.state, "idle_watchdogs", None)
    if table is None:
        table = []
        app.state.idle_watchdogs = table
    return table


async def _sync_idle_watchdogs(request: Request) -> None:
    """把空闲秒数变更同步给看门狗（不用重启）。"""
    cfg = get_cfg(request)
    registry = get_registry(request)
    app = request.app
    table = _watchdog_table(request)
    by_name = {entry["name"]: entry for entry in table}
    tasks: list[Any] = getattr(app.state, "idle_tasks", [])
    app.state.idle_tasks = tasks

    for adapter in registry.gpu_adapters():
        if adapter.name == "tts" and cfg.tts_backend != "engine":
            continue
        desired = int(adapter.idle_unload_seconds())
        entry = by_name.get(adapter.name)

        if desired <= 0:
            if entry is not None:
                entry["task"].cancel()
                table.remove(entry)
            continue

        if entry is not None and not entry["task"].done():
            entry["watchdog"].idle_seconds = desired
            continue

        if entry is not None:
            table.remove(entry)
        watchdog = IdleWatchdog(adapter, desired)
        task = asyncio.create_task(watchdog.run())
        tasks.append(task)
        table.append({"name": adapter.name, "watchdog": watchdog, "task": task})


async def _sync_parallel_gates(request: Request) -> None:
    """并行度变更时同步适配器闸门（容量只会按配置调整）。"""
    cfg = get_cfg(request)
    registry = get_registry(request)
    for adapter in registry.gpu_adapters():
        if not getattr(adapter, "CAN_INFER_IN_PARALLEL", False):
            continue
        capacity = cfg.jobs_parallel_for(adapter.name)
        try:
            await adapter.infer_lock.set_capacity(
                capacity, reason="管理界面修改并行度"
            )
            adapter.parallel_requested = int(capacity)
            adapter.parallel_capacity = int(capacity)
        except Exception:  # noqa: BLE001 - 单点调整失败不影响保存
            logger.exception("调整 %s 并发闸门失败", adapter.name)


@router.put("/settings", summary="保存配置（白名单 + 类型/范围校验）")
async def put_settings(payload: SettingsUpdateRequest, request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    values: dict[str, Any] = {}
    # 先放扁平键，再让 values 覆盖同名键。
    for key, value in (payload.model_extra or {}).items():
        values[str(key)] = value
    values.update(payload.values or {})

    try:
        result = update_settings(cfg, values)
    except SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    changed = set(result.get("applied") or [])
    if changed & {
        "idle_unload",
        "asr_idle_unload",
        "whisper_idle_unload",
        "speaker_idle_unload",
        "separate_idle_unload",
        "translate_idle_unload",
    }:
        await _sync_idle_watchdogs(request)
    if changed & {"asr_parallel", "jobs_parallel_tts", "jobs_parallel_separate"}:
        await _sync_parallel_gates(request)
    if "vram_margin_mb" in changed:
        vram = get_vram_manager(request)
        vram.margin_mb = max(0, int(cfg.vram_margin_mb))
    if "token" in changed:
        result.setdefault("warnings", []).append(
            "访问令牌已更新：请立即在页面右上角「令牌」里填入新值，否则后续请求会 401。"
        )
    return result


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
@router.get("/models", summary="各能力模型清单（是否下载 / 是否当前档位 / 缺失文件）")
async def get_models(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    items = await asyncio.to_thread(model_catalog, cfg)
    return {
        "ok": True,
        "models_root": str(cfg.home / "models"),
        "models": items,
    }


@router.get("/models/search-paths", summary="读取已有模型的搜索路径（默认空）")
async def get_model_search_paths(request: Request) -> dict[str, Any]:
    return search_path_snapshot(get_cfg(request))


@router.put("/models/search-paths", summary="保存搜索路径（校验存在/去重/最多 20 个）")
async def put_model_search_paths(
    payload: SearchPathsUpdateRequest, request: Request
) -> dict[str, Any]:
    cfg = get_cfg(request)
    fields = payload.model_fields_set
    try:
        return update_search_paths(
            cfg,
            paths=payload.paths if "paths" in fields else None,
            add=payload.add if "add" in fields else None,
            remove=payload.remove if "remove" in fields else None,
        )
    except DiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/scan", summary="后台扫描已有模型目录（只读）")
async def start_model_scan(payload: ModelScanRequest, request: Request) -> dict[str, Any]:
    scanner: ModelScanner = request.app.state.model_scanner
    try:
        return scanner.start(
            get_cfg(request), payload.paths, max_depth=payload.max_depth
        )
    except DiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ScanBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/models/scan/{task_id}", summary="查看模型扫描进度与结果")
async def model_scan_status(task_id: str, request: Request) -> dict[str, Any]:
    scanner: ModelScanner = request.app.state.model_scanner
    snapshot = scanner.status(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"扫描任务不存在：{task_id}")
    return snapshot


@router.post("/models/adopt", summary="采用已有模型（只写 config.json，建链接/不复制）")
async def adopt_existing_model(
    payload: ModelAdoptRequest, request: Request
) -> dict[str, Any]:
    try:
        return adopt_model(
            get_cfg(request),
            payload.capability,
            payload.path,
            name=payload.name,
        )
    except DiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/download", summary="异步下载模型（同一时刻只允许一个任务）")
async def download_model(payload: ModelDownloadRequest, request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    downloader: ModelDownloader = request.app.state.model_downloader
    try:
        return downloader.start(cfg, payload.capability, payload.name)
    except DownloadBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DownloadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/tasks/{task_id}", summary="查看下载任务进度")
async def download_task(task_id: str, request: Request) -> dict[str, Any]:
    downloader: ModelDownloader = request.app.state.model_downloader
    snapshot = downloader.status(task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"下载任务不存在：{task_id}")
    return {"ok": True, **snapshot}


@router.post("/models/activate", summary="切换当前模型档位（写配置，不重启进程）")
async def activate_model(payload: ModelActivateRequest, request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    capability = str(payload.capability or "").strip().lower()
    name = str(payload.name or "").strip()
    lower = name.lower()

    if capability == "whisper":
        if lower not in ("tiny", "base", "small", "medium", "large-v3"):
            raise HTTPException(
                status_code=400,
                detail="whisper 档位必须是 tiny / base / small / medium / large-v3",
            )
        key, value = "whisper_model", lower
        note = (
            f"whisper 默认档位已切到 {lower}；下次 whisper 请求会按新档位加载，"
            "已加载的旧档位会由空闲看门狗/显存管家回收。"
        )
    elif capability in ("asr", "sensevoice"):
        if lower not in ("sensevoice", "whisper"):
            raise HTTPException(
                status_code=400,
                detail="ASR 默认引擎必须是 sensevoice 或 whisper",
            )
        key, value = "asr_engine", lower
        note = f"默认识别引擎已切到 {lower}；下一次识别请求立即生效。"
    elif capability in ("translate", "nllb"):
        raise HTTPException(
            status_code=400,
            detail="翻译目前只有一个模型（nllb-200-distilled-600M），无需切换",
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"能力 {capability!r} 不支持切换档位；可切换：whisper / asr",
        )

    try:
        result = update_settings(cfg, {key: value})
    except SettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "capability": capability,
        "name": value,
        "active": value,
        "requires_restart": bool(result.get("requires_restart")),
        "warnings": result.get("warnings") or [],
        "note": note,
    }


# ---------------------------------------------------------------------------
# 系统
# ---------------------------------------------------------------------------
@router.get("/system", summary="版本 / GPU / 磁盘 / 计划任务 / 监听地址")
async def get_system(request: Request) -> dict[str, Any]:
    cfg = get_cfg(request)
    vram = get_vram_manager(request)
    registry = get_registry(request)
    started_at = getattr(request.app.state, "started_at", None)
    jobs_stats: Optional[dict[str, Any]] = None
    try:
        scheduler = get_scheduler(request)
        jobs_stats = await scheduler.stats()
    except Exception:  # noqa: BLE001 - 调度器未启动时也要能看系统页
        jobs_stats = None
    return await asyncio.to_thread(
        system_info,
        cfg,
        vram=vram,
        registry=registry,
        started_at=started_at,
        jobs_stats=jobs_stats,
    )


@router.get("/autostart", summary="开机自启（计划任务）状态")
async def get_autostart(request: Request) -> dict[str, Any]:
    status = await asyncio.to_thread(scheduled_task_status)
    return {"ok": True, **status}


@router.post("/autostart", summary="启用 / 禁用开机自启")
async def post_autostart(payload: AutostartRequest, request: Request) -> dict[str, Any]:
    if os.name != "nt":
        raise HTTPException(status_code=501, detail="开机自启管理只支持 Windows")
    cfg = get_cfg(request)
    try:
        result = await asyncio.to_thread(set_autostart, cfg, bool(payload.enabled))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/restart", summary="异步重启服务（必须 confirm=true）")
async def restart_service(payload: RestartRequest, request: Request) -> JSONResponse:
    if not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="重启服务必须显式传 {\"confirm\": true}，防止误触。",
        )
    cfg = get_cfg(request)
    try:
        result = schedule_restart(request.app, cfg)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(status_code=202, content=result)


@router.get("/logs", summary="查看 service.out.log / service.err.log 尾部")
async def get_logs(
    request: Request,
    lines: int = Query(200, ge=1, le=5000, description="返回最后多少行"),
    file: str = Query("service", description="service / out / err"),
) -> dict[str, Any]:
    cfg = get_cfg(request)
    try:
        return await asyncio.to_thread(read_log_tail, cfg, file, lines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
