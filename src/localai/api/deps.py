"""路由共享依赖：从 ``request.app.state`` 取配置 / 注册表 / 显存管家。"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from ..config import Config
from ..core.registry import EngineRegistry
from ..core.vram import VramManager
from ..jobs import JobScheduler, JobStore


def get_cfg(request: Request) -> Config:
    return request.app.state.cfg


def get_registry(request: Request) -> EngineRegistry:
    return request.app.state.registry


def get_vram_manager(request: Request) -> VramManager:
    return request.app.state.vram


def get_client(request: Request) -> Any:
    return getattr(request.app.state, "client", None)


def get_scheduler(request: Request) -> JobScheduler:
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("任务调度器尚未启动")
    return scheduler


def get_jobs_store(request: Request) -> JobStore:
    store = getattr(request.app.state, "jobs_store", None)
    if store is None:
        raise RuntimeError("任务库尚未初始化")
    return store
