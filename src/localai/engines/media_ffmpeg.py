"""媒体适配器：ffmpeg（探测 / 转 16k wav / 抽帧 / 抽音轨）。

不占显存、不加载模型；ffmpeg 路径由 :mod:`localai.config` 统一探测
（``LOCALAI_FFMPEG`` -> config.json 的 ``ffmpeg`` 字段 -> PATH）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.media")


class MediaAdapter(BaseAdapter):
    name = "media"
    title = "媒体工具（ffmpeg）"
    model_name = "ffmpeg"
    requires_gpu = False
    DEFAULT_ESTIMATE_MB = 0
    ENDPOINTS = (
        "/media/probe",
        "/media/to_wav",
        "/media/frames",
        "/media/audio",
        "/media/scenes",
        "/media/subtitles",
        "/media/subtitles/extract",
    )

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("ffmpeg_tool")
        return self._module

    def status(self) -> dict[str, Any]:
        try:
            module = self.get_module()
            info = module.status()
        except Exception as exc:  # noqa: BLE001 - health 不能让整个接口挂掉
            return {
                "ready": False,
                "loaded": False,
                "model": "ffmpeg",
                "ffmpeg": "missing",
                "path": None,
                "detail": f"媒体模块不可用：{exc}",
            }
        if info.get("ffmpeg") != "ok":
            info.setdefault(
                "hint",
                "把 ffmpeg.exe 的完整路径写进 config.json 的 ffmpeg 字段，或设置 LOCALAI_FFMPEG",
            )
        info["ready"] = info.get("ffmpeg") == "ok"
        info["model"] = "ffmpeg"
        info["loaded"] = False
        return info

    def health(self) -> dict[str, Any]:
        """给 /health 用的 ffmpeg 状态；只查文件是否存在，不执行 ffmpeg。"""
        try:
            module = self.get_module()
            info = module.status()
        except Exception as exc:  # noqa: BLE001 - health 不能让整个接口挂掉
            return {"ffmpeg": "missing", "path": None, "detail": f"媒体模块不可用：{exc}"}
        if info.get("ffmpeg") != "ok":
            info.setdefault(
                "hint",
                "把 ffmpeg.exe 的完整路径写进 config.json 的 ffmpeg 字段，或设置 LOCALAI_FFMPEG",
            )
        return info

    @staticmethod
    def _http_error(exc: Exception) -> HTTPException:
        """把 ffmpeg 的异常翻译成合适的 HTTP 状态码（路径问题 503、超时 504、其它 400）。"""
        text = str(exc)
        if "找不到" in text or "无法启动" in text:
            return HTTPException(status_code=503, detail=text)
        if "超时" in text:
            return HTTPException(status_code=504, detail=text)
        return HTTPException(status_code=400, detail=text)

    async def call(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        """在线程池里跑 ffmpeg 工具函数，避免阻塞事件循环。"""
        try:
            module = self.get_module()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"媒体模块不可用：{type(exc).__name__}: {exc}",
            ) from exc
        func = getattr(module, func_name)
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except module.MediaError as exc:
            raise self._http_error(exc) from exc

    def load(self) -> None:
        """媒体能力没有模型，load 是空操作。"""

    def unload(self) -> None:
        """媒体能力没有模型，unload 是空操作。"""

    def loaded(self) -> bool:
        return False
