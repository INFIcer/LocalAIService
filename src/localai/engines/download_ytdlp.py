"""下载能力适配器：把 ``engine/download_engine.py`` 接进统一能力注册表。

特点
----
- **不占显存**：``requires_gpu=False``，因此不进显存管家 LRU、不挂空闲看门狗；
- 不加载任何模型，``load/unload`` 是空操作；
- ``estimate_mb()`` 固定返回 0；
- ``status()`` 暴露 yt-dlp 版本 / 路径 / aria2 可用性 / 下载目录与限额，
  供 ``/capabilities``、``/admin/models``、``/admin/system`` 使用。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ._native import import_engine_module
from .base import BaseAdapter


class DownloadAdapter(BaseAdapter):
    name = "download"
    title = "通用下载（yt-dlp + 直链兜底）"
    model_name = "yt-dlp"
    requires_gpu = False
    DEFAULT_ESTIMATE_MB = 0
    ENDPOINTS = (
        "/media/download",
        "/media/download/sync",
        "/media/download/status",
        "/media/analyze",
        "/jobs/analyze",
    )

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("download_engine")
        return self._module

    def status(self) -> dict[str, Any]:
        try:
            module = self.get_module()
            info = module.status_info()
        except Exception as exc:  # noqa: BLE001 - 状态查询不能把 /capabilities 打挂
            return {
                "ready": False,
                "loaded": False,
                "model": "yt-dlp",
                "version": None,
                "detail": f"下载引擎不可用：{type(exc).__name__}: {exc}",
            }
        ytdlp = info.get("yt_dlp") or {}
        ready = bool(ytdlp.get("installed"))
        return {
            "ready": ready,
            "loaded": False,
            "model": "yt-dlp",
            "version": ytdlp.get("version"),
            "path": ytdlp.get("path"),
            "python": ytdlp.get("python"),
            "aria2c": info.get("aria2c"),
            "download_dir": info.get("download_dir"),
            "today_dir": info.get("today_dir"),
            "limits": info.get("limits"),
            "ffmpeg": info.get("ffmpeg"),
            "cookies": info.get("cookies"),
            "ssrf": info.get("ssrf"),
            "detail": (
                f"yt-dlp {ytdlp.get('version')} 就绪"
                if ready
                else "yt-dlp 未安装：在 engine/.venv 执行 python -m pip install -U yt-dlp"
            ),
        }

    async def call(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        """在线程池里跑同步下载函数，避免阻塞事件循环。"""
        module = self.get_module()
        func = getattr(module, func_name)
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except module.DownloadBlocked as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except module.DownloadCancelled as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except module.DownloadTimeout as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except module.DownloadError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 没有模型，load / unload / loaded 都是空操作。
    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def loaded(self) -> bool:
        return False

    def estimate_mb(self) -> int:
        return 0
