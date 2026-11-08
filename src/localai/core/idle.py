"""通用空闲看门狗。

每个 GPU 能力一个独立任务：只动自己的模型，互不影响。
加载仍然发生在第一次请求（不预热），这里只负责「长时间没人用就释放显存」。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .concurrency import exclusive

logger = logging.getLogger("localai.idle")


class IdleWatchdog:
    """盯住一个 adapter，空闲超过 ``idle_seconds`` 就卸载。"""

    def __init__(
        self,
        adapter: Any,
        idle_seconds: int,
        *,
        label: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.adapter = adapter
        self.idle_seconds = int(idle_seconds)
        self.label = label or adapter.title
        self.log = logger_ or logger

    async def run(self) -> None:
        if self.idle_seconds <= 0:
            return
        self.log.info(
            "%s 空闲卸载已启用：%d 秒没有请求就卸载 %s",
            self.label,
            self.idle_seconds,
            self.adapter.model_name,
        )
        interval = max(5, min(30, self.idle_seconds // 4))
        while True:
            try:
                await asyncio.sleep(interval)
                idle = self.adapter.idle_seconds()
                if idle is None or idle < self.idle_seconds:
                    continue
                if not self.adapter.loaded():
                    continue
                # 独占推理闸门：如果此刻正在推理（可能有多个并发），
                # 就等它们全部跑完再卸，避免卸到一半。
                async with exclusive(self.adapter.infer_lock):
                    if not self.adapter.loaded():
                        continue
                    await asyncio.to_thread(self.adapter.unload)
                self.adapter.reset_activity()
                self.log.info(
                    "%s 空闲 %.0f 秒，已卸载 %s 腾出显存",
                    self.label,
                    idle,
                    self.adapter.model_name,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 看门狗不能把服务搞死
                self.log.warning("%s 空闲卸载检查出错：%s", self.label, exc)
