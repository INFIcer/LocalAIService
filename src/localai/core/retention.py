"""运行产物保留期清理：任务历史、outputs 与临时目录。

为什么需要
----------
任务表、outputs 里的转码产物、以及各临时目录默认只增不减；进程被强杀时
还会留下半截文件。这里按配置的保留期统一清理：

- jobs 表：只删超过 jobs_retention_days 的终态记录（见 JobStore.prune）；
- outputs：按 output_retention_days 删过期文件；
- 临时目录（asr_tmp / media_tmp / frames / scenes / speaker_tmp /
  separate_tmp）：按 tmp_retention_hours 删整条目（包括目录）。

jobs_upload_dir 不在这里清：排队/执行中的任务还要读里面的输入文件，
调度器在任务结束时自己删。0 表示对应项不清理；任何清理失败只记 warning，
绝不影响服务启动与运行。
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ..jobs.store import JobStore

logger = logging.getLogger("localai.retention")

#: 定时清理间隔（秒）。启动时先清一次，之后每小时一次。
RETENTION_INTERVAL_SECONDS = 3600.0


def _tmp_roots(cfg: Any) -> list[Path]:
    return [
        cfg.asr_tmp_dir,
        cfg.media_tmp_dir,
        cfg.frames_dir,
        cfg.scenes_dir,
        cfg.speaker_tmp_dir,
        cfg.separate_tmp_dir,
    ]


def _remove(path: Path) -> bool:
    """删除文件或目录；失败只记 warning（占用中的文件下次再删）。"""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("清理失败（下次重试）：%s：%s", path, exc)
        return False
    return True


def _sweep_dir(root: Path, max_age_seconds: float) -> int:
    """删掉 root 下 mtime 超过保留期的条目；返回删除数量。"""
    if max_age_seconds <= 0:
        return 0
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.warning("无法列出清理目录 %s：%s", root, exc)
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for entry in entries:
        try:
            if entry.stat(follow_symlinks=False).st_mtime >= cutoff:
                continue
        except OSError:
            continue
        if _remove(entry):
            removed += 1
    return removed


def _run_sync(cfg: Any, store: JobStore) -> dict[str, int]:
    result = {"jobs": 0, "outputs": 0, "tmp": 0}
    if cfg.jobs_retention_days > 0:
        result["jobs"] = store._prune_sync(cfg.jobs_retention_days)  # noqa: SLF001 - 同一清理职责
    if cfg.output_retention_days > 0:
        result["outputs"] = _sweep_dir(
            cfg.output_dir, cfg.output_retention_days * 86400
        )
    if cfg.tmp_retention_hours > 0:
        for root in _tmp_roots(cfg):
            result["tmp"] += _sweep_dir(root, cfg.tmp_retention_hours * 3600)
    if any(result.values()):
        logger.info(
            "保留期清理：任务 %d 条、outputs %d 个、临时条目 %d 个",
            result["jobs"],
            result["outputs"],
            result["tmp"],
        )
    return result


async def run_retention(cfg: Any, store: JobStore) -> dict[str, int]:
    """清理一次过期产物；返回各项删除数量。"""
    return await asyncio.to_thread(_run_sync, cfg, store)


async def retention_loop(
    cfg: Any,
    store: JobStore,
    interval_seconds: float = RETENTION_INTERVAL_SECONDS,
) -> None:
    """启动时先清一次，之后周期性清理；由 app 生命周期负责取消。"""
    while True:
        try:
            await run_retention(cfg, store)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 清理失败不能拖垮服务
            logger.warning("保留期清理异常：%s: %s", type(exc).__name__, exc)
        await asyncio.sleep(interval_seconds)
