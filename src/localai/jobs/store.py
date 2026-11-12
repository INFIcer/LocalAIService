"""持久化任务存储（SQLite，标准库 sqlite3，不引第三方）。

为什么用 SQLite
---------------
- 服务重启后队列不能丢，JSON 文件做不到「一条一条原子更新 + 按条件查询」；
- 标准库自带，Windows 上零依赖；
- 单进程、单 worker 的写入量很小，WAL 模式足够。

线程模型
--------
``sqlite3.Connection`` 默认不允许跨线程使用。这里用
``check_same_thread=False`` + 一把 ``threading.Lock``，并把所有数据库
操作都放进 ``asyncio.to_thread``（见每个 ``async`` 方法），**绝不在事件
循环里做长查询**。同一时刻只有一个连接、一把锁，写入天然串行。

表结构见 :data:`SCHEMA`；状态机::

    queued --claim--> running --finish--> done
       |                 |  \\--fail--> failed
       |                 \\--cancel--> canceled
       \\--cancel--> canceled
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("localai.jobs.store")

#: 合法状态（也是 /jobs?state= 的合法取值）。
JOB_STATES = ("queued", "running", "done", "failed", "canceled")
#: 终态：不会再变化。
TERMINAL_STATES = ("done", "failed", "canceled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    action      TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 0,
    state       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    payload     TEXT,
    result      TEXT,
    error       TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    client_ip   TEXT,
    meta        TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state_priority_created
    ON jobs(state, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_kind_state
    ON jobs(kind, state);
"""


def now_iso() -> str:
    """当前本地时间（带时区），ISO8601 毫秒精度。"""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NewJob:
    """提交任务时的入参（id 由调用方生成，便于先把上传文件落到 id 目录）。"""

    id: str
    kind: str
    action: str
    payload: dict[str, Any]
    priority: int = 0
    client_ip: str = ""
    meta: Optional[dict[str, Any]] = None


class JobStore:
    """``<HOME>/data/jobs.db`` 的读写封装；所有公开方法都是 async。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._conn is not None:
                return
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=15.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # WAL：读写不互相阻塞，崩溃后也能恢复到一致状态。
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()
            self._conn = conn
        logger.info("任务库就绪：%s（WAL，%s）", self.db_path, "新建/复用")

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            # 允许「没调用 init 就查询」的容错路径（主要给单元测试）。
            self._init_sync()
        assert self._conn is not None
        return self._conn

    # ------------------------------------------------------------------
    # 行 <-> dict
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item.pop("_rowid", None)  # pick_next 的排序辅助列，不对外暴露
        item["payload"] = _json_loads(item.get("payload")) or {}
        item["result"] = _json_loads(item.get("result"))
        item["meta"] = _json_loads(item.get("meta")) or {}
        return item

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    async def create(self, job: NewJob) -> dict[str, Any]:
        """插入一条 queued 任务并返回完整记录。"""
        return await asyncio.to_thread(self._create_sync, job)

    def _create_sync(self, job: NewJob) -> dict[str, Any]:
        created_at = now_iso()
        with self._lock:
            conn = self._connection()
            conn.execute(
                "INSERT INTO jobs (id, kind, action, priority, state, created_at,"
                " payload, attempts, client_ip, meta)"
                " VALUES (?, ?, ?, ?, 'queued', ?, ?, 0, ?, ?)",
                (
                    job.id,
                    job.kind,
                    job.action,
                    int(job.priority),
                    created_at,
                    _json_dumps(job.payload or {}),
                    job.client_ip or "",
                    _json_dumps(job.meta or {}),
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job.id,)).fetchone()
        item = self._row_to_dict(row)
        logger.debug(
            "任务入库：id=%s kind=%s action=%s priority=%d",
            job.id,
            job.kind,
            job.action,
            int(job.priority),
        )
        return item

    async def mark_running(self, job_id: str) -> bool:
        """queued -> running；如果期间被取消/已被别人领走则返回 False。"""
        return await asyncio.to_thread(self._mark_running_sync, job_id)

    def _mark_running_sync(self, job_id: str) -> bool:
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "UPDATE jobs SET state = 'running', started_at = ?,"
                " attempts = attempts + 1"
                " WHERE id = ? AND state = 'queued'",
                (now_iso(), job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def finish(self, job_id: str, result: dict[str, Any]) -> bool:
        """running -> done；只写一次，取消后的任务不会被结果覆盖。"""
        return await asyncio.to_thread(self._finish_sync, job_id, result)

    def _finish_sync(self, job_id: str, result: dict[str, Any]) -> bool:
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "UPDATE jobs SET state = 'done', finished_at = ?, result = ?,"
                " error = NULL WHERE id = ? AND state = 'running'",
                (now_iso(), _json_dumps(result or {}), job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def fail(self, job_id: str, error: str) -> bool:
        """running -> failed（也允许把 queued 直接判失败，用于恢复兜底）。"""
        return await asyncio.to_thread(self._fail_sync, job_id, error)

    def _fail_sync(self, job_id: str, error: str) -> bool:
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "UPDATE jobs SET state = 'failed', finished_at = ?, error = ?"
                " WHERE id = ? AND state IN ('queued', 'running')",
                (now_iso(), error[:4000], job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def cancel_queued(self, job_id: str) -> Optional[dict[str, Any]]:
        """queued -> canceled；返回取消后的记录，任务不存在返回 None。"""
        return await asyncio.to_thread(self._cancel_queued_sync, job_id)

    def _cancel_queued_sync(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connection()
            conn.execute(
                "UPDATE jobs SET state = 'canceled', finished_at = ?,"
                " error = COALESCE(error, '已取消')"
                " WHERE id = ? AND state = 'queued'",
                (now_iso(), job_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    async def set_cancel_requested(self, job_id: str) -> Optional[dict[str, Any]]:
        """running 任务：只打 ``meta.cancel_requested``，等执行器退出后再置 canceled。"""
        return await asyncio.to_thread(self._set_cancel_requested_sync, job_id)

    async def update_meta(self, job_id: str, patch: dict[str, Any]) -> bool:
        """把 ``patch`` 合并进任务 meta（下载进度用；只更新 running 任务）。

        合并是浅合并：``{"progress": {...}}`` 会整体替换旧的 progress，
        正是进度上报需要的语义。任务已结束或不存在时返回 False。
        """
        return await asyncio.to_thread(self._update_meta_sync, job_id, patch)

    def _update_meta_sync(self, job_id: str, patch: dict[str, Any]) -> bool:
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT meta, state FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None or row["state"] != "running":
                return False
            meta = _json_loads(row["meta"]) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta.update(patch or {})
            conn.execute(
                "UPDATE jobs SET meta = ? WHERE id = ?",
                (_json_dumps(meta), job_id),
            )
            conn.commit()
        return True

    def _set_cancel_requested_sync(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = self._connection()
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["state"] == "running":
                meta = _json_loads(row["meta"]) or {}
                meta["cancel_requested"] = True
                conn.execute(
                    "UPDATE jobs SET meta = ? WHERE id = ?",
                    (_json_dumps(meta), job_id),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    async def mark_canceled(self, job_id: str, error: str = "已取消") -> bool:
        """running -> canceled（执行器退出后由调度器调用）。"""
        return await asyncio.to_thread(self._mark_canceled_sync, job_id, error)

    def _mark_canceled_sync(self, job_id: str, error: str) -> bool:
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "UPDATE jobs SET state = 'canceled', finished_at = ?, error = ?"
                " WHERE id = ? AND state = 'running'",
                (now_iso(), error[:4000], job_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # 读取 / 调度选择
    # ------------------------------------------------------------------
    async def get(self, job_id: str) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._get_sync, job_id)

    def _get_sync(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    async def list_jobs(
        self,
        *,
        state: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """按创建时间倒序分页；返回 (记录列表, 过滤后的总数)。"""
        return await asyncio.to_thread(
            self._list_sync, state, kind, int(limit), int(offset)
        )

    def _list_sync(
        self,
        state: Optional[str],
        kind: Optional[str],
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        where: list[str] = []
        params: list[Any] = []
        if state:
            where.append("state = ?")
            params.append(state)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            conn = self._connection()
            total = int(
                conn.execute(f"SELECT COUNT(*) FROM jobs{clause}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT * FROM jobs{clause}"
                " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                [*params, max(1, limit), max(0, offset)],
            ).fetchall()
        return [self._row_to_dict(row) for row in rows], total

    async def pick_next(
        self,
        *,
        current_kind: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """调度器选下一个任务：**同类型优先 > 优先级 > 提交时间**（没有 aging）。

        规则（与 ``docs/SERVICE_API.md`` 一致）::

            排序键 = (priority DESC, created_at ASC, rowid ASC)
            1. current_kind 有效、且该 kind 队列非空
               -> 只在这个 kind 里按排序键取（别的 kind 优先级再高也不切走）；
            2. 否则在所有 kind 里按排序键取一个，它的 kind 成为新的 current_kind。

        这就是「同类型优先」的全部实现：当前 kind 只要还有排队任务，就绝不
        切换到别的类型；只有同类彻底清空（唯一的切换出口）才会换 kind。
        代价是同类型持续涌入时其它类型会被饿死 —— 这是明确接受的取舍，
        没有 aging、没有提权。

        ``rowid`` 只做同毫秒提交的 FIFO 兜底（``created_at`` 是毫秒精度）。
        """
        return await asyncio.to_thread(self._pick_next_sync, current_kind)

    def _pick_next_sync(self, current_kind: Optional[str]) -> Optional[dict[str, Any]]:
        with self._lock:
            rows = self._connection().execute(
                "SELECT rowid AS _rowid, * FROM jobs WHERE state = 'queued'"
            ).fetchall()
        if not rows:
            return None

        def order(row: sqlite3.Row) -> tuple[int, str, int]:
            return (
                -int(row["priority"] or 0),
                row["created_at"] or "",
                int(row["_rowid"]),
            )

        same = [row for row in rows if row["kind"] == current_kind] if current_kind else []
        chosen = min(same, key=order) if same else min(rows, key=order)
        return self._row_to_dict(chosen)

    async def queued_count(self, kind: Optional[str] = None) -> int:
        return await asyncio.to_thread(self._queued_count_sync, kind)

    def _queued_count_sync(self, kind: Optional[str]) -> int:
        with self._lock:
            conn = self._connection()
            if kind:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state = 'queued' AND kind = ?",
                    (kind,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state = 'queued'"
                ).fetchone()
        return int(row[0])

    async def health(self) -> dict[str, Any]:
        """``/health`` 用的轻量计数。"""
        return await asyncio.to_thread(self._health_sync)

    def _health_sync(self) -> dict[str, Any]:
        with self._lock:
            conn = self._connection()
            rows = conn.execute(
                "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
            ).fetchall()
        counts = {str(row["state"]): int(row["n"]) for row in rows}
        return {
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "canceled": counts.get("canceled", 0),
        }

    async def stats(self) -> dict[str, Any]:
        """任务统计：状态/能力计数、平均等待与执行耗时、最老排队时长。"""
        return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, Any]:
        """任务统计：全部用 SQL 聚合，历史再长也不把整张表读进内存。"""
        with self._lock:
            conn = self._connection()
            by_state = {state: 0 for state in JOB_STATES}
            for row in conn.execute(
                "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
            ):
                by_state[str(row["state"])] = int(row["n"])
            by_kind = {
                str(row["kind"]): int(row["n"])
                for row in conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM jobs GROUP BY kind"
                )
            }
            queued_by_kind = {
                str(row["kind"]): int(row["n"])
                for row in conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM jobs"
                    " WHERE state = 'queued' GROUP BY kind"
                )
            }
            running_by_kind = {
                str(row["kind"]): int(row["n"])
                for row in conn.execute(
                    "SELECT kind, COUNT(*) AS n FROM jobs"
                    " WHERE state = 'running' GROUP BY kind"
                )
            }
            wait_row = conn.execute(
                "SELECT AVG((julianday(started_at) - julianday(created_at))"
                " * 86400000.0) AS avg_wait FROM jobs"
                " WHERE started_at IS NOT NULL AND created_at IS NOT NULL"
            ).fetchone()
            run_row = conn.execute(
                "SELECT AVG((julianday(finished_at) - julianday(started_at))"
                " * 86400000.0) AS avg_run FROM jobs"
                " WHERE finished_at IS NOT NULL AND started_at IS NOT NULL"
            ).fetchone()
            oldest_row = conn.execute(
                "SELECT (julianday('now') - julianday(MIN(created_at)))"
                " * 86400.0 AS age FROM jobs WHERE state = 'queued'"
            ).fetchone()

        avg_wait = float(wait_row["avg_wait"] or 0.0) if wait_row else 0.0
        avg_run = float(run_row["avg_run"] or 0.0) if run_row else 0.0
        oldest_age = float(oldest_row["age"] or 0.0) if oldest_row else 0.0
        total = sum(by_state.values())

        return {
            "total": total,
            "queued": by_state.get("queued", 0),
            "running": by_state.get("running", 0),
            "done": by_state.get("done", 0),
            "failed": by_state.get("failed", 0),
            "canceled": by_state.get("canceled", 0),
            "by_state": by_state,
            "by_kind": by_kind,
            "queued_by_kind": queued_by_kind,
            "running_by_kind": running_by_kind,
            "avg_wait_ms": round(max(0.0, avg_wait), 1),
            "avg_run_ms": round(max(0.0, avg_run), 1),
            "oldest_queued_seconds": round(max(0.0, oldest_age), 1),
        }

    # ------------------------------------------------------------------
    # 历史清理
    # ------------------------------------------------------------------
    async def prune(self, older_than_days: int) -> int:
        """删除超过保留期的终态任务；返回删除行数（0=不清理）。

        只删 done/failed/canceled：排队中或执行中的任务永远不会被清掉。
        """
        return await asyncio.to_thread(self._prune_sync, older_than_days)

    def _prune_sync(self, older_than_days: int) -> int:
        if older_than_days <= 0:
            return 0
        cutoff = (
            datetime.now().astimezone() - timedelta(days=older_than_days)
        ).isoformat(timespec="milliseconds")
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "DELETE FROM jobs WHERE state IN (?, ?, ?)"
                " AND COALESCE(finished_at, created_at) < ?",
                (*TERMINAL_STATES, cutoff),
            )
            deleted = int(cursor.rowcount or 0)
            conn.commit()
            if deleted:
                # 删除大量行后 VACUUM 回收文件空间；必须在事务外执行。
                conn.execute("VACUUM")
        if deleted:
            logger.info("任务历史清理：删除 %d 条终态记录（保留 %d 天）", deleted, older_than_days)
        return deleted

    # ------------------------------------------------------------------
    # 崩溃恢复
    # ------------------------------------------------------------------
    async def recover_running(self, mode: str = "requeue") -> dict[str, int]:
        """把上次进程遗留的 running 任务重新排队或判失败。

        - ``requeue``（默认）：状态回 queued，保留 createdAt/priority，
          ``started_at`` 清空，``error`` 写恢复原因；
        - ``fail``：直接置 failed，不重复执行可能只做了一半的 GPU 任务。
        """
        return await asyncio.to_thread(self._recover_running_sync, mode)

    def _recover_running_sync(self, mode: str) -> dict[str, int]:
        with self._lock:
            conn = self._connection()
            rows = conn.execute("SELECT id FROM jobs WHERE state = 'running'").fetchall()
            ids = [str(row["id"]) for row in rows]
            if not ids:
                return {"requeued": 0, "failed": 0}
            if mode == "fail":
                conn.execute(
                    "UPDATE jobs SET state = 'failed', finished_at = ?,"
                    " error = '服务重启中断' WHERE state = 'running'",
                    (now_iso(),),
                )
                conn.commit()
                logger.warning("崩溃恢复：%d 个 running 任务置 failed（LOCALAI_JOBS_RECOVER=fail）", len(ids))
                return {"requeued": 0, "failed": len(ids)}
            conn.execute(
                "UPDATE jobs SET state = 'queued', started_at = NULL,"
                " finished_at = NULL, error = '服务重启中断，已重新排队'"
                " WHERE state = 'running'"
            )
            conn.commit()
        logger.warning(
            "崩溃恢复：%d 个 running 任务重新排队（LOCALAI_JOBS_RECOVER=requeue）", len(ids)
        )
        return {"requeued": len(ids), "failed": 0}
