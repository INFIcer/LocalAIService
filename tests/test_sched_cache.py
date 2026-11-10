"""计划任务状态的短期缓存：系统页每次刷新都要读一遍，不能每次都起 PowerShell。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from localai.admin import system as S  # noqa: E402


def _reset() -> None:
    S._SCHED_CACHE["at"] = 0.0
    S._SCHED_CACHE["info"] = None


def test_second_call_hits_cache(monkeypatch) -> None:
    _reset()
    calls = {"n": 0}

    def fake() -> dict:
        calls["n"] += 1
        return {"supported": True, "exists": True, "state": "Ready", "enabled": True}

    monkeypatch.setattr(S, "_scheduled_task_status_uncached", fake)
    first = S.scheduled_task_status()
    second = S.scheduled_task_status()
    assert calls["n"] == 1
    assert first == second
    assert first["state"] == "Ready"


def test_force_bypasses_cache(monkeypatch) -> None:
    _reset()
    calls = {"n": 0}

    def fake() -> dict:
        calls["n"] += 1
        return {"supported": True, "exists": True, "state": f"Ready-{calls['n']}", "enabled": True}

    monkeypatch.setattr(S, "_scheduled_task_status_uncached", fake)
    assert S.scheduled_task_status()["state"] == "Ready-1"
    assert S.scheduled_task_status(force=True)["state"] == "Ready-2"


def test_expired_cache_reasks(monkeypatch) -> None:
    _reset()
    calls = {"n": 0}

    def fake() -> dict:
        calls["n"] += 1
        return {"supported": True, "exists": False}

    monkeypatch.setattr(S, "_scheduled_task_status_uncached", fake)
    S.scheduled_task_status()
    S._SCHED_CACHE["at"] = S._SCHED_CACHE["at"] - (S._SCHED_CACHE_SECONDS + 1.0)
    S.scheduled_task_status()
    assert calls["n"] == 2


def test_returned_dict_is_a_copy(monkeypatch) -> None:
    _reset()

    def fake() -> dict:
        return {"supported": True, "exists": True, "state": "Ready", "extra": {"a": 1}}

    monkeypatch.setattr(S, "_scheduled_task_status_uncached", fake)
    first = S.scheduled_task_status()
    first["state"] = "改坏了"
    assert S.scheduled_task_status()["state"] == "Ready"
