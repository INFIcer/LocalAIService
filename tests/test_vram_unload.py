"""`POST /vram/unload` 的「忙的模型跳过」语义。

不加载任何真实模型：显存管家里的能力都是假 entry（`is_loaded_fn` /
`unload_fn` 可控），显存读数直接替换 `read_memory`，测试不碰 torch 与
nvidia-smi。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localai.api.schemas import VramUnloadRequest
from localai.core.concurrency import ConcurrencyGate
from localai.core.vram import VramManager


class FakeModel:
    """假能力：可控装载状态 + 卸载计数。"""

    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded
        self.unload_calls = 0
        self.lock = ConcurrencyGate("fake", capacity=1)

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


def _manager(tmp_path: Path) -> VramManager:
    """临时目录里的显存管家；读显存固定返回 (4096, 8192)。"""
    manager = VramManager(estimates_path=tmp_path / "vram_estimates.json")
    manager.read_memory = lambda *, initialize=False: (4096.0, 8192.0)  # type: ignore[method-assign]
    return manager


def _register(manager: VramManager, name: str, model: FakeModel) -> FakeModel:
    manager.register(
        name,
        model.unload,
        100.0,
        is_loaded_fn=lambda: model.loaded,
        infer_lock=model.lock,
    )
    return model


def test_schema_defaults_to_skip_busy() -> None:
    """HTTP 手动释放默认跳过忙模型；显式传 False 时仍可等锁。"""
    assert VramUnloadRequest().skip_busy is True
    assert VramUnloadRequest(skip_busy=False).skip_busy is False


@pytest.mark.asyncio
async def test_skip_busy_skips_active_model(tmp_path: Path) -> None:
    """有请求正在使用（active>0）时既不卸载，也不阻塞。"""
    manager = _manager(tmp_path)
    tts = _register(manager, "tts", FakeModel())
    manager.begin_use("tts")
    try:
        started = time.monotonic()
        result = await manager.unload_models(["tts"], skip_busy=True)
        elapsed = time.monotonic() - started
    finally:
        manager.end_use("tts")
    assert result["released"] == []
    assert result["skipped"] == ["tts"]
    assert tts.unload_calls == 0
    assert tts.loaded is True
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_skip_busy_gives_up_on_held_lock(tmp_path: Path) -> None:
    """推理占着闸门但 active 还没标记：短超时后记为跳过。"""
    manager = _manager(tmp_path)
    tts = _register(manager, "tts", FakeModel())
    async with tts.lock:
        started = time.monotonic()
        result = await manager.unload_models(["tts"], skip_busy=True)
        elapsed = time.monotonic() - started
    assert result["released"] == []
    assert result["skipped"] == ["tts"]
    assert tts.unload_calls == 0
    assert tts.loaded is True
    assert 0.2 <= elapsed < 3.0

    # 跳过没有破坏闸门状态：推理退出后，默认语义仍能正常卸载。
    result = await manager.unload_models(["tts"])
    assert result["released"] == ["tts"]
    assert tts.loaded is False


@pytest.mark.asyncio
async def test_default_waits_for_running_inference(tmp_path: Path) -> None:
    """skip_busy=False 保持原语义：等推理锁，再卸载。"""
    manager = _manager(tmp_path)
    tts = _register(manager, "tts", FakeModel())
    entered = asyncio.Event()
    release = asyncio.Event()

    async def busy() -> None:
        async with tts.lock:
            entered.set()
            await release.wait()

    holder = asyncio.create_task(busy())
    await entered.wait()
    unload = asyncio.create_task(manager.unload_models(["tts"]))
    await asyncio.sleep(0.1)
    assert not unload.done(), "默认语义应当等锁，不能提前返回"
    assert tts.unload_calls == 0
    release.set()
    result = await asyncio.wait_for(unload, timeout=3.0)
    await holder
    assert result["released"] == ["tts"]
    assert result["skipped"] == []
    assert tts.unload_calls == 1


@pytest.mark.asyncio
async def test_unload_all_reports_released_and_skipped(tmp_path: Path) -> None:
    """全部卸载：只报实际卸下的与忙的；未加载的名字两边都不出现。"""
    manager = _manager(tmp_path)
    tts = _register(manager, "tts", FakeModel())
    asr = _register(manager, "asr", FakeModel())
    speaker = _register(manager, "speaker", FakeModel(loaded=False))
    manager.begin_use("tts")
    try:
        result = await manager.unload_models(None, unload_all=True, skip_busy=True)
    finally:
        manager.end_use("tts")
    assert result["ok"] is True
    assert result["released"] == ["asr"]
    assert result["skipped"] == ["tts"]
    assert result["free_mb"] == 4096
    assert result["total_mb"] == 8192
    assert asr.unload_calls == 1 and asr.loaded is False
    assert tts.unload_calls == 0 and tts.loaded is True
    assert speaker.unload_calls == 0


def test_route_forwards_skip_busy_and_returns_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """路由把请求体的 skip_busy 透传给显存管家，并原样带出 skipped。"""
    from localai.api.app import app

    calls: dict[str, object] = {}

    class FakeVram:
        async def unload_models(
            self,
            names=None,
            *,
            unload_all: bool = False,
            skip_busy: bool = False,
        ) -> dict[str, object]:
            calls["names"] = names
            calls["unload_all"] = unload_all
            calls["skip_busy"] = skip_busy
            return {
                "ok": True,
                "released": ["asr"],
                "skipped": ["tts"],
                "free_mb": 7000,
                "total_mb": 8192,
            }

    monkeypatch.setattr(app.state, "vram", FakeVram())
    with TestClient(app) as client:
        response = client.post("/vram/unload", json={"all": True})
    assert response.status_code == 200
    assert response.json()["released"] == ["asr"]
    assert response.json()["skipped"] == ["tts"]
    # 请求体没写 skip_busy 时按模型默认值 True 透传。
    assert calls == {"names": None, "unload_all": True, "skip_busy": True}
