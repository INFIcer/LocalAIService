"""关键帧挑选代理的单元测试：全部用假 VLM / 假媒体，不加载任何模型。

覆盖：
- agent 协议（extract -> final）能终止；
- 模型输出带 markdown 围栏也能解析；
- 补抽到 max_rounds 强制结束；
- hybrid 分数不足时会补抽；
- 感知哈希去重 + 亮度/方差空白预筛；
- VLM 故障时 degraded 回退、进程不崩。
"""

from __future__ import annotations

import asyncio

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from PIL import Image

from localai.config import Config
from localai.core.frames_agent import FramesAgent, parse_json_loose


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class FakeMedia:
    """按时间点生成图片；1s 纯白、5/9s 同一张噪声、13s 另一张噪声。"""

    def __init__(self) -> None:
        self.extracted: list[float] = []

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "probe":
            return {"duration": 60.0, "video": {"codec": "h264"}}
        if name == "detect_scenes":
            return {"scenes": [{"time": 20.0}, {"time": 40.0}], "duration": 60.0}
        if name == "extract_frames":
            out_dir = Path(args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for time_sec in kwargs.get("times") or []:
                point = round(float(time_sec), 3)
                path = out_dir / f"frame_t{point:07.2f}.jpg"
                _make_image(path, point)
                frames.append(
                    {"path": str(path), "time_sec": point, "width": 768, "height": 432}
                )
                self.extracted.append(point)
            return frames
        raise AssertionError(f"FakeMedia 不认识：{name}")


def _make_image(path: Path, time_sec: float) -> None:
    if time_sec == 1.0:
        image = Image.new("RGB", (96, 54), (255, 255, 255))
    elif time_sec in (5.0, 9.0):
        array = np.random.default_rng(7).integers(0, 256, (54, 96, 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
    elif time_sec == 13.0:
        array = np.random.default_rng(99).integers(0, 256, (54, 96, 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
    else:
        # 其余时间点用可区分的高方差图案（避免被误判 blank/重复）。
        array = np.random.default_rng(int(time_sec * 100)).integers(
            0, 256, (54, 96, 3), dtype=np.uint8
        )
        image = Image.fromarray(array, mode="RGB")
    image.save(path, format="JPEG", quality=90)


class FakeVision:
    """可配置的假 VLM：按图片文件名里的时间点返回评分。"""

    def __init__(
        self,
        score_fn: Any,
        agent_replies: Optional[list[Any]] = None,
        plan: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.score_fn = score_fn
        self.agent_replies = list(agent_replies or [])
        self.plan = plan or []
        self.image_calls = 0
        self.text_calls = 0

    @staticmethod
    def _time_of(image_path: str) -> float:
        match = re.search(r"t(\d+\.\d+)", Path(image_path).name)
        return float(match.group(1)) if match else 0.0

    async def describe_image(self, image_path: str, prompt: str, **_: Any) -> dict[str, Any]:
        self.image_calls += 1
        point = self._time_of(image_path)
        result = self.score_fn(point, self.image_calls)
        if isinstance(result, Exception):
            raise result
        return {"text": json.dumps(result, ensure_ascii=False), "elapsed_ms": 1}

    async def describe_text(self, prompt: str, *, system_prompt: str = "", **_: Any) -> dict[str, Any]:
        self.text_calls += 1
        if "规划器" in (system_prompt or ""):
            return {"text": json.dumps({"candidates": self.plan}, ensure_ascii=False)}
        if not self.agent_replies:
            return {"text": json.dumps({"action": "final", "frames": []}, ensure_ascii=False)}
        reply = self.agent_replies.pop(0)
        if isinstance(reply, str):
            return {"text": reply}
        return {"text": json.dumps(reply, ensure_ascii=False)}


class FakeRegistry:
    def __init__(self, media: Any = None, vision: Any = None) -> None:
        self.media = media
        self.vision = vision

    def get(self, name: str) -> Any:
        if name == "media":
            return self.media
        if name == "vision":
            return self.vision
        raise KeyError(name)

    def find(self, name: str) -> Any:
        try:
            return self.get(name)
        except KeyError:
            return None


def _score(usable: bool, score: int, kind: str = "chart") -> dict[str, Any]:
    return {
        "usable": usable,
        "kind": kind,
        "score": score,
        "ocr_text": "",
        "summary": f"评分{score}",
        "why": "测试",
    }


def _make_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vision: Any,
    media: Any = None,
) -> FramesAgent:
    for name in list(__import__("os").environ):
        if name.startswith(("LOCALAI_", "INDEX_TTS_")):
            monkeypatch.delenv(name, raising=False)
    cfg = Config.from_env(tmp_path)
    cfg.vision_planner = "hybrid"
    media = media or FakeMedia()
    registry = FakeRegistry(media=media, vision=vision)
    return FramesAgent(cfg, registry, enable_asr=False)


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

def test_parse_json_with_markdown_fence() -> None:
    text = '这是结果：\n```json\n{"action":"extract","times":[1.5,2.5],}\n```\n'
    data = parse_json_loose(text)
    assert data is not None
    assert data["action"] == "extract"
    assert data["times"] == [1.5, 2.5]


def test_parse_json_invalid_returns_none() -> None:
    assert parse_json_loose("完全没有 JSON") is None
    assert parse_json_loose("{不是 JSON}") is None


# ---------------------------------------------------------------------------
# 预筛：blank + 感知哈希去重
# ---------------------------------------------------------------------------

def test_prefilter_blank_and_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = FakeMedia()
    agent = _make_agent(tmp_path, monkeypatch, FakeVision(lambda t, n: _score(True, 5)), media)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")
    stats: dict[str, Any] = {"candidates": 0, "vlm_calls": 0, "blank_skipped": 0, "duplicates_skipped": 0}
    batch = asyncio.run(agent._extract_and_screen(
        media, video, tmp_path / "frames", [1.0, 5.0, 9.0, 13.0], stats, []
    ))
    assert stats["candidates"] == 4
    assert stats["blank_skipped"] == 1
    assert stats["duplicates_skipped"] == 1
    prefiltered = {item["time"]: item.get("prefiltered") for item in batch}
    assert prefiltered[1.0] == "blank"
    assert prefiltered[9.0] == "duplicate"
    assert prefiltered[5.0] is None
    assert prefiltered[13.0] is None


# ---------------------------------------------------------------------------
# hybrid：分数不足时补抽
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_supplements_until_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 前 8 次评分故意给低分，迫使它进入补抽；之后高分。
    vision = FakeVision(
        lambda t, n: _score(False, 1) if n <= 8 else _score(True, 5)
    )
    media = FakeMedia()
    agent = _make_agent(tmp_path, monkeypatch, vision, media)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video,
        target_frames=2,
        planner="hybrid",
        max_rounds=4,
        subtitles=[{"start": 10.0, "end": 12.0, "text": "参数对比"}],
        scene_times=[20.0, 40.0],
    )
    assert result["ok"] is True
    assert len(result["frames"]) == 2
    assert result["rounds_used"] >= 2
    assert result["stats"]["candidates"] > 8
    assert result["stats"]["vlm_calls"] > 8
    assert "degraded" not in result


# ---------------------------------------------------------------------------
# agent 协议：extract -> final 能终止
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_protocol_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vision = FakeVision(
        lambda t, n: _score(True, 5) if abs(t - 57.0) < 1.0 else _score(False, 1),
        agent_replies=[
            {"action": "extract", "times": [57.0], "reason": "结论页"},
            {"action": "final", "frames": [{"time": 57.0, "why": "结论图"}], "strategy": "测试"},
        ],
        plan=[{"time": 57.0, "priority": 1, "reason": "结论"}],
    )
    media = FakeMedia()
    agent = _make_agent(tmp_path, monkeypatch, vision, media)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video,
        target_frames=1,
        planner="agent",
        max_rounds=4,
        subtitles=[{"start": 56.0, "end": 59.0, "text": "总结"}],
        scene_times=[30.0],
    )
    assert result["ok"] is True
    assert result["frames"], "agent 的 final 应至少返回一张"
    assert result["rounds_used"] <= 4
    assert vision.text_calls >= 1


@pytest.mark.asyncio
async def test_agent_stops_at_max_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = {"value": 0}

    def next_extract(*_args: Any) -> dict[str, Any]:
        counter["value"] += 1
        return {"action": "extract", "times": [3.0 + counter["value"] * 5], "reason": "继续"}

    vision = FakeVision(
        lambda t, n: _score(False, 0),
        agent_replies=[next_extract, next_extract, next_extract, next_extract],
    )
    media = FakeMedia()
    agent = _make_agent(tmp_path, monkeypatch, vision, media)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video,
        target_frames=3,
        planner="agent",
        max_rounds=2,
        subtitles=[],
        scene_times=[20.0],
    )
    assert result["ok"] is True
    assert result["rounds_used"] <= 2
    # 全低分时应走降级回退，而不是死循环。
    assert result.get("degraded") is True
    assert vision.text_calls <= 3


# ---------------------------------------------------------------------------
# VLM 故障：degraded 回退
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vlm_failure_degrades_without_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_time: float, _n: int) -> Exception:
        return RuntimeError("幻觉引擎掉线")

    vision = FakeVision(boom)
    media = FakeMedia()
    agent = _make_agent(tmp_path, monkeypatch, vision, media)
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video,
        target_frames=2,
        planner="hybrid",
        max_rounds=3,
        subtitles=[],
        scene_times=[20.0],
    )
    assert result["ok"] is True
    assert result.get("degraded") is True
    assert result["frames"], "VLM 全挂时仍应返回方差筛选出的帧"
    assert all(item.get("degraded") for item in result["frames"])
    assert result.get("degraded_reason")