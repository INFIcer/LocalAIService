"""超长视频覆盖硬规则测试：8000 秒、target=3、effort=auto。

用假 VLM / 假抽帧验证（不读大文件、不加载模型）：

- 超长片自动落 high 覆盖策略，K=24；
- 每段先抽过 1 次之前，绝不从同段抽第 2 次；
- 最终 3 帧落在 3 个不同段，首尾相距 ≥ 1/3 时长；
- 空白段自动换时间点重抽，3 次仍失败记进 skipped_segments。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from localai.config import Config
from localai.core.frames_agent import FramesAgent, segment_index

DURATION = 8000.0
K = 24


class FakeMedia:
    """随机图案帧；可指定某段一直空白，或仅第一次空白。"""

    def __init__(
        self,
        *,
        blank_all: set[int] | None = None,
        blank_first: set[int] | None = None,
    ) -> None:
        self.blank_all = set(blank_all or set())
        self.blank_first = set(blank_first or set())
        self.extracted: list[float] = []
        self.events: list[int] = []
        self.segment_counts: dict[int, int] = {}

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "probe":
            return {"duration": DURATION, "video": {"codec": "h264"}}
        if name == "detect_scenes":
            return {"scenes": [], "duration": DURATION}
        if name == "extract_frames":
            out_dir = Path(args[1])
            out_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for time_sec in kwargs.get("times") or []:
                point = round(float(time_sec), 3)
                index = segment_index(point, DURATION, K)
                self.events.append(index)
                count = self.segment_counts.get(index, 0)
                self.segment_counts[index] = count + 1
                blank = index in self.blank_all or (
                    index in self.blank_first and count == 0
                )
                path = out_dir / f"frame_{point:09.3f}.jpg"
                _make_image(path, point, blank=blank)
                self.extracted.append(point)
                frames.append(
                    {"path": str(path), "time_sec": point, "width": 96, "height": 54}
                )
            return frames
        raise AssertionError(f"FakeMedia 不认识：{name}")


class FakeVision:
    def __init__(self) -> None:
        self.image_calls = 0

    async def describe_image(self, image_path: str, prompt: str, **_: Any) -> dict[str, Any]:
        import re

        self.image_calls += 1
        match = re.search(r"(\d+\.\d+)", Path(image_path).stem)
        point = float(match.group(1)) if match else 0.0
        return {
            "text": json.dumps(
                {
                    "usable": True,
                    "kind": "scene",
                    "score": 5,
                    "ocr_text": "",
                    "summary": f"第 {point:.0f} 秒的有信息画面",
                    "why": "覆盖测试",
                },
                ensure_ascii=False,
            )
        }


class FakeRegistry:
    def __init__(self, media: Any, vision: Any) -> None:
        self.media = media
        self.vision = vision

    def get(self, name: str) -> Any:
        if name == "media":
            return self.media
        if name == "vision":
            return self.vision
        raise KeyError(name)


def _make_image(path: Path, time_sec: float, *, blank: bool = False) -> None:
    if blank:
        image = Image.new("RGB", (96, 54), (255, 255, 255))
    else:
        seed = int(time_sec * 100) % (2**32 - 1)
        array = np.random.default_rng(seed).integers(0, 256, (54, 96, 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
    image.save(path, format="JPEG", quality=90)


def _agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, media: FakeMedia) -> FramesAgent:
    for name in list(__import__("os").environ):
        if name.startswith(("LOCALAI_", "INDEX_TTS_")):
            monkeypatch.delenv(name, raising=False)
    cfg = Config.from_env(tmp_path)
    cfg.home = tmp_path
    cfg.output_dir = tmp_path / "outputs"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.vision_planner = "hybrid"
    return FramesAgent(
        cfg,
        FakeRegistry(media, FakeVision()),
        enable_asr=False,
    )


@pytest.mark.asyncio
async def test_long_video_auto_covers_all_segments_before_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = FakeMedia()
    agent = _agent(tmp_path, monkeypatch, media)
    video = tmp_path / "long.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video,
        target_frames=3,
        effort="auto",
        subtitles=[],
        scene_times=[],
    )
    assert result["effort"] == "auto"
    assert result["effort_profile"] == "high"
    assert result["coverage_segments"] == K
    assert result["budget"]["coverage_segments"] == K

    frames = result["frames"]
    assert len(frames) == 3
    segments = [segment_index(item["time"], DURATION, K) for item in frames]
    assert len(set(segments)) == 3, f"3 帧应来自 3 个不同段：{segments}"
    span = max(item["time"] for item in frames) - min(item["time"] for item in frames)
    assert span >= DURATION / 3, f"首尾跨度不足：{span}"

    # 每段都实际抽过；且第一次出现「同段第二抽」之前，24 段已全部覆盖。
    assert all(item["extracted"] >= 1 for item in result["segments_seen"])
    seen: set[int] = set()
    for index in media.events:
        if index in seen:
            break
        seen.add(index)
    assert len(seen) == K, f"出现重复段时只覆盖了 {len(seen)}/{K} 段"
    assert result["skipped_segments"] == []
    assert result["uncovered_segments"] == []


@pytest.mark.asyncio
async def test_blank_segment_retries_same_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = FakeMedia(blank_first={0})
    agent = _agent(tmp_path, monkeypatch, media)
    video = tmp_path / "long.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video, target_frames=3, effort="auto", subtitles=[], scene_times=[]
    )
    assert result["segments_seen"][0]["extracted"] >= 2, "空白段应换时间点重抽"
    assert 0 not in result["skipped_segments"]
    assert len(result["frames"]) == 3


@pytest.mark.asyncio
async def test_blank_segment_records_skipped_after_three_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = FakeMedia(blank_all={5})
    agent = _agent(tmp_path, monkeypatch, media)
    video = tmp_path / "long.mp4"
    video.write_bytes(b"fake")

    result = await agent.run(
        video, target_frames=3, effort="auto", subtitles=[], scene_times=[]
    )
    assert 5 in result["skipped_segments"]
    # 首轮 1 次 + 重抽 3 次 = 4 次，之后不再无限重试。
    assert result["segments_seen"][5]["extracted"] == 4
    assert len(result["frames"]) == 3
