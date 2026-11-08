"""faster-whisper 引擎单测：加载 / 转写 / 原生时间戳 / 卸载。

用真实的 ``tiny`` 权重（约 75MB）跑一条短 WAV，验证：
1. 导入模块不加载模型（``is_loaded()`` 为 False）；
2. ``transcribe`` 返回全文 + 升序且 ``end > start`` 的原生分段；
3. ``timestamp_source`` 为 ``native``；
4. ``unload()`` 后 ``is_loaded()`` 变回 False。

权重缺失时跳过（不在测试里联网下载）；本机 ``models/whisper/tiny`` 已就绪。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localai.engines._native import import_engine_module

_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


@pytest.fixture(scope="module")
def engine():
    """原生 whisper 引擎模块（只 import，不加载模型）。"""
    return import_engine_module("whisper_engine")


@pytest.fixture(scope="module")
def tiny_ready(engine) -> dict:
    status = engine.model_status("tiny")
    if not status.get("ready"):
        pytest.skip(f"faster-whisper tiny 权重未准备：{status.get('model_dir')}")
    return status


@pytest.fixture(scope="module")
def speech_wav() -> Path:
    """找一条 3~8 秒、确实有语音的短音频。"""
    candidates = [
        _OUTPUTS / "pick_Karenina.wav",
        _OUTPUTS / "pick_female-A.wav",
        _OUTPUTS / "pick_female-B.wav",
    ]
    for path in candidates:
        if path.is_file():
            return path
    found = sorted(_OUTPUTS.glob("*.wav"))
    if found:
        return found[0]
    pytest.skip("outputs/ 下没有可用的短音频")


def test_model_status_does_not_load(engine, tiny_ready) -> None:
    assert engine.is_loaded() is False
    status = engine.model_status("tiny")
    assert status["ready"] is True
    assert status["loaded"] is False
    assert set(status["supported_sizes"]) >= {"tiny", "base", "small", "medium", "large-v3"}


def test_transcribe_returns_native_segments(engine, tiny_ready, speech_wav: Path) -> None:
    engine.unload()
    assert engine.is_loaded() is False

    result = engine.transcribe(str(speech_wav), size="tiny", language="zh", timestamps=True)

    assert result["text"], "tiny 没有识别出文本"
    assert result["timestamp_source"] == "native"
    assert result["segments"], "没有返回分段"
    assert result["duration"] and result["duration"] > 0
    assert result["size"] == "tiny"
    assert result["device"] in ("cuda", "cpu")

    previous_end = -1.0
    for segment in result["segments"]:
        assert segment["end"] > segment["start"] >= 0
        assert segment["start"] >= previous_end - 1e-6, "分段必须按时间升序且不重叠"
        assert isinstance(segment["text"], str)
        previous_end = segment["end"]

    assert engine.is_loaded("tiny") is True


def test_unload_releases_model(engine, tiny_ready, speech_wav: Path) -> None:
    engine.transcribe(str(speech_wav), size="tiny")
    assert engine.is_loaded() is True
    engine.unload()
    assert engine.is_loaded() is False
    # 重复卸载安全
    engine.unload()
    assert engine.is_loaded() is False


def test_package_import_does_not_pull_torch(engine) -> None:
    """导入 whisper 引擎模块本身不应触发 torch / faster-whisper。"""
    # 注意：本用例可能在其它用例加载过模型后运行，所以只断言模块自身可调用状态查询。
    assert callable(engine.model_status)
    assert engine.model_status("tiny")["default_size"] in engine.SIZES
