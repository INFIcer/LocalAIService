"""能力适配器集合。

新增能力时只需要：
1. 在 ``localai/engines/`` 下加一个 ``BaseAdapter`` 子类；
2. 在 :func:`build_registry` 里注册；
3. 在 ``localai/api/routers/`` 下加路由（如果它需要 HTTP 接口）。

主程序、显存管家、空闲看门狗、``/capabilities`` 都不需要改。
"""

from __future__ import annotations

from typing import Optional

from ..config import Config, get_config
from ..core.registry import EngineRegistry
from .asr_fasterwhisper import WhisperAdapter
from .asr_sensevoice import AsrAdapter
from .base import BaseAdapter
from .media_ffmpeg import MediaAdapter
from .separate_demucs import SeparateAdapter
from .speaker_campplus import SpeakerAdapter
from .tts_indextts import TtsAdapter

__all__ = [
    "BaseAdapter",
    "TtsAdapter",
    "AsrAdapter",
    "WhisperAdapter",
    "SpeakerAdapter",
    "SeparateAdapter",
    "MediaAdapter",
    "build_registry",
]


def build_registry(cfg: Optional[Config] = None) -> EngineRegistry:
    """构造本服务支持的全部能力（只建对象，不 import 原生引擎、不加载模型）。"""
    cfg = cfg or get_config()
    registry = EngineRegistry()
    # 注册顺序决定 /vram 里模型的展示顺序。
    registry.register(TtsAdapter(cfg))
    registry.register(AsrAdapter(cfg))
    registry.register(WhisperAdapter(cfg))
    registry.register(SpeakerAdapter(cfg))
    registry.register(SeparateAdapter(cfg))
    registry.register(MediaAdapter(cfg))
    return registry
