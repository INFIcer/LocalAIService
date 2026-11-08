"""按能力拆分的 FastAPI 路由。"""

from __future__ import annotations

from . import asr, media, separate, speakers, system, tts

__all__ = ["system", "tts", "asr", "speakers", "separate", "media"]
