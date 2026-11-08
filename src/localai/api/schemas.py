"""HTTP 请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..constants import TTS_MAX_TEXT


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=TTS_MAX_TEXT, description="要合成的文本")
    voice: str = Field("default", description="<HOME>/voices 下的音色名（文件名去扩展名）")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="语速，1.0 正常；范围 0.5~2.0")
    emotion: Optional[str] = Field(None, description="可选：8 维情感向量或情感文本")
    seed: Optional[int] = Field(None, ge=0, le=4294967295, description="随机种子；不填随机")
    language: Optional[str] = Field(None, description="zh/en/ja/es/ar；不填自动判断")


class SpeechRequest(BaseModel):
    """OpenAI 兼容的 /v1/audio/speech 请求体。"""

    model: str = Field("indextts")
    input: str = Field(..., min_length=1, max_length=TTS_MAX_TEXT)
    voice: str = Field("default")
    response_format: str = Field("wav")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    language: Optional[str] = None
    emotion: Optional[str] = None
    seed: Optional[int] = Field(None, ge=0, le=4294967295)


class CompatTaskRequest(BaseModel):
    """DSH 插件 /api/v1/tts/tasks 的请求体；多余字段一律忽略。"""

    text: str = ""
    prompt_audio: str = ""
    speed: float = 1.0
    seed: Optional[int] = None
    language: Optional[str] = None
    model_config = {"extra": "allow"}


class VramUnloadRequest(BaseModel):
    """POST /vram/unload 的请求体。"""

    models: list[str] = Field(default_factory=list)
    all: bool = False


class JobCreateRequest(BaseModel):
    """POST /jobs 的通用请求体。"""

    kind: str = Field(..., description="tts / asr / speaker / separate / media")
    action: Optional[str] = Field(None, description="不填则用该 kind 的默认 action")
    priority: int = Field(0, description="越大越先；等待久了会自动提权")
    payload: dict[str, Any] = Field(default_factory=dict, description="能力参数")
    meta: Optional[dict[str, Any]] = Field(None, description="预留扩展字段")


class JobTtsRequest(TTSRequest):
    """POST /jobs/tts：TTS 参数 + 优先级。"""

    priority: int = 0
