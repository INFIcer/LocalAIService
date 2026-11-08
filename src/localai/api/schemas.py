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


class TranslateRequest(BaseModel):
    """POST /translate：单条文本翻译。

    长度/语言等业务校验在路由里做，保证超限返回 400 而不是 422。
    """

    text: str = Field(..., description="要翻译的文本（上限 5000 字）")
    source: str = Field("auto", description="auto / zh / en / zho_Hans ...；auto 自动检测")
    target: str = Field("en", description="目标语言：en / ja / eng_Latn ...")
    engine: Optional[str] = Field("nllb", description="翻译引擎；目前只支持 nllb")
    max_len: int = Field(400, ge=32, le=1024, description="每块最大 token 数")
    batch_size: int = Field(8, ge=1, le=64, description="内部推理批大小")


class TranslateBatchRequest(BaseModel):
    """POST /translate/batch：一批文本翻译（内部批量推理）。"""

    texts: list[str] = Field(..., description="待翻译文本列表（上限 500 条 / 20 万字）")
    source: str = Field("auto", description="auto / zh / en / zho_Hans ...")
    target: str = Field("en", description="目标语言")
    engine: Optional[str] = Field("nllb", description="翻译引擎；目前只支持 nllb")
    max_len: int = Field(400, ge=32, le=1024)
    batch_size: int = Field(8, ge=1, le=64)


class JobTranslateRequest(TranslateBatchRequest):
    """POST /jobs/translate：超大批量进任务队列。"""

    priority: int = 0
