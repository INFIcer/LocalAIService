"""HTTP 请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..constants import TTS_MAX_TEXT


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=TTS_MAX_TEXT, description="要合成的文本")
    voice: Optional[str] = Field(
        None, description="<HOME>/voices 下的音色名（文件名去扩展名）；不填用服务端默认音色"
    )
    speed: float = Field(1.0, ge=0.5, le=2.0, description="语速，1.0 正常；范围 0.5~2.0")
    emotion: Optional[str] = Field(None, description="可选：8 维情感向量或情感文本")
    seed: Optional[int] = Field(None, ge=0, le=4294967295, description="随机种子；不填随机")
    language: Optional[str] = Field(None, description="zh/en/ja/es/ar；不填自动判断")


class SpeechRequest(BaseModel):
    """OpenAI 兼容的 /v1/audio/speech 请求体。"""

    model: str = Field("indextts")
    input: str = Field(..., min_length=1, max_length=TTS_MAX_TEXT)
    voice: Optional[str] = Field(None, description="不填用服务端默认音色")
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

    kind: str = Field(..., description="tts / asr / separate / media")
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
    max_len: Optional[int] = Field(
        None, ge=32, le=1024, description="每块最大 token 数；不填用服务端默认值"
    )
    batch_size: Optional[int] = Field(
        None, ge=1, le=64, description="内部推理批大小；不填用服务端默认值"
    )


class TranslateBatchRequest(BaseModel):
    """POST /translate/batch：一批文本翻译（内部批量推理）。"""

    texts: list[str] = Field(..., description="待翻译文本列表（上限 500 条 / 20 万字）")
    source: str = Field("auto", description="auto / zh / en / zho_Hans ...")
    target: str = Field("en", description="目标语言")
    engine: Optional[str] = Field("nllb", description="翻译引擎；目前只支持 nllb")
    max_len: Optional[int] = Field(None, ge=32, le=1024)
    batch_size: Optional[int] = Field(None, ge=1, le=64)


class JobTranslateRequest(TranslateBatchRequest):
    """POST /jobs/translate：超大批量进任务队列。"""

    priority: int = 0


class MediaDownloadRequest(BaseModel):
    """POST /media/download：提交一个异步下载任务。"""

    url: str = Field(..., description="http/https 链接（B站 / 一般站点 / 直链）")
    quality: str = Field("720p", description="720p / best / audio")
    subtitles: bool = Field(True, description="是否尝试下载站点字幕（下不到不报错）")
    cookies: Optional[str] = Field(
        None, description="可选：Netscape cookie 文件路径；不填时 B站自动读取本机凭证"
    )
    priority: int = Field(0, description="优先级，越大越先")


class MediaAnalyzeRequest(BaseModel):
    """POST /media/analyze：下载 + 字幕/ASR + 场景边界。"""

    url: str = Field(..., description="http/https 链接")
    engine: str = Field("auto", description="whisper / sensevoice / auto（站点无字幕时用）")
    max_frames: int = Field(0, ge=0, le=200, description="0=不抽帧；>0 均匀+场景吸附抽帧")
    quality: str = Field("720p", description="720p / best / audio")
    subtitles: bool | str = Field(
        True,
        description=(
            "true=下载站点字幕（旧行为）；false=不下载；"
            "str \"auto\"=下载后先探测内嵌/外挂软字幕，没有再 ASR"
        ),
    )
    filter: str = Field(
        "",
        description="可选 vision：分析完直接让 VLM 挑关键帧（不传保持旧行为）",
    )
    target_frames: int = Field(
        3, ge=1, le=10, description="filter=vision 时挑几张关键帧"
    )
    effort: str = Field(
        "medium", description="关键帧分析力度：low / medium / high（filter=vision 时生效）"
    )


class JobAnalyzeRequest(MediaAnalyzeRequest):
    """POST /jobs/analyze：异步链接分析。"""

    priority: int = 0


class VisionFramesRequest(BaseModel):
    """POST /vision/frames：按字幕 + 时长循环抽帧。

    ``video_path`` 与 ``job_id`` 二选一：前者是服务端本地路径，后者先查任务
    结果里的 ``path``（例如刚跑完的下载 / 分析任务）。
    """

    video_path: Optional[str] = Field(None, description="服务端本地的视频路径")
    job_id: Optional[str] = Field(None, description="任务 id（结果里带 path 时用它）")
    target_frames: int = Field(3, ge=1, le=10, description="要凑够几张有效关键帧")
    planner: str = Field(
        "hybrid", description="hybrid=逐帧评分补抽（默认）/ agent=多轮代理协议"
    )
    effort: str = Field(
        "auto",
        description=(
            "分析力度 auto（默认，按时长定档并加密分段）/ low / medium / high："
            "决定候选数、轮数、VLM 与 clip 预算"
        ),
    )
    max_rounds: int = Field(6, ge=1, le=12, description="最多补抽轮数（会被 effort 上限收紧）")
    subtitles: Optional[list[dict[str, Any]] | str] = Field(
        None,
        description=(
            "可选：[{start,end,text}] 或字幕文件路径；"
            "传字符串 \"auto\" 时先走 /media/subtitles，有软字幕就不跑 ASR"
        ),
    )
    scene_times: Optional[list[float]] = Field(
        None, description="可选：已知场景边界（秒）；不给就现场检测"
    )


class JobVisionRequest(VisionFramesRequest):
    """POST /jobs/vision：异步关键帧挑选。"""

    priority: int = 0


class VisionVideoRequest(BaseModel):
    """POST /vision/video：段级细看，只返回文字。

    ``start`` / ``end`` 都不给 = 看全片（自动降 fps，总帧数 ≤ 64）；
    给了就只切这一段（≤ 48 帧），用 ``clip`` 裁出来再一次性交给 VLM。
    """

    video_path: str = Field(..., description="服务端本地的视频路径")
    start: Optional[float] = Field(None, description="窗口起点（秒）；不给=全片开头")
    end: Optional[float] = Field(None, description="窗口终点（秒）；不给=全片结尾")
    fps: float = Field(1.0, gt=0.0, le=5.0, description="抽帧 fps（0.5~3；超上限会自动降）")
    question: Optional[str] = Field(
        None, description="可选：自定义问题，会附加在默认的「这段讲了什么」之后"
    )
