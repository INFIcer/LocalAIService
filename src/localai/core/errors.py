"""服务层统一异常。

对照 HTTP 状态码：

======================  ======  ==========================================
异常                     状态码  含义
======================  ======  ==========================================
VoiceNotFound           404     请求的音色不存在
InsufficientVramError   507     显存不足（已按 LRU 腾过，仍不够）
EngineBackendError      503     TTS 引擎加载/推理失败（部分参数错误为 400）
AsrBackendError         503     ASR 引擎加载/识别失败
TranslateBackendError   503     翻译引擎加载/推理失败（语言参数错误为 400）
SpeakerBackendError     503     声纹引擎加载/计算失败（参数错误为 400）
SeparateBackendError    503     人声分离引擎加载/推理失败
ComfyError              502/504 ComfyUI 上游错误 / 等待超时
JobFailedError          500     任务失败且没有可复用的原始异常
JobTimeoutError         504     任务执行超时
JobCanceledError        409     任务被取消
======================  ======  ==========================================
"""

from __future__ import annotations

from typing import Any, Optional


class LocalAIError(Exception):
    """本服务所有业务异常的基类。"""

    status_code = 500
    error_code = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class ComfyError(LocalAIError):
    """ComfyUI 侧错误（仅 TTS_BACKEND=comfy 时可能出现）。"""

    error_code = "comfy_error"

    def __init__(
        self,
        message: str,
        *,
        comfy_error: Optional[str] = None,
        messages: Optional[list[Any]] = None,
        prompt_id: Optional[str] = None,
        status_code: int = 502,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.comfy_error = comfy_error
        self.messages = messages or []
        self.prompt_id = prompt_id


class VoiceNotFound(LocalAIError):
    """请求的音色不存在。"""

    status_code = 404
    error_code = "voice_not_found"


class InsufficientVramError(LocalAIError):
    """显存不够：由全局异常处理器转成 HTTP 507 的明确 JSON。"""

    status_code = 507
    error_code = "insufficient_vram"

    def __init__(self, result: dict[str, Any], message: Optional[str] = None) -> None:
        self.result = result
        super().__init__(message or result.get("reason") or "显存不足")


class EngineBackendError(LocalAIError):
    """TTS 引擎加载/推理失败。"""

    status_code = 503
    error_code = "engine_error"


class AsrBackendError(LocalAIError):
    """ASR 引擎加载/识别失败。"""

    status_code = 503
    error_code = "asr_error"


class TranslateBackendError(LocalAIError):
    """翻译引擎加载/翻译失败；语言码不合法时用 400。"""

    status_code = 503
    error_code = "translate_error"


class SpeakerBackendError(LocalAIError):
    """声纹引擎加载/计算失败。"""

    status_code = 503
    error_code = "speaker_error"


class SeparateBackendError(LocalAIError):
    """人声分离引擎加载/推理失败。"""

    status_code = 503
    error_code = "separate_error"


class VisionBackendError(LocalAIError):
    """视觉（VLM）引擎加载/推理失败。"""

    status_code = 503
    error_code = "vision_error"


# ---------------------------------------------------------------------------
# 任务队列（localai.jobs）
# ---------------------------------------------------------------------------

class JobError(LocalAIError):
    """任务队列相关错误的基类。"""

    status_code = 500
    error_code = "job_error"


class JobFailedError(JobError):
    """任务失败，且原始异常已无法恢复（例如服务重启后读取历史记录）。"""

    status_code = 500
    error_code = "job_failed"


class JobTimeoutError(JobError):
    """任务执行超过 ``LOCALAI_JOB_TIMEOUT``。"""

    status_code = 504
    error_code = "job_timeout"


class JobCanceledError(JobError):
    """同步接口等待的任务被取消。"""

    status_code = 409
    error_code = "job_canceled"


def is_cuda_oom(exc: BaseException) -> bool:
    """粗判异常是不是 CUDA 显存不足；只用于把错误翻译成 507。"""
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        or "allocation on device" in text
        or "cublas_status_alloc_failed" in text
    )
