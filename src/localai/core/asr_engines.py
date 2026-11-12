"""识别引擎选择：把请求里的 ``engine`` 参数解析成具体适配器。

两个引擎的服务层名字：
- ``sensevoice`` -> 注册表里的 ``asr``（SenseVoiceSmall，默认）
- ``whisper``    -> 注册表里的 ``whisper``（faster-whisper，原生时间戳）

约定
----
- 请求不带 ``engine``（或空字符串）= **用配置默认值**（``LOCALAI_ASR_ENGINE``）；
- ``engine=auto``：**需要时间戳**时用 whisper（原生 segment 时间戳），
  否则用配置默认值；
- 其它取值一律 400，避免拼写错误悄悄换成别的引擎。
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..constants import ASR_ENGINE_CHOICES, ASR_ENGINES

#: 服务层引擎名 -> 注册表能力名。
ADAPTER_NAMES = {
    "sensevoice": "asr",
    "whisper": "whisper",
}


class EngineChoiceError(ValueError):
    """``engine`` 取值不合法。"""


def normalize_engine(
    value: Any,
    cfg: Config,
    *,
    need_timestamps: bool = False,
) -> str:
    """解析请求里的 engine，返回 ``sensevoice`` / ``whisper``。

    参数
    ----
    value: 请求里的原始值（``None`` / ``""`` = 未指定）。
    cfg:   当前配置，提供默认引擎。
    need_timestamps: 本次请求是否要时间戳；``auto`` 据此选 whisper。
    """
    raw = str(value or "").strip().lower()
    if raw in ("", "default"):
        return cfg.asr_engine
    if raw == "auto":
        return "whisper" if need_timestamps else cfg.asr_engine
    if raw in ASR_ENGINES:
        return raw
    raise EngineChoiceError(
        f"不支持的 engine={value!r}，可选：{', '.join(ASR_ENGINE_CHOICES)}"
    )


def adapter_for(registry: Any, engine: str) -> Any:
    """按服务层引擎名取适配器。"""
    name = ADAPTER_NAMES.get(engine)
    if name is None:
        raise EngineChoiceError(
            f"不支持的 engine={engine!r}，可选：{', '.join(ASR_ENGINE_CHOICES)}"
        )
    return registry.get(name)
