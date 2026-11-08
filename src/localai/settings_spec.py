"""管理界面可修改设置的「唯一清单」。

为什么单独一个模块
------------------
``config.py`` 在导入时就要读 ``config.json``，而管理接口又要把同一份白名单
以 JSON 形式返回给前端。如果规则写在路由里，配置文件加载和接口校验会各写
一遍，时间一久必然漂移。这里把「键名 / 类型 / 范围 / 默认值 / 环境变量 /
是否需要重启 / 中文说明」集中定义，两边都只引用它。

约定
----
- ``key`` 与 :class:`localai.config.Config` 的字段名保持一致；
- ``env`` 按「新变量在前、旧变量在后」排列，任意一个非空就算被环境变量覆盖；
- ``requires_restart`` 只表示「当前进程能立即用上新值，但真正生效还需要重启」，
  例如监听端口、设备、精度、后端实现；
- ``secret`` 为真时接口永远不回显真实值，只回显是否已设置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

#: 分组标题（顺序即前端展示顺序）。
GROUP_TITLES: dict[str, str] = {
    "service": "服务",
    "vram": "显存",
    "idle": "空闲卸载",
    "parallel": "并发",
    "tts": "TTS",
    "asr": "ASR",
    "translate": "翻译",
    "media": "媒体",
}


@dataclass(frozen=True, slots=True)
class Setting:
    """一个可管理配置项。"""

    key: str
    group: str
    title: str
    type: str
    default: Any
    env: tuple[str, ...] = ()
    description: str = ""
    min: Optional[int] = None
    max: Optional[int] = None
    choices: tuple[str, ...] = ()
    requires_restart: bool = False
    secret: bool = False
    #: 路径类设置允许空字符串表示「自动探测」。
    allow_empty: bool = False


#: 所有可写设置（顺序即接口返回顺序）。
SETTINGS: tuple[Setting, ...] = (
    # ---- 服务 ---------------------------------------------------------
    Setting(
        key="host",
        group="service",
        title="监听地址",
        type="str",
        default="127.0.0.1",
        env=("LOCALAI_HOST", "INDEX_TTS_HOST"),
        description="HTTP 监听地址。默认只监听回环地址；改成 0.0.0.0 会暴露到局域网。",
        requires_restart=True,
    ),
    Setting(
        key="port",
        group="service",
        title="监听端口",
        type="int",
        default=7880,
        min=1,
        max=65535,
        env=("LOCALAI_PORT", "INDEX_TTS_PORT"),
        description="HTTP 监听端口。所有调用方默认连 7880，改端口前请同步改客户端。",
        requires_restart=True,
    ),
    Setting(
        key="token",
        group="service",
        title="访问令牌",
        type="str",
        default="",
        env=("LOCALAI_TOKEN",),
        description="可选的 Bearer 令牌。留空表示不鉴权；设置后所有非公开接口都要求 Authorization: Bearer <token>。",
        requires_restart=True,
        secret=True,
        allow_empty=True,
    ),
    # ---- 显存 ---------------------------------------------------------
    Setting(
        key="vram_margin_mb",
        group="vram",
        title="显存安全余量",
        type="int",
        default=512,
        min=0,
        max=65536,
        env=("LOCALAI_VRAM_MARGIN_MB", "INDEX_TTS_VRAM_MARGIN_MB"),
        description="显存管家在估算可用空间时预留的余量（MB）。频繁 OOM 时可调大到 768/1024。",
    ),
    # ---- 空闲卸载 -----------------------------------------------------
    Setting(
        key="idle_unload",
        group="idle",
        title="TTS 空闲卸载",
        type="int",
        default=900,
        min=0,
        max=86400,
        env=("LOCALAI_IDLE_UNLOAD", "INDEX_TTS_IDLE_UNLOAD"),
        description="TTS 模型空闲多少秒后卸载；0 表示关闭空闲卸载。",
    ),
    Setting(
        key="asr_idle_unload",
        group="idle",
        title="SenseVoice 空闲卸载",
        type="int",
        default=900,
        min=0,
        max=86400,
        env=("LOCALAI_ASR_IDLE_UNLOAD", "INDEX_TTS_ASR_IDLE_UNLOAD"),
        description="SenseVoice 空闲多少秒后卸载；0 表示关闭。",
    ),
    Setting(
        key="whisper_idle_unload",
        group="idle",
        title="whisper 空闲卸载",
        type="int",
        default=900,
        min=0,
        max=86400,
        env=("LOCALAI_WHISPER_IDLE_UNLOAD",),
        description="faster-whisper 空闲多少秒后卸载；0 表示关闭。",
    ),
    Setting(
        key="speaker_idle_unload",
        group="idle",
        title="声纹空闲卸载",
        type="int",
        default=900,
        min=0,
        max=86400,
        env=("LOCALAI_SPEAKER_IDLE_UNLOAD", "INDEX_TTS_SPEAKER_IDLE_UNLOAD"),
        description="campplus 声纹模型空闲多少秒后卸载；0 表示关闭。",
    ),
    Setting(
        key="separate_idle_unload",
        group="idle",
        title="人声分离空闲卸载",
        type="int",
        default=900,
        min=0,
        max=86400,
        env=("LOCALAI_SEPARATE_IDLE_UNLOAD", "INDEX_TTS_SEPARATE_IDLE_UNLOAD"),
        description="Demucs 空闲多少秒后卸载；0 表示关闭。",
    ),
    Setting(
        key="translate_idle_unload",
        group="idle",
        title="翻译空闲卸载",
        type="int",
        default=900,
        min=0,
        max=86400,
        env=("LOCALAI_TRANSLATE_IDLE_UNLOAD",),
        description="NLLB 翻译模型空闲多少秒后卸载；0 表示关闭。",
    ),
    # ---- 并发 ---------------------------------------------------------
    Setting(
        key="asr_parallel",
        group="parallel",
        title="ASR 并发容量",
        type="int",
        default=2,
        min=1,
        max=4,
        env=("LOCALAI_ASR_PARALLEL", "LOCALAI_JOBS_PARALLEL_ASR"),
        description="SenseVoice 同时允许的推理数（1~4）。调度器与原生引擎共用这个值。",
    ),
    Setting(
        key="jobs_parallel_tts",
        group="parallel",
        title="TTS 并行上限",
        type="int",
        default=1,
        min=1,
        max=4,
        env=("LOCALAI_JOBS_PARALLEL_TTS",),
        description="任务队列里 TTS 同类的并行上限。实测无收益，建议保持 1。",
    ),
    Setting(
        key="jobs_parallel_separate",
        group="parallel",
        title="人声分离并行上限",
        type="int",
        default=1,
        min=1,
        max=4,
        env=("LOCALAI_JOBS_PARALLEL_SEPARATE",),
        description="任务队列里人声分离同类的并行上限。实测无收益，建议保持 1。",
    ),
    Setting(
        key="jobs_parallel_extra_mb",
        group="parallel",
        title="并行额外预留显存",
        type="int",
        default=600,
        min=0,
        max=65536,
        env=("LOCALAI_JOBS_PARALLEL_EXTRA_MB",),
        description="每新开一个并行任务前，要求额外空出的显存（MB）。",
    ),
    Setting(
        key="jobs_max_parallel",
        group="parallel",
        title="总并发硬上限",
        type="int",
        default=1,
        min=1,
        max=16,
        env=("LOCALAI_JOBS_MAX_PARALLEL",),
        description="跨能力总并发硬上限；1 表示不额外限制（由每类并行度决定）。",
    ),
    # ---- TTS ----------------------------------------------------------
    Setting(
        key="tts_backend",
        group="tts",
        title="TTS 后端",
        type="choice",
        default="engine",
        choices=("engine", "comfy"),
        env=("LOCALAI_TTS_BACKEND", "TTS_BACKEND"),
        description="engine=进程内 IndexTTS；comfy=转发到 ComfyUI。切换后需要重启。",
        requires_restart=True,
    ),
    Setting(
        key="tts_default_voice",
        group="tts",
        title="默认音色",
        type="str",
        default="default",
        env=("LOCALAI_TTS_DEFAULT_VOICE",),
        description="请求没有显式给 voice 时使用的音色名（<HOME>/voices 下的文件名去扩展名）。",
    ),
    # ---- ASR ----------------------------------------------------------
    Setting(
        key="asr_engine",
        group="asr",
        title="默认识别引擎",
        type="choice",
        default="sensevoice",
        choices=("sensevoice", "whisper"),
        env=("LOCALAI_ASR_ENGINE",),
        description="请求不指定 engine 时使用的识别引擎；下次请求立即生效。",
    ),
    Setting(
        key="whisper_model",
        group="asr",
        title="whisper 档位",
        type="choice",
        default="small",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        env=("LOCALAI_WHISPER_MODEL",),
        description="faster-whisper 默认档位；切换后下次请求按新档位加载。",
    ),
    Setting(
        key="whisper_device",
        group="asr",
        title="whisper 设备",
        type="str",
        default="cuda",
        env=("LOCALAI_WHISPER_DEVICE",),
        description="faster-whisper 设备，例如 cuda / cpu / cuda:0。",
        requires_restart=True,
    ),
    Setting(
        key="whisper_compute",
        group="asr",
        title="whisper 计算精度",
        type="str",
        default="int8_float16",
        env=("LOCALAI_WHISPER_COMPUTE",),
        description="faster-whisper 计算类型，例如 int8_float16 / float16 / int8。",
        requires_restart=True,
    ),
    # ---- 翻译 ---------------------------------------------------------
    Setting(
        key="translate_fp16",
        group="translate",
        title="翻译使用 fp16",
        type="bool",
        default=True,
        env=("LOCALAI_TRANSLATE_FP16",),
        description="GPU 上是否用 fp16 跑 NLLB；CPU 会自动回退 fp32。",
        requires_restart=True,
    ),
    Setting(
        key="translate_num_beams",
        group="translate",
        title="默认 beam 数",
        type="int",
        default=4,
        min=1,
        max=8,
        env=("LOCALAI_TRANSLATE_NUM_BEAMS",),
        description="翻译请求没有显式指定时的 beam 数；越大越慢、通常越稳。",
    ),
    Setting(
        key="translate_batch_size",
        group="translate",
        title="默认批大小",
        type="int",
        default=8,
        min=1,
        max=32,
        env=("LOCALAI_TRANSLATE_BATCH_SIZE",),
        description="翻译请求没有显式指定时的内部推理批大小。",
    ),
    Setting(
        key="translate_max_len",
        group="translate",
        title="默认单块长度",
        type="int",
        default=400,
        min=32,
        max=1024,
        env=("LOCALAI_TRANSLATE_MAX_LEN",),
        description="翻译请求没有显式指定 max_len 时，每个切分块的最大 token 数。",
    ),
    # ---- 媒体 ---------------------------------------------------------
    Setting(
        key="ffmpeg",
        group="media",
        title="ffmpeg 路径",
        type="path",
        default=None,
        env=("LOCALAI_FFMPEG", "INDEX_TTS_FFMPEG"),
        description="ffmpeg.exe 的完整路径；留空表示按「本机默认路径 → PATH」自动探测。",
        allow_empty=True,
    ),
)

#: 键 -> 定义。
BY_KEY: dict[str, Setting] = {item.key: item for item in SETTINGS}


def env_present(item: Setting) -> bool:
    """本进程启动时是否有非空环境变量覆盖了这个设置。"""
    for name in item.env:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return True
    return False


def coerce_value(item: Setting, raw: Any) -> Any:
    """把接口/配置文件里的原始值转成合法值；不合法抛 ``ValueError``（中文）。"""
    if item.type == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return bool(raw)
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "on", "y", "是", "开"):
            return True
        if text in ("0", "false", "no", "off", "n", "否", "关"):
            return False
        raise ValueError(f"{item.title}（{item.key}）必须是 true/false")

    if item.type == "int":
        if isinstance(raw, bool):
            raise ValueError(f"{item.title}（{item.key}）必须是整数")
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item.title}（{item.key}）必须是整数") from exc
        if item.min is not None and value < item.min:
            raise ValueError(f"{item.title}（{item.key}）不能小于 {item.min}")
        if item.max is not None and value > item.max:
            raise ValueError(f"{item.title}（{item.key}）不能大于 {item.max}")
        return value

    if item.type == "choice":
        value = str(raw).strip().lower()
        if value not in item.choices:
            raise ValueError(
                f"{item.title}（{item.key}）只能是：{', '.join(item.choices)}"
            )
        return value

    if item.type == "path":
        value = str(raw).strip().strip('"')
        if not value:
            if item.allow_empty:
                return None
            raise ValueError(f"{item.title}（{item.key}）不能为空")
        return value

    # str
    if raw is None:
        value = ""
    else:
        value = str(raw).strip()
    if not value and not item.allow_empty:
        raise ValueError(f"{item.title}（{item.key}）不能为空")
    if len(value) > 4096:
        raise ValueError(f"{item.title}（{item.key}）过长（上限 4096 字符）")
    return value


__all__ = [
    "GROUP_TITLES",
    "SETTINGS",
    "BY_KEY",
    "Setting",
    "env_present",
    "coerce_value",
]
