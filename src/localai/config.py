"""集中配置：所有可能因机器而变的东西都在这里。

设计原则
--------
1. **一个入口**：``LOCALAI_HOME`` 决定数据根目录，其余子目录全部由它推导。
   换机器只需要改 ``LOCALAI_HOME``（或环境变量），代码里不出现 ``E:\\``。
2. **旧变量兼容**：以前用 ``INDEX_TTS_*`` 的地方继续认；新变量 ``LOCALAI_*``
   优先。现有 ``start_ai_service.ps1`` 的语义保持不变。
3. **不 import 模型**：本模块只读环境变量、算路径、探测 ffmpeg，不碰 torch。
4. **可覆盖**：每一项都能用环境变量单独覆盖（见 ``docs/INSTALL.md`` 表格）。

常用环境变量::

    LOCALAI_HOME              数据根目录（默认：项目根）
    LOCALAI_HOST / LOCALAI_PORT
    LOCALAI_ENGINE_DIR        原生引擎目录（默认 <HOME>/engine）
    LOCALAI_TTS_BACKEND       engine / comfy
    LOCALAI_TTS_DEFAULT_VOICE 请求未给 voice 时的默认音色（默认 default）
    LOCALAI_MODEL_DIR         IndexTTS-2.5 权重目录
    LOCALAI_ASR_MODEL_DIR     SenseVoiceSmall 目录
    LOCALAI_CAMPPUS_DIR       campplus 目录
    LOCALAI_VOICES_DIR        音色目录
    LOCALAI_OUTPUT_DIR        输出目录
    LOCALAI_LOGS_DIR          日志目录
    LOCALAI_TORCH_HOME        Demucs/torch 缓存目录
    LOCALAI_FFMPEG            ffmpeg.exe 完整路径（默认读 config.json，再退回 PATH）
    LOCALAI_FFPROBE           ffprobe.exe 完整路径
    LOCALAI_MEDIA_OUTPUT_MAX_MB 转码产物返回上限（默认 512）
    LOCALAI_VRAM_MARGIN_MB    显存安全余量（默认 512）
    LOCALAI_IDLE_UNLOAD       TTS 空闲卸载秒数（默认 900，0=关闭）
    LOCALAI_ASR_IDLE_UNLOAD
    LOCALAI_SPEAKER_IDLE_UNLOAD
    LOCALAI_SEPARATE_IDLE_UNLOAD
    LOCALAI_ASR_PARALLEL      识别引擎并发容量（默认 2，范围 1~4；设 1 = 完全串行）
    LOCALAI_ASR_UNLOAD_TIMEOUT 识别引擎卸载等待推理退出的超时秒数（默认 30）
    LOCALAI_ASR_ENGINE        识别引擎默认值：sensevoice（默认）/ whisper
    LOCALAI_WHISPER_DIR       faster-whisper 权重根目录（默认 <HOME>/models/whisper）
    LOCALAI_WHISPER_MODEL     faster-whisper 档位（tiny/base/small/medium/large-v3，默认 small）
    LOCALAI_WHISPER_DEVICE    faster-whisper 设备（默认 cuda，可设 cpu）
    LOCALAI_WHISPER_COMPUTE   faster-whisper 计算类型（默认 int8_float16；CPU 回退 int8）
    LOCALAI_WHISPER_IDLE_UNLOAD faster-whisper 空闲卸载秒数（默认 900，0=关闭）
    LOCALAI_TRANSLATE_DIR     翻译权重根目录（默认 <HOME>/models/nllb）
    LOCALAI_TRANSLATE_IDLE_UNLOAD 翻译空闲卸载秒数（默认 900，0=关闭）
    LOCALAI_TRANSLATE_FP16    GPU 上是否用 fp16（默认 1；设 0 用 fp32）
    LOCALAI_TRANSLATE_NUM_BEAMS 翻译 beam 数（默认 2，范围 1~8）
    LOCALAI_TRANSLATE_BATCH_SIZE 翻译默认批大小（默认 8，范围 1~32）
    LOCALAI_TRANSLATE_MAX_LEN  翻译默认单块 token 上限（默认 400，范围 32~1024）
    LOCALAI_TOKEN             可选的 Bearer 鉴权令牌（默认空=不鉴权）
    LOCALAI_MODEL_SEARCH_PATHS 扫描已有模型的目录（; 分隔，默认空列表；
                              只在显式设置时生效，程序不预置任何本机路径）
    LOCALAI_WHISPER_MODEL_DIRS 按档位覆盖 whisper 目录（JSON：{"small":"D:\\..."}）
    LOCALAI_TRANSLATE_MODEL_DIR NLLB 权重目录（直接指向模型目录，不再拼模型名）

视觉 / 关键帧挑选（新增，见 ``docs/SERVICE_API.md``）::

    LOCALAI_VISION_DIR        视觉模型目录（默认 <HOME>/models/vision）
    LOCALAI_VISION_PORT       本地 llama-server 端口（默认 7891，只监听 127.0.0.1）
    LOCALAI_VISION_IDLE_UNLOAD 视觉模型空闲卸载秒数（默认 900，0=关闭）
    LOCALAI_VISION_CTX        llama-server 上下文长度（默认 4096）
    LOCALAI_VISION_GPU_LAYERS 卸载到 GPU 的层数（默认 99=全部）
    LOCALAI_VISION_TIMEOUT    单次视觉推理超时秒数（默认 120）
    LOCALAI_VISION_MODEL      GGUF 文件名（默认 Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf）
    LOCALAI_VISION_MMPROJ     视觉投影文件名（默认 mmproj-Q8_0.gguf）
    LOCALAI_VISION_SERVER     llama-server.exe 文件名（默认 llama-server.exe）
    LOCALAI_VISION_PLANNER    关键帧默认 planner：hybrid（默认）/ agent
    LOCALAI_VISION_PHASH_DISTANCE 感知哈希去重汉明距离阈值（默认 6）

下载 / 链接分析（新增，见 ``docs/SERVICE_API.md``）::

    LOCALAI_DOWNLOAD_MAX_MB      单个下载文件大小上限（默认 2048）
    LOCALAI_DOWNLOAD_SYNC_MAX_MB 同步下载接口允许的大小上限（默认 200）
    LOCALAI_DOWNLOAD_TIMEOUT     单次下载 / 分析任务超时秒数（默认 1800）
    LOCALAI_BILI_COOKIE          B站 cookie JSON 路径（默认空=沿用引擎里的本机默认路径）
    LOCALAI_DOWNLOAD_ALLOW_PRIVATE 1=允许私有地址（仅测试；生产不要设）

任务队列（详见 ``localai.jobs``）::

    LOCALAI_JOBS_RECOVER       重启后遗留 running 的处理：requeue（默认）/ fail
    LOCALAI_JOBS_MAX_PARALLEL  总并发硬上限（默认 1 = 不额外限制，由每类并行度决定）
    LOCALAI_JOBS_PARALLEL_TTS      TTS 同类并行上限（实测无收益，默认 1）
    LOCALAI_JOBS_PARALLEL_ASR      ASR 同类并行上限（默认跟随 LOCALAI_ASR_PARALLEL）
    LOCALAI_JOBS_PARALLEL_SEPARATE 人声分离同类并行上限（实测无收益，默认 1）
    LOCALAI_JOBS_PARALLEL_EXTRA_MB 每个新增并行任务需预留的显存（默认 600）
    LOCALAI_JOB_TIMEOUT        单个任务超时秒数（默认 600）
    LOCALAI_JOBS_KEEP_UPLOADS  任务完成后是否保留上传文件（默认 0=删除）
    LOCALAI_JOBS_RETENTION_DAYS 终态任务保留天数（默认 30，0=不清理）
    LOCALAI_OUTPUT_RETENTION_DAYS outputs/ 产物保留天数（默认 30，0=不清理）
    LOCALAI_TMP_RETENTION_HOURS 临时目录残留保留小时数（默认 24，0=不清理）
    LOCALAI_JOBS_POLL          调度器空转轮询间隔秒（默认 0.2）
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .constants import ASR_ENGINES, WHISPER_SIZES
from .settings_spec import SETTINGS, coerce_value, env_present

logger = logging.getLogger("localai.config")

#: 管理界面写入的配置文件（相对 ``LOCALAI_HOME``）。
CONFIG_FILE_NAME = "config.json"

# 本机专用路径（ffmpeg、ComfyUI 权重目录）不写死在代码里：优先环境变量，
# 其次 git 忽略的 ``config.json``（管理界面也写这个文件），最后退回 PATH。


def _pick(*names: str, default: Optional[str] = None) -> Optional[str]:
    """按顺序取第一个非空环境变量。"""
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip().strip('"')
    return default


def _pick_int(default: int, *names: str) -> int:
    raw = _pick(*names)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _pick_float(default: float, *names: str) -> float:
    raw = _pick(*names)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _pick_bool(default: bool, *names: str) -> bool:
    raw = _pick(*names)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _clean_paths(values: Any) -> list[str]:
    """去空、去重、展开用户目录；顺序保留，最多 20 个。"""
    if values is None:
        return []
    if isinstance(values, str):
        items = values.replace(os.pathsep, ";").split(";")
    elif isinstance(values, (list, tuple, set)):
        items = list(values)
    else:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        raw = str(item or "").strip().strip('"')
        if not raw:
            continue
        try:
            path = Path(raw).expanduser()
            # 不要求路径存在：环境变量里的路径可能暂时离线；扫描时再报错。
            text = str(path)
            key = os.path.normcase(os.path.abspath(text))
        except (OSError, ValueError):
            text = raw
            key = raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= 20:
            break
    return cleaned


def _pick_paths(*names: str) -> list[str]:
    raw = _pick(*names)
    return _clean_paths(raw) if raw is not None else []


def _parse_whisper_dirs(raw: Optional[str]) -> dict[str, str]:
    """解析 ``LOCALAI_WHISPER_MODEL_DIRS``（JSON 对象）；坏值按空处理。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("LOCALAI_WHISPER_MODEL_DIRS 不是合法 JSON，已忽略")
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in data.items():
        size = str(key or "").strip().lower()
        path = str(value or "").strip().strip('"')
        if size in WHISPER_SIZES and path:
            result[size] = str(Path(path).expanduser())
    return result


def _env_any(*names: str) -> bool:
    return any(
        os.environ.get(name) is not None and str(os.environ.get(name)).strip() != ""
        for name in names
    )


def _default_home() -> Path:
    """推导数据根目录。

    顺序：
    1. ``LOCALAI_HOME`` / ``INDEX_TTS_HOME``；
    2. 以源码方式安装时：``<项目根>/src/localai/config.py`` → 往上两级；
    3. 当前工作目录（或祖先）里同时有 ``engine/`` 与 ``svc/`` 的目录；
    4. 最后退回当前工作目录。
    """
    env_home = _pick("LOCALAI_HOME", "INDEX_TTS_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "pyproject.toml").is_file() or (package_root / "engine").is_dir():
        return package_root

    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / "engine").is_dir() and (base / "svc").is_dir():
            return base
        if (base / "pyproject.toml").is_file() and (base / "src" / "localai").is_dir():
            return base
    return cwd


def _config_text(data: dict[str, Any], key: str) -> Optional[str]:
    """取 config.json 里的非空字符串值；缺失或空白返回 None。"""
    raw = str(data.get(key) or "").strip().strip('"')
    return raw or None


def _machine_default(data: dict[str, Any], key: str) -> Optional[Path]:
    """取 config.json 里的本机可执行文件路径（只有真实文件才算数）。"""
    raw = _config_text(data, key)
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        return path if path.is_file() else None
    except OSError:
        return None


def _resolve_ffmpeg(config_data: dict[str, Any]) -> tuple[Optional[Path], str]:
    """按「LOCALAI_FFMPEG → config.json → PATH」找 ffmpeg。"""
    configured = _pick("LOCALAI_FFMPEG", "INDEX_TTS_FFMPEG")
    if configured:
        return Path(configured), "env"
    default = _machine_default(config_data, "ffmpeg")
    if default is not None:
        return default, "file"
    found = shutil.which("ffmpeg")
    if found:
        return Path(found), "path"
    return None, "missing"


def _resolve_ffprobe(
    ffmpeg: Optional[Path], config_data: dict[str, Any]
) -> tuple[Optional[Path], str]:
    """按「LOCALAI_FFPROBE → ffmpeg 同目录 → config.json → PATH」找 ffprobe。"""
    configured = _pick("LOCALAI_FFPROBE", "INDEX_TTS_FFPROBE")
    if configured:
        return Path(configured), "env"
    if ffmpeg is not None:
        sibling = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        try:
            if sibling.is_file():
                return sibling, "sibling"
        except OSError:
            pass
    default = _machine_default(config_data, "ffprobe")
    if default is not None:
        return default, "file"
    found = shutil.which("ffprobe")
    if found:
        return Path(found), "path"
    return None, "missing"


def config_file_path(home: Optional[Path] = None) -> Path:
    """管理界面配置文件路径：``<LOCALAI_HOME>/config.json``。"""
    return Path(home or _default_home()) / CONFIG_FILE_NAME


def load_config_file(home: Optional[Path] = None) -> dict[str, Any]:
    """读取 ``config.json``；不存在或不是合法 JSON 对象时返回空字典。

    这里故意不抛异常：一个手滑写坏的配置文件不应该让服务起不来，
    而是按「没有文件」处理，管理界面会显示来源为 default。
    """
    path = config_file_path(home)
    try:
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning("读取配置文件失败，按空配置处理：%s（%s）", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _apply_config_file(
    cfg: "Config",
    home: Path,
    overrides: Optional[dict[str, bool]] = None,
) -> None:
    """把 ``config.json`` 里的白名单值覆盖到配置对象上。

    优先级固定为 **环境变量 > config.json > 代码默认值**：只要某个键在启动时
    被环境变量覆盖（``overrides[key]`` 为真），这里就跳过，避免管理界面写入
    的值看起来生效、实际却被启动脚本压住。
    """
    data = load_config_file(home)
    if not data:
        return
    flags = overrides or {}
    for item in SETTINGS:
        if item.key not in data:
            continue
        if flags.get(item.key):
            continue
        try:
            value = coerce_value(item, data[item.key])
        except ValueError as exc:
            logger.warning("忽略配置文件里的非法值 %s=%r：%s", item.key, data[item.key], exc)
            continue

        if item.key == "asr_parallel":
            # 调度器与原生引擎必须同值，避免「调度开 2、引擎仍 1」。
            cfg.asr_parallel = int(value)
            cfg.jobs_parallel_asr = int(value)
        elif item.key == "ffmpeg":
            if value:
                cfg.ffmpeg = Path(str(value)).expanduser()
                cfg.ffmpeg_source = "file"
                sibling = cfg.ffmpeg.with_name(
                    "ffprobe.exe" if os.name == "nt" else "ffprobe"
                )
                if sibling.is_file():
                    cfg.ffprobe = sibling
                    cfg.ffprobe_source = "file"
            else:
                cfg.ffmpeg = None
                cfg.ffmpeg_source = "missing"
        elif item.key == "token":
            cfg.token = str(value or "")
        else:
            setattr(cfg, item.key, value)


#: 「配置指向」写入的路径键 -> 对应的环境变量（环境变量优先）。
PATH_CONFIG_ENV: dict[str, tuple[str, ...]] = {
    "tts_model_dir": ("LOCALAI_MODEL_DIR", "INDEX_TTS_MODEL_DIR"),
    "comfy_model_dir": ("LOCALAI_COMFY_MODEL_DIR", "INDEX_TTS_COMFY_MODEL_DIR"),
    "asr_model_dir": ("LOCALAI_ASR_MODEL_DIR", "INDEX_TTS_ASR_MODEL_DIR"),
    "translate_model_dir": ("LOCALAI_TRANSLATE_MODEL_DIR",),
    "speaker_model_dir": ("LOCALAI_CAMPPUS_DIR", "INDEX_TTS_CAMPPUS_DIR"),
    "demucs_repo": ("LOCALAI_DEMUCS_REPO", "INDEX_TTS_DEMUCS_REPO"),
    "vision_model_dir": ("LOCALAI_VISION_DIR",),
    "vad_model_dir": ("LOCALAI_ASR_VAD_DIR",),
    "whisper_model_dirs": ("LOCALAI_WHISPER_MODEL_DIRS",),
}


def _apply_path_overrides(cfg: "Config", home: Path) -> None:
    """应用 config.json 里「采用已有模型」写入的路径。

    这些键不在 /admin/settings 白名单里（由模型页的 adopt 接口管理），但必须和
    普通设置一样遵守 **环境变量 > config.json > 代码默认值**。
    """
    data = load_config_file(home)
    if not data:
        return

    def allowed(key: str) -> bool:
        return not _env_any(*PATH_CONFIG_ENV.get(key, ()))

    if "tts_model_dir" in data and allowed("tts_model_dir"):
        raw = str(data["tts_model_dir"] or "").strip()
        if raw:
            cfg.model_dir = Path(raw).expanduser()
    if "asr_model_dir" in data and allowed("asr_model_dir"):
        raw = str(data["asr_model_dir"] or "").strip()
        if raw:
            cfg.asr_model_dir = Path(raw).expanduser()
    if "translate_model_dir" in data and allowed("translate_model_dir"):
        raw = str(data["translate_model_dir"] or "").strip()
        cfg.translate_model_dir = Path(raw).expanduser() if raw else None
    if "speaker_model_dir" in data and allowed("speaker_model_dir"):
        raw = str(data["speaker_model_dir"] or "").strip()
        if raw:
            cfg.speaker_model_dir = Path(raw).expanduser()
    if "comfy_model_dir" in data and allowed("comfy_model_dir"):
        raw = str(data["comfy_model_dir"] or "").strip()
        if raw:
            cfg.comfy_model_dir = Path(raw).expanduser()
    if "demucs_repo" in data and allowed("demucs_repo"):
        raw = str(data["demucs_repo"] or "").strip()
        if raw:
            cfg.demucs_repo = Path(raw).expanduser()
    if "vision_model_dir" in data and allowed("vision_model_dir"):
        raw = str(data["vision_model_dir"] or "").strip()
        if raw:
            cfg.vision_dir = Path(raw).expanduser()
    if "vad_model_dir" in data and allowed("vad_model_dir"):
        raw = str(data["vad_model_dir"] or "").strip()
        if raw:
            cfg.asr_vad_dir = Path(raw).expanduser()
    if "whisper_model_dirs" in data and allowed("whisper_model_dirs"):
        raw = data["whisper_model_dirs"]
        if isinstance(raw, dict):
            cfg.whisper_model_dirs = _parse_whisper_dirs(json.dumps(raw, ensure_ascii=False))
        else:
            logger.warning("忽略 config.json 里的非法 whisper_model_dirs：%r", raw)


@dataclass(slots=True)
class Config:
    """一次服务进程的完整配置。"""

    home: Path
    host: str = "127.0.0.1"
    port: int = 7880
    engine_dir: Path = field(default_factory=Path)

    tts_backend: str = "engine"
    #: 请求没有显式给 voice 时使用的音色名。
    tts_default_voice: str = "default"
    model_dir: Path = field(default_factory=Path)
    asr_model_dir: Path = field(default_factory=Path)
    #: 分段转写用的 fsmn-vad 权重目录（SenseVoice 没有原生时间戳）。
    asr_vad_dir: Path = field(default_factory=Path)
    #: 识别引擎默认值（sensevoice / whisper）；任务级可用 engine= 覆盖。
    asr_engine: str = "sensevoice"
    #: faster-whisper 权重根目录；每档权重在 ``<whisper_dir>/<size>/``。
    whisper_dir: Path = field(default_factory=Path)
    #: 采用已有模型时按档位覆盖目录：``{"small": "D:\\..."}``。
    whisper_model_dirs: dict[str, str] = field(default_factory=dict)
    whisper_model: str = "small"
    whisper_device: str = "cuda"
    whisper_compute: str = "int8_float16"
    whisper_idle_unload: int = 900
    #: 翻译（NLLB-200）权重根目录；权重在 ``<translate_dir>/nllb-200-distilled-600M/``。
    translate_dir: Path = field(default_factory=Path)
    #: 采用已有模型时直接指向权重目录（不再拼模型名）；None=沿用 translate_dir。
    translate_model_dir: Optional[Path] = None
    translate_idle_unload: int = 900
    translate_fp16: bool = True
    translate_num_beams: int = 2
    translate_batch_size: int = 8
    #: 请求没有显式给 max_len 时使用的默认单块 token 上限。
    translate_max_len: int = 400
    #: 下载能力：单文件上限 / 同步接口上限 / 单次超时（秒）。
    download_max_mb: int = 2048
    download_sync_max_mb: int = 200
    download_timeout: float = 1800.0
    #: 视觉（VLM）：llama.cpp 本地服务 + Qwen2.5-VL-3B GGUF。
    vision_dir: Path = field(default_factory=Path)
    vision_port: int = 7891
    vision_idle_unload: int = 900
    vision_ctx: int = 4096
    vision_gpu_layers: int = 99
    vision_timeout: float = 120.0
    vision_model_file: str = "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    vision_mmproj_file: str = "mmproj-Q8_0.gguf"
    vision_server_file: str = "llama-server.exe"
    #: 关键帧挑选默认 planner（hybrid / agent）与去重汉明距离阈值。
    vision_planner: str = "hybrid"
    vision_phash_distance: int = 6
    #: 可选：显式指定 B站 cookie JSON；空=沿用引擎里的本机默认路径。
    bili_cookie_file: str = ""
    speaker_model_dir: Path = field(default_factory=Path)
    comfy_model_dir: Path = field(default_factory=Path)
    comfy_base: str = "http://127.0.0.1:8188"

    voices_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    logs_dir: Path = field(default_factory=Path)
    torch_home: Path = field(default_factory=Path)
    demucs_repo: Path = field(default_factory=Path)

    vram_estimates_path: Path = field(default_factory=Path)
    vram_margin_mb: int = 512

    idle_unload: int = 900
    asr_idle_unload: int = 900
    speaker_idle_unload: int = 900
    separate_idle_unload: int = 900

    ffmpeg: Optional[Path] = None
    ffprobe: Optional[Path] = None
    ffmpeg_source: str = "missing"
    ffprobe_source: str = "missing"
    #: 转码产物（/media/to_wav、/media/audio）单次返回上限（MB，0=不限制）。
    media_output_max_mb: int = 512

    token: str = ""
    #: 已有模型扫描目录；默认空，程序不预置任何本机路径。
    model_search_paths: list[str] = field(default_factory=list)
    log_level: str = "INFO"
    poll_interval: float = 0.4
    job_timeout: float = 180.0
    #: 启动时哪些管理设置被环境变量覆盖（键 -> bool），供 /admin/settings 判断来源。
    env_overrides: dict[str, bool] = field(default_factory=dict)

    # ---- 任务队列（新增，不影响旧配置） ---------------------------------
    jobs_recover: str = "requeue"
    #: 总并发硬上限。1（默认）= 不额外限制，实际并发由下面每类并行度决定；
    #: >=2 时再叠一道全局闸门（跨类仍串行，通常用不到）。
    jobs_max_parallel: int = 1
    jobs_parallel_tts: int = 1
    jobs_parallel_asr: int = 1
    jobs_parallel_separate: int = 1
    jobs_parallel_extra_mb: int = 600
    jobs_timeout: float = 600.0
    jobs_keep_uploads: bool = False
    jobs_poll_interval: float = 0.2
    #: 运行产物保留期（0=不清理）：终态任务行 / outputs 产物 / 临时目录残留。
    jobs_retention_days: int = 30
    output_retention_days: int = 30
    tmp_retention_hours: int = 24

    #: 识别引擎（SenseVoice）真实并发容量：默认 2，1 = 完全串行，范围 1~4。
    #: 这是原生引擎 ``LOCALAI_ASR_PARALLEL`` 的服务层投影；调度器与适配器
    #: 都从它对齐，避免「调度开 2、引擎仍串行」的两套配置打架。
    asr_parallel: int = 2

    # ---- 派生目录 -------------------------------------------------------
    @property
    def tts_model_dir(self) -> Path:
        """IndexTTS 权重目录（adopt 接口里的配置名，等价于 model_dir）。"""
        return self.model_dir

    @property
    def vision_model_dir(self) -> Path:
        """视觉模型目录（adopt 接口里的配置名，等价于 vision_dir）。"""
        return self.vision_dir

    @property
    def vad_model_dir(self) -> Path:
        """fsmn-vad 目录（adopt 接口里的配置名，等价于 asr_vad_dir）。"""
        return self.asr_vad_dir

    @property
    def translate_weight_dir(self) -> Path:
        """NLLB 实际权重目录：adopt 的精确目录优先，否则沿用根目录拼模型名。"""
        if self.translate_model_dir is not None:
            return self.translate_model_dir
        return self.translate_dir / "nllb-200-distilled-600M"

    def whisper_weights_dir(self, size: Optional[str] = None) -> Path:
        """某档 whisper 实际权重目录（adopt 的按档映射优先）。"""
        key = (size or self.whisper_model or "small").strip().lower()
        override = str(self.whisper_model_dirs.get(key) or "").strip()
        if override:
            return Path(override).expanduser()
        return self.whisper_dir / key

    @property
    def data_dir(self) -> Path:
        """任务库与上传文件的根目录：``<HOME>/data``（已被 .gitignore 排除）。"""
        return self.home / "data"

    @property
    def jobs_db_path(self) -> Path:
        """SQLite 任务库路径：``<HOME>/data/jobs.db``。"""
        return self.data_dir / "jobs.db"

    @property
    def downloads_dir(self) -> Path:
        """下载产物根目录：``<HOME>/data/downloads/<日期>/``（已被 .gitignore 排除）。"""
        return self.data_dir / "downloads"

    @property
    def jobs_upload_dir(self) -> Path:
        """异步任务的上传落盘根目录：``<HOME>/data/uploads/<job_id>/``。"""
        return self.data_dir / "uploads"

    @property
    def asr_tmp_dir(self) -> Path:
        return self.output_dir / "asr_tmp"

    @property
    def media_tmp_dir(self) -> Path:
        return self.output_dir / "media_tmp"

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def scenes_dir(self) -> Path:
        """场景边界检测结果落盘目录：``<OUTPUT>/scenes/``。"""
        return self.output_dir / "scenes"

    @property
    def speaker_tmp_dir(self) -> Path:
        return self.output_dir / "speaker_tmp"

    @property
    def separate_tmp_dir(self) -> Path:
        return self.output_dir / "separate_tmp"

    @property
    def separate_dir(self) -> Path:
        return self.output_dir / "separate"

    @property
    def base_url(self) -> str:
        host = self.host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    # ---- 构造 -----------------------------------------------------------
    @classmethod
    def from_env(cls, home: Optional[str | Path] = None) -> "Config":
        root = Path(home).expanduser().resolve() if home else _default_home()
        engine_dir = Path(
            _pick("LOCALAI_ENGINE_DIR", "INDEX_TTS_ENGINE_DIR", default=str(root / "engine"))
        ).expanduser()

        # 本机专用路径只放 git 忽略的 config.json：代码里不预置任何机器路径。
        config_data = load_config_file(root)
        ffmpeg, ffmpeg_source = _resolve_ffmpeg(config_data)
        ffprobe, ffprobe_source = _resolve_ffprobe(ffmpeg, config_data)

        backend = (_pick("LOCALAI_TTS_BACKEND", "TTS_BACKEND", default="engine") or "engine").lower()
        if backend not in ("engine", "comfy"):
            backend = "engine"

        torch_home = Path(
            _pick("LOCALAI_TORCH_HOME", "INDEX_TTS_TORCH_HOME", default=str(root / "models" / "torch"))
        ).expanduser()
        demucs_repo = Path(
            _pick(
                "LOCALAI_DEMUCS_REPO",
                "INDEX_TTS_DEMUCS_REPO",
                default=str(torch_home / "hub" / "checkpoints"),
            )
        ).expanduser()

        # 识别并发：新变量 LOCALAI_ASR_PARALLEL 优先；只设旧的
        # LOCALAI_JOBS_PARALLEL_ASR 时也认。两者最终投影到同一个值，
        # 再分别注入原生引擎和调度器，避免「调度开 2、引擎仍 1」。
        asr_parallel = max(
            1,
            min(
                4,
                _pick_int(
                    2, "LOCALAI_ASR_PARALLEL", "LOCALAI_JOBS_PARALLEL_ASR"
                ),
            ),
        )

        # 识别引擎默认值：只认 sensevoice / whisper；写错时回落到 sensevoice，
        # 保证旧行为不变（非法值不会让服务变成不可用）。
        asr_engine = (
            _pick("LOCALAI_ASR_ENGINE", default="sensevoice") or "sensevoice"
        ).strip().lower()
        if asr_engine not in ASR_ENGINES:
            asr_engine = "sensevoice"

        # whisper 档位：非法值回落到 small（默认档）。
        whisper_model = (
            _pick("LOCALAI_WHISPER_MODEL", default="small") or "small"
        ).strip().lower()
        if whisper_model not in WHISPER_SIZES:
            whisper_model = "small"

        cfg = cls(
            home=root,
            host=_pick("LOCALAI_HOST", "INDEX_TTS_HOST", default="127.0.0.1") or "127.0.0.1",
            port=_pick_int(7880, "LOCALAI_PORT", "INDEX_TTS_PORT"),
            engine_dir=engine_dir,
            tts_backend=backend,
            tts_default_voice=(
                _pick("LOCALAI_TTS_DEFAULT_VOICE", default="default") or "default"
            ).strip(),
            model_dir=Path(
                _pick(
                    "LOCALAI_MODEL_DIR",
                    "INDEX_TTS_MODEL_DIR",
                    default=str(root / "models" / "TTS" / "IndexTTS-2.5"),
                )
            ).expanduser(),
            asr_model_dir=Path(
                _pick(
                    "LOCALAI_ASR_MODEL_DIR",
                    "INDEX_TTS_ASR_MODEL_DIR",
                    default=str(root / "models" / "SenseVoiceSmall"),
                )
            ).expanduser(),
            asr_engine=asr_engine,
            asr_vad_dir=Path(
                _pick(
                    "LOCALAI_ASR_VAD_DIR",
                    default=str(root / "models" / "vad" / "fsmn-vad"),
                )
            ).expanduser(),
            whisper_dir=Path(
                _pick(
                    "LOCALAI_WHISPER_DIR",
                    default=str(root / "models" / "whisper"),
                )
            ).expanduser(),
            whisper_model_dirs=_parse_whisper_dirs(
                _pick("LOCALAI_WHISPER_MODEL_DIRS")
            ),
            whisper_model=whisper_model,
            whisper_device=(
                _pick("LOCALAI_WHISPER_DEVICE", default="cuda") or "cuda"
            ).strip(),
            whisper_compute=(
                _pick("LOCALAI_WHISPER_COMPUTE", default="int8_float16") or "int8_float16"
            ).strip(),
            whisper_idle_unload=_pick_int(900, "LOCALAI_WHISPER_IDLE_UNLOAD"),
            translate_dir=Path(
                _pick(
                    "LOCALAI_TRANSLATE_DIR",
                    default=str(root / "models" / "nllb"),
                )
            ).expanduser(),
            translate_model_dir=(
                Path(_pick("LOCALAI_TRANSLATE_MODEL_DIR")).expanduser()
                if _pick("LOCALAI_TRANSLATE_MODEL_DIR")
                else None
            ),
            translate_idle_unload=_pick_int(900, "LOCALAI_TRANSLATE_IDLE_UNLOAD"),
            translate_fp16=_pick_bool(True, "LOCALAI_TRANSLATE_FP16"),
            translate_num_beams=max(
                1, min(8, _pick_int(2, "LOCALAI_TRANSLATE_NUM_BEAMS"))
            ),
            translate_batch_size=max(
                1, min(32, _pick_int(8, "LOCALAI_TRANSLATE_BATCH_SIZE"))
            ),
            translate_max_len=max(
                32, min(1024, _pick_int(400, "LOCALAI_TRANSLATE_MAX_LEN"))
            ),
            download_max_mb=max(1, _pick_int(2048, "LOCALAI_DOWNLOAD_MAX_MB")),
            download_sync_max_mb=max(
                1, _pick_int(200, "LOCALAI_DOWNLOAD_SYNC_MAX_MB")
            ),
            download_timeout=max(1.0, _pick_float(1800.0, "LOCALAI_DOWNLOAD_TIMEOUT")),
            vision_dir=Path(
                _pick(
                    "LOCALAI_VISION_DIR",
                    default=str(root / "models" / "vision"),
                )
            ).expanduser(),
            vision_port=max(1, min(65535, _pick_int(7891, "LOCALAI_VISION_PORT"))),
            vision_idle_unload=_pick_int(900, "LOCALAI_VISION_IDLE_UNLOAD"),
            vision_ctx=max(2048, min(32768, _pick_int(4096, "LOCALAI_VISION_CTX"))),
            vision_gpu_layers=max(0, min(999, _pick_int(99, "LOCALAI_VISION_GPU_LAYERS"))),
            vision_timeout=max(10.0, _pick_float(120.0, "LOCALAI_VISION_TIMEOUT")),
            vision_model_file=(
                _pick("LOCALAI_VISION_MODEL", default="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
                or "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
            ).strip(),
            vision_mmproj_file=(
                _pick("LOCALAI_VISION_MMPROJ", default="mmproj-Q8_0.gguf")
                or "mmproj-Q8_0.gguf"
            ).strip(),
            vision_server_file=(
                _pick("LOCALAI_VISION_SERVER", default="llama-server.exe")
                or "llama-server.exe"
            ).strip(),
            vision_planner=(
                _pick("LOCALAI_VISION_PLANNER", default="hybrid") or "hybrid"
            ).strip().lower(),
            vision_phash_distance=max(
                0, min(32, _pick_int(6, "LOCALAI_VISION_PHASH_DISTANCE"))
            ),
            bili_cookie_file=(_pick("LOCALAI_BILI_COOKIE") or "").strip(),
            speaker_model_dir=Path(
                _pick(
                    "LOCALAI_CAMPPUS_DIR",
                    "INDEX_TTS_CAMPPUS_DIR",
                    default=str(root / "models" / "campplus"),
                )
            ).expanduser(),
            comfy_model_dir=Path(
                _pick(
                    "LOCALAI_COMFY_MODEL_DIR",
                    "INDEX_TTS_COMFY_MODEL_DIR",
                    default=(
                        _config_text(config_data, "comfy_model_dir")
                        or str(root / "models" / "TTS" / "IndexTTS-2.5")
                    ),
                )
            ).expanduser(),
            comfy_base=(_pick("LOCALAI_COMFY_BASE", "COMFY_BASE", default="http://127.0.0.1:8188") or "").rstrip("/"),
            voices_dir=Path(
                _pick("LOCALAI_VOICES_DIR", "INDEX_TTS_VOICES", default=str(root / "voices"))
            ).expanduser(),
            output_dir=Path(
                _pick("LOCALAI_OUTPUT_DIR", "INDEX_TTS_OUTPUT", default=str(root / "outputs"))
            ).expanduser(),
            logs_dir=Path(
                _pick("LOCALAI_LOGS_DIR", "INDEX_TTS_LOGS", default=str(root / "logs"))
            ).expanduser(),
            torch_home=torch_home,
            demucs_repo=demucs_repo,
            vram_estimates_path=Path(
                _pick(
                    "LOCALAI_VRAM_ESTIMATES",
                    "INDEX_TTS_VRAM_ESTIMATES",
                    default=str(root / "logs" / "vram_estimates.json"),
                )
            ).expanduser(),
            vram_margin_mb=_pick_int(
                512, "LOCALAI_VRAM_MARGIN_MB", "INDEX_TTS_VRAM_MARGIN_MB"
            ),
            idle_unload=_pick_int(
                900, "LOCALAI_IDLE_UNLOAD", "INDEX_TTS_IDLE_UNLOAD"
            ),
            asr_idle_unload=_pick_int(
                900, "LOCALAI_ASR_IDLE_UNLOAD", "INDEX_TTS_ASR_IDLE_UNLOAD"
            ),
            speaker_idle_unload=_pick_int(
                900, "LOCALAI_SPEAKER_IDLE_UNLOAD", "INDEX_TTS_SPEAKER_IDLE_UNLOAD"
            ),
            separate_idle_unload=_pick_int(
                900, "LOCALAI_SEPARATE_IDLE_UNLOAD", "INDEX_TTS_SEPARATE_IDLE_UNLOAD"
            ),
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            ffmpeg_source=ffmpeg_source,
            ffprobe_source=ffprobe_source,
            token=(_pick("LOCALAI_TOKEN") or ""),
            model_search_paths=_pick_paths("LOCALAI_MODEL_SEARCH_PATHS"),
            log_level=(_pick("LOCALAI_LOG_LEVEL", default="INFO") or "INFO").upper(),
            poll_interval=max(0.05, _pick_float(0.4, "LOCALAI_POLL", "INDEX_TTS_POLL")),
            job_timeout=max(1.0, _pick_float(180.0, "LOCALAI_TIMEOUT", "INDEX_TTS_TIMEOUT")),
            jobs_recover=(
                "fail"
                if (_pick("LOCALAI_JOBS_RECOVER", default="requeue") or "requeue").strip().lower()
                == "fail"
                else "requeue"
            ),
            jobs_max_parallel=max(1, _pick_int(1, "LOCALAI_JOBS_MAX_PARALLEL")),
            jobs_parallel_tts=max(1, _pick_int(1, "LOCALAI_JOBS_PARALLEL_TTS")),
            jobs_parallel_asr=asr_parallel,
            jobs_parallel_separate=max(
                1, _pick_int(1, "LOCALAI_JOBS_PARALLEL_SEPARATE")
            ),
            asr_parallel=asr_parallel,
            jobs_parallel_extra_mb=max(
                0, _pick_int(600, "LOCALAI_JOBS_PARALLEL_EXTRA_MB")
            ),
            jobs_timeout=max(
                1.0, _pick_float(600.0, "LOCALAI_JOB_TIMEOUT")
            ),
            jobs_keep_uploads=_pick_bool(False, "LOCALAI_JOBS_KEEP_UPLOADS"),
            jobs_poll_interval=max(
                0.05, _pick_float(0.2, "LOCALAI_JOBS_POLL")
            ),
            media_output_max_mb=max(
                0, _pick_int(512, "LOCALAI_MEDIA_OUTPUT_MAX_MB")
            ),
            jobs_retention_days=max(0, _pick_int(30, "LOCALAI_JOBS_RETENTION_DAYS")),
            output_retention_days=max(
                0, _pick_int(30, "LOCALAI_OUTPUT_RETENTION_DAYS")
            ),
            tmp_retention_hours=max(
                0, _pick_int(24, "LOCALAI_TMP_RETENTION_HOURS")
            ),
        )

        # 记录「启动时哪些键被环境变量覆盖」，必须在 export_env() 之前算：
        # export_env 会把配置写回同名环境变量，事后再判断会把 file/default
        # 误判成 env。
        if cfg.vision_planner not in ("hybrid", "agent"):
            logger.warning(
                "LOCALAI_VISION_PLANNER=%r 不合法，回落到 hybrid（可选 hybrid / agent）",
                cfg.vision_planner,
            )
            cfg.vision_planner = "hybrid"
        cfg.env_overrides = {item.key: False for item in SETTINGS}
        for item in SETTINGS:
            cfg.env_overrides[item.key] = env_present(item)

        # 文件覆盖只对「没有被环境变量覆盖」的键生效。
        _apply_config_file(cfg, root, cfg.env_overrides)
        # adopt 写入的路径覆盖（不在设置白名单里，但同样 env > file）。
        _apply_path_overrides(cfg, root)
        for key, names in PATH_CONFIG_ENV.items():
            cfg.env_overrides[key] = _env_any(*names)
        cfg.env_overrides["model_search_paths"] = env_present(
            next(item for item in SETTINGS if item.key == "model_search_paths")
        )
        return cfg

    # ---- 给原生引擎写回环境变量 -----------------------------------------
    def export_env(self) -> None:
        """把配置写回环境变量，供 ``engine/`` 下的原生引擎模块读取。

        原生模块（engine.py / asr_engine.py / ...）在 import 时读取
        ``INDEX_TTS_*`` 环境变量并据此推导路径；服务层统一在这里注入，
        保证「配置只有一个来源」，同时不修改这些模块。
        """
        # 识别并发以「真正生效的闸门容量」为准写回；
        # LOCALAI_ASR_PARALLEL（原生引擎）与 LOCALAI_JOBS_PARALLEL_ASR
        # （调度器）始终同值，避免两处配置打架。
        asr_parallel = self.asr_parallel_effective()
        mapping = {
            "TTS_BACKEND": self.tts_backend,
            "LOCALAI_TTS_DEFAULT_VOICE": self.tts_default_voice,
            "LOCALAI_ASR_PARALLEL": str(asr_parallel),
            "LOCALAI_JOBS_PARALLEL_ASR": str(asr_parallel),
            "INDEX_TTS_HOME": str(self.home),
            "INDEX_TTS_MODEL_DIR": str(self.model_dir),
            "INDEX_TTS_ASR_MODEL_DIR": str(self.asr_model_dir),
            "LOCALAI_ASR_VAD_DIR": str(self.asr_vad_dir),
            "LOCALAI_ASR_ENGINE": self.asr_engine,
            "LOCALAI_WHISPER_DIR": str(self.whisper_dir),
            "LOCALAI_WHISPER_MODEL": self.whisper_model,
            "LOCALAI_WHISPER_DEVICE": self.whisper_device,
            "LOCALAI_WHISPER_COMPUTE": self.whisper_compute,
            "LOCALAI_WHISPER_IDLE_UNLOAD": str(self.whisper_idle_unload),
            "LOCALAI_TRANSLATE_DIR": str(self.translate_dir),
            "LOCALAI_TRANSLATE_IDLE_UNLOAD": str(self.translate_idle_unload),
            "LOCALAI_TRANSLATE_FP16": "1" if self.translate_fp16 else "0",
            "LOCALAI_TRANSLATE_NUM_BEAMS": str(self.translate_num_beams),
            "LOCALAI_TRANSLATE_BATCH_SIZE": str(self.translate_batch_size),
            "LOCALAI_TRANSLATE_MAX_LEN": str(self.translate_max_len),
            "INDEX_TTS_CAMPPUS_DIR": str(self.speaker_model_dir),
            "INDEX_TTS_TORCH_HOME": str(self.torch_home),
            "INDEX_TTS_DEMUCS_REPO": str(self.demucs_repo),
            "INDEX_TTS_VOICES": str(self.voices_dir),
            "INDEX_TTS_OUTPUT": str(self.output_dir),
            "LOCALAI_LOGS_DIR": str(self.logs_dir),
            "INDEX_TTS_VRAM_ESTIMATES": str(self.vram_estimates_path),
            "INDEX_TTS_VRAM_MARGIN_MB": str(self.vram_margin_mb),
            "INDEX_TTS_IDLE_UNLOAD": str(self.idle_unload),
            "INDEX_TTS_ASR_IDLE_UNLOAD": str(self.asr_idle_unload),
            "INDEX_TTS_SPEAKER_IDLE_UNLOAD": str(self.speaker_idle_unload),
            "INDEX_TTS_SEPARATE_IDLE_UNLOAD": str(self.separate_idle_unload),
            "LOCALAI_DOWNLOAD_MAX_MB": str(self.download_max_mb),
            "LOCALAI_DOWNLOAD_SYNC_MAX_MB": str(self.download_sync_max_mb),
            "LOCALAI_DOWNLOAD_TIMEOUT": str(int(self.download_timeout)),
            "LOCALAI_VISION_DIR": str(self.vision_dir),
            "LOCALAI_VISION_PORT": str(self.vision_port),
            "LOCALAI_VISION_IDLE_UNLOAD": str(self.vision_idle_unload),
            "LOCALAI_VISION_CTX": str(self.vision_ctx),
            "LOCALAI_VISION_GPU_LAYERS": str(self.vision_gpu_layers),
            "LOCALAI_VISION_TIMEOUT": str(self.vision_timeout),
            "LOCALAI_VISION_MODEL": self.vision_model_file,
            "LOCALAI_VISION_MMPROJ": self.vision_mmproj_file,
            "LOCALAI_VISION_SERVER": self.vision_server_file,
            "LOCALAI_VISION_PLANNER": self.vision_planner,
            "LOCALAI_VISION_PHASH_DISTANCE": str(self.vision_phash_distance),
            "TORCH_HOME": str(self.torch_home),
        }
        if self.bili_cookie_file:
            mapping["LOCALAI_BILI_COOKIE"] = self.bili_cookie_file
        if self.ffmpeg is not None:
            mapping["INDEX_TTS_FFMPEG"] = str(self.ffmpeg)
        if self.ffprobe is not None:
            mapping["INDEX_TTS_FFPROBE"] = str(self.ffprobe)
        if self.comfy_base:
            mapping["COMFY_BASE"] = self.comfy_base
        os.environ.update(mapping)

        # 仅在不是「启动时由环境变量指定」时，才把 admin 写入的值同步回环境；
        # 否则会覆盖调用方启动脚本里的显式设置，违背 env > config 约定。
        if not self.env_overrides.get("translate_model_dir"):
            if self.translate_model_dir is not None:
                os.environ["LOCALAI_TRANSLATE_MODEL_DIR"] = str(
                    self.translate_model_dir
                )
            else:
                os.environ.pop("LOCALAI_TRANSLATE_MODEL_DIR", None)
        if not self.env_overrides.get("whisper_model_dirs"):
            if self.whisper_model_dirs:
                os.environ["LOCALAI_WHISPER_MODEL_DIRS"] = json.dumps(
                    self.whisper_model_dirs, ensure_ascii=False
                )
            else:
                os.environ.pop("LOCALAI_WHISPER_MODEL_DIRS", None)
        if not self.env_overrides.get("model_search_paths"):
            if self.model_search_paths:
                os.environ["LOCALAI_MODEL_SEARCH_PATHS"] = os.pathsep.join(
                    self.model_search_paths
                )
            else:
                os.environ.pop("LOCALAI_MODEL_SEARCH_PATHS", None)

    # ---- 诊断 -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """给 /capabilities、doctor、日志用的可打印摘要。"""
        return {
            "home": str(self.home),
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "engine_dir": str(self.engine_dir),
            "tts_backend": self.tts_backend,
            "tts_default_voice": self.tts_default_voice,
            "model_dir": str(self.model_dir),
            "asr_model_dir": str(self.asr_model_dir),
            "asr_engine": self.asr_engine,
            "asr_vad_dir": str(self.asr_vad_dir),
            "model_search_paths": list(self.model_search_paths),
            "whisper": {
                "dir": str(self.whisper_dir),
                "model_dirs": dict(self.whisper_model_dirs),
                "model": self.whisper_model,
                "device": self.whisper_device,
                "compute_type": self.whisper_compute,
                "idle_unload": self.whisper_idle_unload,
            },
            "translate": {
                "dir": str(self.translate_dir),
                "model_dir": str(self.translate_weight_dir),
                "model": "nllb-200-distilled-600M",
                "idle_unload": self.translate_idle_unload,
                "fp16": self.translate_fp16,
                "num_beams": self.translate_num_beams,
                "batch_size": self.translate_batch_size,
                "max_len": self.translate_max_len,
            },
            "download": {
                "dir": str(self.downloads_dir),
                "max_mb": self.download_max_mb,
                "sync_max_mb": self.download_sync_max_mb,
                "timeout_seconds": self.download_timeout,
                "bili_cookie_file": self.bili_cookie_file or None,
            },
            "vision": {
                "dir": str(self.vision_dir),
                "port": self.vision_port,
                "model_file": self.vision_model_file,
                "mmproj_file": self.vision_mmproj_file,
                "server_file": self.vision_server_file,
                "idle_unload": self.vision_idle_unload,
                "ctx": self.vision_ctx,
                "gpu_layers": self.vision_gpu_layers,
                "planner": self.vision_planner,
                "phash_distance": self.vision_phash_distance,
            },
            "speaker_model_dir": str(self.speaker_model_dir),
            "voices_dir": str(self.voices_dir),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "torch_home": str(self.torch_home),
            "demucs_repo": str(self.demucs_repo),
            "ffmpeg": str(self.ffmpeg) if self.ffmpeg else None,
            "ffmpeg_source": self.ffmpeg_source,
            "ffprobe": str(self.ffprobe) if self.ffprobe else None,
            "ffprobe_source": self.ffprobe_source,
            "vram_margin_mb": self.vram_margin_mb,
            "asr_parallel": self.asr_parallel_effective(),
            "idle_unload": self.idle_unload_map(),
            "auth_required": bool(self.token),
            "jobs": {
                "db_path": str(self.jobs_db_path),
                "upload_dir": str(self.jobs_upload_dir),
                "recover": self.jobs_recover,
                "max_parallel": self.jobs_max_parallel,
                "per_kind_parallel": self.jobs_parallel_map(),
                "parallel_extra_mb": self.jobs_parallel_extra_mb,
                "timeout_seconds": self.jobs_timeout,
                "keep_uploads": self.jobs_keep_uploads,
            },
        }

    # ---- 空闲卸载 -------------------------------------------------------
    def idle_unload_map(self) -> dict[str, int]:
        """各能力的空闲卸载阈值（秒，0=关闭）。

        适配器、/capabilities 与配置摘要都从这里取，新增能力只改这一处。
        """
        return {
            "tts": self.idle_unload,
            "asr": self.asr_idle_unload,
            "whisper": self.whisper_idle_unload,
            "speaker": self.speaker_idle_unload,
            "separate": self.separate_idle_unload,
            "translate": self.translate_idle_unload,
            "vision": self.vision_idle_unload,
        }

    # ---- 任务队列并发度 -------------------------------------------------
    def asr_parallel_effective(self) -> int:
        """识别并发最终生效值（两个配置字段取较大者，夹在 1~4）。

        ``from_env()`` 会让 ``asr_parallel`` 与 ``jobs_parallel_asr`` 相同；
        这里取 max 是为了兼容直接构造 / ``dataclasses.replace`` 只改其中一个
        字段的旧代码，避免「字段显示 2、适配器却仍 1」。
        """
        return max(
            1, min(4, max(int(self.asr_parallel), int(self.jobs_parallel_asr)))
        )

    def jobs_parallel_for(self, kind: str) -> int:
        """某类任务的配置并行上限（未单独配置的能力固定 1）。"""
        if kind == "asr":
            return self.asr_parallel_effective()
        # 下载是纯网络 / IO，不占显存；同类并行默认 1，避免同站并发请求。
        # 分析内部要跑 ASR，因此固定串行。
        return {
            "tts": self.jobs_parallel_tts,
            "separate": self.jobs_parallel_separate,
            "download": 1,
            "analyze": 1,
            "vision": 1,
        }.get(kind, 1)

    def job_timeout_for(self, kind: str) -> float:
        """某类任务的执行超时（秒）。

        下载 / 链接分析可能是几十分钟的长任务，用 ``LOCALAI_DOWNLOAD_TIMEOUT``；
        关键帧挑选可能要多轮 VLM 推理，至少给 20 分钟；其余能力沿用旧的
        ``LOCALAI_JOB_TIMEOUT``，旧语义不变。
        """
        if kind in ("download", "analyze"):
            return float(self.download_timeout)
        if kind == "vision":
            return max(float(self.jobs_timeout), 1200.0)
        return float(self.jobs_timeout)

    def jobs_parallel_map(self) -> dict[str, int]:
        """所有能力当前配置的并行上限（给 /jobs/stats、/capabilities）。"""
        return {
            "tts": self.jobs_parallel_tts,
            "asr": self.asr_parallel_effective(),
            "separate": self.jobs_parallel_separate,
            "media": 1,
            "translate": 1,
            "download": 1,
            "analyze": 1,
            "vision": 1,
        }

    def jobs_global_cap(self) -> Optional[int]:
        """总并发硬上限；``None`` 表示不额外限制（由每类并行度决定）。

        ``LOCALAI_JOBS_MAX_PARALLEL`` 默认 1：沿用旧值「1」表示「不额外限制」，
        避免旧脚本把它当成「只能跑一个」而把同类型并行整个关掉。显式设成
        2 及以上时才作为全局硬上限生效。
        """
        return self.jobs_max_parallel if self.jobs_max_parallel > 1 else None


# 进程级唯一配置。导入时即把路径写回环境变量，保证后面任何模块
# （包括原生引擎）看到的都是同一套值。
CONFIG = Config.from_env()
CONFIG.export_env()


def get_config() -> Config:
    """取当前进程配置（测试里可以 monkeypatch 这个函数）。"""
    return CONFIG


def ensure_directories(cfg: Optional[Config] = None) -> list[Path]:
    """创建服务运行需要的目录，返回实际创建的列表。"""
    cfg = cfg or get_config()
    wanted = [
        cfg.voices_dir,
        cfg.output_dir,
        cfg.logs_dir,
        cfg.asr_tmp_dir,
        cfg.media_tmp_dir,
        cfg.frames_dir,
        cfg.scenes_dir,
        cfg.whisper_dir,
        cfg.translate_dir,
        cfg.asr_vad_dir,
        cfg.speaker_tmp_dir,
        cfg.separate_tmp_dir,
        cfg.separate_dir,
        cfg.vram_estimates_path.parent,
        cfg.data_dir,
        cfg.jobs_upload_dir,
        cfg.downloads_dir,
        cfg.vision_dir,
    ]
    created: list[Path] = []
    for path in wanted:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created
