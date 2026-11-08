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
    LOCALAI_MODEL_DIR         IndexTTS-2.5 权重目录
    LOCALAI_ASR_MODEL_DIR     SenseVoiceSmall 目录
    LOCALAI_CAMPPUS_DIR       campplus 目录
    LOCALAI_VOICES_DIR        音色目录
    LOCALAI_OUTPUT_DIR        输出目录
    LOCALAI_SPEAKERS_DIR      声纹库目录
    LOCALAI_LOGS_DIR          日志目录
    LOCALAI_TORCH_HOME        Demucs/torch 缓存目录
    LOCALAI_FFMPEG            ffmpeg.exe 完整路径
    LOCALAI_FFPROBE           ffprobe.exe 完整路径
    LOCALAI_VRAM_MARGIN_MB    显存安全余量（默认 512）
    LOCALAI_IDLE_UNLOAD       TTS 空闲卸载秒数（默认 900，0=关闭）
    LOCALAI_ASR_IDLE_UNLOAD
    LOCALAI_SPEAKER_IDLE_UNLOAD
    LOCALAI_SEPARATE_IDLE_UNLOAD
    LOCALAI_TOKEN             可选的 Bearer 鉴权令牌（默认空=不鉴权）
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# 本机现有的 ffmpeg 常见位置；换机器探测不到就走 PATH。
DEFAULT_FFMPEG = r"C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffmpeg.exe"
DEFAULT_FFPROBE = r"C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffprobe.exe"


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


def _resolve_ffmpeg() -> tuple[Optional[Path], str]:
    """按「LOCALAI_FFMPEG → 本机默认路径 → PATH」找 ffmpeg。"""
    configured = _pick("LOCALAI_FFMPEG", "INDEX_TTS_FFMPEG")
    if configured:
        return Path(configured), "env"
    default = Path(DEFAULT_FFMPEG)
    try:
        if default.is_file():
            return default, "default"
    except OSError:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return Path(found), "path"
    return None, "missing"


def _resolve_ffprobe(ffmpeg: Optional[Path]) -> tuple[Optional[Path], str]:
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
    default = Path(DEFAULT_FFPROBE)
    try:
        if default.is_file():
            return default, "default"
    except OSError:
        pass
    found = shutil.which("ffprobe")
    if found:
        return Path(found), "path"
    return None, "missing"


@dataclass(slots=True)
class Config:
    """一次服务进程的完整配置。"""

    home: Path
    host: str = "127.0.0.1"
    port: int = 7880
    engine_dir: Path = field(default_factory=Path)

    tts_backend: str = "engine"
    model_dir: Path = field(default_factory=Path)
    asr_model_dir: Path = field(default_factory=Path)
    speaker_model_dir: Path = field(default_factory=Path)
    comfy_model_dir: Path = field(default_factory=Path)
    comfy_base: str = "http://127.0.0.1:8188"

    voices_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    logs_dir: Path = field(default_factory=Path)
    speakers_dir: Path = field(default_factory=Path)
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

    token: str = ""
    log_level: str = "INFO"
    poll_interval: float = 0.4
    job_timeout: float = 180.0

    # ---- 派生目录 -------------------------------------------------------
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

        ffmpeg, ffmpeg_source = _resolve_ffmpeg()
        ffprobe, ffprobe_source = _resolve_ffprobe(ffmpeg)

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

        return cls(
            home=root,
            host=_pick("LOCALAI_HOST", "INDEX_TTS_HOST", default="127.0.0.1") or "127.0.0.1",
            port=_pick_int(7880, "LOCALAI_PORT", "INDEX_TTS_PORT"),
            engine_dir=engine_dir,
            tts_backend=backend,
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
                    default=r"D:\Comfy-Desktop\视频\ComfyUI\models\TTS\IndexTTS-2.5",
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
            speakers_dir=Path(
                _pick("LOCALAI_SPEAKERS_DIR", "INDEX_TTS_SPEAKERS_DIR", default=str(root / "speakers"))
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
            log_level=(_pick("LOCALAI_LOG_LEVEL", default="INFO") or "INFO").upper(),
            poll_interval=float(_pick("LOCALAI_POLL", "INDEX_TTS_POLL", default="0.4") or 0.4),
            job_timeout=float(_pick("LOCALAI_TIMEOUT", "INDEX_TTS_TIMEOUT", default="180") or 180),
        )

    # ---- 给原生引擎写回环境变量 -----------------------------------------
    def export_env(self) -> None:
        """把配置写回环境变量，供 ``engine/`` 下的原生引擎模块读取。

        原生模块（engine.py / asr_engine.py / ...）在 import 时读取
        ``INDEX_TTS_*`` 环境变量并据此推导路径；服务层统一在这里注入，
        保证「配置只有一个来源」，同时不修改这些模块。
        """
        mapping = {
            "TTS_BACKEND": self.tts_backend,
            "INDEX_TTS_HOME": str(self.home),
            "INDEX_TTS_MODEL_DIR": str(self.model_dir),
            "INDEX_TTS_ASR_MODEL_DIR": str(self.asr_model_dir),
            "INDEX_TTS_CAMPPUS_DIR": str(self.speaker_model_dir),
            "INDEX_TTS_SPEAKERS_DIR": str(self.speakers_dir),
            "INDEX_TTS_TORCH_HOME": str(self.torch_home),
            "INDEX_TTS_DEMUCS_REPO": str(self.demucs_repo),
            "INDEX_TTS_VOICES": str(self.voices_dir),
            "INDEX_TTS_OUTPUT": str(self.output_dir),
            "INDEX_TTS_VRAM_ESTIMATES": str(self.vram_estimates_path),
            "INDEX_TTS_VRAM_MARGIN_MB": str(self.vram_margin_mb),
            "INDEX_TTS_IDLE_UNLOAD": str(self.idle_unload),
            "INDEX_TTS_ASR_IDLE_UNLOAD": str(self.asr_idle_unload),
            "INDEX_TTS_SPEAKER_IDLE_UNLOAD": str(self.speaker_idle_unload),
            "INDEX_TTS_SEPARATE_IDLE_UNLOAD": str(self.separate_idle_unload),
            "TORCH_HOME": str(self.torch_home),
        }
        if self.ffmpeg is not None:
            mapping["INDEX_TTS_FFMPEG"] = str(self.ffmpeg)
        if self.ffprobe is not None:
            mapping["INDEX_TTS_FFPROBE"] = str(self.ffprobe)
        if self.comfy_base:
            mapping["COMFY_BASE"] = self.comfy_base
        os.environ.update(mapping)

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
            "model_dir": str(self.model_dir),
            "asr_model_dir": str(self.asr_model_dir),
            "speaker_model_dir": str(self.speaker_model_dir),
            "voices_dir": str(self.voices_dir),
            "output_dir": str(self.output_dir),
            "logs_dir": str(self.logs_dir),
            "speakers_dir": str(self.speakers_dir),
            "torch_home": str(self.torch_home),
            "demucs_repo": str(self.demucs_repo),
            "ffmpeg": str(self.ffmpeg) if self.ffmpeg else None,
            "ffmpeg_source": self.ffmpeg_source,
            "ffprobe": str(self.ffprobe) if self.ffprobe else None,
            "ffprobe_source": self.ffprobe_source,
            "vram_margin_mb": self.vram_margin_mb,
            "idle_unload": {
                "tts": self.idle_unload,
                "asr": self.asr_idle_unload,
                "speaker": self.speaker_idle_unload,
                "separate": self.separate_idle_unload,
            },
            "auth_required": bool(self.token),
        }


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
        cfg.speakers_dir,
        cfg.asr_tmp_dir,
        cfg.media_tmp_dir,
        cfg.frames_dir,
        cfg.speaker_tmp_dir,
        cfg.separate_tmp_dir,
        cfg.separate_dir,
        cfg.vram_estimates_path.parent,
    ]
    created: list[Path] = []
    for path in wanted:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created
