"""音色目录：每次请求重新扫描，新增/删除文件不需要重启。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import soundfile as sf

from ..config import get_config
from ..constants import AUDIO_EXTENSIONS
from .errors import VoiceNotFound

logger = logging.getLogger("localai.voices")


@dataclass
class Voice:
    name: str
    path: Path
    duration: Optional[float] = None
    format: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "filename": self.path.name,
            "format": self.format,
            "duration": round(self.duration, 3) if self.duration else None,
        }


def _probe_duration(path: Path) -> Optional[float]:
    """用 soundfile 读时长；读不出来（如部分 m4a）就返回 None，不影响使用。"""
    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:  # noqa: BLE001 - 时长只是锦上添花
        return None


def scan_voices() -> list[Voice]:
    """扫描音色目录。同名文件按文件名排序，第二个起自动加 _2、_3 后缀。"""
    voices_dir = get_config().voices_dir
    voices_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = sorted(
            (p for p in voices_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
    except OSError as exc:
        logger.error("扫描音色目录失败 %s: %s", voices_dir, exc)
        return []

    voices: list[Voice] = []
    used_names: set[str] = set()
    for path in files:
        stem = path.stem
        name = stem
        suffix = 2
        while name in used_names:
            name = f"{stem}_{suffix}"
            suffix += 1
        used_names.add(name)
        voices.append(
            Voice(
                name=name,
                path=path,
                duration=_probe_duration(path),
                format=path.suffix.lower().lstrip("."),
            )
        )
    return voices


def resolve_voice(name: str) -> Voice:
    voices = scan_voices()
    for voice in voices:
        if voice.name == name:
            return voice
    available = ", ".join(v.name for v in voices) or "（空）"
    raise VoiceNotFound(f"音色 {name!r} 不存在。可用音色：{available}")


def pick_voice_name(raw: str) -> str:
    """把「音色名 / 文件名 / 完整路径」统一归一成音色名（DSH 兼容接口用）。"""
    value = (raw or "").strip()
    if not value:
        voices = scan_voices()
        return voices[0].name if voices else "default"
    name = Path(value).name
    if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
        name = Path(name).stem
    voices = scan_voices()
    if any(v.name == name for v in voices):
        return name
    # 传的是不认识的名字：回落到第一个可用音色，避免整条请求失败。
    return voices[0].name if voices else name
