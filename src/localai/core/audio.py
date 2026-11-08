"""音频小工具：WAV 转换、归档命名、HTTP 响应头。"""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import soundfile as sf

from ..config import get_config
from .errors import ComfyError


def to_wav(raw: bytes) -> bytes:
    """把任意（ComfyUI 常产出 FLAC）音频转成 16bit PCM WAV；本来就是 WAV 就原样返回。"""
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return raw
    try:
        with sf.SoundFile(io.BytesIO(raw)) as source:
            audio = source.read(dtype="float32", always_2d=True)
            sample_rate = source.samplerate
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise ComfyError(f"把上游输出转换为 WAV 失败：{exc}") from exc


def safe_name(value: str, fallback: str = "voice") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._-")
    return cleaned[:40] or fallback


def archive_wav(wav: bytes, voice_name: str, seed: int) -> Path:
    """服务端留档：``<OUTPUT>/20261108-123456_default_123.wav``。"""
    output_dir = get_config().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{safe_name(voice_name)}_{seed}"
    path = output_dir / f"{stem}.wav"
    counter = 2
    while path.exists():
        path = output_dir / f"{stem}_{counter}.wav"
        counter += 1
    path.write_bytes(wav)
    return path


def archive_engine_wav(tmp_path: Path, voice_name: str, seed: int) -> Path:
    """把引擎写出的临时 WAV 改名成和 Comfy 模式一致的留档名。"""
    output_dir = get_config().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{safe_name(voice_name)}_{seed}"
    path = output_dir / f"{stem}.wav"
    counter = 2
    while path.exists():
        path = output_dir / f"{stem}_{counter}.wav"
        counter += 1
    try:
        tmp_path.replace(path)
    except OSError:
        # 极端情况下（跨盘/被占用）退回复制+删除
        path.write_bytes(tmp_path.read_bytes())
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return path


def audio_headers(path: Optional[Path], seed: int) -> dict[str, str]:
    headers = {"Cache-Control": "no-store", "X-Seed": str(seed)}
    if path is None:
        return headers
    filename = path.name
    headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    headers["X-Audio-Filename"] = filename
    try:
        headers["X-Audio-Path"] = str(path)
        headers["X-Audio-Path"].encode("latin-1")
    except UnicodeEncodeError:
        # 输出目录含中文时 HTTP 头不能直接放，改成百分号编码。
        headers["X-Audio-Path"] = quote(str(path), safe="")
    return headers
