"""上传落盘工具：统一临时文件命名与大小限制，处理完立即删除。"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from ..config import Config
from ..constants import AUDIO_EXTENSIONS, AUDIO_MAX_BYTES, MEDIA_MAX_BYTES


def _suffix(name: str, fallback: str = ".bin") -> str:
    suffix = Path(name).suffix.lower()
    return suffix or fallback


async def save_speaker_upload(cfg: Config, file: UploadFile, prefix: str) -> Path:
    """声纹/音色接口：整段读进内存（上限 50MB），返回临时文件路径。"""
    raw_name = Path(file.filename or "audio.wav").name
    suffix = _suffix(raw_name)
    if suffix not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"只支持这些音频格式：{', '.join(sorted(AUDIO_EXTENSIONS))}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > AUDIO_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"音频太大（上限 {AUDIO_MAX_BYTES // (1024 * 1024)}MB）",
        )
    cfg.speaker_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.speaker_tmp_dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(data)
    return tmp_path


async def save_media_upload(cfg: Config, file: UploadFile, prefix: str) -> Path:
    """媒体接口：分块写临时目录（不整段读进内存），返回临时文件路径。"""
    raw_name = Path(file.filename or "upload.bin").name
    suffix = _suffix(raw_name)
    cfg.media_tmp_dir.mkdir(parents=True, exist_ok=True)
    target = cfg.media_tmp_dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MEDIA_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件太大（上限 {MEDIA_MAX_BYTES // (1024 * 1024)}MB）",
                    )
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传内容为空")
    return target


def unlink_quiet(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
