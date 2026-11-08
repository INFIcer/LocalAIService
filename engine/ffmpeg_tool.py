"""
ffmpeg 媒体工具（本地 AI 服务共用）
====================================

给 TTS / ASR / 声纹 / 视频处理提供一层薄封装，只依赖 ffmpeg 与 ffprobe 可执行文件，
不 import torch / funasr，所以随时 import 都很便宜。

定位可执行文件（按顺序）：
1. 环境变量 ``INDEX_TTS_FFMPEG`` / ``INDEX_TTS_FFPROBE``（推荐换机时设置）；
2. 本机默认路径 ``C:\\Users\\Administrator\\Documents\\videoTrans\\ffmpeg\\bin\\``；
3. PATH 里的 ``ffmpeg`` / ``ffprobe``。

两个环境变量都找不到时会抛 ``MediaError``，错误信息里带上期望路径，方便排查。

对外函数：
    status()                             ffmpeg / ffprobe 是否就绪
    probe(path)                          媒体信息（时长、容器、音频参数、视频分辨率/帧率）
    to_wav16k(src, dst=None)             任意音视频 -> 16kHz 单声道 PCM16 wav（ASR 用）
    extract_frames(src, out_dir, ...)    抽帧（按数量 / 时间点 / fps），返回文件与尺寸
    extract_frames_window(src, start, end) 在 [start,end] 窗口内按 fps 抽帧，命名带绝对时间点
    clip(src, start, end, ...)          裁剪一段（默认重编码成 h264 mp4，时间点精确）
    extract_audio(src, out_dir, fmt=...) 只抽音轨（保留原采样率）
    detect_scenes(src, ...)              场景/镜头边界检测（ffmpeg scene 分数）

命令行自检：
    python ffmpeg_tool.py status
    python ffmpeg_tool.py probe  D:\\a.mp4
    python ffmpeg_tool.py wav16k D:\\a.mp4 --out D:\\a.wav
    python ffmpeg_tool.py frames D:\\a.mp4 --out D:\\frames --count 3
    python ffmpeg_tool.py window D:\\a.mp4 --start 200 --end 260 --fps 1 --out D:\\frames
    python ffmpeg_tool.py clip   D:\\a.mp4 --start 200 --end 260 --out D:\\clips
    python ffmpeg_tool.py audio  D:\\a.mp4 --out D:\\audio --format mp3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 本机默认安装位置（videoTrans 项目自带的全功能构建，2025-12-31）。
DEFAULT_FFMPEG = r"C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffmpeg.exe"
DEFAULT_FFPROBE = r"C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffprobe.exe"

# 单条命令默认超时（秒）；大视频转码可以调大。
DEFAULT_TIMEOUT = float(os.environ.get("INDEX_TTS_FFMPEG_TIMEOUT", "300"))

# 抽帧数量上限：防止调用方误传 count=100000 把磁盘写满。
MAX_FRAMES = 200

# Windows 下用 CREATE_NO_WINDOW：服务是后台进程，避免每跑一次 ffmpeg 闪一个黑框。
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class MediaError(Exception):
    """ffmpeg / ffprobe 相关错误；message 里带可直接给用户看的中文说明。"""


# ---------------------------------------------------------------------------
# 定位可执行文件
# ---------------------------------------------------------------------------


def _resolve_binary(env_name: str, default: str, exe_name: str) -> Path:
    """按「环境变量 -> 默认路径 -> PATH」的顺序找可执行文件。

    环境变量一旦设置就**必须**有效：路径写错时直接报错，而不是悄悄退回默认路径，
    否则换机后会出现「明明配了却还用旧路径」的迷惑行为。
    """
    env_value = (os.environ.get(env_name) or "").strip().strip('"')
    if env_value:
        candidate = Path(env_value)
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
        raise MediaError(
            f"{env_name} 指向的文件不存在：{env_value}。"
            f"请修正该环境变量，或删除它（删除后会回退到默认路径 {default}）。"
        )

    try:
        if Path(default).is_file():
            return Path(default)
    except OSError:
        pass

    found = shutil.which(exe_name)
    if found:
        return Path(found)

    raise MediaError(
        f"找不到 {exe_name}。请设置环境变量 {env_name}，或确认默认路径存在：{default}"
    )


def ffmpeg_path() -> Path:
    """ffmpeg.exe 的绝对路径。"""
    return _resolve_binary("INDEX_TTS_FFMPEG", DEFAULT_FFMPEG, "ffmpeg")


def ffprobe_path() -> Path:
    """ffprobe.exe 的绝对路径。"""
    env_name = "INDEX_TTS_FFPROBE"
    default = DEFAULT_FFPROBE
    # 只设置了 INDEX_TTS_FFMPEG 时，优先用同目录的 ffprobe，避免两个变量配不一致。
    ffmpeg_env = (os.environ.get("INDEX_TTS_FFMPEG") or "").strip().strip('"')
    if ffmpeg_env and not (os.environ.get(env_name) or "").strip():
        sibling = Path(ffmpeg_env).with_name("ffprobe.exe")
        if sibling.is_file():
            return sibling
    return _resolve_binary(env_name, default, "ffprobe")


def status() -> dict[str, Any]:
    """给 /health 用的探活结果，找不到时把期望路径一起返回。"""
    result: dict[str, Any] = {"ffmpeg": "missing", "path": None, "ffprobe": "missing"}
    try:
        result["path"] = str(ffmpeg_path())
        result["ffmpeg"] = "ok"
    except MediaError as exc:
        result["expected"] = DEFAULT_FFMPEG
        result["detail"] = str(exc)
        return result
    try:
        result["ffprobe_path"] = str(ffprobe_path())
        result["ffprobe"] = "ok"
    except MediaError as exc:
        result["ffprobe_expected"] = DEFAULT_FFPROBE
        result["ffprobe_detail"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# 执行 ffmpeg / ffprobe
# ---------------------------------------------------------------------------


def _run(
    args: Sequence[str],
    *,
    exe: Optional[Path] = None,
    timeout: Optional[float] = None,
    probe: bool = False,
) -> subprocess.CompletedProcess:
    """执行一条命令；失败时把 stderr 尾部 20 行塞进 MediaError。"""
    binary = exe or (ffprobe_path() if probe else ffmpeg_path())
    # ffprobe 不认 -y；ffmpeg 统一加 -y 覆盖输出，免得卡在交互确认上。
    head = ["-hide_banner"]
    if not probe:
        head.append("-y")
    cmd = [str(binary), *head, *[str(item) for item in args]]
    wait = DEFAULT_TIMEOUT if timeout is None else float(timeout)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=wait,
            creationflags=_CREATE_NO_WINDOW,
        )
    except FileNotFoundError as exc:  # 理论上前面已定位过，兜底
        raise MediaError(f"无法启动 {binary}：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{binary.name} 执行超时（>{wait:.0f} 秒）：{' '.join(cmd)}") from exc

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-20:]) or "(stderr 为空)"
        raise MediaError(f"{binary.name} 执行失败（退出码 {proc.returncode}）：\n{tail}")
    return proc


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_fps(value: Any) -> Optional[float]:
    """把 ffprobe 的 ``30000/1001`` 这种分数帧率换算成小数。"""
    if value is None:
        return None
    text = str(value).strip()
    if "/" in text:
        num, _, den = text.partition("/")
        numerator = _as_float(num)
        denominator = _as_float(den)
        if numerator is None or not denominator:
            return None
        return round(numerator / denominator, 3)
    parsed = _as_float(text)
    return None if parsed is None else round(parsed, 3)


def _require_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_file():
        raise MediaError(f"文件不存在：{target}")
    return target


# ---------------------------------------------------------------------------
# 媒体信息
# ---------------------------------------------------------------------------


def probe(path: str | Path) -> dict[str, Any]:
    """读取媒体信息：时长、容器、音频参数、视频分辨率/帧率/编码。

    返回示例::

        {
          "path": "D:\\\\a.mp4",
          "duration": 5.0,
          "format": {"name": "mov,mp4,m4a,3gp,3g2,mj2", "long_name": "QuickTime / MOV", ...},
          "audio": {"codec": "aac", "sample_rate": 44100, "channels": 2, ...},
          "video": {"codec": "h264", "width": 640, "height": 360, "fps": 25.0, ...}
        }
    """
    target = _require_file(path)
    proc = _run(
        ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(target)],
        probe=True,
    )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe 返回的不是合法 JSON：{exc}") from exc

    fmt = data.get("format") or {}
    streams = data.get("streams") or []

    audio_stream: Optional[dict[str, Any]] = None
    video_stream: Optional[dict[str, Any]] = None
    for stream in streams:
        codec_type = (stream.get("codec_type") or "").lower()
        if codec_type == "audio" and audio_stream is None:
            audio_stream = stream
        elif codec_type == "video" and video_stream is None:
            # 封面图（attached_pic=1）不算视频流
            disposition = stream.get("disposition") or {}
            if int(disposition.get("attached_pic") or 0) == 1:
                continue
            video_stream = stream

    duration = _as_float(fmt.get("duration"))
    if duration is None:
        for stream in (video_stream, audio_stream):
            if stream:
                duration = _as_float(stream.get("duration"))
                if duration is not None:
                    break

    result: dict[str, Any] = {
        "path": str(target),
        "duration": None if duration is None else round(duration, 3),
        "size": _as_int(fmt.get("size")),
        "bit_rate": _as_int(fmt.get("bit_rate")),
        "format": {
            "name": fmt.get("format_name"),
            "long_name": fmt.get("format_long_name"),
            "duration": None if duration is None else round(duration, 3),
            "size": _as_int(fmt.get("size")),
            "bit_rate": _as_int(fmt.get("bit_rate")),
        },
        "audio": None,
        "video": None,
        "streams": len(streams),
    }

    if audio_stream:
        result["audio"] = {
            "codec": audio_stream.get("codec_name"),
            "codec_long_name": audio_stream.get("codec_long_name"),
            "sample_rate": _as_int(audio_stream.get("sample_rate")),
            "channels": _as_int(audio_stream.get("channels")),
            "channel_layout": audio_stream.get("channel_layout"),
            "bit_rate": _as_int(audio_stream.get("bit_rate")),
            "duration": _as_float(audio_stream.get("duration")),
        }

    if video_stream:
        result["video"] = {
            "codec": video_stream.get("codec_name"),
            "codec_long_name": video_stream.get("codec_long_name"),
            "width": _as_int(video_stream.get("width")),
            "height": _as_int(video_stream.get("height")),
            "fps": _parse_fps(video_stream.get("avg_frame_rate"))
            or _parse_fps(video_stream.get("r_frame_rate")),
            "frames": _as_int(video_stream.get("nb_frames")),
            "pix_fmt": video_stream.get("pix_fmt"),
            "duration": _as_float(video_stream.get("duration")),
            "bit_rate": _as_int(video_stream.get("bit_rate")),
        }

    if not audio_stream and not video_stream:
        raise MediaError(f"这个文件里没有可识别的音频或视频流：{target}")
    return result


# ---------------------------------------------------------------------------
# 转 16k 单声道 wav（ASR 输入标准）
# ---------------------------------------------------------------------------


def to_wav16k(src: str | Path, dst: str | Path | None = None, *, timeout: float | None = None) -> dict[str, Any]:
    """任意音视频 -> 16kHz / 单声道 / PCM16 wav。

    不传 ``dst`` 时写到系统临时目录（调用方负责删除）。返回::

        {"path": "...wav", "duration": 5.0, "sample_rate": 16000, "channels": 1, "size": 160044}

    注意：即使源文件已经是 16k 单声道 wav 也会重新编码一遍，保证下游拿到的格式绝对统一。
    """
    source = _require_file(src)
    created_temp = False
    if dst is None:
        fd, temp_name = tempfile.mkstemp(prefix="ffmpeg_wav16k_", suffix=".wav")
        os.close(fd)
        target = Path(temp_name)
        created_temp = True
    else:
        target = Path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)

    try:
        _run(
            [
                "-v", "error",
                "-i", str(source),
                "-vn", "-sn", "-dn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                str(target),
            ],
            timeout=timeout,
        )
        if not target.is_file() or target.stat().st_size == 0:
            raise MediaError(f"ffmpeg 没有生成 wav 输出：{target}")
        info = probe(target)
    except Exception:
        if created_temp:
            target.unlink(missing_ok=True)
        raise

    return {
        "path": str(target),
        "duration": info["duration"],
        "sample_rate": 16000,
        "channels": 1,
        "size": target.stat().st_size,
        "temp": created_temp,
    }


# ---------------------------------------------------------------------------
# 转 44.1k 立体声 wav（Demucs 分离输入标准）
# ---------------------------------------------------------------------------


def to_wav44k(src: str | Path, dst: str | Path | None = None, *, timeout: float | None = None) -> dict[str, Any]:
    """任意音视频 -> 44.1kHz / 立体声 / PCM16 wav。

    Demucs 的 htdemucs 系列模型固定按 44.1kHz 立体声训练，先把输入归一成这个
    格式，能避免各容器/采样率带来的隐性重采样差异。返回::

        {"path": "...wav", "duration": 9.752, "sample_rate": 44100, "channels": 2, "size": 1720398}

    不传 ``dst`` 时写到系统临时目录（调用方负责删除）；传了会自动建父目录。
    """
    source = _require_file(src)
    created_temp = False
    if dst is None:
        fd, temp_name = tempfile.mkstemp(prefix="ffmpeg_wav44k_", suffix=".wav")
        os.close(fd)
        target = Path(temp_name)
        created_temp = True
    else:
        target = Path(dst)
        target.parent.mkdir(parents=True, exist_ok=True)

    try:
        _run(
            [
                "-v", "error",
                "-i", str(source),
                "-vn", "-sn", "-dn",
                "-ac", "2",
                "-ar", "44100",
                "-c:a", "pcm_s16le",
                "-f", "wav",
                str(target),
            ],
            timeout=timeout,
        )
        if not target.is_file() or target.stat().st_size == 0:
            raise MediaError(f"ffmpeg 没有生成 wav 输出：{target}")
        info = probe(target)
    except Exception:
        if created_temp:
            target.unlink(missing_ok=True)
        raise

    return {
        "path": str(target),
        "duration": info["duration"],
        "sample_rate": 44100,
        "channels": 2,
        "size": target.stat().st_size,
        "temp": created_temp,
    }


# ---------------------------------------------------------------------------
# 抽帧
# ---------------------------------------------------------------------------


def _image_size(path: Path) -> tuple[Optional[int], Optional[int]]:
    """不装 Pillow，直接读 JPEG/PNG 文件头拿宽高。"""
    try:
        with path.open("rb") as fh:
            head = fh.read(512 * 1024)
    except OSError:
        return None, None

    # PNG: 8 字节签名 + IHDR（宽高在 16..24）
    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)

    # JPEG: 逐段扫描，遇到 SOFn 就读高宽
    if head[:2] != b"\xff\xd8":
        return None, None
    index = 2
    while index + 9 < len(head):
        if head[index] != 0xFF:
            index += 1
            continue
        marker = head[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        segment_length = struct.unpack(">H", head[index + 2:index + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height, width = struct.unpack(">HH", head[index + 5:index + 9])
            return int(width), int(height)
        index += 2 + segment_length
    return None, None


def _frame_name(index: int, seconds: float, fmt: str) -> str:
    """帧文件名带时间点，例如 ``frame_0001_12.500s.jpg``。"""
    return f"frame_{index:04d}_{seconds:.3f}s.{fmt}"


def _grab_series(
    source: Path,
    out_dir: Path,
    times: Iterable[float],
    *,
    fmt: str,
    width: Optional[int],
    quality: int,
    timeout: Optional[float],
) -> list[dict[str, Any]]:
    """按时间点逐张抽帧（输入定位 -ss，快且时间点准）。"""
    frames: list[dict[str, Any]] = []
    for index, seconds in enumerate(times, start=1):
        out_path = out_dir / _frame_name(index, seconds, fmt)
        args = [
            "-v", "error",
            "-ss", f"{max(0.0, seconds):.3f}",
            "-i", str(source),
            "-frames:v", "1",
            "-an", "-sn", "-dn",
        ]
        if width:
            # -2 保证高度是偶数（jpg/png 编码器要求）
            args += ["-vf", f"scale={int(width)}:-2"]
        if fmt == "jpg":
            args += ["-q:v", str(int(quality))]
        else:
            args += ["-compression_level", "6"]
        args.append(str(out_path))
        _run(args, timeout=timeout)

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise MediaError(f"在 {seconds:.3f}s 处没有抽到画面（可能超出视频时长）：{source}")
        frame_width, frame_height = _image_size(out_path)
        frames.append(
            {
                "path": str(out_path),
                "time_sec": round(seconds, 3),
                "width": frame_width,
                "height": frame_height,
                "size": out_path.stat().st_size,
            }
        )
    return frames


def _grab_by_fps(
    source: Path,
    out_dir: Path,
    fps: float,
    *,
    fmt: str,
    width: Optional[int],
    quality: int,
    timeout: Optional[float],
) -> list[dict[str, Any]]:
    """按 fps 抽帧：一次 ffmpeg 跑完，再把序号换算成时间点重命名。"""
    pattern = out_dir / f"raw_%05d.{fmt}"
    filters = [f"fps={fps}"]
    if width:
        filters.append(f"scale={int(width)}:-2")
    args = ["-v", "error", "-i", str(source), "-an", "-sn", "-dn", "-vf", ",".join(filters)]
    if fmt == "jpg":
        args += ["-q:v", str(int(quality))]
    else:
        args += ["-compression_level", "6"]
    args.append(str(pattern))
    _run(args, timeout=timeout)

    raw_files = sorted(out_dir.glob(f"raw_*.{fmt}"))
    if not raw_files:
        raise MediaError(f"按 fps={fps} 没有抽到任何帧：{source}")

    frames: list[dict[str, Any]] = []
    for index, raw_path in enumerate(raw_files, start=1):
        seconds = (index - 1) / float(fps)
        final_path = out_dir / _frame_name(index, seconds, fmt)
        raw_path.replace(final_path)
        frame_width, frame_height = _image_size(final_path)
        frames.append(
            {
                "path": str(final_path),
                "time_sec": round(seconds, 3),
                "width": frame_width,
                "height": frame_height,
                "size": final_path.stat().st_size,
            }
        )
    return frames


def extract_frames(
    src: str | Path,
    out_dir: str | Path,
    *,
    count: Optional[int] = None,
    times: Optional[Sequence[float]] = None,
    fps: Optional[float] = None,
    width: Optional[int] = None,
    fmt: str = "jpg",
    quality: int = 3,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """抽帧成 jpg/png。

    三种模式**只能选一种**（都不传时默认抽 1 张）：

    - ``count=N``：在整段视频里均匀抽 N 张（取每段的中点，避开片头黑场）；
    - ``times=[秒, ...]``：在指定时间点各抽 1 张；
    - ``fps=2``：按每秒 2 张抽。

    ``width`` 指定输出宽度（高度按比例，自动取偶数）。``quality`` 是 jpg 的
    ``-q:v``（1 最好、31 最差，默认 3）。返回
    ``[{"path", "time_sec", "width", "height", "size"}, ...]``。
    """
    source = _require_file(src)
    fmt = (fmt or "jpg").strip().lower().lstrip(".")
    if fmt not in ("jpg", "jpeg", "png"):
        raise MediaError(f"不支持的图片格式：{fmt}（只支持 jpg / png）")
    if fmt == "jpeg":
        fmt = "jpg"

    modes = [value is not None for value in (times, count, fps)]
    if sum(modes) > 1:
        raise MediaError("count / times / fps 只能给一个")
    if width is not None and int(width) <= 0:
        raise MediaError(f"width 必须是正整数，收到 {width}")

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    info = probe(source)
    duration = info.get("duration")
    if not info.get("video"):
        raise MediaError(f"这个文件没有视频流，抽不了帧：{source}")
    if not duration or duration <= 0:
        raise MediaError(f"读不出视频时长，无法抽帧：{source}")

    if times is not None:
        points = [float(value) for value in times]
        if not points:
            raise MediaError("times 不能是空列表")
        if len(points) > MAX_FRAMES:
            raise MediaError(f"一次最多抽 {MAX_FRAMES} 张，收到 {len(points)}")
        for point in points:
            if point < 0 or point > duration:
                raise MediaError(f"时间点 {point:.3f}s 超出视频范围 0~{duration:.3f}s")
        # 落在最后一帧之后一点点会抽不到画面，往前收 50ms 兜底。
        points = [min(point, max(0.0, duration - 0.05)) for point in points]
        return _grab_series(
            source, target_dir, points, fmt=fmt, width=width, quality=quality, timeout=timeout
        )

    if count is not None:
        number = int(count)
        if number <= 0:
            raise MediaError(f"count 必须是正整数，收到 {count}")
        if number > MAX_FRAMES:
            raise MediaError(f"一次最多抽 {MAX_FRAMES} 张，收到 {number}")
        step = duration / number
        points = [step * (index + 0.5) for index in range(number)]
        return _grab_series(
            source, target_dir, points, fmt=fmt, width=width, quality=quality, timeout=timeout
        )

    if fps is not None:
        rate = float(fps)
        if rate <= 0:
            raise MediaError(f"fps 必须大于 0，收到 {fps}")
        estimated = int(duration * rate) + 1
        if estimated > MAX_FRAMES:
            raise MediaError(
                f"按 fps={rate} 预计要抽 {estimated} 帧，超过上限 {MAX_FRAMES}；请调小 fps"
            )
        return _grab_by_fps(
            source, target_dir, rate, fmt=fmt, width=width, quality=quality, timeout=timeout
        )

    # 什么都不传：抽中间那一张
    return _grab_series(
        source, target_dir, [duration / 2], fmt=fmt, width=width, quality=quality, timeout=timeout
    )


# ---------------------------------------------------------------------------
# 裁剪一段 + 窗口抽帧（细看某段）
# ---------------------------------------------------------------------------

#: 窗口抽帧允许的 fps 范围（太小看不出连续性，太大 token 爆）。
WINDOW_MIN_FPS = 0.5
WINDOW_MAX_FPS = 3.0
#: 单次窗口抽帧最多输出多少张（防止窗口 + fps 组合把磁盘写满）。
WINDOW_MAX_FRAMES = 120


def _validate_window(
    src: str | Path, start: Any, end: Any
) -> tuple[Path, dict[str, Any], float, float]:
    """校验 ``0 <= start < end <= duration``，返回归一后的窗口。

    时间点允许 50ms 的浮点误差（调用方常用 round 后的秒数）。
    """
    source = _require_file(src)
    info = probe(source)
    if not info.get("video"):
        raise MediaError(f"这个文件没有视频流：{source}")
    duration = info.get("duration")
    if not duration or float(duration) <= 0:
        raise MediaError(f"读不出视频时长，无法裁剪/抽窗口帧：{source}")
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"start/end 必须是数字，收到 {start!r} / {end!r}") from exc
    if start_value < 0:
        raise MediaError(f"start 不能是负数，收到 {start_value}")
    if end_value <= start_value:
        raise MediaError(f"end 必须大于 start（start={start_value}, end={end_value}）")
    if end_value > float(duration) + 0.05:
        raise MediaError(
            f"end={end_value:.3f}s 超出视频时长 {float(duration):.3f}s"
        )
    end_value = min(end_value, float(duration))
    return source, info, round(start_value, 3), round(end_value, 3)


def clip(
    src: str | Path,
    start: float,
    end: float,
    *,
    out: str | Path | None = None,
    reencode: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """把 ``[start, end]`` 剪成一段独立视频，时间点精确到裁剪位置。

    - 默认**重编码**成 h264 mp4（``libx264 -preset veryfast -crf 23`` + aac），
      这样 ``-ss`` 能按解码时间精确切，不依赖关键帧位置；
    - ``reencode=False`` 时走 ``-c copy``（很快，但会从最近的关键帧开始，
      实际时长可能比请求的长）；
    - ``out`` 不给时落到系统临时目录 ``localai_clips/``，文件名带起止秒数。

    返回 ``{"path", "start", "end", "duration", "size_mb"}``；
    失败抛 :class:`MediaError`，message 里带 ffmpeg stderr 尾部（由 ``_run`` 负责）。
    """
    source, _info, start_value, end_value = _validate_window(src, start, end)
    span = round(end_value - start_value, 3)
    if span <= 0:
        raise MediaError(f"裁剪窗口为空（start={start_value}, end={end_value}）")

    if out is None:
        target_dir = Path(tempfile.gettempdir()) / "localai_clips"
        target = target_dir / (
            f"{source.stem}_clip_{start_value:.3f}-{end_value:.3f}.mp4"
        )
    else:
        target = Path(out)
        if target.suffix == "":
            target = target.with_suffix(".mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    # -ss 放在 -i 前面：重编码时会先跳到最近关键帧再解码丢弃，速度与精度兼顾；
    # -t 用窗口长度而不是 -to，避免部分容器时间基准不一致。
    args = [
        "-v", "error",
        "-ss", f"{start_value:.3f}",
        "-i", str(source),
        "-t", f"{span:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-sn", "-dn",
    ]
    if reencode:
        args += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
        ]
    else:
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    args.append(str(target))
    _run(args, timeout=timeout)

    if not target.is_file() or target.stat().st_size == 0:
        raise MediaError(f"ffmpeg 没有生成裁剪结果：{target}")
    out_info = probe(target)
    out_duration = out_info.get("duration")
    if out_duration is None:
        out_duration = span
    return {
        "path": str(target),
        "start": start_value,
        "end": end_value,
        "duration": round(float(out_duration), 3),
        "size_mb": round(target.stat().st_size / 1024**2, 2),
    }


def extract_frames_window(
    src: str | Path,
    start: float,
    end: float,
    *,
    fps: float = 1.0,
    width: int | None = None,
    out_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """在 ``[start, end]`` 窗口内按 ``fps`` 抽帧，文件名带**绝对时间点**。

    和 :func:`extract_frames` 的 ``fps`` 模式不同，这里只处理一个窗口，
    并且把 ffmpeg 输出的相对序号换算成 ``start + index / fps``，
    返回 ``[{"path", "time_sec", "width", "height"}]``，供「细看某段」直接使用。

    ``fps`` 限制在 0.5~3；窗口时长 × fps 超过 ``WINDOW_MAX_FRAMES`` 时直接拒绝，
    避免一次抽几百张把 VLM 上下文和磁盘都撑爆。
    """
    source, _info, start_value, end_value = _validate_window(src, start, end)
    span = end_value - start_value
    try:
        rate = float(fps)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"fps 必须是数字，收到 {fps!r}") from exc
    if not WINDOW_MIN_FPS <= rate <= WINDOW_MAX_FPS:
        raise MediaError(
            f"fps 必须在 {WINDOW_MIN_FPS}~{WINDOW_MAX_FPS} 之间，收到 {rate}"
        )
    estimated = int(span * rate) + 1
    if estimated > WINDOW_MAX_FRAMES:
        raise MediaError(
            f"窗口 {span:.1f}s × fps={rate} 预计 {estimated} 张，"
            f"超过上限 {WINDOW_MAX_FRAMES}；请调小 fps 或缩短窗口"
        )
    if width is not None and int(width) <= 0:
        raise MediaError(f"width 必须是正整数，收到 {width}")

    target_dir = Path(out_dir) if out_dir is not None else (
        Path(tempfile.gettempdir()) / "localai_windows"
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # 每次调用一个唯一前缀：并发/连续裁剪时不会互相抢 raw 文件。
    token = f"{int(time.time() * 1000):013d}_{os.getpid()}_{id(src) % 10000:04d}"
    pattern = target_dir / f"raw_win_{token}_%05d.jpg"
    filters = [f"fps={rate}"]
    if width:
        filters.append(f"scale={int(width)}:-2")
    args = [
        "-v", "error",
        "-ss", f"{start_value:.3f}",
        "-i", str(source),
        "-t", f"{span:.3f}",
        "-an", "-sn", "-dn",
        "-vf", ",".join(filters),
        "-q:v", "3",
        str(pattern),
    ]
    _run(args, timeout=None)

    raw_files = sorted(target_dir.glob(f"raw_win_{token}_*.jpg"))
    if not raw_files:
        raise MediaError(
            f"窗口 {start_value:.3f}~{end_value:.3f}s 内没有抽到画面：{source}"
        )

    frames: list[dict[str, Any]] = []
    for index, raw_path in enumerate(raw_files):
        seconds = round(start_value + index / rate, 3)
        final_path = target_dir / f"window_{index + 1:04d}_{seconds:.3f}s.jpg"
        raw_path.replace(final_path)
        frame_width, frame_height = _image_size(final_path)
        frames.append(
            {
                "path": str(final_path),
                "time_sec": seconds,
                "width": frame_width,
                "height": frame_height,
            }
        )
    return frames


# ---------------------------------------------------------------------------
# 只抽音轨
# ---------------------------------------------------------------------------


def extract_audio(
    src: str | Path,
    out_dir: str | Path,
    *,
    fmt: str = "wav",
    timeout: float | None = None,
) -> dict[str, Any]:
    """只抽音轨（不重采样，保留原来的采样率/声道）。

    - ``fmt="wav"``：PCM16 wav，采样率保持源文件；
    - ``fmt="mp3"``：libmp3lame，``-q:a 2``（约 190kbps VBR）。

    返回 ``{"path", "duration", "sample_rate", "channels", "format", "size"}``。
    """
    source = _require_file(src)
    audio_format = (fmt or "wav").strip().lower().lstrip(".")
    if audio_format not in ("wav", "mp3"):
        raise MediaError(f"不支持的音频格式：{fmt}（只支持 wav / mp3）")

    info = probe(source)
    if not info.get("audio"):
        raise MediaError(f"这个文件没有音轨，抽不出音频：{source}")

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"audio_{source.stem}.{audio_format}"
    if target.exists():
        # 同名不覆盖：加时间戳后缀
        target = target_dir / f"audio_{source.stem}_{int(time.time())}.{audio_format}"

    args = ["-v", "error", "-i", str(source), "-vn", "-sn", "-dn"]
    if audio_format == "wav":
        args += ["-c:a", "pcm_s16le", "-f", "wav"]
    else:
        args += ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]
    args.append(str(target))
    _run(args, timeout=timeout)

    if not target.is_file() or target.stat().st_size == 0:
        raise MediaError(f"ffmpeg 没有生成音频文件：{target}")

    out_info = probe(target)
    return {
        "path": str(target),
        "duration": out_info["duration"],
        "sample_rate": (out_info.get("audio") or {}).get("sample_rate"),
        "channels": (out_info.get("audio") or {}).get("channels"),
        "format": audio_format,
        "size": target.stat().st_size,
    }


# ---------------------------------------------------------------------------
# 场景 / 镜头边界检测
# ---------------------------------------------------------------------------

#: 场景分数阈值：越大越只挑「明显切换」。
SCENE_DEFAULT_THRESHOLD = 0.3
#: 单次最多返回的边界数（防止长视频返回几万个点）。
SCENE_MAX_SCENES = 200
#: 最小间隔（秒）：两次切换太近时只保留前一个，去掉抖动。
SCENE_MIN_GAP = 1.0

#: 从 ffmpeg 日志里抠时间点 / 场景分数。
_PTS_TIME_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scene_score=\s*([0-9]+(?:\.[0-9]+)?)")


def _debounce_scenes(
    points: list[tuple[float, float]],
    *,
    min_gap: float,
) -> list[tuple[float, float]]:
    """最小间隔去抖：时间升序，间隔小于 ``min_gap`` 的只保留第一个。

    ffmpeg 的 scene 分数在快速运镜 / 闪白时可能连续多帧超阈值，
    去抖后每个镜头只留一个边界。
    """
    kept: list[tuple[float, float]] = []
    for seconds, score in sorted(points, key=lambda item: item[0]):
        if seconds < 0:
            continue
        if kept and seconds - kept[-1][0] < float(min_gap):
            continue
        kept.append((round(seconds, 3), round(score, 4)))
    return kept


def detect_scenes(
    src: str | Path,
    *,
    threshold: float = SCENE_DEFAULT_THRESHOLD,
    max_scenes: int = SCENE_MAX_SCENES,
    min_gap: float = SCENE_MIN_GAP,
    save_dir: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """检测视频的镜头切换点（ffmpeg ``scene`` 分数）。

    实现：``-vf "select='gt(scene,TH)',metadata=mode=print:file=-,showinfo"
    -f null -``

    - ``showinfo`` 把每个被选中帧的 ``pts_time`` 打到 stderr；
    - ``metadata=mode=print`` 把 ``lavfi.scene_score`` 打到 stdout；
    - 两边按顺序对齐；万一 stdout 没解析到，就退回只用 stderr 的时间点、
      分数记 0。

    参数
    ----
    threshold:  场景分数阈值，默认 0.3（越大越严格）。
    max_scenes: 最多返回多少个边界，默认 200。
    min_gap:    最小间隔（秒），默认 1.0；更近的只保留前一个。
    save_dir:   给了就把结果写一份 ``<save_dir>/<时间戳>.json``。

    返回::

        {"count": 2, "scenes": [{"time": 1.52, "score": 1.0}, ...],
         "duration": 4.56, "threshold": 0.3, "max_scenes": 200,
         "min_gap": 1.0, "json_path": "...json"（可选）}

    错误抛 :class:`MediaError`（文件不存在 / 没有视频流 / ffmpeg 失败）。
    """
    source = _require_file(src)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"threshold 必须是数字，收到 {threshold!r}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise MediaError(f"threshold 必须在 0~1 之间，收到 {threshold}")
    max_scenes = int(max_scenes)
    if max_scenes <= 0:
        raise MediaError(f"max_scenes 必须是正整数，收到 {max_scenes}")
    try:
        min_gap = float(min_gap)
    except (TypeError, ValueError) as exc:
        raise MediaError(f"min_gap 必须是数字，收到 {min_gap!r}") from exc
    if min_gap < 0:
        raise MediaError(f"min_gap 不能是负数，收到 {min_gap}")

    info = probe(source)
    if not info.get("video"):
        raise MediaError(f"这个文件没有视频流，做不了场景检测：{source}")

    # 9 位小数足够把 ffmpeg 的 pts_time 原样带过来。
    scene_filter = (
        f"select='gt(scene,{threshold:.6f})',"
        "metadata=mode=print:file=-,showinfo"
    )
    proc = _run(
        ["-v", "info", "-i", str(source), "-vf", scene_filter, "-f", "null", "-"],
        timeout=timeout,
    )

    # stdout：成对的 pts_time / lavfi.scene_score（按帧顺序）。
    pairs: list[tuple[float, float]] = []
    pending_time: Optional[float] = None
    for line in (proc.stdout or "").splitlines():
        time_match = _PTS_TIME_RE.search(line)
        if time_match:
            pending_time = float(time_match.group(1))
            continue
        score_match = _SCENE_SCORE_RE.search(line)
        if score_match:
            score = float(score_match.group(1))
            if pending_time is not None:
                pairs.append((pending_time, score))
                pending_time = None

    # stderr：showinfo 的 pts_time（stdout 没解析到时的兜底）。
    if not pairs:
        for line in (proc.stderr or "").splitlines():
            if "Parsed_showinfo" not in line:
                continue
            match = _PTS_TIME_RE.search(line)
            if match:
                pairs.append((float(match.group(1)), 0.0))

    scenes = _debounce_scenes(pairs, min_gap=min_gap)[:max_scenes]
    result: dict[str, Any] = {
        "count": len(scenes),
        "scenes": [{"time": seconds, "score": score} for seconds, score in scenes],
        "duration": info.get("duration"),
        "threshold": threshold,
        "max_scenes": max_scenes,
        "min_gap": min_gap,
        "source": str(source),
    }

    if save_dir is not None:
        # 落盘文件名用「时间戳 + 毫秒 + 短随机」，既不撞名又一眼能看出时间。
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time() * 1000) % 1000
        json_path = target_dir / f"{stamp}_{millis:03d}.json"
        payload = {
            "source": str(source),
            "duration": info.get("duration"),
            "threshold": threshold,
            "max_scenes": max_scenes,
            "min_gap": min_gap,
            "count": len(scenes),
            "scenes": result["scenes"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["json_path"] = str(json_path)

    return result


# ---------------------------------------------------------------------------
# 命令行自检
# ---------------------------------------------------------------------------


def _main() -> int:
    parser = argparse.ArgumentParser(description="ffmpeg 媒体工具自检")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="查看 ffmpeg / ffprobe 是否就绪")

    probe_parser = sub.add_parser("probe", help="读取媒体信息")
    probe_parser.add_argument("path")

    wav_parser = sub.add_parser("wav16k", help="转成 16k 单声道 wav")
    wav_parser.add_argument("path")
    wav_parser.add_argument("--out", default=None)

    frames_parser = sub.add_parser("frames", help="抽帧")
    frames_parser.add_argument("path")
    frames_parser.add_argument("--out", required=True)
    frames_parser.add_argument("--count", type=int, default=None)
    frames_parser.add_argument("--fps", type=float, default=None)
    frames_parser.add_argument("--times", default=None, help="逗号分隔的秒数，如 1,2.5,4")
    frames_parser.add_argument("--width", type=int, default=None)
    frames_parser.add_argument("--format", default="jpg", choices=("jpg", "png"))

    audio_parser = sub.add_parser("audio", help="只抽音轨")
    audio_parser.add_argument("path")
    audio_parser.add_argument("--out", required=True)
    audio_parser.add_argument("--format", default="wav", choices=("wav", "mp3"))

    window_parser = sub.add_parser("window", help="在 [start,end] 窗口内按 fps 抽帧")
    window_parser.add_argument("path")
    window_parser.add_argument("--start", type=float, required=True)
    window_parser.add_argument("--end", type=float, required=True)
    window_parser.add_argument("--fps", type=float, default=1.0)
    window_parser.add_argument("--width", type=int, default=None)
    window_parser.add_argument("--out", required=True)

    clip_parser = sub.add_parser("clip", help="裁剪一段（默认重编码 mp4）")
    clip_parser.add_argument("path")
    clip_parser.add_argument("--start", type=float, required=True)
    clip_parser.add_argument("--end", type=float, required=True)
    clip_parser.add_argument("--out", default=None)
    clip_parser.add_argument("--copy", action="store_true", help="不重编码（快但不精确）")

    scenes_parser = sub.add_parser("scenes", help="场景/镜头边界检测")
    scenes_parser.add_argument("path")
    scenes_parser.add_argument("--threshold", type=float, default=SCENE_DEFAULT_THRESHOLD)
    scenes_parser.add_argument("--max-scenes", type=int, default=SCENE_MAX_SCENES)
    scenes_parser.add_argument("--min-gap", type=float, default=SCENE_MIN_GAP)
    scenes_parser.add_argument("--out", default=None, help="结果 json 落盘目录（可选）")

    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "probe":
        print(json.dumps(probe(args.path), ensure_ascii=False, indent=2))
        return 0

    if args.command == "scenes":
        result = detect_scenes(
            args.path,
            threshold=args.threshold,
            max_scenes=args.max_scenes,
            min_gap=args.min_gap,
            save_dir=args.out,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "wav16k":
        result = to_wav16k(args.path, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "frames":
        points = None
        if args.times:
            points = [float(part) for part in args.times.split(",") if part.strip()]
        frames = extract_frames(
            args.path,
            args.out,
            count=args.count,
            times=points,
            fps=args.fps,
            width=args.width,
            fmt=args.format,
        )
        print(json.dumps({"count": len(frames), "frames": frames}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "window":
        frames = extract_frames_window(
            args.path,
            args.start,
            args.end,
            fps=args.fps,
            width=args.width,
            out_dir=args.out,
        )
        print(json.dumps({"count": len(frames), "frames": frames}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "clip":
        result = clip(
            args.path,
            args.start,
            args.end,
            out=args.out,
            reencode=not args.copy,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result = extract_audio(args.path, args.out, fmt=args.format)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
