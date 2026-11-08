"""通用视频 / 音频下载引擎（yt-dlp 为核心，直链兜底）。

定位
----
本模块只做「拿一个 URL，把媒体落到本地」这件事，**不 import torch / 不依赖
服务层**，因此服务启动、doctor、独立脚本都可以随时 import。服务层通过
``localai.engines.download_ytdlp.DownloadAdapter`` 把它接进能力注册表。

两条下载路径
------------
- :func:`download`：交给 yt-dlp。支持站点解析、格式选择、合并、站点字幕、
  进度回调、取消与超时。默认 ``bv*[height<=720]+ba/b[height<=720]`` 合并成 mp4。
- :func:`download_direct`：普通 http(s) 直链。httpx 流式下载，支持 Range 断点
  续传、Content-Length 校验、进度回调，同样有大小上限与取消检查。

安全（SSRF）
------------
只允许 http/https；解析域名后拒绝私有网段 / 回环 / 链路本地 / 保留地址。
直链下载会**逐跳**检查重定向（不自动跟随）；yt-dlp 在开始前会复查探测到的
最终页面地址，并限制为同样的 scheme / 地址策略。拒绝 ``file://`` 等所有其它
scheme。测试本机临时 HTTP 服务器时可用 ``LOCALAI_DOWNLOAD_ALLOW_PRIVATE=1``
临时放开私有地址（**生产不要设**）。

Cookie
------
``cookies`` 参数接受 Netscape 格式 cookie 文件路径。服务层另外提供「按域名
自动取 cookie」：bilibili.com / b23.tv 读取本机 ``bili_credential.json``，
生成临时 Netscape 文件、用完即删；其它站点不会自动携带。cookie 内容不进日志。

环境变量
--------
``LOCALAI_HOME``                数据根目录（下载目录 = <HOME>/data/downloads/<日期>/）
``LOCALAI_FFMPEG``              ffmpeg 路径（合并 / 音频转码 / 时长探测用）
``LOCALAI_DOWNLOAD_MAX_MB``     单文件大小上限（默认 2048）
``LOCALAI_DOWNLOAD_TIMEOUT``    单次下载超时秒数（默认 1800）
``LOCALAI_BILI_COOKIE``         B站 cookie JSON 路径（默认 ~/.dsh/bili-duty/bili_credential.json）
``LOCALAI_DOWNLOAD_ALLOW_PRIVATE`` 1=允许私有地址（仅测试用）
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

import httpx

logger = logging.getLogger("download_engine")

# ---------------------------------------------------------------------------
# 常量 / 默认值
# ---------------------------------------------------------------------------

#: B站 cookie JSON 的默认位置（与刷B站插件共用；文件不提交进仓库）。
DEFAULT_BILI_COOKIE = Path(
    r"C:\Users\Administrator\.dsh\bili-duty\bili_credential.json"
)

#: yt-dlp 支持的下载画质选项。
QUALITY_CHOICES = ("720p", "best", "audio")

#: 站点字幕语言优先级：中文优先，其次英文。
SUBTITLE_LANGS = ("zh-Hans", "zh-CN", "zh", "en", "en-US", "en-GB")

#: 进度回调节流（秒）：避免每个分片都写一次任务库。
PROGRESS_INTERVAL = 0.8

#: 直链下载单次读取的块大小。
CHUNK_SIZE = 256 * 1024

#: 直链重定向最多跟随几跳（每一跳都复查目标地址）。
MAX_REDIRECTS = 5

#: 音频模式转码目标。
AUDIO_FORMAT = "m4a"

#: 浏览器 UA：部分站点会拒绝空 UA。
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class DownloadError(RuntimeError):
    """下载失败；message 里带可以直接展示给用户的原因。"""


class DownloadCancelled(DownloadError):
    """调用方要求取消（进度回调 / 检查点）。"""

    canceled = True


class DownloadTimeout(DownloadError):
    """超过整体超时时间。"""


class DownloadBlocked(DownloadError):
    """URL 被 SSRF / scheme 策略拒绝。"""


# ---------------------------------------------------------------------------
# 配置读取（模块级，不依赖 localai.config）
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def max_download_mb() -> int:
    """单文件大小上限（MB）。"""
    return max(1, _env_int("LOCALAI_DOWNLOAD_MAX_MB", 2048))


def default_timeout() -> float:
    """单次下载超时（秒）。"""
    return max(1.0, float(_env_int("LOCALAI_DOWNLOAD_TIMEOUT", 1800)))


def default_home() -> Path:
    """数据根目录：LOCALAI_HOME -> 项目根（本文件在 engine/ 下）。"""
    env = (os.environ.get("LOCALAI_HOME") or "").strip().strip('"')
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1]


def downloads_root() -> Path:
    """下载根目录：``<LOCALAI_HOME>/data/downloads``。"""
    return default_home() / "data" / "downloads"


def today_dir() -> Path:
    """当天的下载目录：``<下载根>/<YYYY-MM-DD>``。"""
    return downloads_root() / time.strftime("%Y-%m-%d")


def ffmpeg_location() -> Optional[str]:
    """ffmpeg 所在目录（yt-dlp 的 ffmpeg_location 只认目录或可执行文件）。"""
    raw = (os.environ.get("LOCALAI_FFMPEG") or "").strip().strip('"')
    if raw:
        path = Path(raw)
        if path.is_file():
            return str(path)
    # 与服务层相同的本机默认位置；换机器可用 LOCALAI_FFMPEG 覆盖。
    default = Path(r"C:\Users\Administrator\Documents\videoTrans\ffmpeg\bin\ffmpeg.exe")
    if default.is_file():
        return str(default)
    found = shutil.which("ffmpeg")
    return found or None


# ---------------------------------------------------------------------------
# SSRF 防护
# ---------------------------------------------------------------------------


def _allow_private() -> bool:
    """测试开关：仅本机临时 HTTP 服务器用例会打开。"""
    return _env_bool("LOCALAI_DOWNLOAD_ALLOW_PRIVATE")


def _ip_is_public(value: str) -> bool:
    """IP 是否属于可公开路由的地址（私有 / 回环 / 链路本地一律 False）。"""
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return bool(ip.is_global)


def validate_url(url: str, *, resolve: bool = True) -> str:
    """校验 URL 的 scheme 与目标地址，返回规范化后的字符串。

    - 只允许 http / https；
    - 拒绝 localhost、私有网段、回环、链路本地、保留地址（IPv4 / IPv6）；
    - ``resolve=False`` 时只查 scheme / 主机名，不做 DNS（用于请求参数预校验）。
    """
    text = str(url or "").strip()
    if not text:
        raise DownloadBlocked("URL 不能为空")
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise DownloadBlocked(
            f"只允许 http/https 链接，拒绝 scheme={parsed.scheme or '(空)'!r}"
        )
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise DownloadBlocked(f"URL 里没有主机名：{text}")
    if host in ("localhost", "localhost.localdomain") and not _allow_private():
        raise DownloadBlocked("拒绝访问本机地址（SSRF 防护）")
    if _allow_private():
        return text
    if not resolve:
        return text

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80))
    except socket.gaierror as exc:
        raise DownloadError(f"域名解析失败：{host}（{exc}）") from exc
    except OSError as exc:
        raise DownloadError(f"域名解析失败：{host}（{exc}）") from exc
    if not infos:
        raise DownloadError(f"域名解析不到任何地址：{host}")
    seen: set[str] = set()
    for info in infos:
        address = str(info[4][0])
        if address in seen:
            continue
        seen.add(address)
        if not _ip_is_public(address):
            raise DownloadBlocked(
                f"拒绝访问非公网地址：{host} -> {address}（SSRF 防护）"
            )
    return text


def _check_redirect_target(base: str, location: str) -> str:
    """把相对 Location 补成绝对地址，并重新做一次 SSRF 校验。"""
    from urllib.parse import urljoin

    target = urljoin(base, location)
    return validate_url(target)


# ---------------------------------------------------------------------------
# Cookie（B站自动读取）
# ---------------------------------------------------------------------------


#: 凭证 JSON 的键（大小写不敏感）-> Netscape cookie 名。
_BILI_COOKIE_FIELDS = {
    "sessdata": "SESSDATA",
    "bili_jct": "bili_jct",
    "buvid3": "buvid3",
    "dedeuserid": "DedeUserID",
    "buvid4": "buvid4",
    "ac_time_value": "ac_time_value",
}


def bili_cookie_source() -> Path:
    raw = (os.environ.get("LOCALAI_BILI_COOKIE") or "").strip().strip('"')
    return Path(raw) if raw else DEFAULT_BILI_COOKIE


def bili_site(url: str) -> bool:
    """URL 是否属于 B站（含 b23.tv 短链）。"""
    host = (urlparse(str(url or "")).hostname or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in ("bilibili.com", "b23.tv")
    )


def build_bili_cookie_file(source: Optional[Path] = None) -> Optional[Path]:
    """把 ``bili_credential.json`` 转成临时 Netscape cookie 文件。

    返回临时文件路径；文件不存在或没有可用字段时返回 ``None``。
    **不打印任何 cookie 值**，只在调试日志里说生成了几个字段。
    """
    path = Path(source) if source else bili_cookie_source()
    try:
        if not path.is_file():
            logger.warning("B站 cookie 文件不存在，按未登录下载：%s", path)
            return None
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning("读取 B站 cookie 失败，按未登录下载：%s", exc)
        return None
    if not isinstance(data, dict):
        logger.warning("B站 cookie 文件不是 JSON 对象，按未登录下载")
        return None

    lowered = {str(key).strip().lower(): value for key, value in data.items()}
    values: list[tuple[str, str]] = []
    for key, cookie_name in _BILI_COOKIE_FIELDS.items():
        value = lowered.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        values.append((cookie_name, text))
    if not values:
        logger.warning("B站 cookie 文件里没有可用字段，按未登录下载")
        return None

    expiry = int(time.time()) + 30 * 24 * 3600
    lines = [
        "# Netscape HTTP Cookie File",
        "# 由 localai download_engine 临时生成，用完即删",
    ]
    for name, value in values:
        lines.append(
            f".bilibili.com\tTRUE\t/\tFALSE\t{expiry}\t{name}\t{value}"
        )
    fd, temp_name = tempfile.mkstemp(prefix="localai_bili_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        try:
            os.chmod(temp_name, 0o600)
        except OSError:
            pass
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    logger.info("已生成临时 B站 cookie 文件（字段数=%d）", len(values))
    return Path(temp_name)


def resolve_cookie_file(url: str, cookies: Optional[str | Path]) -> tuple[Optional[Path], Optional[Path]]:
    """决定本次下载用哪个 cookie 文件。

    返回 ``(cookie 文件, 需要事后删除的临时文件)``。
    - 显式传了 ``cookies``：只校验存在，不自动替换；
    - 没传且是 B站：自动生成临时文件；
    - 其它站点：不带 cookie。
    """
    if cookies:
        path = Path(cookies).expanduser()
        if not path.is_file():
            raise DownloadError(f"cookie 文件不存在：{path}")
        return path, None
    if bili_site(url):
        temp = build_bili_cookie_file()
        if temp is not None:
            return temp, temp
    return None, None


# ---------------------------------------------------------------------------
# yt-dlp 延迟导入 / 状态
# ---------------------------------------------------------------------------


def _ytdlp() -> Any:
    try:
        import yt_dlp  # noqa: PLC0415 - 延迟导入，启动时 0 依赖
    except ImportError as exc:  # pragma: no cover - 安装后不会走到
        raise DownloadError(
            "未安装 yt-dlp。请在 engine/.venv 执行："
            "python -m pip install -U yt-dlp"
        ) from exc
    return yt_dlp


def ytdlp_version() -> Optional[str]:
    """已安装的 yt-dlp 版本；未安装返回 None。"""
    try:
        return str(_ytdlp().version.__version__)
    except Exception:  # noqa: BLE001 - 状态查询不能抛
        return None


def ytdlp_path() -> Optional[str]:
    try:
        return str(Path(_ytdlp().__file__).resolve())
    except Exception:  # noqa: BLE001
        return None


def aria2c_path() -> Optional[str]:
    return shutil.which("aria2c")


def status_info() -> dict[str, Any]:
    """``/media/download/status`` 与 doctor 共用的状态快照。"""
    version = ytdlp_version()
    return {
        "yt_dlp": {
            "installed": version is not None,
            "version": version,
            "path": ytdlp_path(),
            "python": os.sys.executable,
        },
        "aria2c": {
            "available": aria2c_path() is not None,
            "path": aria2c_path(),
            "note": "可选：装了就用 --downloader aria2c，没装用 yt-dlp 内置下载器",
        },
        "download_dir": str(downloads_root()),
        "today_dir": str(today_dir()),
        "limits": {
            "max_mb": max_download_mb(),
            "timeout_seconds": int(default_timeout()),
            "qualities": list(QUALITY_CHOICES),
        },
        "ffmpeg": ffmpeg_location(),
        "cookies": {
            "auto_bilibili": True,
            "source": str(bili_cookie_source()),
            "source_exists": bili_cookie_source().is_file(),
            "explicit_path_supported": True,
        },
        "ssrf": {
            "schemes": ["http", "https"],
            "blocked": [
                "127.0.0.0/8",
                "10.0.0.0/8",
                "172.16.0.0/12",
                "192.168.0.0/16",
                "169.254.0.0/16",
                "::1",
                "fc00::/7",
                "fe80::/10",
            ],
            "check_redirects": True,
            "allow_private_env": "LOCALAI_DOWNLOAD_ALLOW_PRIVATE（仅测试）",
        },
    }


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def sanitize_filename(name: str, *, fallback: str = "download") -> str:
    """把标题 / URL 文件名净化成 Windows 可用的文件名。"""
    text = unquote(str(name or "")).strip()
    text = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text or text in (".", ".."):
        text = fallback
    return text[:150]


def _format_suffix(quality: str) -> str:
    return ".m4a" if quality == "audio" else ".mp4"


def _pick_title(info: dict[str, Any]) -> str:
    title = info.get("title") or info.get("fulltitle") or info.get("id") or "未命名"
    return str(title)


def _find_entry(info: dict[str, Any]) -> dict[str, Any]:
    """播放列表只取第一条（本服务不做批量抓取）。"""
    if info.get("_type") in ("playlist", "multi_video") and info.get("entries"):
        entries = [item for item in info["entries"] if item]
        if entries:
            return entries[0]
    return info


def _format_row(item: dict[str, Any]) -> dict[str, Any]:
    """把 yt-dlp 的格式结构压成对外稳定字段。"""
    return {
        "format_id": item.get("format_id"),
        "ext": item.get("ext"),
        "height": item.get("height"),
        "width": item.get("width"),
        "fps": item.get("fps"),
        "vcodec": item.get("vcodec"),
        "acodec": item.get("acodec"),
        "filesize": item.get("filesize") or item.get("filesize_approx"),
        "tbr": item.get("tbr"),
        "note": item.get("format_note"),
    }


def _probe_duration(path: Path) -> Optional[float]:
    """用 ffprobe 读时长（直链下载后补 duration；失败返回 None）。"""
    try:
        from ffmpeg_tool import probe  # engine/ 在 sys.path 上时可用

        return probe(path).get("duration")
    except Exception:  # noqa: BLE001 - 时长只是附加信息
        return None


def _safe_out_dir(out_dir: str | Path) -> Path:
    target = Path(out_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _emit(progress_cb: Optional[Callable[[dict[str, Any]], None]], payload: dict[str, Any]) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(payload)
    except Exception:  # noqa: BLE001 - 进度回调失败不能影响下载
        logger.debug("进度回调失败", exc_info=True)


def _progress_payload(
    *,
    status: str,
    downloaded: int,
    total: Optional[int],
    started: float,
    speed: Optional[float] = None,
    eta: Optional[float] = None,
) -> dict[str, Any]:
    now = time.monotonic()
    elapsed = max(0.001, now - started)
    measured = downloaded / elapsed
    percent = None
    if total and total > 0:
        percent = round(min(100.0, downloaded * 100.0 / total), 1)
    return {
        "status": status,
        "percent": percent,
        "downloaded_bytes": int(downloaded),
        "total_bytes": int(total) if total else None,
        "speed": round(float(speed or measured), 1) if downloaded else 0,
        "eta": int(eta) if eta is not None else None,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# yt-dlp 选项
# ---------------------------------------------------------------------------


def _format_selector(quality: str) -> str:
    if quality == "audio":
        return "bestaudio/best"
    if quality == "best":
        return "bv*+ba/b"
    # 默认 720p：视频 + 音频，最后用 b 兜底（单文件站点）。
    return "bv*[height<=720]+ba/b[height<=720]/b"


def _base_opts(out_dir: Path, *, cookie_file: Optional[Path]) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "outtmpl": str(out_dir / "%(title).150B [%(id)s].%(ext)s"),
        "paths": {"home": str(out_dir)},
        "windowsfilenames": True,
        "trim_file_name": 150,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "retries": 2,
        "fragment_retries": 2,
        "extractor_retries": 1,
        "socket_timeout": 30,
        "http_headers": {"User-Agent": USER_AGENT},
        "ffmpeg_location": ffmpeg_location(),
        "merge_output_format": "mp4",
        "overwrites": False,
    }
    if cookie_file is not None:
        opts["cookiefile"] = str(cookie_file)
    aria = aria2c_path()
    if aria:
        # 装了 aria2 就用它；没装不影响，yt-dlp 自带下载器。
        opts["downloader"] = "aria2c"
        opts["downloader_args"] = {"aria2c": ["-x", "8", "-s", "8", "-k", "1M"]}
    return opts


def _apply_audio_opts(opts: dict[str, Any]) -> None:
    opts["postprocessors"] = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": AUDIO_FORMAT,
            "preferredquality": "192",
        }
    ]


def _apply_subtitle_opts(opts: dict[str, Any]) -> None:
    opts.update(
        {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": list(SUBTITLE_LANGS),
            # 优先 srt，其次 vtt；都没有就保留站点给的格式。
            "subtitlesformat": "srt/vtt/best",
            "postprocessors": list(opts.get("postprocessors") or [])
            + [
                {
                    "key": "FFmpegSubtitlesConvertor",
                    "format": "srt",
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def probe_url(
    url: str,
    *,
    cookies: Optional[str | Path] = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """用 yt-dlp ``extract_info(download=False)`` 读取链接信息。

    返回 ``{ok, url, title, duration, site, webpage_url, is_live, subtitles,
    formats[], thumbnail}``。失败抛 :class:`DownloadError`，message 里带
    yt-dlp 原始错误摘要。
    """
    validate_url(url)
    cookie_file, temp = resolve_cookie_file(url, cookies)
    ytdlp = _ytdlp()
    opts = _base_opts(Path(tempfile.gettempdir()), cookie_file=cookie_file)
    opts.update({"skip_download": True, "extract_flat": False})
    started = time.perf_counter()
    try:
        with ytdlp.YoutubeDL(opts) as ydl:
            raw = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - 统一翻译成 DownloadError
        raise DownloadError(f"yt-dlp 解析失败：{_summarize_error(exc)}") from exc
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    info = _find_entry(raw or {})
    # 探测到的最终页面地址也要复查一次（重定向后的目标）。
    webpage = str(info.get("webpage_url") or url)
    if webpage.startswith(("http://", "https://")):
        try:
            validate_url(webpage)
        except DownloadBlocked:
            raise
        except DownloadError:
            pass

    formats = [
        _format_row(item)
        for item in (info.get("formats") or [])
        if item.get("url") or item.get("format_id")
    ]
    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    subtitle_langs = sorted(set(subtitles) | set(automatic))
    result = {
        "ok": True,
        "url": url,
        "webpage_url": webpage,
        "title": _pick_title(info),
        "duration": info.get("duration"),
        "site": info.get("extractor_key") or info.get("extractor"),
        "uploader": info.get("uploader"),
        "is_live": bool(info.get("is_live")),
        "thumbnail": info.get("thumbnail"),
        "subtitles": {
            "available": bool(subtitles or automatic),
            "manual": sorted(subtitles),
            "automatic": sorted(automatic),
            "languages": subtitle_langs[:50],
        },
        "formats": formats,
        "format_count": len(formats),
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }
    logger.info(
        "链接探测：site=%s 时长=%ss 格式=%d 字幕=%s 用时=%dms",
        result["site"],
        result["duration"],
        result["format_count"],
        "有" if result["subtitles"]["available"] else "无",
        result["elapsed_ms"],
    )
    return result


def _summarize_error(exc: BaseException) -> str:
    """去掉 yt-dlp 的冗长前缀，保留可读的原始错误摘要。"""
    text = str(exc).strip()
    text = re.sub(r"^ERROR:\s*", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [f"{type(exc).__name__}"]
    summary = " | ".join(lines[:4])
    return summary[:1200]


# ---------------------------------------------------------------------------
# download（yt-dlp）
# ---------------------------------------------------------------------------


@dataclass
class _AbortState:
    """进度钩子里的中止原因（yt-dlp 可能把它包一层，外层靠这里取原文）。"""

    error: Optional[BaseException] = None
    last_emit: float = field(default=0.0)


def download(
    url: str,
    out_dir: str | Path,
    *,
    quality: str = "720p",
    subtitles: bool = True,
    cookies: Optional[str | Path] = None,
    progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
    timeout: Optional[float] = None,
    max_mb: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """用 yt-dlp 下载一个链接，返回落盘结果。

    参数
    ----
    quality:    ``720p``（默认）/ ``best`` / ``audio``。
    subtitles:  是否尝试站点字幕（下不到不报错）。
    cookies:    Netscape cookie 文件路径；B站不传时自动从本机凭证生成。
    progress_cb: 进度回调，入参 ``{percent, downloaded_bytes, total_bytes, speed, eta}``。
    timeout:    整体超时秒数，默认 ``LOCALAI_DOWNLOAD_TIMEOUT``。
    max_mb:     单文件上限 MB，默认 ``LOCALAI_DOWNLOAD_MAX_MB``。
    should_cancel: 返回 True 时在下一个进度检查点中止并清理半成品。

    返回::

        {"ok", "path", "title", "duration", "site", "subtitles": [...],
         "size_mb", "quality", "elapsed_seconds", "yt_dlp_version"}
    """
    url = validate_url(url)
    quality = (quality or "720p").strip().lower()
    if quality not in QUALITY_CHOICES:
        raise DownloadError(f"不支持的 quality={quality!r}；可选：{', '.join(QUALITY_CHOICES)}")
    out_dir_path = _safe_out_dir(out_dir)
    limit_mb = int(max_mb if max_mb is not None else max_download_mb())
    limit_bytes = limit_mb * 1024 * 1024
    wait = float(timeout if timeout is not None else default_timeout())
    deadline = time.monotonic() + wait

    ytdlp = _ytdlp()
    cookie_file, temp_cookie = resolve_cookie_file(url, cookies)
    opts = _base_opts(out_dir_path, cookie_file=cookie_file)
    opts.update(
        {
            "format": _format_selector(quality),
            "noplaylist": True,
            "max_filesize": limit_bytes,
        }
    )
    if quality == "audio":
        _apply_audio_opts(opts)
    if subtitles:
        _apply_subtitle_opts(opts)

    state = _AbortState()
    started = time.monotonic()

    def hook(data: dict[str, Any]) -> None:
        if should_cancel is not None and should_cancel():
            state.error = DownloadCancelled("下载已取消（任务取消请求）")
            raise state.error
        if time.monotonic() > deadline:
            state.error = DownloadTimeout(f"下载超时（>{wait:.0f} 秒）")
            raise state.error
        if data.get("status") == "downloading":
            downloaded = int(data.get("downloaded_bytes") or 0)
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            if total:
                estimated_total = max(int(total), downloaded)
            else:
                estimated_total = None
            if downloaded > limit_bytes:
                state.error = DownloadError(
                    f"文件超过大小上限 {limit_mb}MB，已中止下载"
                )
                raise state.error
            now = time.monotonic()
            if now - state.last_emit >= PROGRESS_INTERVAL:
                state.last_emit = now
                _emit(
                    progress_cb,
                    _progress_payload(
                        status="downloading",
                        downloaded=downloaded,
                        total=estimated_total,
                        started=started,
                        speed=data.get("speed"),
                        eta=data.get("eta"),
                    ),
                )
        elif data.get("status") == "finished":
            downloaded = int(data.get("downloaded_bytes") or 0)
            _emit(
                progress_cb,
                _progress_payload(
                    status="processing",
                    downloaded=downloaded,
                    total=downloaded,
                    started=started,
                    speed=None,
                    eta=0,
                ),
            )

    opts["progress_hooks"] = [hook]
    _emit(
        progress_cb,
        {
            "status": "starting",
            "percent": 0,
            "downloaded_bytes": 0,
            "total_bytes": None,
            "speed": 0,
            "eta": None,
        },
    )

    info: dict[str, Any] = {}
    try:
        with ytdlp.YoutubeDL(opts) as ydl:
            raw = ydl.extract_info(url, download=True)
            info = _find_entry(raw or {})
    except BaseException as exc:  # noqa: BLE001 - 统一转成 DownloadError
        if state.error is not None:
            _cleanup_partials(out_dir_path, info, url)
            raise state.error from exc
        if should_cancel is not None and should_cancel():
            _cleanup_partials(out_dir_path, info, url)
            raise DownloadCancelled("下载已取消（任务取消请求）") from exc
        _cleanup_partials(out_dir_path, info, url)
        raise DownloadError(f"yt-dlp 下载失败：{_summarize_error(exc)}") from exc
    finally:
        if temp_cookie is not None:
            temp_cookie.unlink(missing_ok=True)

    final_path = _resolve_download_path(ydl, info, out_dir_path)
    if final_path is None or not final_path.is_file():
        raise DownloadError("yt-dlp 报告下载完成，但没有找到输出文件")
    size_bytes = final_path.stat().st_size
    if size_bytes > limit_bytes:
        final_path.unlink(missing_ok=True)
        raise DownloadError(f"文件超过大小上限 {limit_mb}MB，已删除")

    subtitle_files = _collect_subtitles(final_path)
    duration = info.get("duration")
    if duration is None:
        duration = _probe_duration(final_path)
    result = {
        "ok": True,
        "path": str(final_path),
        "title": _pick_title(info),
        "duration": None if duration is None else round(float(duration), 3),
        "site": info.get("extractor_key") or info.get("extractor"),
        "webpage_url": info.get("webpage_url") or url,
        "subtitles": [str(path) for path in subtitle_files],
        "subtitle_langs": [
            path.stem.split(".")[-1] for path in subtitle_files
        ],
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "quality": quality,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "yt_dlp_version": ytdlp_version(),
        "downloader": "aria2c" if aria2c_path() else "builtin",
    }
    _emit(
        progress_cb,
        {
            "status": "done",
            "percent": 100,
            "downloaded_bytes": size_bytes,
            "total_bytes": size_bytes,
            "speed": 0,
            "eta": 0,
            "elapsed_seconds": result["elapsed_seconds"],
        },
    )
    logger.info(
        "下载完成：site=%s 画质=%s 大小=%.2fMB 字幕=%d 耗时=%.1fs 路径=%s",
        result["site"],
        quality,
        result["size_mb"],
        len(subtitle_files),
        result["elapsed_seconds"],
        final_path,
    )
    return result


def _resolve_download_path(
    ydl: Any, info: dict[str, Any], out_dir: Path
) -> Optional[Path]:
    """从 yt-dlp 的结果里定位最终文件（合并 / 转码后扩展名可能变化）。"""
    requested = info.get("requested_downloads") or []
    for item in requested:
        raw = item.get("filepath") or item.get("_filename")
        if raw:
            path = Path(raw)
            if path.is_file():
                return path
            # 合并后文件名可能被替换成 mp4 / m4a。
            for suffix in (".mp4", ".mkv", ".webm", AUDIO_FORMAT):
                candidate = path.with_suffix(suffix)
                if candidate.is_file():
                    return candidate
    try:
        prepared = Path(ydl.prepare_filename(info))
    except Exception:  # noqa: BLE001
        prepared = None
    if prepared is not None:
        if prepared.is_file():
            return prepared
        for suffix in (".mp4", ".mkv", ".webm", AUDIO_FORMAT):
            candidate = prepared.with_suffix(suffix)
            if candidate.is_file():
                return candidate
    # 最后兜底：目录里最新的媒体文件。
    media_suffixes = {".mp4", ".mkv", ".webm", ".m4a", ".mp3", ".flv"}
    candidates = [
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix.lower() in media_suffixes
    ]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _collect_subtitles(media_path: Path) -> list[Path]:
    """收集与媒体同名的字幕文件（转 srt 后扩展名可能是 srt / vtt / ass）。"""
    stem = media_path.stem
    found: list[Path] = []
    try:
        for path in media_path.parent.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".srt", ".vtt", ".ass", ".lrc"):
                continue
            if path.name.startswith(stem):
                found.append(path)
    except OSError:
        return []
    return sorted(found)


def _cleanup_partials(
    out_dir: Path, info: dict[str, Any], url: str
) -> None:
    """失败 / 取消后删除本次产生的半成品（.part / .ytdl）。"""
    for pattern in ("*.part", "*.ytdl", "*.temp"):
        for path in out_dir.glob(pattern):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("删除半成品失败：%s", path)


# ---------------------------------------------------------------------------
# 直链兜底
# ---------------------------------------------------------------------------


def _content_disposition_name(value: str) -> Optional[str]:
    """从 Content-Disposition 里抠文件名（支持 RFC 5987）。"""
    if not value:
        return None
    match = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.I)
    if match:
        return unquote(match.group(1).strip().strip('"'))
    match = re.search(r'filename="?([^";]+)"?', value, flags=re.I)
    if match:
        return match.group(1).strip()
    return None


def _guess_suffix(name: str, content_type: str, url: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix and len(suffix) <= 8:
        return suffix
    content_map = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    for key, value in content_map.items():
        if key in (content_type or "").lower():
            return value
    parsed_suffix = Path(urlparse(url).path).suffix.lower()
    return parsed_suffix if parsed_suffix else ".bin"


def download_direct(
    url: str,
    out_dir: str | Path,
    *,
    filename: Optional[str] = None,
    progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
    timeout: Optional[float] = None,
    max_mb: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    resume: bool = True,
) -> dict[str, Any]:
    """普通 http(s) 直链下载（httpx 流式，支持 Range 断点续传）。

    返回 ``{ok, path, title, duration, site, size_mb, resumed, elapsed_seconds}``。
    超限会中止并删除半成品；重定向逐跳复查 SSRF。
    """
    url = validate_url(url)
    out_dir_path = _safe_out_dir(out_dir)
    limit_mb = int(max_mb if max_mb is not None else max_download_mb())
    limit_bytes = limit_mb * 1024 * 1024
    wait = float(timeout if timeout is not None else default_timeout())
    deadline = time.monotonic() + wait
    started = time.monotonic()

    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    current = url
    response: Optional[httpx.Response] = None
    client = httpx.Client(
        timeout=httpx.Timeout(wait, connect=15.0, read=120.0, write=30.0),
        follow_redirects=False,
    )
    try:
        # 手动跟重定向：每一跳都重新做 SSRF 校验，不信任服务器给的 Location。
        for _hop in range(MAX_REDIRECTS + 1):
            if should_cancel is not None and should_cancel():
                raise DownloadCancelled("下载已取消")
            request = client.build_request("GET", current, headers=headers)
            response = client.send(request, stream=True)
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                response.close()
                if not location:
                    raise DownloadError(f"服务器返回 {response.status_code} 但没有 Location")
                current = _check_redirect_target(current, location)
                continue
            break
        else:
            raise DownloadError(f"重定向超过 {MAX_REDIRECTS} 跳，已中止")

        if response is None or response.status_code >= 400:
            code = response.status_code if response is not None else "?"
            detail = response.text[:300] if response is not None else ""
            raise DownloadError(f"直链返回 HTTP {code}：{detail}".strip())

        final_url = str(response.url)
        content_type = response.headers.get("content-type", "")
        raw_name = (
            filename
            or _content_disposition_name(response.headers.get("content-disposition", ""))
            or Path(unquote(urlparse(final_url).path)).name
            or "download"
        )
        suffix = _guess_suffix(raw_name, content_type, final_url)
        stem = sanitize_filename(Path(raw_name).stem or "download")
        base_name = f"{stem}{suffix}"
        target = out_dir_path / base_name
        if target.exists():
            target = out_dir_path / f"{stem}_{int(time.time())}{suffix}"
        part = target.with_name(target.name + ".part")

        existing = part.stat().st_size if (resume and part.is_file()) else 0
        initial_existing = existing
        if existing:
            # 已经有一个 .part：第一次请求先不带 Range（为了拿到文件名/长度），
            # 确认目标后带 Range 重发一次，真正走断点续传。
            headers["Range"] = f"bytes={existing}-"
            response.close()
            request = client.build_request("GET", final_url, headers=headers)
            response = client.send(request, stream=True)
            if response.status_code == 206:
                pass
            elif response.status_code == 200:
                # 服务器无视 Range 强制整包：丢掉旧 part，从头写。
                existing = 0
                part.unlink(missing_ok=True)
            elif response.status_code == 416:
                part.unlink(missing_ok=True)
                raise DownloadError("断点续传范围无效，已删除半成品，请重试")
            else:
                raise DownloadError(f"断点续传失败：HTTP {response.status_code}")

        content_length = response.headers.get("content-length")
        declared = int(content_length) if content_length and content_length.isdigit() else None
        total = (existing + declared) if declared is not None else None
        if total is not None and total > limit_bytes:
            response.close()
            raise DownloadError(
                f"文件大小 {total / 1024 / 1024:.1f}MB 超过上限 {limit_mb}MB"
            )

        written = existing
        mode = "ab" if existing else "wb"
        last_emit = 0.0
        with part.open(mode) as fh:
            for chunk in response.iter_bytes(CHUNK_SIZE):
                if not chunk:
                    continue
                if should_cancel is not None and should_cancel():
                    raise DownloadCancelled("下载已取消")
                if time.monotonic() > deadline:
                    raise DownloadTimeout(f"下载超时（>{wait:.0f} 秒）")
                fh.write(chunk)
                written += len(chunk)
                if written > limit_bytes:
                    raise DownloadError(f"文件超过大小上限 {limit_mb}MB，已中止")
                now = time.monotonic()
                if now - last_emit >= PROGRESS_INTERVAL:
                    last_emit = now
                    _emit(
                        progress_cb,
                        _progress_payload(
                            status="downloading",
                            downloaded=written,
                            total=total,
                            started=started,
                        ),
                    )
        response.close()

        if declared is not None:
            actual = part.stat().st_size
            expected = (existing + declared) if response.status_code == 206 else declared
            if actual != expected:
                raise DownloadError(
                    f"大小校验失败：Content-Length={declared}，实际写入 {actual}"
                )
        part.replace(target)
    except BaseException:
        # 失败 / 取消时删除半成品，避免被误认为完整文件；下次重下会重新开始。
        try:
            if "part" in locals() and part.exists():
                part.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        try:
            if response is not None:
                response.close()
        except Exception:  # noqa: BLE001
            pass
        client.close()

    size_bytes = target.stat().st_size
    duration = _probe_duration(target)
    result = {
        "ok": True,
        "path": str(target),
        "title": target.stem,
        "duration": None if duration is None else round(float(duration), 3),
        "site": urlparse(final_url).hostname,
        "webpage_url": final_url,
        "subtitles": [],
        "subtitle_langs": [],
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "quality": "direct",
        "content_type": content_type,
        "resumed": bool(initial_existing),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    _emit(
        progress_cb,
        {
            "status": "done",
            "percent": 100,
            "downloaded_bytes": size_bytes,
            "total_bytes": size_bytes,
            "speed": 0,
            "eta": 0,
        },
    )
    logger.info(
        "直链下载完成：host=%s 大小=%.2fMB 耗时=%.1fs 路径=%s",
        result["site"],
        result["size_mb"],
        result["elapsed_seconds"],
        target,
    )
    return result


# ---------------------------------------------------------------------------
# 统一入口：yt-dlp 优先，明确的「不支持的 URL」再走直链
# ---------------------------------------------------------------------------


def download_any(
    url: str,
    out_dir: str | Path,
    *,
    quality: str = "720p",
    subtitles: bool = True,
    cookies: Optional[str | Path] = None,
    progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
    timeout: Optional[float] = None,
    max_mb: Optional[int] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict[str, Any]:
    """先交给 yt-dlp；只有它明确不支持这个 URL 时才退回直链下载。

    站点风控 / 需要登录 / 格式缺失等错误**不会**盲目改走直链，避免把登录页
    当成视频存下来；直链兜底只针对 ``Unsupported URL`` 这类明确判断。
    """
    try:
        return download(
            url,
            out_dir,
            quality=quality,
            subtitles=subtitles,
            cookies=cookies,
            progress_cb=progress_cb,
            timeout=timeout,
            max_mb=max_mb,
            should_cancel=should_cancel,
        )
    except (DownloadBlocked, DownloadCancelled, DownloadTimeout):
        raise
    except DownloadError as exc:
        text = str(exc).lower()
        fallback_markers = (
            "unsupported url",
            "no video formats",
            "no suitable",
            "is not a valid url",
        )
        if not any(marker in text for marker in fallback_markers):
            raise
        logger.info("yt-dlp 不支持该链接，尝试直链兜底：%s", str(exc)[:200])
        try:
            return download_direct(
                url,
                out_dir,
                progress_cb=progress_cb,
                timeout=timeout,
                max_mb=max_mb,
                should_cancel=should_cancel,
            )
        except DownloadError as direct_exc:
            raise DownloadError(
                f"yt-dlp：{_summarize_error(exc)}；直链兜底也失败：{direct_exc}"
            ) from direct_exc


# ---------------------------------------------------------------------------
# 命令行自检
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="通用下载引擎自检")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="查看 yt-dlp / aria2 / 下载目录状态")
    probe_parser = sub.add_parser("probe", help="探测链接")
    probe_parser.add_argument("url")
    down_parser = sub.add_parser("download", help="下载链接")
    down_parser.add_argument("url")
    down_parser.add_argument("--quality", default="720p", choices=QUALITY_CHOICES)
    down_parser.add_argument("--out", default=None)
    down_parser.add_argument("--no-subs", action="store_true")
    direct_parser = sub.add_parser("direct", help="直链下载")
    direct_parser.add_argument("url")
    direct_parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(status_info(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "probe":
        print(json.dumps(probe_url(args.url), ensure_ascii=False, indent=2))
        return 0
    if args.command == "direct":
        result = download_direct(args.url, args.out or today_dir())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = download(
        args.url,
        args.out or today_dir(),
        quality=args.quality,
        subtitles=not args.no_subs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
