"""下载能力测试（全部离线）：SSRF / scheme / 限额 / 直链本地服务器 / cookie / 分析编排。

测试策略
--------
- 直链下载用**本机临时 HTTP 服务器**验证真实落盘与时长；因为 SSRF 默认拦截
  私有地址，这类用例临时打开 ``LOCALAI_DOWNLOAD_ALLOW_PRIVATE=1``。
- 私有地址 / file:// / 重定向的拦截用默认配置单独验证。
- 分析编排用假适配器 mock，不加载任何模型、不联网。
"""

from __future__ import annotations

import asyncio
import http.server
import json
import socket
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterator

import pytest

from localai.config import Config
from localai.engines._native import import_engine_module
from localai.jobs.executors import JobContext, execute_job, prepare_payload

ENGINE = import_engine_module("download_engine")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:  # noqa: D401 - 静音
        pass

    def handle_one_request(self) -> None:
        # 客户端因「超过大小上限」主动断开时，服务器端会拿到 BrokenPipe；
        # 测试里静音即可，不影响断言。
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class _QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """固定内容 + 支持 Range 的最小服务器（验证断点续传）。"""

    payload = b"0123456789" * 1000

    def log_message(self, *_args: Any) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 命名
        range_header = self.headers.get("Range") or ""
        start = 0
        if range_header.startswith("bytes="):
            try:
                start = int(range_header[6:].split("-", 1)[0])
            except ValueError:
                start = 0
        body = self.payload[start:]
        if start:
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def http_dir(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """起一个只读的临时 HTTP 服务器，返回 (目录, base_url)。"""
    root = tmp_path / "www"
    root.mkdir()
    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args, directory=str(root), **kwargs
    )
    server = _QuietServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _make_mp4(path: Path) -> bool:
    """用 ffmpeg 合成 5 秒 testsrc（带正弦音轨）；ffmpeg 不可用返回 False。"""
    cfg = Config.from_env()
    ffmpeg = cfg.ffmpeg
    if not ffmpeg:
        return False
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=25:duration=5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=5",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    try:
        subprocess.run(command, check=True, timeout=60, capture_output=True)
    except Exception:  # noqa: BLE001
        return False
    return path.is_file()


def _allow_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAI_DOWNLOAD_ALLOW_PRIVATE", "1")


# ---------------------------------------------------------------------------
# SSRF / scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/win.ini",
        "ftp://example.com/a.mp4",
        "gopher://example.com/",
        "data:text/plain,hello",
        "javascript:alert(1)",
    ],
)
def test_reject_non_http_scheme(url: str) -> None:
    with pytest.raises(ENGINE.DownloadBlocked):
        ENGINE.validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/a.mp4",
        "http://10.0.0.5/a.mp4",
        "http://172.16.3.4/a.mp4",
        "http://192.168.1.1/a.mp4",
        "http://169.254.1.1/a.mp4",
        "http://[::1]/a.mp4",
        "http://localhost/a.mp4",
    ],
)
def test_reject_private_addresses(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALAI_DOWNLOAD_ALLOW_PRIVATE", raising=False)
    with pytest.raises(ENGINE.DownloadBlocked):
        ENGINE.validate_url(url)


def test_accept_public_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALAI_DOWNLOAD_ALLOW_PRIVATE", raising=False)
    assert ENGINE.validate_url("https://8.8.8.8/a.mp4") == "https://8.8.8.8/a.mp4"


def test_redirect_to_private_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCALAI_DOWNLOAD_ALLOW_PRIVATE", raising=False)
    with pytest.raises(ENGINE.DownloadBlocked):
        ENGINE._check_redirect_target("https://example.com/a", "http://127.0.0.1/b.mp4")
    with pytest.raises(ENGINE.DownloadBlocked):
        ENGINE._check_redirect_target("https://example.com/a", "file:///C:/x.mp4")


# ---------------------------------------------------------------------------
# 直链下载
# ---------------------------------------------------------------------------


def test_direct_download_and_probe_duration(
    http_dir: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base = http_dir
    sample = root / "sample.mp4"
    if not _make_mp4(sample):
        pytest.skip("本机 ffmpeg 不可用，跳过直链下载用例")
    _allow_private(monkeypatch)
    out = tmp_path / "out"
    result = ENGINE.download_direct(
        f"{base}/sample.mp4", out, max_mb=50, timeout=120
    )
    path = Path(result["path"])
    assert path.is_file()
    assert path.name == "sample.mp4"
    assert result["size_mb"] > 0
    assert abs((result["duration"] or 0) - 5.0) < 0.5
    assert result["site"] == "127.0.0.1"


def test_direct_download_size_limit(
    http_dir: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base = http_dir
    big = root / "big.bin"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    _allow_private(monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(ENGINE.DownloadError):
        ENGINE.download_direct(f"{base}/big.bin", out, max_mb=1, timeout=60)
    # 半成品必须删掉，不能留下一个被误认为完整的文件。
    assert not (out / "big.bin").exists()
    assert not list(out.glob("*.part"))


def test_direct_download_progress(
    http_dir: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base = http_dir
    (root / "small.bin").write_bytes(b"x" * 4096)
    _allow_private(monkeypatch)
    events: list[dict[str, Any]] = []
    ENGINE.download_direct(
        f"{base}/small.bin",
        tmp_path / "out",
        max_mb=10,
        timeout=60,
        progress_cb=events.append,
    )
    assert events
    assert events[-1]["status"] == "done"
    assert events[-1]["percent"] == 100


def test_direct_download_range_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已有 .part 时带 Range 续传，最终内容与完整文件一致。"""
    _allow_private(monkeypatch)
    server = _QuietServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        out = tmp_path / "out"
        out.mkdir()
        part = out / "media.bin.part"
        part.write_bytes(_RangeHandler.payload[:5000])
        result = ENGINE.download_direct(
            f"http://127.0.0.1:{server.server_port}/media.bin",
            out,
            max_mb=10,
            timeout=60,
        )
        target = Path(result["path"])
        assert target.read_bytes() == _RangeHandler.payload
        assert result["resumed"] is True
        assert not part.exists()
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Cookie
# ---------------------------------------------------------------------------


def test_bili_cookie_file_generated_and_cleaned(tmp_path: Path) -> None:
    credential = tmp_path / "bili_credential.json"
    credential.write_text(
        json.dumps(
            {
                "SESSDATA": "test-sess",
                "bili_jct": "test-jct",
                "buvid3": "test-buvid",
                "DedeUserID": "12345",
            }
        ),
        encoding="utf-8",
    )
    cookie = ENGINE.build_bili_cookie_file(credential)
    assert cookie is not None and cookie.is_file()
    try:
        text = cookie.read_text(encoding="utf-8")
        # 只检查格式与字段名存在，不在测试输出里打印内容。
        assert text.startswith("# Netscape HTTP Cookie File")
        for name in ("SESSDATA", "bili_jct", "buvid3", "DedeUserID"):
            assert name in text
        assert ".bilibili.com" in text
    finally:
        cookie.unlink(missing_ok=True)
    assert not cookie.exists()


def test_bili_cookie_missing_file_returns_none(tmp_path: Path) -> None:
    assert ENGINE.build_bili_cookie_file(tmp_path / "nope.json") is None


def test_bili_site_detection() -> None:
    assert ENGINE.bili_site("https://www.bilibili.com/video/BV1xx")
    assert ENGINE.bili_site("https://b23.tv/abcd")
    assert not ENGINE.bili_site("https://example.com/video")


# ---------------------------------------------------------------------------
# 分析编排（mock，不联网 / 不加载模型）
# ---------------------------------------------------------------------------


class _FakeDownload:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def call(self, func: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        assert func == "download_any"
        if kwargs.get("progress_cb"):
            kwargs["progress_cb"]({"percent": 50, "status": "downloading"})
        return self.result


class _FakeMedia:
    async def call(self, func: str, *args: Any, **kwargs: Any) -> Any:
        if func == "detect_scenes":
            return {
                "scenes": [{"time": 1.0, "score": 0.9}, {"time": 3.0, "score": 0.8}],
                "duration": 5.0,
                "json_path": "scenes.json",
            }
        if func == "extract_frames":
            return [{"path": "frame_1.jpg", "time_sec": 1.05}]
        if func == "to_wav16k":
            return {"path": args[0]}
        raise AssertionError(func)


class _FakeAsr:
    def __init__(self) -> None:
        self.called = False

    async def infer_segments(self, _path: str, _language: str) -> dict[str, Any]:
        self.called = True
        return {
            "segments": [
                {"start": 0.0, "end": 1.2, "text": "第一句"},
                {"start": 1.2, "end": 2.6, "text": "第二句"},
            ],
            "text": "第一句第二句",
            "duration": 5.0,
            "timestamp_source": "vad",
            "engine": "sensevoice",
            "language": "zh",
        }


class _FakeRegistry:
    def __init__(self, download: Any, media: Any, asr: Any) -> None:
        self._items = {"download": download, "media": media, "asr": asr}

    def get(self, name: str) -> Any:
        return self._items[name]


def _analyze_payload(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    media = tmp_path / "downloaded.mp4"
    media.write_bytes(b"fake")
    return {
        "url": "https://example.com/video",
        "quality": "720p",
        "subtitles": True,
        "cookies": None,
        "engine": "sensevoice",
        "max_frames": 1,
        **kwargs,
    }


def _ctx(tmp_path: Path, registry: _FakeRegistry) -> JobContext:
    cfg = Config.from_env(home=tmp_path)
    progress: list[dict[str, Any]] = []
    return JobContext(
        cfg=cfg,
        registry=registry,
        progress=progress.append,
    )


@pytest.mark.parametrize("with_site_subs", [True, False])
def test_analyze_orchestration(tmp_path: Path, with_site_subs: bool) -> None:
    srt = tmp_path / "site.zh-CN.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,500\n站点第一句\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n站点第二句\n",
        encoding="utf-8",
    )
    download = {
        "ok": True,
        "path": str(tmp_path / "downloaded.mp4"),
        "title": "测试视频",
        "duration": 5.0,
        "site": "Example",
        "subtitles": [str(srt)] if with_site_subs else [],
        "size_mb": 1.2,
        "quality": "720p",
        "elapsed_seconds": 2.1,
    }
    asr = _FakeAsr()
    registry = _FakeRegistry(_FakeDownload(download), _FakeMedia(), asr)
    job = {
        "id": "j1",
        "kind": "analyze",
        "action": "analyze.url",
        "payload": _analyze_payload(tmp_path),
    }
    result = asyncio.run(execute_job(job, _ctx(tmp_path, registry)))
    assert result["ok"] is True
    assert result["title"] == "测试视频"
    assert result["duration"] == 5.0
    assert result["subtitles_from"] == ("site" if with_site_subs else "asr")
    assert len(result["segments"]) == 2
    assert result["segments"][0]["text"] == ("站点第一句" if with_site_subs else "第一句")
    assert len(result["scenes"]) == 2
    assert len(result["frames"]) == 1
    assert asr.called is (not with_site_subs)


def test_prepare_payload_rejects_bad_quality_and_scheme(tmp_path: Path) -> None:
    cfg = Config.from_env(home=tmp_path)
    with pytest.raises(ValueError):
        prepare_payload("download", None, {"url": "file:///C:/x.mp4"}, cfg)
    with pytest.raises(ValueError):
        prepare_payload(
            "download", None, {"url": "https://example.com/a", "quality": "4k"}, cfg
        )
    action, payload = prepare_payload(
        "analyze", None, {"url": "https://example.com/a", "max_frames": 3}, cfg
    )
    assert action == "analyze.url"
    assert payload["max_frames"] == 3
    assert payload["quality"] == "720p"
