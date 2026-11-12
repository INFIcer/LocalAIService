"""QQ 语音条转写小工具（给 dsh-qq-bridge 的「转写命令」用）
=============================================================

用法（桥配置里填这个模板，{file} 会被替换成声音文件路径）：

    "E:\\LocalAIService\\engine\\.venv\\Scripts\\python.exe" "E:\\LocalAIService\\tools\\asr_cli.py" "{file}"

行为：把音频 POST 到本机 LocalAIService 的 /asr（它会自动 ffmpeg 转码、懒加载 SenseVoice），
把识别出的文字打到 stdout（桥会把它当消息内容）。识别失败时退出码非 0，并往 stderr 写原因。

可选环境变量：
    LOCALAI_BASE          服务地址，默认 http://127.0.0.1:7880
    LOCALAI_ASR_SPEAKER   1=同时算声纹向量（默认 1），0=不算
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("LOCALAI_BASE", "http://127.0.0.1:7880").rstrip("/")
USE_SPEAKER = os.environ.get("LOCALAI_ASR_SPEAKER", "1") != "0"


def post_file(url: str, path: Path, fields: dict[str, str], timeout: float = 180) -> dict:
    """标准库实现 multipart/form-data POST。"""
    boundary = "----localai" + str(int(time.time() * 1000))
    body = bytearray()
    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
    body += f"Content-Type: {mime}\r\n\r\n".encode()
    body += path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: asr_cli.py <音频文件>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"文件不存在: {src}", file=sys.stderr)
        return 2
    url = f"{BASE}/asr" + ("" if USE_SPEAKER else "?speaker=0")
    try:
        data = post_file(url, src, {"language": "auto"})
    except urllib.error.URLError as exc:
        print(f"连不上 LocalAIService（{BASE}）：{exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(f"转写失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    text = (data.get("text") or "").strip()

    # 顺手留一份原始语音，方便之后登记声纹（桥转完就把临时文件删了）
    try:
        import shutil

        inbox = Path(os.environ.get("LOCALAI_VOICE_INBOX", r"E:\LocalAIService\outputs\voice_inbox"))
        inbox.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = inbox / f"{stamp}_{src.suffix.lstrip('.')}{src.suffix}"
        shutil.copy2(src, target)
        # 只留最近 50 条
        old = sorted(inbox.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[50:]
        for p in old:
            try:
                p.unlink()
            except OSError:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"[voice-inbox] 留档失败：{exc}", file=sys.stderr)

    if not text:
        print("（语音里没识别出内容）")
        return 0
    # 默认只输出文字本身（桥会把它当消息内容）；向量信息放 stderr，不污染正文。
    # 谁是谁由鱼讯的记忆层判断，这里只报「算出了多大的向量」。
    speaker = data.get("speaker") or {}
    if speaker.get("embedding"):
        print(
            f"[声纹向量] dim={speaker.get('dim')} model={speaker.get('model')}",
            file=sys.stderr,
        )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
