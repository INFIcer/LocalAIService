"""
最小链路自检脚本
================

不加载 IndexTTS 模型，只跑 `LoadAudio -> SaveAudio`，验证：

    上传参考音频 -> POST /prompt -> 轮询 /history -> GET /view -> 转 WAV 落盘

用法（在 E:\\LocalAIService\\svc 下）：
    .\\.venv\\Scripts\\python.exe selftest.py

成功时会打印 prompt_id、总耗时、ComfyUI 原始文件和本地 wav 路径。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import service


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    voices = service.scan_voices()
    if not voices:
        print(f"[失败] 音色目录为空：{service.VOICES_DIR}")
        return 1

    voice = next((v for v in voices if v.name == "default"), voices[0])

    async with service._http_client() as client:
        if not await service.comfy_up(client):
            print(f"[失败] 连不上 ComfyUI：{service.COMFY_BASE}")
            return 1

        print(f"[1/4] 上传参考音频：{voice.path}")
        started = time.perf_counter()
        audio_name = await service.ensure_uploaded(client, voice)

        print(f"[2/4] 提交最小工作流（LoadAudio -> SaveAudio），input={audio_name}")
        workflow = service.build_passthrough_workflow(audio_name, "svc_selftest")
        entry, raw = await service.run_workflow(workflow, client=client)
        elapsed = time.perf_counter() - started

        print("[3/4] 下载完成，转换为 WAV")
        wav = service.to_wav(raw)
        output_path = Path(service.OUTPUT_DIR) / "selftest_minimal.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav)

        item = service.find_audio_output(entry)
        prompt_id = ""
        try:
            prompt_id = entry["prompt"][1]
        except Exception:  # noqa: BLE001
            pass

    print("[4/4] 结果")
    print(
        json.dumps(
            {
                "ok": True,
                "prompt_id": prompt_id,
                "elapsed_seconds": round(elapsed, 3),
                "comfy_output": item,
                "comfy_bytes": len(raw),
                "wav_bytes": len(wav),
                "wav_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
