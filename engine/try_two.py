"""连续两次合成的最小复现脚本（排错用）。"""
import gc
import sys
import time
from pathlib import Path

# 按脚本位置推目录，改名 / 走 junction 都不用改这里
ENGINE_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = ENGINE_DIR.parent
sys.path.insert(0, str(ENGINE_DIR))
import torch

import engine

TEXT = "主人，我是大肥鱼。"
VOICE = str(SERVICE_ROOT / "voices" / "default.mp3")

for index in (1, 2):
    print(f"\n===== 第 {index} 次 =====", flush=True)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("前：allocated=%.2fGB reserved=%.2fGB free=%.2fGB" % (
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
            free / 1024**3,
        ), flush=True)
    started = time.perf_counter()
    meta = engine.synthesize(
        TEXT,
        VOICE,
        str(SERVICE_ROOT / "outputs" / f"engine_twice_{index}.wav"),
        "zh",
        1.0,
        seed=1234,
    )
    print("结果：", meta, flush=True)
    print("墙钟：%.2fs" % (time.perf_counter() - started), flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print("后：allocated=%.2fGB reserved=%.2fGB free=%.2fGB" % (
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
            free / 1024**3,
        ), flush=True)
print("DONE", flush=True)
