"""识别引擎实测脚本（准确率 / RTF / 显存 / 长音频时间戳）。

用法（在服务根目录）::

    engine\\.venv\\Scripts\\python.exe tools\\bench_asr.py
    engine\\.venv\\Scripts\\python.exe tools\\bench_asr.py --no-cpu

读取 ``outputs/eval/manifest.json``（由 tools 里的一次性脚本生成：8 条已知
文本的 TTS 音频 + 一条 30~60 秒长音频），依次测：

1. sensevoice / whisper 四档（tiny/base/small/medium）的 CER、单条延迟、RTF；
2. 每档加载时的整卡显存（nvidia-smi 每 100ms 采样）；
3. CPU（int8）跑 small 一档；
4. 长音频 whisper small 的 segments，并与「合成时的真实句子起点」对齐；
5. 结果写 ``logs/bench_asr.json``，表格打印到 stdout。

只用于本机实测，不属于服务运行路径。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engine"
sys.path.insert(0, str(ENGINE_DIR))

import asr_engine  # noqa: E402
import ffmpeg_tool  # noqa: E402
import whisper_engine  # noqa: E402

MANIFEST = ROOT / "outputs" / "eval" / "manifest.json"
OUT_JSON = ROOT / "logs" / "bench_asr.json"

CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """去掉标点 / 空白，只保留文字与数字（中英文都适用）。"""
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)


def edit_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def gpu_used_mb() -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
            encoding="utf-8",
            errors="ignore",
            creationflags=CREATIONFLAGS,
        )
        return int(next(line.strip() for line in output.splitlines() if line.strip()))
    except Exception:
        return -1


class GpuSampler:
    """加载期间每 100ms 采一次整卡已用显存。"""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            value = gpu_used_mb()
            if value >= 0:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def free_gpu() -> None:
    asr_engine.unload()
    whisper_engine.unload()
    gc.collect()
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# 单个引擎 / 模型
# ---------------------------------------------------------------------------


def measure_whisper(
    size: str,
    clips: list[dict[str, Any]],
    *,
    device: str = "cuda",
    compute_type: str = "int8_float16",
    accuracy: bool = True,
) -> dict[str, Any]:
    free_gpu()
    gc.collect()
    time.sleep(0.5)
    before = gpu_used_mb()
    if device == "cuda":
        # 冷加载：从卸干净的状态开始，采样器每 100ms 记一次整卡 used。
        with GpuSampler() as sampler:
            load_started = time.perf_counter()
            whisper_engine.ensure_loaded(size, device, compute_type)
            load_seconds = time.perf_counter() - load_started
        after = gpu_used_mb()
        peak = max(sampler.samples) if sampler.samples else after
    else:
        load_started = time.perf_counter()
        whisper_engine.ensure_loaded(size, device, compute_type)
        load_seconds = time.perf_counter() - load_started
        before = after = peak = gpu_used_mb()

    # 预热（第一次推理包含 kernel 初始化，不计入延迟）。
    whisper_engine.transcribe(clips[0]["file"], size, "zh", timestamps=True)

    rows: list[dict[str, Any]] = []
    for item in clips:
        started = time.perf_counter()
        result = whisper_engine.transcribe(
            item["file"], size, "zh", timestamps=True
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "file": item["file"],
                "duration": item["duration"],
                "text": result.get("text") or "",
                "elapsed": round(elapsed, 3),
                "reference": item["text"],
                "cer": round(cer(item["text"], result.get("text") or ""), 4),
            }
        )

    total_audio = sum(row["duration"] for row in rows)
    total_time = sum(row["elapsed"] for row in rows)
    result_row: dict[str, Any] = {
        "engine": "whisper",
        "model": size,
        "device": device,
        "compute_type": compute_type,
        "load_seconds": round(load_seconds, 2),
        "vram": {
            "before_mb": before,
            "after_mb": after,
            "peak_mb": peak,
            "delta_mb": after - before,
            "peak_delta_mb": peak - before,
            "samples": len(sampler.samples) if device == "cuda" else 0,
        },
        "clips": rows,
        "cer": round(statistics.mean(row["cer"] for row in rows), 4) if accuracy else None,
        "rtf": round(total_audio / total_time, 3) if total_time else None,
        "latency_mean": round(total_time / len(rows), 3) if rows else None,
        "latency_median": round(statistics.median(row["elapsed"] for row in rows), 3),
        "total_audio": round(total_audio, 3),
        "total_time": round(total_time, 3),
    }
    whisper_engine.unload()
    gc.collect()
    time.sleep(0.3)
    result_row["vram"]["after_unload_mb"] = gpu_used_mb()
    return result_row


def measure_sensevoice(clips: list[dict[str, Any]]) -> dict[str, Any]:
    free_gpu()
    gc.collect()
    time.sleep(0.5)
    before = gpu_used_mb()
    with GpuSampler() as sampler:
        load_started = time.perf_counter()
        asr_engine.ensure_loaded()
        load_seconds = time.perf_counter() - load_started
    after = gpu_used_mb()
    peak = max(sampler.samples) if sampler.samples else after

    asr_engine.transcribe(clips[0]["file"], language="zh")

    rows: list[dict[str, Any]] = []
    for item in clips:
        started = time.perf_counter()
        result = asr_engine.transcribe(item["file"], language="zh")
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "file": item["file"],
                "duration": item["duration"],
                "text": result.get("text") or "",
                "elapsed": round(elapsed, 3),
                "reference": item["text"],
                "cer": round(cer(item["text"], result.get("text") or ""), 4),
            }
        )

    total_audio = sum(row["duration"] for row in rows)
    total_time = sum(row["elapsed"] for row in rows)
    result_row: dict[str, Any] = {
        "engine": "sensevoice",
        "model": "SenseVoiceSmall",
        "device": "cuda:0",
        "compute_type": "fp32",
        "load_seconds": round(load_seconds, 2),
        "vram": {
            "before_mb": before,
            "after_mb": after,
            "peak_mb": peak,
            "delta_mb": after - before,
            "peak_delta_mb": peak - before,
            "samples": len(sampler.samples),
        },
        "clips": rows,
        "cer": round(statistics.mean(row["cer"] for row in rows), 4),
        "rtf": round(total_audio / total_time, 3) if total_time else None,
        "latency_mean": round(total_time / len(rows), 3) if rows else None,
        "latency_median": round(statistics.median(row["elapsed"] for row in rows), 3),
        "total_audio": round(total_audio, 3),
        "total_time": round(total_time, 3),
    }
    asr_engine.unload()
    gc.collect()
    time.sleep(0.3)
    result_row["vram"]["after_unload_mb"] = gpu_used_mb()
    return result_row


# ---------------------------------------------------------------------------
# 长音频：segments 与真实句子起点对齐
# ---------------------------------------------------------------------------


def expected_sentence_starts(manifest: dict[str, Any]) -> list[float]:
    """按合成方式推算每句在长音频里的真实起点（秒）。"""
    starts: list[float] = []
    cursor = 0.0
    order = [item for item in manifest["segments"]] + manifest["segments"][:4]
    gap = 0.6
    for index, item in enumerate(order):
        starts.append(round(cursor, 3))
        cursor += item["duration"] + (gap if index < len(order) - 1 else 0.0)
    return starts


def cut_audio(src: Path, start: float, end: float, dst: Path) -> None:
    duration = max(0.05, end - start)
    subprocess.run(
        [
            str(ffmpeg_tool.ffmpeg_path()),
            "-hide_banner",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
        creationflags=CREATIONFLAGS,
    )


def measure_long_audio(manifest: dict[str, Any]) -> dict[str, Any]:
    long_info = manifest["long"]
    path = Path(long_info["file"])
    free_gpu()
    whisper_engine.ensure_loaded("small", "cuda", "int8_float16")
    started = time.perf_counter()
    result = whisper_engine.transcribe(str(path), "small", "zh", timestamps=True)
    elapsed = time.perf_counter() - started
    segments = result.get("segments") or []
    duration = result.get("duration") or long_info["duration"]

    expected_starts = expected_sentence_starts(manifest)
    alignments = []
    for segment in segments:
        nearest = min(expected_starts, key=lambda value: abs(value - segment["start"]))
        alignments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "nearest_expected_start": nearest,
                "start_error": round(segment["start"] - nearest, 3),
            }
        )

    # 抽查 3 段：裁出来分别用 whisper small 与 sensevoice 复recognize，和原段文本比 CER。
    checks = []
    asr_engine.ensure_loaded()
    temp_dir = ROOT / "outputs" / "eval" / "crops"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for index in (0, len(segments) // 2, len(segments) - 1):
        if index < 0 or not segments:
            continue
        segment = segments[index]
        crop = temp_dir / f"segment_{index:02d}.wav"
        cut_audio(path, segment["start"], segment["end"], crop)
        whisper_again = whisper_engine.transcribe(str(crop), "small", "zh", timestamps=False)
        sense = asr_engine.transcribe(str(crop), language="zh")
        checks.append(
            {
                "index": index,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "whisper_crop_text": whisper_again.get("text") or "",
                "whisper_crop_cer": round(cer(segment["text"], whisper_again.get("text") or ""), 4),
                "sensevoice_crop_text": sense.get("text") or "",
                "sensevoice_crop_cer": round(cer(segment["text"], sense.get("text") or ""), 4),
            }
        )

    asr_engine.unload()
    whisper_engine.unload()
    gc.collect()
    return {
        "file": str(path),
        "duration": duration,
        "elapsed": round(elapsed, 3),
        "rtf": round(duration / elapsed, 3) if elapsed else None,
        "segments": len(segments),
        "seconds_per_segment": round(duration / len(segments), 2) if segments else None,
        "segments_preview": segments,
        "expected_starts": expected_starts,
        "alignments": alignments,
        "crop_checks": checks,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _print_table(results: dict[str, Any]) -> None:
    print("\n=== 准确率 / 速度 ===")
    header = f"{'engine':<12}{'model':<10}{'CER':>8}{'RTF':>8}{'延迟均值':>10}{'延迟中位':>10}{'加载秒':>8}"
    print(header)
    print("-" * len(header))
    for row in results["runs"]:
        cer_text = f"{row['cer']:.2%}" if row.get("cer") is not None else "-"
        print(
            f"{row['engine']:<12}{row['model']:<10}{cer_text:>8}"
            f"{row['rtf']:>8.2f}{row['latency_mean']:>10.2f}"
            f"{row['latency_median']:>10.2f}{row['load_seconds']:>8.1f}"
        )

    print("\n=== 显存（整卡 used，nvidia-smi 100ms 采样）===")
    header = f"{'model':<22}{'加载前':>10}{'加载后':>10}{'峰值':>10}{'增量':>10}{'卸载后':>10}"
    print(header)
    print("-" * len(header))
    for row in results["runs"]:
        vram = row["vram"]
        print(
            f"{(row['engine'] + '/' + str(row['model'])):<22}"
            f"{vram['before_mb']:>10}{vram['after_mb']:>10}{vram['peak_mb']:>10}"
            f"{vram['delta_mb']:>10}{vram.get('after_unload_mb', 0):>10}"
        )

    print("\n=== 长音频（whisper small）===")
    long_result = results["long_audio"]
    print(
        f"时长 {long_result['duration']:.2f}s / 处理 {long_result['elapsed']:.2f}s / "
        f"RTF {long_result['rtf']:.2f} / {long_result['segments']} 段"
    )
    for check in long_result["crop_checks"]:
        print(
            f"  段{check['index']}: {check['start']:.2f}~{check['end']:.2f}s "
            f"原文本 CER=whisper {check['whisper_crop_cer']:.2%} / "
            f"sensevoice {check['sensevoice_crop_cer']:.2%}"
        )
    for alignment in long_result["alignments"][:12]:
        print(
            f"  seg {alignment['start']:.2f}~{alignment['end']:.2f} "
            f"起点误差 {alignment['start_error']:+.2f}s  {alignment['text'][:40]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="识别引擎实测")
    parser.add_argument("--no-cpu", action="store_true", help="跳过 CPU small 测试")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest["segments"]
    print(f"评测集：{len(clips)} 条短音频，长音频 {manifest['long']['duration']}s")

    results: dict[str, Any] = {"runs": [], "manifest": str(MANIFEST)}
    started_all = time.perf_counter()

    for size in ("tiny", "base", "small", "medium"):
        print(f"\n>>> whisper {size}（cuda / int8_float16）", flush=True)
        row = measure_whisper(size, clips)
        row["stage"] = "gpu"
        results["runs"].append(row)
        print(
            f"    CER {row['cer']:.2%}  RTF {row['rtf']:.2f}  "
            f"加载 {row['load_seconds']:.1f}s  显存 +{row['vram']['delta_mb']}MB"
            f"（峰值 +{row['vram']['peak_delta_mb']}MB）",
            flush=True,
        )

    print("\n>>> sensevoice（cuda:0）", flush=True)
    row = measure_sensevoice(clips)
    row["stage"] = "gpu"
    results["runs"].append(row)
    print(
        f"    CER {row['cer']:.2%}  RTF {row['rtf']:.2f}  "
        f"加载 {row['load_seconds']:.1f}s  显存 +{row['vram']['delta_mb']}MB"
        f"（峰值 +{row['vram']['peak_delta_mb']}MB）",
        flush=True,
    )

    if not args.no_cpu:
        print("\n>>> whisper small（cpu / int8）", flush=True)
        row = measure_whisper(
            "small", clips, device="cpu", compute_type="int8", accuracy=False
        )
        row["stage"] = "cpu"
        results["runs"].append(row)
        print(f"    RTF {row['rtf']:.2f}  延迟 {row['latency_mean']:.2f}s", flush=True)

    print("\n>>> 长音频 whisper small", flush=True)
    results["long_audio"] = measure_long_audio(manifest)

    results["total_seconds"] = round(time.perf_counter() - started_all, 1)
    results["gpu_name"] = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
            creationflags=CREATIONFLAGS,
        )
        .strip()
        .splitlines()[0]
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print_table(results)
    print(f"\n结果已写入：{OUT_JSON}（总耗时 {results['total_seconds']}s）")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
