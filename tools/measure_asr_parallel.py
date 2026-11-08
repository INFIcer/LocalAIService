"""实测识别引擎并发：同一批 wav -> 逐条延迟 / 总耗时 / 吞吐 / 峰值显存。

用法（服务已经按目标并发启动）::

    python tools/measure_asr_parallel.py --parallel 2 --count 8 \
        --out logs/asr_parallel_p2.json

说明：
- 走真实 HTTP ``POST /jobs`` + 轮询 ``GET /jobs/{id}``，不绕开调度器；
- 批量前先跑一条预热，确保 SenseVoice 已加载，测量窗口内不再包含加载时间；
- 每 100ms 调一次 ``nvidia-smi`` 记录整卡 used，取峰值；
- 逐条延迟优先用任务结果里的 ``elapsed_ms``（服务端处理耗时），
  同时保留客户端从提交到完成的墙钟耗时，便于核对排队影响。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

BASE = "http://127.0.0.1:7880"
NVIDIA_SMI = Path(r"C:\Windows\System32\nvidia-smi.exe")


class VramSampler(threading.Thread):
    """后台采样整卡 used（MB），用于峰值显存。"""

    def __init__(self, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        str(NVIDIA_SMI),
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                self.samples.append(int(out.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001 - 采样失败不影响主流程
                pass
            self._stop.wait(self.interval)

    def stop(self) -> int:
        self._stop.set()
        self.join(5)
        return max(self.samples) if self.samples else -1


def _gpu_used_mb() -> int:
    out = subprocess.check_output(
        [str(NVIDIA_SMI), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    return int(out.strip().splitlines()[0])


def _api(path: str) -> Any:
    response = requests.get(f"{BASE}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _submit(path: Path) -> str:
    response = requests.post(
        f"{BASE}/jobs",
        json={
            "kind": "asr",
            "payload": {"file": str(path), "language": "auto", "speaker": False},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["id"]


def _wait(job_id: str, timeout: float = 120) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = _api(f"/jobs/{job_id}")
        if job["state"] in ("done", "failed", "canceled"):
            return job
        time.sleep(0.03)
    raise TimeoutError(f"任务 {job_id} 超时未结束")


def _prewarm(path: Path) -> None:
    job = _wait(_submit(path), timeout=180)
    if job["state"] != "done":
        raise RuntimeError(f"预热失败：{job}")


def _batch(files: list[Path]) -> dict[str, Any]:
    sampler = VramSampler()
    sampler.start()
    time.sleep(0.5)
    baseline = _gpu_used_mb()
    started = time.perf_counter()
    submitted: list[tuple[str, Path, float]] = []
    for path in files:
        job_id = _submit(path)
        submitted.append((job_id, path, time.perf_counter()))
    jobs = []
    for job_id, path, submitted_at in submitted:
        job = _wait(job_id)
        finished = time.perf_counter()
        result = job.get("result") or {}
        jobs.append(
            {
                "file": path.name,
                "state": job["state"],
                "text": result.get("text"),
                "elapsed_ms": result.get("elapsed_ms"),
                "wait_seconds": round(finished - submitted_at, 3),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "error": job.get("error"),
            }
        )
    wall = time.perf_counter() - started
    time.sleep(1.0)
    final = _gpu_used_mb()
    peak = sampler.stop()
    latencies = [item["elapsed_ms"] / 1000 for item in jobs if item["elapsed_ms"] is not None]
    return {
        "count": len(jobs),
        "wall_seconds": round(wall, 3),
        "throughput_per_second": round(len(jobs) / wall, 3) if wall else None,
        "latency_seconds": [round(value, 3) for value in latencies],
        "latency_mean_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_median_seconds": round(statistics.median(latencies), 3) if latencies else None,
        "peak_vram_mb": peak,
        "baseline_vram_mb": baseline,
        "final_vram_mb": final,
        "failed": [item["file"] for item in jobs if item["state"] != "done"],
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, required=True, help="期望的服务并发容量")
    parser.add_argument("--count", type=int, default=8, help="本次跑多少条")
    parser.add_argument("--out", required=True, help="结果 JSON 路径")
    parser.add_argument(
        "--files",
        nargs="*",
        help="可选：显式指定 wav 列表；默认自动从 outputs 挑 3~10 秒的素材",
    )
    args = parser.parse_args()

    caps = _api("/capabilities")
    capacity = (caps["capabilities"]["asr"]["parallel"] or {}).get("capacity")
    asr_parallel = caps["jobs"].get("asr_parallel")
    if capacity != args.parallel or asr_parallel != args.parallel:
        raise SystemExit(
            f"服务并发不是 {args.parallel}：引擎={capacity} 调度={asr_parallel}；"
            "请先按目标配置重启服务"
        )

    if args.files:
        files = [Path(item) for item in args.files]
    else:
        import soundfile as sf

        candidates: list[Path] = []
        for path in sorted((Path(__file__).resolve().parents[1] / "outputs").glob("*.wav")):
            try:
                duration = sf.info(str(path)).duration
            except Exception:  # noqa: BLE001
                continue
            if 3.0 <= duration <= 10.0:
                candidates.append(path)
        # 去重：同一内容常有多个重命名副本，按文件大小 + 时长挑唯一素材。
        unique: list[Path] = []
        seen: set[tuple[int, int]] = set()
        for path in candidates:
            key = (path.stat().st_size, int(sf.info(str(path)).duration * 1000))
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        files = unique

    if not files:
        raise SystemExit("没有找到 3~10 秒的 wav，请先用 TTS 合成")
    if args.count > len(files):
        files = (files * ((args.count // len(files)) + 1))[: args.count]
    else:
        files = files[: args.count]

    print(f"预热：{files[0].name}")
    _prewarm(files[0])
    time.sleep(1.0)
    print(f"开始测量：parallel={args.parallel} count={len(files)}")
    outcome = _batch(files)
    outcome.update(
        {
            "parallel": args.parallel,
            "engine_capacity": capacity,
            "scheduler_parallel": asr_parallel,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"完成：总耗时={outcome['wall_seconds']}s "
        f"吞吐={outcome['throughput_per_second']} 条/s "
        f"峰值显存={outcome['peak_vram_mb']}MB "
        f"失败={len(outcome['failed'])}"
    )
    print("逐条延迟(s)：" + ", ".join(str(v) for v in outcome["latency_seconds"]))
    if outcome["failed"]:
        print("失败文件：" + ", ".join(outcome["failed"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
