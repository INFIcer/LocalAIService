"""识别引擎并发闸门测试。

覆盖四件事（全部用假模型，不加载真实 SenseVoice、不占显存）：
1. 容量 1：同一时刻只有一个推理，行为与旧版串行锁一致；
2. 容量 2：两个推理真的同时跑，且各自结果不会串；
3. 卸载会等推理退出，期间模型不被拆掉；
4. 卸载等待超时只放弃本次卸载，不破坏信号量计数，之后可正常卸载。

测试直接操作 ``engine/asr_engine.py`` 的 ``_InferenceGate`` 与模块级
``_INFER_GATE``；需要真实模型的路径由 ``tests/test_jobs.py`` 的假执行器覆盖。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import soundfile as sf

from localai.engines._native import import_engine_module


@pytest.fixture()
def asr_engine():
    """加载原生识别引擎模块（只 import，不加载模型）。"""
    return import_engine_module("asr_engine")


class _FakeModel:
    """线程安全的假 SenseVoice：记录并发峰值，按输入文件名返回不同文本。"""

    def __init__(self, delay: float = 0.15) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.outputs: list[str] = []

    def generate(self, input: str, **_: object) -> list[dict[str, str]]:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if self.block:
                assert self.release.wait(10), "测试等待超时：推理没有收到放行信号"
            else:
                time.sleep(self.delay)
            text = f"<|zh|><|NEUTRAL|><|Speech|><|withitn|>{Path(input).stem}"
            with self._lock:
                self.outputs.append(text)
            return [{"text": text}]
        finally:
            with self._lock:
                self.active -= 1


def _make_wavs(root: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = root / f"clip_{index}.wav"
        # 0.2 秒静音；识别本身用假模型，时长只用于结果字段。
        sf.write(str(path), [0.0] * 3200, 16000, subtype="PCM_16")
        paths.append(path)
    return paths


def _install_fake(
    asr_engine,
    monkeypatch: pytest.MonkeyPatch,
    capacity: int,
    model: _FakeModel,
) -> None:
    """替换模块级闸门 / 模型 / 显存日志，走真实 transcribe 路径。"""
    monkeypatch.setattr(asr_engine, "_INFER_GATE", asr_engine._InferenceGate(capacity))
    monkeypatch.setattr(asr_engine, "_model", model)
    monkeypatch.setattr(asr_engine, "_loaded_device", "cpu")
    monkeypatch.setattr(asr_engine, "_log_vram", lambda _tag: None)
    monkeypatch.setattr(asr_engine, "_audio_duration", lambda _path: 0.2)
    # 不碰 CUDA：真实 _free_cuda() 会 import torch，破坏「导入服务层不带 torch」。
    monkeypatch.setattr(asr_engine, "_free_cuda", lambda: None)


def test_capacity_one_is_serial(asr_engine, tmp_path: Path, monkeypatch) -> None:
    """容量 1：4 个推理并发提交，峰值必须只有 1。"""
    model = _FakeModel(delay=0.12)
    _install_fake(asr_engine, monkeypatch, 1, model)
    wavs = _make_wavs(tmp_path, 4)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda p: asr_engine.transcribe(str(p)), wavs))
    elapsed = time.perf_counter() - started

    assert model.peak == 1, f"容量 1 出现并发：peak={model.peak}"
    assert elapsed >= 0.45, f"串行总耗时异常：{elapsed:.3f}s"
    assert len({item["text"] for item in results}) == 4
    assert asr_engine._INFER_GATE.snapshot()["in_use"] == 0


def test_capacity_two_really_parallel(asr_engine, tmp_path: Path, monkeypatch) -> None:
    """容量 2：两个推理真的同时进入，且结果不串。"""
    model = _FakeModel(delay=0.25)
    _install_fake(asr_engine, monkeypatch, 2, model)
    wavs = _make_wavs(tmp_path, 4)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda p: asr_engine.transcribe(str(p)), wavs))
    elapsed = time.perf_counter() - started

    assert model.peak == 2, f"容量 2 没有真并发：peak={model.peak}"
    # 4 条 x 0.25s，2 路并行的理想总耗时约 0.5s；串行会是 1.0s。
    assert elapsed < 0.85, f"并发总耗时没有下降：{elapsed:.3f}s"
    assert len({item["text"] for item in results}) == 4
    assert asr_engine._INFER_GATE.snapshot() == {
        "capacity": 2,
        "in_use": 0,
        "peak": 2,
    }


def test_unload_waits_for_inference(asr_engine, tmp_path: Path, monkeypatch) -> None:
    """卸载要先等推理退出，不能把正在使用的模型拆掉。"""
    model = _FakeModel()
    model.block = True
    _install_fake(asr_engine, monkeypatch, 2, model)
    wav = _make_wavs(tmp_path, 1)[0]

    inference = threading.Thread(target=asr_engine.transcribe, args=(str(wav),))
    inference.start()
    assert model.started.wait(5), "假推理没有启动"

    unload = threading.Thread(target=asr_engine.unload)
    unload.start()
    time.sleep(0.2)

    assert unload.is_alive(), "卸载没有等待正在跑的推理"
    assert asr_engine.is_loaded(), "卸载提前拆掉了模型"

    model.release.set()
    inference.join(5)
    unload.join(5)
    assert not inference.is_alive()
    assert not unload.is_alive()
    assert not asr_engine.is_loaded(), "推理退出后卸载没有完成"


def test_unload_timeout_keeps_model_and_semaphore_healthy(
    asr_engine, tmp_path: Path, monkeypatch
) -> None:
    """卸载等待超时：只放弃本次卸载，模型还在，信号量计数不乱。"""
    model = _FakeModel()
    model.block = True
    _install_fake(asr_engine, monkeypatch, 2, model)
    monkeypatch.setattr(asr_engine, "ASR_UNLOAD_TIMEOUT", 0.2)
    wav = _make_wavs(tmp_path, 1)[0]

    inference = threading.Thread(target=asr_engine.transcribe, args=(str(wav),))
    inference.start()
    assert model.started.wait(5)

    started = time.perf_counter()
    asr_engine.unload()  # 应在约 0.2 秒后放弃，而不是一直卡住
    elapsed = time.perf_counter() - started
    assert 0.15 <= elapsed < 2.0, f"卸载超时未生效：{elapsed:.3f}s"
    assert asr_engine.is_loaded(), "超时后模型不应被拆掉"

    model.release.set()
    inference.join(5)
    assert not inference.is_alive()

    # 计数健康：再卸一次应当成功。
    asr_engine.unload()
    assert not asr_engine.is_loaded()
    assert asr_engine._INFER_GATE.snapshot()["in_use"] == 0


def test_adapter_capacity_follows_config(tmp_path: Path) -> None:
    """适配器闸门容量 = 服务配置，且同步写回引擎环境变量。"""
    from localai.config import Config
    from localai.engines.asr_sensevoice import AsrAdapter

    cfg = Config.from_env(home=str(tmp_path))
    cfg.jobs_parallel_asr = 2
    cfg.asr_parallel = 2
    adapter = AsrAdapter(cfg)

    assert adapter.parallel_capacity == 2
    assert adapter.CAN_INFER_IN_PARALLEL is True
    assert adapter.infer_lock.capacity == 2
    # 引擎模块的闸门容量也是 2。
    assert adapter.get_module().parallel_status()["capacity"] == 2
    # 还原，避免影响同进程里后面的用例。
    adapter.get_module().configure_parallel(1)


def test_config_asr_parallel_precedence_and_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新名优先；只设旧名也认；超出 1~4 会被夹紧。"""
    from localai.config import Config

    monkeypatch.setenv("LOCALAI_ASR_PARALLEL", "1")
    monkeypatch.setenv("LOCALAI_JOBS_PARALLEL_ASR", "3")
    cfg = Config.from_env(home=str(tmp_path))
    assert cfg.asr_parallel == 1
    assert cfg.jobs_parallel_asr == 1

    monkeypatch.delenv("LOCALAI_ASR_PARALLEL")
    cfg = Config.from_env(home=str(tmp_path))
    assert cfg.asr_parallel == 3
    assert cfg.jobs_parallel_asr == 3

    monkeypatch.setenv("LOCALAI_ASR_PARALLEL", "9")
    cfg = Config.from_env(home=str(tmp_path))
    assert cfg.asr_parallel == 4
