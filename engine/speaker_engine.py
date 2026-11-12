"""声纹向量引擎（campplus 192 维 embedding）
================================================

本模块只做**纯分析**：一段音频 -> 一个 192 维、L2 归一化的声纹向量。
「这个向量是谁」不在这里判断——名字、样本、阈值、比对策略都由调用方的记忆层负责，
服务端不保存任何「名字 -> 样本」映射，也不认识未知聚类。

设计要点
--------
1. **懒加载**：服务启动不加载；第一次调 ``embedding()`` 才把 FunASR 的 campplus
   模型读进内存（GPU 可用就上卡），空闲由服务层的看门狗调用 ``unload()`` 释放。
2. **可比性**：返回值统一做 L2 归一化，调用方用余弦相似度就能直接比较。
3. **隐私**：只在本机内存里算向量，不落任何身份数据，也不上传。

模型目录：``E:/LocalAIService/models/campplus``（campplus_cn_common.bin + config.yaml）
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("indextts-speaker")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 服务根目录（speaker_engine.py 在 engine\ 下）；改名或走 junction 都会解析到真实目录。
SERVICE_ROOT = Path(__file__).resolve().parent.parent

CAMPPUS_DIR = Path(
    os.environ.get("INDEX_TTS_CAMPPUS_DIR", str(SERVICE_ROOT / "models" / "campplus"))
)
EMBED_DIM = 192
MODEL_NAME = "campplus"

_MODEL: Any = None
_LOAD_LOCK = threading.Lock()
_LOAD_ERROR: Optional[str] = None
_LOAD_SECONDS: Optional[float] = None


# ---------------------------------------------------------------------------
# 模型状态
# ---------------------------------------------------------------------------


def model_status() -> dict[str, Any]:
    """检查 campplus 模型文件是否齐全。"""
    missing = [n for n in ("campplus_cn_common.bin", "config.yaml") if not (CAMPPUS_DIR / n).exists()]
    return {
        "ready": not missing,
        "loaded": is_loaded(),
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "model_dir": str(CAMPPUS_DIR),
        "missing": missing,
        "detail": "声纹模型文件就绪" if not missing else f"缺少文件：{', '.join(missing)}",
        "load_seconds": _LOAD_SECONDS,
        "load_error": _LOAD_ERROR,
    }


def is_loaded() -> bool:
    return _MODEL is not None


def ensure_loaded() -> Any:
    """加载 campplus（线程安全，重复调用只加载一次）。"""
    global _MODEL, _LOAD_ERROR, _LOAD_SECONDS
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        status = model_status()
        if not status["ready"]:
            raise RuntimeError(status["detail"])
        started = time.perf_counter()
        try:
            from funasr import AutoModel  # 延迟 import：没装 funasr 也能起服务

            device = "cuda:0"
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
            logger.info("加载 campplus 声纹模型：%s (device=%s)", CAMPPUS_DIR, device)
            _MODEL = AutoModel(
                model=str(CAMPPUS_DIR),
                device=device,
                disable_update=True,
            )
            _LOAD_ERROR = None
            _LOAD_SECONDS = round(time.perf_counter() - started, 2)
            logger.info("campplus 加载完成：%.2fs", _LOAD_SECONDS)
            return _MODEL
        except Exception as exc:  # noqa: BLE001
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            logger.error("campplus 加载失败：%s", _LOAD_ERROR)
            raise RuntimeError(f"声纹模型加载失败：{_LOAD_ERROR}") from exc


def unload() -> None:
    """释放声纹模型，把显存还回去。"""
    global _MODEL, _LOAD_SECONDS
    with _LOAD_LOCK:
        if _MODEL is None:
            return
        _MODEL = None
        _LOAD_SECONDS = None
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    logger.info("campplus 声纹模型已卸载")


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------


def _to_np(value: Any) -> np.ndarray:
    """把 funasr 返回的 tensor/list 统一成 float32 numpy 向量。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise RuntimeError("声纹 embedding 为空（音频可能是静音）")
    return vec / norm


def embedding(audio_path: str) -> np.ndarray:
    """算一段音频的 192 维声纹向量（已做 L2 归一化）。"""
    model = ensure_loaded()
    result = model.generate(input=str(audio_path), cache={}, disable_pbar=True)
    if not result:
        raise RuntimeError("声纹模型没有返回结果")
    item = result[0] if isinstance(result, (list, tuple)) else result
    for key in ("spk_embedding", "embeddings", "embedding", "spk_emb"):
        if isinstance(item, dict) and key in item:
            vec = _to_np(item[key])
            if vec.size >= EMBED_DIM:
                return _normalize(vec[:EMBED_DIM])
    # 有些版本直接把向量放在返回值里
    try:
        vec = _to_np(item)
        if vec.size >= EMBED_DIM:
            return _normalize(vec[:EMBED_DIM])
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(f"没从声纹模型输出里找到 embedding，键={list(item) if isinstance(item, dict) else type(item)}")
