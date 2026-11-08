"""声纹识别引擎（campplus 192 维 embedding）
================================================

用途：判断两段语音是不是同一个人说的。

设计要点
--------
1. **懒加载**：服务启动不加载；第一次调 ``embedding()``/``identify()`` 才把 FunASR 的
   campplus 模型读进内存（GPU 可用就上卡），空闲由服务层的看门狗调用 ``unload()`` 释放。
2. **比对方式**：embedding 先 L2 归一化，再用余弦相似度（-1~1，实际人脸/声纹一般 0~1）。
   同一个人的两条语音通常 > 0.7；不同人通常 < 0.6。
3. **已知声纹**：每个名字一个 json，存多条样本的均值向量，放 ``SPEAKERS_DIR``。
4. **未知声纹聚类**：不认识的声音不直接丢掉，而是和 ``unknown.json`` 里的历史聚类比对，
   够像就复用同一个 ``未知-N``，不够像才新建一个。这样同一个人说三次不会变成三个未知。
5. **隐私**：只在本机存 192 维向量和名字，不存原始音频、不上传任何地方。

模型目录：``E:\\LocalAIService\\models\\campplus``（campplus_cn_common.bin + config.yaml）
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
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
SPEAKERS_DIR = Path(os.environ.get("INDEX_TTS_SPEAKERS_DIR", str(SERVICE_ROOT / "speakers")))
# 判定「同一个已知人」的阈值；未知聚类用更低一点的阈值，避免把不同人并成一个。
KNOWN_THRESHOLD = float(os.environ.get("INDEX_TTS_SPEAKER_THRESHOLD", "0.65"))
UNKNOWN_THRESHOLD = float(os.environ.get("INDEX_TTS_SPEAKER_UNKNOWN_THRESHOLD", "0.62"))
EMBED_DIM = 192

_MODEL: Any = None
_LOAD_LOCK = threading.Lock()
_LOAD_ERROR: Optional[str] = None
_LOAD_SECONDS: Optional[float] = None

_UNKNOWN_FILE = SPEAKERS_DIR / "unknown.json"


# ---------------------------------------------------------------------------
# 模型状态
# ---------------------------------------------------------------------------


def model_status() -> dict[str, Any]:
    """检查 campplus 模型文件是否齐全。"""
    missing = [n for n in ("campplus_cn_common.bin", "config.yaml") if not (CAMPPUS_DIR / n).exists()]
    return {
        "ready": not missing,
        "loaded": is_loaded(),
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# 已知声纹库
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    cleaned = "".join(ch for ch in str(name or "").strip() if ch not in '\\/:*?"<>|').strip()
    if not cleaned:
        raise ValueError("名字不能为空，也不能只包含非法字符")
    return cleaned[:60]


def _speaker_file(name: str) -> Path:
    return SPEAKERS_DIR / f"{name}.json"


def list_speakers() -> list[dict[str, Any]]:
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for path in sorted(SPEAKERS_DIR.glob("*.json")):
        if path.name == _UNKNOWN_FILE.name:
            continue
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out.append(
            {
                "name": data.get("name") or path.stem,
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "samples": len(data.get("embeddings") or []),
            }
        )
    return out


def enroll(name: str, audio_path: str) -> dict[str, Any]:
    """把一段音频登记成某个人的声纹；同名累加样本并重新取平均。"""
    safe = _safe_name(name)
    vec = embedding(audio_path)
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = _speaker_file(safe)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    data: dict[str, Any] = {"name": safe, "created_at": now, "embeddings": []}
    if path.exists():
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            pass
    data["name"] = safe
    data.setdefault("created_at", now)
    data.setdefault("embeddings", [])
    data["embeddings"].append([float(x) for x in vec])
    data["updated_at"] = now
    data["source_files"] = list(dict.fromkeys([*(data.get("source_files") or []), Path(audio_path).name]))[:20]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    logger.info("声纹登记：%s（累计 %d 条样本）", safe, len(data["embeddings"]))
    return {"ok": True, "name": safe, "samples": len(data["embeddings"]), "created_at": data["created_at"]}


def delete_speaker(name: str) -> bool:
    safe = _safe_name(name)
    path = _speaker_file(safe)
    if not path.exists():
        return False
    path.unlink()
    logger.info("删除声纹：%s", safe)
    return True


def _speaker_centroid(data: dict[str, Any]) -> Optional[np.ndarray]:
    embeddings = data.get("embeddings") or []
    if not embeddings:
        return None
    stacked = np.asarray(embeddings, dtype=np.float32)
    return _normalize(stacked.mean(axis=0))


def identify(audio_path: str) -> dict[str, Any]:
    """跟已知声纹库比对，返回最像的人（可能没有）。"""
    vec = embedding(audio_path)
    matches: list[dict[str, Any]] = []
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SPEAKERS_DIR.glob("*.json")):
        if path.name == _UNKNOWN_FILE.name:
            continue
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        centroid = _speaker_centroid(data)
        if centroid is None:
            continue
        matches.append({"name": data.get("name") or path.stem, "score": round(cosine(vec, centroid), 4)})
    matches.sort(key=lambda m: m["score"], reverse=True)
    best = matches[0] if matches else {"name": None, "score": 0.0}
    known = bool(best["name"]) and best["score"] >= KNOWN_THRESHOLD
    return {
        "ok": True,
        "embedding": vec,  # 内部用，接口层剔除
        "matches": matches[:5],
        "best": best["name"] if known else None,
        "score": best["score"],
        "threshold": KNOWN_THRESHOLD,
        "known": known,
    }


# ---------------------------------------------------------------------------
# 未知声纹聚类（跨请求保持稳定）
# ---------------------------------------------------------------------------


def _load_unknown() -> dict[str, Any]:
    if not _UNKNOWN_FILE.exists():
        return {"next_id": 1, "clusters": []}
    try:
        data = json.loads(_UNKNOWN_FILE.read_text("utf-8"))
        data.setdefault("clusters", [])
        data.setdefault("next_id", len(data["clusters"]) + 1)
        return data
    except Exception:  # noqa: BLE001
        return {"next_id": 1, "clusters": []}


def _save_unknown(data: dict[str, Any]) -> None:
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    _UNKNOWN_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def classify_unknown(vec: np.ndarray) -> dict[str, Any]:
    """把未知声纹归到稳定的「未知-N」；不够像就新建一个。"""
    data = _load_unknown()
    best_name = None
    best_score = 0.0
    best_cluster = None
    for cluster in data["clusters"]:
        centroid = _speaker_centroid(cluster)
        if centroid is None:
            continue
        score = cosine(vec, centroid)
        if score > best_score:
            best_score = score
            best_name = cluster.get("name")
            best_cluster = cluster
    if best_cluster is not None and best_score >= UNKNOWN_THRESHOLD:
        best_cluster.setdefault("embeddings", []).append([float(x) for x in vec])
        best_cluster["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_unknown(data)
        return {"name": best_name, "score": round(best_score, 4), "known": False, "cluster_score": round(best_score, 4)}

    name = f"未知-{data['next_id']}"
    data["next_id"] = int(data["next_id"]) + 1
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["clusters"].append(
        {
            "name": name,
            "embeddings": [[float(x) for x in vec]],
            "created_at": now,
            "last_seen": now,
        }
    )
    _save_unknown(data)
    return {"name": name, "score": 0.0, "known": False, "cluster_score": 0.0}


def speaker_for(audio_path: str) -> dict[str, Any]:
    """一站式：先认已知，不中再归未知聚类。返回可直接塞进 /asr 的字段。"""
    result = identify(audio_path)
    vec = result.pop("embedding")
    if result["known"]:
        return {"name": result["best"], "score": result["score"], "known": True}
    unknown = classify_unknown(vec)
    return {"name": unknown["name"], "score": unknown["score"], "known": False}


def unknown_clusters() -> list[dict[str, Any]]:
    data = _load_unknown()
    return [
        {
            "name": c.get("name"),
            "samples": len(c.get("embeddings") or []),
            "created_at": c.get("created_at"),
            "last_seen": c.get("last_seen"),
        }
        for c in data["clusters"]
    ]
