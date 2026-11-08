"""模型清单与后台下载任务。

复用原则
--------
- whisper / 翻译：直接调用 ``engine/whisper_engine.py``、``engine/translate_engine.py``
  里**已有的** ``ensure_weights()``，不另写下载器；
- TTS / ASR / 声纹 / Demucs：直接调用 ``localai.cli`` 里 ``models download``
  用的同一批函数（``_download_snapshot`` / ``_download_demucs``）；
- fsmn-vad：调用 ``engine/asr_engine.py`` 的 ``_ensure_vad_weights()``。

同一时刻只允许一个下载任务；任务跑在独立线程里，不阻塞事件循环。
进度用「目标目录当前字节数 - 开始时字节数」表示，因为 ModelScope /
HuggingFace 的 snapshot_download 不提供统一的总字节回调。
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..constants import (
    MODEL_SPECS,
    REQUIRED_AUX_FILES,
    REQUIRED_MODEL_FILES,
    WHISPER_SIZES,
)
from ..engines._native import import_engine_module

logger = logging.getLogger("localai.admin.models")

#: 与人声分离权重保持一致（和 cli.py 用同一份清单）。
from ..cli import DEMUCS_FILES, DEMUCS_FT_FILES  # noqa: E402

#: 目录里是否存在这些文件就算「已下载」。
WHISPER_REQUIRED = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
TRANSLATE_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
TRANSLATE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
VAD_REQUIRED = ("model.pt", "config.yaml")

#: 能力展示顺序与中文名。
CAPABILITY_TITLES = {
    "tts": "语音合成",
    "asr": "语音识别",
    "whisper": "faster-whisper",
    "translate": "翻译",
    "speaker": "声纹",
    "separate": "人声分离",
    "vad": "语音端点检测",
    "download": "通用下载",
}

#: 下载时 capability 的别名。
_CAPABILITY_ALIASES = {
    "tts": "tts",
    "indextts": "tts",
    "indextts-2.5": "tts",
    "asr": "asr",
    "sensevoice": "asr",
    "whisper": "whisper",
    "faster-whisper": "whisper",
    "translate": "translate",
    "nllb": "translate",
    "speaker": "speaker",
    "campplus": "speaker",
    "separate": "separate",
    "demucs": "separate",
    "vad": "vad",
    "fsmn-vad": "vad",
}


def _normalize_capability(raw: str) -> str:
    key = str(raw or "").strip().lower().replace("_", "-")
    return _CAPABILITY_ALIASES.get(key, "")


def _check_files(directory: Path, names: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for name in names:
        candidate = directory / str(name).replace("\\", "/")
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                present.append(str(name))
            else:
                missing.append(str(name))
        except OSError:
            missing.append(str(name))
    return present, missing


def _dir_size_mb(directory: Path) -> float:
    total = 0
    try:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return round(total / (1024 * 1024), 1)


def _dir_size_bytes(directory: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _files_size_mb(directory: Path, names: tuple[str, ...] | list[str]) -> float:
    total = 0
    for name in names:
        try:
            path = directory / str(name).replace("\\", "/")
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return round(total / (1024 * 1024), 1)


def _translate_dir(cfg: Config) -> Path:
    return cfg.translate_dir / "nllb-200-distilled-600M"


def _item(
    *,
    capability: str,
    name: str,
    path: Path,
    expected: list[str] | tuple[str, ...],
    present: bool,
    missing: list[str],
    size_mb: float,
    active: bool,
    options: list[str] | tuple[str, ...] = (),
    note: str = "",
) -> dict[str, Any]:
    return {
        "capability": capability,
        "capability_title": CAPABILITY_TITLES.get(capability, capability),
        "name": name,
        "path": str(path),
        "present": bool(present),
        "size_mb": size_mb,
        "expected_files": list(expected),
        "missing_files": missing,
        "active": bool(active),
        "options": list(options),
        "note": note,
    }


def model_catalog(cfg: Config) -> list[dict[str, Any]]:
    """返回全部能力的模型清单（只读文件，不加载任何模型）。"""
    items: list[dict[str, Any]] = []

    # ---- TTS ---------------------------------------------------------
    tts_expected = list(REQUIRED_MODEL_FILES) + list(REQUIRED_AUX_FILES)
    _, tts_missing = _check_files(cfg.model_dir, tts_expected)
    items.append(
        _item(
            capability="tts",
            name="IndexTTS-2.5",
            path=cfg.model_dir,
            expected=tts_expected,
            present=not tts_missing,
            missing=tts_missing,
            size_mb=_dir_size_mb(cfg.model_dir),
            active=cfg.tts_backend == "engine",
            options=["engine", "comfy"],
            note="当前后端：" + cfg.tts_backend,
        )
    )

    # ---- SenseVoice --------------------------------------------------
    asr_expected = list(MODEL_SPECS["asr"]["required"])
    _, asr_missing = _check_files(cfg.asr_model_dir, asr_expected)
    items.append(
        _item(
            capability="asr",
            name="SenseVoiceSmall",
            path=cfg.asr_model_dir,
            expected=asr_expected,
            present=not asr_missing,
            missing=asr_missing,
            size_mb=_dir_size_mb(cfg.asr_model_dir),
            active=cfg.asr_engine == "sensevoice",
            options=["sensevoice", "whisper"],
            note="默认识别引擎：" + cfg.asr_engine,
        )
    )

    # ---- faster-whisper（每档一行，方便单独下载/切换） -----------------
    for size in WHISPER_SIZES:
        target = cfg.whisper_dir / size
        _, missing = _check_files(target, WHISPER_REQUIRED)
        items.append(
            _item(
                capability="whisper",
                name=size,
                path=target,
                expected=list(WHISPER_REQUIRED),
                present=not missing,
                missing=missing,
                size_mb=_dir_size_mb(target),
                active=cfg.whisper_model == size,
                options=list(WHISPER_SIZES),
                note="当前档位：" + cfg.whisper_model,
            )
        )

    # ---- NLLB 翻译 ---------------------------------------------------
    translate_path = _translate_dir(cfg)
    _, translate_missing = _check_files(translate_path, TRANSLATE_MODEL_FILES)
    has_weight = any(
        (translate_path / name).is_file() and (translate_path / name).stat().st_size > 0
        for name in TRANSLATE_WEIGHT_FILES
    )
    if not has_weight:
        translate_missing.append("model.safetensors 或 pytorch_model.bin")
    translate_expected = list(TRANSLATE_MODEL_FILES) + ["model.safetensors 或 pytorch_model.bin"]
    items.append(
        _item(
            capability="translate",
            name="nllb-200-distilled-600M",
            path=translate_path,
            expected=translate_expected,
            present=not translate_missing,
            missing=translate_missing,
            size_mb=_dir_size_mb(translate_path),
            active=True,
            options=["nllb"],
            note="唯一翻译模型",
        )
    )

    # ---- 声纹 --------------------------------------------------------
    speaker_expected = list(MODEL_SPECS["speaker"]["required"])
    _, speaker_missing = _check_files(cfg.speaker_model_dir, speaker_expected)
    items.append(
        _item(
            capability="speaker",
            name="campplus",
            path=cfg.speaker_model_dir,
            expected=speaker_expected,
            present=not speaker_missing,
            missing=speaker_missing,
            size_mb=_dir_size_mb(cfg.speaker_model_dir),
            active=True,
        )
    )

    # ---- Demucs ------------------------------------------------------
    separate_options = ["htdemucs", "htdemucs_6s", "htdemucs_ft"]
    for model_name, files in DEMUCS_FILES.items():
        _, missing = _check_files(cfg.demucs_repo, files)
        items.append(
            _item(
                capability="separate",
                name=model_name,
                path=cfg.demucs_repo,
                expected=list(files),
                present=not missing,
                missing=missing,
                size_mb=_files_size_mb(cfg.demucs_repo, files),
                active=model_name == "htdemucs",
                options=separate_options,
            )
        )
    _, ft_missing = _check_files(cfg.demucs_repo, DEMUCS_FT_FILES)
    items.append(
        _item(
            capability="separate",
            name="htdemucs_ft",
            path=cfg.demucs_repo,
            expected=list(DEMUCS_FT_FILES),
            present=not ft_missing,
            missing=ft_missing,
            size_mb=_files_size_mb(cfg.demucs_repo, DEMUCS_FT_FILES),
            active=False,
            options=separate_options,
            note="可选：4 个模型组成的 bag，约 4 倍显存/耗时，默认不下载",
        )
    )

    # ---- fsmn-vad ----------------------------------------------------
    _, vad_missing = _check_files(cfg.asr_vad_dir, VAD_REQUIRED)
    items.append(
        _item(
            capability="vad",
            name="fsmn-vad",
            path=cfg.asr_vad_dir,
            expected=list(VAD_REQUIRED),
            present=not vad_missing,
            missing=vad_missing,
            size_mb=_dir_size_mb(cfg.asr_vad_dir),
            active=True,
            note="SenseVoice 分段转写用；whisper 不需要",
        )
    )
    # ---- yt-dlp（不是权重，但同样在模型页展示版本与状态） --------------
    try:
        from ..engines._native import import_engine_module

        dl_module = import_engine_module("download_engine")
        dl_info = dl_module.status_info()
        dl_version = (dl_info.get("yt_dlp") or {}).get("version")
        dl_path = (dl_info.get("yt_dlp") or {}).get("path")
    except Exception as exc:  # noqa: BLE001 - 状态读取失败不影响模型清单
        dl_version = None
        dl_path = str(exc)
    items.append(
        _item(
            capability="download",
            name="yt-dlp",
            path=Path(dl_path) if dl_path and Path(dl_path).is_file() else cfg.downloads_dir,
            expected=[],
            present=bool(dl_version),
            missing=[] if dl_version else ["yt-dlp"],
            size_mb=0.0,
            active=True,
            note=(
                f"版本 {dl_version}；下载目录 {cfg.downloads_dir}；"
                f"单文件上限 {cfg.download_max_mb}MB"
                if dl_version
                else "未安装：engine/.venv 执行 python -m pip install -U yt-dlp"
            ),
        )
    )
    return items


# ---------------------------------------------------------------------------
# 下载任务
# ---------------------------------------------------------------------------
class DownloadError(ValueError):
    """下载参数不合法（路由转 400）。"""


class DownloadBusy(RuntimeError):
    """已有下载任务在运行（路由转 409）。"""


class ModelDownloader:
    """单例下载任务管理器；挂在 ``app.state.model_downloader``。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._task: Optional[dict[str, Any]] = None

    # ---- 参数校验 -----------------------------------------------------
    @staticmethod
    def _resolve(
        cfg: Config, capability: str, name: str
    ) -> tuple[str, str, Path]:
        cap = _normalize_capability(capability)
        if not cap:
            raise DownloadError(
                f"不支持的 capability={capability!r}；"
                f"可选：{', '.join(sorted(set(_CAPABILITY_ALIASES.values())))}"
            )
        raw_name = str(name or "").strip()
        lower = raw_name.lower()

        if cap == "whisper":
            if lower not in WHISPER_SIZES:
                raise DownloadError(
                    f"whisper 档位必须是：{', '.join(WHISPER_SIZES)}（收到 {name!r}）"
                )
            return cap, lower, cfg.whisper_dir / lower

        if cap == "asr":
            if lower not in ("", "sensevoice", "sensevoicesmall"):
                raise DownloadError("SenseVoice 的 name 只能是 sensevoice（或留空）")
            return cap, "SenseVoiceSmall", cfg.asr_model_dir

        if cap == "tts":
            return cap, "IndexTTS-2.5", cfg.model_dir

        if cap == "translate":
            if lower not in ("", "nllb", "nllb-200-distilled-600m"):
                raise DownloadError("翻译模型名只能是 nllb-200-distilled-600M（或留空）")
            return cap, "nllb-200-distilled-600M", _translate_dir(cfg)

        if cap == "speaker":
            return cap, "campplus", cfg.speaker_model_dir

        if cap == "separate":
            if lower not in ("", *separate_names()):
                raise DownloadError(
                    "人声分离模型名必须是：htdemucs / htdemucs_6s / htdemucs_ft"
                )
            return cap, lower or "htdemucs", cfg.demucs_repo

        if cap == "vad":
            return cap, "fsmn-vad", cfg.asr_vad_dir

        raise DownloadError(f"不支持的 capability={capability!r}")

    # ---- 启动 ---------------------------------------------------------
    def start(self, cfg: Config, capability: str, name: str) -> dict[str, Any]:
        cap, resolved_name, target = self._resolve(cfg, capability, name)
        with self._lock:
            if self._task is not None and self._task.get("status") == "running":
                raise DownloadBusy(
                    f"已有下载任务在运行：{self._task.get('capability')}/"
                    f"{self._task.get('name')}（{self._task.get('task_id')}）"
                )
            task_id = uuid.uuid4().hex[:12]
            task: dict[str, Any] = {
                "task_id": task_id,
                "capability": cap,
                "name": resolved_name,
                "target": str(target),
                "status": "running",
                "started_at": _now_iso(),
                "finished_at": None,
                "started_bytes": _dir_size_bytes(target),
                "error": None,
                "detail": "后台下载已启动",
                "source": None,
            }
            self._task = task
        thread = threading.Thread(
            target=self._run,
            args=(cfg, task),
            name=f"model-download-{task_id}",
            daemon=True,
        )
        thread.start()
        return {"ok": True, "task_id": task_id, "capability": cap, "name": resolved_name}

    # ---- 状态 ---------------------------------------------------------
    def status(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            task = self._task
            if task is None or task.get("task_id") != task_id:
                return None
            snapshot = dict(task)
        target = Path(str(snapshot.get("target") or ""))
        current = _dir_size_bytes(target)
        started = int(snapshot.get("started_bytes") or 0)
        snapshot.update(
            {
                "running": snapshot.get("status") == "running",
                "done": snapshot.get("status") == "done",
                "failed": snapshot.get("status") == "failed",
                "size_bytes": current,
                "size_mb": round(current / (1024 * 1024), 1),
                "downloaded_bytes": max(0, current - started),
                "downloaded_mb": round(max(0, current - started) / (1024 * 1024), 1),
                "percent": None,
            }
        )
        return snapshot

    def latest(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._task) if self._task else None

    # ---- 线程体 -------------------------------------------------------
    def _run(self, cfg: Config, task: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            cfg.export_env()
            source = _download_one(cfg, task["capability"], task["name"])
            missing = _missing_after(cfg, task["capability"], task["name"])
            if missing:
                raise RuntimeError(
                    "下载结束但文件仍不完整，缺少：" + "、".join(missing)
                )
            with self._lock:
                task["status"] = "done"
                task["source"] = source
                task["detail"] = "下载完成"
                task["finished_at"] = _now_iso()
                task["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        except Exception as exc:  # noqa: BLE001 - 任务失败要落进任务记录，不能炸线程
            logger.exception("模型下载失败：%s/%s", task["capability"], task["name"])
            with self._lock:
                task["status"] = "failed"
                task["error"] = f"{type(exc).__name__}: {exc}"
                task["detail"] = "下载失败"
                task["finished_at"] = _now_iso()
                task["elapsed_seconds"] = round(time.perf_counter() - started, 1)


def separate_names() -> tuple[str, ...]:
    return ("htdemucs", "htdemucs_6s", "htdemucs_ft")


def _missing_after(cfg: Config, capability: str, name: str) -> list[str]:
    """下载结束后按清单复查；返回仍缺的文件。"""
    if capability == "tts":
        return _check_files(
            cfg.model_dir, list(REQUIRED_MODEL_FILES) + list(REQUIRED_AUX_FILES)
        )[1]
    if capability == "asr":
        return _check_files(cfg.asr_model_dir, MODEL_SPECS["asr"]["required"])[1]
    if capability == "whisper":
        return _check_files(cfg.whisper_dir / name, WHISPER_REQUIRED)[1]
    if capability == "translate":
        missing = _check_files(_translate_dir(cfg), TRANSLATE_MODEL_FILES)[1]
        target = _translate_dir(cfg)
        if not any((target / item).is_file() for item in TRANSLATE_WEIGHT_FILES):
            missing.append("model.safetensors 或 pytorch_model.bin")
        return missing
    if capability == "speaker":
        return _check_files(cfg.speaker_model_dir, MODEL_SPECS["speaker"]["required"])[1]
    if capability == "vad":
        return _check_files(cfg.asr_vad_dir, VAD_REQUIRED)[1]
    if capability == "separate":
        if name == "htdemucs_ft":
            return _check_files(cfg.demucs_repo, DEMUCS_FT_FILES)[1]
        files = DEMUCS_FILES.get(name, ())
        return _check_files(cfg.demucs_repo, files)[1]
    return []


def _download_one(cfg: Config, capability: str, name: str) -> str:
    """在后台线程里执行真正的下载；返回来源说明。"""
    if capability == "whisper":
        module = import_engine_module("whisper_engine")
        info = module.ensure_weights(name)
        return str(info.get("source") or "whisper")

    if capability == "translate":
        module = import_engine_module("translate_engine")
        info = module.ensure_weights()
        return str(info.get("source") or "nllb")

    if capability == "vad":
        module = import_engine_module("asr_engine")
        getattr(module, "_ensure_vad_weights")()
        return "modelscope"

    # 其余复用 `localai models download` 的同一份代码。
    from ..cli import _download_demucs, _download_snapshot

    if capability == "separate":
        # 与 CLI 一致：一次准备 htdemucs + htdemucs_6s；htdemucs_ft 默认不拉。
        _download_demucs(cfg)
        return "demucs-official"

    key = {"tts": "tts", "asr": "asr", "speaker": "speaker"}[capability]
    _download_snapshot(key, cfg)
    return "modelscope/huggingface"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


__all__ = [
    "CAPABILITY_TITLES",
    "DownloadBusy",
    "DownloadError",
    "ModelDownloader",
    "model_catalog",
]
