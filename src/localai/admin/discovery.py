"""本地已有模型的递归探测与「配置指向」采用。

设计约束
--------
1. **只读**：扫描只读文件名、大小与少量样本字节；绝不复制、移动、删除或建链接。
2. **默认不预置路径**：搜索路径来自 ``config.json`` / 环境变量，默认空列表。
3. **配置指向**：采用模型只写 ``<LOCALAI_HOME>/config.json``，让各引擎改用用户
   已有的目录。
4. **防误判**：文件名签名 + 已知大小区间 + 首尾 8MB 抽样哈希；任何一项不够就
   降级为 ``partial``，但仍可展示/采用。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config, config_file_path, load_config_file
from ..constants import WHISPER_SIZES
from .settings import SettingsError, _atomic_write, update_settings

logger = logging.getLogger("localai.admin.discovery")

#: 规则版本；签名规则一变，旧缓存自动失效。
RULES_VERSION = "2026-11-08.1"
#: 目录签名缓存时间（秒）。
CACHE_TTL_SEC = 600
#: 抽样哈希读取的字节数（首 8MB + 尾 8MB）。
SAMPLE_BYTES = 8 * 1024 * 1024
#: 搜索路径上限。
MAX_SEARCH_PATHS = 20
#: 递归深度上限。
MAX_DEPTH = 12

#: 扫描时跳过的目录名（大小写不敏感）；以 ``.`` 开头的隐藏目录也会跳过。
NOISE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "site-packages",
    "dist-info",
    "egg-info",
    "$recycle.bin",
    "system volume information",
    "lost+found",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

#: 已知模型的主文件大小区间（MiB）。只用于置信度，不用于拒绝。
SIZE_RANGES_MB: dict[tuple[str, str], tuple[float, float]] = {
    ("tts", "gpt.pth"): (2900.0, 3220.0),
    ("asr", "model.pt"): (850.0, 1000.0),
    ("speaker", "campplus_cn_common.bin"): (20.0, 35.0),
    ("vad", "model.pt"): (1.0, 10.0),
    ("translate", "pytorch_model.bin"): (2200.0, 2500.0),
    ("translate", "model.safetensors"): (2200.0, 2500.0),
}

WHISPER_SIZE_RANGES_MB: dict[str, tuple[float, float]] = {
    "tiny": (65.0, 85.0),
    "base": (120.0, 165.0),
    "small": (450.0, 500.0),
    "medium": (1400.0, 1580.0),
    "large-v3": (2800.0, 3400.0),
}

#: 已知样本的「首 8MB + 尾 8MB」SHA256（从本机真实权重预算，不联网）。
#: 哈希不匹配时只降置信度，不阻止采用（可能是合法的另一份转换/量化）。
KNOWN_SAMPLE_HASHES: dict[tuple[str, str], str] = {
    ("tts", "gpt.pth"): "c8799d415f5fe5db873ce016b2a57a2485e4296e60c78afdf888d8846c7cbe3d",
    ("asr", "model.pt"): "18f30a62189dcecf5e6bdb2962e8cabaca10b0c39a934e316fd36676da456298",
    ("speaker", "campplus_cn_common.bin"): "be02dcd8b4704cac6efc98cd00911d85b8361ed2f0b93354e76f543a7d7a1080",
    ("vad", "model.pt"): "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5",
    ("translate", "pytorch_model.bin"): "bb1f6a9f6f96d805f61e2829e7ad259f38e4411dc89e5b6e5023ca7810a19cc4",
    ("vision", "qwen2.5-vl-3b-instruct-q4_k_m.gguf"): "90c120dafc4dcef12dadc5c9acfb801d425314408b05f8e4ab1dae271ca4ccb0",
    ("vision", "mmproj-f16.gguf"): "5b8c96fcf15c8076020c0f4fe6f0c5c7b590354a5ead642466ede444dfed43b0",
}
KNOWN_WHISPER_HASHES: dict[str, str] = {
    "tiny": "7ee2e49bc06cc1b9625c85855bd6f3ebccde018b321b4134ba0112d0cfdb992d",
    "base": "f0e7561d619bb227a47953bb07a7052d44d6d8fc8c2a23cb5e20d56795f489bb",
    "small": "7e4e3315b729119758461589705dbe4c174637ce975a5aa3d61fb8bb8c036287",
    "medium": "d92b838baa15d679928598688dcf6f04f50ecb3370dbaf8ebd30eb3e5ebb17f6",
}

#: Demucs 已知权重：文件 -> (模型名, 期望大小 MB 下限, 上限)。
DEMUCS_FILES: dict[str, tuple[str, float, float]] = {
    "955717e8-8726e21a.th": ("htdemucs", 60.0, 100.0),
    "5c90dfd2-34c22ccb.th": ("htdemucs_6s", 40.0, 70.0),
    "f7e0c4bc-ba3fe64a.th": ("htdemucs_ft", 50.0, 120.0),
    "d12395a8-e57c48e6.th": ("htdemucs_ft", 50.0, 120.0),
    "92cfc3b6-ef3bcb9c.th": ("htdemucs_ft", 50.0, 120.0),
    "04573f0d-f3cf25b2.th": ("htdemucs_ft", 50.0, 120.0),
}
DEMUCS_YAML = {
    "htdemucs": "htdemucs.yaml",
    "htdemucs_6s": "htdemucs_6s.yaml",
    "htdemucs_ft": "htdemucs_ft.yaml",
}
DEMUCS_MODEL_FILES: dict[str, list[str]] = {
    model: [name for name, (owner, _lo, _hi) in DEMUCS_FILES.items() if owner == model]
    for model in DEMUCS_YAML
}
KNOWN_DEMUCS_HASHES = {
    "5c90dfd2-34c22ccb.th": "4e6c0a8a817f861558cca81a2be9d96f80c47c8df7a800f1085e065214d26489",
    "955717e8-8726e21a.th": "5e3574c94151752d85c91ba2b71b047e45877b884c925ff850dda081e726caa4",
}

SENSEVOICE_REQUIRED = (
    "model.pt",
    "tokens.json",
    "am.mvn",
    "config.yaml",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)
WHISPER_REQUIRED = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
NLLB_REQUIRED = (
    "config.json",
    "tokenizer.json",
    "sentencepiece.bpe.model",
    "model.safetensors / pytorch_model.bin",
)
TTS_REQUIRED = ("gpt.pth", "s2mel.pth", "codec.pth")
VISION_MMPROJ = "mmproj*.gguf"

#: 能力中文名。
CAPABILITY_TITLES = {
    "tts": "语音合成",
    "asr": "语音识别",
    "whisper": "faster-whisper",
    "translate": "翻译",
    "speaker": "声纹",
    "separate": "人声分离",
    "vad": "语音端点检测",
    "vision": "视觉理解",
}

#: 采用时写入的配置键。
ADOPT_CONFIG_KEYS = {
    "tts": "tts_model_dir",
    "asr": "asr_model_dir",
    "whisper": "whisper_model_dirs",
    "translate": "translate_model_dir",
    "speaker": "speaker_model_dir",
    "separate": "demucs_repo",
    "vision": "vision_model_dir",
    "vad": "vad_model_dir",
}

#: capability 别名（与下载接口保持一致）。
CAPABILITY_ALIASES = {
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
    "vision": "vision",
    "vlm": "vision",
    "qwen2.5-vl": "vision",
    "qwen2.5-vl-3b": "vision",
}


class DiscoveryError(ValueError):
    """路径 / 模型校验失败（路由转 400）。"""


class ScanBusy(RuntimeError):
    """已有扫描任务在运行（路由转 409）。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def normalize_capability(raw: str) -> str:
    key = str(raw or "").strip().lower().replace("_", "-")
    return CAPABILITY_ALIASES.get(key, "")


def _norm_key(path: Path | str) -> str:
    try:
        text = str(Path(path).expanduser().resolve(strict=False))
    except (OSError, ValueError):
        text = str(path)
    return os.path.normcase(text)


def _is_reparse_point(entry: os.DirEntry[str]) -> bool:
    """Windows junction / 符号链接都当重解析点跳过，避免环路与跨盘意外。"""
    try:
        if entry.is_symlink():
            return True
        stat = entry.stat(follow_symlinks=False)
        attrs = getattr(stat, "st_file_attributes", 0)
        return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except OSError:
        return True


def _readable_dir(path: Path) -> None:
    """确认目录存在且至少能列出一项；失败抛中文 :class:`DiscoveryError`。"""
    if not path.exists():
        raise DiscoveryError(f"路径不存在：{path}")
    if not path.is_dir():
        raise DiscoveryError(f"不是目录：{path}")
    try:
        with os.scandir(path) as iterator:
            next(iterator, None)
    except OSError as exc:
        raise DiscoveryError(f"目录不可读：{path}（{exc}）") from exc


def normalize_search_paths(paths: list[str] | tuple[str, ...]) -> list[str]:
    """校验、去重、规范化为绝对路径；最多 20 个。"""
    if not isinstance(paths, (list, tuple)):
        raise DiscoveryError("paths 必须是字符串列表")
    if len(paths) > MAX_SEARCH_PATHS:
        raise DiscoveryError(f"最多只能配置 {MAX_SEARCH_PATHS} 个搜索路径")
    result: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        text = str(raw or "").strip().strip('"')
        if not text:
            continue
        path = Path(text).expanduser()
        _readable_dir(path)
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        result.append(str(resolved))
    if len(result) > MAX_SEARCH_PATHS:
        raise DiscoveryError(f"最多只能配置 {MAX_SEARCH_PATHS} 个搜索路径")
    return result


def search_path_snapshot(cfg: Config) -> dict[str, Any]:
    """GET /admin/models/search-paths 的响应。"""
    paths = list(cfg.model_search_paths or [])
    source = "env" if cfg.env_overrides.get("model_search_paths") else "config"
    return {"ok": True, "paths": paths, "count": len(paths), "source": source}


def update_search_paths(
    cfg: Config,
    *,
    paths: Optional[list[str]] = None,
    add: Optional[str | list[str]] = None,
    remove: Optional[str | list[str]] = None,
) -> dict[str, Any]:
    """整体替换或按 add/remove 修改搜索路径，并写入 config.json。"""
    if paths is not None and (add is not None or remove is not None):
        raise DiscoveryError("paths 与 add/remove 只能二选一")

    def _as_list(value: Optional[str | list[str]]) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        raise DiscoveryError("add/remove 必须是字符串或字符串列表")

    current = list(cfg.model_search_paths or [])
    if paths is not None:
        if not isinstance(paths, list):
            raise DiscoveryError("paths 必须是字符串列表")
        candidate = list(paths)
    elif add is not None:
        candidate = current + _as_list(add)
    elif remove is not None:
        removing = {_norm_key(item) for item in _as_list(remove)}
        kept = [item for item in current if _norm_key(item) not in removing]
        if len(kept) == len(current):
            raise DiscoveryError("要删除的路径不在当前列表里")
        candidate = kept
    else:
        raise DiscoveryError("请提供 paths，或 add/remove 之一")

    normalized = normalize_search_paths(candidate)
    try:
        result = update_settings(cfg, {"model_search_paths": normalized})
    except SettingsError as exc:
        raise DiscoveryError(str(exc)) from exc

    response = search_path_snapshot(cfg)
    response["warnings"] = list(result.get("warnings") or [])
    return response


def clear_cache() -> None:
    """清空目录签名缓存（测试/规则热更新用）。"""
    with _CACHE_LOCK:
        _MATCH_CACHE.clear()


# ---------------------------------------------------------------------------
# 文件与签名
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_MATCH_CACHE: dict[tuple[str, int, str, bool], tuple[float, list[dict[str, Any]]]] = {}


def _file_map(directory: Path) -> dict[str, int]:
    """当前目录的文件名（小写）-> 字节数；隐藏文件也保留。"""
    result: dict[str, int] = {}
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                try:
                    if entry.is_file(follow_symlinks=False):
                        result[entry.name.lower()] = int(entry.stat().st_size)
                except OSError:
                    continue
    except OSError:
        return result
    return result


def _has(files: dict[str, int], name: str) -> bool:
    return files.get(name.lower(), 0) > 0


def _missing(files: dict[str, int], required: list[str] | tuple[str, ...]) -> list[str]:
    return [name for name in required if not _has(files, name)]


def _found(files: dict[str, int], names: list[str] | tuple[str, ...]) -> list[str]:
    return [name for name in names if _has(files, name)]


def _dir_size_mb(directory: Path) -> float:
    total = 0
    try:
        for root, _dirs, names in os.walk(directory):
            for name in names:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return round(total / (1024 * 1024), 1)


def _files_size_mb(files: dict[str, int], names: list[str] | tuple[str, ...]) -> float:
    total = sum(files.get(name.lower(), 0) for name in names)
    return round(total / (1024 * 1024), 1)


def sample_sha256(path: Path) -> str:
    """文件首 8MB + 尾 8MB 的 SHA256（小文件就是全文）。"""
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = handle.read(SAMPLE_BYTES)
        digest.update(first)
        if size > SAMPLE_BYTES:
            handle.seek(max(SAMPLE_BYTES, size - SAMPLE_BYTES))
            digest.update(handle.read(SAMPLE_BYTES))
    return digest.hexdigest()


def _infer_whisper_size(directory_name: str, main_size_mb: float) -> str:
    name = directory_name.lower().replace("_", "-")
    for size in ("large-v3", "medium", "small", "base", "tiny"):
        if re.search(rf"(?<![a-z0-9]){re.escape(size)}(?![a-z0-9])", name):
            return size
    for size, (low, high) in WHISPER_SIZE_RANGES_MB.items():
        if low <= main_size_mb <= high:
            return size
    return ""


def _whisper_size_of(cap: str, model: str, files: dict[str, int]) -> tuple[str, Optional[tuple[float, float]]]:
    if cap != "whisper":
        return model, None
    size = model if model in WHISPER_SIZES else ""
    return size, WHISPER_SIZE_RANGES_MB.get(size)


def _size_check(cap: str, model: str, files: dict[str, int], directory: Path) -> tuple[str, Optional[tuple[float, float]]]:
    """返回 (主文件名, 已知大小区间)；没有主文件时返回空。"""
    if cap == "tts":
        return "gpt.pth", SIZE_RANGES_MB[("tts", "gpt.pth")]
    if cap == "asr":
        return "model.pt", SIZE_RANGES_MB[("asr", "model.pt")]
    if cap == "whisper":
        _size, rng = _whisper_size_of(cap, model, files)
        return "model.bin", rng
    if cap == "speaker":
        return "campplus_cn_common.bin", SIZE_RANGES_MB[("speaker", "campplus_cn_common.bin")]
    if cap == "vad":
        return "model.pt", SIZE_RANGES_MB[("vad", "model.pt")]
    if cap == "translate":
        for name in ("pytorch_model.bin", "model.safetensors"):
            if _has(files, name):
                return name, SIZE_RANGES_MB[("translate", name)]
        return "pytorch_model.bin", None
    if cap == "separate":
        for name, (owner, low, high) in DEMUCS_FILES.items():
            if owner != model or not _has(files, name):
                continue
            return name, (low, high)
        return "", None
    if cap == "vision":
        main = _vision_main(files)
        if main:
            return main, (1500.0, 4000.0)
        return "", None
    return "", None


def _hash_for(cap: str, model: str, main_name: str) -> Optional[str]:
    if not main_name:
        return None
    if cap == "whisper":
        return KNOWN_WHISPER_HASHES.get(model)
    if cap == "separate":
        return KNOWN_DEMUCS_HASHES.get(main_name)
    if cap == "vision":
        return KNOWN_SAMPLE_HASHES.get(("vision", main_name.lower()))
    return KNOWN_SAMPLE_HASHES.get((cap, main_name.lower()))


def _make_hit(
    *,
    cap: str,
    model: str,
    directory: Path,
    files: dict[str, int],
    required: list[str] | tuple[str, ...],
    optional: list[str] | tuple[str, ...],
    version_hint: str = "",
    verify_sizes: bool = True,
    matched_by: str = "filename",
    extra_missing: Optional[list[str]] = None,
) -> dict[str, Any]:
    """按签名、大小区间与抽样哈希构造一个候选；置信度只降不拒。"""
    missing = _missing(files, required)
    if extra_missing:
        missing.extend(extra_missing)
    files_found = _found(files, list(required) + list(optional))
    if not files_found:
        return {}

    main_name, size_range = _size_check(cap, model, files, directory)
    size_in_range: Optional[bool] = None
    hash_verified: Optional[bool] = None
    if verify_sizes and main_name and _has(files, main_name):
        main_size = files[main_name.lower()] / (1024 * 1024)
        if size_range is not None:
            size_in_range = size_range[0] <= main_size <= size_range[1]
        known_hash = _hash_for(cap, model, main_name)
        if known_hash:
            try:
                hash_verified = sample_sha256(directory / main_name) == known_hash
            except OSError:
                hash_verified = None

    confidence = "exact"
    detail = ""
    if missing:
        confidence = "partial"
        detail = "缺少签名文件：" + "、".join(missing)
    elif verify_sizes and size_in_range is False:
        confidence = "partial"
        detail = (
            f"{main_name} 大小 {files.get(main_name.lower(), 0) / (1024 * 1024):.1f}MB "
            f"不在已知区间 {size_range[0]:.0f}–{size_range[1]:.0f}MB"
        )
    elif verify_sizes and hash_verified is False:
        confidence = "partial"
        detail = "抽样哈希与已知样本不一致（可能是不同转换/量化）"

    if verify_sizes and size_in_range is not None:
        matched_by = "filename+size"

    item: dict[str, Any] = {
        "capability": cap,
        "capability_title": CAPABILITY_TITLES.get(cap, cap),
        "model": model,
        "path": str(directory),
        "size_mb": (
            _files_size_mb(files, list(required) + list(optional))
            if cap == "separate"
            else _dir_size_mb(directory)
        ),
        "files_found": files_found,
        "missing": missing,
        "confidence": confidence,
        "version_hint": version_hint,
        "matched_by": matched_by,
        "hash_verified": hash_verified,
        "detail": detail,
    }
    return item


def _vision_main(files: dict[str, int]) -> str:
    ggufs = [
        name
        for name, size in files.items()
        if name.endswith(".gguf") and not name.startswith("mmproj") and size > 0
    ]
    if not ggufs:
        return ""
    qwen = [name for name in ggufs if "qwen" in name and "vl" in name]
    return max(qwen or ggufs, key=lambda name: files[name])


def _vision_has_mmproj(files: dict[str, int]) -> bool:
    return bool(_vision_mmproj_name(files))


def _vision_mmproj_name(files: dict[str, int]) -> str:
    for name, size in files.items():
        if name.startswith("mmproj") and name.endswith(".gguf") and size > 0:
            return name
    return ""


def _skip_matching(directory: Path) -> bool:
    """辅助模型缓存/子模块不是独立可采用的模型，避免同签名重复命中。"""
    parts = {part.lower() for part in directory.parts}
    return bool(parts & {"hf_cache", "qwen0.6bemo4-merge", "w2v-bert-2.0"})


def match_directory(
    directory: Path,
    files: Optional[dict[str, int]] = None,
    *,
    verify_sizes: bool = True,
) -> list[dict[str, Any]]:
    """识别一个目录里可能存在的模型（同一层文件集合）。"""
    files = files if files is not None else _file_map(directory)
    if not files:
        return []
    hits: list[dict[str, Any]] = []
    lname = directory.name.lower()

    # ---- IndexTTS-2.5 ------------------------------------------------
    tts_present = sum(1 for name in TTS_REQUIRED if _has(files, name))
    if tts_present >= 2 or (tts_present >= 1 and "indextts" in lname):
        hit = _make_hit(
            cap="tts",
            model="IndexTTS-2.5",
            directory=directory,
            files=files,
            required=TTS_REQUIRED,
            optional=("config.yaml", "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt"),
            version_hint="2.5",
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    # ---- SenseVoiceSmall --------------------------------------------
    sv_present = sum(1 for name in SENSEVOICE_REQUIRED if _has(files, name))
    is_vad_dir = "fsmn" in lname or lname.strip() == "vad" or lname.endswith("-vad")
    if not is_vad_dir and (sv_present >= 2 or ("sensevoice" in lname and sv_present >= 1)):
        hit = _make_hit(
            cap="asr",
            model="SenseVoiceSmall",
            directory=directory,
            files=files,
            required=SENSEVOICE_REQUIRED,
            optional=("configuration.json",),
            version_hint="SenseVoiceSmall",
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    # ---- faster-whisper ---------------------------------------------
    wh_present = sum(1 for name in WHISPER_REQUIRED if _has(files, name))
    main_size = files.get("model.bin", 0) / (1024 * 1024)
    size = _infer_whisper_size(directory.name, main_size)
    if _has(files, "model.bin") or ("whisper" in lname and wh_present >= 2):
        hit = _make_hit(
            cap="whisper",
            model=size or "unknown",
            directory=directory,
            files=files,
            required=WHISPER_REQUIRED,
            optional=("configuration.json",),
            version_hint=size,
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    # ---- NLLB --------------------------------------------------------
    nllb_weight = _has(files, "model.safetensors") or _has(files, "pytorch_model.bin")
    nllb_signal = (
        _has(files, "sentencepiece.bpe.model")
        and _has(files, "tokenizer.json")
        and _has(files, "config.json")
    )
    if nllb_signal or (nllb_weight and _has(files, "config.json")):
        required = (
            "config.json",
            "tokenizer.json",
            "sentencepiece.bpe.model",
            "model.safetensors" if _has(files, "model.safetensors") else "pytorch_model.bin",
        )
        hit = _make_hit(
            cap="translate",
            model="nllb-200-distilled-600M",
            directory=directory,
            files=files,
            required=required,
            optional=("generation_config.json", "tokenizer_config.json"),
            version_hint="600M",
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    # ---- campplus ----------------------------------------------------
    if _has(files, "campplus_cn_common.bin") or "campplus" in lname:
        hit = _make_hit(
            cap="speaker",
            model="campplus",
            directory=directory,
            files=files,
            required=("campplus_cn_common.bin", "config.yaml"),
            optional=("configuration.json",),
            version_hint="16k-common",
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    # ---- Demucs ------------------------------------------------------
    present_demucs = {
        name: meta for name, meta in DEMUCS_FILES.items() if _has(files, name)
    }
    if present_demucs:
        for model, names in DEMUCS_MODEL_FILES.items():
            model_present = [name for name in names if _has(files, name)]
            if not model_present:
                continue
            required = (DEMUCS_YAML[model], *names)
            hit = _make_hit(
                cap="separate",
                model=model,
                directory=directory,
                files=files,
                required=required,
                optional=(),
                version_hint="htdemucs",
                verify_sizes=verify_sizes,
            )
            if hit:
                hits.append(hit)

    # ---- fsmn-vad ----------------------------------------------------
    if "fsmn" in lname or ("vad" in lname and _has(files, "model.pt")):
        hit = _make_hit(
            cap="vad",
            model="fsmn-vad",
            directory=directory,
            files=files,
            required=("model.pt", "config.yaml"),
            optional=("am.mvn", "configuration.json"),
            version_hint="fsmn-vad",
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    # ---- 视觉 GGUF ---------------------------------------------------
    main_gguf = _vision_main(files)
    has_mmproj = _vision_has_mmproj(files)
    main_mb = files.get(main_gguf, 0) / (1024 * 1024) if main_gguf else 0.0
    looks_qwen = bool(main_gguf and "qwen" in main_gguf and "vl" in main_gguf)
    if looks_qwen or (has_mmproj and main_mb > 1500):
        mmproj = _vision_mmproj_name(files)
        required = [name for name in (main_gguf, mmproj) if name]
        if not mmproj:
            required.append(VISION_MMPROJ)
        hit = _make_hit(
            cap="vision",
            model=("Qwen2.5-VL-3B" if looks_qwen else (Path(main_gguf).stem if main_gguf else "Qwen2.5-VL")),
            directory=directory,
            files=files,
            required=required,
            optional=("llama-server.exe",),
            version_hint=main_gguf,
            verify_sizes=verify_sizes,
        )
        if hit:
            hits.append(hit)

    return hits


def nllb_like(files: dict[str, int]) -> bool:
    """辅助判断：有 NLLB 特征时不要把同目录误认成 SenseVoice。"""
    return _has(files, "sentencepiece.bpe.model") and _has(files, "tokenizer.json")


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
def _cached_matches(
    directory: Path,
    files: dict[str, int],
    mtime_ns: int,
    verify_sizes: bool,
) -> list[dict[str, Any]]:
    key = (
        os.path.normcase(str(directory)),
        int(mtime_ns),
        RULES_VERSION,
        bool(verify_sizes),
    )
    now = time.time()
    with _CACHE_LOCK:
        entry = _MATCH_CACHE.get(key)
        if entry is not None and now - entry[0] <= CACHE_TTL_SEC:
            return [dict(item) for item in entry[1]]
    matches = match_directory(directory, files, verify_sizes=verify_sizes)
    with _CACHE_LOCK:
        # 顺手清掉过期项，避免长时间运行的进程无限增长。
        if len(_MATCH_CACHE) > 4096:
            for old_key, (created, _value) in list(_MATCH_CACHE.items()):
                if now - created > CACHE_TTL_SEC:
                    _MATCH_CACHE.pop(old_key, None)
        _MATCH_CACHE[key] = (now, [dict(item) for item in matches])
    return matches


def scan(
    paths: list[str] | tuple[str, ...],
    *,
    max_depth: int = 6,
    max_files: int = 200000,
    timeout_sec: float = 300,
    verify_sizes: bool = True,
    progress: Optional[Callable[[dict[str, int]], None]] = None,
) -> dict[str, Any]:
    """递归扫描目录并返回命中的模型签名。

    只读；遇到符号链接/junction 不进入，并用 realpath 集合兜底防环。
    超过 ``max_files`` / ``timeout_sec`` 时停止并置 ``truncated``。
    """
    started = time.perf_counter()
    max_depth = max(0, min(MAX_DEPTH, int(max_depth)))
    deadline = started + max(0.1, float(timeout_sec))
    matches: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_real: set[str] = set()
    visited_dirs = 0
    visited_files = 0
    truncated = False
    stop = False

    def report() -> None:
        if progress is not None:
            try:
                progress(
                    {
                        "visited_dirs": visited_dirs,
                        "visited_files": visited_files,
                        "matches": len(matches),
                    }
                )
            except Exception:  # noqa: BLE001 - 进度回调不能影响扫描
                pass

    roots = [Path(str(path)).expanduser() for path in paths]
    for root in roots:
        if stop:
            break
        try:
            _readable_dir(root)
        except DiscoveryError as exc:
            errors.append({"path": str(root), "error": str(exc)})
            continue
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            if stop:
                break
            directory, depth = stack.pop()
            if time.perf_counter() > deadline:
                truncated = True
                errors.append(
                    {
                        "path": str(directory),
                        "error": f"扫描超过 {timeout_sec:.0f} 秒，已提前结束",
                    }
                )
                stop = True
                break
            if visited_files >= max_files:
                truncated = True
                errors.append(
                    {
                        "path": str(directory),
                        "error": f"文件数超过上限 {max_files}，已提前结束",
                    }
                )
                stop = True
                break

            try:
                real = os.path.normcase(os.path.realpath(directory))
            except OSError:
                real = os.path.normcase(str(directory))
            if real in seen_real:
                continue
            seen_real.add(real)

            try:
                with os.scandir(directory) as iterator:
                    entries = list(iterator)
            except OSError as exc:
                if len(errors) < 100:
                    errors.append({"path": str(directory), "error": f"无法读取目录：{exc}"})
                continue

            visited_dirs += 1
            files: dict[str, int] = {}
            child_dirs: list[Path] = []
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        visited_files += 1
                        files[entry.name.lower()] = int(entry.stat().st_size)
                    elif entry.is_dir(follow_symlinks=False):
                        if _is_reparse_point(entry):
                            continue
                        child_dirs.append(Path(entry.path))
                except OSError:
                    continue

            try:
                mtime_ns = directory.stat().st_mtime_ns
            except OSError:
                mtime_ns = 0
            try:
                found = (
                    []
                    if _skip_matching(directory)
                    else _cached_matches(directory, files, mtime_ns, verify_sizes)
                )
            except Exception as exc:  # noqa: BLE001 - 单目录识别失败不终止整轮
                logger.exception("识别目录失败：%s", directory)
                if len(errors) < 100:
                    errors.append(
                        {"path": str(directory), "error": f"识别失败：{type(exc).__name__}: {exc}"}
                    )
                found = []
            if found:
                matches.extend(found)
            report()

            if depth < max_depth:
                for child in child_dirs:
                    name = child.name.lower()
                    if name.startswith(".") or name in NOISE_DIRS:
                        continue
                    stack.append((child, depth + 1))
            if time.perf_counter() > deadline:
                truncated = True
                errors.append(
                    {
                        "path": str(directory),
                        "error": f"扫描超过 {timeout_sec:.0f} 秒，已提前结束",
                    }
                )
                stop = True

    # 同一模型可能被不同搜索根重复扫到；按能力/路径/模型去重。
    deduped: list[dict[str, Any]] = []
    seen_hits: set[tuple[str, str, str]] = set()
    for item in matches:
        key = (item["capability"], os.path.normcase(item["path"]), item["model"])
        if key in seen_hits:
            continue
        seen_hits.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: (item["capability"], item["path"].lower(), item["model"]))
    report()
    elapsed = round(time.perf_counter() - started, 2)
    return {
        "matches": deduped,
        "errors": errors,
        "stats": {
            "roots": [str(path) for path in roots],
            "visited_dirs": visited_dirs,
            "visited_files": visited_files,
            "matches": len(deduped),
            "max_depth": max_depth,
            "max_files": max_files,
            "timeout_sec": timeout_sec,
            "elapsed_seconds": elapsed,
            "truncated": truncated,
            "cached_rule_version": RULES_VERSION,
        },
    }


# ---------------------------------------------------------------------------
# 后台扫描任务
# ---------------------------------------------------------------------------
class ModelScanner:
    """单例扫描管理器；挂在 ``app.state.model_scanner``。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._task: Optional[dict[str, Any]] = None

    def start(
        self,
        cfg: Config,
        paths: Optional[list[str]] = None,
        *,
        max_depth: int = 6,
    ) -> dict[str, Any]:
        raw_paths = list(paths if paths is not None else (cfg.model_search_paths or []))
        if not raw_paths:
            raise DiscoveryError(
                "尚未配置模型搜索路径：请先在「模型」页添加目录，"
                "或传 paths 后再扫描"
            )
        normalized = normalize_search_paths(raw_paths)
        if not normalized:
            raise DiscoveryError("没有可扫描的目录")
        with self._lock:
            if self._task is not None and self._task.get("status") == "running":
                raise ScanBusy(
                    f"已有扫描任务在运行：{self._task.get('task_id')}；请等它结束"
                )
            task_id = uuid4_hex()
            task: dict[str, Any] = {
                "task_id": task_id,
                "status": "running",
                "started_at": _now_iso(),
                "finished_at": None,
                "paths": normalized,
                "max_depth": max(0, min(MAX_DEPTH, int(max_depth))),
                "progress": {
                    "visited_dirs": 0,
                    "visited_files": 0,
                    "matches": 0,
                },
                "result": None,
                "error": None,
            }
            self._task = task
        thread = threading.Thread(
            target=self._run,
            args=(task,),
            name=f"model-scan-{task_id}",
            daemon=True,
        )
        thread.start()
        return {
            "ok": True,
            "task_id": task_id,
            "paths": normalized,
            "max_depth": task["max_depth"],
        }

    def status(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            task = self._task
            if task is None or task.get("task_id") != task_id:
                return None
            snapshot = dict(task)
            snapshot["progress"] = dict(task.get("progress") or {})
            snapshot["result"] = task.get("result")
        snapshot["ok"] = True
        return snapshot

    def latest(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._task) if self._task else None

    def _run(self, task: dict[str, Any]) -> None:
        started = time.perf_counter()

        def on_progress(progress: dict[str, int]) -> None:
            with self._lock:
                task["progress"] = dict(progress)

        try:
            result = scan(
                task["paths"],
                max_depth=task["max_depth"],
                verify_sizes=True,
                progress=on_progress,
            )
            with self._lock:
                task["status"] = "done"
                task["result"] = result
                task["progress"] = {
                    "visited_dirs": result["stats"]["visited_dirs"],
                    "visited_files": result["stats"]["visited_files"],
                    "matches": result["stats"]["matches"],
                }
                task["finished_at"] = _now_iso()
                task["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        except Exception as exc:  # noqa: BLE001 - 任务失败要落状态，不能炸线程
            logger.exception("模型扫描失败")
            with self._lock:
                task["status"] = "failed"
                task["error"] = f"{type(exc).__name__}: {exc}"
                task["finished_at"] = _now_iso()
                task["elapsed_seconds"] = round(time.perf_counter() - started, 2)


def uuid4_hex() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 采用（只写 config.json）
# ---------------------------------------------------------------------------
def _set_config_key(cfg: Config, key: str, value: Any) -> None:
    """写入白名单外的路径键，保留 config.json 里不认识的键。"""
    file_path = config_file_path(cfg.home)
    data = load_config_file(cfg.home)
    if value in (None, "", {}, []):
        data.pop(key, None)
    else:
        data[key] = value
    try:
        _atomic_write(file_path, data)
    except OSError as exc:
        raise DiscoveryError(f"写入配置文件失败：{file_path}（{exc}）") from exc


def _sync_native_paths(cfg: Config, capability: str) -> None:
    """原生引擎模块已 import 时同步其路径常量；未 import 的靠环境变量。"""
    try:
        if capability == "tts" and "engine" in sys.modules:
            sys.modules["engine"].MODEL_DIR = cfg.model_dir
        elif capability == "asr" and "asr_engine" in sys.modules:
            sys.modules["asr_engine"].MODEL_DIR = cfg.asr_model_dir
        elif capability == "vad" and "asr_engine" in sys.modules:
            sys.modules["asr_engine"].VAD_MODEL_DIR = cfg.asr_vad_dir
        elif capability == "speaker" and "speaker_engine" in sys.modules:
            sys.modules["speaker_engine"].CAMPPUS_DIR = cfg.speaker_model_dir
        elif capability == "translate" and "translate_engine" in sys.modules:
            module = sys.modules["translate_engine"]
            module.MODEL_DIR = cfg.translate_weight_dir
            module.MODELS_ROOT = cfg.translate_dir
        elif capability == "separate" and "separator_engine" in sys.modules:
            sys.modules["separator_engine"].WEIGHTS_DIR = cfg.demucs_repo
        elif capability == "whisper" and "whisper_engine" in sys.modules:
            sys.modules["whisper_engine"].SIZE_DIRS = {
                size: Path(path)
                for size, path in (cfg.whisper_model_dirs or {}).items()
            }
        # vision_engine 每次调用 vision_dir() 都读环境变量，无需改模块常量。
    except Exception:  # noqa: BLE001 - 热同步失败由「下次加载/重启」兜底
        logger.exception("同步 %s 路径到原生引擎失败", capability)


def _adopt_hit(cfg: Config, capability: str, target: Path, name: str) -> dict[str, Any]:
    try:
        files = _file_map(target)
    except OSError as exc:
        raise DiscoveryError(f"无法读取目录：{target}（{exc}）") from exc
    hits = match_directory(target, files, verify_sizes=False)
    candidates = [item for item in hits if item["capability"] == capability]
    if not candidates:
        raise DiscoveryError(
            f"{target} 里没有找到 {CAPABILITY_TITLES.get(capability, capability)} 的签名文件；"
            "请确认目录指向的是模型根目录（不是上一级）"
        )
    hit = candidates[0]

    if capability == "whisper":
        requested = str(name or "").strip().lower()
        if requested and requested not in WHISPER_SIZES:
            raise DiscoveryError(
                "whisper 档位必须是：" + " / ".join(WHISPER_SIZES)
            )
        size = requested or hit.get("model") or ""
        if size == "unknown":
            size = ""
        if not size:
            main_size = files.get("model.bin", 0) / (1024 * 1024)
            size = _infer_whisper_size(target.name, main_size)
        if size not in WHISPER_SIZES:
            raise DiscoveryError(
                "无法判断 whisper 档位：请在请求里传 name（tiny/base/small/medium/large-v3）"
            )
        if hit.get("missing"):
            raise DiscoveryError(
                f"{target} 的 whisper 签名不完整，缺少：" + "、".join(hit["missing"])
            )
    elif hit.get("missing"):
        raise DiscoveryError(
            f"{target} 的 {CAPABILITY_TITLES.get(capability, capability)} 签名不完整，缺少："
            + "、".join(hit["missing"])
        )

    key = ADOPT_CONFIG_KEYS[capability]
    env_key = key
    env_overridden = bool(cfg.env_overrides.get(env_key))
    if capability == "whisper":
        saved = load_config_file(cfg.home).get("whisper_model_dirs")
        base = dict(saved) if isinstance(saved, dict) else {}
        value = {**base, size: str(target)}
    else:
        value = str(target)

    # 只写 config.json；若该路径由环境变量覆盖，则不动当前进程的 Config。
    _set_config_key(cfg, key, value)
    if not env_overridden:
        if capability == "tts":
            cfg.model_dir = target
        elif capability == "asr":
            cfg.asr_model_dir = target
        elif capability == "whisper":
            cfg.whisper_model_dirs = dict(value)
        elif capability == "translate":
            cfg.translate_model_dir = target
        elif capability == "speaker":
            cfg.speaker_model_dir = target
        elif capability == "separate":
            cfg.demucs_repo = target
        elif capability == "vision":
            cfg.vision_dir = target
        elif capability == "vad":
            cfg.asr_vad_dir = target
        cfg.export_env()
        _sync_native_paths(cfg, capability)

    warnings: list[str] = []
    if env_overridden:
        warnings.append(
            "对应路径已由环境变量覆盖：配置已写入，但当前进程和下次启动仍以环境变量为准。"
        )
    note = (
        f"已把 {CAPABILITY_TITLES.get(capability, capability)} 指向 {target}。"
        "下次加载使用新路径；如果该能力已有模型加载，请先卸载或等空闲卸载。"
        "未移动、复制或删除任何用户文件。"
    )
    if warnings:
        note += " " + warnings[0]

    return {
        "ok": True,
        "capability": capability,
        "model": hit.get("model"),
        "name": size if capability == "whisper" else hit.get("model"),
        "path": str(target),
        "config_key": key,
        "value": str(target) if capability == "whisper" else value,
        "config_value": value,
        "requires_restart": False,
        "warnings": warnings,
        "note": note,
    }


def adopt_model(
    cfg: Config,
    capability: str,
    path: str,
    *,
    name: str = "",
) -> dict[str, Any]:
    """校验签名后把已有模型目录写进配置；只读用户目录。"""
    cap = normalize_capability(capability)
    if not cap:
        raise DiscoveryError(
            f"不支持的 capability={capability!r}；可选："
            + "、".join(sorted(ADOPT_CONFIG_KEYS))
        )
    text = str(path or "").strip().strip('"')
    if not text:
        raise DiscoveryError("path 不能为空")
    target = Path(text).expanduser()
    _readable_dir(target)
    target = target.resolve()
    return _adopt_hit(cfg, cap, target, name)


__all__ = [
    "ADOPT_CONFIG_KEYS",
    "CACHE_TTL_SEC",
    "CAPABILITY_TITLES",
    "DEMUCS_FILES",
    "DiscoveryError",
    "ModelScanner",
    "RULES_VERSION",
    "ScanBusy",
    "adopt_model",
    "clear_cache",
    "match_directory",
    "normalize_search_paths",
    "sample_sha256",
    "scan",
    "search_path_snapshot",
    "update_search_paths",
]
