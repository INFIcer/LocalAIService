"""命令行入口：``localai serve | doctor | models | version``。

安装后可直接用 ``localai``；未安装时也可以用::

    set PYTHONPATH=E:\\LocalAIService\\src
    python -m localai doctor
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

from . import API_VERSION, __version__
from .config import DEFAULT_FFMPEG, Config, get_config
from .constants import MODEL_SPECS, REQUIRED_AUX_FILES, REQUIRED_MODEL_FILES

# Demucs 官方权重（不在 ModelScope 上，单独按 URL 下载）。
DEMUCS_BASE = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"
DEMUCS_FILES = {
    "htdemucs": ("htdemucs.yaml", "955717e8-8726e21a.th"),
    "htdemucs_6s": ("htdemucs_6s.yaml", "5c90dfd2-34c22ccb.th"),
}
# htdemucs_ft 是 4 个模型的 bag，官方每个模型一个 .th；缺任何一个都不算就绪。
DEMUCS_FT_FILES = (
    "htdemucs_ft.yaml",
    "f7e0c4bc-ba3fe64a.th",
    "d12395a8-e57c48e6.th",
    "92cfc3b6-ef3bcb9c.th",
    "04573f0d-f3cf25b2.th",
)

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"
INFO = "[INFO]"

_MODEL_ALIASES = {
    "tts": "tts",
    "indextts": "tts",
    "indextts-2.5": "tts",
    "asr": "asr",
    "sensevoice": "asr",
    "speaker": "speaker",
    "campplus": "speaker",
    "separate": "separate",
    "demucs": "separate",
    "all": "all",
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _version_of(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(module: str) -> Optional[str]:
    """尽量拿到模块版本：优先查发行包元数据（不 import，避免副作用）。"""
    version = _version_of(module)
    if version:
        return version
    if importlib.util.find_spec(module) is None:
        return None
    try:
        imported = __import__(module)
        value = getattr(imported, "__version__", None)
        if value:
            return str(value)
    except Exception:  # noqa: BLE001
        pass
    return None


def _check_files(directory: Path, names: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for name in names:
        candidate = directory / str(name).replace("\\", os.sep)
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


def _port_free(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((probe_host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _http_probe(url: str, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            body = response.read(4096).decode("utf-8", errors="replace")
        return True, body
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

def cmd_version(args: argparse.Namespace) -> int:
    cfg = get_config()
    print(f"localai-service {__version__}（HTTP 契约 API v{API_VERSION}）")
    print(f"Python       {platform.python_version()}  {sys.executable}")
    print(f"LOCALAI_HOME {cfg.home}")
    print(f"监听         {cfg.base_url}")
    print(f"引擎目录     {cfg.engine_dir}")
    return 0


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import app

    cfg = get_config()
    host = args.host or cfg.host
    port = args.port or cfg.port
    log_level = (args.log_level or cfg.log_level).lower()
    print(f"localai-service {__version__} 监听 http://{host}:{port}  (docs: /docs)")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def _doctor_dependencies() -> list[tuple[str, str, bool]]:
    """返回 (名称, 版本或状态, 是否必需/存在)。"""
    required = ["fastapi", "uvicorn", "httpx", "soundfile", "numpy", "pydantic"]
    optional = [
        "torch",
        "funasr",
        "demucs",
        "transformers",
        "modelscope",
        "faster_whisper",
        "ctranslate2",
        "langdetect",
    ]
    rows: list[tuple[str, str, bool]] = []
    for name in required:
        version = _module_version(name)
        rows.append((name, version or "未安装", version is not None))
    for name in optional:
        version = _module_version(name)
        rows.append((f"{name}（GPU 引擎）", version or "未安装", version is not None))
    return rows


def _doctor_models(cfg: Config) -> list[tuple[str, str, bool, str]]:
    rows: list[tuple[str, str, bool, str]] = []

    tts_missing = _check_files(cfg.model_dir, REQUIRED_MODEL_FILES)[1]
    aux_missing = _check_files(cfg.model_dir, REQUIRED_AUX_FILES)[1]
    tts_ok = not tts_missing and not aux_missing
    rows.append(
        (
            "IndexTTS-2.5",
            str(cfg.model_dir),
            tts_ok,
            (
                f"主权重 {len(REQUIRED_MODEL_FILES) - len(tts_missing)}/{len(REQUIRED_MODEL_FILES)}，"
                f"辅助模型 {len(REQUIRED_AUX_FILES) - len(aux_missing)}/{len(REQUIRED_AUX_FILES)}"
                + (f"；缺 {', '.join(tts_missing + aux_missing)}" if not tts_ok else "")
            ),
        )
    )

    asr_missing = _check_files(cfg.asr_model_dir, MODEL_SPECS["asr"]["required"])[1]
    rows.append(
        (
            "SenseVoiceSmall",
            str(cfg.asr_model_dir),
            not asr_missing,
            "文件齐全" if not asr_missing else f"缺 {', '.join(asr_missing)}",
        )
    )

    spk_missing = _check_files(cfg.speaker_model_dir, MODEL_SPECS["speaker"]["required"])[1]
    rows.append(
        (
            "campplus",
            str(cfg.speaker_model_dir),
            not spk_missing,
            "文件齐全" if not spk_missing else f"缺 {', '.join(spk_missing)}",
        )
    )

    # faster-whisper（可选第二识别引擎）：权重缺失不影响启动，首次请求会自动下载。
    whisper_dir = cfg.whisper_dir / cfg.whisper_model
    whisper_missing = _check_files(
        whisper_dir, ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
    )[1]
    rows.append(
        (
            f"faster-whisper-{cfg.whisper_model}",
            str(whisper_dir),
            not whisper_missing,
            (
                "权重齐全"
                if not whisper_missing
                else "可选：未下载（首次 /asr engine=whisper 时自动从 ModelScope 下载）"
            ),
        )
    )

    # 翻译（NLLB-200，可选能力）：权重缺失不影响启动，首次 /translate 会自动下载。
    translate_dir = cfg.translate_dir / "nllb-200-distilled-600M"
    translate_missing = _check_files(
        translate_dir,
        (
            "config.json",
            "sentencepiece.bpe.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
    )[1]
    if not any(
        (translate_dir / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        translate_missing.append("pytorch_model.bin")
    rows.append(
        (
            "NLLB-200 翻译",
            str(translate_dir),
            not translate_missing,
            (
                "权重齐全"
                if not translate_missing
                else "可选：未下载（首次 /translate 时自动从 ModelScope 下载）"
            ),
        )
    )

    # fsmn-vad：SenseVoice 分段转写用，缺了只影响 /asr/segments 的 sensevoice 分支。
    vad_missing = _check_files(cfg.asr_vad_dir, ("model.pt", "config.yaml"))[1]
    rows.append(
        (
            "fsmn-vad",
            str(cfg.asr_vad_dir),
            not vad_missing,
            "权重齐全" if not vad_missing else "可选：未下载（首次分段请求时自动下载）",
        )
    )

    demucs_missing: list[str] = []
    for files in DEMUCS_FILES.values():
        demucs_missing.extend(_check_files(cfg.demucs_repo, files)[1])
    rows.append(
        (
            "Demucs（htdemucs / 6s）",
            str(cfg.demucs_repo),
            not demucs_missing,
            "权重齐全" if not demucs_missing else f"缺 {', '.join(demucs_missing)}",
        )
    )

    # 视觉（VLM）：可选能力，缺失只提示；llama.cpp 二进制与 GGUF 都在 models/vision/。
    try:
        from .engines._native import import_engine_module

        vision_status = import_engine_module("vision_engine").model_status()
        vision_ok = bool(vision_status.get("ready"))
        vision_detail = (
            f"llama.cpp {vision_status.get('runtime_version')}，"
            f"权重 {vision_status.get('model_size_mb')}MB + 投影层 "
            f"{vision_status.get('mmproj_size_mb')}MB"
            if vision_ok
            else "可选：未安装（下载：POST /admin/models/download capability=vision）"
        )
    except Exception as exc:  # noqa: BLE001 - 视觉状态不影响整体体检
        vision_ok = False
        vision_detail = f"可选：状态读取失败（{type(exc).__name__}: {exc}）"
    rows.append(("Qwen2.5-VL-3B（视觉）", str(cfg.vision_dir), vision_ok, vision_detail))

    ft_missing = _check_files(cfg.demucs_repo, DEMUCS_FT_FILES)[1]
    rows.append(
        (
            "Demucs htdemucs_ft",
            str(cfg.demucs_repo),
            not ft_missing,
            "可选：未下载（缺失属正常；需要时按 demucs 官方说明手动放置）",
        )
    )
    return rows


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = get_config()
    problems = 0
    warnings = 0

    print("=" * 72)
    print(f"localai doctor  ——  localai-service {__version__}（API v{API_VERSION}）")
    print("=" * 72)

    print(f"\n{INFO} 运行环境")
    print(f"       Python      {platform.python_version()}  ({sys.executable})")
    print(f"       平台        {platform.platform()}")
    print(f"       LOCALAI_HOME {cfg.home}")
    print(f"       引擎目录     {cfg.engine_dir}")
    print(f"       监听         {cfg.base_url}")
    print(f"       TTS 后端     {cfg.tts_backend}")

    print(f"\n{INFO} 依赖")
    for name, version, present in _doctor_dependencies():
        if "GPU 引擎" in name:
            mark = OK if present else WARN
        else:
            mark = OK if present else FAIL
            if not present:
                problems += 1
        print(f"       {mark} {name:<24} {version}")

    print(f"\n{INFO} GPU")
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            output = subprocess.check_output(
                [smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                 "--format=csv,noheader"],
                timeout=10,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in output.splitlines():
                if line.strip():
                    print(f"       {OK} {line.strip()}")
        except Exception as exc:  # noqa: BLE001
            print(f"       {WARN} nvidia-smi 执行失败：{exc}")
            warnings += 1
    else:
        print(f"       {WARN} 找不到 nvidia-smi（无独显或驱动未装）")
        warnings += 1

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            print(
                f"       {OK} torch {torch.__version__}  CUDA {torch.version.cuda}  "
                f"{device}  空闲 {free / 1024**2:.0f}/{total / 1024**2:.0f} MB"
            )
        else:
            print(f"       {WARN} torch {torch.__version__} 存在，但 CUDA 不可用")
            warnings += 1
    except Exception as exc:  # noqa: BLE001
        print(f"       {WARN} 无法使用 torch：{type(exc).__name__}: {exc}")
        warnings += 1

    print(f"\n{INFO} ffmpeg")
    if cfg.ffmpeg and Path(cfg.ffmpeg).is_file():
        print(f"       {OK} ffmpeg  {cfg.ffmpeg}（来源：{cfg.ffmpeg_source}）")
    else:
        print(f"       {FAIL} 找不到 ffmpeg；期望 {DEFAULT_FFMPEG}，或设置 LOCALAI_FFMPEG")
        problems += 1
    if cfg.ffprobe and Path(cfg.ffprobe).is_file():
        print(f"       {OK} ffprobe {cfg.ffprobe}（来源：{cfg.ffprobe_source}）")
    else:
        print(f"       {WARN} 找不到 ffprobe；抽帧/探测会受影响")
        warnings += 1

    print(f"\n{INFO} 下载 / 链接分析")
    try:
        from .engines._native import import_engine_module

        dl_status = import_engine_module("download_engine").status_info()
        dl_version = (dl_status.get("yt_dlp") or {}).get("version")
        if dl_version:
            print(f"       {OK} yt-dlp {dl_version}  {(dl_status.get('yt_dlp') or {}).get('path')}")
        else:
            # 可选能力：缺失只提示，不算问题、不增加警告。
            print(
                f"       {INFO} yt-dlp 未安装（可选）："
                "engine/.venv 里执行 python -m pip install -U yt-dlp"
            )
        aria = dl_status.get("aria2c") or {}
        print(
            f"       {OK if aria.get('available') else INFO} aria2c "
            f"{aria.get('path') or '未安装（可选，用 yt-dlp 内置下载器）'}"
        )
        print(f"       {INFO} 下载目录 {dl_status.get('download_dir')}")
        print(
            f"       {INFO} 限额：单文件 {cfg.download_max_mb}MB，"
            f"同步接口 {cfg.download_sync_max_mb}MB，超时 {cfg.download_timeout:.0f}s"
        )
        print(
            f"       {INFO} SSRF：仅 http/https；私有/回环/链路本地地址与 file:// 拒绝"
        )
    except Exception as exc:  # noqa: BLE001 - 下载引擎状态不影响整体体检
        print(f"       {INFO} 下载引擎状态读取失败：{type(exc).__name__}: {exc}")

    print(f"\n{INFO} 模型文件")
    for name, directory, ok, detail in _doctor_models(cfg):
        if ok:
            mark = OK
        elif (
            "htdemucs_ft" in name
            or "faster-whisper" in name
            or "fsmn-vad" in name
            or "NLLB" in name
            or "视觉" in name
        ):
            # 可选组件：缺失只提示，不算问题、不增加警告（doctor 要能 0 警告）。
            mark = INFO
        else:
            mark = FAIL
            problems += 1
        print(f"       {mark} {name:<22} {directory}")
        print(f"              {detail}  ({_dir_size_mb(Path(directory))} MB)")

    print(f"\n{INFO} 数据目录")
    for label, path in (
        ("音色", cfg.voices_dir),
        ("输出", cfg.output_dir),
        ("声纹库", cfg.speakers_dir),
        ("日志", cfg.logs_dir),
        ("显存估算", cfg.vram_estimates_path.parent),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            writable = os.access(path, os.W_OK)
            mark = OK if writable else FAIL
            if not writable:
                problems += 1
        except OSError as exc:
            mark = FAIL
            problems += 1
            print(f"       {mark} {label:<8} {path}  ({exc})")
            continue
        print(f"       {mark} {label:<8} {path}")

    print(f"\n{INFO} 端口")
    if _port_free(cfg.host, cfg.port):
        print(f"       {OK} {cfg.host}:{cfg.port} 可绑定（服务未启动）")
    else:
        healthy, body = _http_probe(f"{cfg.base_url}/health")
        if healthy:
            print(f"       {OK} {cfg.host}:{cfg.port} 已被本服务占用，/health 正常")
            print(f"              {body[:200]}")
        else:
            print(f"       {WARN} {cfg.host}:{cfg.port} 被占用，且 /health 无响应")
            warnings += 1

    print(f"\n{INFO} 配置提示")
    print(f"       LOCALAI_VRAM_MARGIN_MB={cfg.vram_margin_mb}")
    print(
        "       idle_unload: "
        f"tts={cfg.idle_unload}s asr={cfg.asr_idle_unload}s "
        f"whisper={cfg.whisper_idle_unload}s "
        f"speaker={cfg.speaker_idle_unload}s separate={cfg.separate_idle_unload}s "
        f"translate={cfg.translate_idle_unload}s vision={cfg.vision_idle_unload}s"
    )
    print(
        f"       翻译：nllb-200-distilled-600M fp16={'开' if cfg.translate_fp16 else '关'}"
        f" beams={cfg.translate_num_beams} batch={cfg.translate_batch_size}"
    )
    print(f"       默认识别引擎：{cfg.asr_engine}；whisper 档位：{cfg.whisper_model}"
          f"（{cfg.whisper_device}/{cfg.whisper_compute}）")
    print(
        f"       视觉：Qwen2.5-VL-3B Q4_K_M（llama.cpp，端口 {cfg.vision_port}）"
        f" 默认 planner={cfg.vision_planner} 空闲 {cfg.vision_idle_unload}s 卸载"
    )
    print(f"       鉴权：{'已启用（LOCALAI_TOKEN）' if cfg.token else '未启用（默认）'}")

    print("\n" + "-" * 72)
    if problems:
        print(f"{FAIL} 体检完成：{problems} 个问题，{warnings} 个警告。")
        print("       排错见 docs/INSTALL.md；模型可用 `localai models list` 查看。")
        return 1
    print(f"{OK} 体检完成：关键项全部通过（{warnings} 个警告）。")
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def _model_directory(key: str, cfg: Config) -> Path:
    return {
        "tts": cfg.model_dir,
        "asr": cfg.asr_model_dir,
        "speaker": cfg.speaker_model_dir,
        "separate": cfg.demucs_repo,
    }[key]


def cmd_models_list(args: argparse.Namespace) -> int:
    cfg = get_config()
    print(f"模型目录根：{cfg.home / 'models'}\n")

    print("[1] TTS / IndexTTS-2.5")
    present, missing = _check_files(cfg.model_dir, REQUIRED_MODEL_FILES)
    aux_present, aux_missing = _check_files(cfg.model_dir, REQUIRED_AUX_FILES)
    print(f"    {cfg.model_dir}  ({_dir_size_mb(cfg.model_dir)} MB)")
    print(f"    主权重   {len(present)}/{len(REQUIRED_MODEL_FILES)} 就绪" + (f"，缺：{', '.join(missing)}" if missing else ""))
    print(f"    辅助模型 {len(aux_present)}/{len(REQUIRED_AUX_FILES)} 就绪" + (f"，缺：{', '.join(aux_missing)}" if aux_missing else ""))

    print("\n[2] ASR / SenseVoiceSmall")
    present, missing = _check_files(cfg.asr_model_dir, MODEL_SPECS["asr"]["required"])
    print(f"    {cfg.asr_model_dir}  ({_dir_size_mb(cfg.asr_model_dir)} MB)")
    print(f"    {len(present)}/{len(MODEL_SPECS['asr']['required'])} 就绪" + (f"，缺：{', '.join(missing)}" if missing else ""))

    print("\n[3] 声纹 / campplus")
    present, missing = _check_files(cfg.speaker_model_dir, MODEL_SPECS["speaker"]["required"])
    print(f"    {cfg.speaker_model_dir}  ({_dir_size_mb(cfg.speaker_model_dir)} MB)")
    print(f"    {len(present)}/{len(MODEL_SPECS['speaker']['required'])} 就绪" + (f"，缺：{', '.join(missing)}" if missing else ""))

    print("\n[4] 视觉 / Qwen2.5-VL-3B")
    try:
        from .engines._native import import_engine_module

        vision_status = import_engine_module("vision_engine").model_status()
        print(f"    {cfg.vision_dir}  ({_dir_size_mb(cfg.vision_dir)} MB)")
        print(
            "    "
            + ("就绪" if vision_status.get("ready") else "未安装")
            + f"：llama.cpp {vision_status.get('runtime_version')}，"
            + f"权重 {vision_status.get('model_size_mb')}MB + 投影层 {vision_status.get('mmproj_size_mb')}MB"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    状态读取失败：{type(exc).__name__}: {exc}")

    print("\n[5] 人声分离 / Demucs")
    print(f"    {cfg.demucs_repo}  ({_dir_size_mb(cfg.demucs_repo)} MB)")
    for model, files in DEMUCS_FILES.items():
        _, missing = _check_files(cfg.demucs_repo, files)
        print(f"    {model:<14} {'就绪' if not missing else '缺：' + ', '.join(missing)}")
    _, ft_missing = _check_files(cfg.demucs_repo, DEMUCS_FT_FILES)
    print(
        "    htdemucs_ft    "
        + (
            "就绪"
            if not ft_missing
            else "未下载（可选；4 个 bag 权重，显存/耗时约为 htdemucs 的 4 倍）"
        )
    )

    print("\n下载：localai models download <tts|asr|speaker|demucs|all>")
    return 0


def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    tmp.replace(target)


def _download_snapshot(key: str, cfg: Config) -> list[str]:
    """用 ModelScope 优先下载一个模型目录；返回缺少的文件。"""
    spec = MODEL_SPECS[key]
    target = _model_directory(key, cfg)
    target.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    modelscope_id = spec.get("modelscope")
    hf_id = spec.get("huggingface")

    if modelscope_id and importlib.util.find_spec("modelscope") is not None:
        print(f"    使用 ModelScope：{modelscope_id} -> {target}")
        try:
            from modelscope import snapshot_download  # noqa: PLC0415

            snapshot_download(modelscope_id, local_dir=str(target))
            return []
        except Exception as exc:  # noqa: BLE001
            messages.append(f"ModelScope 下载失败：{type(exc).__name__}: {exc}")
            print(f"    [WARN] {messages[-1]}")

    if hf_id and importlib.util.find_spec("huggingface_hub") is not None:
        print(f"    使用 HuggingFace Hub：{hf_id} -> {target}")
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            snapshot_download(repo_id=hf_id, local_dir=str(target))
            return []
        except Exception as exc:  # noqa: BLE001
            messages.append(f"HuggingFace 下载失败：{type(exc).__name__}: {exc}")
            print(f"    [WARN] {messages[-1]}")

    raise RuntimeError(
        "；".join(messages)
        or "没有可用的下载后端：请先 pip install modelscope 或 huggingface_hub"
    )


def _download_demucs(cfg: Config, *, include_ft: bool = False) -> int:
    cfg.demucs_repo.mkdir(parents=True, exist_ok=True)
    for model, files in DEMUCS_FILES.items():
        for name in files:
            target = cfg.demucs_repo / name
            if target.is_file() and target.stat().st_size > 0:
                print(f"    {OK} 已存在 {name}")
                continue
            url = f"{DEMUCS_BASE}/{name}"
            print(f"    下载 {url}")
            try:
                _download_url(url, target)
                print(f"    {OK} {name}（{target.stat().st_size / 1024**2:.1f} MB）")
            except Exception as exc:  # noqa: BLE001
                print(f"    {FAIL} {name} 下载失败：{type(exc).__name__}: {exc}")
                print("           可手动从 https://github.com/facebookresearch/demucs 获取权重")
                return 1

    ft_present = not _check_files(cfg.demucs_repo, DEMUCS_FT_FILES)[1]
    if include_ft and not ft_present:
        print(
            f"    {WARN} htdemucs_ft 未下载：它是 4 个模型的 bag（约 4 倍显存/耗时），"
            "本项目默认不拉；需要时按 demucs 官方说明手动放置 "
            + "、".join(DEMUCS_FT_FILES)
        )
    elif not ft_present:
        print(f"    {INFO} htdemucs_ft 未下载（可选，缺失不影响 htdemucs / htdemucs_6s）")
    return 0


def cmd_models_download(args: argparse.Namespace) -> int:
    cfg = get_config()
    raw = (args.name or "").strip().lower().replace("_", "-")
    key = _MODEL_ALIASES.get(raw)
    if key is None:
        print(f"不认识的名字：{args.name}")
        print("可选：tts / asr / speaker / demucs / all")
        return 2

    if key == "separate":
        return _download_demucs(cfg)

    keys = ["tts", "asr", "speaker"] if key == "all" else [key]
    failed = 0
    for item in keys:
        label = MODEL_SPECS[item]["label"]
        print(f"\n[{item}] {label}")
        try:
            _download_snapshot(item, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"    {FAIL} 下载失败：{exc}")
            failed += 1
            continue
        target = _model_directory(item, cfg)
        missing = _check_files(target, MODEL_SPECS[item]["required"])[1]
        if missing:
            print(f"    {WARN} 下载完成但仍缺：{', '.join(missing)}")
            failed += 1
        else:
            print(f"    {OK} {label} 就绪（{target}）")

    if key == "all":
        print("\n[separate] Demucs")
        failed += _download_demucs(cfg)

    if failed:
        print(f"\n{FAIL} 有 {failed} 项未完成；可再跑一次，已下载的文件会跳过。")
        return 1
    print(f"\n{OK} 模型下载完成。运行 `localai doctor` 复查。")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localai",
        description="本地 AI 服务层：TTS / ASR / 声纹 / 人声分离 / 媒体工具",
    )
    parser.add_argument("--version", action="version", version=f"localai-service {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", help="监听地址（默认 LOCALAI_HOST / 127.0.0.1）")
    serve.add_argument("--port", type=int, help="监听端口（默认 LOCALAI_PORT / 7880）")
    serve.add_argument("--log-level", help="日志级别（debug/info/warning）")
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="体检：Python/依赖/GPU/ffmpeg/模型/端口")
    doctor.set_defaults(func=cmd_doctor)

    version = sub.add_parser("version", help="打印版本与路径")
    version.set_defaults(func=cmd_version)

    models = sub.add_parser("models", help="模型管理")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list", help="检查本机模型文件")
    models_list.set_defaults(func=cmd_models_list)
    models_download = models_sub.add_parser("download", help="下载缺失权重（ModelScope 优先）")
    models_download.add_argument(
        "name",
        help="tts / asr / speaker / demucs / all",
    )
    models_download.set_defaults(func=cmd_models_download)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
