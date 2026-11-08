"""ComfyUI 后端（旧路径，保留兼容）。

当前机器因 transformers 5.x 不可用，``TTS_BACKEND=comfy`` 不是默认值；
但接口语义保持不变：上传参考音频 -> 提交工作流 -> 轮询 /history -> 下载音频
-> 转 WAV -> 归档。所有调用都走 HTTP，不 import ComfyUI 的任何模块。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import re
import secrets
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config import Config, get_config
from ..constants import AUDIO_EXTENSIONS
from ..core.audio import archive_wav, to_wav
from ..core.errors import ComfyError
from ..core.text import duration_factor_for, normalize_language, parse_emotion
from ..core.voices import Voice, resolve_voice

logger = logging.getLogger("localai.engines.comfy")

# 固定的 client_id：同一个服务进程内复用，方便在 ComfyUI 日志里识别。
CLIENT_ID = f"localai-svc-{uuid.uuid4()}"

# 已上传音色的缓存：文件内容 sha256 -> ComfyUI input 目录里的文件名。
_UPLOAD_CACHE: dict[str, str] = {}
_UPLOAD_LOCK = asyncio.Lock()


def _http_client(cfg: Optional[Config] = None) -> httpx.AsyncClient:
    cfg = cfg or get_config()
    return httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.job_timeout + 30.0, connect=5.0, write=30.0),
        follow_redirects=True,
    )


async def comfy_up(client: httpx.AsyncClient, cfg: Optional[Config] = None) -> bool:
    cfg = cfg or get_config()
    try:
        response = await client.get(f"{cfg.comfy_base}/system_stats", timeout=3.0)
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False


async def ensure_uploaded(client: httpx.AsyncClient, voice: Voice) -> str:
    """把参考音频上传到 ComfyUI 的 input 目录，返回 LoadAudio 可用的文件名。"""
    data = voice.path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    async with _UPLOAD_LOCK:
        cached = _UPLOAD_CACHE.get(digest)
        if cached:
            return cached

        safe_stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", voice.path.stem)[:40] or "voice"
        filename = f"svc_{safe_stem}_{digest[:10]}{voice.path.suffix.lower()}"
        mime = mimetypes.guess_type(voice.path.name)[0] or "application/octet-stream"

        try:
            response = await client.post(
                f"{get_config().comfy_base}/upload/image",
                files={"image": (filename, data, mime)},
                data={"type": "input", "overwrite": "true"},
            )
        except httpx.HTTPError as exc:
            raise ComfyError(f"上传参考音频到 ComfyUI 失败：{exc}") from exc

        if response.status_code != 200:
            raise ComfyError(
                f"上传参考音频失败（HTTP {response.status_code}）",
                comfy_error=response.text,
            )

        try:
            payload = response.json()
        except ValueError:
            payload = {}
        name = payload.get("name") or filename
        _UPLOAD_CACHE[digest] = name
        logger.info("参考音频已上传：%s -> ComfyUI input/%s", voice.path.name, name)
        return name


def build_tts_workflow(
    *,
    text: str,
    audio_name: str,
    language: str,
    duration_factor: float,
    seed: int,
    emotion: Optional[str] = None,
) -> dict[str, Any]:
    """LoadAudio -> IndexTTS2_5Node -> SaveAudio（API 格式）。"""
    node_inputs: dict[str, Any] = {
        "text": text,
        "reference_audio": ["1", 0],
        "language": language,
        "duration_factor": duration_factor,
        "seed": seed,
    }
    if emotion:
        node_inputs.update(parse_emotion(emotion))

    prefix = "svc_tts/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6]
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "2": {"class_type": "IndexTTS2_5Node", "inputs": node_inputs},
        "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": prefix}},
    }


def build_passthrough_workflow(audio_name: str, prefix: str = "svc_selftest") -> dict[str, Any]:
    """最小链路：LoadAudio -> SaveAudio，不加载 TTS 模型。"""
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "2": {"class_type": "SaveAudio", "inputs": {"audio": ["1", 0], "filename_prefix": prefix}},
    }


def _format_execution_error(entry: dict[str, Any]) -> tuple[Optional[str], list[Any]]:
    """从 history 的 status.messages 里提取 execution_error 原文。"""
    status = entry.get("status") or {}
    messages = status.get("messages") or []
    chunks: list[str] = []
    for item in messages:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        kind, payload = item
        if kind not in ("execution_error", "execution_interrupted"):
            continue
        if not isinstance(payload, dict):
            chunks.append(f"{kind}: {payload}")
            continue
        node = payload.get("node_type") or payload.get("node_id") or "?"
        exc_type = payload.get("exception_type") or "Error"
        exc_msg = payload.get("exception_message") or ""
        chunks.append(f"{kind}: 节点 {node} 抛出 {exc_type}: {exc_msg}")
        traceback_text = payload.get("traceback")
        if traceback_text:
            tb = "\n".join(traceback_text) if isinstance(traceback_text, list) else str(traceback_text)
            chunks.append("traceback:\n" + tb[-4000:])
    return ("\n".join(chunks) if chunks else None), messages


async def submit_prompt(client: httpx.AsyncClient, workflow: dict[str, Any]) -> str:
    """提交工作流，返回 prompt_id；ComfyUI 的校验错误原样抛出。"""
    try:
        response = await client.post(
            f"{get_config().comfy_base}/prompt",
            json={"prompt": workflow, "client_id": CLIENT_ID},
        )
    except httpx.HTTPError as exc:
        raise ComfyError(f"提交工作流到 ComfyUI 失败：{exc}") from exc

    if response.status_code != 200:
        raise ComfyError(
            f"ComfyUI /prompt 返回 HTTP {response.status_code}",
            comfy_error=response.text,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ComfyError("ComfyUI /prompt 返回的不是合法 JSON", comfy_error=response.text) from exc

    if payload.get("error") or payload.get("node_errors"):
        raise ComfyError(
            "ComfyUI 拒绝了工作流（节点校验失败）",
            comfy_error=json.dumps(payload, ensure_ascii=False, indent=2),
            prompt_id=payload.get("prompt_id"),
        )

    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise ComfyError("ComfyUI 没有返回 prompt_id", comfy_error=response.text)
    logger.info("已提交 ComfyUI 任务 prompt_id=%s", prompt_id)
    return prompt_id


async def wait_history(
    client: httpx.AsyncClient,
    prompt_id: str,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """轮询 /history/{prompt_id}，直到成功或失败。"""
    cfg = get_config()
    timeout = cfg.job_timeout if timeout is None else timeout
    deadline = time.monotonic() + timeout
    last_status = ""
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{cfg.comfy_base}/history/{prompt_id}")
        except httpx.HTTPError as exc:
            raise ComfyError(f"查询 ComfyUI 任务状态失败：{exc}", prompt_id=prompt_id) from exc

        if response.status_code != 200:
            await asyncio.sleep(cfg.poll_interval)
            continue

        try:
            payload = response.json()
        except ValueError:
            await asyncio.sleep(cfg.poll_interval)
            continue

        entry = payload.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status") or {}
            status_str = status.get("status_str") or ""
            if status_str and status_str != last_status:
                logger.info("任务 %s 状态：%s", prompt_id, status_str)
                last_status = status_str

            if status_str == "error":
                comfy_error, messages = _format_execution_error(entry)
                if not comfy_error:
                    comfy_error = json.dumps(status, ensure_ascii=False, indent=2)
                raise ComfyError(
                    "ComfyUI 执行失败（详见 comfy_error）",
                    comfy_error=comfy_error,
                    messages=messages,
                    prompt_id=prompt_id,
                )
            if status.get("completed") is True or status_str == "success":
                return entry
            if entry.get("outputs"):
                return entry

        await asyncio.sleep(cfg.poll_interval)

    raise ComfyError(
        f"等待 ComfyUI 任务超时（{timeout:.0f} 秒）",
        prompt_id=prompt_id,
        status_code=504,
    )


def find_audio_output(entry: dict[str, Any]) -> dict[str, Any]:
    """从 outputs 里找 SaveAudio 产出的音频文件描述。"""
    outputs = entry.get("outputs") or {}
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename") or ""
                if item.get("type") == "output" and Path(filename).suffix.lower() in AUDIO_EXTENSIONS:
                    return item
    raise ComfyError(
        "ComfyUI 任务完成，但 outputs 里没有找到音频文件",
        comfy_error=json.dumps(entry.get("outputs") or {}, ensure_ascii=False, indent=2),
    )


async def download_output(client: httpx.AsyncClient, item: dict[str, Any]) -> bytes:
    params = {
        "filename": item.get("filename", ""),
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    }
    try:
        response = await client.get(f"{get_config().comfy_base}/view", params=params)
    except httpx.HTTPError as exc:
        raise ComfyError(f"从 ComfyUI 下载音频失败：{exc}") from exc
    if response.status_code != 200:
        raise ComfyError(
            f"从 ComfyUI 下载音频失败（HTTP {response.status_code}）",
            comfy_error=response.text,
        )
    return response.content


async def run_workflow(
    workflow: dict[str, Any],
    timeout: Optional[float] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> tuple[dict[str, Any], bytes]:
    """提交 -> 轮询 -> 下载，返回 (history entry, 原始音频字节)。"""
    own_client = client is None
    if own_client:
        client = _http_client()
    assert client is not None
    try:
        prompt_id = await submit_prompt(client, workflow)
        entry = await wait_history(client, prompt_id, timeout=timeout)
        item = find_audio_output(entry)
        raw = await download_output(client, item)
        return entry, raw
    finally:
        if own_client:
            await client.aclose()


async def synthesize_comfy(
    *,
    client: Optional[httpx.AsyncClient],
    text: str,
    voice_name: str,
    speed: float,
    emotion: Optional[str],
    seed: Optional[int],
    language: Optional[str],
) -> Any:
    """ComfyUI 后端完整链路；返回 TtsResult。"""
    from .tts_indextts import TtsResult

    cfg = get_config()
    voice = resolve_voice(voice_name)
    actual_seed = seed if seed is not None else secrets.randbelow(2**32)
    actual_language = normalize_language(language, text)

    own_client = client is None
    if own_client:
        client = _http_client(cfg)
    assert client is not None
    try:
        audio_name = await ensure_uploaded(client, voice)
        workflow = build_tts_workflow(
            text=text,
            audio_name=audio_name,
            language=actual_language,
            duration_factor=duration_factor_for(speed),
            seed=actual_seed,
            emotion=emotion,
        )
        entry, raw = await run_workflow(workflow, client=client)
    finally:
        if own_client:
            await client.aclose()

    wav = to_wav(raw)
    path = archive_wav(wav, voice_name, actual_seed)
    comfy_filename = ""
    try:
        comfy_filename = str(find_audio_output(entry).get("filename") or "")
    except ComfyError:
        pass

    logger.info(
        "合成完成（comfy）：voice=%s seed=%s language=%s bytes=%d -> %s",
        voice_name,
        actual_seed,
        actual_language,
        len(wav),
        path,
    )
    return TtsResult(
        wav=wav,
        path=path,
        seed=actual_seed,
        language=actual_language,
        comfy_filename=comfy_filename,
    )
