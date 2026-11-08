"""视觉适配器：本地 Qwen2.5-VL-3B（能力名 ``vision``）。

生命周期与其它 GPU 能力完全同构：
- 服务启动时 **0 模型**，第一次视觉请求才拉起 llama-server；
- 注册进显存管家，估算约 3400 MB，加载后按实测增量校准；
- 和 TTS / ASR / 翻译按 LRU 互相腾显存（8G 卡上不能同时驻留）；
- 空闲看门狗按 ``LOCALAI_VISION_IDLE_UNLOAD`` 卸载（默认 900 秒，0=关闭）。

与 transformers 方案不同，VLM 跑在独立子进程里，所以 ``unload()`` 只需要
停掉子进程，不会在服务进程里留下 torch 的显存碎片。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Sequence

from ..core.errors import InsufficientVramError, VisionBackendError
from ..core.vram import ensure_model_loaded, get_vram
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.vision")


class VisionAdapter(BaseAdapter):
    """Qwen2.5-VL-3B（llama.cpp 子进程）适配器。"""

    name = "vision"
    title = "视觉理解（Qwen2.5-VL-3B）"
    model_name = "Qwen2.5-VL-3B-Instruct-Q4_K_M"
    requires_gpu = True
    DEFAULT_ESTIMATE_MB = 3400
    #: 校准下限：Q4 权重约 1.93GB + mmproj 1.34GB，别被 WDDM 缓存低估。
    MIN_ESTIMATE_MB = 2400
    MAX_ESTIMATE_MB: float = 4000
    ENDPOINTS = (
        "/vision/describe",
        "/vision/scan",
        "/vision/frames",
        "/vision/video",
        "/vision/frame",
        "/vision/status",
        "/jobs/vision",
    )
    #: 单个 llama-server 只开一个 slot，强制串行；批量靠一次 /vision/scan。
    CAN_INFER_IN_PARALLEL = False
    PARALLEL_NOTE = "llama-server 单 slot，强制串行"

    @property
    def loader_hint(self) -> str:
        return "Qwen2.5-VL-3B-Instruct Q4_K_M（llama.cpp 子进程）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("vision_engine")
        return self._module

    # ------------------------------------------------------------------
    # 状态 / 加载 / 卸载
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        try:
            module = self.get_module()
            status = module.model_status()
        except Exception as exc:  # noqa: BLE001 - /health 不能因此挂掉
            return {
                "ready": False,
                "loaded": False,
                "model": self.model_name,
                "model_dir": str(self.cfg.vision_dir),
                "detail": f"视觉引擎不可用：{type(exc).__name__}: {exc}",
                "missing": [],
            }
        status["model"] = status.get("model") or self.model_name
        if status.get("ready"):
            status["detail"] = status.get("detail") or (
                f"{self.model_name} 已就绪"
                + ("（已加载）" if status.get("loaded") else "（首次视觉请求时加载）")
            )
        return status

    def load(self) -> None:
        """同步加载：拉起 llama-server 并等 /health（调用方放线程池）。"""
        module = self.get_module()
        module.ensure_loaded()
        # CUDA 的 mem_get_info 看不到 Vulkan 子进程；用引擎测到的 nvidia-smi
        # 增量回填显存管家，保证 /vram 的估算与实测一致。
        try:
            delta = module.vram_delta_mb()
            if delta:
                get_vram().record_measured(self.name, delta)
        except Exception:  # noqa: BLE001 - 校准失败不影响加载
            logger.debug("视觉显存校准失败", exc_info=True)

    def ensure_loaded(self) -> None:
        self.load()

    def unload(self) -> None:
        """停止 llama-server 子进程（可重复调用）。"""
        module = self._module
        if module is None:
            return
        module.unload()

    def loaded(self) -> bool:
        module = self._module
        if module is None:
            return False
        try:
            return bool(module.is_loaded())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    async def describe_image(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """单图理解；返回 ``{ok,text,elapsed_ms,model,loaded}``。"""
        started = time.perf_counter()
        text = await self._run(
            "describe",
            str(image_path),
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "ok": True,
            "text": text,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": self.model_name,
            "loaded": self.loaded(),
        }

    async def describe_many(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """多图批量（一次加载、逐张推理）；返回 ``{ok,texts,elapsed_ms}``。"""
        started = time.perf_counter()
        texts = await self._run(
            "describe_many",
            [str(item) for item in images],
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "ok": True,
            "texts": texts,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": self.model_name,
        }

    async def describe_frames(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 768,
        max_side: int = 256,
        temperature: float = 0.0,
        labels: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """多帧一次性理解（``/vision/video`` 用）；返回 ``{ok,text,elapsed_ms}``。

        所有帧塞进同一个请求，模型能横向比较整段画面的变化；
        ``max_side`` 控制每帧缩略图尺寸，``labels`` 给每帧加时间标签，
        避免小模型只盯第一帧。
        """
        started = time.perf_counter()
        text = await self._run(
            "describe_frames",
            [str(item) for item in images],
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            max_side=max_side,
            temperature=temperature,
            labels=list(labels) if labels is not None else None,
        )
        return {
            "ok": True,
            "text": text,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": self.model_name,
            "loaded": self.loaded(),
        }

    async def describe_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """纯文本推理（规划器）；返回 ``{ok,text,elapsed_ms}``。"""
        started = time.perf_counter()
        text = await self._run(
            "describe_text",
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "ok": True,
            "text": text,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "model": self.model_name,
        }

    async def _run(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        """统一走「推理锁 -> 腾显存 -> 加载 -> 线程池调用 -> 记活跃」。"""
        module = self.get_module()
        vram = get_vram()
        self.mark_activity()
        try:
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    await ensure_model_loaded(self)
                    self.mark_activity()
                    func = getattr(module, func_name)
                    result = await asyncio.to_thread(func, *args, **kwargs)
                    vram.touch(self.name)
                    self.mark_activity()
                    return result
                finally:
                    vram.end_use(self.name)
        except InsufficientVramError:
            raise
        except VisionBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一成 503
            vision_error = getattr(module, "VisionError", None)
            if vision_error is not None and isinstance(exc, vision_error):
                raise VisionBackendError(str(exc), detail=str(exc)) from exc
            raise VisionBackendError(
                f"视觉推理失败：{type(exc).__name__}: {exc}",
                detail=str(exc),
            ) from exc


__all__ = ["VisionAdapter"]
