"""翻译适配器：NLLB-200-distilled-600M（能力名 ``translate``）。

和其它 GPU 能力完全同构：
- 第一次翻译才加载（服务启动仍然 0 模型）；
- 空闲看门狗按 ``LOCALAI_TRANSLATE_IDLE_UNLOAD`` 卸载（默认 900 秒，0=关闭）；
- 注册进显存管家，估算 1500 MB，加载后按实测增量校准；
- 和 TTS / ASR / Demucs 按 LRU 互相腾显存（``vram.ensure_capacity``）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, Sequence

from ..config import get_config
from ..constants import TRANSLATE_ESTIMATE_MB
from ..core.errors import InsufficientVramError, TranslateBackendError
from ..core.vram import ensure_model_loaded, get_vram
from ._native import import_engine_module
from .base import BaseAdapter

logger = logging.getLogger("localai.engines.translate")


class TranslateAdapter(BaseAdapter):
    """NLLB-200 翻译适配器。"""

    name = "translate"
    title = "翻译（NLLB-200）"
    model_name = "nllb-200-distilled-600M"
    requires_gpu = True
    DEFAULT_ESTIMATE_MB = TRANSLATE_ESTIMATE_MB
    #: 校准下限：fp16 权重约 1.2GB，加上生成时的 KV/中间张量，别被低估到危险值。
    MIN_ESTIMATE_MB = 600
    MAX_ESTIMATE_MB: float = 1800
    ENDPOINTS = (
        "/translate",
        "/translate/batch",
        "/translate/status",
        "/jobs/translate",
    )
    #: transformers 的 generate 没有验证过同一模型并发安全，强制串行；
    #: 批量吞吐靠一次喂多条（batch_size），而不是多请求并发。
    CAN_INFER_IN_PARALLEL = False
    PARALLEL_NOTE = "transformers.generate 未实测可并发，强制串行；吞吐靠批内 batch_size"

    @property
    def loader_hint(self) -> str:
        return "nllb-200-distilled-600M（翻译，fp16/GPU）"

    def get_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is None:
                self._module = import_engine_module("translate_engine")
        return self._module

    # ------------------------------------------------------------------
    # 状态 / 显存估算
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        try:
            module = self.get_module()
            status = module.model_status()
        except Exception as exc:  # noqa: BLE001 - health 不能让整个接口挂掉
            return {
                "ready": False,
                "detail": f"翻译引擎不可用：{exc}",
                "missing": [],
                "loaded": False,
                "load_seconds": None,
                "model_dir": str(self.cfg.translate_weight_dir),
                "model": self.model_name,
            }

        if status.get("ready"):
            detail = f"{self.model_name} 权重已就绪" + (
                "（模型已加载）" if status.get("loaded") else "（首次翻译请求时加载）"
            )
        else:
            detail = (
                f"{self.model_name} 权重未下载：{status.get('model_dir')}"
                "（首次翻译时自动从 ModelScope 下载，失败再退 HF 镜像；"
                "日志见 logs/nllb_install.log）"
            )
        return {
            "ready": bool(status.get("ready")),
            "detail": detail,
            "missing": status.get("missing") or [],
            "loaded": bool(status.get("loaded")),
            "load_seconds": status.get("load_seconds"),
            "load_error": status.get("load_error"),
            "model_dir": status.get("model_dir"),
            "model": status.get("model") or self.model_name,
            "device": status.get("device"),
            "dtype": status.get("dtype"),
            "fp16": status.get("fp16"),
            "size_mb": status.get("size_mb"),
            "weights_source": status.get("weights_source"),
            "install_log": status.get("install_log"),
            "package": status.get("package"),
        }

    def load(self) -> None:
        """同步加载（调用方放线程池）。"""
        self.get_module().ensure_loaded()

    def loaded(self) -> bool:
        module = self._module
        if module is None:
            return False
        try:
            return bool(module.is_loaded())
        except Exception:  # noqa: BLE001
            return False

    def languages(self) -> list[dict[str, str]]:
        """语言表（懒加载；引擎模块 import 不碰 torch）。"""
        try:
            return list(self.get_module().language_table())
        except Exception:  # noqa: BLE001 - 状态页不能因此挂掉
            return []

    def normalize_language(self, code: str, *, allow_auto: bool = False) -> str:
        """别名 -> NLLB 码；不合法时抛引擎侧 ``LangError``（路由转 400）。"""
        return self.get_module().normalize_lang(code, allow_auto=allow_auto)

    def detect_language(self, text: str) -> str:
        """自动检测语言，返回 NLLB 码；失败抛 ``LangError``。"""
        return self.get_module().detect_language(text)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    async def infer(
        self,
        texts: Sequence[str],
        *,
        source: str = "auto",
        target: str = "en",
        batch_size: Optional[int] = None,
        num_beams: Optional[int] = None,
        max_len: int = 400,
    ) -> dict[str, Any]:
        """翻译一批文本；返回 ``{engine, source, target, docs, elapsed_ms}``。"""
        module = self.get_module()
        vram = get_vram()
        batch = int(batch_size or self.cfg.translate_batch_size)
        beams = int(num_beams or self.cfg.translate_num_beams)
        started = time.perf_counter()
        self.mark_activity()
        try:
            async with self.infer_lock:
                vram.begin_use(self.name)
                try:
                    # 加载前统一腾显存；已加载时只 touch，不会重复加载。
                    await ensure_model_loaded(self)
                    self.mark_activity()
                    docs = await asyncio.to_thread(
                        module.translate_documents,
                        list(texts),
                        source,
                        target,
                        max_len,
                        batch,
                        beams,
                    )
                    vram.touch(self.name)
                    self.mark_activity()
                finally:
                    vram.end_use(self.name)
        except InsufficientVramError:
            raise
        except Exception as exc:  # noqa: BLE001 - 统一成 JSON 错误
            lang_error = getattr(module, "LangError", None)
            if lang_error is not None and isinstance(exc, lang_error):
                raise TranslateBackendError(str(exc), status_code=400) from exc
            if isinstance(exc, TranslateBackendError):
                raise
            raise TranslateBackendError(
                f"NLLB 翻译失败：{type(exc).__name__}: {exc}",
                detail=str(exc),
            ) from exc

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        source_code = docs[0].get("source") if docs else source
        target_code = docs[0].get("target") if docs else target
        chunks = sum(int(doc.get("chunks") or 0) for doc in docs)
        return {
            "engine": "nllb",
            "model": self.model_name,
            "source": source_code,
            "target": target_code,
            "docs": docs,
            "chunks": chunks,
            "elapsed_ms": elapsed_ms,
            "loaded": self.loaded(),
            "device": (self.status().get("device") if self.loaded() else None),
        }


__all__ = ["TranslateAdapter"]
