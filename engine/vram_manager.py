"""兼容旧路径：统一显存管家已迁移到 ``src/localai/core/vram.py``。

服务层现在只从 ``localai.core.vram`` 导入；本文件保留为一个薄转发层，
防止旧脚本/旧文档里的 ``import vram_manager`` 断掉。
新代码请直接 ``from localai.core.vram import ...``。
"""

from __future__ import annotations

from localai.core.vram import (  # noqa: F401
    FALLBACK_BASE_MB,
    FALLBACK_TOTAL_MB,
    ModelEntry,
    VramManager,
)

__all__ = ["VramManager", "ModelEntry", "FALLBACK_TOTAL_MB", "FALLBACK_BASE_MB"]
