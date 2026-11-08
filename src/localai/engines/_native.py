"""原生引擎模块的惰性导入。

``engine/`` 目录是随仓库分发的原生推理实现（IndexTTS-2.5 / SenseVoice /
campplus / Demucs / ffmpeg 封装）。服务层不把 torch 代码搬进包里，而是：

1. 用 :mod:`localai.config` 统一算出所有路径并写回 ``INDEX_TTS_*`` 环境变量；
2. 第一次真正用到某能力时才把 ``engine/`` 加入 ``sys.path`` 并 import 对应模块。

这样换机器只需要改配置，不需要改任何 import。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from ..config import get_config


class NativeEngineMissing(RuntimeError):
    """engine 目录不存在或原生模块导入失败。"""


def ensure_engine_path() -> Path:
    cfg = get_config()
    engine_dir = Path(cfg.engine_dir)
    if not engine_dir.is_dir():
        raise NativeEngineMissing(
            f"原生引擎目录不存在：{engine_dir}。"
            "请设置 LOCALAI_ENGINE_DIR 指向 engine 目录（见 docs/INSTALL.md）。"
        )
    path_text = str(engine_dir)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return engine_dir


def import_engine_module(module_name: str) -> Any:
    """导入 ``engine/<module_name>.py``；失败时给出可读的错误。"""
    engine_dir = ensure_engine_path()
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise NativeEngineMissing(
            f"无法导入原生引擎模块 {module_name}.py（{engine_dir}）："
            f"{type(exc).__name__}: {exc}"
        ) from exc
