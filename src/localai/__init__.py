"""LocalAI Service —— 可移植的本地 AI 服务层。

本包只做两件事：

1. **把模型能力封装成 HTTP 接口**（TTS / ASR / 声纹 / 人声分离 / 媒体工具）；
2. **管理这些能力的生命周期**（懒加载、空闲卸载、显存管家、统一错误码）。

任何程序都可以只通过 HTTP 使用它，不需要 import 本包，也不需要知道
模型文件、虚拟环境、ffmpeg 在哪里。给调用方的文档见 ``docs/SERVICE_API.md``；
给未来 DSH 插件的契约见 ``docs/PLUGIN_CONTRACT.md``。

目录分层::

    localai.config          唯一配置入口（环境变量 / LOCALAI_HOME）
    localai.core            与协议无关的底座：显存 / 空闲看门狗 / 注册表 / 错误
    localai.engines         各能力的适配器（统一 status/load/infer/unload/estimate）
    localai.api             FastAPI 装配与按能力拆分的路由
    localai.cli             localai serve / doctor / models / version
    localai.legacy          旧入口兼容（svc/service.py 只做转发）
"""

from __future__ import annotations

__all__ = ["__version__", "API_VERSION"]

# 服务版本（/capabilities、/docs、localai version 都会展示）
__version__ = "2.2.0"

# 对外 HTTP 契约版本：接口发生不兼容变化时才 +1。
# 插件应当拿它做版本协商，见 docs/PLUGIN_CONTRACT.md。
API_VERSION = "1"
