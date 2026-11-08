"""旧入口兼容层。

``svc/service.py`` 现在只是一层薄壳：``from localai.legacy import app, main``。
这样旧调用（``python svc/service.py``、旧快捷方式、旧脚本）都不会断，
而真正的实现全部在 ``localai`` 包里。
"""

from __future__ import annotations

import logging

from .api.app import app, create_app
from .config import get_config

__all__ = ["app", "create_app", "main"]


def main() -> None:
    """``python svc/service.py`` 的兼容入口：等价于 ``localai serve``。"""
    import uvicorn

    cfg = get_config()
    logging.getLogger("localai").info("旧入口启动：%s", cfg.base_url)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
