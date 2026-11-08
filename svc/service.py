"""兼容壳（旧入口）。

历史调用 ``python svc\\service.py`` / ``powershell start_ai_service.ps1`` 都还可用，
但实现已经全部搬到 ``src/localai`` 包里：

- 服务装配：``localai.api.app``
- 配置：``localai.config``
- 引擎适配器：``localai.engines``
- 显存 / 空闲看门狗 / 注册表：``localai.core``

改成两层架构后，本文件只负责把旧路径指向新包，不要再往这里加业务逻辑。
新入口：``python -m localai serve`` 或 ``localai serve``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许「源码直接运行」：即使没 pip install -e .，也能从 ../src 找到包。
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from localai.legacy import app, create_app, main  # noqa: E402

__all__ = ["app", "create_app", "main"]


if __name__ == "__main__":
    main()
