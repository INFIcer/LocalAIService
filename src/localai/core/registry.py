"""能力注册表：新增能力只需要注册一个 adapter，主程序不用改 if-else。

统一接口（鸭子类型，见 ``localai.engines.base.BaseAdapter``）::

    name / title / requires_gpu / loader_hint
    status() -> dict      当前是否 ready / loaded，模型名与目录
    loaded() -> bool
    load()                加载（同步，调用方放线程池）
    infer(...)            能力相关（异步）
    unload()              卸载（同步，可重复调用）
    estimate_mb()         当前显存估算
    idle_seconds()        空闲秒数（没调用过返回 None）
"""

from __future__ import annotations

from typing import Any, Iterator, Optional


class EngineRegistry:
    """能力的名字 -> 适配器映射。"""

    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}

    def register(self, adapter: Any) -> Any:
        name = getattr(adapter, "name", "")
        if not name:
            raise ValueError("adapter 必须有 name")
        if name in self._adapters:
            raise ValueError(f"能力 {name!r} 已经注册过了")
        self._adapters[name] = adapter
        return adapter

    def get(self, name: str) -> Any:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(f"未注册的能力：{name}") from exc

    def find(self, name: str) -> Optional[Any]:
        return self._adapters.get(name)

    def all(self) -> list[Any]:
        return list(self._adapters.values())

    def names(self) -> list[str]:
        return list(self._adapters)

    def gpu_adapters(self) -> list[Any]:
        return [adapter for adapter in self._adapters.values() if adapter.requires_gpu]

    def __contains__(self, name: object) -> bool:
        return name in self._adapters

    def __iter__(self) -> Iterator[Any]:
        return iter(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)
