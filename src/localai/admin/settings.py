"""管理设置的读取 / 校验 / 写入。

设计要点
--------
1. **白名单**：只认 :data:`localai.settings_spec.SETTINGS` 里的键，其它键一律
   400；新增设置只改清单，路由不用动。
2. **环境变量优先**：启动时被环境变量覆盖的键仍然可以写进 ``config.json``，
   但接口会明确返回 ``source: env`` 和警告，避免“保存成功却不生效”的误会。
3. **只写改动项**：读取已有 ``config.json``，保留不认识的键；值等于默认值时
   直接把该键删掉，文件始终保持最小。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..config import Config, config_file_path, load_config_file
from ..settings_spec import BY_KEY, GROUP_TITLES, SETTINGS, coerce_value

logger = logging.getLogger("localai.admin.settings")


class SettingsError(ValueError):
    """设置校验失败（路由转 400，消息直接展示给用户）。"""


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def _source(item_key: str, cfg: Config, file_data: dict[str, Any]) -> str:
    """返回 value 的来源：env / file / default。

    只认 ``cfg.env_overrides``（启动时快照），不能在请求时读 ``os.environ``：
    ``Config.export_env()`` 会把配置写回同名环境变量，事后读会把 file/default
    误判成 env。
    """
    if cfg.env_overrides.get(item_key):
        return "env"
    if item_key in file_data:
        return "file"
    return "default"


def _item_value(item: Any, cfg: Config) -> Any:
    """把配置对象上的有效值转成接口可序列化的形式。"""
    if item.key == "token":
        # 永远不回显真实令牌；只告诉前端“已设置/未设置”。
        return ""
    if item.key == "ffmpeg":
        return str(cfg.ffmpeg) if cfg.ffmpeg else ""
    if item.key == "asr_parallel":
        return cfg.asr_parallel_effective()
    if item.type == "list":
        return list(getattr(cfg, item.key, item.default) or [])
    return getattr(cfg, item.key, item.default)


def settings_snapshot(cfg: Config) -> dict[str, Any]:
    """构造 ``GET /admin/settings`` 的响应（按分组 + 扁平两份）。"""
    file_path = config_file_path(cfg.home)
    file_data = load_config_file(cfg.home)
    groups: list[dict[str, Any]] = []
    flat: dict[str, dict[str, Any]] = {}

    for group_key, group_title in GROUP_TITLES.items():
        items: list[dict[str, Any]] = []
        for item in SETTINGS:
            if item.group != group_key:
                continue
            value = _item_value(item, cfg)
            entry: dict[str, Any] = {
                "key": item.key,
                "group": item.group,
                "title": item.title,
                "type": item.type,
                "value": value,
                "default": item.default,
                "source": _source(item.key, cfg, file_data),
                "requires_restart": item.requires_restart,
                "description": item.description,
                "secret": item.secret,
                "env": list(item.env),
            }
            if item.type == "list":
                entry["readonly"] = True
                entry["count"] = len(value or [])
            if item.min is not None:
                entry["min"] = item.min
            if item.max is not None:
                entry["max"] = item.max
            if item.choices:
                entry["choices"] = list(item.choices)
            if item.secret:
                entry["enabled"] = bool(cfg.token)
                entry["has_value"] = bool(cfg.token)
            items.append(entry)
            flat[item.key] = entry
        if items:
            groups.append({"group": group_key, "title": group_title, "items": items})

    return {
        "ok": True,
        "config_file": str(file_path),
        "config_file_exists": file_path.is_file(),
        "groups": groups,
        "settings": flat,
    }


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """原子写 JSON（先写 .tmp 再替换），避免服务崩溃留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _apply_runtime(cfg: Config, item_key: str, value: Any) -> None:
    """把新值同步到当前进程的 Config 对象（不需要重启的项立即生效）。"""
    if item_key == "asr_parallel":
        cfg.asr_parallel = int(value)
        cfg.jobs_parallel_asr = int(value)
    elif item_key == "ffmpeg":
        if value:
            cfg.ffmpeg = Path(str(value)).expanduser()
            cfg.ffmpeg_source = "file"
            sibling = cfg.ffmpeg.with_name(
                "ffprobe.exe" if os.name == "nt" else "ffprobe"
            )
            if sibling.is_file():
                cfg.ffprobe = sibling
                cfg.ffprobe_source = "file"
        else:
            cfg.ffmpeg = None
            cfg.ffmpeg_source = "missing"
    elif item_key == "token":
        cfg.token = str(value or "")
    elif BY_KEY.get(item_key) is not None and BY_KEY[item_key].type == "list":
        setattr(cfg, item_key, list(value or []))
    else:
        setattr(cfg, item_key, value)


def update_settings(cfg: Config, values: dict[str, Any]) -> dict[str, Any]:
    """校验并写入一批设置；返回 applied / requires_restart / warnings。

    ``values`` 为 ``{key: raw_value}``。未知键、类型错误、越界都会抛
    :class:`SettingsError`，此时**一个键都不会写**（先全量校验再落盘）。
    """
    if not isinstance(values, dict) or not values:
        raise SettingsError("没有要保存的设置")

    # 第一遍：全量校验，任何一项不合法都不动文件。
    validated: list[tuple[Any, Any]] = []
    for raw_key, raw_value in values.items():
        item_key = str(raw_key).strip()
        item = BY_KEY.get(item_key)
        if item is None:
            raise SettingsError(
                f"不认识的设置项：{item_key}；可写项见 GET /admin/settings 的 settings 字段"
            )
        try:
            value = coerce_value(item, raw_value)
        except ValueError as exc:
            raise SettingsError(str(exc)) from exc
        validated.append((item, value))

    file_path = config_file_path(cfg.home)
    file_data = load_config_file(cfg.home)

    applied: list[str] = []
    requires_restart: list[str] = []
    warnings: list[str] = []
    overridden: list[str] = []

    for item, value in validated:
        # 值等于默认值就删掉文件里的键，保证「只保存改过的项」。
        if value is None or value == item.default:
            file_data.pop(item.key, None)
        else:
            file_data[item.key] = value

        env_overridden = bool(cfg.env_overrides.get(item.key))
        # 被环境变量覆盖的键只落盘、不改运行时，避免出现「页面显示新值，
        # 实际原生引擎仍按旧环境变量跑」的假象。
        if not env_overridden:
            _apply_runtime(cfg, item.key, value)
        applied.append(item.key)
        if item.requires_restart:
            requires_restart.append(item.key)
        if env_overridden:
            overridden.append(item.key)
            names = " / ".join(item.env)
            warnings.append(
                f"{item.title}（{item.key}）已由环境变量 {names} 覆盖："
                "写入已保存，但当前进程和下次启动仍以环境变量为准。"
            )

    try:
        _atomic_write(file_path, file_data)
    except OSError as exc:
        raise SettingsError(f"写入配置文件失败：{file_path}（{exc}）") from exc

    # 路径与数值变化后，把最新值写回原生引擎读取的环境变量。
    try:
        cfg.export_env()
    except Exception:  # noqa: BLE001 - 环境变量写回失败不影响保存结果
        logger.exception("保存设置后写回环境变量失败")

    return {
        "ok": True,
        "applied": applied,
        "requires_restart": requires_restart,
        "warnings": warnings,
        "overridden_by_env": overridden,
        "config_file": str(file_path),
    }


__all__ = ["SettingsError", "settings_snapshot", "update_settings"]
