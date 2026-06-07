"""Load, mutate, and write CloudAI TOML scenario configs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a CloudAI TOML scenario config into a nested dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_value(config: dict[str, Any], dotted_key: str) -> Any:
    """Read a nested value via a dotted key path, e.g. 'serving.batch_size'."""
    node: Any = config
    for part in dotted_key.split("."):
        node = node[part]
    return node


def set_value(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Return a copy of `config` with `dotted_key` set to `value`."""
    result = _deep_copy(config)
    node = result
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return result


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def to_toml(config: dict[str, Any]) -> str:
    """Serialize a config dict back to TOML text (handles the nesting we use)."""
    lines: list[str] = []
    _write_table(config, [], lines)
    return "\n".join(lines) + "\n"


def _write_table(table: dict[str, Any], path: list[str], lines: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    subtables = {k: v for k, v in table.items() if isinstance(v, dict)}

    if path:
        lines.append(f"[{'.'.join(path)}]")
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    if scalars and subtables:
        lines.append("")
    for i, (key, value) in enumerate(subtables.items()):
        _write_table(value, path + [key], lines)
        if i < len(subtables) - 1:
            lines.append("")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def write_config(config: dict[str, Any], path: Path | str) -> Path:
    """Write a config dict to `path` as TOML and return the path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_toml(config))
    return out_path


def derive_config(
    base_path: Path | str,
    overrides: dict[str, Any],
    out_path: Path | str,
) -> Path:
    """Load `base_path`, apply dotted-key `overrides`, write to `out_path`."""
    config = load_config(base_path)
    for dotted_key, value in overrides.items():
        config = set_value(config, dotted_key, value)
    return write_config(config, out_path)
