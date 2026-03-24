"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and merge it with the default config if needed."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    default_path = CODE_ROOT / "configs" / "default.yaml"
    if config_path.resolve() == default_path.resolve() or not default_path.exists():
        return config

    with default_path.open("r", encoding="utf-8") as handle:
        default_cfg = yaml.safe_load(handle) or {}
    return merge_dicts(default_cfg, config)


def resolve_path(path_value: str | Path | None) -> Path | None:
    """Resolve a path relative to the code root."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return CODE_ROOT / path

