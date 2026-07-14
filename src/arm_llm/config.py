# armenian_llm_training/src/arm_llm/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file.

    We keep this simple now.
    Later we can replace it with Hydra/OmegaConf if the project grows.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML dictionary: {config_path}")

    return config


def require_key(config: dict[str, Any], dotted_key: str) -> Any:
    """
    Read required nested config values.

    Example:
        require_key(config, "data.train_file")
    """
    current: Any = config

    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current: #if the part is not in the dictionary or the current is not a dictionary, raise a KeyError
            raise KeyError(f"Missing required config key: {dotted_key}")
        current = current[part]

    return current


