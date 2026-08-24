from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a mapping")
    return cast(dict[str, Any], config)
