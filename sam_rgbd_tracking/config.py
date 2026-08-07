from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Tiny read-only-ish nested config with attribute access."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name not in self._data:
            raise AttributeError(name)
        value = self._data[name]
        return Config(value) if isinstance(value, dict) else value

    def get(self, name: str, default: Any = None) -> Any:
        value = self._data.get(name, default)
        return Config(value) if isinstance(value, dict) else value

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)


def load_config(path: str | Path, *, tracker: str | None = None) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if tracker is not None:
        data.setdefault("tracker", {})["backend"] = tracker
    return Config(data)
