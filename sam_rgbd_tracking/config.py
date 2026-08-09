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


def load_config(
    path: str | Path,
    *,
    tracker: str | None = None,
    efficient_tam_execution_mode: str | None = None,
) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    tracker_cfg = data.setdefault("tracker", {})
    if tracker is not None:
        tracker_cfg["backend"] = tracker

    if efficient_tam_execution_mode is not None:
        mode = str(efficient_tam_execution_mode).strip().lower()
        if mode not in {"sequential", "fixed_batch"}:
            raise ValueError(
                "efficient_tam_execution_mode must be 'sequential' or "
                f"'fixed_batch', got {efficient_tam_execution_mode!r}"
            )
        tracker_cfg.setdefault("efficient_tam", {})["execution_mode"] = mode

    return Config(data)
