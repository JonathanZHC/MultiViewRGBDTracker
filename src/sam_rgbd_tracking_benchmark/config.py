from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


class ConfigNode(Mapping[str, Any]):
    """Read-only attribute-access wrapper around a nested configuration mapping."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data: dict[str, Any] = {}
        for key, value in data.items():
            self._data[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get_path(self, path: str, default: Any = None) -> Any:
        current: Any = self
        for key in path.split("."):
            if not isinstance(current, ConfigNode) or key not in current:
                return default
            current = current[key]
        return current

    def to_dict(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, ConfigNode):
                return {key: unwrap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            return copy.deepcopy(value)

        return unwrap(self)


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got: {raw}")
    key, value = raw.split("=", 1)
    return key, yaml.safe_load(value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ConfigNode:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for override in overrides or []:
        key, value = _parse_override(override)
        _set_nested(data, key, value)
    return ConfigNode(data)
