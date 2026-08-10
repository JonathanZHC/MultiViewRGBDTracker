from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config


@dataclass(frozen=True, slots=True)
class SlotSpec:
    slot_index: int
    track_id: int
    semantic_label: str
    class_slot: int


def _dict_like(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return value.as_dict()
    if isinstance(value, dict):
        return dict(value)
    return {}


def prompt_capacities(config) -> list[tuple[str, int]]:
    """Return ordered ``(semantic_label, capacity)`` pairs.

    Preferred compact YAML format::

        detector:
          prompts:
            - [ball, 2]
            - [red and white can, 1]

    The legacy split format (plain string prompts plus
    ``max_instances_per_class``) is still accepted so older configs keep working.
    A mapping form is also accepted for readability, e.g.
    ``{name: ball, max_instances: 2}``.
    """

    raw_prompts = list(config.detector.get("prompts", []))
    legacy = _dict_like(config.detector.get("max_instances_per_class", {}))
    specs: list[tuple[str, int]] = []
    seen: set[str] = set()

    for item in raw_prompts:
        label: str
        capacity: int

        if isinstance(item, str):
            label = item.strip()
            capacity = int(legacy.get(label, 1))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            label = str(item[0]).strip()
            capacity = int(item[1])
        elif isinstance(item, dict):
            raw_label = item.get("name", item.get("text", item.get("prompt")))
            raw_capacity = item.get(
                "max_instances",
                item.get("capacity", item.get("count", 1)),
            )
            if raw_label is None:
                raise ValueError(
                    "detector.prompts mapping entries need name/text/prompt"
                )
            label = str(raw_label).strip()
            capacity = int(raw_capacity)
        else:
            raise ValueError(
                "Each detector.prompts entry must be a string, [label, capacity], "
                "or a mapping such as {name: ball, max_instances: 2}. "
                f"Got {item!r}."
            )

        if not label:
            raise ValueError("detector.prompts contains an empty semantic label")
        if capacity <= 0:
            raise ValueError(
                f"detector prompt capacity for {label!r} must be > 0, got {capacity}"
            )
        if label in seen:
            raise ValueError(
                f"detector.prompts contains duplicate semantic class {label!r}; "
                "put its capacity on the same entry instead"
            )
        seen.add(label)
        specs.append((label, capacity))

    if not specs:
        raise ValueError("detector.prompts must contain at least one semantic class")
    return specs


def class_capacities(config) -> dict[str, int]:
    return dict(prompt_capacities(config))


def build_slot_layout(config) -> list[SlotSpec]:
    layout: list[SlotSpec] = []
    for label, capacity in prompt_capacities(config):
        for class_slot in range(capacity):
            slot_index = len(layout)
            layout.append(
                SlotSpec(
                    slot_index=slot_index,
                    track_id=slot_index + 1,
                    semantic_label=label,
                    class_slot=class_slot,
                )
            )
    return layout


def slot_layout_key(config) -> tuple[str, ...]:
    return tuple(
        f"{label}:{capacity}" for label, capacity in prompt_capacities(config)
    )


def object_slots_per_view(config) -> int:
    return sum(capacity for _, capacity in prompt_capacities(config))
