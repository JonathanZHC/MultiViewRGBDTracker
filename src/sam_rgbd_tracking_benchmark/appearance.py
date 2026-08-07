from __future__ import annotations

import numpy as np


def masked_color_embedding(
    rgb: np.ndarray,
    mask: np.ndarray,
    bins_per_channel: int = 8,
) -> np.ndarray | None:
    """Return a compact normalized RGB histogram for keyframe association.

    This is intentionally lightweight and model-free. It is not a semantic CLIP
    embedding; it provides stable appearance evidence for separating instances
    with the same text label. The detector adapter can later replace it with a
    learned embedding without changing the association interface.
    """
    pixels = np.asarray(rgb, dtype=np.uint8)[np.asarray(mask, dtype=bool)]
    if pixels.shape[0] < 8:
        return None
    bins = max(2, int(bins_per_channel))
    quantized = np.minimum((pixels.astype(np.int32) * bins) // 256, bins - 1)
    indices = quantized[:, 0] * bins * bins + quantized[:, 1] * bins + quantized[:, 2]
    histogram = np.bincount(indices, minlength=bins**3).astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    if norm <= 0:
        return None
    return histogram / norm


def cosine_distance(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.0
    a = np.asarray(first, dtype=np.float32).reshape(-1)
    b = np.asarray(second, dtype=np.float32).reshape(-1)
    if a.size != b.size or a.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def blend_embedding(
    previous: np.ndarray | None,
    current: np.ndarray | None,
    alpha: float = 0.25,
) -> np.ndarray | None:
    if current is None:
        return previous
    if previous is None or previous.shape != current.shape:
        return current.copy()
    blended = (1.0 - alpha) * previous + alpha * current
    norm = float(np.linalg.norm(blended))
    return blended / norm if norm > 1e-12 else current.copy()
