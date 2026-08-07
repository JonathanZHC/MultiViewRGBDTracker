#!/usr/bin/env python3
"""NVIDIA Replicator/Warp camera corruption kernels.

The kernels are attached to Replicator annotators.  When the source annotator
uses ``device="cuda"``, corruption stays on the GPU until ``get_data()`` is
converted to NumPy for the current Python ROS publisher.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import warp as wp


@wp.kernel
def rgb_camera_corruption_wp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    noise_std_255: float,
    exposure_fraction: float,
    seed: int,
):
    """Apply one exposure multiplier and independent RGB Gaussian noise.

    Replicator supplies a launch-specific seed to the kernel.  Every pixel uses
    the same exposure value for a frame, while channel noise is independent.
    Alpha is copied unchanged when the input annotator provides RGBA.
    """

    row, col = wp.tid()
    height = data_in.shape[0]
    width = data_in.shape[1]
    pixel_count = height * width
    pixel_id = row * width + col

    exposure_state = wp.rand_init(seed, pixel_count * 4 + 17)
    exposure_random = wp.randf(exposure_state)
    exposure = 1.0 + (2.0 * exposure_random - 1.0) * exposure_fraction

    state_r = wp.rand_init(seed, pixel_id + pixel_count * 0)
    state_g = wp.rand_init(seed, pixel_id + pixel_count * 1)
    state_b = wp.rand_init(seed, pixel_id + pixel_count * 2)

    red = (
        wp.float32(data_in[row, col, 0]) * exposure
        + noise_std_255 * wp.randn(state_r)
    )
    green = (
        wp.float32(data_in[row, col, 1]) * exposure
        + noise_std_255 * wp.randn(state_g)
    )
    blue = (
        wp.float32(data_in[row, col, 2]) * exposure
        + noise_std_255 * wp.randn(state_b)
    )

    data_out[row, col, 0] = wp.uint8(wp.clamp(red, 0.0, 255.0))
    data_out[row, col, 1] = wp.uint8(wp.clamp(green, 0.0, 255.0))
    data_out[row, col, 2] = wp.uint8(wp.clamp(blue, 0.0, 255.0))

    if data_out.shape[2] > 3:
        data_out[row, col, 3] = data_in[row, col, 3]


@wp.func
def _is_depth_edge(
    data_in: wp.array2d(dtype=wp.float32),
    row: int,
    col: int,
    center: float,
    threshold: float,
) -> bool:
    """Return true when a valid 4-neighbour depth jump exceeds threshold."""

    height = data_in.shape[0]
    width = data_in.shape[1]
    edge = False

    if col > 0:
        value = data_in[row, col - 1]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or (wp.abs(value - center) > threshold)

    if col + 1 < width:
        value = data_in[row, col + 1]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or (wp.abs(value - center) > threshold)

    if row > 0:
        value = data_in[row - 1, col]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or (wp.abs(value - center) > threshold)

    if row + 1 < height:
        value = data_in[row + 1, col]
        if wp.isfinite(value) and value > 0.0:
            edge = edge or (wp.abs(value - center) > threshold)

    return edge


@wp.kernel
def depth_camera_corruption_wp(
    data_in: wp.array2d(dtype=wp.float32),
    data_out: wp.array2d(dtype=wp.float32),
    noise_base_m: float,
    noise_quadratic: float,
    quantization_m: float,
    random_dropout_probability: float,
    edge_dropout_probability: float,
    edge_threshold_m: float,
    seed: int,
):
    """Apply a fused RGB-D style depth corruption model in one GPU kernel.

    The model contains distance-dependent Gaussian noise, quantization,
    independent dropout and depth-discontinuity dropout.  Invalid output is NaN
    so the existing point-cloud validity mask removes it.
    """

    row, col = wp.tid()
    width = data_in.shape[1]
    pixel_id = row * width + col
    clean_depth = data_in[row, col]

    valid = wp.isfinite(clean_depth) and clean_depth > 0.0
    if not valid:
        data_out[row, col] = wp.float32(wp.NAN)
    else:
        state = wp.rand_init(seed, pixel_id)

        sigma = noise_base_m + noise_quadratic * clean_depth * clean_depth
        noisy_depth = clean_depth + sigma * wp.randn(state)

        if quantization_m > 0.0:
            noisy_depth = (
                wp.round(noisy_depth / quantization_m)
                * quantization_m
            )

        random_dropout = (
            wp.randf(state) < random_dropout_probability
        )
        edge = _is_depth_edge(
            data_in,
            row,
            col,
            clean_depth,
            edge_threshold_m,
        )
        edge_dropout = (
            edge
            and wp.randf(state) < edge_dropout_probability
        )

        output_valid = (
            wp.isfinite(noisy_depth)
            and noisy_depth > 0.0
            and not random_dropout
            and not edge_dropout
        )

        if output_valid:
            data_out[row, col] = noisy_depth
        else:
            data_out[row, col] = wp.float32(wp.NAN)


def enable_replicator_warp_runtime() -> None:
    """Enable Replicator Python/Warp augmentation graph execution."""

    import carb.settings

    carb.settings.get_settings().set_bool(
        "/app/omni.graph.scriptnode/opt_in",
        True,
    )
    wp.init()


def make_rgb_augmentation(config: Any, seed: int):
    """Create a Replicator augmentation wrapping the RGB Warp kernel."""

    import omni.replicator.core as rep

    return rep.annotators.Augmentation.from_function(
        rgb_camera_corruption_wp,
        noise_std_255=float(config.rgb_noise_std_255),
        exposure_fraction=float(config.exposure_fraction),
        seed=int(seed),
    )


def make_depth_augmentation(config: Any, seed: int):
    """Create a Replicator augmentation wrapping the depth Warp kernel."""

    import omni.replicator.core as rep

    return rep.annotators.Augmentation.from_function(
        depth_camera_corruption_wp,
        noise_base_m=float(config.depth_noise_base_m),
        noise_quadratic=float(config.depth_noise_quadratic),
        quantization_m=float(config.depth_quantization_m),
        random_dropout_probability=float(
            config.random_dropout_probability
        ),
        edge_dropout_probability=float(
            config.edge_dropout_probability
        ),
        edge_threshold_m=float(config.edge_threshold_m),
        seed=int(seed),
    )


def corruption_config_to_dict(config: Any) -> dict[str, Any]:
    """Serialize a dataclass-like corruption configuration."""

    try:
        return asdict(config)
    except TypeError:
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
