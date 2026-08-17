#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam_rgbd_tracking.alignment import SharedWorldVoxelizer
from sam_rgbd_tracking.config import load_config
from sam_rgbd_tracking.data_types import CameraIntrinsics, RGBDFrame
from sam_rgbd_tracking.gpu_geometry import (
    GeometrySamples,
    GPUSparseGeometryBackend,
)


@dataclass
class Record:
    view_index: int


def print_stats(
    name: str,
    values: np.ndarray,
) -> None:
    values = np.asarray(values, dtype=np.float64)

    print()
    print(name)
    print(f"  mean   : {values.mean():.3f} ms")
    print(f"  median : {np.median(values):.3f} ms")
    print(f"  p95    : {np.percentile(values, 95):.3f} ms")
    print(f"  p99    : {np.percentile(values, 99):.3f} ms")
    print(f"  min    : {values.min():.3f} ms")
    print(f"  max    : {values.max():.3f} ms")


def make_test_data(
    points_per_record: int = 4096,
    cameras: int = 2,
    records_per_camera: int = 3,
):
    rng = np.random.default_rng(9)

    h = 480
    w = 640

    frames = []
    records = []
    samples = []

    for view in range(cameras):
        K = CameraIntrinsics(
            615.3,
            614.8,
            319.5,
            239.5,
            w,
            h,
        )

        T = np.eye(4, dtype=np.float32)

        T[:3, 3] = np.array(
            [
                0.15 * view,
                -0.03 * view,
                0.2,
            ],
            dtype=np.float32,
        )

        frame = RGBDFrame(
            camera_name=f"camera_{view}",
            frame_index=0,
            timestamp_ns=0,
            rgb=np.zeros(
                (h, w, 3),
                dtype=np.uint8,
            ),
            depth_m=np.ones(
                (h, w),
                dtype=np.float32,
            ),
            intrinsics=K,
            world_from_camera=T,
        )

        frames.append(frame)

        for _ in range(records_per_camera):
            n = points_per_record

            xs = rng.integers(
                40,
                w - 40,
                n,
                dtype=np.int64,
            )

            ys = rng.integers(
                40,
                h - 40,
                n,
                dtype=np.int64,
            )

            z = rng.uniform(
                0.3,
                1.8,
                n,
            ).astype(np.float32)

            colors = rng.integers(
                0,
                256,
                (n, 3),
                dtype=np.uint8,
            )

            records.append(
                Record(
                    view_index=view,
                )
            )

            samples.append(
                GeometrySamples(
                    ys=ys,
                    xs=xs,
                    z=z,
                    colors_rgb=colors,
                )
            )

    return records, frames, samples


def benchmark_compute(
    backend,
    records,
    frames,
    samples,
    *,
    warmup: int,
    iterations: int,
) -> np.ndarray:
    """
    Measure synchronized backend.compute() latency.

    This includes:
      CPU staging/submission
      H2D
      fused Warp geometry
      sorting
      parallel dedup

    It does NOT include materialize().
    """

    for _ in range(warmup):
        backend.compute(
            records,
            frames,
            samples,
        )
        torch.cuda.synchronize()

    times = np.empty(
        iterations,
        dtype=np.float64,
    )

    for i in range(iterations):
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        backend.compute(
            records,
            frames,
            samples,
        )

        torch.cuda.synchronize()

        t1 = time.perf_counter()

        times[i] = (
            t1 - t0
        ) * 1000.0

    return times


def benchmark_materialize(
    backend,
    records,
    frames,
    samples,
    *,
    warmup: int,
    iterations: int,
) -> np.ndarray:
    """
    Measure materialize() alone.

    compute() is fully synchronized BEFORE timing starts,
    therefore the result does not accidentally include
    outstanding geometry GPU work.

    This measures the current GPU -> CPU compatibility
    boundary plus NumPy-side processing.
    """

    for _ in range(warmup):
        pending = backend.compute(
            records,
            frames,
            samples,
        )

        torch.cuda.synchronize()

        backend.materialize(
            pending,
            records,
            frames,
        )

    times = np.empty(
        iterations,
        dtype=np.float64,
    )

    for i in range(iterations):
        pending = backend.compute(
            records,
            frames,
            samples,
        )

        # Important:
        # remove compute GPU execution from materialize timing.
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        backend.materialize(
            pending,
            records,
            frames,
        )

        t1 = time.perf_counter()

        times[i] = (
            t1 - t0
        ) * 1000.0

    return times


def benchmark_full_geometry(
    backend,
    records,
    frames,
    samples,
    *,
    warmup: int,
    iterations: int,
) -> np.ndarray:
    """
    Measure the geometry path exactly as currently used:

        compute()
            ->
        materialize()

    There is deliberately NO synchronization between the two.

    materialize() therefore waits for whatever GPU work it
    actually depends on, exactly like the current tracker.
    """

    for _ in range(warmup):
        pending = backend.compute(
            records,
            frames,
            samples,
        )

        backend.materialize(
            pending,
            records,
            frames,
        )

        torch.cuda.synchronize()

    times = np.empty(
        iterations,
        dtype=np.float64,
    )

    for i in range(iterations):
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        pending = backend.compute(
            records,
            frames,
            samples,
        )

        backend.materialize(
            pending,
            records,
            frames,
        )

        torch.cuda.synchronize()

        t1 = time.perf_counter()

        times[i] = (
            t1 - t0
        ) * 1000.0

    return times


def benchmark_gpu_stages(
    backend,
    records,
    frames,
    samples,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, np.ndarray]:
    """
    Fine-grained CUDA-event profiling of backend.compute().
    """

    stage_names = [
        "h2d",
        "fused_warp",
        "sort_keys",
        "index_select",
        "sort_records",
        "mark_unique",
        "prefix_sum",
        "scatter_unique",
    ]

    if not hasattr(
        backend,
        "enable_gpu_profile",
    ):
        raise RuntimeError(
            "Backend does not provide "
            "enable_gpu_profile()."
        )

    if not hasattr(
        backend,
        "gpu_profile_ms",
    ):
        raise RuntimeError(
            "Backend does not provide "
            "gpu_profile_ms()."
        )

    backend.enable_gpu_profile(True)

    stage_times = {
        name: np.empty(
            iterations,
            dtype=np.float64,
        )
        for name in stage_names
    }

    # Warm up profiling itself.
    for _ in range(warmup):
        backend.compute(
            records,
            frames,
            samples,
        )

        torch.cuda.synchronize()

        timing = backend.gpu_profile_ms()

        if timing is None:
            raise RuntimeError(
                "gpu_profile_ms() returned None"
            )

    for i in range(iterations):
        backend.compute(
            records,
            frames,
            samples,
        )

        torch.cuda.synchronize()

        timing = backend.gpu_profile_ms()

        if timing is None:
            raise RuntimeError(
                "gpu_profile_ms() returned None"
            )

        for name in stage_names:
            stage_times[name][i] = float(
                timing[name]
            )

    backend.enable_gpu_profile(False)

    return stage_times


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this benchmark"
        )

    # ---------------------------------------------------------
    # Benchmark configuration
    # ---------------------------------------------------------

    WARMUP = 50
    ITERATIONS = 500

    CAMERAS = 2
    RECORDS_PER_CAMERA = 3
    POINTS_PER_RECORD = 4096

    TOTAL_RECORDS = (
        CAMERAS
        * RECORDS_PER_CAMERA
    )

    TOTAL_POINTS = (
        TOTAL_RECORDS
        * POINTS_PER_RECORD
    )

    print("=" * 72)
    print("Fused GPU geometry benchmark")
    print("=" * 72)

    print(
        f"GPU               : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"cameras           : "
        f"{CAMERAS}"
    )

    print(
        f"records/camera    : "
        f"{RECORDS_PER_CAMERA}"
    )

    print(
        f"total records     : "
        f"{TOTAL_RECORDS}"
    )

    print(
        f"points/record     : "
        f"{POINTS_PER_RECORD}"
    )

    print(
        f"total sparse pts  : "
        f"{TOTAL_POINTS}"
    )

    print(
        f"warmup            : "
        f"{WARMUP}"
    )

    print(
        f"iterations        : "
        f"{ITERATIONS}"
    )

    # ---------------------------------------------------------
    # Backend
    # ---------------------------------------------------------

    cfg = load_config(
        ROOT / "configs" / "tracking.yaml"
    )

    voxelizer = SharedWorldVoxelizer(
        cfg
    )

    backend = GPUSparseGeometryBackend(
        torch.device("cuda:0"),
        voxelizer,
    )

    records, frames, samples = make_test_data(
        points_per_record=POINTS_PER_RECORD,
        cameras=CAMERAS,
        records_per_camera=RECORDS_PER_CAMERA,
    )

    # ---------------------------------------------------------
    # Initial run / correctness sanity
    # ---------------------------------------------------------

    pending = backend.compute(
        records,
        frames,
        samples,
    )

    got = backend.materialize(
        pending,
        records,
        frames,
    )

    torch.cuda.synchronize()

    if len(got) != TOTAL_RECORDS:
        raise AssertionError(
            f"Expected {TOTAL_RECORDS} records, "
            f"got {len(got)}"
        )

    print()
    print(
        "Initial geometry/materialize sanity: OK"
    )

    print(
        f"backend capacity  : "
        f"{backend.capacity}"
    )

    # =========================================================
    # 1. Compute only
    # =========================================================

    compute_ms = benchmark_compute(
        backend,
        records,
        frames,
        samples,
        warmup=WARMUP,
        iterations=ITERATIONS,
    )

    print()
    print("=" * 72)
    print("1. compute() only")
    print("=" * 72)

    print_stats(
        "backend.compute()",
        compute_ms,
    )

    # =========================================================
    # 2. Materialize only
    # =========================================================

    materialize_ms = benchmark_materialize(
        backend,
        records,
        frames,
        samples,
        warmup=WARMUP,
        iterations=ITERATIONS,
    )

    print()
    print("=" * 72)
    print("2. materialize() only")
    print("=" * 72)

    print_stats(
        "backend.materialize()",
        materialize_ms,
    )

    # =========================================================
    # 3. Current real geometry path
    # =========================================================

    full_ms = benchmark_full_geometry(
        backend,
        records,
        frames,
        samples,
        warmup=WARMUP,
        iterations=ITERATIONS,
    )

    print()
    print("=" * 72)
    print("3. compute() + materialize()")
    print("=" * 72)

    print_stats(
        "current geometry path",
        full_ms,
    )

    # =========================================================
    # 4. GPU stage breakdown
    # =========================================================

    stage_times = benchmark_gpu_stages(
        backend,
        records,
        frames,
        samples,
        warmup=WARMUP,
        iterations=ITERATIONS,
    )

    print()
    print("=" * 72)
    print("4. GPU compute stage breakdown")
    print("=" * 72)
    print()

    stage_names = list(
        stage_times.keys()
    )

    for name in stage_names:
        x = stage_times[name]

        print(
            f"{name:15s} "
            f"mean={x.mean():7.3f} ms  "
            f"median={np.median(x):7.3f} ms  "
            f"p95={np.percentile(x, 95):7.3f} ms  "
            f"p99={np.percentile(x, 99):7.3f} ms  "
            f"max={x.max():7.3f} ms"
        )

    gpu_total = np.zeros(
        ITERATIONS,
        dtype=np.float64,
    )

    for name in stage_names:
        gpu_total += stage_times[name]

    print()

    print(
        f"{'GPU total':15s} "
        f"mean={gpu_total.mean():7.3f} ms  "
        f"median={np.median(gpu_total):7.3f} ms  "
        f"p95={np.percentile(gpu_total, 95):7.3f} ms  "
        f"p99={np.percentile(gpu_total, 99):7.3f} ms  "
        f"max={gpu_total.max():7.3f} ms"
    )

    mean_gpu = float(
        gpu_total.mean()
    )

    print()
    print("Mean GPU time distribution")

    for name in stage_names:
        mean_stage = float(
            stage_times[name].mean()
        )

        percentage = (
            mean_stage
            / mean_gpu
            * 100.0
            if mean_gpu > 0.0
            else 0.0
        )

        print(
            f"  {name:15s}: "
            f"{mean_stage:7.3f} ms "
            f"({percentage:5.1f}%)"
        )

    # =========================================================
    # Summary
    # =========================================================

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)

    compute_median = float(
        np.median(compute_ms)
    )

    materialize_median = float(
        np.median(materialize_ms)
    )

    full_median = float(
        np.median(full_ms)
    )

    print(
        f"compute median              : "
        f"{compute_median:.3f} ms"
    )

    print(
        f"materialize median          : "
        f"{materialize_median:.3f} ms"
    )

    print(
        f"compute + materialize median: "
        f"{full_median:.3f} ms"
    )

    if full_median > 0.0:
        print(
            f"current geometry rate       : "
            f"{1000.0 / full_median:.1f} Hz"
        )

    if compute_median > 0.0:
        print(
            f"materialize / compute ratio : "
            f"{materialize_median / compute_median:.2f}x"
        )


if __name__ == "__main__":
    main()