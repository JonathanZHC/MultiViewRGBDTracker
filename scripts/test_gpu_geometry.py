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
from sam_rgbd_tracking.gpu_geometry import GeometrySamples, GPUSparseGeometryBackend


@dataclass
class Record:
    view_index: int


def cpu_reference(frame, sample, voxelizer):
    k = frame.intrinsics

    x_ray = (
        np.arange(k.width, dtype=np.float32) - np.float32(k.cx)
    ) / np.float32(k.fx)

    y_ray = (
        np.arange(k.height, dtype=np.float32) - np.float32(k.cy)
    ) / np.float32(k.fy)

    pc = np.empty((len(sample.z), 3), dtype=np.float32)
    pc[:, 0] = x_ray[sample.xs] * sample.z
    pc[:, 1] = y_ray[sample.ys] * sample.z
    pc[:, 2] = sample.z

    T = np.asarray(frame.world_from_camera, dtype=np.float32)

    pw = np.ascontiguousarray(
        pc @ T[:3, :3].T + T[:3, 3][None, :],
        dtype=np.float32,
    )

    prepared = voxelizer.prepare_points(
        pw,
        sample.colors_rgb,
    )

    return pc, pw, prepared


def print_stats(
    name: str,
    values: np.ndarray,
) -> None:
    print()
    print(name)
    print(f"  mean   : {np.mean(values):.3f} ms")
    print(f"  median : {np.median(values):.3f} ms")
    print(f"  p95    : {np.percentile(values, 95):.3f} ms")
    print(f"  p99    : {np.percentile(values, 99):.3f} ms")
    print(f"  min    : {np.min(values):.3f} ms")
    print(f"  max    : {np.max(values):.3f} ms")


def benchmark_compute(
    backend,
    records,
    frames,
    samples,
    warmup: int = 50,
    iterations: int = 500,
) -> None:
    print()
    print("Steady-state compute breakdown")
    print(f"  warmup     : {warmup}")
    print(f"  iterations : {iterations}")

    # ---------------------------------------------------------
    # Safe warmup.
    #
    # compute() reuses persistent pinned host staging buffers,
    # so synchronize before those buffers are overwritten by the
    # next iteration.
    # ---------------------------------------------------------
    for _ in range(warmup):
        backend.compute(
            records,
            frames,
            samples,
        )
        torch.cuda.synchronize()

    submit_ms = np.empty(
        iterations,
        dtype=np.float64,
    )

    wait_ms = np.empty(
        iterations,
        dtype=np.float64,
    )

    total_ms = np.empty(
        iterations,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Split wall-clock latency into:
    #
    # submit:
    #   CPU staging
    #   bookkeeping
    #   H2D enqueue
    #   CUDA/Warp/Torch dispatch
    #
    # wait:
    #   outstanding GPU execution after compute() returns
    #
    # total:
    #   complete synchronized compute latency
    # ---------------------------------------------------------
    for i in range(iterations):
        torch.cuda.synchronize()

        t0 = time.perf_counter()

        backend.compute(
            records,
            frames,
            samples,
        )

        t1 = time.perf_counter()

        torch.cuda.synchronize()

        t2 = time.perf_counter()

        submit_ms[i] = (
            t1 - t0
        ) * 1000.0

        wait_ms[i] = (
            t2 - t1
        ) * 1000.0

        total_ms[i] = (
            t2 - t0
        ) * 1000.0

    print_stats(
        "CPU submit / staging time",
        submit_ms,
    )

    print_stats(
        "GPU outstanding wait time",
        wait_ms,
    )

    print_stats(
        "Total synchronized compute time",
        total_ms,
    )

    mean_total = float(
        np.mean(total_ms)
    )

    print()
    print(
        "Geometry-only equivalent rate: "
        f"{1000.0 / mean_total:.1f} Hz"
    )


def benchmark_gpu_stages(
    backend,
    records,
    frames,
    samples,
    warmup: int = 50,
    iterations: int = 500,
) -> None:
    print()
    print("Per-stage CUDA benchmark")
    print(f"  warmup     : {warmup}")
    print(f"  iterations : {iterations}")

    if not hasattr(
        backend,
        "enable_gpu_profile",
    ):
        raise RuntimeError(
            "GPUSparseGeometryBackend does not provide "
            "enable_gpu_profile(). Update gpu_geometry.py first."
        )

    if not hasattr(
        backend,
        "gpu_profile_ms",
    ):
        raise RuntimeError(
            "GPUSparseGeometryBackend does not provide "
            "gpu_profile_ms(). Update gpu_geometry.py first."
        )

    backend.enable_gpu_profile(True)

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

    stage_times = {
        name: np.empty(
            iterations,
            dtype=np.float64,
        )
        for name in stage_names
    }

    # ---------------------------------------------------------
    # Profile-enabled warmup.
    #
    # This also ensures CUDA events themselves are no longer
    # first-use operations when measurement starts.
    # ---------------------------------------------------------
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
                "GPU profiling is enabled but "
                "gpu_profile_ms() returned None."
            )

    # ---------------------------------------------------------
    # CUDA-event timing.
    #
    # Each iteration synchronizes after compute() so all recorded
    # CUDA events are complete before elapsed_time() is queried.
    # ---------------------------------------------------------
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
                "gpu_profile_ms() returned None."
            )

        for name in stage_names:
            stage_times[name][i] = float(
                timing[name]
            )

    print()
    print("GPU stage breakdown")
    print()

    for name in stage_names:
        x = stage_times[name]

        print(
            f"{name:14s} "
            f"mean={x.mean():7.3f} ms  "
            f"median={np.median(x):7.3f} ms  "
            f"p95={np.percentile(x, 95):7.3f} ms  "
            f"p99={np.percentile(x, 99):7.3f} ms  "
            f"min={x.min():7.3f} ms  "
            f"max={x.max():7.3f} ms"
        )

    total_gpu = np.zeros(
        iterations,
        dtype=np.float64,
    )

    for name in stage_names:
        total_gpu += stage_times[name]

    print()
    print(
        f"{'GPU total':14s} "
        f"mean={total_gpu.mean():7.3f} ms  "
        f"median={np.median(total_gpu):7.3f} ms  "
        f"p95={np.percentile(total_gpu, 95):7.3f} ms  "
        f"p99={np.percentile(total_gpu, 99):7.3f} ms  "
        f"min={total_gpu.min():7.3f} ms  "
        f"max={total_gpu.max():7.3f} ms"
    )

    # ---------------------------------------------------------
    # Fraction of mean GPU time spent in each stage.
    # Useful for immediately spotting the dominant operation.
    # ---------------------------------------------------------
    mean_gpu = float(
        total_gpu.mean()
    )

    print()
    print("Mean GPU time distribution")

    for name in stage_names:
        mean_stage = float(
            stage_times[name].mean()
        )

        fraction = (
            100.0 * mean_stage / mean_gpu
            if mean_gpu > 0.0
            else 0.0
        )

        print(
            f"  {name:14s}: "
            f"{mean_stage:7.3f} ms "
            f"({fraction:5.1f}%)"
        )

    backend.enable_gpu_profile(False)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this validation"
        )

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

    rng = np.random.default_rng(9)

    h = 480
    w = 640

    frames = []
    samples = []
    records = []

    # ---------------------------------------------------------
    # Synthetic test data
    #
    # 2 cameras
    # × 3 instances per camera
    # × 4096 sparse points per instance
    #
    # Total:
    #
    #   2 × 3 × 4096
    #   = 24576 sparse points
    # ---------------------------------------------------------
    for view in range(2):
        K = CameraIntrinsics(
            615.3,
            614.8,
            319.5,
            239.5,
            w,
            h,
        )

        T = np.eye(
            4,
            dtype=np.float32,
        )

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

        frames.append(
            frame
        )

        for _ in range(3):
            n = 4096

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
            ).astype(
                np.float32
            )

            colors = rng.integers(
                0,
                256,
                (n, 3),
                dtype=np.uint8,
            )

            records.append(
                Record(view)
            )

            samples.append(
                GeometrySamples(
                    ys=ys,
                    xs=xs,
                    z=z,
                    colors_rgb=colors,
                )
            )

    # =========================================================
    # Correctness validation
    # =========================================================

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

    max_camera = 0.0
    max_world = 0.0

    for i, (
        record,
        sample,
        item,
    ) in enumerate(
        zip(
            records,
            samples,
            got,
        )
    ):
        pc, pw, prepared = cpu_reference(
            frames[record.view_index],
            sample,
            voxelizer,
        )

        (
            coords,
            keys,
            points,
            colors,
            bmin,
            bmax,
        ) = prepared

        max_camera = max(
            max_camera,
            float(
                np.max(
                    np.abs(
                        pc
                        - item.points_camera
                    )
                )
            ),
        )

        max_world = max(
            max_world,
            float(
                np.max(
                    np.abs(
                        pw
                        - item.points_world
                    )
                )
            ),
        )

        gpu_keys = (
            item.voxel_keys.astype(
                np.uint64,
                copy=False,
            )
        )

        if not np.array_equal(
            keys,
            gpu_keys,
        ):
            # GPU FMA can move a point lying extremely close
            # to a voxel boundary. Allow only a tiny discrepancy.
            overlap = len(
                np.intersect1d(
                    keys,
                    gpu_keys,
                )
            )

            ratio = (
                overlap
                / max(
                    1,
                    len(keys),
                )
            )

            if ratio < 0.999:
                raise AssertionError(
                    f"record {i}: "
                    f"voxel-key overlap only "
                    f"{ratio:.6f}"
                )

        if (
            len(item.voxel_points)
            != len(item.voxel_keys)
        ):
            raise AssertionError(
                f"record {i}: "
                "voxel point/key length mismatch"
            )

        if (
            item.voxel_bbox_min is None
            or item.voxel_bbox_max is None
        ):
            raise AssertionError(
                f"record {i}: "
                "missing voxel bbox"
            )

        if (
            len(colors)
            != len(item.voxel_colors)
        ):
            raise AssertionError(
                f"record {i}: "
                "voxel color count mismatch"
            )

    print(
        "Fused Warp GPU geometry validation OK on "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "max camera XYZ abs error: "
        f"{max_camera:.3e} m"
    )

    print(
        "max world  XYZ abs error: "
        f"{max_world:.3e} m"
    )

    print(
        "staging capacity: "
        f"{backend.capacity} sparse points"
    )

    # =========================================================
    # End-to-end compute benchmark
    # =========================================================

    benchmark_compute(
        backend,
        records,
        frames,
        samples,
        warmup=50,
        iterations=500,
    )

    # =========================================================
    # Detailed GPU stage benchmark
    # =========================================================

    benchmark_gpu_stages(
        backend,
        records,
        frames,
        samples,
        warmup=50,
        iterations=500,
    )


if __name__ == "__main__":
    main()