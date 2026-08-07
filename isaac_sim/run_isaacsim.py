#!/usr/bin/env python3
"""Run the multi-camera RGB-D, GT instance and ROS publisher forever.

This follows the standalone structure used by ScenePredictor's YOLOE branch:
create SimulationApp first, then import Omniverse/local sensor modules, create
USD cameras and annotators, warm up the renderer, and publish inside a simple
``simulation_app.is_running()`` loop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

WARMUP_FRAMES = 30
CAMERA_READY_MAX_ATTEMPTS = 90
CAMERA_READY_LOG_INTERVAL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish Isaac Sim RGB-D and instance ground truth."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/isaac.yaml"))
    parser.add_argument("--scene", choices=("static", "dynamic", "occlusion"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--rgbd-hz", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--motion-speed-scale", type=float)
    parser.add_argument("--profile-every", type=int, default=120)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.config).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    if args.scene is not None:
        raw["scene"] = args.scene
    if args.headless:
        raw["headless"] = True
    if args.width is not None:
        raw["width"] = args.width
    if args.height is not None:
        raw["height"] = args.height
    if args.rgbd_hz is not None:
        raw["rgbd_hz"] = args.rgbd_hz
    if args.duration is not None:
        raw["duration_seconds"] = args.duration
    if args.motion_speed_scale is not None:
        raw["motion_speed_scale"] = args.motion_speed_scale
    raw["_config_path"] = str(path)
    return raw


def _validate(config: dict[str, Any], profile_every: int) -> None:
    if int(config["width"]) <= 0 or int(config["height"]) <= 0:
        raise ValueError("width and height must be positive.")
    if float(config["rgbd_hz"]) <= 0.0:
        raise ValueError("rgbd_hz must be positive.")
    if float(config.get("duration_seconds", 0.0)) < 0.0:
        raise ValueError("duration_seconds must be non-negative.")
    if float(config.get("motion_speed_scale", 1.0)) <= 0.0:
        raise ValueError("motion_speed_scale must be positive.")
    if profile_every <= 0:
        raise ValueError("profile_every must be positive.")
    cameras = config.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("configs/isaac.yaml must contain at least one camera.")


def _wait_until_cameras_ready(
    simulation_app: Any,
    cameras: list[Any],
    rig: Any,
    capture_all_cameras: Any,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, CAMERA_READY_MAX_ATTEMPTS + 1):
        try:
            frames = capture_all_cameras(cameras, rig)
        except RuntimeError as error:
            last_error = error
            if attempt == 1 or attempt % CAMERA_READY_LOG_INTERVAL == 0:
                print(
                    "Camera annotators are not ready "
                    f"({attempt}/{CAMERA_READY_MAX_ATTEMPTS}): {error}",
                    flush=True,
                )
            simulation_app.update()
            continue
        summary = ", ".join(
            f"{name}: rgb={frame.rgb.shape}, depth={frame.depth_m.shape}, "
            f"instances={len(frame.instance_metadata)}"
            for name, frame in frames.items()
        )
        print(
            f"Camera annotators ready after {attempt} attempt(s): {summary}",
            flush=True,
        )
        return
    raise RuntimeError(
        "Camera annotators remained empty after "
        f"{CAMERA_READY_MAX_ATTEMPTS} attempts. Last error: {last_error}"
    ) from last_error


def _print_profile(samples: list[dict[str, float]]) -> None:
    import numpy as np

    print("[ISAAC PROFILE]", flush=True)
    for key in ("capture_ms", "ros_ms", "pipeline_ms", "actual_period_ms"):
        values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
        if key == "actual_period_ms":
            values = values[values > 0.0]
        if values.size == 0:
            continue
        print(
            f"  {key:<18} mean={values.mean():8.3f} "
            f"p95={np.percentile(values, 95):8.3f} max={values.max():8.3f}",
            flush=True,
        )
    periods = np.asarray(
        [sample["actual_period_ms"] for sample in samples if sample["actual_period_ms"] > 0.0],
        dtype=np.float64,
    )
    if periods.size:
        print(f"  achieved_hz        {1000.0 / periods.mean():8.3f}", flush=True)


def main() -> None:
    args = parse_args()
    config = _load_config(args)
    _validate(config, args.profile_every)

    os.environ.setdefault("ROS_DISTRO", "jazzy")
    os.environ.setdefault("ROS_DOMAIN_ID", str(config.get("ros_domain_id", 117)))
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    # SimulationApp must be created before importing omni, pxr, Replicator, or
    # local modules that import them.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(config.get("headless", False)),
            "renderer": "RaytracedLighting",
            "width": int(config["width"]),
            "height": int(config["height"]),
        }
    )

    publisher = None
    timeline = None
    try:
        import omni.timeline
        import omni.usd

        from camera_settings import (
            CameraRigConfig,
            CameraSpec,
            capture_all_cameras,
            create_cameras,
        )
        from ros_camera_publisher import RosCameraPublisher
        from scene_settings import build_scene

        stage = omni.usd.get_context().get_stage()
        controller = build_scene(
            stage,
            scene_mode=str(config["scene"]),
            speed_scale=float(config.get("motion_speed_scale", 1.0)),
        )
        specs = tuple(
            CameraSpec(
                name=str(item["name"]),
                prim_path=str(item["prim_path"]),
                position_world=tuple(float(v) for v in item["position"]),
                look_at_world=tuple(float(v) for v in item["look_at"]),
                focal_length_mm=float(item.get("focal_length_mm", 18.0)),
                horizontal_aperture_mm=float(item.get("horizontal_aperture_mm", 20.955)),
                near_m=float(item.get("near_m", 0.05)),
                far_m=float(item.get("far_m", 10.0)),
            )
            for item in config["cameras"]
        )
        rig = CameraRigConfig(
            width=int(config["width"]),
            height=int(config["height"]),
            camera_specs=specs,
            world_frame_id=str(config.get("world_frame", "world")),
            max_depth_m=float(config.get("max_depth_m", 6.0)),
        )
        cameras = create_cameras(stage, rig)
        noise = config.get("depth_noise", {})
        publisher = RosCameraPublisher(
            cameras,
            rig,
            depth_noise_enabled=bool(noise.get("enabled", False)),
            depth_noise_sigma_m=float(noise.get("gaussian_sigma_m", 0.0)),
            depth_dropout_probability=float(noise.get("dropout_probability", 0.0)),
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        print(
            f"Warming up the renderer for {WARMUP_FRAMES} frames.",
            flush=True,
        )
        for _ in range(WARMUP_FRAMES):
            simulation_app.update()
        _wait_until_cameras_ready(
            simulation_app,
            cameras,
            rig,
            capture_all_cameras,
        )

        hz = float(config["rgbd_hz"])
        duration = float(config.get("duration_seconds", 0.0))
        publish_period = 1.0 / hz
        start_time = time.perf_counter()
        next_publish_time = start_time
        last_publish_time: float | None = None
        published_frames = 0
        profile_samples: list[dict[str, float]] = []
        print(
            "Running: "
            f"scene={config['scene']}, cameras={[spec.name for spec in specs]}, "
            f"resolution={rig.width}x{rig.height}, rgbd_hz={hz}, "
            f"duration={'forever' if duration == 0.0 else duration}, "
            f"motion={controller.description()}, "
            "topics=rgb+depth+camera_info+instance_gt+metadata",
            flush=True,
        )

        while simulation_app.is_running():
            now = time.perf_counter()
            elapsed = now - start_time
            if duration > 0.0 and elapsed >= duration:
                print(f"Configured duration reached: {duration:.3f}s", flush=True)
                break

            controller.update(elapsed)
            # Update/render after changing USD transforms, exactly as in the
            # ScenePredictor standalone loop.
            simulation_app.update()
            now = time.perf_counter()
            if now < next_publish_time:
                continue
            next_publish_time = max(next_publish_time + publish_period, now)

            pipeline_start = time.perf_counter()
            frames = capture_all_cameras(cameras, rig)
            capture_end = time.perf_counter()
            publisher.publish(frames)
            ros_end = time.perf_counter()
            actual_period_ms = (
                0.0
                if last_publish_time is None
                else 1000.0 * (ros_end - last_publish_time)
            )
            last_publish_time = ros_end
            profile_samples.append(
                {
                    "capture_ms": 1000.0 * (capture_end - pipeline_start),
                    "ros_ms": 1000.0 * (ros_end - capture_end),
                    "pipeline_ms": 1000.0 * (ros_end - pipeline_start),
                    "actual_period_ms": actual_period_ms,
                }
            )
            published_frames += 1
            if published_frames % max(1, int(round(hz))) == 0:
                counts = ", ".join(
                    f"{name}={len(frame.instance_metadata)}"
                    for name, frame in frames.items()
                )
                print(
                    f"frame={published_frames:06d} visible_instances[{counts}]",
                    flush=True,
                )
            if len(profile_samples) >= args.profile_every:
                _print_profile(profile_samples)
                profile_samples.clear()

    except KeyboardInterrupt:
        print("Interrupted by the user.", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if publisher is not None:
            publisher.shutdown()
        if timeline is not None:
            try:
                timeline.stop()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
