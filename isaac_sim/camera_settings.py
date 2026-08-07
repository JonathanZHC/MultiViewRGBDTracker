from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

import numpy as np
import omni.replicator.core as rep
from pxr import Gf, UsdGeom

from camera_math import camera_pose


@dataclass(frozen=True)
class CameraSpec:
    name: str
    prim_path: str
    position_world: tuple[float, float, float]
    look_at_world: tuple[float, float, float]
    focal_length_mm: float = 18.0
    horizontal_aperture_mm: float = 20.955
    near_m: float = 0.05
    far_m: float = 10.0


@dataclass(frozen=True)
class CameraRigConfig:
    width: int
    height: int
    camera_specs: tuple[CameraSpec, ...]
    world_frame_id: str = "world"
    max_depth_m: float = 6.0


@dataclass
class CameraRuntime:
    spec: CameraSpec
    render_product: Any
    rgb_annotator: Any
    depth_annotator: Any
    instance_annotator: Any
    K: np.ndarray
    T_world_from_camera_optical: np.ndarray


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    instance_map: np.ndarray
    instance_metadata: dict[int, dict[str, str]]


def camera_intrinsics(
    width: int,
    height: int,
    focal_length_mm: float,
    horizontal_aperture_mm: float,
) -> tuple[np.ndarray, float]:
    vertical_aperture_mm = (
        horizontal_aperture_mm * float(height) / float(width)
    )
    fx = float(width) * focal_length_mm / horizontal_aperture_mm
    fy = float(height) * focal_length_mm / vertical_aperture_mm
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    matrix = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix, vertical_aperture_mm


def _set_usd_camera_pose(
    prim: Any,
    translation_world: np.ndarray,
    quaternion_usd_xyzw: np.ndarray,
) -> None:
    """Set a USD camera pose using the exact float quaternion type it expects."""
    x, y, z, w = [float(value) for value in quaternion_usd_xyzw]
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*[float(value) for value in translation_world])
    )
    # UsdGeom.Camera's orient op is GfQuatf in Isaac Sim 6.0. Using Quatd is a
    # type error and was the reason the previous implementation shut down.
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(w, Gf.Vec3f(x, y, z))
    )


def create_cameras(stage: Any, rig: CameraRigConfig) -> list[CameraRuntime]:
    if rig.width <= 0 or rig.height <= 0:
        raise ValueError("Camera resolution must be positive.")
    if not rig.camera_specs:
        raise ValueError("At least one camera is required.")

    cameras: list[CameraRuntime] = []
    for spec in rig.camera_specs:
        transform, quaternion_usd_xyzw = camera_pose(
            np.asarray(spec.position_world, dtype=np.float64),
            np.asarray(spec.look_at_world, dtype=np.float64),
        )
        camera = UsdGeom.Camera.Define(stage, spec.prim_path)
        camera.CreateFocalLengthAttr(float(spec.focal_length_mm))
        camera.CreateHorizontalApertureAttr(
            float(spec.horizontal_aperture_mm)
        )
        intrinsics, vertical_aperture_mm = camera_intrinsics(
            rig.width,
            rig.height,
            spec.focal_length_mm,
            spec.horizontal_aperture_mm,
        )
        camera.CreateVerticalApertureAttr(float(vertical_aperture_mm))
        camera.CreateHorizontalApertureOffsetAttr(0.0)
        camera.CreateVerticalApertureOffsetAttr(0.0)
        camera.CreateClippingRangeAttr(
            Gf.Vec2f(float(spec.near_m), float(spec.far_m))
        )
        _set_usd_camera_pose(
            camera.GetPrim(),
            np.asarray(spec.position_world, dtype=np.float64),
            quaternion_usd_xyzw,
        )

        render_product = rep.create.render_product(
            spec.prim_path,
            resolution=(rig.width, rig.height),
            name=f"{spec.name}_render_product",
        )
        rgb_annotator = rep.annotators.get("rgb", device="cuda")
        depth_annotator = rep.annotators.get(
            "distance_to_image_plane",
            device="cuda",
        )
        # Metadata contains idToLabels/idToSemantics, so keep this annotator on
        # CPU. RGB and depth remain CUDA annotators like ScenePredictor.
        instance_annotator = rep.annotators.get(
            "instance_segmentation_fast",
            init_params={"colorize": False},
        )
        rgb_annotator.attach(render_product)
        depth_annotator.attach(render_product)
        instance_annotator.attach(render_product)

        cameras.append(
            CameraRuntime(
                spec=spec,
                render_product=render_product,
                rgb_annotator=rgb_annotator,
                depth_annotator=depth_annotator,
                instance_annotator=instance_annotator,
                K=intrinsics,
                T_world_from_camera_optical=transform,
            )
        )
    return cameras


def _unwrap_data(value: Any, label: str) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        raise RuntimeError(f"{label} annotator returned no data.")
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "numpy"):
        array = value.numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        raise RuntimeError(f"{label} annotator returned empty data.")
    return np.asarray(array)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _semantic_label(value: Any) -> str | None:
    if isinstance(value, dict):
        preferred = value.get("class")
        if preferred is not None:
            return str(preferred).split(",")[-1].strip()
        for candidate in value.values():
            text = str(candidate).strip()
            if text:
                return text.split(",")[-1].strip()
    elif value is not None:
        text = str(value).strip()
        if text:
            return text.split(",")[-1].strip()
    return None


def _normalize_instance_annotation(
    annotation: Any,
) -> tuple[np.ndarray, dict[int, dict[str, str]]]:
    if not isinstance(annotation, dict):
        raise RuntimeError(
            "instance_segmentation_fast did not return its metadata dictionary."
        )
    raw_map = _unwrap_data(annotation, "instance").astype(np.int32, copy=False)
    if raw_map.ndim == 3 and raw_map.shape[-1] == 1:
        raw_map = raw_map[..., 0]
    if raw_map.ndim != 2:
        raise RuntimeError(
            f"Instance map must be HxW, received shape {raw_map.shape}."
        )

    info = annotation.get("info", {})
    if not isinstance(info, dict):
        info = {}
    id_to_labels = _mapping(
        info.get("idToLabels", info.get("id_to_labels"))
    )
    if isinstance(id_to_labels.get("idToLabels"), dict):
        id_to_labels = dict(id_to_labels["idToLabels"])
    # Different Replicator releases use singular/plural spellings.
    id_to_semantics = _mapping(
        info.get(
            "idToSemantics",
            info.get("idToSemantic", info.get("id_to_semantics")),
        )
    )
    if isinstance(id_to_semantics.get("idToSemantics"), dict):
        id_to_semantics = dict(id_to_semantics["idToSemantics"])

    metadata: dict[int, dict[str, str]] = {}
    valid_ids: list[int] = []
    for raw_id in np.unique(raw_map).tolist():
        instance_id = int(raw_id)
        if instance_id <= 1:
            continue
        semantic_value = id_to_semantics.get(
            str(instance_id), id_to_semantics.get(instance_id)
        )
        label = _semantic_label(semantic_value)
        prim_path_value = id_to_labels.get(
            str(instance_id), id_to_labels.get(instance_id, "")
        )
        prim_path = str(prim_path_value)
        if label is None and prim_path:
            label = prim_path.rstrip("/").split("/")[-1].lower()
        if not label or label.upper() in {"BACKGROUND", "UNLABELLED"}:
            continue
        valid_ids.append(instance_id)
        metadata[instance_id] = {
            "label": label,
            "prim_path": prim_path,
        }

    # Remove renderer background/unlabelled ids and any unlabeled geometry.
    if valid_ids:
        valid = np.isin(raw_map, np.asarray(valid_ids, dtype=np.int32))
        normalized = np.where(valid, raw_map, 0).astype(np.int32, copy=False)
    else:
        normalized = np.zeros_like(raw_map, dtype=np.int32)
    return normalized, metadata


def capture_all_cameras(
    cameras: Iterable[CameraRuntime],
    rig: CameraRigConfig,
) -> dict[str, CameraFrame]:
    frames: dict[str, CameraFrame] = {}
    for runtime in cameras:
        rgb = _unwrap_data(
            runtime.rgb_annotator.get_data(),
            f"{runtime.spec.name}/rgb",
        )
        depth = _unwrap_data(
            runtime.depth_annotator.get_data(),
            f"{runtime.spec.name}/depth",
        )
        instance_map, metadata = _normalize_instance_annotation(
            runtime.instance_annotator.get_data()
        )

        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(
                f"{runtime.spec.name} RGB shape is {rgb.shape}, expected HxWx3/4."
            )
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        if depth.shape != (rig.height, rig.width):
            raise RuntimeError(
                f"{runtime.spec.name} depth shape is {depth.shape}, "
                f"expected {(rig.height, rig.width)}."
            )
        depth[(~np.isfinite(depth)) | (depth <= 0.0) | (depth > rig.max_depth_m)] = np.nan
        if instance_map.shape != depth.shape:
            raise RuntimeError(
                f"{runtime.spec.name} instance shape {instance_map.shape} "
                f"does not match depth shape {depth.shape}."
            )
        frames[runtime.spec.name] = CameraFrame(
            rgb=rgb,
            depth_m=depth,
            instance_map=np.ascontiguousarray(instance_map, dtype=np.int32),
            instance_metadata=metadata,
        )
    return frames
