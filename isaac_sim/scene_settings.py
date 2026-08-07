from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pxr import Gf, UsdGeom, UsdLux

try:
    from isaacsim.core.experimental.utils.semantics import add_labels

    def _add_class_label(prim: Any, label: str) -> None:
        add_labels(prim, labels=[label], taxonomy="class")
except ImportError:  # Isaac Sim 5.x compatibility
    from isaacsim.core.utils.semantics import add_labels

    def _add_class_label(prim: Any, label: str) -> None:
        add_labels(prim, [label], instance_name="class")


SCENE_CHOICES = ("static", "dynamic", "occlusion")
TABLE_SURFACE_Z = 0.70


@dataclass
class AnimatedObject:
    name: str
    label: str
    translate_op: Any
    rotate_op: Any
    base_position: tuple[float, float, float]
    base_rotation_deg: tuple[float, float, float]

    def set_pose(
        self,
        position: tuple[float, float, float],
        rotation_deg: tuple[float, float, float] | None = None,
    ) -> None:
        self.translate_op.Set(Gf.Vec3d(*[float(v) for v in position]))
        rotation = self.base_rotation_deg if rotation_deg is None else rotation_deg
        self.rotate_op.Set(Gf.Vec3f(*[float(v) for v in rotation]))


class SceneController:
    def __init__(
        self,
        mode: str,
        objects: dict[str, AnimatedObject],
        speed_scale: float,
    ) -> None:
        self.mode = mode
        self.objects = objects
        self.speed_scale = float(speed_scale)

    def description(self) -> str:
        return f"{self.mode}:" + ",".join(sorted(self.objects))

    def update(self, elapsed_s: float) -> None:
        t = float(elapsed_s) * self.speed_scale
        if self.mode == "static":
            return
        if self.mode == "dynamic":
            bottle = self.objects["bottle"]
            bottle.set_pose(
                (
                    0.26 + 0.23 * math.sin(0.72 * t),
                    0.10 + 0.10 * math.cos(0.52 * t),
                    TABLE_SURFACE_Z,
                ),
                (0.0, 0.0, 25.0 * math.sin(0.65 * t)),
            )
            box = self.objects["box"]
            box.set_pose(
                (
                    -0.28 + 0.16 * math.sin(0.48 * t + 1.2),
                    0.18 + 0.12 * math.sin(0.81 * t),
                    TABLE_SURFACE_Z,
                ),
                (0.0, 0.0, 30.0 + 20.0 * math.sin(0.57 * t)),
            )
            shelf = self.objects["shelf"]
            shelf.set_pose(
                (
                    0.42 * math.sin(0.36 * t),
                    -0.18,
                    TABLE_SURFACE_Z,
                ),
                (0.0, 0.0, 8.0 * math.sin(0.42 * t)),
            )
            return

        # The shelf travels between both cameras and the two target objects.
        # It creates partial, full and recovery intervals without deleting the
        # targets, which is the intended benchmark for identity persistence.
        sweep = 0.54 * math.sin(2.0 * math.pi * t / 7.0)
        self.objects["shelf"].set_pose(
            (sweep, -0.10, TABLE_SURFACE_Z),
            (0.0, 0.0, 3.0 * math.sin(0.4 * t)),
        )
        self.objects["bottle"].set_pose(
            (0.18 + 0.025 * math.sin(0.6 * t), 0.22, TABLE_SURFACE_Z),
            (0.0, 0.0, 10.0 * math.sin(0.35 * t)),
        )
        self.objects["box"].set_pose(
            (-0.27, 0.18 + 0.018 * math.sin(0.5 * t), TABLE_SURFACE_Z),
            (0.0, 0.0, -18.0),
        )


def _define_xform(stage: Any, path: str) -> Any:
    return UsdGeom.Xform.Define(stage, path).GetPrim()


def _set_local_transform(
    prim: Any,
    translation: tuple[float, float, float],
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*[float(v) for v in translation])
    )
    xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*[float(v) for v in rotation_deg])
    )
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Vec3f(*[float(v) for v in scale])
    )


def _set_display_color(geometry: Any, color: tuple[float, float, float]) -> None:
    geometry.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])


def _animated_root(
    stage: Any,
    name: str,
    label: str,
    position: tuple[float, float, float],
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> AnimatedObject:
    root = UsdGeom.Xform.Define(stage, f"/World/Objects/{name}")
    xformable = UsdGeom.Xformable(root.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(Gf.Vec3d(*position))
    rotate_op.Set(Gf.Vec3f(*rotation_deg))
    _add_class_label(root.GetPrim(), label)
    return AnimatedObject(
        name=name,
        label=label,
        translate_op=translate_op,
        rotate_op=rotate_op,
        base_position=position,
        base_rotation_deg=rotation_deg,
    )


def _create_bottle(stage: Any, position: tuple[float, float, float]) -> AnimatedObject:
    runtime = _animated_root(stage, "Bottle", "bottle", position)
    body = UsdGeom.Cylinder.Define(stage, "/World/Objects/Bottle/Body")
    body.CreateAxisAttr("Z")
    body.CreateRadiusAttr(0.065)
    body.CreateHeightAttr(0.28)
    _set_local_transform(body.GetPrim(), (0.0, 0.0, 0.14))
    _set_display_color(body, (0.93, 0.73, 0.10))
    neck = UsdGeom.Cylinder.Define(stage, "/World/Objects/Bottle/Neck")
    neck.CreateAxisAttr("Z")
    neck.CreateRadiusAttr(0.035)
    neck.CreateHeightAttr(0.09)
    _set_local_transform(neck.GetPrim(), (0.0, 0.0, 0.325))
    _set_display_color(neck, (0.96, 0.82, 0.20))
    cap = UsdGeom.Cylinder.Define(stage, "/World/Objects/Bottle/Cap")
    cap.CreateAxisAttr("Z")
    cap.CreateRadiusAttr(0.041)
    cap.CreateHeightAttr(0.035)
    _set_local_transform(cap.GetPrim(), (0.0, 0.0, 0.3875))
    _set_display_color(cap, (0.15, 0.12, 0.08))
    return runtime


def _create_box(stage: Any, position: tuple[float, float, float]) -> AnimatedObject:
    runtime = _animated_root(stage, "Box", "box", position, (0.0, 0.0, -18.0))
    box = UsdGeom.Cube.Define(stage, "/World/Objects/Box/Geometry")
    box.CreateSizeAttr(1.0)
    _set_local_transform(box.GetPrim(), (0.0, 0.0, 0.17), scale=(0.22, 0.14, 0.34))
    _set_display_color(box, (0.15, 0.55, 0.92))
    return runtime


def _create_shelf(stage: Any, position: tuple[float, float, float]) -> AnimatedObject:
    runtime = _animated_root(stage, "Shelf", "shelf", position)
    color = (0.70, 0.24, 0.18)
    # A framed shelf gives the tracker thin structures while the broad back
    # panel guarantees a controlled full-occlusion interval.
    parts = {
        "Back": ((0.0, 0.0, 0.30), (0.48, 0.055, 0.60)),
        "LeftPost": ((-0.27, 0.0, 0.30), (0.055, 0.10, 0.60)),
        "RightPost": ((0.27, 0.0, 0.30), (0.055, 0.10, 0.60)),
        "Top": ((0.0, 0.0, 0.59), (0.60, 0.10, 0.055)),
        "Middle": ((0.0, 0.0, 0.30), (0.60, 0.10, 0.045)),
    }
    for part_name, (translation, scale) in parts.items():
        cube = UsdGeom.Cube.Define(
            stage, f"/World/Objects/Shelf/{part_name}"
        )
        cube.CreateSizeAttr(1.0)
        _set_local_transform(cube.GetPrim(), translation, scale=scale)
        _set_display_color(cube, color)
    return runtime


def _create_environment(stage: Any) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for path in (
        "/World",
        "/World/Environment",
        "/World/Objects",
        "/World/Cameras",
        "/World/Lights",
    ):
        _define_xform(stage, path)

    ground = UsdGeom.Cube.Define(stage, "/World/Environment/Ground")
    ground.CreateSizeAttr(1.0)
    _set_local_transform(ground.GetPrim(), (0.0, 0.0, -0.04), scale=(4.0, 4.0, 0.08))
    _set_display_color(ground, (0.16, 0.17, 0.19))

    top = UsdGeom.Cube.Define(stage, "/World/Environment/TableTop")
    top.CreateSizeAttr(1.0)
    _set_local_transform(top.GetPrim(), (0.0, 0.0, 0.65), scale=(1.75, 1.35, 0.10))
    _set_display_color(top, (0.48, 0.31, 0.18))
    for index, (x, y) in enumerate(((-0.72, -0.52), (-0.72, 0.52), (0.72, -0.52), (0.72, 0.52))):
        leg = UsdGeom.Cube.Define(stage, f"/World/Environment/TableLeg_{index}")
        leg.CreateSizeAttr(1.0)
        _set_local_transform(leg.GetPrim(), (x, y, 0.30), scale=(0.10, 0.10, 0.60))
        _set_display_color(leg, (0.34, 0.22, 0.13))

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(550.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(1.0)
    _set_local_transform(key.GetPrim(), (0.0, 0.0, 3.0), (-38.0, 20.0, 24.0))
    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(18000.0)
    fill.CreateRadiusAttr(0.35)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.70))
    _set_local_transform(fill.GetPrim(), (-1.3, -1.0, 2.2))


def build_scene(stage: Any, scene_mode: str, speed_scale: float = 1.0) -> SceneController:
    if scene_mode not in SCENE_CHOICES:
        raise ValueError(
            f"Unknown scene {scene_mode!r}; expected one of {SCENE_CHOICES}."
        )
    if speed_scale <= 0.0:
        raise ValueError("speed_scale must be positive.")
    _create_environment(stage)

    if scene_mode == "static":
        objects = {
            "bottle": _create_bottle(stage, (0.24, 0.10, TABLE_SURFACE_Z)),
            "box": _create_box(stage, (-0.30, 0.16, TABLE_SURFACE_Z)),
            "shelf": _create_shelf(stage, (0.02, -0.28, TABLE_SURFACE_Z)),
        }
    elif scene_mode == "dynamic":
        objects = {
            "bottle": _create_bottle(stage, (0.26, 0.10, TABLE_SURFACE_Z)),
            "box": _create_box(stage, (-0.28, 0.18, TABLE_SURFACE_Z)),
            "shelf": _create_shelf(stage, (0.0, -0.18, TABLE_SURFACE_Z)),
        }
    else:
        objects = {
            "bottle": _create_bottle(stage, (0.18, 0.22, TABLE_SURFACE_Z)),
            "box": _create_box(stage, (-0.27, 0.18, TABLE_SURFACE_Z)),
            "shelf": _create_shelf(stage, (-0.54, -0.10, TABLE_SURFACE_Z)),
        }
    controller = SceneController(scene_mode, objects, speed_scale)
    controller.update(0.0)
    return controller
