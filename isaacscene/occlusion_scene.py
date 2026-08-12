#!/usr/bin/env python3
"""Deterministic foreground occluder for RGB-D tracking tests.

The occluder is a simple opaque panel that moves laterally between the camera
rig and the rear-center target objects. One half-cycle is intentionally:

    clear -> partial -> full -> partial -> clear

The motion then reverses and repeats. The panel is a test fixture rather than a
semantic target, so it is deliberately not part of the SAM3 prompt vocabulary.

SimulationApp must be created before importing this module.
"""

from __future__ import annotations

import math

from pxr import Gf, Sdf, UsdGeom, UsdShade


class OcclusionController:
    """Create and animate one opaque foreground occluder panel."""

    ROOT_PATH = "/World/OcclusionTest"
    PANEL_PATH = f"{ROOT_PATH}/OccluderPanel"
    MATERIAL_PATH = "/World/Materials/Occluder"

    # Geometry is chosen for the existing two-camera rig and the rear target
    # row. The static can/bottle and the translated dynamic ball all sit behind
    # this panel. A wide panel is intentional: at the centered FULL phase it
    # covers both cameras' sight-lines to the ball across its whole x motion.
    PANEL_SIZE_X_M = 1.20
    PANEL_SIZE_Y_M = 0.045
    PANEL_SIZE_Z_M = 0.62
    PANEL_Y_M = 0.05
    X_AMPLITUDE_M = 1.20
    PERIOD_S = 16.0

    # These thresholds are only diagnostic labels. Actual image-space
    # visibility remains determined by the renderer and segmentation model.
    FULL_X_THRESHOLD_M = 0.08
    CLEAR_X_THRESHOLD_M = 1.10

    def __init__(
        self,
        stage,
        *,
        table_surface_z: float,
        speed_scale: float = 1.0,
    ) -> None:
        if speed_scale <= 0.0:
            raise ValueError("speed_scale must be positive.")

        self.speed_scale = float(speed_scale)
        self.table_surface_z = float(table_surface_z)
        self._last_state: str | None = None

        UsdGeom.Xform.Define(stage, Sdf.Path(self.ROOT_PATH))

        material = UsdShade.Material.Define(
            stage,
            Sdf.Path(self.MATERIAL_PATH),
        )
        shader = UsdShade.Shader.Define(
            stage,
            Sdf.Path(f"{self.MATERIAL_PATH}/Shader"),
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).Set(Gf.Vec3f(0.08, 0.12, 0.18))
        shader.CreateInput(
            "roughness",
            Sdf.ValueTypeNames.Float,
        ).Set(0.78)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(),
            "surface",
        )

        panel = UsdGeom.Cube.Define(
            stage,
            Sdf.Path(self.PANEL_PATH),
        )
        panel.CreateSizeAttr(1.0)
        prim = panel.GetPrim()
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        self._translate_op = xformable.AddTranslateOp()
        xformable.AddScaleOp().Set(
            Gf.Vec3f(
                self.PANEL_SIZE_X_M,
                self.PANEL_SIZE_Y_M,
                self.PANEL_SIZE_Z_M,
            )
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        self.update(0.0)

    @staticmethod
    def _smooth_lerp(start: float, end: float, alpha: float) -> float:
        alpha = min(1.0, max(0.0, float(alpha)))
        weight = 0.5 - 0.5 * math.cos(math.pi * alpha)
        return (1.0 - weight) * start + weight * end

    def _x_position(self, elapsed_s: float) -> float:
        # Explicit holds make the scene useful for a ~1 Hz SAM3 refresh: both
        # clear and fully occluded states persist for many RGB-D frames.
        t = (max(0.0, float(elapsed_s)) * self.speed_scale) % self.PERIOD_S
        a = self.X_AMPLITUDE_M

        if t < 1.5:
            return a
        if t < 3.5:
            return self._smooth_lerp(a, 0.0, (t - 1.5) / 2.0)
        if t < 5.0:
            return 0.0
        if t < 7.0:
            return self._smooth_lerp(0.0, -a, (t - 5.0) / 2.0)
        if t < 9.0:
            return -a
        if t < 11.0:
            return self._smooth_lerp(-a, 0.0, (t - 9.0) / 2.0)
        if t < 12.5:
            return 0.0
        if t < 14.5:
            return self._smooth_lerp(0.0, a, (t - 12.5) / 2.0)
        return a

    @classmethod
    def _state_from_x(cls, x: float) -> str:
        abs_x = abs(float(x))
        if abs_x <= cls.FULL_X_THRESHOLD_M:
            return "full"
        if abs_x >= cls.CLEAR_X_THRESHOLD_M:
            return "clear"
        return "partial"

    def update(self, elapsed_s: float) -> None:
        x = self._x_position(elapsed_s)
        z = self.table_surface_z + 0.5 * self.PANEL_SIZE_Z_M
        self._translate_op.Set(
            Gf.Vec3d(
                float(x),
                self.PANEL_Y_M,
                float(z),
            )
        )

        state = self._state_from_x(x)
        if state != self._last_state:
            print(
                "[OCCLUSION] "
                f"t={max(0.0, float(elapsed_s)):.2f}s "
                f"state={state} x={x:+.3f}m",
                flush=True,
            )
            self._last_state = state

    def description(self) -> str:
        return (
            "occluder_panel(" 
            "clear->partial->full->partial->clear, "
            f"period={self.PERIOD_S / self.speed_scale:.1f}s)"
        )
