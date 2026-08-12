#!/usr/bin/env python3
"""Pure-Python geometry sanity check for the occlusion test scene.

Checks that the centered panel can hide both static targets and the entire
translated dynamic-ball envelope from both cameras, while the two clear hold
positions move the panel fully outside those lines of sight.
"""

from __future__ import annotations

from itertools import product


CAMERAS = (
    (1.55, -1.70, 1.55),
    (-1.45, -1.45, 1.35),
)
PANEL_Y = 0.05
PANEL_HALF_WIDTH = 0.60
PANEL_FULL_CENTER_X = 0.0
PANEL_CLEAR_CENTERS_X = (-1.20, +1.20)
PANEL_Z_MIN = 0.79
PANEL_Z_MAX = 1.41

# Approximate target points are sufficient for this visibility sanity check.
STATIC_TARGETS = (
    ("food_can", -0.38, 0.22, 0.90),
    ("mustard_bottle", 0.00, 0.22, 0.92),
)

# Occlusion-mode ball is dynamic-scene motion rigidly translated:
# x = 0.38 +/- 0.055, y = 0.22 +/- 0.012, root z = 0.79..1.01 m.
BALL_ENVELOPE = tuple(
    ("ball", x, y, z)
    for x, y, z in product(
        (0.325, 0.435),
        (0.208, 0.232),
        (0.79, 1.01),
    )
)


def sight_at_panel(camera, target):
    _, tx, ty, tz = target
    alpha = (PANEL_Y - camera[1]) / (ty - camera[1])
    x = camera[0] + alpha * (tx - camera[0])
    z = camera[2] + alpha * (tz - camera[2])
    return x, z


def panel_covers_x(center_x: float, x: float) -> bool:
    return abs(x - center_x) < PANEL_HALF_WIDTH


def main() -> None:
    targets = STATIC_TARGETS + BALL_ENVELOPE
    all_crossings = []

    for camera_index, camera in enumerate(CAMERAS):
        camera_crossings = []
        for target in targets:
            x, z = sight_at_panel(camera, target)
            camera_crossings.append((x, z, target[0]))
            all_crossings.append(x)

            if not panel_covers_x(PANEL_FULL_CENTER_X, x):
                raise RuntimeError(
                    f"camera_{camera_index} FULL panel misses {target[0]} "
                    f"line of sight in x: x={x:+.3f} m"
                )
            if not (PANEL_Z_MIN <= z <= PANEL_Z_MAX):
                raise RuntimeError(
                    f"camera_{camera_index} panel misses {target[0]} "
                    f"line of sight in z: z={z:.3f} m"
                )

        xs = [item[0] for item in camera_crossings]
        zs = [item[1] for item in camera_crossings]
        print(
            f"camera_{camera_index}: target LOS envelope at panel plane "
            f"x=[{min(xs):+.3f}, {max(xs):+.3f}] m, "
            f"z=[{min(zs):.3f}, {max(zs):.3f}] m"
        )

    for clear_center in PANEL_CLEAR_CENTERS_X:
        overlapping = [
            x for x in all_crossings
            if panel_covers_x(clear_center, x)
        ]
        if overlapping:
            raise RuntimeError(
                f"CLEAR panel at x={clear_center:+.2f} still overlaps "
                f"{len(overlapping)} target sight-lines"
            )

    print(
        "FULL panel covers all static/dynamic target sight-lines; "
        "both CLEAR holds uncover them."
    )
    print("OCCLUSION_GEOMETRY_OK")


if __name__ == "__main__":
    main()
