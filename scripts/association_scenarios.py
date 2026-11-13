"""The scenes of the association benchmark: object classes, concrete objects, counts, 2 - 4 cameras, sensing faults."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from association_render import Camera, Obj, Sensing, box, chair, cylinder, look_at, miscalibrate, person, ring, rod, sphere, table

GROUND = Obj(0, None, [box([0.0, 0.0, -0.05], [14.0, 14.0, 0.1])])
AWAY = 30.0                                      # x of an object that is not in the scene


@dataclass
class Scenario:
    name: str
    note: str
    cameras: list[Camera]
    capacities: dict[str, int]
    objects: Callable[[float], list[Obj]]
    duration_s: float = 6.0
    sensing: Sensing = field(default_factory=Sensing)
    faults: Callable[[float], dict] | None = None


def _ping(t, low, high, speed):
    """Back and forth between low and high."""
    span = high - low
    phase = (t * speed) % (2.0 * span)
    return low + (phase if phase <= span else 2.0 * span - phase), (1.0 if phase <= span else -1.0)


def _walker(ident, t, *, x=0.0, low=-1.2, high=1.2, speed=1.0, along="y", start=0.0, label="person", height=1.75):
    pos, sign = _ping(t + start, low, high, speed)
    yaw = (math.pi / 2 if along == "y" else 0.0) + (0.0 if sign > 0 else math.pi)
    xy = (x, pos) if along == "y" else (pos, x)
    return Obj(ident, label, person(xy[0], xy[1], yaw, phase=2.0 * math.pi * 1.8 * (t + start) * speed, height=height))


def _can(ident, x, y, z, label="can", radius=0.033, height=0.10):
    return Obj(ident, label, [cylinder([x, y, z + 0.5 * height], radius, height)])


def _balls(t, count, seed, label="ball"):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        base = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(0.5, 1.6)])
        amp = np.array([rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5), rng.uniform(0.1, 0.3)])
        freq, phase = rng.uniform(0.2, 0.6, 3), rng.uniform(0.0, 6.28, 3)
        out.append(Obj(100 + k, label, [sphere(base + amp * np.sin(2.0 * math.pi * freq * t + phase), rng.uniform(0.06, 0.11))]))
    return out


# ---- the scenes ---------------------------------------------------------------------------------------------------------
def s01(t):
    return [GROUND, _walker(1, t)]


def s02(t):
    angle = 0.9 * t
    return [GROUND, Obj(1, "person", person(0.7 * math.cos(angle), 0.7 * math.sin(angle), angle + math.pi / 2, phase=8.0 * t))]


def s03(t):
    return [GROUND, _walker(1, t, x=0.3), _walker(2, t, x=-0.3, start=2.4)]


def s04(t):
    angle = 0.6 * t
    centre = np.array([0.9 * math.cos(angle), 0.9 * math.sin(angle)])
    radial = np.array([math.cos(angle), math.sin(angle)])
    a, b = centre + 0.29 * radial, centre - 0.29 * radial
    return [GROUND, Obj(1, "person", person(a[0], a[1], angle + math.pi / 2, phase=7.0 * t)),
            Obj(2, "person", person(b[0], b[1], angle + math.pi / 2, phase=7.0 * t + 1.0))]


def s05(t):
    third = _walker(3, t, x=-0.9, low=-1.0, high=1.0, speed=0.8, along="x") if 2.0 <= t <= 5.0 else Obj(3, "person", person(AWAY, 0.0, 0.0))
    return [GROUND, _walker(1, t, x=0.9, speed=1.2), Obj(2, "person", person(-0.2, -0.9, 0.8 * t, height=1.6)), third]


def s06(t):
    swing = 1.2 * math.sin(2.0 * math.pi * 0.7 * t)
    hand = np.array([-0.05, -0.30, 1.25]) + 0.25 * np.array([math.cos(swing), 0.0, math.sin(swing)])
    tip = hand + 0.8 * np.array([math.cos(swing) * 0.8, -0.35, math.sin(swing) * 0.8 + 0.3]) / np.linalg.norm([math.cos(swing) * 0.8, -0.35, math.sin(swing) * 0.8 + 0.3])
    return [GROUND, Obj(1, "person", person(-0.3, 0.0, 0.0, reach=hand)), Obj(2, "stick", [rod(hand, tip, 0.02)])]


def _table_scene(extra):
    return [GROUND, Obj(9, None, table(0.0, 0.0))] + extra


def s07(t):
    slide = 0.10 * math.sin(2.0 * math.pi * 0.4 * t)
    return _table_scene([_can(1, -0.12, slide, 0.74), _can(2, 0.0, 0.0, 0.74), _can(3, 0.12, -slide, 0.74)])


def s08(t):
    shift = 0.15 * math.sin(2.0 * math.pi * 0.3 * t)
    return _table_scene([_can(1 + k, (k - 1) * 0.076, shift, 0.74) for k in range(3)])


def s09(t):
    return [GROUND, Obj(1, "person", person(0.9, 0.2, math.pi, phase=0.0))] + _balls(t, 8, seed=3)


def s10(t):
    return [GROUND] + _balls(t, 8, seed=11)


def s11(t):
    roll = 1.0 * math.sin(2.0 * math.pi * 0.25 * t)
    angle = 0.7 * t
    return [GROUND, Obj(1, "table", table(0.0, 0.0)), Obj(2, "chair", chair(-0.9, 0.6, 0.3)), Obj(3, "chair", chair(0.2, -0.95, 1.8)),
            Obj(4, "cracker box", [box([-0.3, 0.1, 0.74 + 0.105], [0.16, 0.06, 0.21], 0.4)]),
            _can(5, 0.25, -0.15, 0.74, label="mug", radius=0.045, height=0.095),
            Obj(6, "bottle", [cylinder([0.1, 0.2, 0.74 + 0.10], 0.035, 0.20), cylinder([0.1, 0.2, 0.74 + 0.225], 0.014, 0.05)]),
            Obj(7, "ball", [sphere([1.1, roll, 0.11], 0.11)]),
            Obj(8, "person", person(1.7 * math.cos(angle), 1.7 * math.sin(angle), angle + math.pi / 2, phase=7.0 * t))]


def s12(t):
    lift = 0.15 * math.sin(2.0 * math.pi * 0.3 * t)
    return [GROUND, Obj(1, "pipe", [rod([0.0, -1.2, 1.0 + lift], [0.0, 1.2, 1.0 + lift], 0.03)]),
            Obj(2, "pipe", [rod([0.35, -1.2, 1.3 - lift], [0.35, 1.2, 1.3 - lift], 0.03)])]


def s13(t):
    return [GROUND, _walker(1, t, low=-1.0, high=1.0, speed=0.7),
            Obj(8, None, [box([0.6, 0.0, 0.5], [0.3, 3.2, 1.0])]),            # counter towards cam0: hides the legs
            Obj(9, None, [box([-0.6, 0.0, 1.65], [0.05, 3.2, 1.2])])]         # hanging panel towards cam1: hides the upper body


def s14(t):
    drift = 0.25 * math.sin(2.0 * math.pi * 0.2 * t)
    crate = lambda ident, x, y: Obj(ident, "crate", [box([x, y, 0.2], [0.4, 0.4, 0.4], 0.3 * ident)])
    return [GROUND, Obj(9, None, [box([-0.6, 0.0, 0.75], [0.1, 2.4, 1.5])]), crate(1, 0.5, 0.6 + drift), crate(2, 0.5, -0.6 + drift),
            crate(3, -1.3, drift)]


def s15(t):
    ball = lambda ident, y, start: Obj(ident, "ball", [sphere([(-3.5 + 1.2 * (t - start)) if start <= t <= start + 5.8 else AWAY, y, 0.11], 0.11)])
    return [GROUND, ball(1, -0.5, 0.0), ball(2, 0.0, 1.0), ball(3, 0.5, 2.0)]


def s16(t):
    share = min(max((t - 2.0) / 2.0, 0.0), 1.0)
    hold = np.array([-0.12 + 0.24 * share, -0.05, 1.15])
    return [GROUND, Obj(1, "person", person(-0.5, 0.2, 0.0, reach=hold + [-0.04, 0.0, 0.0] if share < 1.0 else None)),
            Obj(2, "person", person(0.5, -0.3, math.pi, reach=hold + [0.04, 0.0, 0.0] if share > 0.0 else None)),
            Obj(3, "bottle", [cylinder(hold, 0.035, 0.25)])]


def s17(t):
    lift = min(max(t - 3.0, 0.0), 1.5)
    top = [0.0 + 0.4 * lift, 0.0, 0.74 + 0.06 + 0.03 + 0.2 * min(lift, 0.5)]
    flat = lambda ident, centre: Obj(ident, "cracker box", [box(centre, [0.21, 0.16, 0.06], 0.2)])
    return _table_scene([flat(1, [0.0, 0.0, 0.74 + 0.03]), flat(2, top), flat(3, [-0.26, 0.02, 0.74 + 0.03])])


def s18(t):
    cycle = t % 1.5
    ball = [-2.5 + 5.0 * cycle, 0.6, 1.2 + 3.0 * cycle - 4.9 * cycle * cycle] if cycle < 1.0 else [AWAY, 0.0, 1.0]
    return [GROUND, Obj(1, "person", person(-0.4, -0.5, 0.5)), Obj(2, "ball", [sphere(ball, 0.06)])]


def s19(t):
    return [GROUND, _walker(1, t, x=1.0), _walker(2, t, x=-1.0, start=1.3),
            Obj(7, None, [cylinder([0.0, 1.9, 0.85], 0.12, 1.7), sphere([0.0, 1.9, 1.75], 0.14)])]      # coat rack


def s19_faults(t):
    faults = {"drop": set(), "label": {}, "swap": set()}
    if 1.0 <= t <= 1.5:
        faults["drop"].add((0, 1))                   # camera 0 loses person 1
    if 2.0 <= t <= 4.0:
        faults["label"][(1, 7)] = "person"           # camera 1 calls the coat rack a person
    if 4.5 <= t <= 4.6:
        faults["swap"].add((2, 1, 2))                # camera 2's slots jump between the two persons
    return faults


def s20(t):
    swing = 0.8 + 0.7 * math.sin(2.0 * math.pi * 0.25 * t)
    hinge = np.array([-0.8, -0.9])
    leaf = hinge + 0.45 * np.array([math.cos(swing), math.sin(swing)])
    return [GROUND, Obj(1, "door", [box([leaf[0], leaf[1], 1.0], [0.9, 0.04, 2.0], swing)]), _walker(2, t, x=0.5, speed=0.9),
            Obj(3, "chair", chair(-0.4, 0.9, 2.5)), Obj(4, "chair", chair(1.2, -1.0, 0.7))]


def _table_cams(count, **kwargs):
    return ring(count, radius=1.6, height=1.5, target=(0.0, 0.0, 0.8), **kwargs)


def _pipe_cams():
    return [look_at("cam0", [2.5, -0.9, 1.6], [0.35, -0.9, 1.1], focal=900.0), look_at("cam1", [2.5, 0.9, 1.6], [0.35, 0.9, 1.1], focal=900.0)]


def scenarios() -> list[Scenario]:
    hard_cameras = ring(4, start_deg=45.0)
    miscalibrate(hard_cameras, 0.015, 0.7, seed=5)
    return [
        Scenario("01_person_front_back", "1 person walking, cameras opposite", ring(2), {"person": 1}, s01),
        Scenario("02_person_spare_capacity", "1 person, capacity 2, cameras at 90 deg", ring(2, span_deg=90.0, start_deg=-45.0), {"person": 2}, s02),
        Scenario("03_two_persons_crossing", "2 persons pass 0.6 m apart, hide each other", ring(2), {"person": 2}, s03),
        Scenario("04_two_persons_side_by_side", "2 persons walk shoulder to shoulder", ring(3), {"person": 2}, s04),
        Scenario("05_three_persons_enter_leave", "3 persons, one enters and leaves, capacity 4", ring(4, start_deg=45.0), {"person": 4}, s05, 7.0),
        Scenario("06_person_with_stick", "person swings a 0.8 m stick", ring(2, span_deg=120.0, start_deg=-60.0), {"person": 1, "stick": 1}, s06,
                 sensing=Sensing(erode_px=1)),
        Scenario("07_cans_spaced", "3 soup cans 12 cm apart on a table", _table_cams(2, span_deg=100.0, start_deg=-50.0), {"can": 3}, s07),
        Scenario("08_cans_touching", "3 soup cans 1 cm apart", _table_cams(3), {"can": 3}, s08),
        Scenario("09_eight_balls_person", "8 balls around a person", ring(2, span_deg=110.0, start_deg=-55.0), {"ball": 8, "person": 1}, s09),
        Scenario("10_eight_balls_four_cams", "8 balls", ring(4, start_deg=45.0), {"ball": 8}, s10),
        Scenario("11_mixed_room", "table, 2 chairs, box, mug, bottle, ball, person", ring(3, start_deg=20.0),
                 {"table": 1, "chair": 2, "cracker box": 1, "mug": 1, "bottle": 1, "ball": 1, "person": 1}, s11),
        Scenario("12_pipes_partial_views", "2 pipes, each camera sees another half", _pipe_cams(), {"pipe": 2}, s12),
        Scenario("13_person_split_by_occluders", "one camera sees the upper body, the other the legs", ring(2), {"person": 1}, s13),
        Scenario("14_crate_one_view_only", "3 crates, one seen by a single camera",
                 [look_at("cam0", [2.8, -1.0, 1.9], [0.0, 0.0, 0.5]), look_at("cam1", [2.8, 1.0, 1.9], [0.0, 0.0, 0.5]),
                  look_at("cam2", [-2.8, 0.0, 1.9], [0.0, 0.0, 0.5])], {"crate": 3}, s14),
        Scenario("15_balls_enter_leave", "3 balls roll through one after the other", ring(2, span_deg=90.0, start_deg=45.0), {"ball": 3}, s15, 8.0),
        Scenario("16_bottle_handover", "a bottle goes from one person's hand to another's", ring(3, start_deg=30.0), {"person": 2, "bottle": 1}, s16),
        Scenario("17_stacked_boxes", "2 stacked boxes and one beside; the top one is lifted off", _table_cams(4, start_deg=45.0), {"cracker box": 3}, s17),
        Scenario("18_thrown_ball", "ball thrown at 5 m/s past a person", ring(2, span_deg=120.0, start_deg=-60.0), {"person": 1, "ball": 1}, s18),
        Scenario("19_detection_faults", "false positive, lost detection, slot jump, flicker", ring(3), {"person": 2}, s19,
                 sensing=Sensing(flicker=0.03), faults=s19_faults),
        Scenario("20_door_bad_sensing", "door, person, 2 chairs; 1.5 cm / 0.7 deg calibration error, 2x noise, flying pixels", hard_cameras,
                 {"door": 1, "person": 1, "chair": 2}, s20, sensing=Sensing(noise=0.003, flying=0.2)),
    ]
