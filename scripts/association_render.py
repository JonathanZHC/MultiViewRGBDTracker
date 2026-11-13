"""Synthetic multi-camera observations with known identities, for the cross-view / cross-frame association benchmark.

A scene is a set of objects made of spheres, boxes and capped cylinders. Every camera casts its pixel rays against all
of them (occlusion included), so each camera gets what the tracker gets after its 2-D stage: per visible object one
instance with the world points of the pixels that survive the mask erosion, depth noise and (optionally) flying pixels,
a calibration error of the camera, detection faults, and a per-camera slot id that stays on the object like the 2-D
tracker's. The true object of every instance is kept in ``instance.truth``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from sam_rgbd_tracking.data_types import FrameResult, ProcessedInstance, VisibilityState

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPHERE, BOX, CYLINDER = 0, 1, 2


# ---- scene description ------------------------------------------------------------------------------------------------
@dataclass
class Prim:
    kind: int
    center: np.ndarray               # world
    rot: np.ndarray                  # [3, 3] local -> world
    size: np.ndarray                 # sphere (r, -, -), box half extents, cylinder (r, half height along local z, -)


@dataclass
class Obj:
    ident: int
    label: str | None                # semantic class; None: not tracked (occluder, background)
    prims: list[Prim]


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_from_z(direction) -> np.ndarray:
    """A rotation whose local z axis is ``direction``."""
    z = np.asarray(direction, float)
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def sphere(center, radius) -> Prim:
    return Prim(SPHERE, np.asarray(center, float), np.eye(3), np.array([radius, 0.0, 0.0]))


def box(center, size, yaw=0.0, rot=None) -> Prim:
    return Prim(BOX, np.asarray(center, float), rot_z(yaw) if rot is None else rot, 0.5 * np.asarray(size, float))


def cylinder(center, radius, height, axis=(0.0, 0.0, 1.0)) -> Prim:
    return Prim(CYLINDER, np.asarray(center, float), rot_from_z(axis), np.array([radius, 0.5 * height, 0.0]))


def rod(start, end, radius) -> Prim:
    start, end = np.asarray(start, float), np.asarray(end, float)
    return cylinder(0.5 * (start + end), radius, float(np.linalg.norm(end - start)), end - start)


def person(x, y, yaw, phase=0.0, *, height=1.75, reach=None) -> list[Prim]:
    """A walking person: torso, head, swinging legs and arms. ``reach``: world point the right hand holds out to."""
    s = height / 1.75
    turn = rot_z(yaw)
    at = lambda local: np.array([x, y, 0.0]) + turn @ (s * np.asarray(local, float))
    swing = 0.45 * math.sin(phase)
    prims = [box(at([0.0, 0.0, 1.17]), s * np.array([0.22, 0.38, 0.62]), yaw), sphere(at([0.0, 0.0, 1.63]), s * 0.11)]
    for side, sign in ((0.10, 1.0), (-0.10, -1.0)):
        hip = at([0.0, side, 0.90])
        foot = hip + turn @ (s * 0.88 * np.array([math.sin(sign * swing), 0.0, -math.cos(sign * swing)]))
        prims.append(rod(hip, foot, s * 0.075))
    for side, sign in ((0.24, -1.0), (-0.24, 1.0)):
        shoulder = at([0.0, side, 1.43])
        hand = shoulder + turn @ (s * 0.60 * np.array([math.sin(sign * swing), 0.0, -math.cos(sign * swing)]))
        if reach is not None and side < 0.0:
            hand = np.asarray(reach, float)
        prims.append(rod(shoulder, hand, s * 0.045))
    return prims


def table(x, y, yaw=0.0, size=(1.2, 0.7), height=0.74) -> list[Prim]:
    turn = rot_z(yaw)
    prims = [box([x, y, height - 0.02], [size[0], size[1], 0.04], yaw)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner = np.array([x, y, 0.0]) + turn @ np.array([sx * (0.5 * size[0] - 0.05), sy * (0.5 * size[1] - 0.05), 0.5 * (height - 0.04)])
            prims.append(cylinder(corner, 0.025, height - 0.04))
    return prims


def chair(x, y, yaw=0.0) -> list[Prim]:
    turn = rot_z(yaw)
    at = lambda local: np.array([x, y, 0.0]) + turn @ np.asarray(local, float)
    prims = [box(at([0.0, 0.0, 0.45]), [0.42, 0.42, 0.04], yaw), box(at([-0.20, 0.0, 0.72]), [0.03, 0.40, 0.50], yaw)]
    prims += [cylinder(at([sx * 0.18, sy * 0.18, 0.215]), 0.02, 0.43) for sx in (-1, 1) for sy in (-1, 1)]
    return prims


# ---- cameras ----------------------------------------------------------------------------------------------------------
@dataclass
class Camera:
    name: str
    eye: np.ndarray
    rot: np.ndarray                  # [3, 3] camera (x right, y down, z forward) -> world
    width: int = 640
    height: int = 360
    focal: float = 300.0             # px; 640 px wide: 94 degrees
    offset: np.ndarray = field(default_factory=lambda: np.eye(4))     # calibration error applied to its world points

    def rays(self) -> torch.Tensor:
        u = torch.arange(self.width, device=DEVICE, dtype=torch.float32) + 0.5 - 0.5 * self.width
        v = torch.arange(self.height, device=DEVICE, dtype=torch.float32) + 0.5 - 0.5 * self.height
        grid = torch.stack(torch.broadcast_tensors(u[None, :] / self.focal, v[:, None] / self.focal, torch.ones(1, 1, device=DEVICE)), -1)
        return torch.nn.functional.normalize(grid.reshape(-1, 3), dim=1)          # camera frame, [P, 3]


def look_at(name, eye, target, **kwargs) -> Camera:
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return Camera(name, eye, np.stack([right, np.cross(forward, right), forward], axis=1), **kwargs)


def ring(count, radius=2.8, height=1.9, target=(0.0, 0.0, 0.9), start_deg=0.0, span_deg=360.0) -> list[Camera]:
    """``count`` cameras on a circle around ``target``; span 360: evenly around, 180 with 2: opposite each other."""
    step = span_deg / count if span_deg >= 360.0 else span_deg / max(count - 1, 1)
    cameras = []
    for k in range(count):
        angle = math.radians(start_deg + k * step)
        eye = [target[0] + radius * math.cos(angle), target[1] + radius * math.sin(angle), height]
        cameras.append(look_at(f"cam{k}", eye, target))
    return cameras


def miscalibrate(cameras, shift_m: float, angle_deg: float, seed: int = 0) -> None:
    """A small rigid error of every camera's extrinsics (about the workspace centre)."""
    rng = np.random.default_rng(seed)
    for camera in cameras:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = math.radians(angle_deg)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        offset = np.eye(4)
        offset[:3, :3] = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
        direction = rng.normal(size=3)
        offset[:3, 3] = shift_m * direction / np.linalg.norm(direction) + (np.eye(3) - offset[:3, :3]) @ np.array([0.0, 0.0, 0.9])
        camera.offset = offset


# ---- ray casting ------------------------------------------------------------------------------------------------------
def _hits(origin: torch.Tensor, directions: torch.Tensor, prims: list[Prim]) -> torch.Tensor:
    """Distance along every ray to every primitive, inf without a hit. [P, n]"""
    kinds = torch.tensor([prim.kind for prim in prims], device=DEVICE)
    center = torch.tensor(np.stack([prim.center for prim in prims]), device=DEVICE, dtype=torch.float32)
    rot = torch.tensor(np.stack([prim.rot for prim in prims]), device=DEVICE, dtype=torch.float32)
    size = torch.tensor(np.stack([prim.size for prim in prims]), device=DEVICE, dtype=torch.float32)
    local_origin = torch.einsum("nji,nj->ni", rot, origin[None] - center)                  # [n, 3]
    local_dir = torch.einsum("nji,pj->pni", rot, directions)                               # [P, n, 3]
    inf = torch.full((directions.shape[0], len(prims)), float("inf"), device=DEVICE)
    out = inf.clone()

    pick = (kinds == SPHERE).nonzero().flatten()
    if pick.numel():
        o, d, radius = local_origin[pick], local_dir[:, pick], size[pick, 0]
        b = (o[None] * d).sum(-1)
        disc = b * b - ((o * o).sum(-1) - radius * radius)[None]
        t = -b - torch.sqrt(disc.clamp_min(0.0))
        out[:, pick] = torch.where((disc >= 0.0) & (t > 1e-3), t, inf[:, pick])

    pick = (kinds == BOX).nonzero().flatten()
    if pick.numel():
        o, d, half = local_origin[pick], local_dir[:, pick], size[pick]
        inverse = 1.0 / torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
        t1, t2 = (-half[None] - o[None]) * inverse, (half[None] - o[None]) * inverse
        near, far = torch.minimum(t1, t2).amax(-1), torch.maximum(t1, t2).amin(-1)
        out[:, pick] = torch.where((far >= near) & (near > 1e-3), near, inf[:, pick])

    pick = (kinds == CYLINDER).nonzero().flatten()
    if pick.numel():
        o, d, radius, half = local_origin[pick], local_dir[:, pick], size[pick, 0], size[pick, 1]
        a = (d[..., :2] ** 2).sum(-1).clamp_min(1e-12)
        b = (o[None, :, :2] * d[..., :2]).sum(-1)
        disc = b * b - a * ((o[:, :2] ** 2).sum(-1) - radius * radius)[None]
        side = (-b - torch.sqrt(disc.clamp_min(0.0))) / a
        side_z = o[None, :, 2] + side * d[..., 2]
        side = torch.where((disc >= 0.0) & (side > 1e-3) & (side_z.abs() <= half[None]), side, inf[:, pick])
        dz = torch.where(d[..., 2].abs() < 1e-9, torch.full_like(d[..., 2], 1e-9), d[..., 2])
        cap = (-half[None] * torch.sign(dz) - o[None, :, 2]) / dz
        cap_xy = o[None, :, :2] + cap[..., None] * d[..., :2]
        cap = torch.where((cap > 1e-3) & ((cap_xy ** 2).sum(-1) <= (radius * radius)[None]), cap, inf[:, pick])
        out[:, pick] = torch.minimum(side, cap)
    return out


@dataclass
class Rendered:
    distance: torch.Tensor           # [H, W] along the ray to the first surface (inf: nothing)
    behind: torch.Tensor             # [H, W] to the first surface of ANOTHER object behind it
    owner: torch.Tensor              # [H, W] index into the object list, -1: nothing
    directions: torch.Tensor         # [H, W, 3] world


def render(camera: Camera, objects: list[Obj]) -> Rendered:
    prims = [prim for obj in objects for prim in obj.prims]
    prim_owner = torch.tensor([index for index, obj in enumerate(objects) for _ in obj.prims], device=DEVICE)
    directions = camera.rays() @ torch.tensor(camera.rot, device=DEVICE, dtype=torch.float32).T
    hits = _hits(torch.tensor(camera.eye, device=DEVICE, dtype=torch.float32), directions, prims)
    distance, first = hits.min(1)
    owner = torch.where(torch.isfinite(distance), prim_owner[first], torch.full_like(first, -1))
    behind = torch.where(prim_owner[None, :] == owner[:, None], torch.full_like(hits, float("inf")), hits).amin(1)
    shape = (camera.height, camera.width)
    return Rendered(distance.view(shape), behind.view(shape), owner.view(shape), directions.view(*shape, 3))


# ---- what the tracker's 2-D stage hands over --------------------------------------------------------------------------
@dataclass
class Sensing:
    erode_px: int = 2                # mask erosion (the tracker's erosion + depth-boundary filter)
    noise: float = 0.0015            # depth noise sigma = noise * z^2 (m)
    flying: float = 0.0              # share of the outermost kept mask pixels whose depth lies between fore- and background
    min_pixels: int = 40             # smaller masks are not detected
    flicker: float = 0.0             # chance per frame that a detection is missing
    lost_frames: int = 15            # a slot is released after this many frames without its object


class Frame:
    def __init__(self, name: str, stamp_ns: int) -> None:
        self.camera_name, self.timestamp_ns = name, stamp_ns


class Observer:
    """Per camera: the instances of every frame and the slots they keep."""

    def __init__(self, cameras: list[Camera], capacities: dict[str, int], sensing: Sensing, voxelizer, seed: int = 0, gpu: bool = False) -> None:
        """``gpu``: the clouds stay on the device as in the deployed tracker (postprocess.gpu_alignment); the host
        only gets the voxel indices."""
        self.cameras, self.capacities, self.sensing, self.voxelizer, self.gpu = cameras, capacities, sensing, voxelizer, gpu
        self.labels = list(capacities)
        self.generator = torch.Generator(device=DEVICE).manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.slots = [dict() for _ in cameras]        # object ident -> (label, slot)
        self.missing = [dict() for _ in cameras]      # object ident -> frames without it

    def _eroded(self, owner: torch.Tensor, pixels: int) -> torch.Tensor:
        if pixels <= 0:
            return owner
        value = owner[None, None].float()
        size = 2 * pixels + 1
        high = torch.nn.functional.max_pool2d(value, size, 1, pixels)[0, 0]
        low = -torch.nn.functional.max_pool2d(-value, size, 1, pixels)[0, 0]
        return torch.where(high == low, owner, torch.full_like(owner, -1))

    def observe(self, objects: list[Obj], stamp_ns: int, faults: dict | None = None) -> list[FrameResult]:
        """``faults``: {"drop": {(view, ident)}, "label": {(view, ident): label}, "swap": {(view, ident a, ident b)}}"""
        faults = faults or {}
        sensing = self.sensing
        results = []
        for view, camera in enumerate(self.cameras):
            image = render(camera, objects)
            kept = self._eroded(image.owner, sensing.erode_px)
            distance = image.distance
            depth = distance * (image.directions @ torch.tensor(camera.rot[:, 2], device=DEVICE, dtype=torch.float32))
            noisy = distance + sensing.noise * depth * depth * torch.randn(distance.shape, device=DEVICE, generator=self.generator)
            if sensing.flying > 0.0:
                rim = (kept >= 0) & (self._eroded(kept, 1) < 0)
                fly = rim & (torch.rand(distance.shape, device=DEVICE, generator=self.generator) < sensing.flying)
                back = torch.minimum(image.behind, distance + 0.5)
                noisy = torch.where(fly, distance + torch.rand(distance.shape, device=DEVICE, generator=self.generator) * (back - distance), noisy)
            points = torch.tensor(camera.eye, device=DEVICE, dtype=torch.float32) + noisy[..., None] * image.directions
            offset = torch.tensor(camera.offset, device=DEVICE, dtype=torch.float32)
            points = points @ offset[:3, :3].T + offset[:3, 3]

            detected = []
            for index, obj in enumerate(objects):
                label = faults.get("label", {}).get((view, obj.ident), obj.label)
                if label is None or label not in self.capacities or (view, obj.ident) in faults.get("drop", ()):
                    continue
                mask = kept == index
                count = int(mask.sum())
                if count < sensing.min_pixels or self.rng.random() < sensing.flicker:
                    continue
                detected.append((count, obj.ident, label, points[mask].cpu().numpy()))

            slots, missing = self.slots[view], self.missing[view]
            seen = {ident for _, ident, _, _ in detected}
            for ident in list(slots):
                missing[ident] = 0 if ident in seen else missing.get(ident, 0) + 1
                if missing[ident] > sensing.lost_frames:
                    del slots[ident], missing[ident]
            instances = []
            for count, ident, label, cloud in sorted(detected, key=lambda item: -item[0]):      # largest first when slots run out
                if ident not in slots or slots[ident][0] != label:
                    taken = {slot for other, (other_label, slot) in slots.items() if other_label == label}
                    free = [slot for slot in range(self.capacities[label]) if slot not in taken]
                    if not free:
                        continue
                    slots[ident] = (label, free[0])
                    missing[ident] = 0
                instances.append((ident, label, slots[ident][1], cloud))
            swap = {(a, b) for v, a, b in faults.get("swap", ()) if v == view}
            slot_of = {ident: slot for ident, _, slot, _ in instances}
            for a, b in swap:                                   # the 2-D tracker's slots jump between two objects
                if a in slot_of and b in slot_of:
                    slot_of[a], slot_of[b] = slot_of[b], slot_of[a]
            results.append(FrameResult(
                frame=Frame(camera.name, stamp_ns), exclusion_mask_gpu=None, owner_track_map=None, keyframe=False, timings_ms={},
                instances=[self._instance(label, 100 * self.labels.index(label) + slot_of[ident] + 1, cloud, ident)
                           for ident, label, _, cloud in instances],
            ))
        return results

    def _instance(self, label: str, track_id: int, cloud: np.ndarray, ident: int) -> ProcessedInstance:
        coords, keys, points, _, low, high = self.voxelizer.prepare_points(cloud, None)
        instance = ProcessedInstance(
            track_id=track_id, label=label, semantic_confidence=1.0, tracking_confidence=1.0, motion_prediction_confidence=1.0,
            raw_mask=np.empty((0, 0), bool), mask=np.empty((0, 0), bool), points_camera=np.empty((0, 3), np.float32),
            points_world=None if self.gpu else points, colors_rgb=np.empty((0, 3), np.uint8), centroid_camera=None,
            centroid_world=points.mean(0), bbox_min=None, bbox_max=None, bbox_2d_xyxy=(0, 0, 1, 1), status=VisibilityState.VISIBLE,
            voxel_coords=coords, voxel_keys=keys, voxel_points=None if self.gpu else points, voxel_bbox_min=low, voxel_bbox_max=high,
            voxel_points_gpu=torch.from_numpy(points).to(DEVICE) if self.gpu else None,
        )
        instance.truth = ident
        return instance
