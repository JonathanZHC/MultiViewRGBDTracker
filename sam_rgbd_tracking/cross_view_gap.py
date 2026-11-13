"""Cross-view association by surface gap under a per-class instance count (shared_voxel_grid.association: gap).

Which observations of the cameras are the same object is decided per semantic class from two things:

  gap     how far the two observed surfaces are apart: the distance within which ``quantile`` of one cloud's voxels
          find the other cloud (the smaller of the two directions, metres). Two views of one object give a few
          centimetres whatever its size, also when their boxes do not overlap at all (each camera sees another part
          of it); two objects give their distance. The world boxes are only used to skip pairs that are far apart.
  count   a class has at most ``capacity`` instances (the prompt capacities). Observations merge on their own below
          ``merge_m``. If more groups than ``capacity`` are left, the merge threshold is raised just far enough for
          the count to fit (never beyond ``forced_max_m``): the nearest observations are then the same object without
          any absolute threshold having to hold. A count that cannot be reached is left as it is - nothing is dropped.

Per pair of cameras the observations are matched one to one with the least total gap (Hungarian), the matches of all
camera pairs are united smallest gap first with at most one observation per camera in a group.

Over time a pair is known by its two per-camera slots (the 2-D tracker keeps a slot on its object). A merged pair stays
merged up to ``split_m``, so a gap near the threshold does not flicker; a merge forced by the count outlives the count
by ``memory_s`` (another object of the class hidden for a moment). Nothing is smoothed: when the geometry changes - a
slot jumps to another object - the grouping follows in the same frame. A known pair is measured again every
``measure_every`` frames, or at once when its two boxes have moved against each other by half of what its gap is
away from the threshold that would change its state (the gap cannot change by more than the clouds move). Every gap
is measured the same way, up to ``forced_max_m``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

_ALL_MATCHES = 1.0e3      # a threshold above every gap: as many matches as possible, least total gap


@dataclass(frozen=True)
class GapSettings:
    # Two views of one object: gap 1 - 3 cm, up to 12 cm for cameras opposite each other (they share only the
    # eroded silhouette rim); thinning a large cloud to ``samples`` adds up to 3 cm (scripts/benchmark_association.py).
    merge_m: float = 0.15          # a gap below this merges on its own
    split_m: float = 0.20          # a merged pair stays merged up to this gap
    forced_max_m: float = 0.50     # the count may force a merge up to this gap
    quantile: float = 0.05         # share of a cloud that has to lie within the gap of the other
    memory_s: float = 0.3          # a forced merge outlives the count by this; a pair unseen for longer starts anew
    samples: int = 96              # voxels per cloud the gap is measured on
    measure_every: int = 3         # frames between two measurements of a pair whose boxes keep their offset (1: every frame)

    @classmethod
    def from_config(cls, values) -> "GapSettings":
        values = dict(values.as_dict() if hasattr(values, "as_dict") else values or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"shared_voxel_grid.gap: unknown keys {sorted(unknown)}")
        settings = cls(**{name: type(getattr(cls, name))(value) for name, value in values.items()})
        if not 0.0 < settings.merge_m <= settings.split_m <= settings.forced_max_m:
            raise ValueError("shared_voxel_grid.gap needs 0 < merge_m <= split_m <= forced_max_m")
        if not 0.0 < settings.quantile < 1.0 or settings.samples < 8 or settings.memory_s < 0.0 or settings.measure_every < 1:
            raise ValueError("shared_voxel_grid.gap: quantile in (0, 1), samples >= 8, memory_s >= 0, measure_every >= 1")
        return settings


# ----------------------------------------------------------------------------------------------------------------------
# The gap of two voxel clouds (voxel units)
# ----------------------------------------------------------------------------------------------------------------------
def box_gap(lo_a, hi_a, lo_b, hi_b):
    """Distance between two voxel index boxes (0 when they overlap); never more than the gap of the clouds in them.
    Broadcasts over leading axes."""
    separation = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0)
    return np.sqrt((separation * separation).sum(-1, dtype=np.float64))


class Cloud:
    """One observation's voxels, thinned once per frame for all the gaps it takes part in."""

    __slots__ = ("count", "lo", "hi", "points", "weight")
    LIMIT = 512                    # voxels kept; the region near another cloud is cut out of these

    def __init__(self, coords: np.ndarray, lo, hi) -> None:
        self.count = len(coords)
        self.lo = (float(lo[0]), float(lo[1]), float(lo[2]))
        self.hi = (float(hi[0]), float(hi[1]), float(hi[2]))
        self.points = coords[:: -(-self.count // self.LIMIT)].astype(np.float32)
        self.weight = self.count / len(self.points)            # voxels a kept voxel stands for

    def near(self, other: "Cloud", reach: float, samples: int):
        """At most ``samples`` of the voxels within ``reach`` of the other cloud's box, and how many voxels that
        region holds. Voxels farther from the box cannot be within ``reach`` of the cloud."""
        lo, hi = other.lo, other.hi
        points = self.points
        if not (self.lo[0] >= lo[0] - reach and self.lo[1] >= lo[1] - reach and self.lo[2] >= lo[2] - reach
                and self.hi[0] <= hi[0] + reach and self.hi[1] <= hi[1] + reach and self.hi[2] <= hi[2] + reach):
            low = np.array([lo[0] - reach, lo[1] - reach, lo[2] - reach], np.float32)
            high = np.array([hi[0] + reach, hi[1] + reach, hi[2] + reach], np.float32)
            points = points[((points >= low) & (points <= high)).all(1)]
        held = len(points) * self.weight
        return points[:: -(-len(points) // samples)] if len(points) > samples else points, held


def surface_gap(a: Cloud, b: Cloud, *, reach: float, quantile: float, samples: int) -> float:
    """The smaller of the two directed gaps of the clouds, capped at ``reach``.

    The directed gap a -> b is the distance within which ``quantile`` of a's voxels find a voxel of b. Only voxels
    within ``reach`` of the other cloud's box can count, so both clouds are cut to that region first and the distances
    are taken between at most ``samples`` evenly thinned voxels of each: where two partial views of an object meet the
    sampling stays dense, whatever the size of the object.
    """
    points_a, held_a = a.near(b, reach, samples)
    points_b, held_b = b.near(a, reach, samples)
    need_a = max(1.0, quantile * a.count)
    need_b = max(1.0, quantile * b.count)
    if len(points_a) == 0 or len(points_b) == 0 or (held_a < need_a and held_b < need_b):
        return reach
    origin = points_a[0]
    points_a = points_a - origin
    points_b = points_b - origin
    squared = (points_a * points_a).sum(1)[:, None] + (points_b * points_b).sum(1)[None, :] - 2.0 * (points_a @ points_b.T)
    gap = reach * reach
    if held_a >= need_a:                # the need-th nearest of the region, on its sample
        nearest = np.sort(squared.min(1))
        gap = min(gap, float(nearest[min(len(nearest), math.ceil(need_a * len(nearest) / held_a)) - 1]))
    if held_b >= need_b:
        nearest = np.sort(squared.min(0))
        gap = min(gap, float(nearest[min(len(nearest), math.ceil(need_b * len(nearest) / held_b)) - 1]))
    return math.sqrt(max(gap, 0.0))


# ----------------------------------------------------------------------------------------------------------------------
# One class: who is the same object
# ----------------------------------------------------------------------------------------------------------------------
def _matches(view, cost, threshold: float, limit: float):
    """Per pair of cameras the one-to-one matching with the least total gap among the pairs below ``threshold``."""
    edges = []
    members: dict[int, list[int]] = {}
    for index, camera in enumerate(view):
        members.setdefault(camera, []).append(index)
    for camera_a, camera_b in combinations(sorted(members), 2):
        rows, cols = members[camera_a], members[camera_b]
        gaps = cost[np.ix_(rows, cols)]
        allowed = (gaps < threshold) & (gaps <= limit)
        if not allowed.any():
            continue
        if gaps.size == 1:
            edges.append((float(gaps[0, 0]), rows[0], cols[0]))
            continue
        # an allowed pair is worth threshold - gap, any other assignment nothing
        for row, col in zip(*linear_sum_assignment(np.where(allowed, gaps - threshold, 0.0))):
            if allowed[row, col]:
                edges.append((float(gaps[row, col]), rows[row], cols[col]))
    return edges


def _unite(view, edges):
    """Unite along the edges, smallest gap first, at most one observation per camera in a group.
    -> the edges that united two groups."""
    parent = list(range(len(view)))
    cameras = [{camera} for camera in view]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    united = []
    for edge in sorted(edges):
        root_a, root_b = find(edge[1]), find(edge[2])
        if root_a == root_b or cameras[root_a] & cameras[root_b]:
            continue
        parent[root_b] = root_a
        cameras[root_a] |= cameras[root_b]
        united.append(edge)
    return united


def associate_class(
    view: list[int],
    near: np.ndarray,
    capacity: int | None,
    *,
    merge_m: float,
    forced_max_m: float,
    far: Callable[[int, int], float] | None = None,
) -> list[tuple[float, int, int, bool]]:
    """The edges (gap, i, j, forced by the count) between observations of one class that are the same object.

    ``view[i]`` is the camera of observation i, ``near[i, j]`` the gap that decides a merge on its own (inf within a
    camera). ``far(i, j)`` is asked for the gap of a pair that did not merge on its own, and only when the count
    ``capacity`` forces merges.
    """
    count = len(view)
    if count == 2:                      # one observation in each of two cameras
        gap = float(near[0, 1])
        if gap < merge_m:
            return [(gap, 0, 1, False)]
        if capacity is not None and capacity < 2 and far is not None:
            gap = far(0, 1)
            return [(gap, 0, 1, True)] if gap <= forced_max_m else []
        return []
    free = _unite(view, _matches(view, near, merge_m, merge_m))
    if capacity is None or count - len(free) <= capacity or far is None:
        return [(*edge, False) for edge in free]

    cost = near.copy()
    for i, j in combinations(range(count), 2):
        if view[i] != view[j] and not near[i, j] < merge_m:
            cost[i, j] = cost[j, i] = max(far(i, j), merge_m)
    united = free
    candidates = np.unique(cost[(cost >= merge_m) & (cost <= forced_max_m)])
    for threshold in [float(value) + 1.0e-6 for value in candidates] + [_ALL_MATCHES]:
        united = _unite(view, _matches(view, cost, threshold, forced_max_m))
        if count - len(united) <= capacity:
            break
    # forced merges the count does not need are undone, largest gap first
    for edge in sorted((edge for edge in united if edge[0] >= merge_m), reverse=True):
        without = _unite(view, [other for other in united if other is not edge])
        if count - len(without) <= capacity:
            united = without
    return [(*edge, edge[0] >= merge_m) for edge in united]


# ----------------------------------------------------------------------------------------------------------------------
# All classes, over time
# ----------------------------------------------------------------------------------------------------------------------
_NEVER = -(10 ** 9)


@dataclass
class _Pair:
    """What is kept of two per-camera slots whose boxes are close, or that the count merged."""

    label: str
    gap_m: float                           # last measurement (capped at forced_max_m)
    offset: tuple                          # of the two box centres when it was measured (voxels)
    measured_frame: int
    last_frame: int                        # both slots were observed
    merged: bool = False
    forced_frame: int = _NEVER             # last frame the count forced this merge


class GapAssociation:
    """The cross-view grouping of every frame.

    An observation needs ``view_index``, ``instance.label``, ``instance.track_id`` (the camera's slot), ``coords``
    (unique voxel indices [N, 3]), ``bbox_min`` and ``bbox_max`` (voxel index box).
    """

    def __init__(self, settings: GapSettings, voxel_size_m: float, capacities: dict[str, int] | None) -> None:
        self.settings = settings
        self.voxel_size_m = float(voxel_size_m)
        self.capacities = dict(capacities or {})
        self._pairs: dict[tuple, _Pair] = {}           # ((camera, slot), (camera, slot)) -> _Pair
        self._forced: dict[str, dict[tuple, _Pair]] = {}       # label -> the pairs the count merged recently
        self._frame = 0
        self._memory = 3                               # memory_s in frames
        self._jitter = 0.02 / self.voxel_size_m       # box centres move this much without the object moving (voxels)
        self.candidate_pairs = 0                       # of the last frame
        self.measured_pairs = 0
        self.forced_merges = 0
        self.over_capacity = 0

    def associate(self, observations, dt_s: float) -> list[tuple[float, int, int]]:
        """-> edges (gap m, index a, index b) into ``observations``: the two are the same object."""
        self._frame += 1
        self.candidate_pairs = self.measured_pairs = self.forced_merges = self.over_capacity = 0
        self._memory = max(1, int(round(self.settings.memory_s / max(dt_s, 1.0e-3))))
        by_label: dict[str, list[int]] = {}
        for index, observation in enumerate(observations):
            if observation.coords is not None and observation.bbox_min is not None:
                by_label.setdefault(observation.instance.label, []).append(index)
        edges = []
        for label, members in by_label.items():
            if len(members) > 1:
                edges.extend((gap, members[i], members[j]) for gap, i, j, _ in self._class_edges(label, [observations[index] for index in members]))
        if self._frame % 256 == 0:
            for name in [name for name, pair in self._pairs.items() if self._frame - pair.last_frame > self._memory]:
                del self._pairs[name]
        return edges

    def _class_edges(self, label: str, observations) -> list[tuple[float, int, int, bool]]:
        """The observations of one class: edges (gap m, i, j, forced) between those that are the same object."""
        settings = self.settings
        voxel = self.voxel_size_m
        frame = self._frame
        view = [observation.view_index for observation in observations]
        lo = [observation.bbox_min.tolist() for observation in observations]
        hi = [observation.bbox_max.tolist() for observation in observations]
        boxes_m, close = self._box_gaps(view, lo, hi)
        if not np.isfinite(boxes_m).any():
            return []                                      # all in one camera
        # the camera's slot identifies an observation over time
        key = [(camera, int(observation.instance.track_id)) for camera, observation in zip(view, observations)]
        clouds: dict[int, Cloud] = {}
        pairs: dict[tuple[int, int], _Pair] = {}           # the pairs whose gap is known in this frame

        def offset(i: int, j: int) -> tuple:               # of the two box centres (voxels)
            return tuple(0.5 * (lo[i][axis] + hi[i][axis] - lo[j][axis] - hi[j][axis]) for axis in range(3))

        def measure(i: int, j: int) -> float:
            for k in (i, j):
                if k not in clouds:
                    clouds[k] = Cloud(observations[k].coords, lo[k], hi[k])
            self.measured_pairs += 1
            return voxel * surface_gap(clouds[i], clouds[j], reach=settings.forced_max_m / voxel, quantile=settings.quantile, samples=settings.samples)

        def known(i: int, j: int) -> _Pair:
            """The pair with its gap: measured now if it is new, due, or its boxes moved against each other."""
            pair = pairs.get((i, j))
            if pair is not None:
                return pair
            name = _name(key[i], key[j])
            pair = self._pairs.get(name)
            if pair is None or frame - pair.last_frame > self._memory:
                pair = self._pairs[name] = _Pair(label, measure(i, j), offset(i, j), frame, frame)
            elif frame - pair.forced_frame > self._memory and (
                    frame - pair.measured_frame >= settings.measure_every or self._shifted(offset(i, j), pair.offset, self._margin(pair))):
                pair.gap_m, pair.offset, pair.measured_frame = measure(i, j), offset(i, j), frame
            pair.last_frame = frame
            pairs[(i, j)] = pair
            return pair

        def far(i: int, j: int) -> float:                  # asked for when the count forces merges
            return known(i, j).gap_m if boxes_m[i, j] < settings.forced_max_m else float(boxes_m[i, j])

        near = boxes_m.copy()                              # pairs that are apart: the distance of their boxes
        hold = settings.split_m - settings.merge_m
        for i, j in close:
            pair = known(i, j)
            # a merged pair stays merged up to split_m: it competes as if it were that much closer
            near[i, j] = near[j, i] = pair.gap_m - (hold if pair.merged else 0.0)
        forced = self._forced.get(label)
        if forced:                                         # merged by the count a moment ago: the merge holds
            index_of = {name: index for index, name in enumerate(key)}
            for name, pair in list(forced.items()):
                if frame - pair.forced_frame > self._memory:
                    del forced[name]
                    continue
                i, j = index_of.get(name[0]), index_of.get(name[1])
                if i is not None and j is not None and boxes_m[i, j] < settings.forced_max_m:
                    i, j = min(i, j), max(i, j)
                    pair.last_frame = frame
                    pairs[(i, j)] = pair
                    near[i, j] = near[j, i] = min(near[i, j], 0.999 * settings.merge_m)

        capacity = self.capacities.get(label)
        united = associate_class(view, near, capacity, merge_m=settings.merge_m, forced_max_m=settings.forced_max_m, far=far)
        if capacity is not None and len(view) - len(united) > capacity:
            self.over_capacity += 1
        merged = {(min(i, j), max(i, j)): is_forced for _, i, j, is_forced in united}
        for index_pair, pair in pairs.items():
            pair.merged = index_pair in merged
        for index_pair, is_forced in merged.items():
            if is_forced:
                self.forced_merges += 1
                pair = pairs[index_pair]
                pair.forced_frame = frame
                self._forced.setdefault(label, {})[_name(key[index_pair[0]], key[index_pair[1]])] = pair
        return united

    def _margin(self, pair: _Pair) -> float:
        """How far the pair's gap is from the nearest threshold that would change its state (m)."""
        settings = self.settings
        state = settings.split_m if pair.merged else settings.merge_m
        return min(abs(pair.gap_m - state), abs(settings.forced_max_m - pair.gap_m))

    def _box_gaps(self, view, lo, hi):
        """-> ([n, n] distance of the boxes of every two observations of different cameras (m), inf within a camera;
        the pairs (i < j) whose boxes are closer than split_m)."""
        count = len(view)
        split = self.settings.split_m
        if count <= 6:                                     # few pairs: plain arithmetic is cheaper than array set-up
            gaps = np.full((count, count), np.inf)
            close = []
            for i in range(count):
                for j in range(i + 1, count):
                    if view[i] == view[j]:
                        continue
                    self.candidate_pairs += 1
                    total = 0.0
                    for axis in range(3):
                        separation = max(lo[i][axis] - hi[j][axis], lo[j][axis] - hi[i][axis], 0.0)
                        total += separation * separation
                    gap = self.voxel_size_m * math.sqrt(total)
                    gaps[i, j] = gaps[j, i] = gap
                    if gap < split:
                        close.append((i, j))
            return gaps, close
        low, high, cameras = np.asarray(lo), np.asarray(hi), np.asarray(view)
        other_camera = cameras[:, None] != cameras[None, :]
        gaps = np.where(other_camera, self.voxel_size_m * box_gap(low[:, None], high[:, None], low[None, :], high[None, :]), np.inf)
        self.candidate_pairs += int(other_camera.sum()) // 2
        rows, cols = np.nonzero(np.triu(gaps < split, 1))
        return gaps, list(zip(rows.tolist(), cols.tolist()))

    def _shifted(self, offset_a, offset_b, margin_m: float) -> bool:
        """The two boxes moved against each other by more than half of ``margin_m`` (and more than they jitter)."""
        limit = max(self._jitter, 0.5 * margin_m / self.voxel_size_m)
        return (abs(offset_a[0] - offset_b[0]) > limit or abs(offset_a[1] - offset_b[1]) > limit or abs(offset_a[2] - offset_b[2]) > limit)


def _name(key_a, key_b):
    return (key_a, key_b) if key_a < key_b else (key_b, key_a)
