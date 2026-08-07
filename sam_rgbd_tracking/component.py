from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .config import Config, load_config
from .data_types import (
    CameraIntrinsics,
    FrameResult,
    ProcessedInstance,
    RGBDFrame,
    TrackState,
    TrackerPrediction,
    TrackerSeed,
    VisibilityState,
)
from .detector import InstanceDetector, build_detector
from .processing import (
    associate,
    backproject_mask,
    bbox_3d,
    blend_depth_model,
    blend_embedding,
    depth_edge_validity,
    erode_and_filter,
    estimate_depth_model,
    exclusive_depth_masks,
    robust_centroid,
    transform_points,
    valid_depth_mask,
)
from .profiler import FrameProfiler
from .trackers import build_tracker
from .trackers.base import MultiObjectTracker

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SAMTrackingComponent:
    """Reusable SAM3 + mask-tracker RGB-D component with no ROS dependency.

    Typical plugin use::

        component = SAMTrackingComponent("configs/tracking.yaml", backend="efficient_tam")
        result = component.process_arrays(
            rgb, depth_m, fx=..., fy=..., cx=..., cy=...,
            world_from_camera=T_world_camera,
        )

    `result.instances` contains persistent IDs, masks and per-instance point clouds.
    """

    def __init__(
        self,
        config: str | Path | Config = "configs/tracking.yaml",
        *,
        camera_name: str = "camera_0",
        backend: str | None = None,
        detector: InstanceDetector | None = None,
        tracker: MultiObjectTracker | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config, tracker=backend)
        elif backend is not None:
            data = config.as_dict()
            if backend is not None:
                data["tracker"]["backend"] = backend
            config = Config(data)
        self.config = config
        self.camera_name = camera_name
        self.detector = detector or build_detector(config)
        self.tracker = tracker or build_tracker(config)
        self.profiler = FrameProfiler(config, name=f"{camera_name}/{config.tracker.backend}")
        self.tracks: dict[int, TrackState] = {}
        self.next_track_id = 1
        self.frame_index = 0
        self.last_keyframe = -10**9
        self.last_prediction: TrackerPrediction | None = None

        hz = float(config.runtime.target_hz)
        self.refresh_frames = max(1, int(round(float(config.detector.refresh_seconds) * hz)))
        phase_seconds = float(config.detector.phase_offsets_seconds.get(camera_name, 0.0))
        self.phase_frames = int(round(phase_seconds * hz))
        self.min_trigger_gap = int(config.detector.min_frames_between_triggers)

        if torch is not None and torch.cuda.is_available() and bool(config.runtime.get("enable_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def process_arrays(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        timestamp_ns: int = 0,
        world_from_camera: np.ndarray | None = None,
    ) -> FrameResult:
        h, w = depth_m.shape
        frame = RGBDFrame(
            camera_name=self.camera_name,
            frame_index=self.frame_index,
            timestamp_ns=int(timestamp_ns),
            rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
            depth_m=np.ascontiguousarray(depth_m, dtype=np.float32),
            intrinsics=CameraIntrinsics(float(fx), float(fy), float(cx), float(cy), w, h),
            world_from_camera=None if world_from_camera is None else np.asarray(world_from_camera, dtype=np.float32),
        )
        self.frame_index += 1
        return self.process(frame)

    def process(self, frame: RGBDFrame) -> FrameResult:
        self.profiler.begin_frame()
        trigger_reasons: list[str] = []
        keyframe = self._periodic_keyframe(frame.frame_index)

        if keyframe:
            trigger_reasons.append("initial" if not self.tracks else "periodic")
            prediction = self._run_keyframe(frame)
        else:
            with self.profiler.stage("tracker_total_gpu", cuda=True):
                prediction = self.tracker.track(frame)
            self._update_raw_observations(prediction)
            if self._anomaly_keyframe(frame.frame_index, prediction):
                trigger_reasons.append("tracking_anomaly")
                prediction = self._run_keyframe(frame)
                keyframe = True

        self.last_prediction = prediction
        with self.profiler.stage("postprocess_cpu", cuda=False):
            instances, owner = self._postprocess(frame, prediction)
        timings = self.profiler.end_frame()
        return FrameResult(
            frame=frame,
            instances=instances,
            owner_track_map=owner,
            keyframe=keyframe,
            timings_ms=timings,
            metadata={"trigger_reasons": trigger_reasons},
        )

    def _periodic_keyframe(self, frame_index: int) -> bool:
        if not self.tracks:
            return True
        if frame_index - self.last_keyframe < self.min_trigger_gap:
            return False
        return (frame_index - self.phase_frames) % self.refresh_frames == 0

    def _anomaly_keyframe(self, frame_index: int, prediction: TrackerPrediction) -> bool:
        if not bool(self.config.detector.trigger_on_anomaly):
            return False
        if frame_index - self.last_keyframe < self.min_trigger_gap:
            return False
        if prediction.presence_scores.size == 0:
            return True
        threshold = float(self.config.detector.get("anomaly_presence_threshold", 0.05))
        return bool(float(np.min(prediction.presence_scores)) < threshold)

    def _run_keyframe(self, frame: RGBDFrame) -> TrackerPrediction:
        with self.profiler.stage("sam3_total_gpu", cuda=True):
            detections = self.detector.detect(frame)
        seeds = self._associate_and_seed(frame, detections)
        self._bootstrap_depth_models(frame, seeds)
        with self.profiler.stage("tracker_total_gpu", cuda=True):
            prediction = self.tracker.correct(frame, seeds)
        self.last_keyframe = frame.frame_index
        return prediction

    def _associate_and_seed(self, frame: RGBDFrame, detections) -> list[TrackerSeed]:
        centroids = [robust_centroid(frame.depth_m, det.mask, frame.intrinsics) for det in detections]
        if not self.tracks:
            seeds = []
            for det, centroid in zip(detections, centroids):
                track_id = self._new_track_id()
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    label=det.label,
                    semantic_confidence=det.score,
                    embedding=det.embedding,
                    last_mask=det.mask.copy(),
                    last_raw_mask=det.mask.copy(),
                    centroid_camera=centroid,
                    last_seen_frame=frame.frame_index,
                )
                seeds.append(TrackerSeed(track_id, det.mask, det.label, det.score))
            return seeds

        matches, unmatched_d, unmatched_t = associate(
            detections, self.tracks, centroids, self.config.association
        )
        seeds: list[TrackerSeed] = []
        for d_idx, track_id in matches:
            det = detections[d_idx]
            track = self.tracks[track_id]
            track.label = det.label
            track.semantic_confidence = det.score
            track.embedding = blend_embedding(track.embedding, det.embedding)
            track.last_raw_mask = None if track.last_mask is None else track.last_mask.copy()
            track.last_mask = det.mask.copy()
            track.centroid_camera = centroids[d_idx]
            track.last_seen_frame = frame.frame_index
            track.missing_frames = 0
            seeds.append(TrackerSeed(track_id, det.mask, det.label, det.score))

        for d_idx in sorted(unmatched_d):
            det = detections[d_idx]
            track_id = self._new_track_id()
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                label=det.label,
                semantic_confidence=det.score,
                embedding=det.embedding,
                last_mask=det.mask.copy(),
                last_raw_mask=det.mask.copy(),
                centroid_camera=centroids[d_idx],
                last_seen_frame=frame.frame_index,
            )
            seeds.append(TrackerSeed(track_id, det.mask, det.label, det.score))

        ttl = int(self.config.association.lost_ttl_frames)
        for track_id in unmatched_t:
            track = self.tracks[track_id]
            track.missing_frames += 1
            fallback = track.last_raw_mask if track.last_raw_mask is not None else track.last_mask
            if track.missing_frames <= ttl and fallback is not None and fallback.any():
                seeds.append(TrackerSeed(track_id, fallback, track.label, track.semantic_confidence))

        for track_id in [k for k, v in self.tracks.items() if v.missing_frames > ttl]:
            self.tracks.pop(track_id, None)
        return seeds

    def _bootstrap_depth_models(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> None:
        min_pixels = int(self.config.postprocess.depth_model_min_pixels)
        for seed in seeds:
            model = estimate_depth_model(frame.depth_m, seed.mask, min_pixels)
            track = self.tracks[seed.track_id]
            track.depth_model = blend_depth_model(track.depth_model, model, alpha=0.4)

    def _ensure_logits(self, prediction: TrackerPrediction, height: int, width: int) -> np.ndarray:
        logits = np.asarray(prediction.mask_logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[None]
        if logits.size == 0:
            return np.empty((0, height, width), np.float32)
        if logits.shape[-2:] != (height, width):
            logits = np.stack(
                [cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) for mask in logits],
                axis=0,
            )
        return logits

    def _update_raw_observations(self, prediction: TrackerPrediction) -> None:
        logits = np.asarray(prediction.mask_logits)
        if logits.ndim == 2:
            logits = logits[None]
        for channel, track_id in enumerate(prediction.track_ids):
            track = self.tracks.get(int(track_id))
            if track is None or channel >= logits.shape[0]:
                continue
            track.last_raw_mask = None if track.last_mask is None else track.last_mask.copy()
            track.last_mask = logits[channel] > float(self.config.postprocess.mask_threshold)
            if channel < prediction.presence_scores.size:
                track.tracking_confidence = float(prediction.presence_scores[channel])

    def _postprocess(self, frame: RGBDFrame, prediction: TrackerPrediction) -> tuple[list[ProcessedInstance], np.ndarray]:
        h, w = frame.depth_m.shape
        logits = self._ensure_logits(prediction, h, w)
        valid_channels = [
            index for index, value in enumerate(prediction.track_ids)
            if int(value) in self.tracks and index < logits.shape[0]
        ]
        track_ids = [int(prediction.track_ids[index]) for index in valid_channels]
        logits = logits[valid_channels] if valid_channels else np.empty((0, h, w), np.float32)
        owner_track_map = np.zeros((h, w), dtype=np.int32)
        if not track_ids:
            return [], owner_track_map

        valid = valid_depth_mask(
            frame.depth_m,
            float(self.config.postprocess.min_valid_depth_m),
            float(self.config.postprocess.max_valid_depth_m),
        )
        valid &= depth_edge_validity(
            frame.depth_m, float(self.config.postprocess.depth_edge_threshold_m)
        )
        exclusive, depth_consistency = exclusive_depth_masks(
            frame.depth_m,
            logits,
            track_ids,
            self.tracks,
            valid,
            float(self.config.postprocess.mask_threshold),
            float(self.config.postprocess.depth_gate_mad_scale),
            float(self.config.postprocess.depth_gate_min_m),
            float(self.config.postprocess.depth_gate_max_m),
            overlap_depth_only=bool(self.config.postprocess.overlap_depth_only),
            logit_weight=float(self.config.postprocess.logit_weight),
        )

        processed: list[ProcessedInstance] = []
        for channel, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            raw_mask = logits[channel] > float(self.config.postprocess.mask_threshold)
            final_mask = erode_and_filter(
                exclusive[channel],
                int(self.config.postprocess.erosion_pixels),
                int(self.config.postprocess.min_component_pixels),
            )
            owner_track_map[final_mask] = track_id
            points_camera, colors = backproject_mask(
                frame.depth_m,
                frame.rgb,
                final_mask,
                frame.intrinsics,
                int(self.config.pointcloud.stride),
                int(self.config.pointcloud.max_points_per_instance),
            )
            points_world = transform_points(points_camera, frame.world_from_camera)
            centroid_camera = None if points_camera.size == 0 else np.median(points_camera, axis=0).astype(np.float32)
            centroid_world = None if points_world is None or points_world.size == 0 else np.median(points_world, axis=0).astype(np.float32)
            bounds_min, bounds_max = bbox_3d(points_world if points_world is not None else points_camera)
            raw_pixels = max(int(raw_mask.sum()), 1)
            visible_ratio = float(final_mask.sum() / raw_pixels)
            consistency = depth_consistency[channel] if channel < len(depth_consistency) else 0.0
            status = self._visibility(visible_ratio, track.missing_frames)
            current_model = estimate_depth_model(
                frame.depth_m, final_mask, int(self.config.postprocess.depth_model_min_pixels)
            )
            track.depth_model = blend_depth_model(track.depth_model, current_model)
            track.last_raw_mask = raw_mask
            track.last_mask = final_mask
            track.centroid_camera = centroid_camera
            track.centroid_world = centroid_world
            track.visible_ratio = visible_ratio
            if channel < prediction.presence_scores.size:
                track.tracking_confidence = float(prediction.presence_scores[channel])
            motion_conf = float(
                np.clip(
                    track.tracking_confidence * math.sqrt(max(visible_ratio, 0.0)) * max(consistency, 0.0),
                    0.0,
                    1.0,
                )
            )
            if final_mask.any():
                track.last_seen_frame = frame.frame_index
                track.missing_frames = 0
            else:
                track.missing_frames += 1
            processed.append(
                ProcessedInstance(
                    track_id=track_id,
                    label=track.label,
                    semantic_confidence=track.semantic_confidence,
                    tracking_confidence=track.tracking_confidence,
                    motion_prediction_confidence=motion_conf,
                    raw_mask=raw_mask,
                    mask=final_mask,
                    points_camera=points_camera,
                    points_world=points_world,
                    colors_rgb=colors,
                    centroid_camera=centroid_camera,
                    centroid_world=centroid_world,
                    bbox_min=bounds_min,
                    bbox_max=bounds_max,
                    visible_ratio=visible_ratio,
                    depth_consistency=float(consistency),
                    status=status,
                )
            )
        return processed, owner_track_map

    def _visibility(self, visible_ratio: float, missing_frames: int) -> VisibilityState:
        if missing_frames > int(self.config.association.lost_ttl_frames):
            return VisibilityState.LOST
        if visible_ratio >= float(self.config.postprocess.visible_ratio_visible):
            return VisibilityState.VISIBLE
        if visible_ratio >= float(self.config.postprocess.visible_ratio_partial):
            return VisibilityState.PARTIAL
        return VisibilityState.OCCLUDED

    def _new_track_id(self) -> int:
        value = self.next_track_id
        self.next_track_id += 1
        return value

    def print_stats(self) -> None:
        self.profiler.print_summary()

    def close(self) -> None:
        self.tracker.close()
