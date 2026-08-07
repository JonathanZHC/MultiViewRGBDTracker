from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .appearance import blend_embedding
from .association import associate_detections
from .data_types import (
    DetectionInstance,
    FrameResult,
    ProcessedInstance,
    RGBDFrame,
    TrackState,
    TrackerPrediction,
    TrackerSeed,
    VisibilityState,
)
from .depth_assignment import assign_depth_ownership, blend_depth_model, estimate_depth_model
from .detector import InstanceDetector, build_detector
from .geometry import (
    backproject_mask,
    bbox_3d,
    robust_centroid,
    transform_points,
    valid_depth_mask,
)
from .mask_ops import (
    channel_owner_to_track_owner,
    depth_edge_validity,
    erode_mask,
    ensure_nhw,
    keep_large_components,
)
from .profiler import FrameProfiler
from .trackers import build_tracker
from .trackers.base import MultiObjectTracker
from .triggers import KeyframeTrigger


class CameraTrackingPipeline:
    """One independent tracking state per RGB-D camera."""

    def __init__(
        self,
        camera_name: str,
        config: Any,
        detector: InstanceDetector | None = None,
        tracker: MultiObjectTracker | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.config = config
        self.detector = detector or build_detector(config)
        self.tracker = tracker or build_tracker(config)
        self.tracks: dict[int, TrackState] = {}
        self.next_track_id = 1
        target_hz = float(config.runtime.target_hz)
        phase_seconds = float(config.detector.phase_offsets_seconds.get(camera_name, 0.0))
        self.phase_offset_frames = round(phase_seconds * target_hz)
        self.trigger = KeyframeTrigger(
            refresh_seconds=float(config.detector.refresh_seconds),
            target_hz=target_hz,
            min_frames_between_triggers=int(config.detector.min_frames_between_triggers),
        )
        log_dir = Path(config.runtime.log_dir) / camera_name / config.tracker.backend
        self.profiler = FrameProfiler(
            csv_path=str(log_dir / "timing.csv"),
            jsonl_path=str(log_dir / "frames.jsonl"),
            use_cuda_events=bool(config.profiling.cuda_events),
            summary_interval=int(config.profiling.summary_interval_frames),
        )
        self.last_prediction: TrackerPrediction | None = None

    def process(self, frame: RGBDFrame, dropped_frames: int = 0) -> FrameResult:
        self.profiler.begin_frame(
            camera=self.camera_name,
            frame_index=frame.frame_index,
            stamp_ns=frame.stamp_ns,
            tracker_backend=self.config.tracker.backend,
            detector_backend=self.config.detector.backend,
            dropped_frames=int(dropped_frames),
        )
        keyframe = False
        anomaly_triggered = False
        trigger_reasons: list[str] = []

        periodic_or_initial = self.trigger.evaluate(
            frame_index=frame.frame_index,
            depth_m=frame.depth_m,
            tracks=self.tracks,
            force=not self.tracks,
            phase_offset_frames=self.phase_offset_frames,
        )

        if not self.tracks or periodic_or_initial.trigger:
            trigger_reasons = periodic_or_initial.reasons
            prediction = self._run_keyframe(frame)
            keyframe = True
        else:
            with self.profiler.stage("tracker_total", cuda=True):
                prediction = self.tracker.track(frame)
            self._update_raw_track_observations(prediction)
            if bool(self.config.detector.trigger_on_anomaly):
                anomaly = self.trigger.evaluate(
                    frame_index=frame.frame_index,
                    depth_m=frame.depth_m,
                    tracks=self.tracks,
                    force=False,
                    phase_offset_frames=10**8,
                )
                if anomaly.trigger:
                    trigger_reasons = anomaly.reasons
                    prediction = self._run_keyframe(frame)
                    keyframe = True
                    anomaly_triggered = True

        self.last_prediction = prediction
        with self.profiler.stage("postprocess", cuda=False):
            instances, owner_track_map = self._postprocess(frame, prediction)

        timings = self.profiler.end_frame(
            keyframe=keyframe,
            anomaly_triggered=anomaly_triggered,
            trigger_reasons=trigger_reasons,
            num_tracks=len(instances),
        )
        return FrameResult(
            frame=frame,
            instances=instances,
            owner_track_map=owner_track_map,
            keyframe=keyframe,
            anomaly_triggered=anomaly_triggered,
            timings_ms=timings,
            metadata={"trigger_reasons": trigger_reasons},
        )

    def _run_keyframe(self, frame: RGBDFrame) -> TrackerPrediction:
        with self.profiler.stage("sam3_total", cuda=True):
            detections = self.detector.detect(frame)
        with self.profiler.stage("keyframe_association"):
            seeds = self._associate_and_make_seeds(frame, detections)
            self._bootstrap_depth_models(frame, seeds)
        with self.profiler.stage("tracker_correction", cuda=True):
            return self.tracker.correct(frame, seeds)

    def _associate_and_make_seeds(
        self,
        frame: RGBDFrame,
        detections: list[DetectionInstance],
    ) -> list[TrackerSeed]:
        if not self.tracks:
            seeds: list[TrackerSeed] = []
            for detection in detections:
                track_id = self._allocate_track_id()
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    label=detection.label,
                    semantic_confidence=detection.score,
                    last_mask=detection.mask.copy(),
                    last_raw_mask=detection.mask.copy(),
                    last_seen_frame=frame.frame_index,
                    appearance_embedding=detection.embedding,
                )
                seeds.append(TrackerSeed(track_id, detection.mask, detection.label, detection.score))
            return seeds

        association = associate_detections(
            self.tracks,
            detections,
            frame.depth_m,
            frame.intrinsics.fx,
            frame.intrinsics.fy,
            frame.intrinsics.cx,
            frame.intrinsics.cy,
            max_centroid_distance_m=float(self.config.association.max_centroid_distance_m),
            min_mask_iou=float(self.config.association.min_mask_iou),
            label_mismatch_cost=float(self.config.association.label_mismatch_cost),
            weight_centroid=float(self.config.association.weight_centroid),
            weight_iou=float(self.config.association.weight_iou),
            weight_depth=float(self.config.association.weight_depth),
            weight_embedding=float(self.config.association.weight_embedding),
        )
        seeds = []
        matched_track_ids: set[int] = set()
        for track_id, detection_index in association.matches:
            detection = detections[detection_index]
            track = self.tracks[track_id]
            track.label = detection.label
            track.semantic_confidence = detection.score
            track.appearance_embedding = blend_embedding(track.appearance_embedding, detection.embedding)
            track.last_raw_mask = track.last_mask.copy()
            track.last_mask = detection.mask.copy()
            track.last_seen_frame = frame.frame_index
            track.missing_frames = 0
            matched_track_ids.add(track_id)
            seeds.append(TrackerSeed(track_id, detection.mask, detection.label, detection.score))

        for detection_index in association.unmatched_detection_indices:
            detection = detections[detection_index]
            track_id = self._allocate_track_id()
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                label=detection.label,
                semantic_confidence=detection.score,
                last_mask=detection.mask.copy(),
                last_raw_mask=detection.mask.copy(),
                last_seen_frame=frame.frame_index,
                appearance_embedding=detection.embedding,
            )
            seeds.append(TrackerSeed(track_id, detection.mask, detection.label, detection.score))

        ttl = int(self.config.association.lost_ttl_frames)
        for track_id in association.unmatched_track_ids:
            track = self.tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames <= ttl and track.last_raw_mask.any():
                # Keep the tracker's current hypothesis as a conditioning mask. Depth
                # filtering will still suppress any foreground pixels assigned to it.
                seeds.append(
                    TrackerSeed(track_id, track.last_raw_mask, track.label, track.semantic_confidence)
                )
            else:
                track.status = VisibilityState.LOST

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missing_frames > ttl
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)
        return seeds

    def _bootstrap_depth_models(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> None:
        min_pixels = int(self.config.postprocess.depth_model_min_pixels)
        for seed in seeds:
            current = estimate_depth_model(frame.depth_m, seed.mask, min_pixels)
            track = self.tracks[seed.track_id]
            track.depth_model = blend_depth_model(track.depth_model, current, alpha=0.4)

    def _update_raw_track_observations(self, prediction: TrackerPrediction) -> None:
        logits = prediction.mask_logits
        for channel, track_id in enumerate(prediction.track_ids):
            if track_id not in self.tracks or channel >= logits.shape[0]:
                continue
            track = self.tracks[track_id]
            track.last_raw_mask = track.last_mask.copy()
            track.last_mask = logits[channel] > float(self.config.postprocess.mask_threshold)
            if channel < prediction.presence_scores.size:
                track.tracking_confidence = float(prediction.presence_scores[channel])

    def _postprocess(
        self,
        frame: RGBDFrame,
        prediction: TrackerPrediction,
    ) -> tuple[list[ProcessedInstance], np.ndarray]:
        h, w = frame.depth_m.shape
        logits = ensure_nhw(prediction.mask_logits, h, w)
        track_ids = [int(value) for value in prediction.track_ids]
        depth_models = [self.tracks[track_id].depth_model for track_id in track_ids]

        with self.profiler.stage("depth_validity"):
            valid = valid_depth_mask(
                frame.depth_m,
                float(self.config.postprocess.min_valid_depth_m),
                float(self.config.postprocess.max_valid_depth_m),
            )
            valid &= depth_edge_validity(
                frame.depth_m,
                float(self.config.postprocess.depth_edge_threshold_m),
            )

        with self.profiler.stage("mask_competition_depth_assignment"):
            assignment = assign_depth_ownership(
                depth_m=frame.depth_m,
                logits=logits,
                depth_models=depth_models,
                threshold=float(self.config.postprocess.mask_threshold),
                valid_depth=valid,
                overlap_depth_only=bool(self.config.postprocess.overlap_depth_only),
                mad_scale=float(self.config.postprocess.depth_gate_mad_scale),
                min_gate_m=float(self.config.postprocess.depth_gate_min_m),
                max_gate_m=float(self.config.postprocess.depth_gate_max_m),
                logit_weight=float(self.config.postprocess.logit_weight),
            )

        final_masks: list[np.ndarray] = []
        rejected_masks: list[np.ndarray] = []
        raw_masks: list[np.ndarray] = []
        with self.profiler.stage("morphology"):
            for channel in range(len(track_ids)):
                raw_mask = logits[channel] > float(self.config.postprocess.mask_threshold)
                exclusive = assignment.exclusive_masks[channel]
                final_mask = assignment.filtered_masks[channel]
                final_mask = erode_mask(final_mask, int(self.config.postprocess.erosion_pixels))
                final_mask = keep_large_components(
                    final_mask,
                    int(self.config.postprocess.min_component_pixels),
                )
                raw_masks.append(raw_mask)
                final_masks.append(final_mask)
                rejected_masks.append(exclusive & ~final_mask)

        point_payloads: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
        with self.profiler.stage("backprojection_tf"):
            for final_mask in final_masks:
                points_camera, colors = backproject_mask(
                    frame.depth_m,
                    final_mask,
                    frame.intrinsics,
                    frame.rgb,
                    stride=int(self.config.pointcloud.stride),
                    max_points=int(self.config.pointcloud.max_points_per_instance),
                )
                points_world = transform_points(points_camera, frame.world_from_camera)
                centroid_camera = robust_centroid(points_camera)
                centroid_world = robust_centroid(points_world)
                bounds_min, bounds_max = bbox_3d(points_world)
                point_payloads.append(
                    (
                        points_camera,
                        points_world,
                        colors,
                        centroid_camera,
                        centroid_world,
                        bounds_min,
                        bounds_max,
                    )
                )

        processed: list[ProcessedInstance] = []
        final_owner_channel = np.full((h, w), -1, dtype=np.int32)
        with self.profiler.stage("track_state_update"):
            for channel, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                raw_mask = raw_masks[channel]
                exclusive = assignment.exclusive_masks[channel]
                final_mask = final_masks[channel]
                rejected = rejected_masks[channel]
                final_owner_channel[final_mask] = channel
                raw_pixels = max(int(raw_mask.sum()), 1)
                visible_ratio = float(final_mask.sum() / raw_pixels)
                depth_consistency = float(assignment.depth_consistency[channel])
                status = self._visibility_state(visible_ratio, track.missing_frames)
                (
                    points_camera,
                    points_world,
                    colors,
                    centroid_camera,
                    centroid_world,
                    bounds_min,
                    bounds_max,
                ) = point_payloads[channel]

                current_depth_model = estimate_depth_model(
                    frame.depth_m,
                    final_mask,
                    int(self.config.postprocess.depth_model_min_pixels),
                )
                track.depth_model = blend_depth_model(track.depth_model, current_depth_model)
                track.last_raw_mask = raw_mask
                track.last_mask = final_mask
                track.centroid_camera = centroid_camera
                track.centroid_world = centroid_world
                track.bbox_3d_min = bounds_min
                track.bbox_3d_max = bounds_max
                track.visible_ratio = visible_ratio
                if channel < prediction.presence_scores.size:
                    track.tracking_confidence = float(prediction.presence_scores[channel])
                track.motion_prediction_confidence = float(
                    np.clip(
                        track.tracking_confidence
                        * math.sqrt(max(visible_ratio, 0.0))
                        * max(depth_consistency, 0.0),
                        0.0,
                        1.0,
                    )
                )
                track.status = status
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
                        motion_prediction_confidence=track.motion_prediction_confidence,
                        raw_mask=raw_mask,
                        exclusive_mask=exclusive,
                        depth_filtered_mask=final_mask,
                        depth_rejected_mask=rejected,
                        points_camera=points_camera,
                        points_world=points_world,
                        colors_rgb=colors,
                        centroid_camera=centroid_camera,
                        centroid_world=centroid_world,
                        bbox_3d_min=bounds_min,
                        bbox_3d_max=bounds_max,
                        visible_ratio=visible_ratio,
                        depth_consistency=depth_consistency,
                        status=status,
                    )
                )

        owner_track_map = channel_owner_to_track_owner(final_owner_channel, track_ids)
        return processed, owner_track_map

    def _visibility_state(self, visible_ratio: float, missing_frames: int) -> VisibilityState:
        if missing_frames > int(self.config.association.lost_ttl_frames):
            return VisibilityState.LOST
        if visible_ratio >= float(self.config.postprocess.visible_ratio_visible):
            return VisibilityState.VISIBLE
        if visible_ratio >= float(self.config.postprocess.visible_ratio_partial):
            return VisibilityState.PARTIAL
        return VisibilityState.OCCLUDED

    def _allocate_track_id(self) -> int:
        value = self.next_track_id
        self.next_track_id += 1
        return value

    def close(self) -> None:
        self.tracker.close()
