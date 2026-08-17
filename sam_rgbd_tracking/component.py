from __future__ import annotations

from copy import copy
from pathlib import Path

import cv2
import numpy as np

from .config import Config, load_config
from .data_types import (
    CameraIntrinsics,
    DetectionInstance,
    FrameResult,
    ProcessedInstance,
    RGBDFrame,
    TrackState,
    TrackerPrediction,
    TrackerSeed,
    VisibilityState,
)
from .processing import (
    backproject_mask,
    bbox_3d,
    erode_filter_and_bbox,
    nonoverlap_owner_map,
    robust_centroid,
    transform_points,
    valid_depth_mask,
    mask_iou,
)
from .profiler import FrameProfiler
from .slots import build_slot_layout


class SAMTrackingComponent:
    """Per-view association, RGB-D geometry, and profiling helper.

    SAM3 and EfficientTAM are owned by the multi-view orchestration layer. This
    class intentionally contains no model backend and no GPU model state.
    """

    def __init__(
        self,
        config: str | Path | Config = "configs/tracking.yaml",
        *,
        camera_name: str = "camera_0",
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_config(config)
        self.config = config
        self.camera_name = camera_name
        self.profiler = FrameProfiler(config, name=f"{camera_name}/efficient_tam")
        self.slot_layout = build_slot_layout(config)
        self.tracks: dict[int, TrackState] = {}
        self.next_track_id = len(self.slot_layout) + 1
        self.release_after_missing_frames = max(
            1, int(config.tracker.get("release_after_missing_frames", 30))
        )
        self.local_slot_iou_threshold = float(
            config.tracker.get("local_slot_iou_threshold", 0.05)
        )
        self.frame_index = 0
        self.last_prediction: TrackerPrediction | None = None

    def make_frame(
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
    ) -> RGBDFrame:
        h, w = depth_m.shape
        frame = RGBDFrame(
            camera_name=self.camera_name,
            frame_index=self.frame_index,
            timestamp_ns=int(timestamp_ns),
            rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
            depth_m=np.ascontiguousarray(depth_m, dtype=np.float32),
            intrinsics=CameraIntrinsics(float(fx), float(fy), float(cx), float(cy), w, h),
            world_from_camera=(
                None
                if world_from_camera is None
                else np.asarray(world_from_camera, dtype=np.float32)
            ),
        )
        self.frame_index += 1
        return frame

    def begin_external_frame(self) -> None:
        self.profiler.begin_frame()

    def initialize_tracks(
        self,
        frame: RGBDFrame,
        detections: list[DetectionInstance],
    ) -> list[TrackerSeed]:
        """Create every configured fixed slot; inactive slots use zero masks."""
        if self.tracks:
            raise RuntimeError("initialize_tracks may only be called once")

        height, width = frame.depth_m.shape
        detections_by_label: dict[str, list[DetectionInstance]] = {}
        for detection in detections:
            detections_by_label.setdefault(detection.label, []).append(detection)
        for values in detections_by_label.values():
            values.sort(key=lambda item: float(item.score), reverse=True)

        seeds: list[TrackerSeed] = []
        with self.profiler.stage("sam3_local_slot_assoc_cpu", cuda=False):
            for spec in self.slot_layout:
                class_detections = detections_by_label.get(spec.semantic_label, [])
                detection = (
                    class_detections[spec.class_slot]
                    if spec.class_slot < len(class_detections)
                    else None
                )
                active = detection is not None
                mask = (
                    np.asarray(detection.mask, dtype=bool).copy()
                    if detection is not None
                    else np.zeros((height, width), dtype=bool)
                )
                centroid = (
                    robust_centroid(frame.depth_m, mask, frame.intrinsics)
                    if active
                    else None
                )
                confidence = float(detection.score) if detection is not None else 0.0
                self.tracks[spec.track_id] = TrackState(
                    track_id=spec.track_id,
                    label=spec.semantic_label,
                    semantic_confidence=confidence,
                    tracker_slot=spec.slot_index,
                    class_slot=spec.class_slot,
                    active=active,
                    embedding=None,
                    last_mask=mask.copy(),
                    last_raw_mask=mask.copy(),
                    centroid_camera=centroid,
                    tracking_confidence=confidence,
                    missing_frames=0,
                    last_seen_frame=frame.frame_index if active else -1,
                )
                seeds.append(
                    TrackerSeed(
                        spec.track_id,
                        mask,
                        spec.semantic_label,
                        confidence,
                    )
                )
        return seeds

    def build_direct_correction_masks(
        self,
        reference_frame: RGBDFrame,
        detections: list[DetectionInstance],
        expected_track_ids: list[int],
        fallback_masks: dict[int, np.ndarray],
    ) -> tuple[list[np.ndarray], int]:
        """Same-class greedy mask-IoU association onto the fixed local slots."""
        expected = [int(track_id) for track_id in expected_track_ids]
        height, width = reference_frame.depth_m.shape
        masks_by_id: dict[int, np.ndarray] = {}
        for track_id in expected:
            fallback = fallback_masks.get(track_id)
            if fallback is None:
                fallback = np.zeros((height, width), dtype=bool)
            masks_by_id[track_id] = np.asarray(fallback, dtype=bool).copy()

        detections_by_label: dict[str, list[DetectionInstance]] = {}
        for detection in detections:
            detections_by_label.setdefault(detection.label, []).append(detection)
        for values in detections_by_label.values():
            values.sort(key=lambda item: float(item.score), reverse=True)

        activated = 0
        for label in {spec.semantic_label for spec in self.slot_layout}:
            class_detections = detections_by_label.get(label, [])
            class_tracks = [
                self.tracks[track_id]
                for track_id in expected
                if track_id in self.tracks and self.tracks[track_id].label == label
            ]
            active_tracks = [track for track in class_tracks if track.active]

            candidate_edges: list[tuple[float, int, int]] = []
            for det_index, detection in enumerate(class_detections):
                for track in active_tracks:
                    iou = mask_iou(masks_by_id[track.track_id], detection.mask)
                    if iou >= self.local_slot_iou_threshold:
                        candidate_edges.append((iou, det_index, track.track_id))

            matched_detections: set[int] = set()
            matched_tracks: set[int] = set()
            for _, det_index, track_id in sorted(candidate_edges, reverse=True):
                if det_index in matched_detections or track_id in matched_tracks:
                    continue
                detection = class_detections[det_index]
                masks_by_id[track_id] = np.asarray(detection.mask, dtype=bool).copy()
                track = self.tracks[track_id]
                track.semantic_confidence = float(detection.score)
                track.active = True
                matched_detections.add(det_index)
                matched_tracks.add(track_id)

            free_tracks = sorted(
                (track for track in class_tracks if not track.active),
                key=lambda track: track.class_slot,
            )
            for det_index, detection in enumerate(class_detections):
                if det_index in matched_detections or not free_tracks:
                    continue
                track = free_tracks.pop(0)
                masks_by_id[track.track_id] = np.asarray(detection.mask, dtype=bool).copy()
                track.active = True
                track.semantic_confidence = float(detection.score)
                track.tracking_confidence = float(detection.score)
                track.missing_frames = 0
                track.last_seen_frame = reference_frame.frame_index
                activated += 1

        return [masks_by_id[track_id] for track_id in expected], activated

    def finalize_external_prediction(
        self,
        frame: RGBDFrame,
        prediction: TrackerPrediction,
        *,
        keyframe: bool,
        trigger_reasons: list[str],
        update_raw_observations: bool = True,
        extra_metadata: dict | None = None,
    ) -> FrameResult:
        if update_raw_observations:
            self._update_raw_observations(prediction)
        self.last_prediction = prediction
        with self.profiler.stage("postprocess_cpu", cuda=False):
            (
                instances,
                owner,
                raw_instance_map,
                filtered_instance_map,
            ) = self._postprocess(frame, prediction)
        timings = self.profiler.end_frame()
        metadata = {
            "trigger_reasons": list(trigger_reasons),
            "num_active_instances_per_view": sum(
                1 for track in self.tracks.values() if track.active
            ),
            "num_dummy_slots_per_view": sum(
                1 for track in self.tracks.values() if not track.active
            ),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return FrameResult(
            frame=frame,
            instances=instances,
            owner_track_map=owner,
            keyframe=bool(keyframe),
            timings_ms=timings,
            raw_instance_map=raw_instance_map,
            filtered_instance_map=filtered_instance_map,
            metadata=metadata,
        )

    def raw_masks_by_track(self, result: FrameResult) -> dict[int, np.ndarray]:
        return {
            int(instance.track_id): np.asarray(instance.raw_mask, dtype=bool).copy()
            for instance in result.instances
        }

    def _ensure_logits(
        self,
        prediction: TrackerPrediction,
        height: int,
        width: int,
    ) -> np.ndarray:
        logits = np.asarray(prediction.mask_logits, dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[None]
        if logits.size == 0:
            return np.empty((0, height, width), np.float32)
        if logits.shape[-2:] != (height, width):
            logits = np.stack(
                [
                    cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for mask in logits
                ],
                axis=0,
            )
        return logits

    def _update_raw_observations(
        self,
        prediction: TrackerPrediction,
    ) -> None:
        logits = np.asarray(prediction.mask_logits)
        if logits.ndim == 2:
            logits = logits[None]
        for channel, track_id in enumerate(prediction.track_ids):
            track = self.tracks.get(int(track_id))
            if track is None or not track.active or channel >= logits.shape[0]:
                continue
            track.last_raw_mask = (
                None if track.last_mask is None else track.last_mask.copy()
            )
            track.last_mask = (
                logits[channel] > float(self.config.postprocess.mask_threshold)
            )
            if channel < prediction.presence_scores.size:
                track.tracking_confidence = float(
                    prediction.presence_scores[channel]
                )

    def _postprocess(
        self,
        frame: RGBDFrame,
        prediction: TrackerPrediction,
    ) -> tuple[
        list[ProcessedInstance],
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """Convert tracker masks to geometry without depth-based mask exclusion.

        Visualization-only raster products are prepared here once, while masks
        are already hot in cache.  This avoids rescanning every instance mask in
        the RViz stage.  They are omitted entirely when visualization is disabled.
        """
        h, w = frame.depth_m.shape
        logits = self._ensure_logits(prediction, h, w)

        valid_channels = [
            index
            for index, value in enumerate(prediction.track_ids)
            if (
                int(value) in self.tracks
                and self.tracks[int(value)].active
                and index < logits.shape[0]
            )
        ]
        track_ids = [
            int(prediction.track_ids[index])
            for index in valid_channels
        ]
        logits = (
            logits[valid_channels]
            if valid_channels
            else np.empty((0, h, w), np.float32)
        )

        visualization_enabled = bool(
            self.config.runtime.get("enable_visualization", True)
        )
        debug_images_enabled = visualization_enabled and bool(
            self.config.runtime.get("publish_debug_images", True)
        )

        if not track_ids:
            empty_owner = np.zeros((h, w), dtype=np.int32)
            filtered_map = (
                np.zeros((h, w), dtype=np.uint8)
                if debug_images_enabled
                else None
            )
            raw_map = (
                np.zeros((h, w), dtype=np.uint8)
                if debug_images_enabled
                else None
            )
            return [], empty_owner, raw_map, filtered_map

        threshold = float(self.config.postprocess.mask_threshold)

        # Threshold all fixed object channels in one NumPy operation rather than
        # allocating one temporary threshold result per Python loop iteration.
        raw_mask_stack = logits > threshold
        final_masks: list[np.ndarray] = []
        bboxes_2d: list[tuple[int, int, int, int] | None] = []
        for channel in range(len(track_ids)):
            filtered, bbox_2d = erode_filter_and_bbox(
                raw_mask_stack[channel],
                int(self.config.postprocess.erosion_pixels),
                int(self.config.postprocess.min_component_pixels),
            )
            final_masks.append(filtered)
            bboxes_2d.append(bbox_2d)

        # Overlap pixels stay 0 for external ownership/occlusion handling.
        owner_track_map = nonoverlap_owner_map(
            final_masks,
            track_ids,
            h,
            w,
        )

        # Build compact visualization maps once in postprocess.  Values are
        # uint8 per-frame instance codes (1..N), not global track IDs.  The true
        # ownership map above remains int32/track-ID based.  Using one byte/pixel
        # cuts debug-raster memory traffic by 4x and feeds OpenCV applyColorMap
        # directly without conversion.
        filtered_instance_map = (
            np.zeros((h, w), dtype=np.uint8)
            if debug_images_enabled
            else None
        )
        raw_instance_map = (
            np.zeros((h, w), dtype=np.uint8)
            if debug_images_enabled
            else None
        )

        # Depth validity affects only 3-D point generation.
        valid_geometry_depth = valid_depth_mask(
            frame.depth_m,
            float(self.config.postprocess.min_valid_depth_m),
            float(self.config.postprocess.max_valid_depth_m),
        )

        processed: list[ProcessedInstance] = []
        for channel, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            raw_mask = raw_mask_stack[channel]
            final_mask = final_masks[channel]
            bbox_2d = bboxes_2d[channel]

            visual_code = np.uint8(min(channel + 1, 255))
            if filtered_instance_map is not None:
                filtered_instance_map[final_mask] = visual_code
            if raw_instance_map is not None:
                raw_instance_map[raw_mask] = visual_code

            geometry_mask = final_mask & valid_geometry_depth
            points_camera, colors = backproject_mask(
                frame.depth_m,
                frame.rgb,
                geometry_mask,
                frame.intrinsics,
                int(self.config.pointcloud.stride),
                int(self.config.pointcloud.max_points_per_instance),
            )
            points_world = transform_points(
                points_camera,
                frame.world_from_camera,
            )

            centroid_camera = (
                None
                if points_camera.size == 0
                else np.median(points_camera, axis=0).astype(np.float32)
            )
            centroid_world = (
                None
                if points_world is None or points_world.size == 0
                else np.median(points_world, axis=0).astype(np.float32)
            )
            bounds_min, bounds_max = bbox_3d(
                points_world if points_world is not None else points_camera
            )

            track.last_raw_mask = raw_mask
            track.last_mask = final_mask
            track.centroid_camera = centroid_camera
            track.centroid_world = centroid_world
            if channel < prediction.presence_scores.size:
                track.tracking_confidence = float(
                    prediction.presence_scores[channel]
                )

            # bbox_2d is already derived from the final filtered mask, so it also
            # tells us whether the mask is empty without another full-image scan.
            if bbox_2d is not None:
                track.last_seen_frame = frame.frame_index
                track.missing_frames = 0
                status = VisibilityState.VISIBLE
            else:
                track.missing_frames += 1
                status = VisibilityState.LOST
                if track.missing_frames >= self.release_after_missing_frames:
                    track.active = False

            motion_conf = min(max(float(track.tracking_confidence), 0.0), 1.0)

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
                    bbox_2d_xyxy=bbox_2d,
                    status=status,
                    tracker_slot=track.tracker_slot,
                    class_slot=track.class_slot,
                )
            )

        return (
            processed,
            owner_track_map,
            raw_instance_map,
            filtered_instance_map,
        )

    def _new_track_id(self) -> int:
        value = self.next_track_id
        self.next_track_id += 1
        return value

    def print_stats(self) -> None:
        self.profiler.print_summary()

    def close(self) -> None:
        self.profiler.close()
