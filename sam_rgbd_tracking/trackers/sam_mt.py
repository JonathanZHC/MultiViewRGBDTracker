from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .base import tracker_profile_context
from .sam2_adapter import Sam2StyleStreamingTracker

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SamMTTracker(Sam2StyleStreamingTracker):
    """SAM-MT adapter using the same optimized streaming state infrastructure."""

    def __init__(self, *args, points_per_object: int = 2, **kwargs) -> None:
        self.num_points_per_object = max(1, int(points_per_object))
        self.points_per_object: list[int] = []
        super().__init__(*args, **kwargs)

    @property
    def backend_name(self) -> str:
        return "sam_mt"

    def _build_predictor(self) -> Any:
        from sam2.build_sam import build_sam2_video_predictor

        return build_sam2_video_predictor(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            mode="eval",
            apply_postprocessing=False,
            hydra_overrides_extra=[
                f"++model.non_overlap_masks={str(self.non_overlap_masks).lower()}"
            ],
            vos_optimized=False,
        )

    def _interior_points(self, mask: np.ndarray) -> np.ndarray:
        binary = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.nonzero(binary)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        selected: list[tuple[float, float]] = []
        work = distance.copy()
        radius = max(
            3,
            int(np.sqrt(binary.sum()) / (self.num_points_per_object + 1)),
        )
        for _ in range(self.num_points_per_object):
            flat = int(np.argmax(work))
            y, x = np.unravel_index(flat, work.shape)
            if work[y, x] <= 0:
                break
            selected.append((float(x), float(y)))
            y0, y1 = max(0, y - radius), min(work.shape[0], y + radius + 1)
            x0, x1 = max(0, x - radius), min(work.shape[1], x + radius + 1)
            work[y0:y1, x0:x1] = 0
        if selected:
            return np.asarray(selected, dtype=np.float32)
        indices = np.linspace(
            0,
            ys.size - 1,
            min(self.num_points_per_object, ys.size),
            dtype=np.int64,
        )
        return np.column_stack((xs[indices], ys[indices])).astype(np.float32)

    def initialize(
        self,
        frame: RGBDFrame,
        seeds: list[TrackerSeed],
    ) -> TrackerPrediction:
        call_started = time.perf_counter()

        with self._gpu_guard():
            reinit_started = time.perf_counter()
            with torch.inference_mode(), self._autocast(), tracker_profile_context(self.profiler):
                with self.profile_stage("tracker_reinit_gpu", cuda=True):
                    self._reset_or_create_stream(frame)
                    assert self.stream is not None
                    self.track_ids = [int(seed.track_id) for seed in seeds]
                    self.points_per_object = []

                    if not seeds:
                        prediction = TrackerPrediction(
                            [],
                            np.empty((0, *frame.depth_m.shape), np.float32),
                            np.empty((0,), np.float32),
                            {"backend": self.backend_name},
                        )
                    else:
                        point_parts = [self._interior_points(seed.mask) for seed in seeds]
                        if any(part.shape[0] == 0 for part in point_parts):
                            raise RuntimeError(
                                "SAM-MT cannot initialize a seed with an empty mask"
                            )
                        self.points_per_object = [
                            int(part.shape[0]) for part in point_parts
                        ]
                        points_np = np.concatenate(point_parts, axis=0).astype(np.float32)
                        labels_np = np.ones(points_np.shape[0], dtype=np.int32)
                        points = torch.from_numpy(points_np).unsqueeze(0).to(self.device)
                        labels = torch.from_numpy(labels_np).unsqueeze(0).to(self.device)

                        with self.profile_stage("tracker_seed_gpu", cuda=True):
                            output = self.predictor.add_new_points_or_box(
                                self.stream.state,
                                frame_idx=0,
                                obj_id=1,
                                points=points,
                                labels=labels,
                                points_per_object=self.points_per_object,
                            )
                        logits = output[-1] if isinstance(output, tuple) else output
                        with self.profile_stage("tracker_output_d2h_cpu", cuda=False):
                            masks = self._to_numpy_logits(logits)
                        if masks.shape[0] != len(seeds):
                            masks = np.stack(
                                [seed.mask.astype(np.float32) for seed in seeds]
                            )
                        prediction = TrackerPrediction(
                            list(self.track_ids),
                            masks,
                            self._presence_from_logits(masks),
                            {
                                "backend": self.backend_name,
                                "points_per_object": list(self.points_per_object),
                            },
                        )

            self.record_profile(
                "tracker_reinit_wall_cpu",
                1000.0 * (time.perf_counter() - reinit_started),
            )

        self.record_profile(
            "tracker_total_wall_cpu",
            1000.0 * (time.perf_counter() - call_started),
        )
        return prediction

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        if self.stream is None:
            raise RuntimeError("SAM-MT has not been initialized")

        call_started = time.perf_counter()
        with self._gpu_guard():
            with torch.inference_mode(), self._autocast(), tracker_profile_context(self.profiler):
                with self.profile_stage("tracker_total_gpu", cuda=True):
                    with self.profile_stage("tracker_append_cpu", cuda=False):
                        frame_idx = self.stream.append(frame.rgb)
                    with self.profile_stage("tracker_propagate_gpu", cuda=True):
                        output = None
                        for output in self.predictor.propagate_in_video(
                            self.stream.state,
                            start_frame_idx=frame_idx,
                            max_frame_num_to_track=1,
                            reverse=False,
                            points_per_object=self.points_per_object,
                        ):
                            pass

                if output is None:
                    raise RuntimeError(
                        f"SAM-MT returned no output for frame {frame_idx}"
                    )
                logits = output[-1]
                with self.profile_stage("tracker_output_d2h_cpu", cuda=False):
                    masks = self._to_numpy_logits(logits)
                prediction = TrackerPrediction(
                    list(self.track_ids),
                    masks,
                    self._presence_from_logits(masks),
                    {"backend": self.backend_name},
                )

        self.record_profile(
            "tracker_total_wall_cpu",
            1000.0 * (time.perf_counter() - call_started),
        )
        return prediction
