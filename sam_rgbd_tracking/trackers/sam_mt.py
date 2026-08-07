from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .sam2_adapter import Sam2StyleStreamingTracker
from .streaming_state import StreamingVideoState

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SamMTTracker(Sam2StyleStreamingTracker):
    """SAM-MT adapter using the official multi-target point interface."""

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
        radius = max(3, int(np.sqrt(binary.sum()) / (self.num_points_per_object + 1)))
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
        indices = np.linspace(0, ys.size - 1, min(self.num_points_per_object, ys.size), dtype=np.int64)
        return np.column_stack((xs[indices], ys[indices])).astype(np.float32)

    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        if self.stream is not None:
            self.stream.close()
        self.stream = StreamingVideoState(
            self.predictor,
            frame.rgb,
            self.offload_video_to_cpu,
            self.offload_state_to_cpu,
        )
        self.track_ids = [seed.track_id for seed in seeds]
        self.points_per_object = []
        if not seeds:
            return TrackerPrediction([], np.empty((0, *frame.depth_m.shape), np.float32), np.empty(0, np.float32), {"backend": self.backend_name})

        point_parts = [self._interior_points(seed.mask) for seed in seeds]
        if any(part.shape[0] == 0 for part in point_parts):
            raise RuntimeError("SAM-MT cannot initialize a seed with an empty mask")
        self.points_per_object = [int(part.shape[0]) for part in point_parts]
        points_np = np.concatenate(point_parts, axis=0).astype(np.float32)
        labels_np = np.ones(points_np.shape[0], dtype=np.int32)
        points = torch.from_numpy(points_np).unsqueeze(0).to(self.device)
        labels = torch.from_numpy(labels_np).unsqueeze(0).to(self.device)
        lock, autocast = self._gpu_context()
        with lock, torch.inference_mode(), autocast:
            output = self.predictor.add_new_points_or_box(
                self.stream.state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
                points_per_object=self.points_per_object,
            )
        logits = output[-1] if isinstance(output, tuple) else output
        masks = self._to_numpy_logits(logits)
        if masks.shape[0] != len(seeds):
            masks = np.stack([seed.mask.astype(np.float32) for seed in seeds])
        return TrackerPrediction(
            list(self.track_ids),
            masks,
            self._presence_from_logits(masks),
            {"backend": self.backend_name, "points_per_object": list(self.points_per_object)},
        )

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        if self.stream is None:
            raise RuntimeError("SAM-MT has not been initialized")
        frame_idx = self.stream.append(frame.rgb)
        lock, autocast = self._gpu_context()
        with lock, torch.inference_mode(), autocast:
            outputs = list(
                self.predictor.propagate_in_video(
                    self.stream.state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1,
                    reverse=False,
                    points_per_object=self.points_per_object,
                )
            )
        if not outputs:
            raise RuntimeError(f"SAM-MT returned no output for frame {frame_idx}")
        output = outputs[-1]
        logits = output[-1]
        masks = self._to_numpy_logits(logits)
        return TrackerPrediction(
            list(self.track_ids),
            masks,
            self._presence_from_logits(masks),
            {"backend": self.backend_name},
        )
