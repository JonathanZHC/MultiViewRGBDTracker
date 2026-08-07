from __future__ import annotations

from typing import Any

import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .sam2_adapter import Sam2StyleStreamingTracker
from .streaming_state import StreamingVideoState

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SamMTTracker(Sam2StyleStreamingTracker):
    """SAM-MT adapter using its official multi-target point interface.

    The current public SAM-MT inference path represents all targets as channels
    inside one model object and expects `points_per_object`. It does not expose
    a public multi-mask conditioning call. Therefore, high-quality interior
    points are sampled from each SAM3 mask at every keyframe. The exact SAM3
    masks still initialize depth models, labels and the first post-processing
    result, while SAM-MT receives the corresponding target prompts.
    """

    def __init__(self, *args, points_per_object: int = 2, **kwargs) -> None:
        self.num_points_per_object = max(1, int(points_per_object))
        self.points_per_object: list[int] = []
        super().__init__(*args, **kwargs)

    @property
    def backend_name(self) -> str:
        return "sam_mt"

    def _build_predictor(self) -> Any:
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            mode="eval",
            apply_postprocessing=False,
            hydra_overrides_extra=[f"++model.non_overlap_masks={str(self.non_overlap_masks).lower()}"],
            vos_optimized=False,
        )
        return predictor

    def initialize(self, frame: RGBDFrame, seeds: list[TrackerSeed]) -> TrackerPrediction:
        if torch is None:
            raise RuntimeError("PyTorch is required for SAM-MT")
        with self._lock(), torch.inference_mode(), self._autocast():
            if self.stream is not None:
                self.stream.close()
            self.stream = StreamingVideoState(
                self.predictor,
                frame.rgb,
                self.offload_video_to_cpu,
                self.offload_state_to_cpu,
            )
            self.track_ids = [int(seed.track_id) for seed in seeds]
            if not seeds:
                h, w = frame.rgb.shape[:2]
                self.points_per_object = []
                return TrackerPrediction([], np.empty((0, h, w), np.float32), np.empty((0,), np.float32))

            point_parts: list[np.ndarray] = []
            self.points_per_object = []
            for seed in seeds:
                points = self._sample_mask_points(seed.mask, self.num_points_per_object)
                if points.shape[0] == 0:
                    continue
                point_parts.append(points)
                self.points_per_object.append(int(points.shape[0]))
            if len(point_parts) != len(seeds):
                raise RuntimeError("SAM-MT cannot initialize a seed with an empty mask")
            points_np = np.concatenate(point_parts, axis=0).astype(np.float32)
            points = torch.from_numpy(points_np).unsqueeze(0).to(self.device)
            labels = torch.ones(points.shape[:2], dtype=torch.int32, device=self.device)
            output = self.predictor.add_new_points_or_box(
                self.stream.state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
                points_per_object=self.points_per_object,
            )
            logits = self._sam_mt_logits_to_numpy(output[-1], frame.rgb.shape[:2])
            return TrackerPrediction(
                track_ids=self.track_ids.copy(),
                mask_logits=logits,
                presence_scores=self._presence_from_logits(logits),
                backend_metadata={
                    "backend": self.backend_name,
                    "frame_index": 0,
                    "points_per_object": self.points_per_object.copy(),
                    "mask_bridge": "interior_points",
                },
            )

    def track(self, frame: RGBDFrame) -> TrackerPrediction:
        if self.stream is None:
            raise RuntimeError("Tracker must be initialized before track()")
        if not self.track_ids:
            h, w = frame.rgb.shape[:2]
            return TrackerPrediction([], np.empty((0, h, w), np.float32), np.empty((0,), np.float32))
        with self._lock(), torch.inference_mode(), self._autocast():
            frame_idx = self.stream.append(frame.rgb)
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
                raise RuntimeError("SAM-MT yielded no output for the appended frame")
            logits = self._sam_mt_logits_to_numpy(output[-1], frame.rgb.shape[:2])
            return TrackerPrediction(
                track_ids=self.track_ids.copy(),
                mask_logits=logits,
                presence_scores=self._presence_from_logits(logits),
                backend_metadata={
                    "backend": self.backend_name,
                    "frame_index": int(output[0]),
                    "stream_length": int(self.stream.state["num_frames"]),
                    "points_per_object": self.points_per_object.copy(),
                },
            )

    @staticmethod
    def _sample_mask_points(mask: np.ndarray, count: int) -> np.ndarray:
        binary = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.nonzero(binary)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if cv2 is not None:
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            selected: list[tuple[float, float]] = []
            working = distance.copy()
            radius = max(3, int(np.sqrt(binary.sum()) / (count + 1)))
            for _ in range(count):
                flat = int(np.argmax(working))
                y, x = np.unravel_index(flat, working.shape)
                if working[y, x] <= 0:
                    break
                selected.append((float(x), float(y)))
                y0, y1 = max(0, y - radius), min(working.shape[0], y + radius + 1)
                x0, x1 = max(0, x - radius), min(working.shape[1], x + radius + 1)
                working[y0:y1, x0:x1] = 0
            if selected:
                return np.asarray(selected, dtype=np.float32)
        indices = np.linspace(0, ys.size - 1, min(count, ys.size), dtype=np.int64)
        return np.column_stack((xs[indices], ys[indices])).astype(np.float32)

    @staticmethod
    def _sam_mt_logits_to_numpy(value: Any, original_hw: tuple[int, int]) -> np.ndarray:
        if hasattr(value, "detach"):
            array = value.detach().float().cpu().numpy()
        else:
            array = np.asarray(value, dtype=np.float32)
        # Public SAM-MT returns [1, N, H, W].
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError(f"Unexpected SAM-MT mask shape: {array.shape}")
        return array.astype(np.float32, copy=False)
