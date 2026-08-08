from __future__ import annotations

import threading
import time
from typing import Any, ClassVar

import cv2
import numpy as np

from ..data_types import RGBDFrame, TrackerPrediction, TrackerSeed
from .base import tracker_profile_context
from .sam2_adapter import Sam2StyleStreamingTracker
from .streaming_state import StreamingVideoState

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class SamMTTracker(Sam2StyleStreamingTracker):
    """SAM-MT adapter using the same optimized streaming state infrastructure.

    SAM-MT keeps its native multi-target propagation path (``vos_optimized=False``)
    and optionally compiles *only* the SAM2 image encoder. This is intentionally
    narrower than full VOS compilation: the SAM-MT-specific multi-target logic,
    memory attention, decoder, and memory encoder stay eager.

    The image encoder compile is enabled by default because it is the lowest-risk
    upstream-supported compile option for this backend. The first compile is
    pre-warmed on the same persistent GPU-owner OS thread used by live inference,
    so the expensive first ``torch.compile`` call does not land in a live cycle.
    """

    _image_encoder_prewarm_lock: ClassVar[threading.Lock] = threading.Lock()
    _image_encoder_prewarm_done: ClassVar[set[tuple[Any, ...]]] = set()

    def __init__(
        self,
        *args,
        points_per_object: int = 2,
        compile_image_encoder: bool = True,
        prewarm_enabled: bool = True,
        prewarm_passes: int = 2,
        **kwargs,
    ) -> None:
        self.num_points_per_object = max(1, int(points_per_object))
        self.points_per_object: list[int] = []
        self.compile_image_encoder = bool(compile_image_encoder)
        self.prewarm_enabled = bool(prewarm_enabled)
        self.prewarm_passes = max(1, int(prewarm_passes))
        super().__init__(*args, **kwargs)

    @property
    def backend_name(self) -> str:
        return "sam_mt"

    def _cache_key(self) -> tuple[str, ...]:
        # The base predictor cache is shared between camera components. Include
        # the image-encoder compile mode so an eager and compiled SAM-MT
        # predictor can never alias the same cached model instance.
        return super()._cache_key() + (str(self.compile_image_encoder),)

    def _build_predictor(self) -> Any:
        from sam2.build_sam import build_sam2_video_predictor

        overrides = [
            f"++model.non_overlap_masks={str(self.non_overlap_masks).lower()}",
            (
                "++model.compile_image_encoder="
                f"{str(self.compile_image_encoder).lower()}"
            ),
        ]

        return build_sam2_video_predictor(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            mode="eval",
            apply_postprocessing=False,
            hydra_overrides_extra=overrides,
            # Keep SAM-MT's native multi-target predictor path. Only the image
            # encoder is compiled through SAM2Base.compile_image_encoder.
            vos_optimized=False,
        )

    def _prewarm_key(self, height: int, width: int) -> tuple[Any, ...]:
        device_index = -1
        if torch is not None and str(self.device).startswith("cuda"):
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
        return (
            id(self.predictor),
            int(height),
            int(width),
            int(getattr(self.predictor, "image_size", 0)),
            str(self.device),
            int(device_index),
            bool(self.use_bf16),
        )

    def prewarm(self, first_rgb: np.ndarray) -> dict[str, Any]:
        """Compile/capture only the SAM-MT image encoder before live tracking.

        A temporary streaming state is used only to obtain the exact normalized
        image tensor shape used by the real pipeline. We then call
        ``predictor.forward_image`` directly so the warm-up does not create any
        SAM-MT prompts, objects, or temporal memory and cannot change live state.

        The ROS node already routes this method through the single persistent
        GPU-owner thread. Keeping compile and live replay on that same OS thread
        avoids PyTorch Inductor CUDAGraph-tree TLS problems.
        """
        if not self.compile_image_encoder or not self.prewarm_enabled:
            return {
                "enabled": bool(self.compile_image_encoder),
                "performed": False,
                "reason": "disabled",
            }
        if torch is None:
            return {"enabled": True, "performed": False, "reason": "no_torch"}

        rgb = np.ascontiguousarray(first_rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]
        key = self._prewarm_key(height, width)

        with self._image_encoder_prewarm_lock:
            if key in self._image_encoder_prewarm_done:
                return {
                    "enabled": True,
                    "performed": False,
                    "reason": "already_warm",
                }

            started = time.perf_counter()
            pass_ms: list[float] = []
            stream: StreamingVideoState | None = None

            try:
                with self._gpu_guard():
                    # Use the exact same preprocessing/state path as live
                    # tracking, but never seed or propagate the temporary state.
                    stream = StreamingVideoState(
                        self.predictor,
                        rgb,
                        self.offload_video_to_cpu,
                        self.offload_state_to_cpu,
                        buffer_frames=2,
                        profiler=None,
                        use_gpu_preprocess=self.gpu_preprocess,
                        pin_input_memory=self.pin_input_memory,
                    )

                    image = stream.state["images"][0:1].to(
                        torch.device(self.device),
                        dtype=torch.float32,
                        non_blocking=True,
                    )

                    with torch.inference_mode(), self._autocast():
                        for pass_index in range(self.prewarm_passes):
                            pass_started = time.perf_counter()
                            backbone_out = self.predictor.forward_image(image)
                            if str(self.device).startswith("cuda"):
                                torch.cuda.synchronize(torch.device(self.device))
                            elapsed_ms = 1000.0 * (
                                time.perf_counter() - pass_started
                            )
                            pass_ms.append(elapsed_ms)

                            # Do not retain CUDAGraph-backed outputs across the
                            # next replay. Live code will immediately consume its
                            # backbone output inside _get_image_feature.
                            del backbone_out

                            print(
                                "[SAM-MT image-encoder warmup] "
                                f"pass={pass_index + 1}/{self.prewarm_passes} "
                                f"wall={elapsed_ms:.1f} ms",
                                flush=True,
                            )

                self._image_encoder_prewarm_done.add(key)
                total_ms = 1000.0 * (time.perf_counter() - started)
                print(
                    "[SAM-MT image-encoder warmup] complete "
                    f"total={total_ms:.1f} ms",
                    flush=True,
                )
                return {
                    "enabled": True,
                    "performed": True,
                    "passes_ms": pass_ms,
                    "total_ms": total_ms,
                }
            finally:
                if stream is not None:
                    stream.close()

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
