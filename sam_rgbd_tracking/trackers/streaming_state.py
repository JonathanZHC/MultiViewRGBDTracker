from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


class StreamingVideoState:
    """Fast live-stream bridge for SAM2/EfficientTAM-style predictors.

    Design goals:
      1. Keep upstream ``init_state`` compatibility for the *first* frame.
      2. Never rebuild the state/JPEG directory on later SAM3 keyframes.
      3. Preprocess live RGB on GPU when video frames live on GPU.
      4. Reuse pinned host input and CUDA scratch buffers.
      5. Replace per-frame ``torch.cat`` with a preallocated frame buffer.

    After the first initialization, a keyframe reset uses predictor.reset_state(),
    replaces frame 0 in the existing buffer, clears the image-feature cache and
    re-seeds the same inference-state object. Therefore the repeated
    ``frame loading (JPEG)`` path disappears from periodic/anomaly keyframes.
    """

    def __init__(
        self,
        predictor: Any,
        first_rgb: np.ndarray,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        *,
        buffer_frames: int = 40,
        profiler: Any | None = None,
        use_gpu_preprocess: bool = True,
        pin_input_memory: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for tracker backends")

        self.predictor = predictor
        self.profiler = profiler
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.offload_state_to_cpu = bool(offload_state_to_cpu)
        self.image_size = int(self.predictor.image_size)
        self.height, self.width = first_rgb.shape[:2]
        self.frame_index = 0
        self._capacity = max(2, int(buffer_frames))
        self._pin_input_memory = bool(pin_input_memory)

        # Keep the one-time upstream initialization for compatibility with both
        # EfficientTAM and SAM-MT/SAM2 state dictionaries.
        self.temp_dir = tempfile.TemporaryDirectory(prefix="sam_rgbd_stream_")
        first_path = Path(self.temp_dir.name) / "000000.jpg"
        Image.fromarray(
            np.ascontiguousarray(first_rgb, dtype=np.uint8),
            mode="RGB",
        ).save(first_path, quality=95)

        self.state = predictor.init_state(
            video_path=self.temp_dir.name,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=False,
        )

        self.compute_device = torch.device(self.state["device"])
        self.storage_device = (
            torch.device("cpu")
            if self.offload_video_to_cpu
            else self.compute_device
        )
        self._gpu_preprocess = bool(
            use_gpu_preprocess
            and self.storage_device.type == "cuda"
            and torch.cuda.is_available()
        )

        loaded_images = self.state["images"]
        if not isinstance(loaded_images, torch.Tensor):
            raise TypeError(
                "Fast streaming requires tensor-backed state['images']; "
                f"got {type(loaded_images)!r}"
            )
        if len(loaded_images) < 1:
            raise RuntimeError("Upstream init_state returned no images")

        # Preallocate the history buffer once. Normal append is now O(1): only
        # the new slot is written and state['images'] is changed to a cheap view.
        self._frame_buffer = torch.empty(
            (self._capacity, 3, self.image_size, self.image_size),
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=(
                self.storage_device.type == "cpu"
                and self._pin_input_memory
                and torch.cuda.is_available()
            ),
        )
        self._frame_buffer[0].copy_(
            loaded_images[0].to(
                self.storage_device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )
        self._set_visible_length(1)

        # Reusable preprocessing tensors. No per-frame mean/std allocation.
        if self._gpu_preprocess:
            try:
                self._host_u8 = torch.empty(
                    (self.height, self.width, 3),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=self._pin_input_memory,
                )
            except RuntimeError:
                self._host_u8 = torch.empty(
                    (self.height, self.width, 3),
                    dtype=torch.uint8,
                    device="cpu",
                )
            self._host_np = self._host_u8.numpy()
            self._device_hwc_u8 = torch.empty(
                (self.height, self.width, 3),
                dtype=torch.uint8,
                device=self.compute_device,
            )
            self._mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
                device=self.compute_device,
            ).view(1, 3, 1, 1)
            self._std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
                device=self.compute_device,
            ).view(1, 3, 1, 1)
        else:
            self._host_u8 = None
            self._host_np = None
            self._device_hwc_u8 = None
            self._mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1)
            self._std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1)

    def _stage(self, name: str, *, cuda: bool = False):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.stage(name, cuda=cuda)

    def _validate_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(
                "Frame resolution changed inside a tracking session: "
                f"expected {(self.height, self.width)}, got {rgb.shape[:2]}"
            )
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB HxWx3 input, got {rgb.shape}")
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _set_visible_length(self, num_frames: int) -> None:
        self.state["images"] = self._frame_buffer[:num_frames]
        self.state["num_frames"] = int(num_frames)

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = max(required, self._capacity * 2)
        stage_name = (
            "tracker_buffer_grow_gpu"
            if self.storage_device.type == "cuda"
            else "tracker_buffer_grow_cpu"
        )
        with self._stage(stage_name, cuda=self.storage_device.type == "cuda"):
            new_buffer = torch.empty(
                (new_capacity, 3, self.image_size, self.image_size),
                dtype=torch.float32,
                device=self.storage_device,
                pin_memory=(
                    self.storage_device.type == "cpu"
                    and self._pin_input_memory
                    and torch.cuda.is_available()
                ),
            )
            visible = int(self.state["num_frames"])
            if visible > 0:
                new_buffer[:visible].copy_(self._frame_buffer[:visible])
            self._frame_buffer = new_buffer
            self._capacity = new_capacity
            self._set_visible_length(visible)

    def _preprocess_into_slot(self, rgb: np.ndarray, slot: int) -> None:
        rgb = self._validate_rgb(rgb)
        self._ensure_capacity(slot + 1)

        if self._gpu_preprocess:
            # Copy camera numpy memory into a persistent pinned allocation first.
            # This makes the following H2D copy eligible for non-blocking DMA.
            with self._stage("tracker_input_host_copy_cpu", cuda=False):
                np.copyto(self._host_np, rgb, casting="no")

            with self._stage("tracker_input_preprocess_gpu", cuda=True):
                self._device_hwc_u8.copy_(
                    self._host_u8,
                    non_blocking=bool(self._host_u8.is_pinned()),
                )
                image = (
                    self._device_hwc_u8
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(dtype=torch.float32)
                )
                image.mul_(1.0 / 255.0)
                image = F.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                image.sub_(self._mean).div_(self._std)
                self._frame_buffer[slot].copy_(image[0], non_blocking=True)
            return

        # Compatibility path for CPU-offloaded video frames or CPU execution.
        with self._stage("tracker_input_host_copy_cpu", cuda=False):
            image = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            image.mul_(1.0 / 255.0)
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            image.sub_(self._mean).div_(self._std)
            self._frame_buffer[slot].copy_(
                image[0].to(self.storage_device, non_blocking=True)
            )

    def reset(self, rgb: np.ndarray) -> None:
        """Reuse the existing inference state for a new SAM3 keyframe.

        This reproduces the old semantic behavior (discard previous tracking
        history and reseed frame 0) without creating a new JPEG directory or
        calling predictor.init_state again.
        """
        self._validate_rgb(rgb)
        self.predictor.reset_state(self.state)

        # Image features are image-dependent and must not survive a keyframe.
        cached = self.state.get("cached_features")
        if hasattr(cached, "clear"):
            cached.clear()
        else:
            self.state["cached_features"] = {}

        # model constants such as the cached mask-memory positional encoding are
        # image independent; keeping them avoids unnecessary recomputation.
        self._preprocess_into_slot(rgb, 0)
        self._set_visible_length(1)
        self.state["video_height"] = self.height
        self.state["video_width"] = self.width
        self.frame_index = 0

    def append(self, rgb: np.ndarray) -> int:
        next_index = int(self.state["num_frames"])
        self._preprocess_into_slot(rgb, next_index)
        self._set_visible_length(next_index + 1)
        self.frame_index = next_index
        return next_index

    def close(self) -> None:
        self.temp_dir.cleanup()
