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


class _LogicalFrameRing:
    """Logical-frame-index view over a fixed-size physical tensor ring.

    EfficientTAM indexes ``inference_state["images"][frame_idx]`` with the
    monotonically increasing logical video frame index.  A plain tensor would
    therefore need to grow forever.  This adapter preserves that indexing API
    while storing only the most recent fixed number of model-input images.
    """

    def __init__(self, owner: "StreamingVideoState") -> None:
        self._owner = owner

    def __len__(self) -> int:
        return int(self._owner.state.get("num_frames", 0))

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError(
                "Bounded EfficientTAM image storage supports logical integer "
                "frame indexing only; slicing would materialize expired history"
            )

        logical_index = int(index)
        if logical_index < 0:
            logical_index += len(self)
        return self._owner._frame_at_logical_index(logical_index)


class StreamingVideoState:
    """Bounded live-stream bridge for EfficientTAM/SAM2-style predictors.

    ``state['num_frames']`` and the frame indices exposed to EfficientTAM remain
    monotonically increasing, but ``state['images']`` is backed by a fixed-size
    ring.  This prevents the previous append-only GPU tensor from doubling until
    an eventual multi-gigabyte allocation/OOM.

    The ring must be large enough to cover every historical image that may still
    be referenced by a cached EfficientTAM feature snapshot.  The multi-view
    adapter enforces that its capacity is at least ``feature_history_frames + 2``.
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

        # One-time upstream initialization preserves the exact state schema.
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
            torch.device("cpu") if self.offload_video_to_cpu else self.compute_device
        )
        self._gpu_preprocess = bool(
            use_gpu_preprocess
            and self.storage_device.type == "cuda"
            and torch.cuda.is_available()
        )

        loaded_images = self.state["images"]
        if not isinstance(loaded_images, torch.Tensor):
            raise TypeError(
                "Fast streaming requires tensor-backed initial state['images']; "
                f"got {type(loaded_images)!r}"
            )
        if len(loaded_images) < 1:
            raise RuntimeError("Upstream init_state returned no images")

        # Fixed physical storage. This allocation never grows during a session.
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
        self._slot_logical_indices = [-1] * self._capacity
        self._frame_buffer[0].copy_(
            loaded_images[0].to(
                self.storage_device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )
        self._slot_logical_indices[0] = 0
        self._image_ring = _LogicalFrameRing(self)
        self.state["images"] = self._image_ring
        self.state["num_frames"] = 1

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
                [0.485, 0.456, 0.406], dtype=torch.float32
            ).view(1, 3, 1, 1)
            self._std = torch.tensor(
                [0.229, 0.224, 0.225], dtype=torch.float32
            ).view(1, 3, 1, 1)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def oldest_available_frame(self) -> int:
        valid = [idx for idx in self._slot_logical_indices if idx >= 0]
        return min(valid) if valid else -1

    @property
    def newest_available_frame(self) -> int:
        valid = [idx for idx in self._slot_logical_indices if idx >= 0]
        return max(valid) if valid else -1

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

    def _frame_at_logical_index(self, logical_index: int):
        num_frames = int(self.state.get("num_frames", 0))
        if logical_index < 0 or logical_index >= num_frames:
            raise IndexError(
                f"Logical frame {logical_index} is outside [0, {num_frames})"
            )
        slot = logical_index % self._capacity
        stored_index = self._slot_logical_indices[slot]
        if stored_index != logical_index:
            raise IndexError(
                "Requested EfficientTAM image frame has expired from the bounded "
                f"ring: requested={logical_index}, stored_in_slot={stored_index}, "
                f"available=[{self.oldest_available_frame}, "
                f"{self.newest_available_frame}], capacity={self._capacity}"
            )
        return self._frame_buffer[slot]

    def _preprocess_into_logical_frame(self, rgb: np.ndarray, logical_index: int) -> None:
        rgb = self._validate_rgb(rgb)
        slot = int(logical_index) % self._capacity

        if self._gpu_preprocess:
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
        else:
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

        # Update the logical tag only after the image copy has been enqueued.
        self._slot_logical_indices[slot] = int(logical_index)

    def _commit_preprocessed_frame(
        self,
        image_chw: Any,
        logical_index: int,
    ) -> None:
        """Store one already-normalized model image in the bounded ring.

        This is used by the multi-view batch preprocessor so synchronized camera
        views share one H2D + resize/normalize pass instead of repeating the same
        preprocessing pipeline independently per view.
        """
        if torch is None or not torch.is_tensor(image_chw):
            raise TypeError("Preprocessed EfficientTAM image must be a torch.Tensor")
        expected = (3, self.image_size, self.image_size)
        if tuple(image_chw.shape) != expected:
            raise ValueError(
                f"Expected preprocessed image shape {expected}, got {tuple(image_chw.shape)}"
            )
        logical_index = int(logical_index)
        slot = logical_index % self._capacity
        self._frame_buffer[slot].copy_(
            image_chw.to(self.storage_device, dtype=torch.float32, non_blocking=True),
            non_blocking=True,
        )
        self._slot_logical_indices[slot] = logical_index

    def append_preprocessed(self, image_chw: Any) -> int:
        next_index = int(self.state["num_frames"])
        self._commit_preprocessed_frame(image_chw, next_index)
        self.state["num_frames"] = next_index + 1
        self.frame_index = next_index
        return next_index

    def reset(self, rgb: np.ndarray) -> None:
        """Reuse the inference-state object and restart logical frame numbering."""
        self._validate_rgb(rgb)
        self.predictor.reset_state(self.state)

        cached = self.state.get("cached_features")
        if hasattr(cached, "clear"):
            cached.clear()
        else:
            self.state["cached_features"] = {}

        self._slot_logical_indices[:] = [-1] * self._capacity
        self._preprocess_into_logical_frame(rgb, 0)
        self.state["images"] = self._image_ring
        self.state["num_frames"] = 1
        self.state["video_height"] = self.height
        self.state["video_width"] = self.width
        self.frame_index = 0

    def append(self, rgb: np.ndarray) -> int:
        next_index = int(self.state["num_frames"])
        self._preprocess_into_logical_frame(rgb, next_index)
        self.state["num_frames"] = next_index + 1
        self.frame_index = next_index
        return next_index

    def close(self) -> None:
        self.temp_dir.cleanup()


class BatchedStreamingPreprocessor:
    """Reusable synchronized RGB preprocessor for multiple live streams.

    The hot path performs one pinned-host staging copy per view, one batched H2D,
    and one batched cast/resize/normalize operation.  The normalized images are
    then copied into each stream's fixed logical-frame ring.
    """

    def __init__(self, streams: list[StreamingVideoState]) -> None:
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required for batched preprocessing")
        if not streams:
            raise ValueError("At least one stream is required")
        if any(not stream._gpu_preprocess for stream in streams):
            raise ValueError("Batched preprocessing requires GPU preprocessing on every stream")

        first = streams[0]
        self.num_views = len(streams)
        self.height = int(first.height)
        self.width = int(first.width)
        self.image_size = int(first.image_size)
        self.device = torch.device(first.compute_device)
        self.pin_input_memory = bool(first._pin_input_memory)

        for stream in streams[1:]:
            if (stream.height, stream.width) != (self.height, self.width):
                raise ValueError("All batched EfficientTAM streams must share one RGB resolution")
            if int(stream.image_size) != self.image_size:
                raise ValueError("All batched EfficientTAM streams must share one model image size")
            if torch.device(stream.compute_device) != self.device:
                raise ValueError("All batched EfficientTAM streams must share one CUDA device")
            if torch.device(stream.storage_device) != self.device:
                raise ValueError("Batched preprocessing requires CUDA-backed stream storage")

        try:
            self.host_u8 = torch.empty(
                (self.num_views, self.height, self.width, 3),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.pin_input_memory,
            )
        except RuntimeError:
            self.host_u8 = torch.empty(
                (self.num_views, self.height, self.width, 3),
                dtype=torch.uint8,
                device="cpu",
            )
        self.host_np = self.host_u8.numpy()
        self.device_u8 = torch.empty(
            (self.num_views, self.height, self.width, 3),
            dtype=torch.uint8,
            device=self.device,
        )
        # Reuse this cast/permute target every frame; F.interpolate still owns
        # its output tensor, but the full-resolution float input is allocation-free.
        self.device_nchw = torch.empty(
            (self.num_views, 3, self.height, self.width),
            dtype=torch.float32,
            device=self.device,
        )
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def compatible(self, streams: list[StreamingVideoState]) -> bool:
        if len(streams) != self.num_views:
            return False
        for stream in streams:
            if not stream._gpu_preprocess:
                return False
            if (stream.height, stream.width) != (self.height, self.width):
                return False
            if int(stream.image_size) != self.image_size:
                return False
            if torch.device(stream.compute_device) != self.device:
                return False
            if torch.device(stream.storage_device) != self.device:
                return False
        return True

    def append(
        self,
        streams: list[StreamingVideoState],
        rgbs: list[np.ndarray],
    ) -> int:
        if len(streams) != self.num_views or len(rgbs) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} synchronized streams/RGBs, "
                f"got streams={len(streams)} rgbs={len(rgbs)}"
            )
        logical_indices = [int(stream.state["num_frames"]) for stream in streams]
        if len(set(logical_indices)) != 1:
            raise RuntimeError(
                "Multi-view streams became misaligned before batched preprocessing: "
                f"frame_indices={logical_indices}"
            )

        for view_index, (stream, rgb) in enumerate(zip(streams, rgbs)):
            rgb = stream._validate_rgb(rgb)
            np.copyto(self.host_np[view_index], rgb, casting="no")

        self.device_u8.copy_(
            self.host_u8, non_blocking=bool(self.host_u8.is_pinned())
        )
        self.device_nchw.copy_(self.device_u8.permute(0, 3, 1, 2))
        self.device_nchw.mul_(1.0 / 255.0)
        normalized = F.interpolate(
            self.device_nchw,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        normalized.sub_(self.mean).div_(self.std)

        next_index = logical_indices[0]
        for view_index, stream in enumerate(streams):
            committed = stream.append_preprocessed(normalized[view_index])
            if committed != next_index:
                raise RuntimeError(
                    "Batched EfficientTAM append returned inconsistent frame index: "
                    f"expected={next_index}, got={committed}"
                )
        return next_index
