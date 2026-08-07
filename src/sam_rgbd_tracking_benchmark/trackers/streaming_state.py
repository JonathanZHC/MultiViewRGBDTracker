from __future__ import annotations

import tempfile
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
    """Turns an offline SAM2-style video predictor into a live frame stream.

    A one-JPEG directory is used only to obtain an upstream-compatible state.
    Subsequent normalized frames are appended directly to `state["images"]`.
    The benchmark resets this state on every detector keyframe, so it remains
    short and bounded.
    """

    def __init__(self, predictor: Any, first_rgb: np.ndarray, offload_video_to_cpu: bool, offload_state_to_cpu: bool) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for real tracker backends")
        self.predictor = predictor
        self.temp_dir = tempfile.TemporaryDirectory(prefix="sam_rgbd_stream_")
        first_path = Path(self.temp_dir.name) / "000000.jpg"
        Image.fromarray(first_rgb.astype(np.uint8), mode="RGB").save(first_path, quality=95)
        self.state = predictor.init_state(
            video_path=self.temp_dir.name,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=False,
        )
        self.original_height, self.original_width = first_rgb.shape[:2]
        self.frame_index = 0

    def append(self, rgb: np.ndarray) -> int:
        if rgb.shape[:2] != (self.original_height, self.original_width):
            raise ValueError("Frame resolution changed inside a tracking session")
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0)
        image = F.interpolate(
            image,
            size=(self.predictor.image_size, self.predictor.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = (image - mean) / std
        target_device = torch.device("cpu") if self.state["offload_video_to_cpu"] else self.state["device"]
        image = image.to(target_device, non_blocking=True)

        images = self.state["images"]
        if isinstance(images, torch.Tensor):
            self.state["images"] = torch.cat([images, image], dim=0)
        elif hasattr(images, "append"):
            images.append(image[0])
        else:
            raise TypeError(
                f"Unsupported upstream image container {type(images)!r}; "
                "update StreamingVideoState for this upstream revision"
            )
        self.state["num_frames"] = int(self.state["num_frames"]) + 1
        self.frame_index += 1
        return self.frame_index

    def close(self) -> None:
        self.temp_dir.cleanup()
