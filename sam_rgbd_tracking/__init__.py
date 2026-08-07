"""Minimal reusable RGB-D SAM tracking component."""

from .component import SAMTrackingComponent
from .config import Config, load_config
from .data_types import CameraIntrinsics, FrameResult, ProcessedInstance, RGBDFrame

__all__ = [
    "SAMTrackingComponent",
    "Config",
    "load_config",
    "CameraIntrinsics",
    "RGBDFrame",
    "ProcessedInstance",
    "FrameResult",
]
