from __future__ import annotations

from typing import Any

from .sam2_adapter import Sam2StyleStreamingTracker


class EfficientTAMTracker(Sam2StyleStreamingTracker):
    @property
    def backend_name(self) -> str:
        return "efficient_tam"

    def _build_predictor(self) -> Any:
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

        return build_efficienttam_video_predictor(
            config_file=self.config_path,
            ckpt_path=self.checkpoint_path,
            device=self.device,
            mode="eval",
            apply_postprocessing=True,
            vos_optimized=self.vos_optimized,
        )
