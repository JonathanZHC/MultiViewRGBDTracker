from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any


EXPECTED_ROOTS = {
    "sam3": Path("/opt/upstream/sam3"),
    "sam_mt": Path("/opt/upstream/sam-mt"),
    "efficient_tam": Path("/opt/upstream/efficient-tam"),
}


def require_parameter(callable_object: Any, parameter: str) -> None:
    signature = inspect.signature(callable_object)
    if parameter not in signature.parameters:
        raise RuntimeError(
            f"{callable_object} no longer exposes required parameter "
            f"{parameter!r}: {signature}"
        )


def require_origin(module: Any, expected_root: Path, name: str) -> str:
    module_file = Path(module.__file__).resolve()
    expected = expected_root.resolve()
    if expected not in module_file.parents:
        raise RuntimeError(
            f"{name} resolved from {module_file}, expected it below {expected}. "
            "Check PYTHONPATH/.pth ordering."
        )
    return str(module_file)


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    # SAM3 currently imports these modules transitively from its image-model
    # builder even though upstream lists them only as optional dependencies.
    import decord
    import pycocotools

    import efficient_track_anything
    import sam2
    import sam3
    from efficient_track_anything.build_efficienttam import (
        build_efficienttam_video_predictor,
    )
    from efficient_track_anything.efficienttam_video_predictor import (
        EfficientTAMVideoPredictor,
    )
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor as SamMTVideoPredictor
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    require_parameter(build_sam3_image_model, "checkpoint_path")
    require_parameter(build_sam3_image_model, "load_from_HF")
    require_parameter(Sam3Processor.set_text_prompt, "prompt")

    require_parameter(build_efficienttam_video_predictor, "config_file")
    require_parameter(EfficientTAMVideoPredictor.add_new_mask, "mask")
    require_parameter(EfficientTAMVideoPredictor.propagate_in_video, "start_frame_idx")

    require_parameter(build_sam2_video_predictor, "config_file")
    require_parameter(SamMTVideoPredictor.add_new_points_or_box, "points_per_object")
    require_parameter(SamMTVideoPredictor.propagate_in_video, "points_per_object")

    print(
        json.dumps(
            {
                "sam3": {
                    "status": "compatible",
                    "origin": require_origin(sam3, EXPECTED_ROOTS["sam3"], "sam3"),
                    "pycocotools": package_version("pycocotools"),
                    "decord": package_version("decord"),
                },
                "efficient_tam": {
                    "status": "compatible",
                    "origin": require_origin(
                        efficient_track_anything,
                        EXPECTED_ROOTS["efficient_tam"],
                        "efficient_track_anything",
                    ),
                },
                "sam_mt": {
                    "status": "compatible",
                    "origin": require_origin(sam2, EXPECTED_ROOTS["sam_mt"], "sam2"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
