import json
from pathlib import Path

from sam_rgbd_tracking_benchmark.timing_report import summarize_jsonl


def test_timing_report_splits_keyframes(tmp_path: Path) -> None:
    path = tmp_path / "frames.jsonl"
    records = [
        {"keyframe": True, "pipeline_total": 40.0, "sam3_total_gpu": 30.0},
        {"keyframe": False, "pipeline_total": 10.0, "tracker_total_gpu": 7.0},
        {"keyframe": False, "pipeline_total": 12.0, "tracker_total_gpu": 8.0},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records))
    report = summarize_jsonl(path)
    assert report["groups"]["keyframes"]["pipeline_total"]["count"] == 1
    assert report["groups"]["tracking_only"]["pipeline_total"]["mean"] == 11.0
