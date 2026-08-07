import csv
from pathlib import Path

from sam_rgbd_tracking_benchmark.profiler import FrameProfiler


def test_csv_schema_expands_after_first_frame(tmp_path: Path) -> None:
    profiler = FrameProfiler(
        str(tmp_path / "timing.csv"),
        str(tmp_path / "frames.jsonl"),
        use_cuda_events=False,
        summary_interval=0,
    )
    profiler.begin_frame(frame_index=0)
    with profiler.stage("sam3_total"):
        pass
    profiler.end_frame(keyframe=True)
    profiler.begin_frame(frame_index=1)
    with profiler.stage("tracker_total"):
        pass
    profiler.end_frame(keyframe=False)
    with (tmp_path / "timing.csv").open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "sam3_total_cpu" in reader.fieldnames
    assert "tracker_total_cpu" in reader.fieldnames
    assert len(rows) == 2
