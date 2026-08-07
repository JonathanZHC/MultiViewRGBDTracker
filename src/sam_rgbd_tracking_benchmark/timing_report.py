from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def summarize_jsonl(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = {
        "all": records,
        "tracking_only": [record for record in records if not record.get("keyframe", False)],
        "keyframes": [record for record in records if record.get("keyframe", False)],
    }
    output: dict[str, Any] = {"source": str(path), "groups": {}}
    for group_name, group_records in groups.items():
        metric_names = sorted(
            {
                key
                for record in group_records
                for key, value in record.items()
                if isinstance(value, (int, float)) and (key.endswith("_cpu") or key.endswith("_gpu") or key == "pipeline_total")
            }
        )
        output["groups"][group_name] = {
            metric: _stats([float(record[metric]) for record in group_records if metric in record])
            for metric in metric_names
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Profiler frames.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = summarize_jsonl(Path(args.jsonl))
    output = Path(args.output) if args.output else Path(args.jsonl).with_name("timing_summary.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
