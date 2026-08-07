from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # compatibility with older huggingface_hub
    from huggingface_hub.utils import HfHubHTTPError


MODELS = {
    "sam3": {
        "repo_id": "facebook/sam3",
        "filename": "sam3.pt",
        "destination": "sam3.pt",
        "gated": True,
    },
    "sam_mt": {
        "repo_id": "FudanCVL/SAM-MT",
        "filename": "checkpoints/sam-mt.pt",
        "destination": "sam_mt.pt",
        "gated": False,
    },
    "efficient_tam": {
        "repo_id": "yunyangx/efficient-track-anything",
        "filename": "efficienttam_s_2.pt",
        "destination": "efficienttam_s_2.pt",
        "gated": False,
    },
}


def fetch(name: str, output: Path, force: bool) -> None:
    spec = MODELS[name]
    destination = output / str(spec["destination"])
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {name}: {destination} already exists")
        return

    print(f"[download] {name}: {spec['repo_id']}/{spec['filename']}")
    try:
        source = Path(
            hf_hub_download(
                repo_id=str(spec["repo_id"]),
                filename=str(spec["filename"]),
                force_download=force,
            )
        )
    except HfHubHTTPError as exc:
        print(f"\nFailed to download {name}: {exc}", file=sys.stderr)
        if spec["gated"]:
            print(
                "SAM3 is gated. Request access at facebook/sam3, then run "
                "./scripts/hf_login.sh before retrying.",
                file=sys.stderr,
            )
        raise SystemExit(2) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    print(f"[done] {destination} ({destination.stat().st_size / 2**20:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark checkpoints")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--skip-sam-mt", action="store_true")
    parser.add_argument("--skip-efficient-tam", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    requested = []
    if not args.skip_sam3:
        requested.append("sam3")
    if not args.skip_sam_mt:
        requested.append("sam_mt")
    if not args.skip_efficient_tam:
        requested.append("efficient_tam")

    for name in requested:
        fetch(name, output, args.force)

    print("\nCheckpoint directory:")
    for path in sorted(output.glob("*.pt")):
        print(f"  {path.name:28s} {path.stat().st_size / 2**20:8.1f} MiB")


if __name__ == "__main__":
    main()
