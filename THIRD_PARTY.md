# Third-party components

The benchmark repository itself is MIT licensed. The Docker build retrieves these official model implementations without adding them as Git submodules:

| Component | Source | Default install location | License remains upstream |
|---|---|---|---|
| SAM3 | `facebookresearch/sam3` | `/opt/upstream/sam3` | See upstream repository |
| SAM-MT | `FudanCVL/SAM-MT` | `/opt/upstream/sam-mt` | See upstream repository/checkpoint terms |
| EfficientTAM | `yformer/EfficientTAM` | `/opt/upstream/efficient-tam` | Apache-2.0 upstream |
| Isaac Sim | NVIDIA Isaac Sim 6.0.1 container | `/isaac-sim` | NVIDIA license/EULA |

No model weights are included in the repository archive. `scripts/download_checkpoints.py` downloads them from their official Hugging Face repositories after the user accepts the corresponding terms.
