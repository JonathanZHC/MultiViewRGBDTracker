# syntax=docker/dockerfile:1.7

# Self-contained Isaac Sim 6.0.1 + ROS 2 Jazzy + SAM3 + EfficientTAM tracking image.
#
# Runtime separation is deliberate:
# Isaac scene: /isaac-sim/python.sh
# Tracking:    /opt/tracking-venv/bin/python
#
# The tracking virtualenv is never activated globally and is never added to
# Isaac Sim's PYTHONPATH. This prevents NumPy/SciPy/package collisions.

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1
FROM ${ISAAC_SIM_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG ISAAC_TORCH_VERSION=2.11.0
ARG ISAAC_TORCHVISION_VERSION=0.26.0
ARG TRACKING_TORCH_VERSION=2.8.0
ARG TRACKING_TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG WARP_VERSION=1.15.0

ARG SAM3_REPOSITORY=https://github.com/facebookresearch/sam3.git
ARG SAM3_REF=main

ARG EFFICIENT_TAM_REPOSITORY=https://github.com/JonathanZHC/EfficientTAM.git
ARG EFFICIENT_TAM_REF=main

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1 \
    ISAAC_SIM_PATH=/isaac-sim \
    ROS_DISTRO=jazzy \
    ROS_DOMAIN_ID=117 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNICLIENT_HUB_MODE=disabled \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    HOME=/isaac-sim \
    HF_HOME=/isaac-sim/.cache/huggingface \
    TORCH_HOME=/isaac-sim/.cache/torch \
    MPLCONFIGDIR=/isaac-sim/.cache/matplotlib

# -----------------------------------------------------------------------------
# Isaac + ROS runtime dependencies.
# -----------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs gnupg2 locales lsb-release \
      software-properties-common \
      python3 python3-dev python3.12 python3.12-dev python3.12-venv \
      build-essential cmake ninja-build pkg-config ffmpeg \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 \
      libxrandr2 libxinerama1 libxcursor1 libxi6 xauth mesa-utils \
      libgomp1 \
    && locale-gen en_US.UTF-8 \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-base \
      ros-jazzy-rmw-fastrtps-cpp \
      ros-jazzy-geometry-msgs \
      ros-jazzy-sensor-msgs \
      ros-jazzy-sensor-msgs-py \
      ros-jazzy-std-msgs \
      ros-jazzy-visualization-msgs \
      ros-jazzy-message-filters \
      ros-jazzy-cv-bridge \
      ros-jazzy-tf2-ros \
      ros-jazzy-tf2-tools \
      ros-jazzy-tf2-geometry-msgs \
      ros-jazzy-rviz2 \
      ros-jazzy-image-transport-plugins \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Isaac-side GPU packages used by the scene/sensor publisher.
# Keep these isolated from the tracking venv because they run in separate
# processes and may require different Torch/package versions.
# -----------------------------------------------------------------------------

RUN mkdir -p /opt/isaac-python-packages \
    && /isaac-sim/python.sh -m pip install --no-cache-dir \
      --target /opt/isaac-python-packages \
      torch==${ISAAC_TORCH_VERSION} \
      torchvision==${ISAAC_TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL} \
    && PYTHONPATH=/opt/isaac-python-packages \
      /isaac-sim/python.sh -m pip install --no-cache-dir \
      --target /opt/isaac-python-packages \
      numpy==1.26.4 \
      warp-lang==${WARP_VERSION}

# Append (never prepend) the extra Isaac packages and bind-mounted repository.
RUN /isaac-sim/python.sh - <<'PY'
from pathlib import Path
import site

site_dirs = site.getsitepackages()
if not site_dirs:
    raise RuntimeError("Isaac Sim site-packages directory was not found")

pth = Path(site_dirs[0]) / "sam_rgbd_isaac_paths.pth"
pth.write_text(
    "import sys; "
    "sys.path.append('/opt/isaac-python-packages'); "
    "sys.path.append('/workspace')\n",
    encoding="utf-8",
)
print("created", pth)
PY

# -----------------------------------------------------------------------------
# Isolated tracking environment: SAM3 + EfficientTAM.
# -----------------------------------------------------------------------------

RUN python3.12 -m venv /opt/tracking-venv \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir \
      "setuptools>=75,<81" "wheel>=0.45,<1" \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir \
      torch==${TRACKING_TORCH_VERSION} \
      torchvision==${TRACKING_TORCHVISION_VERSION} \
      --index-url ${TORCH_INDEX_URL} \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir \
      "numpy>=1.26,<2" \
      "scipy>=1.12,<2" \
      "pandas>=2.2,<3" \
      "opencv-python-headless>=4.9,<4.12" \
      "Pillow>=10,<12" \
      "PyYAML>=6,<7" \
      "tqdm>=4.66" \
      "psutil>=5.9" \
      "hydra-core>=1.3.2,<1.4" \
      "omegaconf>=2.3,<2.4" \
      "huggingface-hub>=0.27,<2" \
      "safetensors>=0.4" \
      "transformers>=4.48,<5" \
      "accelerate>=1,<2" \
      "timm>=1.0.17" \
      "ftfy==6.1.1" regex \
      "iopath>=0.1.10" \
      "portalocker>=2.10" \
      "einops>=0.8" \
      ninja \
      "matplotlib>=3.9" \
      "typing_extensions>=4.12" \
      "pycocotools>=2.0.8,<3" \
      "decord==0.6.0" \
      "scikit-image>=0.24" \
      "scikit-learn>=1.5"

# -----------------------------------------------------------------------------
# Upstream model repositories.
# -----------------------------------------------------------------------------

RUN mkdir -p /opt/upstream \
    && git clone "${SAM3_REPOSITORY}" /opt/upstream/sam3 \
    && git -C /opt/upstream/sam3 checkout "${SAM3_REF}" \
    && git -C /opt/upstream/sam3 submodule update --init --recursive \
    && git clone "${EFFICIENT_TAM_REPOSITORY}" /opt/upstream/efficient-tam \
    && git -C /opt/upstream/efficient-tam checkout "${EFFICIENT_TAM_REF}" \
    && git -C /opt/upstream/efficient-tam submodule update --init --recursive \
    && /opt/tracking-venv/bin/python -m pip install --no-cache-dir --no-deps -e /opt/upstream/sam3

# ROS Python bindings + upstream model repos + bind-mounted benchmark repository.
RUN printf '%s\n' \
      '/opt/ros/jazzy/lib/python3.12/site-packages' \
      '/usr/lib/python3/dist-packages' \
      '/opt/upstream/sam3' \
      '/opt/upstream/efficient-tam' \
      '/workspace' \
      > /opt/tracking-venv/lib/python3.12/site-packages/sam_rgbd_tracking_paths.pth

# -----------------------------------------------------------------------------
# Build-time dependency/API smoke test.
# This verifies package compatibility and confirms that the EfficientTAM
# checkout contains the direct-reference API required by the benchmark.
# -----------------------------------------------------------------------------

RUN source /opt/ros/jazzy/setup.bash \
    && /opt/tracking-venv/bin/python - <<'PY'
import cv2
import matplotlib
import numpy
import rclpy
import torch
import sam3
import efficient_track_anything

from cv_bridge import CvBridge
from sam3.model_builder import build_sam3_image_model
from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)
from efficient_track_anything.efficienttam_video_predictor import (
    EfficientTAMVideoPredictor,
)

required_direct_reference_apis = (
    "snapshot_multiview_image_features",
    "correct_multiview_from_reference",
)

missing = [
    name
    for name in required_direct_reference_apis
    if not hasattr(EfficientTAMVideoPredictor, name)
]

if missing:
    raise RuntimeError(
        "EfficientTAM checkout is missing required direct-reference API(s): "
        f"{missing}"
    )

print("tracking dependencies OK")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__, "opencv", cv2.__version__)
print("matplotlib", matplotlib.__version__)
print("ROS", rclpy.__file__, "CvBridge", CvBridge.__name__)
print("SAM3", build_sam3_image_model)
print("EfficientTAM", build_efficienttam_video_predictor)
print("EfficientTAM direct-reference APIs OK")
PY

# -----------------------------------------------------------------------------
# Common entrypoint: source ROS only. Do not activate the tracking venv.
# -----------------------------------------------------------------------------

RUN cat > /usr/local/bin/sam-rgbd-entrypoint <<'EOF_ENTRYPOINT'
#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

exec "$@"
EOF_ENTRYPOINT

RUN chmod +x /usr/local/bin/sam-rgbd-entrypoint

# -----------------------------------------------------------------------------
# Runtime/cache directories use the Isaac image's normal UID/GID 1234.
# -----------------------------------------------------------------------------

RUN install -d -o 1234 -g 1234 -m 0775 \
      /workspace \
      /workspace/checkpoints \
      /workspace/logs \
      /isaac-sim/kit/cache \
      /isaac-sim/.cache/ov \
      /isaac-sim/.cache/warp \
      /isaac-sim/.cache/matplotlib \
      /isaac-sim/.cache/huggingface \
      /isaac-sim/.cache/torch \
      /isaac-sim/.nv/ComputeCache \
      /isaac-sim/.nvidia-omniverse/logs \
      /isaac-sim/.nvidia-omniverse/config \
      /isaac-sim/.local/share/ov/data \
      /tmp/runtime-1234

USER 1234:1234
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/sam-rgbd-entrypoint"]
CMD ["/bin/bash"]