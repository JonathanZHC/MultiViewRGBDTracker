# syntax=docker/dockerfile:1.7

ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1
FROM ${ISAAC_SIM_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=jazzy
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG SAM3_REPOSITORY=https://github.com/facebookresearch/sam3.git
ARG SAM3_REF=main
ARG SAM_MT_REPOSITORY=https://github.com/FudanCVL/SAM-MT.git
ARG SAM_MT_REF=main
ARG EFFICIENT_TAM_REPOSITORY=https://github.com/yformer/EfficientTAM.git
ARG EFFICIENT_TAM_REF=main

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=${ROS_DISTRO} \
    ROS_DOMAIN_ID=117 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/tracking-venv \
    PATH=/opt/tracking-venv/bin:${PATH} \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache

# System packages. Isaac images can start as a non-root user and can have their
# apt lists directory removed, so recreate it explicitly.
RUN mkdir -p /var/lib/apt/lists/partial && \
    chown -R root:root /var/lib/apt/lists && \
    chmod 0755 /var/lib/apt/lists && \
    chmod 0700 /var/lib/apt/lists/partial && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg lsb-release software-properties-common \
        git git-lfs build-essential cmake ninja-build pkg-config \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        python3-setuptools python3-wheel \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libx11-6 libx11-xcb1 libxcb1 libxcb-xinerama0 libxcb-cursor0 \
        libxcb-keysyms1 libxcb-render-util0 libxcb-icccm4 libxcb-image0 \
        libxcb-shape0 libxcb-randr0 libxcb-render0 libxcb-xfixes0 \
        libxkbcommon-x11-0 libegl1 libgles2 ffmpeg unzip rsync tmux vim nano \
        htop jq shellcheck && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install --system

# System ROS 2 Jazzy is used by both the Isaac standalone process and the
# external tracking/RViz processes. They remain separate Python processes.
RUN mkdir -p /usr/share/keyrings && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    mkdir -p /var/lib/apt/lists/partial && \
    chmod 0700 /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-base \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-image-transport-plugins \
        ros-${ROS_DISTRO}-message-filters \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-sensor-msgs-py \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-visualization-msgs \
        ros-${ROS_DISTRO}-std-msgs \
        ros-${ROS_DISTRO}-tf2 \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-tf2-geometry-msgs \
        ros-${ROS_DISTRO}-tf2-sensor-msgs \
        ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-pcl-conversions \
        ros-${ROS_DISTRO}-pcl-ros \
        ros-${ROS_DISTRO}-rmw-fastrtps-cpp \
        ros-${ROS_DISTRO}-rosbag2 \
        ros-${ROS_DISTRO}-rosbag2-storage-mcap \
        python3-colcon-common-extensions python3-rosdep && \
    rm -rf /var/lib/apt/lists/*

# Python 3.12 venv for SAM3/SAM-MT/EfficientTAM and external ROS nodes. The
# system site packages expose rclpy/cv_bridge without letting pip modify apt
# managed packages.
RUN python3.12 -m venv --system-site-packages /opt/tracking-venv && \
    python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --ignore-installed \
        "setuptools>=75,<81" "wheel>=0.45,<1"

RUN python -m pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX_URL}

RUN python -m pip install --no-cache-dir \
        "numpy>=1.26,<2" "scipy>=1.12,<2" "pandas>=2.2,<3" \
        "opencv-python-headless>=4.9,<4.12" "Pillow>=10,<12" \
        "PyYAML>=6,<7" "tqdm>=4.66" "psutil>=5.9" \
        "hydra-core>=1.3.2,<1.4" "omegaconf>=2.3,<2.4" \
        "huggingface-hub>=0.27,<2" "safetensors>=0.4" \
        "transformers>=4.48,<5" "accelerate>=1,<2" \
        "timm>=1.0.17" "ftfy==6.1.1" regex "iopath>=0.1.10" \
        "portalocker>=2.10" "einops>=0.8" ninja matplotlib \
        "typing_extensions>=4.12" "pycocotools>=2.0.8,<3" \
        "decord==0.6.0" "scikit-image>=0.24" "scikit-learn>=1.5" \
        "pytest>=8,<10"

# Upstream source is fixed in the image, not attached as submodules to this repo.
RUN mkdir -p /opt/upstream && \
    git clone "${SAM3_REPOSITORY}" /opt/upstream/sam3 && \
    git -C /opt/upstream/sam3 checkout "${SAM3_REF}" && \
    git -C /opt/upstream/sam3 submodule update --init --recursive && \
    git clone "${SAM_MT_REPOSITORY}" /opt/upstream/sam-mt && \
    git -C /opt/upstream/sam-mt checkout "${SAM_MT_REF}" && \
    git -C /opt/upstream/sam-mt submodule update --init --recursive && \
    git clone "${EFFICIENT_TAM_REPOSITORY}" /opt/upstream/efficient-tam && \
    git -C /opt/upstream/efficient-tam checkout "${EFFICIENT_TAM_REF}" && \
    git -C /opt/upstream/efficient-tam submodule update --init --recursive

RUN python -m pip install --no-cache-dir --no-deps -e /opt/upstream/sam3

ENV PYTHONPATH=/workspace/sam_rgbd_tracking_benchmark/src:/opt/upstream/sam3:/opt/upstream/sam-mt:/opt/upstream/efficient-tam

RUN printf '%s\n' \
        '/opt/ros/jazzy/lib/python3.12/site-packages' \
        '/usr/lib/python3/dist-packages' \
        '/opt/upstream/sam3' \
        '/opt/upstream/sam-mt' \
        '/opt/upstream/efficient-tam' \
        '/workspace/sam_rgbd_tracking_benchmark/src' \
        > /opt/tracking-venv/lib/python3.12/site-packages/sam_rgbd_tracking.pth

WORKDIR /workspace/sam_rgbd_tracking_benchmark
COPY . /workspace/sam_rgbd_tracking_benchmark
RUN python -m pip install --no-cache-dir --no-deps -e /workspace/sam_rgbd_tracking_benchmark

# Build-time import test. The build has no GPU; CUDA warnings are expected.
RUN source /opt/ros/jazzy/setup.bash && python - <<'PY'
import inspect
import json
import cv2
import decord
import numpy
import rclpy
import torch
import torchvision
import sam3
import sam2
import efficient_track_anything
import sam_rgbd_tracking_benchmark
from cv_bridge import CvBridge
from sam3.model_builder import build_sam3_image_model
from sam2.build_sam import build_sam2_video_predictor
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

print(json.dumps({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_build": torch.version.cuda,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "decord": decord.__version__,
    "rclpy": inspect.getfile(rclpy),
    "cv_bridge": CvBridge.__name__,
    "sam3": inspect.getfile(sam3),
    "sam3_builder": str(inspect.signature(build_sam3_image_model)),
    "sam_mt": inspect.getfile(sam2),
    "sam_mt_builder": str(inspect.signature(build_sam2_video_predictor)),
    "efficient_tam": inspect.getfile(efficient_track_anything),
    "efficient_tam_builder": str(inspect.signature(build_efficienttam_video_predictor)),
    "local_package": inspect.getfile(sam_rgbd_tracking_benchmark),
}, indent=2))
PY

RUN mkdir -p \
        /workspace/.home \
        /workspace/.cache/huggingface \
        /workspace/.cache/torch \
        /workspace/.cache/warp \
        /workspace/.cache/ov \
        /workspace/.cache/pip \
        /workspace/.cache/nvidia/GLCache \
        /workspace/.nv/ComputeCache \
        /workspace/checkpoints \
        /workspace/datasets \
        /workspace/logs \
        /tmp/runtime-1234-sam-rgbd && \
    chown -R 1234:1234 /workspace/.home /workspace/.cache \
        /workspace/checkpoints /workspace/datasets /workspace/logs \
        /tmp/runtime-1234-sam-rgbd && \
    chmod 0777 /workspace/.home /workspace/.cache /workspace/checkpoints \
        /workspace/datasets /workspace/logs && \
    chmod 0700 /tmp/runtime-1234-sam-rgbd && \
    find /workspace/sam_rgbd_tracking_benchmark/scripts -maxdepth 1 -type f -name '*.sh' -exec chmod +x {} +

# Do not source ROS globally; individual launch scripts source Jazzy safely.
RUN cat > /usr/local/bin/sam-rgbd-entrypoint <<'EOF_ENTRY'
#!/usr/bin/env bash
set -e
export VIRTUAL_ENV=/opt/tracking-venv
export PATH="/opt/tracking-venv/bin:${PATH}"
export HOME="${HOME:-/workspace/.home}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-1234-sam-rgbd}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
mkdir -p "${HOME}" "${XDG_RUNTIME_DIR}" "${HF_HOME}" "${TORCH_HOME}"
chmod 0700 "${XDG_RUNTIME_DIR}" 2>/dev/null || true
exec "$@"
EOF_ENTRY
RUN chmod 0755 /usr/local/bin/sam-rgbd-entrypoint

WORKDIR /workspace/sam_rgbd_tracking_benchmark
ENTRYPOINT ["/usr/local/bin/sam-rgbd-entrypoint"]
CMD ["bash"]
