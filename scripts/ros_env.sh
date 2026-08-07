#!/usr/bin/env bash

# Source ROS 2 Jazzy for the system-Python tracking/RViz processes.
# ROS setup scripts read optional variables that are incompatible with nounset.

set +u
source /opt/ros/jazzy/setup.bash
set -u

export VIRTUAL_ENV=/opt/tracking-venv
export PATH="/opt/tracking-venv/bin:${PATH}"
export PYTHONPATH="/workspace/sam_rgbd_tracking_benchmark/src:/opt/upstream/sam3:/opt/upstream/sam-mt:/opt/upstream/efficient-tam:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
