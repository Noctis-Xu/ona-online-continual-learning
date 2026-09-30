#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_BUILD_PATH="/opt/ros/jazzy/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

source /opt/ros/jazzy/setup.bash
set -u
cd "${SCRIPT_DIR}/thesis_ws"

PATH="${ROS_BUILD_PATH}" /usr/bin/colcon build \
  --symlink-install \
  "$@" \
  --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
