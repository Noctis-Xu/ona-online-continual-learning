#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/thesis_ws/install/setup.bash"

exec conda run -n thesis python -m adaptive_sorting.execution.ros2_demo "$@"
