#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <rq1|rq2a|rq2b|rq3a|rq3b|rq4a|rq4b> [ROS 2 launch arguments...]" >&2
  exit 2
fi

research_question="$1"
shift

case "${research_question}" in
  rq1|rq2a|rq2b)
    scene_arguments=(
      "bin_count:=5"
      "context_light_count:=0"
      "show_bin_color_labels:=true"
    )
    ;;
  rq3a)
    scene_arguments=(
      "bin_count:=5"
      "context_light_count:=1"
      "show_bin_color_labels:=true"
    )
    ;;
  rq3b)
    scene_arguments=(
      "bin_count:=5"
      "context_light_count:=2"
      "show_bin_color_labels:=true"
    )
    ;;
  rq4a)
    scene_arguments=(
      "bin_count:=7"
      "context_light_count:=0"
      "show_bin_color_labels:=true"
    )
    ;;
  rq4b)
    scene_arguments=(
      "bin_count:=7"
      "context_light_count:=0"
      "color_target_bins:=true"
    )
    ;;
  *)
    echo "Unknown research question: ${research_question}" >&2
    echo "Expected one of: rq1, rq2a, rq2b, rq3a, rq3b, rq4a, rq4b" >&2
    exit 2
    ;;
esac

scene_arguments+=("research_question:=${research_question^^}")

source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/thesis_ws/install/setup.bash"

exec ros2 launch adaptive_sorting_bringup panda_demo.launch.py \
  "${scene_arguments[@]}" "$@"
