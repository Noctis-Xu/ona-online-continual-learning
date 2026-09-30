#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <rq1|rq2a|rq2b|rq3a|rq3b|rq4a|rq4b> [experiment arguments...]" >&2
  exit 2
fi

research_question="$1"
shift

case "${research_question}" in
  rq1|rq2a|rq2b|rq3a|rq3b|rq4a|rq4b) ;;
  *)
    echo "Unknown research question: ${research_question}" >&2
    echo "Expected one of: rq1, rq2a, rq2b, rq3a, rq3b, rq4a, rq4b" >&2
    exit 2
    ;;
esac

source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/thesis_ws/install/setup.bash"

exec conda run -n thesis python -m \
  adaptive_sorting.execution.ros2_experiment_demo \
  "${research_question}" "$@"
