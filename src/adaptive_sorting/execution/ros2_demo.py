"""Execute one observation and bin-placement action through the ROS 2 backend."""

from __future__ import annotations

import argparse

from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.execution.ros2_client import Ros2SortingClient
from adaptive_sorting.execution.execution_backend import ActionExecutionError
from adaptive_sorting.execution.ros2_backend import Ros2Backend
from adaptive_sorting.experiments.console import run_cli


COLORS = ("red", "orange", "yellow", "green", "blue", "indigo", "violet")
ACTIONS = tuple(f"place_to_bin{index}" for index in range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", choices=COLORS, default="red")
    parser.add_argument("--action", choices=ACTIONS, default="place_to_bin1")
    parser.add_argument("--light-1", choices=("on", "off"), default="off")
    parser.add_argument("--light-2", choices=("on", "off"), default="off")
    parser.add_argument("--service-timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = f"light_1_{args.light_1}_light_2_{args.light_2}"
    observation = Observation(args.color, context)
    with Ros2SortingClient(args.service_timeout) as client:
        backend = Ros2Backend(client=client)
        backend.present(observation)
        result = backend.execute(args.action)
    if not result.success:
        raise ActionExecutionError(args.action, result)
    print(
        f"executed color={args.color} action={args.action} "
        f"total={result.duration_seconds:.3f}s "
        f"planning={result.planning_duration_seconds:.3f}s "
        f"motion={result.motion_duration_seconds:.3f}s "
        f"cache_hits={result.trajectory_cache_hits}/3"
    )


if __name__ == "__main__":
    run_cli(main)
