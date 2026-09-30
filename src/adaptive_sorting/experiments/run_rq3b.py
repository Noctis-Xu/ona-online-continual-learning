"""Run Experiment 3b: Experiment 3a with an independently random irrelevant light L2."""

from __future__ import annotations

from adaptive_sorting.experiments.console import run_cli
from adaptive_sorting.experiments.run_rq3a import L2_RANDOM, build_parser, run_experiment


def main() -> None:
    args = build_parser(__doc__).parse_args()
    run_experiment(
        args,
        research_question="rq3b",
        light_2_mode=L2_RANDOM,
    )


if __name__ == "__main__":
    run_cli(main)
