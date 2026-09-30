"""Run Experiment 4b: new colors with color-labeled bins."""

from __future__ import annotations

from adaptive_sorting.experiments.console import run_cli
from adaptive_sorting.experiments.run_rq4a import build_parser, run_experiment


def main() -> None:
    args = build_parser(__doc__).parse_args()
    run_experiment(
        args,
        research_question="rq4b",
        bin_colors_visible=True,
    )


if __name__ == "__main__":
    run_cli(main)
