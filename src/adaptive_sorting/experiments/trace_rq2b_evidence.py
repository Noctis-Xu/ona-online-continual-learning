"""Replay RQ2b with read-only decision and predictive-memory diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import TextIO

from adaptive_sorting.agents.ona_agent import (
    FLAT_ENCODING,
    GOAL,
    ONAAgent,
    RELATIONAL_SORTING_ENCODING,
)
from adaptive_sorting.analysis.plot_rq2b import (
    load_trials,
    plot_report,
)
from adaptive_sorting.env.sorting_task_env import (
    Action,
    DEFAULT_TASK_RULES,
    Observation,
)
from adaptive_sorting.experiments.defaults import (
    DEFAULT_INITIAL_TRIALS,
    DEFAULT_SUBSEQUENT_TRIALS,
)
from adaptive_sorting.experiments.ona_profile import binary_provenance
from adaptive_sorting.experiments.result_paths import (
    DEFAULT_RESULTS_DIR,
    create_run_dir,
    write_experiment_results,
    write_metadata,
)
from adaptive_sorting.experiments.run_rq2b import (
    PHASE_LABELS,
    run_seed,
)
from adaptive_sorting.experiments.seeding import replicate_seed_metadata


class TracedONAAgent(ONAAgent):
    """Observe existing outputs; diagnostic queries supply no beliefs or goals."""

    def __init__(self, *args, trace: TextIO, snapshot_interval: int, **kwargs) -> None:
        self.trace = trace
        self.snapshot_interval = snapshot_interval
        self.trial = 0
        super().__init__(*args, **kwargs)

    def update(self, observation: Observation, action: Action, reward: int) -> None:
        diagnostic = self.last_decision_diagnostic
        super().update(observation, action, reward)
        self.trial += 1
        record = {
            "trial": self.trial,
            "color": observation.sphere_color,
            "action": action,
            "reward": reward,
            "decision": asdict(diagnostic) if diagnostic else None,
        }
        if self.trial % self.snapshot_interval == 0:
            record["rules_after_feedback"] = [
                line for line in self.send_command("*concepts")
                if GOAL in line and "=/>" in line and "^" in line
            ]
            record["stats_after_feedback"] = self.send_command("*stats")
        self.trace.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ona-binary", type=Path, required=True)
    parser.add_argument("--agent", choices=("flat_ona", "relational_ona"), default="flat_ona")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--phase-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    parser.add_argument("--final-window", type=int, default=200)
    parser.add_argument("--ona-cycles", type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.snapshot_interval <= 0:
        parser.error("snapshot interval must be positive")
    if not 0 < args.final_window <= min(args.initial_trials, args.phase_trials):
        parser.error("final window must fit within every phase")
    run_dir = create_run_dir(DEFAULT_RESULTS_DIR, "rq2b", args.output_dir)
    encoding = RELATIONAL_SORTING_ENCODING if args.agent == "relational_ona" else FLAT_ENCODING
    write_metadata(run_dir, {
        "research_question": "rq2b", "diagnostic_replay": True,
        "agent": args.agent, "interaction_encoding": encoding,
        "phase_labels": PHASE_LABELS,
        **replicate_seed_metadata([args.seed]),
        **binary_provenance(args.ona_binary),
        "initial_trials": args.initial_trials, "phase_trials": args.phase_trials,
        "final_window": args.final_window,
        "inference_cycles": args.ona_cycles,
        "snapshot_interval": args.snapshot_interval,
    })
    with (run_dir / "evidence_trace.jsonl").open("w", encoding="utf-8") as trace:
        def factory(actions, seed):
            return TracedONAAgent(
                actions, seed=seed, binary_path=args.ona_binary,
                inference_cycles=args.ona_cycles, interaction_encoding=encoding,
                trace=trace, snapshot_interval=args.snapshot_interval,
            )

        summaries, rows = run_seed(
            args.seed, args.initial_trials, args.phase_trials, args.final_window,
            DEFAULT_TASK_RULES, factory,
        )
    trial_path, _ = write_experiment_results(run_dir, summaries, rows)
    plot_report(load_trials(trial_path), run_dir / "report.png", args.agent)
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
