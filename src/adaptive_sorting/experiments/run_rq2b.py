"""Run Experiment 2b: repeated mapping changes with one persistent agent per seed."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_FINAL_WINDOW,
    EvaluationConfig,
    evaluate_seed,
)

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from statistics import mean


from adaptive_sorting.experiments.defaults import (
    DEFAULT_SEEDS,
    DEFAULT_INITIAL_TRIALS,
    DEFAULT_SUBSEQUENT_TRIALS,
)

from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    SortingTaskEnv,
    TaskRule,
    load_task_rules,
)
from adaptive_sorting.experiments.agent_factory import (
    AgentFactory,
    add_agent_arguments,
    agent_label,
    agent_metadata,
    build_agent_factory,
    agent_worker_count,
)
from adaptive_sorting.experiments.console import run_cli
from adaptive_sorting.experiments.interaction import run_interaction
from adaptive_sorting.experiments.result_paths import (
    DEFAULT_RESULTS_DIR,
    prepare_run,
    write_experiment_results,
    write_metadata,
)
from adaptive_sorting.experiments.runner import (
    add_worker_argument,
    effective_worker_count,
    run_jobs,
)
from adaptive_sorting.experiments.seeding import (
    derive_replicate_seeds,
    replicate_seed_metadata,
)
from adaptive_sorting.execution.execution_backend import (
    NoOpBackend,
    ExecutionBackend,
)
from adaptive_sorting.perception import PerceptionInterface


RULE_A = "initial"
RULE_B = "local_update_after"
RULE_C = "recurring_update_c"
PHASE_RULES = (RULE_A, RULE_B, RULE_C, RULE_B, RULE_A, RULE_B, RULE_C, RULE_B, RULE_A)
PHASE_LABELS = ("A", "B", "C", "B", "A", "B", "C", "B", "A")


PhaseResult = dict[str, object]


@dataclass(frozen=True)
class SeedJob:
    seed: int
    args: argparse.Namespace


def changed_colors(before: TaskRule, after: TaskRule) -> set[str]:
    if before.mapping.keys() != after.mapping.keys():
        raise ValueError("RQ2b rules must contain the same colors.")
    if before.action_space != after.action_space:
        raise ValueError("RQ2b rules must use the same action space.")
    return {
        color
        for color, action in before.mapping.items()
        if after.mapping[color] != action
    }




def validate_phase_rules(rules: dict[str, TaskRule]) -> None:
    for before_name, after_name in zip(PHASE_RULES, PHASE_RULES[1:]):
        change_count = len(changed_colors(rules[before_name], rules[after_name]))
        if change_count != 2:
            raise ValueError(
                f"RQ2b transition {before_name} -> {after_name} changes "
                f"{change_count} colors, expected 2."
            )


def run_seed(
    seed: int,
    initial_trials: int,
    phase_trials: int,
    final_window: int,
    config_path: Path,
    agent_factory: AgentFactory,
    backend: ExecutionBackend | None = None,
    perception: PerceptionInterface | None = None,
) -> tuple[list[PhaseResult], list[dict[str, object]]]:
    rules = load_task_rules(config_path)
    validate_phase_rules(rules)
    seeds = derive_replicate_seeds(seed)
    env = SortingTaskEnv(
        rules,
        initial_rule=RULE_A,
        rng=random.Random(seeds.environment),
    )
    active_backend = backend if backend is not None else NoOpBackend()
    all_rows: list[dict[str, object]] = []

    with agent_factory(env.action_space, seeds.agent) as agent:
        env.reset()
        for phase_index, (rule_name, label) in enumerate(
            zip(PHASE_RULES, PHASE_LABELS), start=1
        ):
            trial_count = initial_trials if phase_index == 1 else phase_trials
            previous_rule = rules[PHASE_RULES[phase_index - 2]] if phase_index > 1 else None
            changed = changed_colors(previous_rule, rules[rule_name]) if previous_rule else set()
            phase_start = len(all_rows) + 1
            env.set_rule(rule_name)
            for phase_trial in range(1, trial_count + 1):
                interaction = run_interaction(
                    env,
                    agent,
                    active_backend,
                    perception,
                )
                observation = interaction.observation
                action = interaction.action
                outcome = interaction.outcome
                row = {
                    "seed": seed,
                    "trial": len(all_rows) + 1,
                    "phase_index": phase_index,
                    "phase_label": label,
                    "rule_name": rule_name,
                    "phase_trial": phase_trial,
                    "phase_start": phase_start,
                    "sphere_color": observation.sphere_color,
                    "is_changed": observation.sphere_color in changed,
                    "selected_action": action,
                    "correct_action": outcome.correct_action,
                    "execution_success": interaction.execution.success,
                    "execution_duration_seconds": (
                        interaction.execution.duration_seconds
                    ),
                    "planning_duration_seconds": (
                        interaction.execution.planning_duration_seconds
                    ),
                    "motion_duration_seconds": (
                        interaction.execution.motion_duration_seconds
                    ),
                    "trajectory_cache_hits": (
                        interaction.execution.trajectory_cache_hits
                    ),
                    "reward": outcome.reward,
                    "is_correct": outcome.is_correct,
                }
                all_rows.append(row)

    config = EvaluationConfig(final_window=final_window)
    return evaluate_seed("rq2b", all_rows, config, config_path), all_rows


def run_seed_job(
    job: SeedJob,
) -> tuple[list[PhaseResult], list[dict[str, object]]]:
    args = job.args
    return run_seed(
        seed=job.seed,
        initial_trials=args.initial_trials,
        phase_trials=args.phase_trials,
        final_window=args.final_window,
        config_path=args.config,
        agent_factory=build_agent_factory(args),
    )


def print_summary(results: list[PhaseResult]) -> None:
    """Print the per-seed mean cumulative new-task error over transitions."""
    transitions = [result for result in results if result["phase_index"] > 1]
    values = [r["new_task_errors"] for r in transitions]
    value = f"{mean(values):.1f}" if values and all(v is not None for v in values) else "NA"
    print(f"mean_transition_new_task_errors={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--initial-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--phase-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    add_worker_argument(parser)
    add_agent_arguments(parser, include_relational_ona=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TASK_RULES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initial_trials <= 0 or args.phase_trials <= 0:
        raise ValueError("Phase trial counts must be positive.")
    if not 0 < args.final_window <= args.phase_trials:
        raise ValueError("final-window must fit within post-change phases.")

    jobs = [SeedJob(seed=seed, args=args) for seed in args.seeds]
    workers = agent_worker_count(args)
    run_dir = prepare_run(args, "rq2b")
    outcomes = run_jobs(run_seed_job, jobs, workers, checkpoint_dir=run_dir)
    all_results: list[PhaseResult] = []
    all_rows: list[dict[str, object]] = []
    for results, rows in outcomes:
        all_results.extend(results)
        all_rows.extend(rows)

    trial_path, summary_path = write_experiment_results(run_dir, all_results, all_rows)
    write_metadata(
        run_dir,
        {
            "research_question": "rq2b",
            "phase_labels": PHASE_LABELS,
            "phase_rules": PHASE_RULES,
            **replicate_seed_metadata(args.seeds),
            "initial_trials": args.initial_trials,
            "phase_trials": args.phase_trials,
            "final_window": args.final_window,
            "workers": args.workers,
            "effective_workers": effective_worker_count(workers, len(jobs)),
            **agent_metadata(args),
        },
    )
    from adaptive_sorting.analysis.plot_rq2b import load_trials, plot_report

    plot_report(load_trials(trial_path), run_dir / "report.png", agent_label(args.agent, args.ona_binary))
    print_summary(all_results)
    print(f"trial_log={trial_path}")
    print(f"summary_log={summary_path}")
    print(f"report={run_dir / 'report.png'}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    run_cli(main)
