"""Run Experiment 1: initial learning of the color-to-bin mapping."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_CRITERION_WINDOW,
    DEFAULT_FINAL_WINDOW,
    DEFAULT_CRITERION_THRESHOLD,
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
    DEFAULT_RQ1_TRIALS,
)

from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    SortingTaskEnv,
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


SeedResult = dict[str, object]


@dataclass(frozen=True)
class SeedJob:
    seed: int
    args: argparse.Namespace


def run_seed(
    seed: int,
    trials: int,
    final_window: int,
    config_path: Path,
    agent_factory: AgentFactory,
    backend: ExecutionBackend | None = None,
    perception: PerceptionInterface | None = None,
    criterion_window: int = DEFAULT_CRITERION_WINDOW,
    criterion_threshold: float = DEFAULT_CRITERION_THRESHOLD,
) -> tuple[SeedResult, list[dict[str, object]]]:
    rules = load_task_rules(config_path)
    seeds = derive_replicate_seeds(seed)
    env = SortingTaskEnv(
        rules,
        initial_rule="initial",
        rng=random.Random(seeds.environment),
    )
    active_backend = backend if backend is not None else NoOpBackend()
    rows: list[dict[str, object]] = []

    with agent_factory(env.action_space, seeds.agent) as agent:
        env.reset()
        for trial in range(1, trials + 1):
            interaction = run_interaction(env, agent, active_backend, perception)
            observation = interaction.observation
            action = interaction.action
            outcome = interaction.outcome
            rows.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "sphere_color": observation.sphere_color,
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
            )

    return evaluate_seed("rq1", rows, EvaluationConfig(final_window=final_window, criterion_window=criterion_window, criterion_threshold=criterion_threshold), config_path)[0], rows


def run_seed_job(job: SeedJob) -> tuple[SeedResult, list[dict[str, object]]]:
    args = job.args
    return run_seed(
        seed=job.seed,
        trials=args.trials,
        final_window=args.final_window,
        criterion_window=args.criterion_window,
        criterion_threshold=args.criterion_threshold,
        config_path=args.config,
        agent_factory=build_agent_factory(args),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trials", type=int, default=DEFAULT_RQ1_TRIALS)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--criterion-window", type=int, default=DEFAULT_CRITERION_WINDOW)
    parser.add_argument("--criterion-threshold", type=float, default=DEFAULT_CRITERION_THRESHOLD)
    add_worker_argument(parser)
    add_agent_arguments(parser, include_relational_ona=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TASK_RULES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EvaluationConfig(final_window=args.final_window, criterion_window=args.criterion_window, criterion_threshold=args.criterion_threshold)
    if args.trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 < args.final_window <= args.trials:
        raise ValueError("final-window must be between 1 and trials.")

    jobs = [SeedJob(seed=seed, args=args) for seed in args.seeds]
    workers = agent_worker_count(args)
    run_dir = prepare_run(args, "rq1")
    outcomes = run_jobs(run_seed_job, jobs, workers, checkpoint_dir=run_dir)
    results: list[SeedResult] = []
    all_rows: list[dict[str, object]] = []
    for result, rows in outcomes:
        results.append(result)
        all_rows.extend(rows)

    trial_path, summary_path = write_experiment_results(run_dir, results, all_rows)
    write_metadata(
        run_dir,
        {
            "research_question": "rq1",
            **replicate_seed_metadata(args.seeds),
            "trials": args.trials,
            "final_window": args.final_window,
            "criterion_window": args.criterion_window,
            "criterion_threshold": args.criterion_threshold,
            "workers": args.workers,
            "effective_workers": effective_worker_count(workers, len(jobs)),
            **agent_metadata(args),
        },
    )
    from adaptive_sorting.analysis.plot_rq1 import load_trials, plot_report

    plot_report(
        load_trials(trial_path),
        run_dir / "report.png",
        agent_label=agent_label(args.agent, args.ona_binary),
    )
    print(
        f"final_overall_accuracy="
        f"{mean(result['final_overall_accuracy'] for result in results):.3f}"
    )
    print(f"trial_log={trial_path}")
    print(f"summary_log={summary_path}")
    print(f"report={run_dir / 'report.png'}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    run_cli(main)
