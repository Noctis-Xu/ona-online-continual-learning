"""Run Experiment 2a: a single change of two, three, or four color mappings."""

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


BEFORE_RULE = "initial"
AFTER_RULES = {
    2: "local_update_after",
    3: "local_update_3_after",
    4: "local_update_4_after",
}


SeedResult = dict[str, object]


@dataclass(frozen=True)
class SeedJob:
    change_size: int
    seed: int
    args: argparse.Namespace


def changed_colors(before: TaskRule, after: TaskRule) -> set[str]:
    if before.mapping.keys() != after.mapping.keys():
        raise ValueError("RQ2 rules must contain the same colors.")
    if before.action_space != after.action_space:
        raise ValueError("RQ2 rules must use the same action space.")
    return {
        color
        for color, action in before.mapping.items()
        if after.mapping[color] != action
    }




def run_seed(
    change_size: int,
    seed: int,
    before_trials: int,
    after_trials: int,
    final_window: int,
    config_path: Path,
    agent_factory: AgentFactory,
    backend: ExecutionBackend | None = None,
    perception: PerceptionInterface | None = None,
) -> tuple[SeedResult, list[dict[str, object]]]:
    rules = load_task_rules(config_path)
    before = rules[BEFORE_RULE]
    try:
        after_rule = AFTER_RULES[change_size]
    except KeyError as exc:
        raise ValueError(f"Unsupported RQ2 change size: {change_size}.") from exc
    after = rules[after_rule]
    changed = changed_colors(before, after)
    if len(changed) != change_size:
        raise ValueError(
            f"RQ2 rule {after_rule!r} changes {len(changed)} colors, "
            f"expected {change_size}."
        )

    seeds = derive_replicate_seeds(seed)
    env = SortingTaskEnv(
        rules,
        initial_rule=BEFORE_RULE,
        rng=random.Random(seeds.environment),
    )
    active_backend = backend if backend is not None else NoOpBackend()
    rows: list[dict[str, object]] = []

    with agent_factory(env.action_space, seeds.agent) as agent:
        env.reset()
        phases = (
            ("before_update", BEFORE_RULE, before_trials),
            ("after_update", after_rule, after_trials),
        )
        for phase, rule_name, phase_trials in phases:
            env.set_rule(rule_name)
            for phase_trial in range(1, phase_trials + 1):
                interaction = run_interaction(env, agent, active_backend, perception)
                observation = interaction.observation
                action = interaction.action
                outcome = interaction.outcome
                rows.append(
                    {
                        "seed": seed,
                        "change_size": change_size,
                        "trial": len(rows) + 1,
                        "phase": phase,
                        "phase_trial": phase_trial,
                        "change_point": before_trials + 1,
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
                )

    config = EvaluationConfig(final_window)
    return evaluate_seed("rq2a", rows, config, config_path)[0], rows


def run_seed_job(job: SeedJob) -> tuple[SeedResult, list[dict[str, object]]]:
    args = job.args
    return run_seed(
        change_size=job.change_size,
        seed=job.seed,
        before_trials=args.before_trials,
        after_trials=args.after_trials,
        final_window=args.final_window,
        config_path=args.config,
        agent_factory=build_agent_factory(args),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--change-sizes",
        type=int,
        nargs="+",
        choices=sorted(AFTER_RULES),
        default=[2, 3, 4],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--before-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--after-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    add_worker_argument(parser)
    add_agent_arguments(parser, include_relational_ona=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TASK_RULES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.before_trials <= 0 or args.after_trials <= 0:
        raise ValueError("before-trials and after-trials must be positive.")
    if not 0 < args.final_window <= args.after_trials:
        raise ValueError("final-window must fit within the after-update phase.")

    jobs = [
        SeedJob(change_size=change_size, seed=seed, args=args)
        for change_size in args.change_sizes
        for seed in args.seeds
    ]
    workers = agent_worker_count(args)
    run_dir = prepare_run(args, "rq2a")
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
            "research_question": "rq2a",
            "change_sizes": args.change_sizes,
            **replicate_seed_metadata(args.seeds),
            "before_trials": args.before_trials,
            "after_trials": args.after_trials,
            "final_window": args.final_window,
            "workers": args.workers,
            "effective_workers": effective_worker_count(workers, len(jobs)),
            **agent_metadata(args),
        },
    )
    from adaptive_sorting.analysis.plot_rq2a import load_trials, plot_report
    from adaptive_sorting.analysis.plot_rq2a_summary import (
        load_summary,
        plot_comparison,
    )

    for change_size in args.change_sizes:
        plot_report(
            load_trials(trial_path, change_size),
            run_dir / f"{change_size}_changes.png",
            agent_label=agent_label(args.agent, args.ona_binary),
        )
    plot_comparison(load_summary(summary_path), run_dir / "change_size_summary.png")
    representative_size = 2 if 2 in args.change_sizes else args.change_sizes[0]
    representative_results = [
        result for result in results if result["change_size"] == representative_size
    ]
    print(
        f"final_overall_accuracy="
        f"{mean(result['final_overall_accuracy'] for result in representative_results):.3f}"
    )
    print(f"trial_log={trial_path}")
    print(f"summary_log={summary_path}")
    print(f"report_dir={run_dir}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    run_cli(main)
