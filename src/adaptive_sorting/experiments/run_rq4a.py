"""Run Experiment 4a: new colors with numbered bins."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_FINAL_WINDOW,
    EvaluationConfig,
    evaluate_seed,
)

import argparse
from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
import random
from statistics import mean
from typing import Sequence


from adaptive_sorting.analysis.rq4_relational_diagnostics import (
    classify_generalization_rule, write_rq4_diagnostics,
)
from adaptive_sorting.experiments.defaults import (
    DEFAULT_SEEDS,
    DEFAULT_INITIAL_TRIALS,
    DEFAULT_SUBSEQUENT_TRIALS,
)

from adaptive_sorting.agents.ona_agent import (
    ONAAgent,
    RELATIONAL_SORTING_ENCODING,
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


BASE_RULE = "novel_input_base"
EXPANDED_RULE = "expanded_inputs"


SeedResult = dict[str, object]


@dataclass(frozen=True)
class SeedJob:
    seed: int
    args: argparse.Namespace
    bin_colors_visible: bool


@dataclass
class RQ4SeedDiagnostics:
    seed: int
    decisions: list[dict[str, object]] = field(default_factory=list)
    snapshots: dict[str, tuple[str, ...]] = field(default_factory=dict)


COLOR_BASE_RULE = "color_target_base"
COLOR_EXPANDED_RULE = "color_target_expanded"


def observable_bin_colors(rule: TaskRule) -> tuple[tuple[str, str], ...]:
    """Return the stable, action-ordered color labels visible on all seven bins."""

    color_by_action = {action: color for color, action in rule.mapping.items()}
    if len(color_by_action) != len(rule.mapping):
        raise ValueError("RQ4 requires a one-to-one color-to-bin mapping.")
    return tuple(
        (action.removeprefix("place_to_"), color_by_action[action])
        for action in rule.action_space
    )


def validate_rq4_rules(base: TaskRule, expanded: TaskRule) -> set[str]:
    if base.action_space != expanded.action_space or len(base.action_space) != 7:
        raise ValueError("RQ4 rules must use the same seven-action space.")
    if len(base.mapping) != 5 or len(expanded.mapping) != 7:
        raise ValueError("RQ4 must expand five known colors to seven colors.")
    for color, action in base.mapping.items():
        if expanded.mapping.get(color) != action:
            raise ValueError("RQ4 expanded rule must preserve every known mapping.")

    novel_mapping = {
        color: action
        for color, action in expanded.mapping.items()
        if color not in base.mapping
    }
    if set(novel_mapping) != {"indigo", "violet"}:
        raise ValueError("RQ4 must introduce indigo and violet.")
    return set(novel_mapping)




def run_seed(
    seed: int,
    base_trials: int,
    expanded_trials: int,
    final_window: int,
    config_path: Path,
    agent_factory: AgentFactory,
    backend: ExecutionBackend | None = None,
    perception: PerceptionInterface | None = None,
    *,
    bin_colors_visible: bool = False,
    diagnostics: RQ4SeedDiagnostics | None = None,
) -> tuple[SeedResult, list[dict[str, object]]]:
    rules = load_task_rules(config_path)
    base_name, expanded_name = ((COLOR_BASE_RULE, COLOR_EXPANDED_RULE) if bin_colors_visible
                                else (BASE_RULE, EXPANDED_RULE))
    base = rules[base_name]
    expanded = rules[expanded_name]
    novel_colors = validate_rq4_rules(base, expanded)
    seeds = derive_replicate_seeds(seed)
    env = SortingTaskEnv(
        rules,
        initial_rule=base_name,
        rng=random.Random(seeds.environment),
    )
    active_backend = backend if backend is not None else NoOpBackend()
    rows: list[dict[str, object]] = []

    with agent_factory(base.action_space, seeds.agent) as agent:
        if (
            bin_colors_visible
            and isinstance(agent, ONAAgent)
            and agent.interaction_encoding == RELATIONAL_SORTING_ENCODING
        ):
            agent.configure_color_addressed_bin_actions(observable_bin_colors(expanded))
        env.reset()
        phases = (
            ("known_only", base_name, base_trials),
            ("expanded_inputs", expanded_name, expanded_trials),
        )
        for phase, rule_name, phase_trials in phases:
            env.set_rule(rule_name)
            for phase_trial in range(1, phase_trials + 1):
                interaction = run_interaction(env, agent, active_backend, perception)
                observation = interaction.observation
                action = interaction.action
                outcome = interaction.outcome
                decision = (
                    agent.last_decision_diagnostic
                    if isinstance(agent, ONAAgent)
                    else None
                )
                row: dict[str, object] = {
                    "seed": seed,
                    "trial": len(rows) + 1,
                    "phase": phase,
                    "phase_trial": phase_trial,
                    "expanded_phase_start": base_trials + 1,
                    "is_novel": observation.sphere_color in novel_colors,
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
                    "bin_colors_visible": bin_colors_visible,
                    "decision_source": decision.source if decision else "n/a",
                    "driving_rule": decision.implication if decision else None,
                    "driving_rule_class": (
                        classify_generalization_rule(decision.implication)
                        if decision
                        else "n/a"
                    ),
                }
                rows.append(row)
                if diagnostics is not None:
                    diagnostics.decisions.append(dict(row))

            if diagnostics is not None and isinstance(agent, ONAAgent):
                diagnostics.snapshots[phase] = agent.retained_operation_rules()

    config = EvaluationConfig(final_window)
    rq = "rq4b" if bin_colors_visible else "rq4a"
    return evaluate_seed(rq, rows, config, config_path)[0], rows


def run_seed_job(
    job: SeedJob,
) -> tuple[SeedResult, list[dict[str, object]], RQ4SeedDiagnostics | None]:
    args = job.args
    diagnostics = (
        RQ4SeedDiagnostics(job.seed) if args.agent == "relational_ona" else None
    )
    result, rows = run_seed(
        seed=job.seed,
        base_trials=args.base_trials,
        expanded_trials=args.expanded_trials,
        final_window=args.final_window,
        config_path=args.config,
        agent_factory=build_agent_factory(args),
        bin_colors_visible=job.bin_colors_visible,
        diagnostics=diagnostics,
    )
    return result, rows, diagnostics


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--base-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--expanded-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    add_worker_argument(parser)
    add_agent_arguments(parser, include_relational_ona=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TASK_RULES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.base_trials <= 0 or args.expanded_trials <= 0:
        raise ValueError("base-trials and expanded-trials must be positive.")
    if not 0 < args.final_window <= args.expanded_trials:
        raise ValueError("final-window must fit within the expanded phase.")


def run_experiment(
    args: argparse.Namespace,
    *,
    research_question: str,
    bin_colors_visible: bool,
) -> Path:
    validate_args(args)

    jobs = [
        SeedJob(
            seed=seed,
            args=args,
            bin_colors_visible=bin_colors_visible,
        )
        for seed in args.seeds
    ]
    workers = agent_worker_count(args)
    run_dir = prepare_run(args, research_question)
    outcomes = run_jobs(run_seed_job, jobs, workers, checkpoint_dir=run_dir)
    results: list[SeedResult] = []
    all_rows: list[dict[str, object]] = []
    diagnostics: list[RQ4SeedDiagnostics] = []
    for result, rows, seed_diagnostics in outcomes:
        results.append(result)
        all_rows.extend(rows)
        if seed_diagnostics is not None:
            diagnostics.append(seed_diagnostics)

    trial_path, summary_path = write_experiment_results(run_dir, results, all_rows)
    rules = load_task_rules(args.config)
    base = rules[COLOR_BASE_RULE if bin_colors_visible else BASE_RULE]
    expanded = rules[COLOR_EXPANDED_RULE if bin_colors_visible else EXPANDED_RULE]
    novel_colors = set(expanded.mapping) - set(base.mapping)
    diagnostic_paths = write_rq4_diagnostics(run_dir, diagnostics, novel_colors) if diagnostics else {}
    write_metadata(
        run_dir,
        {
            "research_question": research_question,
            "protocol_reference": "rq4a",
            "bin_colors_visible": bin_colors_visible,
            "target_addressing": "color" if bin_colors_visible else "identity",
            "observation_encoding": (
                "sphere_COLOR for flat ONA and bandit agents; sphere is relation for relational ONA; "
                "color-addressed targets shared by all agents"
                if bin_colors_visible
                else "sphere color only"
            ),
            "relational_action_encoding": (
                "<({SELF} * (bin * COLOR)) --> ^place>"
                if bin_colors_visible and args.agent == "relational_ona"
                else "<({SELF} * binN) --> ^place>"
                if args.agent == "relational_ona" else None
            ),
            **replicate_seed_metadata(args.seeds),
            "base_trials": args.base_trials,
            "expanded_trials": args.expanded_trials,
            "final_window": args.final_window,
            "workers": args.workers,
            "effective_workers": effective_worker_count(workers, len(jobs)),
            "diagnostic_artifacts": {
                name: str(path) for name, path in diagnostic_paths.items()
            },
            **agent_metadata(args),
        },
    )
    from adaptive_sorting.analysis.plot_rq4a import load_trials, plot_report

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
    return run_dir


def main() -> None:
    run_experiment(
        parse_args(),
        research_question="rq4a",
        bin_colors_visible=False,
    )


if __name__ == "__main__":
    run_cli(main)
