"""Run one experiment through the persistent ROS 2 backend."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_CRITERION_WINDOW,
    DEFAULT_FINAL_WINDOW,
    METRICS_VERSION,
    DEFAULT_CRITERION_THRESHOLD,
)

from adaptive_sorting.experiments.defaults import (
    DEFAULT_INITIAL_TRIALS,
    DEFAULT_RQ1_TRIALS,
    DEFAULT_SUBSEQUENT_TRIALS,
)


import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import re
import sys
from typing import Sequence

from adaptive_sorting.analysis import (
    plot_rq1,
    plot_rq2a,
    plot_rq2b,
    plot_rq3a,
    plot_rq4a,
)
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    TaskRule,
    load_task_rules,
)
from adaptive_sorting.execution.execution_backend import ExecutionBackend
from adaptive_sorting.execution.ros2_client import (
    Ros2SortingClient,
    SortingSceneConfig,
)
from adaptive_sorting.execution.ros2_backend import Ros2Backend
from adaptive_sorting.experiments import (
    run_rq1,
    run_rq2a,
    run_rq2b,
    run_rq3a,
    run_rq4a,
)
from adaptive_sorting.experiments.agent_factory import (
    add_agent_arguments,
    agent_label,
    agent_metadata,
    build_agent_factory,
)
from adaptive_sorting.experiments.console import run_cli
from adaptive_sorting.experiments.result_paths import (
    DEFAULT_RESULTS_DIR,
    recorded_run,
    write_experiment_results,
    write_metadata,
)
from adaptive_sorting.experiments.seeding import replicate_seed_metadata
from adaptive_sorting.perception import (
    PerceptionInterface,
    Ros2Perception,
)


CONTEXT_RQS = frozenset({"rq3a", "rq3b"})
L2_MODE_BY_RQ = {
    "rq3a": run_rq3a.L2_ABSENT,
    "rq3b": run_rq3a.L2_RANDOM,
}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--service-timeout", type=float, default=20.0)
    parser.add_argument("--status-window", type=int, default=50)
    add_agent_arguments(parser, include_relational_ona=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TASK_RULES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="rq", required=True)

    rq1 = subparsers.add_parser("rq1", help="Initial five-color learning.")
    _add_common_arguments(rq1)
    rq1.add_argument("--trials", type=int, default=DEFAULT_RQ1_TRIALS)
    rq1.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    rq1.add_argument("--criterion-window", type=int, default=DEFAULT_CRITERION_WINDOW)
    rq1.add_argument("--criterion-threshold", type=float, default=DEFAULT_CRITERION_THRESHOLD)

    rq2a = subparsers.add_parser("rq2a", help="Local hidden-rule update.")
    _add_common_arguments(rq2a)
    rq2a.add_argument(
        "--change-size", type=int, choices=sorted(run_rq2a.AFTER_RULES), default=2
    )
    rq2a.add_argument("--before-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    rq2a.add_argument("--after-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    rq2a.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)

    rq2b = subparsers.add_parser("rq2b", help="Recurring local rule updates.")
    _add_common_arguments(rq2b)
    rq2b.add_argument("--initial-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    rq2b.add_argument("--phase-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    rq2b.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)

    for name, help_text in (
        ("rq3a", "Context-dependent switching."),
        ("rq3b", "Context switching with a random irrelevant light."),
    ):
        context_parser = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(context_parser)
        context_parser.add_argument("--before-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
        context_parser.add_argument("--mixed-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
        context_parser.add_argument("--context-block-size", type=int, default=10)
        context_parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)

    for name, help_text in (
        ("rq4a", "Novel-input learning without bin-color labels."),
        ("rq4b", "Generalization with visible bin-color labels."),
    ):
        generalization_parser = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(generalization_parser)
        generalization_parser.add_argument("--base-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
        generalization_parser.add_argument("--expanded-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
        generalization_parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.service_timeout <= 0:
        raise ValueError("service-timeout must be positive.")
    if args.status_window <= 0:
        raise ValueError("status-window must be positive.")
    if args.rq == "rq1":
        if args.trials <= 0:
            raise ValueError("trials must be positive.")
        if not 0 < args.final_window <= args.trials:
            raise ValueError("final-window must fit within the run.")
        return

    if args.rq == "rq2a":
        if args.before_trials <= 0 or args.after_trials <= 0:
            raise ValueError("before-trials and after-trials must be positive.")
        if not 0 < args.final_window <= args.after_trials:
            raise ValueError("final-window must fit within the post-update phase.")
        return

    if args.rq == "rq2b":
        if args.initial_trials <= 0 or args.phase_trials <= 0:
            raise ValueError("initial-trials and phase-trials must be positive.")
        if not 0 < args.final_window <= args.phase_trials:
            raise ValueError("final-window must fit within post-change phases.")
        return

    if args.rq in CONTEXT_RQS:
        if args.before_trials <= 0 or args.mixed_trials <= 0:
            raise ValueError("before-trials and mixed-trials must be positive.")
        if not 0 < args.final_window <= args.mixed_trials:
            raise ValueError("final-window must fit within the mixed phase.")
        run_rq3a.balanced_context_schedule(
            args.mixed_trials,
            args.context_block_size,
            random.Random(0),
        )
        return

    if args.base_trials <= 0 or args.expanded_trials <= 0:
        raise ValueError("base-trials and expanded-trials must be positive.")
    if not 0 < args.final_window <= args.expanded_trials:
        raise ValueError("final-window must fit within the expanded phase.")


def run_demo(
    args: argparse.Namespace,
    backend: ExecutionBackend,
    perception: PerceptionInterface | None,
) -> tuple[object, list[dict[str, object]]]:
    factory = build_agent_factory(args)
    common = {
        "seed": args.seed,
        "config_path": args.config,
        "agent_factory": factory,
        "backend": backend,
        "perception": perception,
    }
    if args.rq == "rq1":
        return run_rq1.run_seed(
            trials=args.trials,
            final_window=args.final_window,
            criterion_window=args.criterion_window,
            criterion_threshold=args.criterion_threshold,
            **common,
        )
    if args.rq == "rq2a":
        return run_rq2a.run_seed(
            change_size=args.change_size,
            before_trials=args.before_trials,
            after_trials=args.after_trials,
            final_window=args.final_window,
            **common,
        )
    if args.rq == "rq2b":
        return run_rq2b.run_seed(
            initial_trials=args.initial_trials,
            phase_trials=args.phase_trials,
            final_window=args.final_window,
            **common,
        )
    if args.rq in CONTEXT_RQS:
        return run_rq3a.run_seed(
            before_trials=args.before_trials,
            mixed_trials=args.mixed_trials,
            context_block_size=args.context_block_size,
            final_window=args.final_window,
            light_2_mode=L2_MODE_BY_RQ[args.rq],
            **common,
        )
    return run_rq4a.run_seed(
        base_trials=args.base_trials,
        expanded_trials=args.expanded_trials,
        final_window=args.final_window,
        bin_colors_visible=args.rq == "rq4b",
        **common,
    )


def _rq_metadata(args: argparse.Namespace) -> dict[str, object]:
    excluded = {
        "rq",
        "service_timeout",
        "status_window",
        "config",
        "results_dir",
        "output_dir",
        "agent",
        "ona_binary",
        "ona_timeout",
        "ona_startup_timeout",
        "ona_cycles",
        "ona_anticipation_confidence",
        "ona_decision_threshold",
        "epsilon",
        "learning_rate",
        "ucb1_exploration",
        "ucb_window",
        "ucb_exploration",
    }
    return {key: value for key, value in vars(args).items() if key not in excluded}


def _context_metadata(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if args.rq not in CONTEXT_RQS:
        return {}
    return run_rq3a.light_2_protocol_metadata(L2_MODE_BY_RQ[args.rq], rows)


def demo_record(args: argparse.Namespace):
    """Use the same configuration snapshot and run identity as quantitative runs."""
    metadata = {
        'backend': 'ros2_moveit',
        'embodied_every_trial': True,
        'checkpoint_directory': None,
        'seeds': [args.seed],
    }
    if args.rq == 'rq2a':
        metadata['change_sizes'] = [args.change_size]
    elif args.rq == 'rq2b':
        metadata['phase_labels'] = list(run_rq2b.PHASE_LABELS)
    return recorded_run(args, args.rq, metadata=metadata)


def save_demo(
    args: argparse.Namespace,
    result: object,
    rows: list[dict[str, object]],
    run_dir: Path,
) -> Path:
    results = result if isinstance(result, list) else [result]
    trial_path, summary_path = write_experiment_results(run_dir, results, rows)
    write_metadata(
        run_dir,
        {
            "status": "running",
            "research_question": args.rq,
            "backend": "ros2_moveit",
            "embodied_every_trial": True,
            "service_timeout_seconds": args.service_timeout,
            "config": str(args.config),
            **_rq_metadata(args),
            "task_config": str(args.config.resolve()),
            "metrics_version": METRICS_VERSION,
            **replicate_seed_metadata(
                [args.seed],
                uses_context_schedule=args.rq in CONTEXT_RQS,
                uses_distractor_schedule=args.rq == "rq3b",
            ),
            **_context_metadata(args, rows),
            **agent_metadata(args),
        },
    )
    label = agent_label(args.agent, args.ona_binary)
    if args.rq == "rq1":
        plot_rq1.plot_report(
            plot_rq1.load_trials(trial_path), run_dir / "report.png", agent_label=label
        )
    elif args.rq == "rq2a":
        plot_rq2a.plot_report(
            plot_rq2a.load_trials(trial_path, args.change_size),
            run_dir / "report.png",
            agent_label=label,
        )
    elif args.rq == "rq2b":
        plot_rq2b.plot_report(
            plot_rq2b.load_trials(trial_path),
            run_dir / "report.png",
            label,
        )
    elif args.rq in CONTEXT_RQS:
        plot_rq3a.plot_report(
            plot_rq3a.load_trials(trial_path),
            run_dir / "report.png",
            switch_output_path=run_dir / "switch_cost.png",
            final_window=args.final_window,
            agent_label=label,
        )
    else:
        plot_rq4a.plot_report(
            plot_rq4a.load_trials(trial_path),
            run_dir / "report.png",
            agent_label=label,
        )

    if args.rq == "rq2b":
        run_rq2b.print_summary(results)
    else:
        print(f"final_overall_accuracy={result['final_overall_accuracy']:.3f}")
    print(f"trial_log={trial_path}")
    print(f"summary_log={summary_path}")
    print(f"report={run_dir / 'report.png'}")
    print(f"run_dir={run_dir}")
    return run_dir


def _warn_about_scene(args: argparse.Namespace, config: SortingSceneConfig) -> None:
    if not config.show_bin_color_labels and not config.color_target_bins:
        print(
            "WARNING: human-only dynamic bin color labels are hidden.",
            file=sys.stderr,
        )


@dataclass(frozen=True)
class SceneRequirements:
    bin_count: int
    colors: tuple[str, ...]
    context_light_count: int
    color_target_bins: bool = False


def _selected_rules(
    args: argparse.Namespace,
    rules: dict[str, TaskRule],
) -> tuple[TaskRule, ...]:
    if args.rq == "rq1":
        names = ("initial",)
    elif args.rq == "rq2a":
        names = (run_rq2a.BEFORE_RULE, run_rq2a.AFTER_RULES[args.change_size])
    elif args.rq == "rq2b":
        names = tuple(dict.fromkeys(run_rq2b.PHASE_RULES))
    elif args.rq in CONTEXT_RQS:
        names = (run_rq3a.OFF_RULE, run_rq3a.ON_RULE)
    elif args.rq == "rq4b":
        names = (run_rq4a.COLOR_BASE_RULE, run_rq4a.COLOR_EXPANDED_RULE)
    else:
        names = (run_rq4a.BASE_RULE, run_rq4a.EXPANDED_RULE)
    return tuple(rules[name] for name in names)


def _scene_requirements(args: argparse.Namespace) -> SceneRequirements:
    rules = _selected_rules(args, load_task_rules(args.config))
    if args.rq == "rq4b":
        actions = set().union(*(set(rule.action_space) for rule in rules))
        target_colors = {action.removeprefix("place_to_bin_") for action in actions}
        return SceneRequirements(
            bin_count=len(actions),
            colors=tuple(sorted(target_colors | set().union(*(set(rule.mapping) for rule in rules)))),
            context_light_count=0,
            color_target_bins=True,
        )
    bin_numbers: set[int] = set()
    colors: set[str] = set()
    for rule in rules:
        colors.update(rule.mapping)
        for action in rule.action_space:
            match = re.fullmatch(r"place_to_bin([1-9][0-9]*)", action)
            if match is None:
                raise ValueError(
                    f"ROS 2 experiment rule {rule.name!r} has unsupported action "
                    f"{action!r}."
                )
            bin_numbers.add(int(match.group(1)))

    bin_count = max(bin_numbers, default=0)
    if bin_numbers != set(range(1, bin_count + 1)):
        raise ValueError("ROS 2 experiment actions must use contiguous bins from 1.")
    return SceneRequirements(
        bin_count=bin_count,
        colors=tuple(sorted(colors)),
        context_light_count=(
            1 if args.rq == "rq3a" else 2 if args.rq in CONTEXT_RQS else 0
        ),
    )


def _phase_lengths(args: argparse.Namespace) -> tuple[int, ...]:
    if args.rq == "rq1":
        return (args.trials,)
    if args.rq == "rq2a":
        return (args.before_trials, args.after_trials)
    if args.rq == "rq2b":
        return (args.initial_trials,) + (args.phase_trials,) * (
            len(run_rq2b.PHASE_RULES) - 1
        )
    if args.rq in CONTEXT_RQS:
        return (args.before_trials, args.mixed_trials)
    return (args.base_trials, args.expanded_trials)


def main() -> None:
    args = parse_args()
    with demo_record(args) as run_dir:
        requirements = _scene_requirements(args)
        with Ros2SortingClient(args.service_timeout) as client:
            perception = Ros2Perception(
                include_context=args.rq in CONTEXT_RQS,
                include_irrelevant_context=args.rq == "rq3b",
                client=client,
            )
            backend = Ros2Backend(client=client)
            scene_config = backend.require_scene_capabilities(
                requirements.bin_count,
                requirements.colors,
                requirements.context_light_count,
                requirements.color_target_bins,
            )
            _warn_about_scene(args, scene_config)
            if args.rq == "rq4a":
                # The expanded rule keeps every known mapping, so its labels
                # are the fixed ground truth for both phases.
                expanded = load_task_rules(args.config)[run_rq4a.EXPANDED_RULE]
                backend.fix_bin_color_labels(expanded.mapping)
            phase_lengths = _phase_lengths(args)
            backend.configure_experiment_status(
                args.rq,
                agent_label(args.agent, args.ona_binary),
                sum(phase_lengths),
                args.status_window,
                phase_lengths,
            )
            result, rows = run_demo(args, backend, perception)
        save_demo(args, result, rows, run_dir)


if __name__ == "__main__":
    run_cli(main)
