"""Run agent comparisons for one or more experiments."""

from __future__ import annotations

import argparse
from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

from adaptive_sorting.agents.ona_agent import (
    DEFAULT_ONA_BINARY,
    DEFAULT_ONA_STARTUP_TIMEOUT,
    DEFAULT_ONA_TIMEOUT,
)
from adaptive_sorting.analysis.evaluation import DEFAULT_FINAL_WINDOW
from adaptive_sorting.experiments.agent_factory import agent_label
from adaptive_sorting.experiments.defaults import (
    DEFAULT_WORKERS,
    DEFAULT_SEED_COUNT,
)
from adaptive_sorting.experiments.ona_profile import ONA_C256_T240_MAX_WORKERS
from adaptive_sorting.experiments.result_paths import (
    DEFAULT_RESULTS_DIR,
    create_run_dir,
)


from adaptive_sorting.experiments.runner import positive_integer, add_worker_argument
from adaptive_sorting.naming import experiment_id


RQ_RUNNERS = {
    "rq1": "adaptive_sorting.experiments.run_rq1",
    "rq2a": "adaptive_sorting.experiments.run_rq2a",
    "rq2b": "adaptive_sorting.experiments.run_rq2b",
    "rq3a": "adaptive_sorting.experiments.run_rq3a",
    "rq3b": "adaptive_sorting.experiments.run_rq3b",
    "rq4a": "adaptive_sorting.experiments.run_rq4a",
    "rq4b": "adaptive_sorting.experiments.run_rq4b",
}
ALL_AGENTS = ("flat_ona", "epsilon_greedy", "ucb1", "sw_ucb", "relational_ona")
ONA_AGENTS = frozenset({"flat_ona", "relational_ona"})
CONTEXT_RQS = frozenset({"rq3a", "rq3b"})
TRIAL_ARGUMENTS = {
    "rq1": ("--trials", None),
    "rq2a": ("--before-trials", "--after-trials"),
    "rq2b": ("--initial-trials", "--phase-trials"),
    "rq3a": ("--before-trials", "--mixed-trials"),
    "rq3b": ("--before-trials", "--mixed-trials"),
    "rq4a": ("--base-trials", "--expanded-trials"),
    "rq4b": ("--base-trials", "--expanded-trials"),
}
PLOT_MODULE = "adaptive_sorting.analysis.plot_agent_comparison"
SUMMARY_MODULE = "adaptive_sorting.analysis.summarize_agent_comparison"

CommandRunner = Callable[[Sequence[str]], int]


def trial_counts(value: str) -> tuple[int, int]:
    if re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must use INITIAL/SUBSEQUENT format")
    initial, subsequent = value.split("/")
    return int(initial), int(subsequent)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "items",
        nargs="+",
        metavar="EXPERIMENT_OR_AGENT",
        help=(
            "Experiment target (rq1, rq2a, rq2b, rq3a, rq3b, rq4a, rq4b, or all) or agent "
            f"({', '.join(ALL_AGENTS)})"
        ),
    )
    parser.add_argument(
        "--seeds",
        "--seed-count",
        dest="seed_count",
        type=positive_integer,
        default=DEFAULT_SEED_COUNT,
        help="number of sequential seeds, starting at 1 (default: 10)",
    )
    add_worker_argument(parser)
    parser.add_argument(
        "--trials",
        type=trial_counts,
        metavar="INITIAL/SUBSEQUENT",
        help=(
            "override initial and subsequent phase lengths; both values are "
            "required (RQ1 uses only INITIAL)"
        ),
    )
    parser.add_argument(
        "--ona-binary",
        type=Path,
        default=DEFAULT_ONA_BINARY,
        help=f"ONA executable (default: {DEFAULT_ONA_BINARY})",
    )
    parser.add_argument(
        "--ona-timeout",
        type=float,
        default=DEFAULT_ONA_TIMEOUT,
        help=f"ONA command timeout in seconds (default: {DEFAULT_ONA_TIMEOUT:g})",
    )
    parser.add_argument(
        "--ona-startup-timeout",
        type=float,
        default=DEFAULT_ONA_STARTUP_TIMEOUT,
        help=(
            "ONA startup timeout in seconds "
            f"(default: {DEFAULT_ONA_STARTUP_TIMEOUT:g})"
        ),
    )
    args = parser.parse_args(argv)

    unknown = [
        item
        for item in args.items
        if item not in RQ_RUNNERS and item != "all" and item not in ALL_AGENTS
    ]
    if unknown:
        parser.error(f"unknown RQ or agent: {unknown[0]}")

    args.targets = [
        item for item in args.items if item in RQ_RUNNERS or item == "all"
    ]
    args.agents = [item for item in args.items if item in ALL_AGENTS]
    if not args.targets:
        parser.error("at least one RQ target is required")
    if len(set(args.targets)) != len(args.targets):
        parser.error("duplicate RQ targets are not allowed")
    if len(set(args.agents)) != len(args.agents):
        parser.error("duplicate agents are not allowed")
    if "all" in args.targets and len(args.targets) != 1:
        parser.error("all cannot be combined with explicit RQ targets")
    selected_targets = (
        tuple(RQ_RUNNERS) if args.targets == ["all"] else args.targets
    )
    if args.trials is not None:
        for target in selected_targets:
            error = trial_count_error(target, args.trials)
            if error is not None:
                parser.error(f"invalid --trials for {target}: {error}")
    return args


def trial_count_error(rq: str, trials: tuple[int, int]) -> str | None:
    initial, subsequent = trials

    if rq == "rq1":
        minimum = DEFAULT_FINAL_WINDOW
        if initial < minimum:
            return f"INITIAL must be at least {minimum}"
        return None
    if rq == "rq2a":
        if initial < 50:
            return "INITIAL must be at least 50"
        minimum = DEFAULT_FINAL_WINDOW
        if subsequent < minimum:
            return f"SUBSEQUENT must be at least {minimum}"
        return None
    if rq == "rq2b":
        if initial < 50:
            return "INITIAL must be at least 50"
        if subsequent < DEFAULT_FINAL_WINDOW:
            return f"SUBSEQUENT must be at least {DEFAULT_FINAL_WINDOW}"
        return None
    if rq in CONTEXT_RQS:
        initial_minimum = 50
        subsequent_minimum = DEFAULT_FINAL_WINDOW
        if initial < initial_minimum:
            return f"INITIAL must be at least {initial_minimum}"
        if subsequent < subsequent_minimum:
            return f"SUBSEQUENT must be at least {subsequent_minimum}"
        if subsequent % 10 != 0:
            return "SUBSEQUENT must be divisible by 10"
        return None
    if rq in {"rq4a", "rq4b"}:
        if initial < 50:
            return "INITIAL must be at least 50"
        minimum = DEFAULT_FINAL_WINDOW
        if subsequent < minimum:
            return f"SUBSEQUENT must be at least {minimum}"
        return None
    raise ValueError(f"Unsupported research question: {rq}")


CHILD_INTERRUPT_GRACE_SECONDS = 60.0


def execute_command(command: Sequence[str]) -> int:
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Ctrl+C also reaches the child; subprocess.run would SIGKILL it after
        # 0.25 s, before it can stop its workers and mark the run interrupted.
        try:
            process.wait(timeout=CHILD_INTERRUPT_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(argument) for argument in arguments)]


def print_timing(label: str, started_at: float, status: str) -> None:
    elapsed = perf_counter() - started_at
    print(
        f"timing={label} status={status} elapsed_seconds={elapsed:.2f}",
        flush=True,
    )


def run_agent(
    rq: str,
    agent: str,
    seeds: Sequence[int],
    workers: int,
    ona_binary: Path,
    ona_timeout: float,
    ona_startup_timeout: float,
    trials: tuple[int, int] | None,
    output_dir: Path,
    execute: CommandRunner,
) -> Path:
    label = agent_label(agent, ona_binary)
    print(f"=== {experiment_id(rq)}: {label} ===", flush=True)
    started_at = perf_counter()
    timing_status = "failed"
    agent_workers = min(workers, ONA_C256_T240_MAX_WORKERS) if agent in ONA_AGENTS else workers
    rq_arguments: list[object] = []
    if trials is not None:
        initial_argument, subsequent_argument = TRIAL_ARGUMENTS[rq]
        rq_arguments.extend((initial_argument, trials[0]))
        if subsequent_argument is not None:
            rq_arguments.extend((subsequent_argument, trials[1]))
    command = module_command(
        RQ_RUNNERS[rq],
        "--agent",
        agent,
        "--seeds",
        *seeds,
        "--workers",
        agent_workers,
        "--ona-binary",
        ona_binary,
        "--ona-timeout",
        ona_timeout,
        "--ona-startup-timeout",
        ona_startup_timeout,
        *rq_arguments,
        "--output-dir",
        output_dir,
    )
    try:
        status = execute(command)
        if status != 0:
            raise RuntimeError(
                f"{agent} experiment failed with exit code {status}."
            )

        trial_path = output_dir / "trials.csv"
        if not trial_path.is_file():
            raise RuntimeError(
                f"{agent} run did not produce a usable {trial_path}."
            )
        timing_status = "completed"
        return trial_path
    finally:
        print_timing(f"{rq}/{agent}", started_at, timing_status)


def comparison_arguments(
    agent_trials: Sequence[tuple[str, Path]],
) -> list[str]:
    arguments: list[str] = []
    for label, trial_path in agent_trials:
        arguments.extend(("--agent-trials", label, str(trial_path)))
    return arguments


def run_comparison_reports(
    rq: str,
    run_dir: Path,
    agent_trials: Sequence[tuple[str, Path]],
    execute: CommandRunner,
) -> None:
    trial_arguments = comparison_arguments(agent_trials)
    print(f"=== {rq}: Agent Comparison ===", flush=True)
    plot_command = module_command(
        PLOT_MODULE,
        "--rq",
        rq,
        *trial_arguments,
        "--output",
        run_dir / "agent_comparison.png",
    )
    if (status := execute(plot_command)) != 0:
        raise RuntimeError(f"{rq} comparison plot failed with exit code {status}.")

    print(f"=== {rq}: Core-Metric Summary ===", flush=True)
    summary_command = module_command(
        SUMMARY_MODULE,
        "--rq",
        rq,
        *trial_arguments,
        "--output-dir",
        run_dir,
    )
    if (status := execute(summary_command)) != 0:
        raise RuntimeError(f"{rq} metric summary failed with exit code {status}.")


def run_rq(
    rq: str,
    agents: Sequence[str],
    seeds: Sequence[int],
    results_dir: Path,
    workers: int = DEFAULT_WORKERS,
    ona_binary: Path = DEFAULT_ONA_BINARY,
    ona_timeout: float = DEFAULT_ONA_TIMEOUT,
    ona_startup_timeout: float = DEFAULT_ONA_STARTUP_TIMEOUT,
    trials: tuple[int, int] | None = None,
    execute: CommandRunner = execute_command,
) -> Path:
    selected_agents = tuple(agents) or ALL_AGENTS
    run_dir = create_run_dir(results_dir, experiment_id(rq))
    agent_trials: list[tuple[str, Path]] = []

    for agent in selected_agents:
        output_dir = run_dir / agent
        trial_path = run_agent(
            rq,
            agent,
            seeds,
            workers,
            ona_binary,
            ona_timeout,
            ona_startup_timeout,
            trials,
            output_dir,
            execute,
        )
        agent_trials.append((agent_label(agent, ona_binary), trial_path))

    if len(agent_trials) > 1:
        run_comparison_reports(rq, run_dir, agent_trials, execute)

    print(f"run_dir={run_dir}")
    for agent in selected_agents:
        print(f"{agent}_results={run_dir / agent}")
    if len(agent_trials) > 1:
        print(f"comparison_report={run_dir / 'agent_comparison.png'}")
        print(f"core_metrics_report={run_dir / 'core_metrics.png'}")
        print(f"core_metrics_summary={run_dir / 'core_metrics.md'}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    started_at = perf_counter()
    status = "failed"
    args = parse_args(argv)
    targets = tuple(RQ_RUNNERS) if args.targets == ["all"] else args.targets
    seeds = tuple(range(1, args.seed_count + 1))
    try:
        for rq in targets:
            run_rq(
                rq,
                args.agents,
                seeds,
                DEFAULT_RESULTS_DIR,
                workers=args.workers,
                ona_binary=args.ona_binary,
                ona_timeout=args.ona_timeout,
                ona_startup_timeout=args.ona_startup_timeout,
                trials=args.trials,
            )
        status = "completed"
    finally:
        print_timing("total", started_at, status)


if __name__ == "__main__":
    main()
