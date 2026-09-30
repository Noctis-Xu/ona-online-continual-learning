"""Run Experiment 3a: a mapping that depends on the relevant light L1."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_FINAL_WINDOW,
    EvaluationConfig,
    evaluate_seed,
)

import argparse
from contextlib import nullcontext
from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
import random
from statistics import mean
from typing import (
    Literal,
    Sequence,
)


from adaptive_sorting.experiments.defaults import (
    DEFAULT_SEEDS,
    DEFAULT_INITIAL_TRIALS,
    DEFAULT_SUBSEQUENT_TRIALS,
)

from adaptive_sorting.agents.ona_agent import (
    ONAAgent,
    RELATIONAL_SORTING_ENCODING,
    RULE_FAMILIES,
    classify_relational_rule,
)
from adaptive_sorting.analysis.rq3_relational_diagnostics import (
    DEFAULT_FAMILY_STATS_INTERVAL,
    DEFAULT_SNAPSHOT_INTERVAL,
    MEMORY_DIRECTORY,
    SeedMemoryWriter,
    snapshot_trials,
    write_relational_diagnostics,
)
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    Observation,
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


OFF_RULE = "context_light_off"
ON_RULE = "context_light_on"
RULE_BY_CONTEXT = {"light_off": OFF_RULE, "light_on": ON_RULE}
L2_ABSENT = "absent"
L2_RANDOM = "random"
Light2Mode = Literal["absent", "random"]
LIGHT_2_MODES = (L2_ABSENT, L2_RANDOM)


SeedResult = dict[str, object]


@dataclass(frozen=True)
class SeedJob:
    seed: int
    args: argparse.Namespace
    light_2_mode: Light2Mode = L2_ABSENT
    memory_dir: Path | None = None


@dataclass
class RQ3SeedDiagnostics:
    seed: int
    decisions: list[dict[str, object]] = field(default_factory=list)
    # Full snapshots and family statistics are streamed to memory_dir; only their trials stay here.
    snapshot_trials: list[int] = field(default_factory=list)


def balanced_context_schedule(
    trials: int,
    block_size: int,
    rng: random.Random,
) -> list[str]:
    if trials <= 0 or block_size <= 0:
        raise ValueError("Trials and context block size must be positive.")
    if block_size % 2 != 0:
        raise ValueError("Context block size must be even.")
    if trials % block_size != 0:
        raise ValueError("Mixed trials must be divisible by context block size.")

    half = block_size // 2
    schedule: list[str] = []
    for _ in range(trials // block_size):
        block = ["light_off"] * half + ["light_on"] * half
        rng.shuffle(block)
        schedule.extend(block)
    return schedule


def irrelevant_light_schedule(trials: int, rng: random.Random) -> list[bool]:
    """Sample the task-irrelevant L2 state independently on every trial."""

    if trials <= 0:
        raise ValueError("Trials must be positive.")
    return [bool(rng.getrandbits(1)) for _ in range(trials)]


def light_2_schedule(
    mode: Light2Mode,
    trials: int,
    rng: random.Random,
) -> list[bool | None]:
    if mode == L2_RANDOM:
        return irrelevant_light_schedule(trials, rng)
    if mode == L2_ABSENT:
        return [None] * trials
    raise ValueError(f"Unsupported L2 mode: {mode!r}.")


def joint_light_context(relevant_context: str, light_2_on: bool) -> str:
    try:
        light_1_state = {
            "light_off": "off",
            "light_on": "on",
        }[relevant_context]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported relevant context: {relevant_context!r}."
        ) from exc
    light_2_state = "on" if light_2_on else "off"
    return f"light_1_{light_1_state}_light_2_{light_2_state}"




def run_seed(
    seed: int,
    before_trials: int,
    mixed_trials: int,
    context_block_size: int,
    final_window: int,
    config_path: Path,
    agent_factory: AgentFactory,
    backend: ExecutionBackend | None = None,
    perception: PerceptionInterface | None = None,
    light_2_mode: Light2Mode = L2_ABSENT,
    diagnostics: RQ3SeedDiagnostics | None = None,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
    family_stats_interval: int = DEFAULT_FAMILY_STATS_INTERVAL,
    memory_dir: Path | None = None,
) -> tuple[SeedResult, list[dict[str, object]]]:
    if light_2_mode not in LIGHT_2_MODES:
        raise ValueError(f"Unsupported L2 mode: {light_2_mode!r}.")
    full_snapshots: set[int] = set()
    if diagnostics is not None:
        if diagnostics.seed != seed:
            raise ValueError("RQ3 diagnostics seed must match the run seed.")
        if family_stats_interval <= 0:
            raise ValueError("family-stats-interval must be positive.")
        if memory_dir is None:
            raise ValueError("RQ3 memory diagnostics require a memory directory.")
        full_snapshots = set(snapshot_trials(before_trials, before_trials + mixed_trials, snapshot_interval))
    rules = load_task_rules(config_path)
    off_rule = rules[OFF_RULE]
    on_rule = rules[ON_RULE]
    if off_rule.mapping.keys() != on_rule.mapping.keys():
        raise ValueError("RQ3 context rules must contain the same colors.")
    if off_rule.action_space != on_rule.action_space:
        raise ValueError("RQ3 context rules must use the same action space.")
    if any(
        off_rule.mapping[color] == on_rule.mapping[color]
        for color in off_rule.mapping
    ):
        raise ValueError("Every RQ3 color must map to a different action by context.")

    seeds = derive_replicate_seeds(seed)
    contexts = balanced_context_schedule(
        mixed_trials,
        context_block_size,
        random.Random(seeds.context_schedule),
    )
    light_2_states = light_2_schedule(
        light_2_mode,
        before_trials + mixed_trials,
        random.Random(seeds.distractor_schedule),
    )
    env = SortingTaskEnv(
        rules,
        initial_rule=OFF_RULE,
        rng=random.Random(seeds.environment),
    )
    active_backend = backend if backend is not None else NoOpBackend()
    rows: list[dict[str, object]] = []
    previous_context: str | None = None
    trial_index = 0

    memory = SeedMemoryWriter(memory_dir, seed) if diagnostics is not None else None
    with agent_factory(env.action_space, seeds.agent) as agent, (memory or nullcontext()):
        if diagnostics is not None and not (
            isinstance(agent, ONAAgent)
            and agent.interaction_encoding == RELATIONAL_SORTING_ENCODING
        ):
            raise TypeError("RQ3 rule diagnostics require a ONA (relational) agent.")
        env.reset()
        phases = (
            ("off_only", ["light_off"] * before_trials),
            ("mixed_contexts", contexts),
        )
        for phase, phase_contexts in phases:
            for phase_trial, context in enumerate(phase_contexts, start=1):
                env.set_rule(RULE_BY_CONTEXT[context])
                light_2_on = light_2_states[trial_index]
                trial_index += 1
                base_observation = env.get_observation()
                presented_observation = None
                if light_2_on is not None:
                    presented_observation = Observation(
                        base_observation.sphere_color,
                        joint_light_context(context, light_2_on),
                    )
                interaction = run_interaction(
                    env,
                    agent,
                    active_backend,
                    perception,
                    presented_observation=presented_observation,
                )
                observation = interaction.observation
                action = interaction.action
                outcome = interaction.outcome
                row: dict[str, object] = {
                    "seed": seed,
                    "trial": len(rows) + 1,
                    "phase": phase,
                    "phase_trial": phase_trial,
                    "mixed_phase_start": before_trials + 1,
                    "mixed_phase_trials": mixed_trials,
                    "context": context,
                    "context_switched": (
                        previous_context is not None and context != previous_context
                    ),
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
                if light_2_on is not None:
                    row.update(
                        {
                            "light_1": context.removeprefix("light_"),
                            "light_2": "on" if light_2_on else "off",
                            "joint_context": observation.context,
                        }
                    )
                if diagnostics is not None:
                    decision = agent.last_decision_diagnostic
                    if decision is None:
                        raise RuntimeError("ONA (relational) did not expose decision provenance.")
                    row.update(
                        {
                            "decision_source": decision.source,
                            "driving_rule_family": decision.rule_family or "",
                        }
                    )
                    diagnostics.decisions.append(
                        {
                            "seed": seed,
                            "trial": row["trial"],
                            "decision_source": decision.source,
                            "driving_rule_family": decision.rule_family,
                            "expectation": decision.expectation,
                            "implication": decision.implication,
                        }
                    )
                rows.append(row)
                trial = len(rows)
                # Reading memory prints concepts only; it runs no inference cycles and draws no randomness.
                if diagnostics is not None and (trial % family_stats_interval == 0 or trial in full_snapshots):
                    classified = tuple(
                        (classify_relational_rule(rule), rule)
                        for rule in agent.retained_operation_rules()
                    )
                    memory.record(trial, classified, full=trial in full_snapshots)
                    if trial in full_snapshots:
                        diagnostics.snapshot_trials.append(trial)
                previous_context = context

    config = EvaluationConfig(final_window)
    rq = "rq3b" if light_2_mode == L2_RANDOM else "rq3a"
    return evaluate_seed(rq, rows, config, config_path)[0], rows


def run_seed_job(
    job: SeedJob,
) -> tuple[
    SeedResult,
    list[dict[str, object]],
    RQ3SeedDiagnostics | None,
]:
    args = job.args
    diagnostics = (
        RQ3SeedDiagnostics(job.seed) if args.agent == "relational_ona" else None
    )
    result, rows = run_seed(
        seed=job.seed,
        before_trials=args.before_trials,
        mixed_trials=args.mixed_trials,
        context_block_size=args.context_block_size,
        final_window=args.final_window,
        config_path=args.config,
        agent_factory=build_agent_factory(args),
        light_2_mode=job.light_2_mode,
        diagnostics=diagnostics,
        snapshot_interval=args.snapshot_interval,
        family_stats_interval=args.family_stats_interval,
        memory_dir=job.memory_dir,
    )
    return result, rows, diagnostics


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--before-trials", type=int, default=DEFAULT_INITIAL_TRIALS)
    parser.add_argument("--mixed-trials", type=int, default=DEFAULT_SUBSEQUENT_TRIALS)
    parser.add_argument("--context-block-size", type=int, default=10)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL, help="Trials between full-text memory snapshots (relational ONA only).")
    parser.add_argument("--family-stats-interval", type=int, default=DEFAULT_FAMILY_STATS_INTERVAL, help="Trials between rule-family count and truth summaries (relational ONA only).")
    add_worker_argument(parser)
    add_agent_arguments(parser, include_relational_ona=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_TASK_RULES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.before_trials <= 0 or args.mixed_trials <= 0:
        raise ValueError("before-trials and mixed-trials must be positive.")
    if not 0 < args.final_window <= args.mixed_trials:
        raise ValueError("final-window must fit within the mixed phase.")
    balanced_context_schedule(
        args.mixed_trials,
        args.context_block_size,
        random.Random(0),
    )
    if args.agent == "relational_ona":
        snapshot_trials(args.before_trials, args.before_trials + args.mixed_trials, args.snapshot_interval)
        if args.family_stats_interval <= 0:
            raise ValueError("family-stats-interval must be positive.")


def distractor_schedule_metadata(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    schedules: list[dict[str, object]] = []
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        states = [
            str(row["light_2"])
            for row in rows
            if int(row["seed"]) == seed
        ]
        schedules.append(
            {
                "seed": seed,
                "light_2_on_trials": states.count("on"),
                "light_2_off_trials": states.count("off"),
                "states": states,
            }
        )
    return schedules


def light_2_protocol_metadata(
    mode: Light2Mode,
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    metadata: dict[str, object] = {"light_2_mode": mode}
    if mode == L2_ABSENT:
        return metadata
    if mode == L2_RANDOM:
        metadata.update(
            {
                "protocol_reference": "rq3a",
                "irrelevant_context": "light_2",
                "light_2_distribution": "iid_bernoulli_0.5",
                "distractor_schedules": distractor_schedule_metadata(rows),
            }
        )
        return metadata
    raise ValueError(f"Unsupported L2 mode: {mode!r}.")


def run_experiment(
    args: argparse.Namespace,
    *,
    research_question: str,
    light_2_mode: Light2Mode,
) -> Path:
    validate_args(args)
    if light_2_mode not in LIGHT_2_MODES:
        raise ValueError(f"Unsupported L2 mode: {light_2_mode!r}.")
    workers = agent_worker_count(args)
    run_dir = prepare_run(args, research_question)
    memory_dir = run_dir / MEMORY_DIRECTORY if args.agent == "relational_ona" else None
    jobs = [
        SeedJob(
            seed=seed,
            args=args,
            light_2_mode=light_2_mode,
            memory_dir=memory_dir,
        )
        for seed in args.seeds
    ]
    outcomes = run_jobs(run_seed_job, jobs, workers, checkpoint_dir=run_dir)
    results: list[SeedResult] = []
    all_rows: list[dict[str, object]] = []
    diagnostics: list[RQ3SeedDiagnostics] = []
    for result, rows, seed_diagnostics in outcomes:
        results.append(result)
        all_rows.extend(rows)
        if seed_diagnostics is not None:
            diagnostics.append(seed_diagnostics)

    trial_path, summary_path = write_experiment_results(run_dir, results, all_rows)
    diagnostic_paths: dict[str, Path] = {}
    if diagnostics:
        diagnostic_paths = write_relational_diagnostics(
            run_dir,
            diagnostics,
            all_rows,
            config=EvaluationConfig(args.final_window),
            snapshot_interval=args.snapshot_interval,
        )
    diagnostic_metadata: dict[str, object] = {}
    if diagnostics:
        diagnostic_metadata = {
            "diagnostic_windows": {"final": args.final_window},
            "snapshot_interval": args.snapshot_interval,
            "family_stats_interval": args.family_stats_interval,
            "rule_families": list(RULE_FAMILIES),
            "diagnostic_artifacts": {
                name: path.name for name, path in diagnostic_paths.items()
            },
        }
    write_metadata(
        run_dir,
        {
            "research_question": research_question,
            **replicate_seed_metadata(
                args.seeds,
                uses_context_schedule=True,
                uses_distractor_schedule=light_2_mode == L2_RANDOM,
            ),
            "before_trials": args.before_trials,
            "mixed_trials": args.mixed_trials,
            "context_block_size": args.context_block_size,
            "final_window": args.final_window,
            **diagnostic_metadata,
            **light_2_protocol_metadata(light_2_mode, all_rows),
            **agent_metadata(args),
        },
    )
    from adaptive_sorting.analysis.plot_rq3a import load_trials, plot_report

    plot_report(
        load_trials(trial_path),
        run_dir / "report.png",
        final_window=args.final_window,
        agent_label=agent_label(args.agent, args.ona_binary),
    )
    print(
        f"final_overall_accuracy="
        f"{mean(result['final_overall_accuracy'] for result in results):.3f}"
    )
    print(f"trial_log={trial_path}")
    print(f"summary_log={summary_path}")
    print(f"report={run_dir / 'report.png'}")
    print(f"switch_report={run_dir / 'switch_cost.png'}")
    for name, path in diagnostic_paths.items():
        print(f"{name}={path}")
    print(f"run_dir={run_dir}")
    return run_dir


def main() -> None:
    run_experiment(
        parse_args(),
        research_question="rq3a",
        light_2_mode=L2_ABSENT,
    )


if __name__ == "__main__":
    run_cli(main)
