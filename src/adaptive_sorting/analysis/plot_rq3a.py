"""Plot RQ3 context learning, retention, and switching performance."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_FINAL_WINDOW,
    DEFAULT_ROLLING_WINDOW,
    rolling_accuracy as shared_rolling_accuracy,
)

from adaptive_sorting.analysis.statistics import curve_summary

from adaptive_sorting.analysis.trial_ticks import set_trial_ticks

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from adaptive_sorting.analysis.plot_style import plt, single_agent_report
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class TrialRecord:
    seed: int
    trial: int
    phase: str
    context: str
    context_switched: bool
    is_correct: bool
    mixed_phase_start: int


def load_trials(path: Path) -> dict[int, list[TrialRecord]]:
    required_fields = {
        "seed",
        "trial",
        "phase",
        "context",
        "context_switched",
        "is_correct",
        "mixed_phase_start",
    }
    grouped: dict[int, list[TrialRecord]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial CSV is missing fields: {sorted(missing)}")
        for row in reader:
            if row["context_switched"] not in {"True", "False"}:
                raise ValueError(
                    f"Invalid context_switched value: {row['context_switched']!r}"
                )
            if row["is_correct"] not in {"True", "False"}:
                raise ValueError(f"Invalid is_correct value: {row['is_correct']!r}")
            record = TrialRecord(
                seed=int(row["seed"]),
                trial=int(row["trial"]),
                phase=row["phase"],
                context=row["context"],
                context_switched=row["context_switched"] == "True",
                is_correct=row["is_correct"] == "True",
                mixed_phase_start=int(row["mixed_phase_start"]),
            )
            grouped.setdefault(record.seed, []).append(record)
    if not grouped:
        raise ValueError("Trial CSV contains no data.")
    for records in grouped.values():
        records.sort(key=lambda record: record.trial)
    return grouped


def rolling_accuracy(records: list[TrialRecord], window: int, context: str | None = None) -> list[float]:
    return shared_rolling_accuracy(records, window, correct=lambda r: r.is_correct, include=lambda r: context is None or r.context == context)

def mean_curve(curves: list[list[float]]) -> list[float]:
    if not curves or len({len(c) for c in curves}) != 1:
        raise ValueError("Seed curves must share a trial horizon.")
    return [mean(values) for values in zip(*curves)]

def stable_switch_rates(
    grouped: dict[int, list[TrialRecord]], final_window: int
) -> tuple[float, float]:
    switch_rates = []
    stay_rates = []
    for records in grouped.values():
        mixed = [record for record in records if record.phase == "mixed_contexts"]
        if len(mixed) < final_window:
            raise ValueError("Final window is larger than the mixed phase.")
        final = mixed[-final_window:]
        switched = [record.is_correct for record in final if record.context_switched]
        stayed = [record.is_correct for record in final if not record.context_switched]
        if not switched or not stayed:
            raise ValueError("Final window must contain switch and stay trials.")
        switch_rates.append(mean(switched))
        stay_rates.append(mean(stayed))
    return mean(switch_rates), mean(stay_rates)


def switch_cost_curve(records: list[TrialRecord], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("Switch-cost window must be positive.")

    mixed = [record for record in records if record.phase == "mixed_contexts"]
    values: list[float] = []
    for index in range(len(mixed)):
        if index + 1 < window:
            values.append(float("nan"))
            continue
        window_records = mixed[max(0, index - window + 1) : index + 1]
        switched = [
            record.is_correct for record in window_records if record.context_switched
        ]
        stayed = [
            record.is_correct for record in window_records if not record.context_switched
        ]
        values.append(
            mean(stayed) - mean(switched)
            if switched and stayed
            else float("nan")
        )
    return values


def plot_report(
    grouped: dict[int, list[TrialRecord]],
    output_path: Path,
    switch_output_path: Path | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    switch_window: int = 100,
    final_window: int = DEFAULT_FINAL_WINDOW,
    random_baseline: float = 0.2,
    agent_label: str = "ONA (flat)",
) -> None:
    records_by_seed = list(grouped.values())
    mixed_phase_start = records_by_seed[0][0].mixed_phase_start
    single_agent_report(grouped, output_path, rolling_window,
        [(name, lambda rows, group=group: rolling_accuracy(rows, rolling_window, group))
         for name, group in [('Overall', None), ('L1 on', 'light_on'), ('L1 off', 'light_off')]],
        label=agent_label,
        baseline=random_baseline, starts=[mixed_phase_start])

    switch_figure, switch_axis = plt.subplots(
        figsize=(8, 5.2), constrained_layout=True
    )
    switch_figure.patch.set_facecolor("white")

    switch_rate, stay_rate = stable_switch_rates(grouped, final_window)
    mixed_trial_count = min(
        sum(record.phase == "mixed_contexts" for record in records)
        for records in records_by_seed
    )
    mixed_trials = list(range(1, mixed_trial_count + 1))
    cost_curve = mean_curve(
        [switch_cost_curve(records, switch_window) for records in records_by_seed]
    )
    _, cost_lower, cost_upper = curve_summary(
        [switch_cost_curve(records, switch_window) for records in records_by_seed]
    )
    switch_axis.fill_between(mixed_trials, cost_lower, cost_upper, color='#F2A541', alpha=.14, linewidth=0)
    switch_axis.plot(
        mixed_trials,
        cost_curve,
        color="#F2A541",
        linewidth=2.1,
    )
    switch_axis.axhline(0, color="#7A858C", linewidth=1.2)
    final_switch_cost = stay_rate - switch_rate
    final_window_start = mixed_trial_count - final_window + 1
    switch_axis.axvspan(
        final_window_start,
        mixed_trial_count,
        color="#5BB8A8",
        alpha=0.12,
    )
    switch_axis.axvline(
        final_window_start,
        color="#5A9F95",
        linestyle=":",
        linewidth=1.2,
    )
    switch_axis.text(
        0.97,
        0.95,
        f"Final {final_window}-trial cost\n{final_switch_cost * 100:.1f} pp",
        transform=switch_axis.transAxes,
        ha="right",
        va="top",
        color="#44515A",
    )
    finite_costs = [value for value in cost_curve if value == value]
    max_abs_cost = max((abs(value) for value in finite_costs), default=0.0)
    limit = max(0.1, min(1.0, max_abs_cost + 0.05))
    switch_axis.set_title("Switch cost")
    switch_axis.set_xlabel("Trials since phase change")
    switch_axis.set_ylabel(
        f"Switch cost ({switch_window}-trial window, percentage points)"
    )
    switch_axis.set_xlim(1, mixed_trial_count)
    switch_axis.set_ylim(-limit, limit)
    switch_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value * 100:.0f}")
    )
    set_trial_ticks(switch_axis, mixed_trial_count)
    switch_axis.grid(axis="y", color="#DDE2E5", linewidth=0.8)

    switch_output_path = switch_output_path or output_path.with_name("switch_cost.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    switch_output_path.parent.mkdir(parents=True, exist_ok=True)
    switch_figure.savefig(
        switch_output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(switch_figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="RQ3 trial CSV.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--switch-output", type=Path)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--switch-window", type=int, default=100)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--random-baseline", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.trials.with_name("report.png")
    grouped = load_trials(args.trials)
    plot_report(
        grouped,
        output_path,
        switch_output_path=args.switch_output,
        rolling_window=args.rolling_window,
        switch_window=args.switch_window,
        final_window=args.final_window,
        random_baseline=args.random_baseline,
    )
    print(f"seeds={len(grouped)}")
    print(f"output={output_path}")
    print(f"switch_output={args.switch_output or output_path.with_name('switch_cost.png')}")


if __name__ == "__main__":
    main()
