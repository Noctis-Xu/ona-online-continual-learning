"""Plot accuracy across the recurring RQ2b rule sequence."""

from __future__ import annotations

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_ROLLING_WINDOW,
    rolling_accuracy as shared_rolling_accuracy,
)


import csv
from dataclasses import dataclass
from pathlib import Path

from adaptive_sorting.analysis.plot_style import plt
from matplotlib.ticker import NullLocator


@dataclass(frozen=True)
class TrialRecord:
    seed: int
    trial: int
    phase_index: int
    phase_label: str
    phase_start: int
    is_correct: bool


def load_trials(path: Path) -> dict[int, list[TrialRecord]]:
    required = {
        "seed",
        "trial",
        "phase_index",
        "phase_label",
        "phase_start",
        "is_correct",
    }
    grouped: dict[int, list[TrialRecord]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial CSV is missing fields: {sorted(missing)}")
        for row in reader:
            record = TrialRecord(
                seed=int(row["seed"]),
                trial=int(row["trial"]),
                phase_index=int(row["phase_index"]),
                phase_label=row["phase_label"],
                phase_start=int(row["phase_start"]),
                is_correct=row["is_correct"] == "True",
            )
            grouped.setdefault(record.seed, []).append(record)
    if not grouped:
        raise ValueError("Trial CSV contains no records.")
    for records in grouped.values():
        records.sort(key=lambda record: record.trial)
    return grouped


def rolling_accuracy(records: list[TrialRecord], window: int) -> list[float]:
    return shared_rolling_accuracy(records, window, correct=lambda r: r.is_correct, include=lambda r: True)

def set_phase_ticks(axis, records: list[TrialRecord]) -> None:
    """Show phase boundaries and the rule applying in each interval."""
    starts = [r for r in records if r.trial == r.phase_start]
    boundaries = [r.phase_start - 1 for r in starts[1:]]
    axis.set_xticks([*boundaries, records[-1].trial],
                    [str(r.phase_start - 1) for r in starts[1:]] + [str(records[-1].trial)])
    axis.xaxis.set_minor_locator(NullLocator())
    for i, record in enumerate(starts):
        end = starts[i + 1].phase_start - 1 if i + 1 < len(starts) else records[-1].trial
        axis.text((record.phase_start + end) / 2, .98, record.phase_label,
                  transform=axis.get_xaxis_transform(), ha='center', va='top', fontsize=9)


def plot_report(
    grouped: dict[int, list[TrialRecord]],
    output_path: Path,
    agent_label: str,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> None:
    from adaptive_sorting.analysis.plot_style import validate_curves, draw_curve, accuracy_axis, finish_figure
    trials = validate_curves({agent_label: grouped})
    records = next(iter(grouped.values()))
    figure, axis = plt.subplots(figsize=(12, 4.8), layout='constrained')
    draw_curve(axis, trials, [rolling_accuracy(rows, rolling_window) for rows in grouped.values()], agent_label, '#087F8C')
    starts = [r.phase_start for r in records if r.trial == r.phase_start and r.phase_start > 1]
    accuracy_axis(axis, len(trials), rolling_window, baseline=.2, starts=starts)
    set_phase_ticks(axis, records)
    axis.set_title(agent_label)
    finish_figure(figure, output_path)
