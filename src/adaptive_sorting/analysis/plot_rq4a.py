"""Plot RQ4 novel-input learning and known-input retention."""

from __future__ import annotations
from adaptive_sorting.analysis.plot_style import single_agent_report

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_ROLLING_WINDOW,
    rolling_accuracy as shared_rolling_accuracy,
)



import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TrialRecord:
    seed: int
    trial: int
    is_novel: bool
    is_correct: bool
    expanded_phase_start: int


def load_trials(path: Path) -> dict[int, list[TrialRecord]]:
    required = {"seed", "trial", "is_novel", "is_correct", "expanded_phase_start"}
    grouped: dict[int, list[TrialRecord]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial CSV is missing fields: {sorted(missing)}")
        for row in reader:
            if row["is_novel"] not in {"True", "False"}:
                raise ValueError(f"Invalid is_novel value: {row['is_novel']!r}")
            if row["is_correct"] not in {"True", "False"}:
                raise ValueError(f"Invalid is_correct value: {row['is_correct']!r}")
            record = TrialRecord(
                seed=int(row["seed"]),
                trial=int(row["trial"]),
                is_novel=row["is_novel"] == "True",
                is_correct=row["is_correct"] == "True",
                expanded_phase_start=int(row["expanded_phase_start"]),
            )
            grouped.setdefault(record.seed, []).append(record)
    if not grouped:
        raise ValueError("Trial CSV contains no data.")
    for records in grouped.values():
        records.sort(key=lambda record: record.trial)
    return grouped


def rolling_accuracy(records: list[TrialRecord], window: int, category: str = "overall") -> list[float]:
    if category not in {"overall", "known", "novel"}:
        raise ValueError(f"Unknown category: {category}")
    return shared_rolling_accuracy(records, window, correct=lambda r: r.is_correct, include=lambda r: category == "overall" or r.is_novel == (category == "novel"))

def plot_report(
    grouped: dict[int, list[TrialRecord]],
    output_path: Path,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    random_baseline: float = 1 / 7,
    agent_label: str = "ONA (flat)",
) -> None:
    first = next(iter(grouped.values()))[0]
    single_agent_report(grouped, output_path, rolling_window,
        [(name, lambda rows, group=group: rolling_accuracy(rows, rolling_window, group))
         for name, group in [('Overall', 'overall'), ('New-task inputs: new colors', 'novel'), ('Old-task inputs: old colors', 'known')]],
        label=agent_label,
        baseline=random_baseline, starts=[first.expanded_phase_start])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--random-baseline", type=float, default=1 / 7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.trials.with_name("report.png")
    plot_report(
        load_trials(args.trials),
        output_path,
        rolling_window=args.rolling_window,
        random_baseline=args.random_baseline,
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
