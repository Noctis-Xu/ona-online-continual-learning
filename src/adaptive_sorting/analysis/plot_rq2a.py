"""Plot RQ2 adaptation and retention curves from an ONA trial CSV."""

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
    change_size: int
    seed: int
    trial: int
    is_changed: bool
    is_correct: bool
    change_point: int


def load_trials(path: Path, change_size: int) -> dict[int, list[TrialRecord]]:
    required_fields = {
        "change_size",
        "seed",
        "trial",
        "is_changed",
        "is_correct",
        "change_point",
    }
    grouped: dict[int, list[TrialRecord]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial CSV is missing fields: {sorted(missing)}")
        for row in reader:
            if int(row["change_size"]) != change_size:
                continue
            if row["is_changed"] not in {"True", "False"}:
                raise ValueError(f"Invalid is_changed value: {row['is_changed']!r}")
            if row["is_correct"] not in {"True", "False"}:
                raise ValueError(f"Invalid is_correct value: {row['is_correct']!r}")
            record = TrialRecord(
                change_size=change_size,
                seed=int(row["seed"]),
                trial=int(row["trial"]),
                is_changed=row["is_changed"] == "True",
                is_correct=row["is_correct"] == "True",
                change_point=int(row["change_point"]),
            )
            grouped.setdefault(record.seed, []).append(record)

    if not grouped:
        raise ValueError(f"Trial CSV contains no data for change size {change_size}.")
    for records in grouped.values():
        records.sort(key=lambda record: record.trial)
    return grouped


def rolling_accuracy(records: list[TrialRecord], window: int, changed: bool | None = None) -> list[float]:
    return shared_rolling_accuracy(records, window, correct=lambda r: r.is_correct, include=lambda r: changed is None or r.is_changed == changed)

def plot_report(
    grouped: dict[int, list[TrialRecord]],
    output_path: Path,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    random_baseline: float = 0.2,
    agent_label: str = "ONA (flat)",
) -> None:
    first = next(iter(grouped.values()))[0]
    single_agent_report(grouped, output_path, rolling_window,
        [(name, lambda rows, group=group: rolling_accuracy(rows, rolling_window, group))
         for name, group in [('Overall', None), ('New-task inputs: changed colors', True), ('Old-task inputs: unchanged colors', False)]],
        label=f'{agent_label}: {first.change_size} changed colors',
        baseline=random_baseline, starts=[first.change_point])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="RQ2 trial CSV.")
    parser.add_argument("--change-size", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent-label", default="ONA (flat)")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--random-baseline", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.trials.with_name(
        f"{args.change_size}_changes.png"
    )
    grouped = load_trials(args.trials, args.change_size)
    plot_report(
        grouped,
        output_path,
        rolling_window=args.rolling_window,
        random_baseline=args.random_baseline,
        agent_label=args.agent_label,
    )
    print(f"seeds={len(grouped)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
