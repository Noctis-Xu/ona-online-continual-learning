"""Create an RQ1 learning report from an ONA trial CSV."""

from __future__ import annotations
from adaptive_sorting.analysis.plot_style import single_agent_report

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_CRITERION_THRESHOLD,
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
    is_correct: bool


def load_trials(path: Path) -> dict[int, list[TrialRecord]]:
    """Load and group trial records by seed."""

    required_fields = {
        "seed",
        "trial",
        "is_correct",
    }
    grouped: dict[int, list[TrialRecord]] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trial CSV is missing fields: {sorted(missing)}")

        for row in reader:
            if row["is_correct"] not in {"True", "False"}:
                raise ValueError(f"Invalid is_correct value: {row['is_correct']!r}")
            record = TrialRecord(
                seed=int(row["seed"]),
                trial=int(row["trial"]),
                is_correct=row["is_correct"] == "True",
            )
            grouped.setdefault(record.seed, []).append(record)

    if not grouped:
        raise ValueError("Trial CSV contains no data.")
    for records in grouped.values():
        records.sort(key=lambda record: record.trial)
    return grouped


def rolling_accuracy(records: list[TrialRecord], window: int) -> list[float]:
    return shared_rolling_accuracy(records, window, correct=lambda r: r.is_correct, include=lambda r: True)

def plot_report(
    grouped: dict[int, list[TrialRecord]],
    output_path: Path,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    threshold: float = DEFAULT_CRITERION_THRESHOLD,
    random_baseline: float = 0.2,
    agent_label: str = "ONA (flat)",
) -> None:
    single_agent_report(grouped, output_path, rolling_window,
        [('Overall', lambda rows: rolling_accuracy(rows, rolling_window))],
        label=agent_label, baseline=random_baseline, reference=threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="RQ1 trial CSV.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent-label", default="ONA (flat)")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CRITERION_THRESHOLD)
    parser.add_argument("--random-baseline", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.trials.with_name("report.png")
    grouped = load_trials(args.trials)
    plot_report(
        grouped,
        output_path,
        rolling_window=args.rolling_window,
        threshold=args.threshold,
        random_baseline=args.random_baseline,
        agent_label=args.agent_label,
    )
    print(f"seeds={len(grouped)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
