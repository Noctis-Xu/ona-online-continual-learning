"""Plot Experiment 2a summaries by the number of remapped colors."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from adaptive_sorting.analysis.summarize_agent_comparison import (
    plot_summary,
    summarize_records,
)


def load_summary(path: Path) -> dict[int, list[dict]]:
    grouped = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        numeric = {"new_task_errors", "old_task_errors", "final_new_task_accuracy", "final_old_task_accuracy", "trials_to_criterion", "criterion_horizon"}
        required = numeric | {"seed", "change_size", "criterion_reached"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Summary lacks the evaluation columns; recompute it from trials.csv.")
        for row in reader:
            record = {key: float(row[key]) if row[key] else None for key in numeric}
            record.update(rq="rq2a", seed=int(row["seed"]), change_size=int(row["change_size"]), criterion_reached=row["criterion_reached"] == "True")
            grouped.setdefault(record["change_size"], []).append(record)
    if not grouped:
        raise ValueError("Summary contains no records.")
    return grouped


def plot_comparison(grouped, output_path):
    records = {"Agent": [r for rows in grouped.values() for r in rows]}
    summaries = summarize_records("rq2a", records)
    plot_summary(output_path, "rq2a", [("Agent", Path())], summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot_comparison(load_summary(args.summary), args.output or args.summary.with_name("comparison.png"))


if __name__ == "__main__":
    main()
