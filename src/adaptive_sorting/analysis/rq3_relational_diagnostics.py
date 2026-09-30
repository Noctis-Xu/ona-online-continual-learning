"""Write focused retained-rule and decision-attribution artifacts for RQ3."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import re
from pathlib import Path
from statistics import mean
from typing import (
    Iterable,
    Mapping,
    Protocol,
    Sequence,
)

from adaptive_sorting.agents.ona_agent import RULE_FAMILIES
from adaptive_sorting.analysis.evaluation import (
    EvaluationConfig,
    decision_category,
    truth,
)


EXTRA_RULE_FAMILY = "unclassified"
FAMILY_NAMES = (*RULE_FAMILIES, EXTRA_RULE_FAMILY)
DECISION_CATEGORIES = ("relevant_only", "relevant_with_l2", "missing_required", "unclassified", "motor_babbling", "unknown")
DEFAULT_SNAPSHOT_INTERVAL = 1
DEFAULT_FAMILY_STATS_INTERVAL = 1
# ONA prints each retained implication with its truth value as "{frequency confidence}".
TRUTH_PATTERN = re.compile(r"\{([0-9.]+) ([0-9.]+)\}\s*$")
FAMILY_STAT_FIELDS = ("count", "mean_expectation", "max_expectation", "mean_confidence")
STATS_FIELDS = ("seed", "trial", "rule_family", *FAMILY_STAT_FIELDS)
MEMORY_DIRECTORY = "memory"


class SeedDiagnostics(Protocol):
    seed: int
    decisions: list[dict[str, object]]
    snapshot_trials: list[int]


def seed_memory_paths(memory_dir: Path, seed: int) -> tuple[Path, Path]:
    return (memory_dir / f"seed_{seed:04d}_snapshots.jsonl.gz",
            memory_dir / f"seed_{seed:04d}_family_stats.csv.gz")


class SeedMemoryWriter:
    """Stream one seed's memory diagnostics to disk so full snapshots never accumulate in memory."""

    def __init__(self, memory_dir: Path, seed: int) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        snapshot_path, stats_path = seed_memory_paths(memory_dir, seed)
        self._snapshots = gzip.open(snapshot_path, "wt", encoding="utf-8")
        self._stats = gzip.open(stats_path, "wt", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stats, fieldnames=STATS_FIELDS)
        self._writer.writeheader()

    def record(self, trial: int, rules: Sequence[tuple[str, str]], *, full: bool) -> None:
        self._writer.writerows(family_statistics(self.seed, trial, rules))
        if full:
            for family, rule in rules:
                self._snapshots.write(json.dumps({"seed": self.seed, "trial": trial, "rule_family": family, "rule": rule}, sort_keys=True) + "\n")

    def close(self) -> None:
        self._snapshots.close()
        self._stats.close()

    def __enter__(self) -> "SeedMemoryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def snapshot_trials(before_trials: int, total_trials: int, interval: int) -> tuple[int, ...]:
    """Full-text snapshot trials: every `interval` trials plus the end of each phase."""
    if interval <= 0 or not 0 < before_trials < total_trials:
        raise ValueError("Snapshot interval and phase lengths must be positive.")
    return tuple(sorted({*range(interval, total_trials + 1, interval), before_trials, total_trials}))


def rule_expectation(rule: str) -> tuple[float, float]:
    """Return (expectation, confidence) of one printed implication; e = c(f - 0.5) + 0.5."""
    match = TRUTH_PATTERN.search(rule)
    if match is None:
        raise ValueError(f"Retained rule has no truth value: {rule!r}")
    frequency, confidence = float(match.group(1)), float(match.group(2))
    return confidence * (frequency - 0.5) + 0.5, confidence


def family_statistics(seed: int, trial: int, rules: Sequence[tuple[str, str]]) -> list[dict[str, object]]:
    """One row per rule family: retained count and truth summaries (empty families have no truth)."""
    grouped: dict[str, list[tuple[float, float]]] = {family: [] for family in FAMILY_NAMES}
    for family, rule in rules:
        grouped[family if family in grouped else EXTRA_RULE_FAMILY].append(rule_expectation(rule))
    rows = []
    for family, values in grouped.items():
        expectations = [e for e, _ in values]
        rows.append({
            "seed": seed, "trial": trial, "rule_family": family, "count": len(values),
            "mean_expectation": mean(expectations) if values else "",
            "max_expectation": max(expectations) if values else "",
            "mean_confidence": mean(c for _, c in values) if values else "",
        })
    return rows


def decision_attribution(trial_rows: Iterable[Mapping[str, object]], config: EvaluationConfig):
    """Decision provenance over the final mixed-phase window shared with final accuracy."""
    seeds = set()
    counts_by_seed = {}
    window_totals = Counter()
    unavailable = set()
    for row in trial_rows:
        seed = int(row['seed'])
        seeds.add(seed)
        if row["phase"] != "mixed_contexts" or int(row["phase_trial"]) <= int(row["mixed_phase_trials"]) - config.final_window:
            continue
        window_totals[seed] += 1
        if row.get('decision_source') not in {'learned_rule', 'motor_babbling', 'unknown'}:
            unavailable.add(seed)
            continue
        category = decision_category(row)
        counts = counts_by_seed.setdefault((seed, category), Counter())
        counts['trials'] += 1
        counts['correct'] += truth(row['is_correct'])
    window = config.final_window
    valid = [seed for seed in sorted(seeds) if window_totals[seed] == window and seed not in unavailable]
    results = []
    for category in DECISION_CATEGORIES:
        counts = [counts_by_seed.get((seed, category), Counter()) for seed in valid]
        conditional = [c['correct']/c['trials'] for c in counts if c['trials']]
        results.append(dict(stage="final_mixed", window_trials=window, decision_category=category,
            trial_count=sum(c['trials'] for c in counts),
            trial_share=mean(c['trials']/window for c in counts) if valid and len(valid)==len(seeds) else '',
            correct_count=sum(c['correct'] for c in counts),
            conditional_accuracy=mean(conditional) if conditional else '',
            conditional_seed_count=len(conditional), valid_seed_count=len(valid), seed_count=len(seeds)))
    return results


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_percentage(value: object) -> str:
    if value == "":
        return "—"
    return f"{float(value) * 100:.1f}%"


def _write_markdown_report(
    path: Path,
    inventory_rows: Sequence[dict[str, object]],
    attribution_rows: Sequence[dict[str, object]],
    phase_end_trials: Sequence[int],
) -> None:
    inventory = {(str(r["rule_family"]), int(r["trial"])): r for r in inventory_rows}
    window = attribution_rows[0]["window_trials"]
    headers = " | ".join(f"Trial {t}" for t in phase_end_trials)
    lines = [
        "# RQ3 relational-encoding diagnostics",
        "",
        "## Retained implications at phase ends (seed mean count / mean max expectation)",
        "",
        f"| Implication family | {headers} |",
        "|---|" + "---:|" * len(phase_end_trials),
    ]
    for family in FAMILY_NAMES:
        cells = []
        for trial in phase_end_trials:
            row = inventory[(family, trial)]
            expectation = row["mean_max_expectation"]
            cells.append(f"{float(row['mean_count']):.2f} / " + ("—" if expectation == "" else f"{float(expectation):.3f}"))
        lines.append(f"| {family} | {' | '.join(cells)} |")
    lines += ["", f"## Decision attribution over the final {window} trials (trial share / conditional accuracy)", "",
              "| Decision category | Final window |", "|---|---:|"]
    for row in attribution_rows:
        lines.append(f"| {row['decision_category']} | {_format_percentage(row['trial_share'])} / {_format_percentage(row['conditional_accuracy'])} |")
    lines += ["", "Retained implications are available hypotheses in memory; decision attribution counts only the implication or "
              "motor-babbling source that actually selected the executed action. Full time series are in "
              "retained_rule_inventory.csv and the per-seed files in memory/.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_relational_diagnostics(
    run_dir: Path,
    diagnostics: Sequence[SeedDiagnostics],
    trial_rows: Sequence[dict[str, object]],
    *,
    config: EvaluationConfig,
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL,
) -> dict[str, Path]:
    """Check streamed per-seed memory files and write decisions and seed-mean summaries."""

    raw_decisions = [record for item in diagnostics for record in item.decisions]
    if len(raw_decisions) != len(trial_rows):
        raise ValueError(
            f"Raw decision diagnostics contain {len(raw_decisions)} records; "
            f"expected {len(trial_rows)}."
        )
    before_trials = int(trial_rows[0]["mixed_phase_start"]) - 1
    total_trials = before_trials + int(trial_rows[0]["mixed_phase_trials"])
    expected_snapshots = list(snapshot_trials(before_trials, total_trials, snapshot_interval))
    memory_dir = run_dir / MEMORY_DIRECTORY
    for item in diagnostics:
        if list(item.snapshot_trials) != expected_snapshots:
            raise ValueError(f"Seed {item.seed} snapshots do not match the expected trials.")
        if not all(path.is_file() for path in seed_memory_paths(memory_dir, item.seed)):
            raise ValueError(f"Seed {item.seed} memory files are missing.")

    decisions_path = run_dir / "decision_diagnostics.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for record in raw_decisions:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    # Stream seed files; truth summaries average only the seeds that retain the family.
    totals: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for item in diagnostics:
        _, stats_path = seed_memory_paths(memory_dir, item.seed)
        with gzip.open(stats_path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total = totals[int(row["trial"]), row["rule_family"]]
                total["seeds"] += 1
                total["count"] += int(row["count"])
                if int(row["count"]):
                    total["present"] += 1
                    total["max_expectation"] += float(row["max_expectation"])
                    total["mean_expectation"] += float(row["mean_expectation"])
                    total["mean_confidence"] += float(row["mean_confidence"])
    inventory_rows = []
    for (trial, family), total in sorted(totals.items()):
        present = total["present"]
        inventory_rows.append({
            "trial": trial, "rule_family": family, "seed_count": total["seeds"],
            "mean_count": total["count"] / total["seeds"], "retaining_seed_count": present,
            "mean_max_expectation": total["max_expectation"] / present if present else "",
            "mean_mean_expectation": total["mean_expectation"] / present if present else "",
            "mean_confidence": total["mean_confidence"] / present if present else "",
        })
    inventory_path = run_dir / "retained_rule_inventory.csv"
    _write_csv(inventory_path, tuple(inventory_rows[0]), inventory_rows)

    attribution_rows = decision_attribution(trial_rows, config)
    attribution_path = run_dir / "decision_attribution.csv"
    _write_csv(attribution_path, tuple(attribution_rows[0]), attribution_rows)
    report_path = run_dir / "relational_diagnostics.md"
    _write_markdown_report(report_path, inventory_rows, attribution_rows, (before_trials, total_trials))
    return {
        "decision_diagnostics": decisions_path,
        "memory_directory": memory_dir,
        "retained_rule_inventory": inventory_path,
        "decision_attribution": attribution_path,
        "relational_diagnostics_report": report_path,
    }
