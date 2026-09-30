"""RQ4 rule classification and diagnostic reports, separate from execution."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean
from typing import Protocol, Sequence

from adaptive_sorting.analysis.evaluation import first_novel_metrics


class SeedDiagnostics(Protocol):
    seed: int
    decisions: list[dict[str, object]]
    snapshots: dict[str, tuple[str, ...]]


VARIABLE_PATTERN = re.compile(r"[#\$]\d+")
SPHERE_COLOR_VARIABLE_PATTERN = re.compile(
    r"\(sphere \* (?P<color>[#\$]\d+)\) --> is"
)
BIN_COLOR_VARIABLE_PATTERN = re.compile(
    r"\((?P<bin>[#\$]\d+) \* (?P<color>[#\$]\d+)\) --> is"
)
OPERATION_BIN_VARIABLE_PATTERN = re.compile(
    r"\(\{SELF\} \* (?P<bin>[#\$]\d+)\) --> \^place"
)
COLOR_ADDRESSED_OPERATION_PATTERN = re.compile(
    r"\(\{SELF\} \* \(bin \* (?P<color>[#\$]\d+)\)\) --> \^place"
)


def classify_generalization_rule(rule: str | None) -> str:
    """Classify whether a decision rule contains relational variables."""

    if not rule or not VARIABLE_PATTERN.search(rule):
        return "grounded"
    sphere_match = SPHERE_COLOR_VARIABLE_PATTERN.search(rule)
    color_addressed_match = COLOR_ADDRESSED_OPERATION_PATTERN.search(rule)
    if (
        sphere_match
        and color_addressed_match
        and sphere_match.group("color") == color_addressed_match.group("color")
    ):
        return "target_match_generalization"
    operation_match = OPERATION_BIN_VARIABLE_PATTERN.search(rule)
    if sphere_match and operation_match:
        target_pair = (
            operation_match.group("bin"),
            sphere_match.group("color"),
        )
        bin_pairs = {
            (match.group("bin"), match.group("color"))
            for match in BIN_COLOR_VARIABLE_PATTERN.finditer(rule)
        }
        if target_pair in bin_pairs:
            return "target_match_generalization"
    return "other_generalized"


def _write_rq4_markdown_report(
    path: Path,
    diagnostics: Sequence[SeedDiagnostics],
    novel_colors: set[str],
) -> None:
    """Write the compact rule-retention and decision-attribution audit."""

    rule_classes = ("grounded", "target_match_generalization", "other_generalized")
    stages = ("known_only", "expanded_known", "expanded_novel")
    decisions_by_stage: dict[str, list[dict[str, object]]] = {
        stage: [] for stage in stages
    }
    for diagnostic in diagnostics:
        for row in diagnostic.decisions:
            if row["phase"] == "known_only":
                stage = "known_only"
            elif bool(row["is_novel"]):
                stage = "expanded_novel"
            else:
                stage = "expanded_known"
            decisions_by_stage[stage].append(row)

    categories = (
        "target_match_generalization",
        "grounded",
        "other_generalized",
        "motor_babbling",
        "unknown",
    )
    category_labels = {
        "target_match_generalization": "identity-matching implication",
        "grounded": "color-specific implication",
        "other_generalized": "other variable implication",
        "motor_babbling": "motor babbling",
        "unknown": "unknown",
    }

    lines = [
        "# RQ4 ONA (relational) diagnostics",
        "",
        "## Mean retained implications per seed",
        "",
        "| Implication class | End known-only | End expanded |",
        "|---|---:|---:|",
    ]
    for rule_class in rule_classes:
        phase_means = []
        for phase in ("known_only", "expanded_inputs"):
            counts = [
                sum(
                    classify_generalization_rule(rule) == rule_class
                    for rule in diagnostic.snapshots.get(phase, ())
                )
                for diagnostic in diagnostics
            ]
            phase_means.append(mean(counts) if counts else 0.0)
        lines.append(
            f"| {category_labels[rule_class]} | "
            f"{phase_means[0]:.2f} | {phase_means[1]:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Full-phase decision attribution (trial share / conditional accuracy)",
            "",
            "These diagnostics cover entire phases, not the early or final evaluation windows.",
            "",
            "| Decision category | Known-only | Expanded known | Expanded novel |",
            "|---|---:|---:|---:|",
        ]
    )
    for category in categories:
        cells = []
        for stage in stages:
            stage_rows = decisions_by_stage[stage]
            selected = [
                row
                for row in stage_rows
                if (
                    str(row["driving_rule_class"])
                    if row["decision_source"] == "learned_rule"
                    else str(row["decision_source"])
                )
                == category
            ]
            shares = []
            accuracies = []
            for diagnostic in diagnostics:
                seed_rows = [r for r in stage_rows if r["seed"] == diagnostic.seed]
                chosen = [r for r in selected if r["seed"] == diagnostic.seed]
                if seed_rows:
                    shares.append(len(chosen) / len(seed_rows))
                if chosen:
                    accuracies.append(mean(bool(r["is_correct"]) for r in chosen))
            share_text = f"{mean(shares):.1%}" if len(shares) == len(diagnostics) else "NA"
            accuracy_text = f"{mean(accuracies):.1%} (n={len(accuracies)})" if accuracies else "NA"
            cells.append(f"{share_text} / {accuracy_text}")
        lines.append(f"| {category_labels[category]} | {' | '.join(cells)} |")

    first_metrics = [first_novel_metrics(d.decisions, novel_colors) for d in diagnostics]
    lines.extend([
        "", "## First novel decisions", "",
        "Per-color decisions precede feedback on that color, but may follow feedback on the other novel color. First-any precedes all novel feedback. Values below are seed means; combined gives equal weight to both colors within each seed.", "",
        "| Scope | Correct | Target rule driven | Correct and target-rule driven |",
        "|---|---:|---:|---:|",
    ])
    for label, scope in [*((color, f"first_{color}") for color in sorted(novel_colors)), ("combined", "first_novel"), ("first-any", "first_any_novel")]:
        cells = []
        for suffix in ("accuracy", "rule_share", "correct_rule_share"):
            values = [m[f"{scope}_{suffix}"] for m in first_metrics]
            if any(v is None for v in values):
                cells.append("NA")
            else:
                total = sum(values)
                cells.append(f"{total:g}/{len(values)} ({mean(values):.1%})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    target_retained = sum(
        any(
            classify_generalization_rule(rule) == "target_match_generalization"
            for rule in diagnostic.snapshots.get("known_only", ())
        )
        for diagnostic in diagnostics
    )
    lines.extend(
        [
            "",
            "## Seed-level mechanism checks",
            "",
            f"- Identity-matching implication retained before expansion: "
            f"{target_retained}/{len(diagnostics)} seeds.",
            "",
            "Retained implications are available hypotheses in memory; decision attribution "
            "counts only the source that actually selected the executed action.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rq4_diagnostics(
    run_dir: Path,
    diagnostics: Sequence[SeedDiagnostics],
    novel_colors: set[str],
) -> dict[str, Path]:
    """Persist decision provenance and retained-rule snapshots for inspection."""

    decision_path = run_dir / "relational_decisions.jsonl"
    decision_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for diagnostic in diagnostics
            for row in diagnostic.decisions
        ),
        encoding="utf-8",
    )
    snapshot_path = run_dir / "relational_rule_snapshots.jsonl"
    snapshot_path.write_text(
        "".join(
            json.dumps(
                {
                    "seed": diagnostic.seed,
                    "phase": phase,
                    "rule": rule,
                    "rule_class": classify_generalization_rule(rule),
                },
                sort_keys=True,
            )
            + "\n"
            for diagnostic in diagnostics
            for phase, rules in diagnostic.snapshots.items()
            for rule in rules
        ),
        encoding="utf-8",
    )
    summary: list[dict[str, object]] = []
    for diagnostic in diagnostics:
        decision_classes = Counter(
            str(row["driving_rule_class"]) for row in diagnostic.decisions
        )
        novel_decision_classes = Counter(
            str(row["driving_rule_class"])
            for row in diagnostic.decisions
            if bool(row["is_novel"])
        )
        retained_by_phase = {
            phase: dict(Counter(classify_generalization_rule(rule) for rule in rules))
            for phase, rules in diagnostic.snapshots.items()
        }
        summary.append(
            {
                "seed": diagnostic.seed,
                "decision_rule_classes": dict(decision_classes),
                "novel_decision_rule_classes": dict(novel_decision_classes),
                "retained_rule_classes_by_phase": retained_by_phase,
            }
        )
    summary_path = run_dir / "relational_diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = run_dir / "relational_diagnostics.md"
    _write_rq4_markdown_report(report_path, diagnostics, novel_colors)
    return {
        "decisions": decision_path,
        "rule_snapshots": snapshot_path,
        "summary": summary_path,
        "report": report_path,
    }


