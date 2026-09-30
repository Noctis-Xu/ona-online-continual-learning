"""Shared trial-level evaluation for runners and offline reports."""
from __future__ import annotations

from collections import (
    defaultdict,
    deque,
)
from dataclasses import (
    asdict,
    dataclass,
)
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import (
    Callable,
    Mapping,
    Sequence,
    TypeVar,
)

import numpy as np

from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    load_task_rules,
)

METRICS_VERSION = "evaluation-v4"
DEFAULT_FINAL_WINDOW = 300
DEFAULT_ROLLING_WINDOW = 50
DEFAULT_CRITERION_WINDOW = 20
DEFAULT_CRITERION_THRESHOLD = 0.9
Row = Mapping[str, object]


@dataclass(frozen=True)
class EvaluationConfig:
    final_window: int = DEFAULT_FINAL_WINDOW
    criterion_window: int = DEFAULT_CRITERION_WINDOW
    criterion_threshold: float = DEFAULT_CRITERION_THRESHOLD

    def __post_init__(self):
        if min(self.final_window, self.criterion_window) <= 0:
            raise ValueError("Evaluation windows must be positive.")
        if not 0 <= self.criterion_threshold <= 1:
            raise ValueError("Criterion threshold must be between zero and one.")

    def metadata(self) -> dict[str, object]:
        return {"metrics_version": METRICS_VERSION, **asdict(self)}


def truth(value: object) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"Expected a boolean trial field, got {value!r}.")


def accuracy(rows: Sequence[Row]) -> float | None:
    return mean(truth(r["is_correct"]) for r in rows) if rows else None


def trials_to_criterion(rows: Sequence[Row], start_trial: int, window: int, threshold: float) -> int | None:
    """Trials since the phase start until the last `window` given rows first reach `threshold`.

    Callers pass only new-task observations, so old-task trials never enter the window.
    """
    if window <= 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid learning criterion.")
    recent: deque[bool] = deque(maxlen=window)
    for row in rows:
        recent.append(truth(row["is_correct"]))
        if len(recent) == window and sum(recent) / window >= threshold:
            return int(row["trial"]) - start_trial + 1
    return None


def decision_category(row: Row) -> str:
    source = str(row.get("decision_source") or "unavailable")
    if source != "learned_rule":
        return source
    family = str(row.get("driving_rule_family") or "unclassified")
    factors = set(family.split("+"))
    if family == "unclassified":
        return "unclassified"
    if {"color", "L1"} <= factors:
        return "relevant_with_l2" if "L2" in factors else "relevant_only"
    return "missing_required"


def first_novel_metrics(rows: Sequence[Row], novel_colors: set[str]) -> dict[str, object]:
    first: dict[str, Row] = {}
    for row in rows:
        if str(row["sphere_color"]) in novel_colors:
            first.setdefault(str(row["sphere_color"]), row)
    metrics: dict[str, object] = {}
    scopes: dict[str, list[Row]] = {f"first_{c}": [first[c]] if c in first else [] for c in sorted(novel_colors)}
    scopes["first_novel"] = list(first.values()) if set(first) == novel_colors else []
    scopes["first_any_novel"] = list(first.values())[:1]
    for scope, selected in scopes.items():
        metrics[f"{scope}_n"] = len(selected)
        metrics[f"{scope}_accuracy"] = accuracy(selected)
        available = bool(selected) and all(r.get("decision_source") in {"learned_rule", "motor_babbling"} and (r.get("decision_source") != "learned_rule" or r.get("driving_rule_class")) for r in selected)
        driven = [r.get("decision_source") == "learned_rule" and r.get("driving_rule_class") == "target_match_generalization" for r in selected]
        metrics[f"{scope}_rule_share"] = mean(driven) if available else None
        metrics[f"{scope}_correct_rule_share"] = mean(d and truth(r["is_correct"]) for d, r in zip(driven, selected)) if available else None
        if len(selected) == 1:
            metrics[f"{scope}_trial"] = int(selected[0]["trial"])
            metrics[f"{scope}_prior_novel_feedback"] = sum(int(r["trial"]) < int(selected[0]["trial"]) and str(r["sphere_color"]) in novel_colors for r in rows)
    return metrics


@lru_cache(maxsize=16)
def cached_task_rules(path: str, mtime_ns: int):
    """Parse each task configuration once per report; the modification time invalidates stale entries."""
    return load_task_rules(path)


def evaluate_seed(
    rq: str,
    rows: Sequence[Row],
    config: EvaluationConfig = EvaluationConfig(),
    task_config: Path = DEFAULT_TASK_RULES,
) -> list[dict[str, object]]:
    """One record per seed/condition/phase; never average recurring phases here."""
    if not rows:
        raise ValueError("Cannot evaluate an empty run.")
    seeds = {int(r["seed"]) for r in rows}
    if len(seeds) != 1:
        raise ValueError("evaluate_seed requires exactly one seed.")
    rules = cached_task_rules(str(Path(task_config).resolve()), Path(task_config).stat().st_mtime_ns)
    conditions: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        conditions[int(row.get("change_size", 0))].append(row)
    records = []
    for change_size, condition in sorted(conditions.items()):
        ordered = sorted(condition, key=lambda r: int(r["trial"]))
        numbers = [int(r["trial"]) for r in ordered]
        if numbers != list(range(1, len(ordered) + 1)):
            raise ValueError("Trial numbers must be unique and continuous within a seed/condition.")
        if rq == "rq1":
            phases = {1: ordered}
        elif rq == "rq2b":
            phases = defaultdict(list)
            for row in ordered:
                phases[int(row["phase_index"])].append(row)
            if sorted(phases) != list(range(1, len(phases) + 1)):
                raise ValueError("Recurring phase indices must be continuous from one.")
        else:
            name = {"rq2a": "after_update", "rq3a": "mixed_contexts", "rq3b": "mixed_contexts", "rq4a": "expanded_inputs", "rq4b": "expanded_inputs"}[rq]
            phases = {2: [r for r in ordered if r["phase"] == name]}
        for phase_index, phase in sorted(phases.items()):
            if not phase:
                raise ValueError("Missing evaluation phase.")
            initial = rq == "rq1" or (rq == "rq2b" and phase_index == 1)
            color_rule = "initial"
            if rq in {"rq3a", "rq3b"}:
                color_rule = "context_light_off"
            elif rq in {"rq4a", "rq4b"}:
                color_rule = "color_target_expanded" if rq == "rq4b" else "expanded_inputs"
            colors = set(rules[color_rule].mapping)
            novel = colors - set(rules["color_target_base" if rq == "rq4b" else "novel_input_base"].mapping) if rq in {"rq4a", "rq4b"} else set()
            # New-task inputs: the correct action changed at this switch; all other inputs are old-task.
            if initial:
                new_task = lambda r: True
            elif rq in {"rq2a", "rq2b"}:
                new_task = lambda r: truth(r["is_changed"])
            elif rq in {"rq3a", "rq3b"}:
                new_task = lambda r: r["context"] == "light_on"
            else:
                new_task = lambda r: str(r["sphere_color"]) in novel
            # Fixed global windows must fit; a short demo gets NA, never a silently shorter window.
            final = phase[-config.final_window:] if len(phase) >= config.final_window else []
            new_rows = [r for r in phase if new_task(r)]
            old_rows = [r for r in phase if not new_task(r)]
            final_new = [r for r in final if new_task(r)]
            final_old = [r for r in final if not new_task(r)]
            errors = lambda selected: sum(not truth(r["is_correct"]) for r in selected) if selected else None
            key = lambda r: f"{r['context']}:{r['sphere_color']}" if rq in {"rq3a", "rq3b"} else str(r["sphere_color"])
            expected = {f"{context}:{c}" for context in ("light_on", "light_off") for c in colors} if rq in {"rq3a", "rq3b"} else colors
            by_group = {g: [r for r in final if key(r) == g] for g in sorted(expected)}
            start_trial = int(phase[0]["trial"])
            hit = trials_to_criterion(new_rows, start_trial, config.criterion_window, config.criterion_threshold)
            record: dict[str, object] = {
                **config.metadata(), "rq": rq, "seed": next(iter(seeds)), "change_size": change_size,
                "phase_index": phase_index, "phase_label": phase[0].get("phase_label", phase[0].get("phase", "initial")),
                "phase_trials": len(phase), "phase_start": start_trial,
                "phase_errors": sum(not truth(r["is_correct"]) for r in phase),
                "new_task_errors": errors(new_rows), "old_task_errors": errors(old_rows),
                "new_task_n": len(new_rows), "old_task_n": len(old_rows),
                "final_overall_accuracy": accuracy(final),
                "final_new_task_accuracy": accuracy(final_new), "final_old_task_accuracy": accuracy(final_old),
                "final_new_task_n": len(final_new), "final_old_task_n": len(final_old),
                "final_window_complete": bool(final),
                "missing_final_groups": ";".join(g for g, samples in by_group.items() if not samples),
                "trials_to_criterion": hit, "criterion_reached": hit is not None, "criterion_horizon": len(phase),
            }
            for group, samples in by_group.items():
                record[f"final_group_{group}_n"] = len(samples)
                record[f"final_group_{group}_accuracy"] = accuracy(samples)
            if rq in {"rq3a", "rq3b"}:
                categories = ("relevant_only", "relevant_with_l2", "missing_required", "unclassified", "motor_babbling", "unknown")
                available = bool(final) and all(r.get("decision_source") in {"learned_rule", "motor_babbling", "unknown"} for r in final)
                for category in categories:
                    record[f"rule_{category}_share"] = mean(decision_category(r) == category for r in final) if available else None
            if rq in {"rq4a", "rq4b"}:
                record.update(first_novel_metrics(phase, novel))
            records.append(record)
    return records


T = TypeVar("T")


def rolling_accuracy(records: Sequence[T], window: int, *, correct: Callable[[T], bool], include: Callable[[T], bool] = lambda r: True) -> list[float]:
    """Trailing global-trial windows; continuous across phases, using available observations at run start."""
    if window <= 0:
        raise ValueError("Rolling window must be positive.")
    selected = np.fromiter((bool(include(r)) for r in records), dtype=bool, count=len(records))
    # correct() is only evaluated for selected records, as filters may exclude rows it cannot score.
    hits = np.fromiter((s and bool(correct(r)) for s, r in zip(selected, records)), dtype=bool, count=len(records))
    def window_sums(flags):
        totals = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
        ends = np.arange(1, len(flags) + 1)
        return totals[ends] - totals[np.maximum(ends - window, 0)]
    counts, correct_counts = window_sums(selected), window_sums(hits)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, correct_counts / counts, np.nan).tolist()
