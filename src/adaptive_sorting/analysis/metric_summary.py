"""Metric definitions, paired evaluation, and statistical table serialization."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_CRITERION_WINDOW, DEFAULT_FINAL_WINDOW,
    DEFAULT_CRITERION_THRESHOLD, EvaluationConfig, evaluate_seed,
)
from adaptive_sorting.analysis.statistics import (
    BOOTSTRAP_SAMPLES, bootstrap_interval, bootstrap_median_interval, wilson_interval,
)
from adaptive_sorting.analysis.report_validation import phase_plan, validate_job, validate_jobs
from adaptive_sorting.experiments.task_config import resolve_task_config

AgentTrials = list[tuple[str, Path]]


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    higher_is_better: bool | None = True
    percentage: bool = True
    binary: bool = False
    # Trials to criterion: median over runs, with runs that never reach it censored beyond the phase.
    censored_median: bool = False

    @property
    def direction(self):
        return 'neutral' if self.higher_is_better is None else 'higher' if self.higher_is_better else 'lower'

    @property
    def unit(self):
        return 'proportion' if self.percentage else 'trials' if self.censored_median else 'errors'

    @property
    def ci_method(self):
        return "wilson" if self.binary else "seed_bootstrap_median" if self.censored_median else "seed_bootstrap"


NEW_ERRORS = MetricDefinition("new_task_errors", "Cumulative errors: new-task inputs", False, False)
OLD_ERRORS = MetricDefinition("old_task_errors", "Cumulative errors: old-task inputs", False, False)
NEW_FINAL = MetricDefinition("final_new_task_accuracy", "Final accuracy: new-task inputs")
OLD_FINAL = MetricDefinition("final_old_task_accuracy", "Final accuracy: old-task inputs")
TRIALS_TO_CRITERION = MetricDefinition("trials_to_criterion", "Trials to criterion", False, False, censored_median=True)
SWITCH_METRICS = (NEW_ERRORS, OLD_ERRORS, NEW_FINAL, OLD_FINAL, TRIALS_TO_CRITERION)
METRICS_BY_RQ = {
    "rq1": (NEW_ERRORS, NEW_FINAL, TRIALS_TO_CRITERION),
    "rq2a": SWITCH_METRICS,
    "rq2b": SWITCH_METRICS,
    "rq3a": SWITCH_METRICS,
    "rq4a": (
        MetricDefinition("first_any_novel_accuracy", "First novel decision accuracy (before feedback)", binary=True),
        MetricDefinition("first_any_novel_rule_share", "Identity-matching implication use at first novel decision", higher_is_better=None, binary=True),
        *SWITCH_METRICS,
    ),
}
METRICS_BY_RQ["rq3b"] = METRICS_BY_RQ["rq3a"]
METRICS_BY_RQ["rq4b"] = METRICS_BY_RQ["rq4a"]


@dataclass(frozen=True)
class MetricSummary:
    agent: str
    scope: str
    metric: MetricDefinition
    average: float | None
    lower: float | None
    upper: float | None
    n_valid: int
    n_expected: int
    # Censored-median metrics: phase length; a median above it means fewer than half reached the criterion.
    horizon: int | None = None


def scope_of(record: dict) -> str:
    if record["rq"] == "rq2a":
        return f"change_size={record['change_size']}"
    if record["rq"] == "rq2b":
        return f"phase={record['phase_index']}:{record['phase_label']}"
    return "evaluation_phase"


def iter_trial_jobs(path):
    """Stream contiguous seed/condition jobs without loading multi-million-row runs."""
    seen = set()
    active = None
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"seed", "trial", "sphere_color", "is_correct"} <= set(reader.fieldnames or ()):
            raise ValueError(f"Missing trial fields in {path}")
        for row in reader:
            key = (int(row["seed"]), int(row.get("change_size", 0)))
            if key != active:
                if active is not None:
                    yield active, rows
                    seen.add(active)
                if key in seen:
                    raise ValueError("Trial jobs must be contiguous within the CSV.")
                active, rows = key, []
            rows.append(row)
    if active is None:
        raise ValueError(f"No trials in {path}")
    yield active, rows


def _input_signature(rows, context_control=False):
    fields = ("trial", "phase", "phase_index", "phase_trial", "sphere_color", "context", "is_changed", "is_novel", "correct_action")
    if not context_control:
        fields += ("light_1", "light_2")
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r:int(r["trial"])):
        digest.update(json.dumps([str(row.get(k, "")) for k in fields], separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.digest()


def evaluate_agents(rq, agent_trials, config, *, context_control=False, require_complete_plan=False):
    """Check task alignment and retain seed IDs, conditions and recurring phases."""
    if context_control and len(agent_trials) != 2:
        raise ValueError("A context comparison requires one L2 run and one no-L2 run.")
    per_agent = {}
    signatures = {}
    control_identity = None
    if len({label for label, _ in agent_trials}) != len(agent_trials):
        raise ValueError("Agent labels must be unique.")
    for agent_index, (label, path) in enumerate(agent_trials):
        meta_path = path.with_name("metadata.json")
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        parameters = metadata.get("parameters", {})
        if context_control and metadata:
            identity = (metadata.get("agent"), metadata.get("binary_sha256"), metadata.get("encoding_version"), *(parameters.get(k) for k in ("ona_cycles", "epsilon", "learning_rate", "ucb1_exploration", "ucb_window", "ucb_exploration")))
            if control_identity is not None and identity != control_identity:
                raise ValueError("Context controls require the same agent and controller configuration.")
            control_identity = identity
        task_config = resolve_task_config(path.parent, metadata)
        records = []
        jobs = []
        plan = phase_plan(rq, metadata)
        if require_complete_plan and (plan is None or not metadata.get("seeds", parameters.get("seeds"))
                                      or (rq == "rq2a" and not metadata.get("change_sizes", parameters.get("change_sizes")))):
            raise ValueError(f"Complete experiment plan unavailable for {label}.")
        for (seed, condition), rows in iter_trial_jobs(path):
            jobs.append((seed, condition))
            validate_job(rq, rows, plan)
            if context_control:
                has_l2 = any(r.get("light_2") in {"on", "off"} for r in rows)
                if has_l2 != (agent_index == 0):
                    raise ValueError("Expected an L2 run followed by a no-L2 control.")
            key = (seed, condition)
            signature = _input_signature(rows, context_control)
            if key in signatures and signatures[key] != signature:
                raise ValueError(f"Unpaired input sequence for seed {seed}, condition {condition}.")
            signatures[key] = signature
            records.extend(evaluate_seed(rq, rows, config, task_config))
        planned_seeds = metadata.get("seeds", parameters.get("seeds"))
        if planned_seeds is not None and {int(r["seed"]) for r in records} != set(planned_seeds):
            raise ValueError(f"Recorded seeds disagree with metadata for {label}.")
        validate_jobs(rq, jobs, metadata)
        per_agent[label] = records
    return per_agent


def supporting_definitions(rq, records):
    if rq in {"rq3a", "rq3b"}:
        keys = [k for k in records[0] if k.startswith("rule_") and k.endswith("_share")]
    elif rq in {"rq4a", "rq4b"}:
        keys = [k for k in records[0] if k.startswith("first_") and (k.endswith("_accuracy") or k.endswith("_share")) and k not in {d.key for d in METRICS_BY_RQ[rq]}]
        colors = [k[6:-2] for k in records[0] if k.startswith('first_') and k.endswith('_n')
                  and k not in {'first_novel_n', 'first_any_novel_n'}]
        keys += [f'final_group_{c}_accuracy' for c in sorted(colors)]
    else:
        keys = []
    return tuple(MetricDefinition(k, k.replace("_", " "),
                 higher_is_better=None if k.endswith("_share") else True,
                 binary=k.startswith('first_') and not k.startswith('first_novel_')) for k in keys)


def _estimate(definition, selected, n_expected, samples):
    """Return (average, lower, upper, n_valid, horizon) for one agent, scope and metric."""
    if definition.censored_median:
        if len(selected) != n_expected:
            return None, None, None, len(selected), None
        horizons = {int(r["criterion_horizon"]) for r in selected}
        if len(horizons) != 1:
            raise ValueError("Attainment horizons must agree within a scope.")
        horizon = horizons.pop()
        # A run that never reaches the criterion ranks after every run that does; only whether the median exceeds
        # the horizon is interpreted for such runs, never a numerical time.
        times = [int(r["trials_to_criterion"]) if r["criterion_reached"] in (True, "True") else horizon + 1 for r in selected]
        reached = sum(r["criterion_reached"] in (True, "True") for r in selected)
        return (*bootstrap_median_interval(times, samples), reached, horizon)
    values = [r[definition.key] for r in selected if r.get(definition.key) is not None]
    if len(values) != n_expected:
        return None, None, None, len(values), None
    estimate = wilson_interval(values) if definition.binary else bootstrap_interval(values, samples)
    return (*estimate, len(values), None)


def summarize_records(rq, per_agent, *, supporting=False, samples=BOOTSTRAP_SAMPLES):
    summaries = []
    scopes = sorted({scope_of(r) for records in per_agent.values() for r in records if rq != "rq2b" or r["phase_index"] > 1})
    for scope in scopes:
        expected = {int(r["seed"]) for records in per_agent.values() for r in records if scope_of(r) == scope}
        for label, records in per_agent.items():
            selected = [r for r in records if scope_of(r) == scope]
            definitions = supporting_definitions(rq, records) if supporting else METRICS_BY_RQ[rq]
            for definition in definitions:
                average, lower, upper, n_valid, horizon = _estimate(definition, selected, len(expected), samples)
                summaries.append(MetricSummary(label, scope, definition, average, lower, upper, n_valid, len(expected), horizon))
    return summaries


def summarize_agents(rq, agent_trials, *, final_window=DEFAULT_FINAL_WINDOW, criterion_window=DEFAULT_CRITERION_WINDOW, criterion_threshold=DEFAULT_CRITERION_THRESHOLD, samples=BOOTSTRAP_SAMPLES):
    config = EvaluationConfig(final_window, criterion_window, criterion_threshold)
    per_agent = evaluate_agents(rq, agent_trials, config)
    return summarize_records(rq, per_agent, samples=samples), per_agent


def paired_comparisons(rq, per_agent, samples=BOOTSTRAP_SAMPLES):
    results = []
    for left, right in combinations(per_agent, 2):
        left_rows = {(scope_of(r), int(r["seed"])): r for r in per_agent[left]}
        right_rows = {(scope_of(r), int(r["seed"])): r for r in per_agent[right]}
        for scope in sorted({s for s, _ in left_rows.keys() | right_rows.keys()}):
            if rq == "rq2b" and scope.startswith("phase=1:"):
                continue
            seeds = {seed for s, seed in left_rows.keys() | right_rows.keys() if s == scope}
            # Censored trials to criterion have no per-seed difference; they are compared as medians.
            for definition in (d for d in METRICS_BY_RQ[rq] if not d.censored_median):
                diffs = []
                missing = []
                for seed in sorted(seeds):
                    a = left_rows.get((scope, seed), {}).get(definition.key)
                    b = right_rows.get((scope, seed), {}).get(definition.key)
                    if a is None or b is None:
                        missing.append(seed)
                    else:
                        diffs.append((float(a) - float(b)) * (100 if definition.percentage else 1))
                average, lower, upper = bootstrap_interval(diffs, samples)
                results.append({"left": left, "right": right, "scope": scope, "metric": definition.key, "difference": "left-minus-right", "mean_difference": average, "ci95_lower": lower, "ci95_upper": upper, "n_pairs": len(diffs), "n_expected": len(seeds), "missing_seeds": ";".join(map(str, missing)), "complete_pairs_only": bool(missing), "ci_method": "seed_bootstrap", "unit": "percentage_points" if definition.percentage else definition.unit, "direction": definition.direction})
    return results


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summaries):
    write_rows(path, [{"agent": s.agent, "scope": s.scope, "metric": s.metric.key, "estimate": "median" if s.metric.censored_median else "mean", "value": s.average, "ci95_lower": s.lower, "ci95_upper": s.upper, "n_valid": s.n_valid, "n_expected": s.n_expected, "horizon": s.horizon, "ci_method": s.metric.ci_method, "direction": s.metric.direction, "unit": s.metric.unit} for s in summaries])


