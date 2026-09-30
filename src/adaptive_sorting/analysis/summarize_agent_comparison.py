"""Render metric tables and expose the standalone comparison CLI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from adaptive_sorting.analysis.evaluation import (
    DEFAULT_CRITERION_WINDOW, DEFAULT_FINAL_WINDOW,
    DEFAULT_CRITERION_THRESHOLD, EvaluationConfig, METRICS_VERSION,
)
from adaptive_sorting.analysis.statistics import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
from adaptive_sorting.analysis.plot_style import plt, TITLES, finish_figure
from adaptive_sorting.naming import EXPERIMENT_IDS
# Re-export the metric API used by the experiment runners and reports.
from adaptive_sorting.analysis.metric_summary import (
    AgentTrials, MetricDefinition, MetricSummary, METRICS_BY_RQ,
    NEW_ERRORS, OLD_ERRORS, NEW_FINAL, OLD_FINAL, TRIALS_TO_CRITERION, scope_of, iter_trial_jobs,
    evaluate_agents, supporting_definitions, summarize_records, summarize_agents,
    paired_comparisons, write_rows, write_summary_csv,
)


def criterion_cell(summary):
    """Median trials to criterion; reached count only when some runs miss it; '>horizon' if most do."""
    if summary.average is None:
        return "NA"
    median = f">{summary.horizon}" if summary.average > summary.horizon else f"{summary.average:.0f}"
    return median if summary.n_valid == summary.n_expected else f"{median} ({summary.n_valid}/{summary.n_expected})"


def format_metric(summary):
    if summary.metric.censored_median:
        value = criterion_cell(summary)
        if summary.lower is not None and summary.upper <= summary.horizon:
            value += f" [{summary.lower:.0f}, {summary.upper:.0f}]"
        return value
    if summary.average is None:
        return f"NA ({summary.n_valid}/{summary.n_expected})"
    factor = 100 if summary.metric.percentage else 1
    suffix = "%" if summary.metric.percentage else ""
    value = f"{summary.average * factor:.2f}{suffix}"
    if summary.lower is not None:
        value += f" [{summary.lower * factor:.2f}, {summary.upper * factor:.2f}]"
    return value + f" (n={summary.n_valid})"


def write_markdown_summary(path, rq, agent_trials, summaries, **settings):
    lines = [f"# {TITLES[rq]} evaluation", "", f"Metrics: {METRICS_VERSION}. Cumulative errors cover the complete evaluated phase for new-task and old-task inputs. Final window: {settings.get('final_window', DEFAULT_FINAL_WINDOW)} global trials, sliced before filtering groups. Trials to criterion: global trials from the switch until {settings.get('criterion_threshold', DEFAULT_CRITERION_THRESHOLD):.0%} accuracy over the last {settings.get('criterion_window', DEFAULT_CRITERION_WINDOW)} new-task observations; median over runs, runs that never reach the criterion count as exceeding the phase, and the count of runs reaching it is shown when some do not.", "", "Values are seed means except medians of trials to criterion. Single binary seed outcomes use 95% Wilson intervals; other means, medians and paired differences use seed percentile bootstrap intervals. NA includes valid/expected counts; incomplete groups are not silently dropped. Paired effects are in paired_differences.csv (left minus right); proportion differences use percentage points. Descriptive rule shares have neutral direction, not a performance ranking. Recurring phases remain separate; they are not independent replicates.", ""]
    labels = [label for label, _ in agent_trials]
    for scope in dict.fromkeys(s.scope for s in summaries):
        lines += [f"## {scope}", "", "| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        selected = [s for s in summaries if s.scope == scope]
        for key in dict.fromkeys(s.metric.key for s in selected):
            cells = {s.agent: s for s in selected if s.metric.key == key}
            lines.append("| " + next(iter(cells.values())).metric.label + " | " + " | ".join(format_metric(cells[label]) for label in labels) + " |")
        lines.append("")
    if rq in {"rq4a", "rq4b"}:
        lines += ["Per-color first encounters can follow feedback on the other novel color. first_any_novel records the single decision before any novel-color feedback; per-color prior feedback counts are in per_seed_metrics.csv. Rule attribution is NA when provenance is unavailable.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_title(rq, definition):
    new = {'rq1': 'all inputs', 'rq3a': 'L1 on', 'rq3b': 'L1 on', 'rq4a': 'new colors', 'rq4b': 'new colors'}.get(rq, 'changed colors')
    old = {'rq3a': 'L1 off', 'rq3b': 'L1 off', 'rq4a': 'old colors', 'rq4b': 'old colors'}.get(rq, 'unchanged colors')
    return {
        'new_task_errors': f'Cumulative errors: new-task inputs ({new})',
        'old_task_errors': f'Cumulative errors: old-task inputs ({old})',
        'final_new_task_accuracy': f'Final accuracy: new-task inputs ({new})',
        'final_old_task_accuracy': f'Final accuracy: old-task inputs ({old})',
        'trials_to_criterion': f'Trials to criterion: new-task inputs ({new})',
    }.get(definition.key, definition.label)


def scope_title(scope):
    if scope.startswith('change_size='):
        return scope.split('=')[1] + ' changed colors'
    if scope.startswith('phase='):
        phase, label = scope.split('=')[1].split(':', 1)
        return f'Switch {int(phase)-1} ({label})'
    return 'Evaluation phase'


def plot_summary(path, rq, agent_trials, summaries):
    scopes = list(dict.fromkeys(s.scope for s in summaries))
    definitions = METRICS_BY_RQ[rq]
    labels = [label for label, _ in agent_trials]
    figure, axes = plt.subplots(len(definitions), 1,
        figsize=(max(9, 3 + 1.35 * len(scopes)), 2.1 * len(definitions)),
        squeeze=False, layout='constrained')
    markdown = [f'# {TITLES[rq]} core metrics', '',
                'Seed means; intervals and sample counts are in core_metrics.csv and supporting_metrics.csv.', '']
    for axis, definition in zip(axes.flat, definitions):
        title = metric_title(rq, definition)
        columns = [(scope_title(scope), scope, definition.key) for scope in scopes]
        if definition.key == 'final_new_task_accuracy' and rq in {'rq4a', 'rq4b'}:
            colors = sorted({s.metric.key for s in summaries if s.metric.key.startswith('final_group_')})
            columns = [('All new colors', scopes[0], definition.key)] + [
                (key.removeprefix('final_group_').removesuffix('_accuracy'), scopes[0], key) for key in colors]
        headers = ['Agent'] + [name for name, _, _ in columns]
        cells = []
        for label in labels:
            row = [label]
            for _, scope, key in columns:
                result = next(s for s in summaries if s.scope == scope and s.agent == label and s.metric.key == key)
                row.append(criterion_cell(result) if definition.censored_median else
                           'NA' if result.average is None else
                           f'{result.average * 100:.1f}%' if definition.percentage else f'{result.average:.1f}')
            cells.append(row)
        axis.axis('off')
        axis.set_title(title, loc='left', fontsize=11)
        widths = [2.5] + [1] * len(columns)
        table = axis.table(cellText=cells, colLabels=headers, cellLoc='center',
                           colWidths=[w/sum(widths) for w in widths], bbox=[0, 0, 1, .9])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#dddddd')
            cell.set_linewidth(.4)
            if row == 0:
                cell.set_facecolor('#eeeeee')
                cell.set_text_props(weight='bold')
            elif col == 0:
                cell.set_text_props(ha='left')
        markdown += [f'## {title}', '', '| ' + ' | '.join(headers) + ' |',
                     '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        markdown += ['| ' + ' | '.join(row) + ' |' for row in cells] + ['']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.stem + '_table.md').write_text('\n'.join(markdown))
    finish_figure(figure, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq", choices=EXPERIMENT_IDS, required=True)
    parser.add_argument("--agent-trials", action="append", nargs=2, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-control", action="append", nargs=2, default=[], metavar=("LABEL", "RQ3_TRIALS"), help="Pair an RQ3b agent with its matching no-L2 run.")
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--criterion-window", type=int, default=DEFAULT_CRITERION_WINDOW)
    parser.add_argument("--criterion-threshold", type=float, default=DEFAULT_CRITERION_THRESHOLD)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("bootstrap-samples must be at least 100")
    paths = [(label, Path(path)) for label, path in args.agent_trials]
    config = EvaluationConfig(args.final_window, args.criterion_window, args.criterion_threshold)
    records = evaluate_agents(args.rq, paths, config)
    summaries = summarize_records(args.rq, records, samples=args.bootstrap_samples)
    supporting = summarize_records(args.rq, records, supporting=True, samples=args.bootstrap_samples)
    control_effects = []
    if args.context_control and args.rq != "rq3b":
        parser.error("context-control applies only to rq3b")
    for label, path in args.context_control:
        if label not in dict(paths):
            parser.error(f"Unknown agent label for context control: {label}")
        matched = evaluate_agents("rq3b", [(label + " with L2", dict(paths)[label]), (label + " without L2", Path(path))], config, context_control=True)
        control_effects.extend(paired_comparisons("rq3b", matched, args.bootstrap_samples))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "per_seed_metrics.csv", [{"agent": label, **r} for label, rows in records.items() for r in rows])
    write_summary_csv(output / "core_metrics.csv", summaries)
    write_summary_csv(output / "supporting_metrics.csv", supporting)
    write_rows(output / "paired_differences.csv", paired_comparisons(args.rq, records, args.bootstrap_samples))
    if control_effects:
        write_rows(output / "context_control_differences.csv", control_effects)
    write_markdown_summary(output / "core_metrics.md", args.rq, paths, summaries + supporting, **config.metadata())
    plot_summary(output / "core_metrics.png", args.rq, paths, summaries + supporting)
    (output / "evaluation_metadata.json").write_text(json.dumps({**config.metadata(), "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": BOOTSTRAP_SEED, "binary_proportion_interval": "wilson", "paired_interval": "seed_bootstrap", "sources": {label: str(path.resolve()) for label, path in paths}}, indent=2) + "\n")
    print(f"evaluation_report={output / 'core_metrics.md'}")


if __name__ == "__main__":
    main()
