"""Generate a reproducible figure report from an explicit source manifest."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor

from adaptive_sorting.naming import experiment_id, EXPERIMENT_IDS
from adaptive_sorting.analysis.report_validation import source_fingerprints, validate_cache, source_code_hashes
from adaptive_sorting.analysis import plot_agent_comparison as curves
from adaptive_sorting.analysis import plot_diagnostics as diagnostics
from adaptive_sorting.analysis import summarize_agent_comparison as summary
from adaptive_sorting.analysis.evaluation import EvaluationConfig, DEFAULT_ROLLING_WINDOW
from adaptive_sorting.analysis.statistics import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
from adaptive_sorting.analysis.plot_style import plt


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def cached_summaries(folder, rq, name, definitions):
    return [summary.MetricSummary(row['agent'], row['scope'], definitions[row['metric']],
            *(float(row[key]) if row[key] else None for key in ('value', 'ci95_lower', 'ci95_upper')),
            int(row['n_valid']), int(row['n_expected']), int(row['horizon']) if row.get('horizon') else None)
            for row in read_rows(folder / name)]


def package_report(output):
    items = [path for rq in summary.METRICS_BY_RQ
             for path in sorted((output / experiment_id(rq)).glob('*.png'))]
    if not items:
        return
    columns = min(4, len(items))
    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.6 * rows), squeeze=False, layout='constrained')
    for axis, path in zip(axes.flat, items):
        axis.imshow(plt.imread(path))
        axis.set_title(f'{path.parent.name}: {path.stem}', fontsize=9)
    for axis in axes.flat:
        axis.axis('off')
    fig.savefig(output / 'overview.png', dpi=100)
    plt.close(fig)
    (output / 'figure_index.json').write_text(json.dumps({i: str(p.relative_to(output)) for i, p in enumerate(items, 1)}, indent=2) + '\n')


def write_report_status(output, status, **details):
    path = output / 'report_status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now().astimezone().isoformat()
    content = {**previous, 'status': status, 'updated_at': now, **details}
    if status == 'running':
        content['started_at'] = now
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(content, indent=2) + '\n')
    temporary.replace(path)


def select_current(report):
    """Select a completed sibling report without replacing a real directory."""
    report = Path(report).resolve()
    status_path = report / 'report_status.json'
    if not status_path.is_file() or json.loads(status_path.read_text()).get('status') != 'complete':
        raise ValueError('Only a completed report can be selected as current.')
    current = report.parent / 'current'
    if current.exists() and not current.is_symlink():
        raise ValueError('Refusing to replace a non-symlink current entry.')
    temporary = report.parent / f'.current-{uuid.uuid4().hex}'
    try:
        temporary.symlink_to(report.name, target_is_directory=True)
        temporary.replace(current)
    finally:
        temporary.unlink(missing_ok=True)


def generate(manifest, output, *, reuse_metrics=None, window=DEFAULT_ROLLING_WINDOW, recompute=(), workers=1):
    if window <= 0:
        raise ValueError('Rolling window must be positive.')
    normalized = dict(manifest)
    for rq, spec in list(normalized.items()):
        if rq not in summary.METRICS_BY_RQ or not spec['sources']:
            raise ValueError(f'Invalid protocol or empty sources: {rq}')
        normalized[rq] = {**spec, 'sources': {label: str(Path(path).resolve())
                                           for label, path in spec['sources'].items()}}
        for path in normalized[rq]['sources'].values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    if not normalized:
        raise ValueError('A report must contain at least one experiment.')
    if 'rq3a' in normalized and 'rq3b' in normalized:
        if set(normalized['rq3a']['sources']) != set(normalized['rq3b']['sources']):
            raise ValueError('L2 comparisons require matching controller labels.')
    recompute = tuple(recompute)
    if set(recompute) - set(normalized):
        raise ValueError('Recompute experiments must be present in the manifest.')
    output = Path(output)
    if output.is_symlink() or output.name == 'current':
        raise ValueError('Generate into a new concrete directory, never current or a symlink.')
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Use a new empty output directory to preserve previous reports.')
    reuse_metrics = Path(reuse_metrics) if reuse_metrics is not None else None
    if reuse_metrics is not None:
        status_path = reuse_metrics / 'report_status.json'
        if not status_path.is_file() or json.loads(status_path.read_text()).get('status') != 'complete':
            raise ValueError('Cache reuse requires a completed report.')
    config = EvaluationConfig()
    hashes = source_code_hashes()
    # Validate every source and cache before creating an output directory.
    prepared = {}
    for rq, spec in normalized.items():
        paths = [(label, Path(path)) for label, path in spec['sources'].items()]
        fingerprints = source_fingerprints(paths)
        old = reuse_metrics / experiment_id(rq) if reuse_metrics and rq not in recompute else None
        if old:
            previous = json.loads((old / 'evaluation_metadata.json').read_text())
            validate_cache(previous, paths, config, fingerprints, hashes[0],
                           BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            for filename in ('per_seed_metrics.csv', 'core_metrics.csv', 'supporting_metrics.csv', 'paired_differences.csv'):
                if not (old / filename).is_file():
                    raise FileNotFoundError(old / filename)
        prepared[rq] = fingerprints
    output.mkdir(parents=True, exist_ok=True)
    write_report_status(output, 'running')
    try:
        _generate(normalized, output, reuse_metrics=reuse_metrics, window=window,
                  recompute=recompute, hashes=hashes, prepared=prepared, workers=workers)
    except BaseException as exc:
        write_report_status(output, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    write_report_status(output, 'complete')


def _protocol(rq, paths, output, old, fingerprints, window, commit, hashes):
    """Metrics and figures of one experiment; runs in a worker process when parallel."""
    config = EvaluationConfig()
    analysis_hash, figure_hash = hashes
    print(f'{datetime.now().isoformat()} {rq}: metrics and figures', flush=True)
    target = output / experiment_id(rq)
    if old:
        previous = json.loads((old / 'evaluation_metadata.json').read_text())
        for filename in ('per_seed_metrics.csv', 'core_metrics.csv', 'supporting_metrics.csv', 'paired_differences.csv'):
            shutil.copy2(old / filename, target / filename)
        core = cached_summaries(target, rq, 'core_metrics.csv', {d.key:d for d in summary.METRICS_BY_RQ[rq]})
        definitions = summary.supporting_definitions(rq, read_rows(target / 'per_seed_metrics.csv'))
        supporting = cached_summaries(target, rq, 'supporting_metrics.csv', {d.key:d for d in definitions})
    else:
        records = summary.evaluate_agents(rq, paths, config, require_complete_plan=True)
        core = summary.summarize_records(rq, records)
        supporting = summary.summarize_records(rq, records, supporting=True)
        summary.write_rows(target / 'per_seed_metrics.csv', [{'agent': label, **r} for label, rows in records.items() for r in rows])
        summary.write_summary_csv(target / 'core_metrics.csv', core)
        summary.write_summary_csv(target / 'supporting_metrics.csv', supporting)
        summary.write_rows(target / 'paired_differences.csv', summary.paired_comparisons(rq, records))
    summary.write_markdown_summary(target / 'core_metrics.md', rq, paths, core + supporting, **config.metadata())
    counts = {label: len({r['seed'] for r in read_rows(target / 'per_seed_metrics.csv') if r['agent'] == label}) for label,_ in paths}
    metadata = {**config.metadata(), 'rolling_window': window,
                'curve_start_policy': 'continuous trailing global trials; available prefix at run start',
                'curve_missing_policy': 'equal mean of available seeds; empty subgroup is missing',
                'bootstrap_samples': BOOTSTRAP_SAMPLES, 'bootstrap_seed': BOOTSTRAP_SEED,
                'binary_proportion_interval': 'wilson', 'paired_interval': 'seed_bootstrap',
                'sources': {label:str(path) for label,path in paths}, 'seed_counts': counts,
                'source_fingerprints': fingerprints,
                'figure_git_commit': commit, 'figure_source_sha256': figure_hash,
                'generated_at': datetime.now().astimezone().isoformat()}
    if not old:
        metadata['analysis_git_commit'] = commit
        metadata['analysis_source_sha256'] = analysis_hash
    if old:
        metadata['metrics_reused_from'] = str(old.resolve())
        metadata['analysis_git_commit'] = previous.get('analysis_git_commit')
        metadata['analysis_source_sha256'] = previous['analysis_source_sha256']
    (target / 'evaluation_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    summary.plot_summary(target / 'core_metrics', rq, paths, core + supporting)
    getattr(curves, f'plot_{rq}')(paths, target / 'learning_curves', window)
    curves.plot_learning_groups(rq, paths, target / 'learning_groups', window)
    if rq in ('rq3a', 'rq3b'):
        diagnostics.rule_figure(target)
        diagnostics.attribution_table(target, paths, config)
        diagnostics.mechanism_curves(target, paths, window)
        diagnostics.memory_curves(target, paths)
    if rq in ('rq4a', 'rq4b'):
        diagnostics.first_figure(target)
    return {'seed_counts': counts, 'missing_core_metrics': [dict(agent=s.agent, metric=s.metric.key, scope=s.scope, n_valid=s.n_valid, n_expected=s.n_expected) for s in core if s.average is None]}


def _context_control(output, selected):
    config = EvaluationConfig()
    target = output / 'rq3b'
    effects = []
    controls = dict(selected['rq3a'])
    for label, path in selected['rq3b']:
        records = summary.evaluate_agents('rq3b', [(label+' with L2',path), (label+' without L2',controls[label])], config, context_control=True, require_complete_plan=True)
        effects.extend(summary.paired_comparisons('rq3b', records))
    summary.write_rows(target / 'context_control_differences.csv', effects)
    diagnostics.control_figure(target)


def _generalization(output, selected, window):
    experiments = {rq: selected[rq] for rq in ('rq4a', 'rq4b')}
    diagnostics.generalization_curves(output / 'rq4b', experiments, window)
    diagnostics.retained_vs_used(output / 'rq4b', experiments)


def _generate(manifest, output, *, reuse_metrics, window, recompute, hashes, prepared, workers):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    selected = {rq: [(label, Path(path).resolve()) for label, path in spec['sources'].items()]
                for rq, spec in manifest.items()}
    (output / 'sources.json').write_text(json.dumps({experiment_id(rq): spec for rq, spec in manifest.items()}, indent=2) + '\n')
    olds = {rq: reuse_metrics / experiment_id(rq) if reuse_metrics and rq not in recompute else None for rq in selected}
    for rq in selected:
        (output / experiment_id(rq)).mkdir()
    tasks = [(_protocol, (rq, paths, output, olds[rq], prepared[rq], window, commit, hashes)) for rq, paths in selected.items()]
    if 'rq3b' in selected and 'rq3a' in selected:
        tasks.append((_context_control, (output, selected)))
    if 'rq4a' in selected and 'rq4b' in selected:
        tasks.append((_generalization, (output, selected, window)))
    if workers > 1:
        # Each task writes only its own files; the figure package below waits for all of them.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [future.result() for future in [pool.submit(function, *args) for function, args in tasks]]
    else:
        results = [function(*args) for function, args in tasks]
    validation = {experiment_id(rq): result for rq, result in zip(selected, results)}
    (output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    package_report(output)
    lines = ['# Experiment figure report', '', '[Overview](overview.png)', '',
             f'Curves: up to {window} global trials, continuous across phase boundaries; available observations are used at run start. Windows are sliced before filtering groups. Empty groups remain missing; each point averages available seeds equally; effective counts are saved in *_coverage.csv.', '',
             'Curves show seed means and pointwise 95% bootstrap intervals. Final windows use 300 global trials. Trials to criterion count global trials from the switch until 90% accuracy over the last 20 new-task observations; the 90% line in figures is a visual reference, not that calculation.', '',
             'Compatible metrics were reused where applicable; per-protocol metadata identifies reused and recomputed statistics.' if reuse_metrics else 'Metrics were recomputed from the selected raw logs.', '',
             '## Figures', '']
    for rq in selected:
        lines.append(f'- {experiment_id(rq)}: [learning curves]({experiment_id(rq)}/learning_curves.png), [core metrics]({experiment_id(rq)}/core_metrics.png), [statistics]({experiment_id(rq)}/core_metrics.md)')
        if rq in ('rq4a', 'rq4b'):
            lines.append(f'  - [First-encounter table]({experiment_id(rq)}/first_encounters_table.md)')
    lines += ['', '## Figure notes', '',
              'Figures carry no in-figure titles; the file name identifies the experiment and figure, and these notes supply what a caption states.', '',
              '- `learning_curves`, `learning_groups`: seed means and pointwise 95% bootstrap intervals over trailing global-trial windows; dashed vertical lines mark switches.',
              '- `core_metrics`: seed means (trials to criterion: medians); 95% intervals are in `core_metrics.md`.',
              f'- `rule_usage`: seed-mean share of decisions by rule category over the final {EvaluationConfig().final_window} global trials.',
              '- `context_control`: with L2 minus without L2 on paired seeds; mean difference and 95% seed-bootstrap interval.',
              '- `first_encounters`: seed proportions of the first novel-color decision with 95% Wilson intervals.',
              f'- `rule_mechanism_<agent>`: (a) seed-mean share of decisions by the condition family of the driving implication, over trailing {window}-trial windows; (b) seed-mean count of retained implications per family at every trial, when the run records the per-trial memory inventory. Series absent throughout a panel are omitted; the dashed line marks the switch.',
              f'- `rq4b/generalization_sources_<agent>`: Experiments 4a and 4b, seed-mean share of decisions by the class of the driving implication (identity-matching, color-specific) or motor babbling, over trailing {window}-trial windows on all inputs; the dashed line marks the onset of new colors. `rq4b/retained_vs_used.md` sets implications retained at each phase end against the decisions they drove.',
              '- `memory_rule_families`: seed-mean count of retained implications and mean strongest expectation per family (over seeds retaining the family); a diagnostic that complements `rule_mechanism`.',
              '', 'Additional learning_groups figures show overall, old-task and new-task inputs per agent. Core metrics are numerical tables; single binary seed proportions use Wilson intervals, trials to criterion are medians with runs that never reach the criterion counted as exceeding the phase, other means and paired differences use seed bootstrap. Detailed intervals remain in CSV and core_metrics.md. RQ4 final-new-color panels include individual colors; first-encounter tables focus on decisions before any novel-color feedback.', '', 'Source selection: `sources.json`. Per-protocol metadata records statistical settings, seed counts, source paths, code identity and metric reuse. Proportion differences use percentage points; descriptive rule shares have neutral direction. Cache reuse checks source/configuration fingerprints, analysis code and statistical settings. Rule usage plots show seed-mean shares; intervals remain in supporting_metrics.csv.', '',
              'Reproduce with `python -m adaptive_sorting.analysis.generate_report --manifest sources.json --output-dir experiment_logs/evaluation/YYYYMMDD_HHMMSS_main`. Add `--reuse-metrics-from PREVIOUS_REPORT` only when those unchanged raw sources and metric definitions are still applicable.', '']
    (output / 'README.md').write_text('\n'.join(lines))
    print(f'Report: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-metrics-from', type=Path)
    parser.add_argument('--recompute', nargs='*', choices=EXPERIMENT_IDS, default=[])
    parser.add_argument('--select-current', action='store_true', help='Select the successfully generated report as its parent directory current entry.')
    parser.add_argument('--rolling-window', type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument('--workers', type=int, default=8, help='Experiments processed in parallel; each worker holds one experiment\'s trial logs in memory.')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for spec in manifest.values():
        spec['sources'] = {label: str((args.manifest.parent / path).resolve()) for label,path in spec['sources'].items()}
    generate(manifest, args.output_dir, reuse_metrics=args.reuse_metrics_from, window=args.rolling_window, recompute=args.recompute, workers=args.workers)
    if args.select_current:
        select_current(args.output_dir)


if __name__ == '__main__':
    main()
