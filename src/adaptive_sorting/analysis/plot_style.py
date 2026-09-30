"""Shared presentation and alignment checks for experiment figures."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'thesis-plot-cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams['legend.fontsize'] = 9

from adaptive_sorting.analysis.statistics import curve_summary
from adaptive_sorting.analysis.trial_ticks import set_trial_ticks

AGENT_COLORS = {
    'ONA': '#087F8C', 'Relational ONA': '#8B5FBF',
    'ONA (flat)': '#087F8C', 'ONA (relational)': '#8B5FBF',
    'ONA-TIP (flat)': '#087F8C', 'ONA-TIP (relational)': '#8B5FBF',
    'Epsilon-greedy': '#D95F59', 'Contextual UCB1': '#D99A2B',
    'Contextual SW-UCB': '#5B8E3E',
}


# Agents drawn dashed; figures that give such an agent its own hue may clear this.
DASHED_AGENTS = ('ONA-TIP',)


def agent_style(label):
    """Hue marks the encoding; ONA-TIP shares its encoding's hue with a dashed line."""
    return {'color': AGENT_COLORS.get(label, '#6C757D'), 'linestyle': '--' if label.startswith(DASHED_AGENTS) else '-'}


GROUP_COLORS = ('#087F8C', '#D95F59', '#4C78A8')
TITLES = {
    'rq1': 'RQ1 — Initial learning',
    'rq2a': 'RQ2 / Experiment 2a — Knowledge revision: single rule change',
    'rq2b': 'RQ2 / Experiment 2b — Knowledge revision: repeated rule changes',
    'rq3a': 'RQ3 / Experiment 3a — Causal relevance: without L2',
    'rq3b': 'RQ3 / Experiment 3b — Causal relevance: with irrelevant L2',
    'rq4a': 'RQ4 / Experiment 4a — Generalization: without target colors',
    'rq4b': 'RQ4 / Experiment 4b — Generalization: with target colors',
}



def phase_key(record):
    for field in ('phase_index', 'phase'):
        if hasattr(record, field):
            return getattr(record, field)
    for field in ('change_point', 'expanded_phase_start'):
        if hasattr(record, field):
            return record.trial >= getattr(record, field)
    return 1


def validate_curves(grouped_by_agent):
    """Require complete, aligned trial coordinates and phase schedules."""
    if not grouped_by_agent or any(not group for group in grouped_by_agent.values()):
        raise ValueError('Curves require nonempty agents and seeds.')
    expected_seeds = set(next(iter(grouped_by_agent.values())))
    reference = None
    for label, grouped in grouped_by_agent.items():
        if set(grouped) != expected_seeds:
            raise ValueError(f'Seed sets differ for {label}.')
        for seed, records in grouped.items():
            trials = [r.trial for r in records]
            if not trials or trials != list(range(1, len(records) + 1)):
                raise ValueError(f'Missing, duplicate, or unordered trials: {label}, seed {seed}.')
            schedule = [(r.trial, phase_key(r)) for r in records]
            if reference is not None and schedule != reference:
                raise ValueError(f'Trial horizons or phase schedules differ: {label}, seed {seed}.')
            reference = schedule
    return [r.trial for r in next(iter(next(iter(grouped_by_agent.values())).values()))]


def draw_curve(axis, trials, curves, label, color, linestyle='-'):
    import numpy as np
    counts = np.isfinite(curves).sum(axis=0).tolist()
    axis._curve_coverage = getattr(axis, "_curve_coverage", []) + [(label, list(trials), counts)]
    average, lower, upper = curve_summary(curves)
    axis.fill_between(trials, lower, upper, color=color, alpha=.14, linewidth=0)
    axis.plot(trials, average, color=color, linestyle=linestyle, linewidth=1.8, label=label)


def accuracy_axis(axis, horizon, window, *, baseline, starts=(), reference=None):
    axis.set(xlabel='Trial', ylabel=f'Accuracy ({window}-trial window)',
             xlim=(.5, horizon + .5), ylim=(0, 1.05))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    set_trial_ticks(axis, horizon, max_labels=8)
    axis.grid(axis='y', alpha=.22)
    axis.axhline(baseline, color='#777777', ls='-.', lw=1, label='Random baseline')
    if reference is not None:
        axis.axhline(reference, color='#777777', ls=':', lw=1,
                     label=f'{reference:.0%} reference')
    for index, start in enumerate(starts):
        axis.axvline(start - .5, color='#555555', ls='--', lw=1,
                     label='Phase change' if index == 0 else None)


def finish_figure(figure, output):
    """Save without an in-figure title; captions and the report README carry descriptions."""
    handles = {}
    for axis in figure.axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    if handles:
        figure.legend(handles.values(), handles.keys(), loc='outside lower center',
                      ncol=min(4, len(handles)), frameon=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Default reports use PNG; explicit output extensions remain supported.
    for suffix in ('.png',) if output.suffix == '' else (output.suffix,):
        figure.savefig(output.with_suffix(suffix), dpi=180, bbox_inches='tight', facecolor='white')
    coverage = [(index, label, trial, count)
                for index, axis in enumerate(figure.axes, 1)
                for label, trials, counts in getattr(axis, '_curve_coverage', [])
                for trial, count in zip(trials, counts)]
    if coverage:
        import csv
        with output.with_name(output.stem + '_coverage.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['panel', 'series', 'trial', 'n_valid_seeds'])
            writer.writerows(coverage)
    plt.close(figure)


def single_agent_report(grouped, output, window, series, *, label, baseline,
                        starts=(), reference=None):
    trials = validate_curves({'agent': grouped})
    figure, axis = plt.subplots(figsize=(8, 4.8), layout='constrained')
    for (label, function), color in zip(series, GROUP_COLORS):
        draw_curve(axis, trials, [function(rows) for rows in grouped.values()], label, color)
    accuracy_axis(axis, len(trials), window, baseline=baseline, starts=starts, reference=reference)
    axis.set_title(label)
    finish_figure(figure, output)
