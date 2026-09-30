"""Learning-curve figures: seed-mean accuracy with pointwise bootstrap intervals."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from adaptive_sorting.analysis import plot_rq1, plot_rq2a, plot_rq2b, plot_rq3a, plot_rq4a, plot_style
from adaptive_sorting.analysis.plot_agent_comparison import add_agent_curves, first_seed_records, plot_rq1 as draw_rq1
from adaptive_sorting.analysis.plot_style import plt, accuracy_axis, draw_curve, finish_figure
from adaptive_sorting.analysis.thesis_figures import (
    CONTRAST_COLOR, MAIN_LABELS, WINDOW, manifest, output_path, printed, source, threshold_run,
)

TRIAL_STEPS = (100, 200, 400, 500, 1000, 2000, 5000)
MAX_TRIAL_LABELS = 7


def save(figure, output: Path) -> None:
    for axis in figure.axes:
        # Regular trial axes get at most MAX_TRIAL_LABELS labels, ending at the last trial.
        if isinstance(axis.xaxis.get_major_locator(), MultipleLocator):
            horizon = round(axis.get_xlim()[1] - .5)
            step = next(step for step in TRIAL_STEPS if horizon / step <= MAX_TRIAL_LABELS)
            axis.xaxis.set_major_locator(MultipleLocator(step, horizon % step))
        # Axis titles appear once per row and column: y on the left column, x on the bottom row.
        if spec := axis.get_subplotspec():
            if not spec.is_first_col():
                axis.set_ylabel('')
            if not spec.is_last_row():
                axis.set_xlabel('')
    finish_figure(figure, output)
    output.with_name(output.name + '_coverage.csv').unlink(missing_ok=True)


def contrast_tip() -> None:
    """Draw ONA-TIP (relational) solid in the contrast hue instead of dashed in the ONA hue."""
    plot_style.AGENT_COLORS['ONA-TIP (relational)'] = CONTRAST_COLOR
    plot_style.DASHED_AGENTS = ()


@printed(12)
def rq1() -> None:
    output = output_path('rq1/learning_curves')
    draw_rq1([(label, source(manifest(), 'rq1', label)) for label in MAIN_LABELS], output, WINDOW)
    output.with_name(output.name + '_coverage.csv').unlink(missing_ok=True)


@printed(15)
def rq2a() -> None:
    sources = manifest()
    figure, axes = plt.subplots(3, 3, figsize=(15, 12.6), layout='constrained')
    for row, size in zip(axes, (2, 3, 4)):
        grouped = {label: plot_rq2a.load_trials(source(sources, 'rq2a', label), size) for label in MAIN_LABELS}
        start = first_seed_records(grouped)[0].change_point
        for axis, (title, changed) in zip(row, [('overall', None), ('new-task inputs', True), ('old-task inputs', False)]):
            n = add_agent_curves(axis, grouped, lambda records, c=changed: plot_rq2a.rolling_accuracy(records, WINDOW, c))
            accuracy_axis(axis, n, WINDOW, baseline=.2, starts=[start])
            axis.set_title(f'{size} remapped colors: {title}')
    save(figure, output_path('rq2a/learning_curves'))


def rq2b_sequence(labels: list[str], output: Path) -> None:
    sources = manifest()
    grouped = {label: plot_rq2b.load_trials(source(sources, 'rq2b', label)) for label in labels}
    figure, axis = plt.subplots(figsize=(12, 4.8), layout='constrained')
    n = add_agent_curves(axis, grouped, lambda records: plot_rq2b.rolling_accuracy(records, WINDOW))
    records = first_seed_records(grouped)
    starts = [r.phase_start for r in records if r.trial == r.phase_start and r.phase_start > 1]
    accuracy_axis(axis, n, WINDOW, baseline=.2, starts=starts)
    plot_rq2b.set_phase_ticks(axis, records)
    axis.set_ylabel(f'Overall accuracy ({WINDOW}-trial window)')
    save(figure, output)


@printed(12)
def rq2b() -> None:
    rq2b_sequence(MAIN_LABELS, output_path('rq2b/learning_curves'))


@printed(12)
def rq2b_tip() -> None:
    contrast_tip()
    rq2b_sequence(['ONA (relational)', 'ONA-TIP (relational)'], output_path('rq2b/learning_curves_tip'))


def context_rows(sources: dict, experiments, labels, output: Path, *, mark=None) -> None:
    """Experiments 3a and 3b as rows of overall, L1-on and L1-off accuracy."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), layout='constrained')
    panels = [('overall', None), ('new-task inputs (L1 on)', 'light_on'), ('old-task inputs (L1 off)', 'light_off')]
    for row, experiment in zip(axes, experiments):
        grouped = {label: plot_rq3a.load_trials(source(sources, experiment, label))
                   for label in labels or sources[experiment]['sources']}
        start = first_seed_records(grouped)[0].mixed_phase_start
        for axis, (title, context) in zip(row, panels):
            n = add_agent_curves(axis, grouped, lambda records, c=context: plot_rq3a.rolling_accuracy(records, WINDOW, c))
            accuracy_axis(axis, n, WINDOW, baseline=.2, starts=[start])
            if mark:
                axis.axvline(mark, color='0.6', linestyle=':', linewidth=1)
            axis.set_title(f'Experiment {experiment[2:]}: {title}')
    save(figure, output)


@printed(15)
def rq3() -> None:
    context_rows(manifest(), ('rq3a', 'rq3b'), MAIN_LABELS, output_path('rq3/learning_curves'))


@printed(15)
def rq3_extended() -> None:
    """ONA (relational) with a mixed phase of 4800 trials; the dotted line marks the end of the main budget."""
    context_rows(manifest(extended=True), ('rq3a', 'rq3b'), None,
                 output_path('appendix/rq3_extended_curves'), mark=1400)


@printed(15)
def rq4() -> None:
    sources = manifest()
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.8), layout='constrained')
    panels = [('overall', 'overall'), ('new-task inputs', 'novel'), ('old-task inputs', 'known')]
    for row, experiment in zip(axes, ('rq4a', 'rq4b')):
        grouped = {label: plot_rq4a.load_trials(source(sources, experiment, label)) for label in MAIN_LABELS}
        start = first_seed_records(grouped)[0].expanded_phase_start
        for axis, (title, category) in zip(row, panels):
            n = add_agent_curves(axis, grouped, lambda records, c=category: plot_rq4a.rolling_accuracy(records, WINDOW, c))
            accuracy_axis(axis, n, WINDOW, baseline=1 / 7, starts=[start])
            axis.set_title(f'Experiment {experiment[2:]}: {title}')
            # Keep the relational ONA curve visible where it coincides with UCB1 at 100%.
            for artist in axis.lines + axis.collections:
                if artist.get_label() == 'ONA (relational)':
                    artist.set_zorder(5)
    save(figure, output_path('rq4/learning_curves_ab'))


@printed(15)
def tip_all_experiments() -> None:
    """Overall accuracy of ONA and ONA-TIP (relational) in every experiment except 2b."""
    contrast_tip()
    labels = ['ONA (relational)', 'ONA-TIP (relational)']
    sources = manifest()
    # experiment id, panel title, loader, overall-accuracy curve, phase-start field, chance level
    panels = [
        ('rq1', 'Experiment 1', plot_rq1.load_trials, lambda r: plot_rq1.rolling_accuracy(r, WINDOW), None, 1 / 5),
        ('rq2a', 'Experiment 2a (two remapped colors)', lambda p: plot_rq2a.load_trials(p, 2),
         lambda r: plot_rq2a.rolling_accuracy(r, WINDOW), 'change_point', 1 / 5),
        ('rq3a', 'Experiment 3a', plot_rq3a.load_trials, lambda r: plot_rq3a.rolling_accuracy(r, WINDOW), 'mixed_phase_start', 1 / 5),
        ('rq3b', 'Experiment 3b', plot_rq3a.load_trials, lambda r: plot_rq3a.rolling_accuracy(r, WINDOW), 'mixed_phase_start', 1 / 5),
        ('rq4a', 'Experiment 4a', plot_rq4a.load_trials, lambda r: plot_rq4a.rolling_accuracy(r, WINDOW), 'expanded_phase_start', 1 / 7),
        ('rq4b', 'Experiment 4b', plot_rq4a.load_trials, lambda r: plot_rq4a.rolling_accuracy(r, WINDOW), 'expanded_phase_start', 1 / 7),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.4), layout='constrained')
    for axis, (experiment, title, loader, curve, start_field, chance) in zip(axes.flat, panels):
        grouped = {label: loader(source(sources, experiment, label)) for label in labels}
        starts = [getattr(first_seed_records(grouped)[0], start_field)] if start_field else []
        n = add_agent_curves(axis, grouped, curve)
        accuracy_axis(axis, n, WINDOW, baseline=chance, starts=starts)
        axis.set_title(title)
    save(figure, output_path('appendix/tip_all_experiments'))


@printed(12, .9)
def babbling_threshold() -> None:
    """ONA (relational) at the suppression threshold of 0.55 (main runs) and 0.65, seeds 1-20.

    The intervention runs cover only seeds 1-20, the first seeds in order and not selected by outcome.
    """
    seeds = list(range(1, 21))
    sources = manifest()
    runs = {experiment: (source(sources, experiment, 'ONA (relational)'), threshold_run(experiment))
            for experiment in ('rq3a', 'rq3b')}
    styles = {'0.55': (plot_style.AGENT_COLORS['ONA (relational)'], '-'), '0.65': (CONTRAST_COLOR, '-')}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for axis, (experiment, paths) in zip(axes, runs.items()):
        for (name, (color, style)), path in zip(styles.items(), paths):
            grouped = {seed: rows for seed, rows in plot_rq3a.load_trials(path).items() if seed in seeds}
            if sorted(grouped) != seeds:
                raise ValueError(f'{path} does not contain seeds 1-20')
            curves = [plot_rq3a.rolling_accuracy(grouped[seed], WINDOW) for seed in seeds]
            trials = np.arange(1, len(curves[0]) + 1)
            draw_curve(axis, trials, curves, f'threshold {name}', color, style)
        start = int(pd.read_csv(paths[0], usecols=['mixed_phase_start'], nrows=1).mixed_phase_start.iloc[0])
        accuracy_axis(axis, len(trials), WINDOW, baseline=.2, starts=[start])
        axis.set_title(f'Experiment {experiment[2:]}')
    save(figure, output_path('appendix/rq3_babbling_threshold'))
