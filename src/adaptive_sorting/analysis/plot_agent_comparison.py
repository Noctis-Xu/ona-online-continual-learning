"""Compare agents using shared curve statistics and presentation."""
from pathlib import Path
import argparse
from adaptive_sorting.analysis.evaluation import DEFAULT_ROLLING_WINDOW
from adaptive_sorting.analysis.statistics import curve_summary
from adaptive_sorting.analysis.plot_style import (
    plt, agent_style, TITLES, validate_curves, draw_curve,
    accuracy_axis, finish_figure,
)
from adaptive_sorting.analysis import (
    plot_rq1 as rq1_plot, plot_rq2a as rq2a_plot, plot_rq2b as rq2b_plot,
    plot_rq3a as rq3a_plot, plot_rq4a as rq4a_plot,
)
AgentTrials = list[tuple[str, Path]]


def add_agent_curves(axis, grouped_by_agent, curve_function):
    trials = validate_curves(grouped_by_agent)
    import numpy as np
    availability = None
    for label, grouped in grouped_by_agent.items():
        values = [curve_function(grouped[seed]) for seed in sorted(grouped)]
        current = np.isfinite(values)
        if availability is not None and not np.array_equal(availability, current):
            raise ValueError('Subgroup seed coverage differs between agents.')
        availability = current
        style = agent_style(label)
        draw_curve(axis, trials, values, label, style['color'], style['linestyle'])
    return len(trials)


def first_seed_records(grouped):
    return next(iter(next(iter(grouped.values())).values()))


def plot_rq1(agent_trials, output, window):
    grouped = {label: rq1_plot.load_trials(path) for label, path in agent_trials}
    trials = validate_curves(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for label, seeds in grouped.items():
        stats = curve_summary([rq1_plot.rolling_accuracy(rows, window) for rows in seeds.values()])
        for axis in axes:
            style = agent_style(label)
            axis.fill_between(trials, stats[1], stats[2], color=style['color'], alpha=.14, linewidth=0)
            axis.plot(trials, stats[0], label=label, lw=1.8, **style)
    for axis in axes:
        accuracy_axis(axis, len(trials), window, baseline=.2, reference=.9)
    early = min(200, len(trials))
    axes[0].set_title(f'Initial learning: first {early} trials')
    axes[0].set_xlim(.5, early + .5)
    from adaptive_sorting.analysis.trial_ticks import set_trial_ticks
    set_trial_ticks(axes[0], early, max_labels=8)
    axes[1].set_title('Full run')
    finish_figure(fig, output)


def plot_rq2a(agent_trials, output, window):
    import csv
    conditions = None
    for _, path in agent_trials:
        with path.open() as stream:
            current = {int(r['change_size']) for r in csv.DictReader(stream)}
        if conditions is not None and conditions != current:
            raise ValueError('Rule-change conditions differ between agents.')
        conditions = current
    fig, axes = plt.subplots(3, len(conditions), figsize=(5 * len(conditions), 10), squeeze=False, layout='constrained')
    for column, size in enumerate(sorted(conditions)):
        grouped = {label: rq2a_plot.load_trials(path, size) for label, path in agent_trials}
        start = first_seed_records(grouped)[0].change_point
        for row, changed in enumerate((None, True, False)):
            axis = axes[row, column]
            n = add_agent_curves(axis, grouped, lambda records: rq2a_plot.rolling_accuracy(records, window, changed))
            accuracy_axis(axis, n, window, baseline=.2, starts=[start])
            name = "overall" if changed is None else "new-task inputs" if changed else "old-task inputs"
            axis.set_title(f'{size} changed colors — {name}')
    finish_figure(fig, output)


def plot_rq2b(agent_trials, output, window):
    grouped = {label: rq2b_plot.load_trials(path) for label, path in agent_trials}
    fig, axis = plt.subplots(figsize=(12, 4.8), layout='constrained')
    n = add_agent_curves(axis, grouped, lambda records: rq2b_plot.rolling_accuracy(records, window))
    records = first_seed_records(grouped)
    starts = [r.phase_start for r in records if r.trial == r.phase_start and r.phase_start > 1]
    accuracy_axis(axis, n, window, baseline=.2, starts=starts)
    rq2b_plot.set_phase_ticks(axis, records)
    axis.set_ylabel(f'Overall accuracy ({window}-trial window)')
    finish_figure(fig, output)


def grouped_comparison(agent_trials, output, window, *, loader, curve, groups, start_field, baseline):
    grouped = {label: loader(path) for label, path in agent_trials}
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.8), squeeze=False, layout='constrained')
    start = getattr(first_seed_records(grouped)[0], start_field)
    for axis, (name, group) in zip(axes.flat, groups):
        n = add_agent_curves(axis, grouped, lambda rows: curve(rows, window, group))
        accuracy_axis(axis, n, window, baseline=baseline, starts=[start])
        axis.set_title(name)
    finish_figure(fig, output)


def plot_rq3a(agent_trials, output, window):
    grouped_comparison(agent_trials, output, window, loader=rq3a_plot.load_trials,
                       curve=rq3a_plot.rolling_accuracy,
                       groups=[('Overall', None), ('New-task inputs: L1 on', 'light_on'), ('Old-task inputs: L1 off', 'light_off')],
                       start_field='mixed_phase_start', baseline=.2)


plot_rq3b = plot_rq3a


def plot_rq4a(agent_trials, output, window):
    grouped_comparison(agent_trials, output, window, loader=rq4a_plot.load_trials,
                       curve=rq4a_plot.rolling_accuracy,
                       groups=[('Overall', 'overall'), ('New-task inputs: new colors', 'novel'), ('Old-task inputs: old colors', 'known')],
                       start_field='expanded_phase_start', baseline=1/7)


plot_rq4b = plot_rq4a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rq",
        choices=tuple(TITLES),
        required=True,
    )
    parser.add_argument(
        "--agent-trials",
        action="append",
        nargs=2,
        required=True,
        metavar=("LABEL", "PATH"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_trials = [(label, Path(path)) for label, path in args.agent_trials]
    labels = [label for label, _ in agent_trials]
    if len(agent_trials) < 2:
        raise ValueError("Comparison requires at least two agents.")
    if len(set(labels)) != len(labels):
        raise ValueError("Agent labels must be unique.")
    window = DEFAULT_ROLLING_WINDOW if args.rolling_window is None else args.rolling_window
    plotters = {
        "rq1": plot_rq1,
        "rq2a": plot_rq2a,
        "rq2b": plot_rq2b,
        "rq3a": plot_rq3a,
        "rq3b": plot_rq3b,
        "rq4a": plot_rq4a,
        "rq4b": plot_rq4b,
    }
    plotters[args.rq](agent_trials, args.output, window)
    print(f"output={args.output}")



def plot_learning_groups(rq, agent_trials, output, window):
    """Show overall, old-task and new-task performance within each agent."""
    import csv
    if rq == 'rq2a':
        with agent_trials[0][1].open() as stream:
            conditions = sorted({int(row['change_size']) for row in csv.DictReader(stream)})
        loader, curve = rq2a_plot.load_trials, rq2a_plot.rolling_accuracy
        groups = [('Overall', None), ('Old-task inputs: unchanged colors', False), ('New-task inputs: changed colors', True)]
        start_field, baseline = 'change_point', .2
    elif rq in {'rq3a', 'rq3b'}:
        conditions = [None]
        loader, curve = rq3a_plot.load_trials, rq3a_plot.rolling_accuracy
        groups = [('Overall', None), ('Old-task inputs: L1 off', 'light_off'), ('New-task inputs: L1 on', 'light_on')]
        start_field, baseline = 'mixed_phase_start', .2
    elif rq in {'rq4a', 'rq4b'}:
        conditions = [None]
        loader, curve = rq4a_plot.load_trials, rq4a_plot.rolling_accuracy
        groups = [('Overall', 'overall'), ('Old-task inputs: old colors', 'known'), ('New-task inputs: new colors', 'novel')]
        start_field, baseline = 'expanded_phase_start', 1/7
    else:
        return
    for condition in conditions:
        grouped = {label: loader(path, condition) if condition is not None else loader(path)
                   for label, path in agent_trials}
        trials = validate_curves(grouped)
        fig, axes = plt.subplots((len(grouped)+1)//2, 2, figsize=(12, 3.5*((len(grouped)+1)//2)),
                                 squeeze=False, layout='constrained')
        for axis, (label, seeds) in zip(axes.flat, grouped.items()):
            for (name, group), color in zip(groups, ('#007f8b', '#4c78a8', '#e45756')):
                draw_curve(axis, trials, [curve(rows, window, group) for rows in seeds.values()], name, color)
            start = getattr(next(iter(seeds.values()))[0], start_field)
            accuracy_axis(axis, len(trials), window, baseline=baseline, starts=[start])
            axis.set_title(label)
        for axis in list(axes.flat)[len(grouped):]:
            axis.remove()
        suffix = f'_{condition}_colors' if condition is not None else ''
        finish_figure(fig, Path(output).with_name(Path(output).name + suffix))


if __name__ == "__main__":
    main()
