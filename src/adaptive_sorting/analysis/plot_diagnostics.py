"""Render rule-use, first-encounter, and paired-context figures."""
import csv
import json
import math
import re

from adaptive_sorting.analysis.plot_style import agent_style, TITLES, finish_figure
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def labels_from(rows):
    return list(dict.fromkeys(row['agent'] for row in rows))

def rule_figure(folder):
    rows = read(folder / 'supporting_metrics.csv')
    categories = {
        'relevant_only': ('Color + L1', '#087F8C'),
        'relevant_with_l2': ('Color + L1 + L2', '#8B5FBF'),
        'missing_required': ('Missing required conditions', '#D99A2B'),
        'unclassified': ('Unclassified implication', '#5B8E3E'),
        'motor_babbling': ('Motor babbling', '#D95F59'),
        'unknown': ('Unknown source', '#A0A0A0'),
    }
    labels = labels_from(rows)
    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    bottoms = np.zeros(len(labels))
    for category, (name, color) in categories.items():
        selected = [next(r for r in rows if r['agent'] == label and
                         r['metric'] == f'rule_{category}_share') for label in labels]
        values = np.array([float(r['value']) if r['value'] else np.nan for r in selected])
        ax.bar(range(len(labels)), values, bottom=bottoms, label=name, color=color, width=.6)
        bottoms += np.nan_to_num(values)
    for i, total in enumerate(bottoms):
        if total == 0:
            ax.text(i, .08, 'NA\nNo rule provenance', ha='center', fontsize=9)
        else:
            assert math.isclose(total, 1.0, abs_tol=1e-9)
    ax.set_xticks(range(len(labels)), [x.replace(' ', '\n', 1) for x in labels])
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Share of decisions (seed mean)')
    finish_figure(fig, folder / 'rule_usage')


def first_figure(folder):
    rows = read(folder / 'core_metrics.csv') + read(folder / 'supporting_metrics.csv')
    groups = [
        ('Before any novel-color feedback', 'first_any_novel_accuracy', 'first_any_novel_rule_share'),
    ]
    labels = labels_from(rows)
    fig, axes = plt.subplots(1, 1, figsize=(10, 2.8), squeeze=False, layout='constrained')
    note = 'Percentages: seed proportion [95% Wilson interval]; NA = unavailable.'
    markdown = [f'# {TITLES[folder.name]} — First novel-color decisions', '', note, '']
    for ax, (title, accuracy, rule_use) in zip(axes.flat, groups):
        headers = ['Agent', 'Correct action (%)', 'Identity-matching implication use (%)']
        cells = []
        for label in labels:
            values = [label]
            for key in (accuracy, rule_use):
                row = next(r for r in rows if r['agent'] == label and r['metric'] == key)
                if not row['value']:
                    value = 'NA'
                else:
                    value = f"{100 * float(row['value']):.1f}"
                    if row['ci95_lower'] and row['ci95_upper']:
                        value += f" [{100 * float(row['ci95_lower']):.1f}, {100 * float(row['ci95_upper']):.1f}]"
                    else:
                        value += ' [CI unavailable]'
                values.append(value)
            cells.append(values)
        ax.axis('off')
        ax.set_title(title, loc='left', fontsize=11)
        table = ax.table(cellText=cells, colLabels=headers, cellLoc='center',
                         colWidths=[.30, .32, .38], bbox=[0, 0, 1, .9])
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
        markdown += [f'## {title}', '', '| ' + ' | '.join(headers) + ' |', '|---|---|---|']
        markdown += ['| ' + ' | '.join(row) + ' |' for row in cells] + ['']
    (folder / 'first_encounters_table.md').write_text('\n'.join(markdown))
    finish_figure(fig, folder / 'first_encounters')


def control_figure(folder):
    rows = read(folder / 'context_control_differences.csv')
    labels = list(dict.fromkeys(row['left'].removesuffix(' with L2') for row in rows))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), layout='constrained')
    metrics = [('new_task_errors', 'Cumulative errors: L1 on (new task)', 'Error difference', 1),
        ('old_task_errors', 'Cumulative errors: L1 off (old task)', 'Error difference', 1),
        ('final_new_task_accuracy', 'Final accuracy: L1 on (new task)', 'Difference (percentage points)', 1),
        ('final_old_task_accuracy', 'Final accuracy: L1 off (old task)', 'Difference (percentage points)', 1)]
    for ax, (key, title, xlabel, scale) in zip(axes, metrics):
        for index, label in enumerate(labels):
            style = agent_style(label)
            row = next(r for r in rows if r['left'] == label + ' with L2' and r['metric'] == key)
            if not row['mean_difference']:
                ax.text(.02, index, 'NA', transform=ax.get_yaxis_transform())
                continue
            expected_unit = 'errors' if key.endswith('_errors') else 'percentage_points'
            if row.get('unit') != expected_unit:
                raise ValueError('Paired-effect units are missing or incompatible; recompute the report.')
            value, lo, hi = [float(row[k]) * scale for k in ('mean_difference', 'ci95_lower', 'ci95_upper')]
            # ONA-TIP: open marker and dashed interval, matching the dashed curves.
            bar = ax.errorbar(value, index, xerr=[[max(0, value-lo)], [max(0, hi-value)]], fmt='o', capsize=4,
                              color=style['color'], mfc='white' if style['linestyle'] == '--' else style['color'])
            bar[2][0].set_linestyle(style['linestyle'])
        ax.axvline(0, color='#777777', ls='--', lw=1)
        ax.set_yticks(range(len(labels)), labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=.2)
    finish_figure(fig, folder / 'context_control')


def attribution_table(folder, paths, config):
    """Decision attribution over the final mixed window, recomputed from the trial logs."""
    from adaptive_sorting.analysis.rq3_relational_diagnostics import decision_attribution
    rows = []
    for label, path in paths:
        with path.open(newline='') as stream:
            rows.extend(dict(agent=label, **r) for r in decision_attribution(csv.DictReader(stream), config))
    with (folder / 'decision_attribution.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Decision attribution', '',
             f'Final mixed window: {config.final_window} global trials, the window of final accuracy.',
             'Shares average all expected seeds; conditional accuracy averages seeds using that category. Counts are in CSV.', '']
    for label in dict(paths):
        lines += [f'## {label}', '', '| Stage | Category | Decision share | Conditional accuracy |', '|---|---|---:|---:|']
        for r in rows:
            if r['agent'] == label:
                values = ['NA' if r[k] == '' else f'{100*r[k]:.1f}%' for k in ('trial_share', 'conditional_accuracy')]
                lines.append(f"| {r['stage']} | {r['decision_category']} | {' | '.join(values)} |")
        lines.append('')
    (folder / 'decision_attribution.md').write_text('\n'.join(lines))


FAMILY_COLORS = {
    'color': '#4C78A8', 'L1': '#F58518', 'L2': '#B279A2', 'color+L1': '#087F8C',
    'color+L2': '#E45756', 'L1+L2': '#9D755D', 'color+L1+L2': '#8B5FBF',
}
# Other decision sources; motor babbling avoids light gray, which reads as an unfilled area.
SOURCE_COLORS = {'motor_babbling': ('Motor babbling', '#EECA3B'),
                 'unclassified': ('Unclassified implication', '#5B8E3E'), 'unknown': ('Unknown source', '#444444')}


def decision_series(row):
    """Rule family of a learned-rule decision; otherwise its non-rule source."""
    if row.get('decision_source') == 'learned_rule':
        family = row.get('driving_rule_family') or 'unclassified'
        return family if family in FAMILY_COLORS else 'unclassified'
    return row.get('decision_source') if row.get('decision_source') in SOURCE_COLORS else 'unknown'


def series_style(name):
    return (name, FAMILY_COLORS[name]) if name in FAMILY_COLORS else SOURCE_COLORS[name]


def slug(label):
    return re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')


def decision_shares(path, window):
    """Seed-mean share of decisions per series over trailing global-trial windows, or None without provenance."""
    from adaptive_sorting.analysis.evaluation import rolling_accuracy
    by_seed = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if 'decision_source' not in (reader.fieldnames or ()):
            return None
        for row in reader:
            by_seed.setdefault(int(row['seed']), []).append(row)
    rows = [sorted(seed_rows, key=lambda r: int(r['trial'])) for seed_rows in by_seed.values()]
    shares = {name: np.nanmean([rolling_accuracy(seed_rows, window, correct=lambda r, n=name: decision_series(r) == n)
                                for seed_rows in rows], axis=0) for name in (*FAMILY_COLORS, *SOURCE_COLORS)}
    return int(rows[0][0]['mixed_phase_start']) - .5, shares


def retained_counts(path):
    """Seed-mean count of retained implications per family per trial, or None without an inventory."""
    inventory = path.with_name('retained_rule_inventory.csv')
    rows = read(inventory) if inventory.is_file() else []
    if not rows:
        return None
    counts = {}
    for row in rows:
        counts.setdefault(row['rule_family'], {})[int(row['trial'])] = float(row['mean_count'])
    trials = sorted(next(iter(counts.values())))
    families = [name for name in (*FAMILY_COLORS, *SOURCE_COLORS) if name in counts]
    return np.array(trials), {family: np.array([counts[family].get(t, 0.0) for t in trials]) for family in families}


def mechanism_curves(folder, paths, window):
    """Per agent: (a) decision shares by driving implication family and (b) retained implication counts, on one trial axis."""
    records = []
    for label, path in paths:
        decisions = decision_shares(path, window)
        if decisions is None:
            continue
        switch, shares = decisions
        retained = retained_counts(path)
        panels = [('(a) Decision sources', 'Share of decisions', np.arange(1, len(shares['color']) + 1), shares)]
        if retained is not None:
            panels.append(('(b) Retained implications', 'Implications (seed mean)', *retained))
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.1 * len(panels)), sharex=True,
                                 squeeze=False, layout='constrained')
        for index, (axis, (title, ylabel, trials, values)) in enumerate(zip(axes.flat, panels)):
            # Series that never occur in a panel are omitted; the CSV keeps every series.
            shown = [name for name in values if np.nanmax(np.nan_to_num(values[name])) > 0]
            if index == 0:
                names, colors = zip(*(series_style(name) for name in shown))
                axis.stackplot(trials, *[np.nan_to_num(values[name]) for name in shown], labels=names, colors=colors, alpha=.9)
                axis.set_ylim(0, 1)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                # Counts stay as separate lines: shares hide the growth that distinguishes the families.
                for name in shown:
                    name_label, color = series_style(name)
                    axis.plot(trials, values[name], color=color, lw=1.6, label=name_label)
                axis.set_ylim(bottom=0)
                axis.grid(axis='y', alpha=.22)
            axis.axvline(switch, color='#333333', ls='--', lw=1)
            axis.set(ylabel=ylabel, xlim=(1, trials[-1]))
            axis.set_title(title, loc='left')
            records += [dict(agent=label, panel=ylabel, trial=int(t), series=name, value=float(values[name][i]))
                        for name in values for i, t in enumerate(trials)]
        axes.flat[-1].set_xlabel('Trial')
        finish_figure(fig, folder / f'rule_mechanism_{slug(label)}')
    if records:
        with (folder / 'rule_mechanism_curves.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def memory_curves(folder, paths):
    """Retained implication counts and strongest expectation per family."""
    panels = []
    for label, path in paths:
        inventory = path.with_name('retained_rule_inventory.csv')
        if not inventory.is_file():
            continue
        rows = read(inventory)
        if not rows:
            continue
        metadata = json.loads(path.with_name('metadata.json').read_text())
        panels.append((label, metadata['before_trials'] + .5, rows))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 3.8 * len(panels)), squeeze=False, layout='constrained')
    for (count_axis, truth_axis), (label, switch, rows) in zip(axes, panels):
        for family, color in FAMILY_COLORS.items():
            selected = [r for r in rows if r['rule_family'] == family]
            trials = [int(r['trial']) for r in selected]
            count_axis.plot(trials, [float(r['mean_count']) for r in selected], color=color, lw=1.4, label=family)
            truth_axis.plot(trials, [float(r['mean_max_expectation']) if r['mean_max_expectation'] else np.nan for r in selected], color=color, lw=1.4)
        for axis in (count_axis, truth_axis):
            axis.axvline(switch, color='#333333', ls='--', lw=1)
            axis.set_xlabel('Trial')
        count_axis.set(title=f'{label}: retained implications', ylabel='Implications (seed mean)')
        truth_axis.set(title=f'{label}: strongest implication', ylabel='Mean max expectation', ylim=(.5, 1))
    finish_figure(fig, folder / 'memory_rule_families')


# RQ4 decision sources; driving_rule_class values keep their code names, labels follow the thesis.
GENERALIZATION_SERIES = {
    'target_match_generalization': ('Identity-matching implication', '#8B5FBF'),
    'other_generalized': ('Other variable implication', '#9D755D'),
    'grounded': ('Color-specific implication', '#4C78A8'),
    'motor_babbling': ('Motor babbling', '#EECA3B'),
    'unknown': ('Unknown source', '#444444'),
}
IMPLICATION_CLASSES = ('target_match_generalization', 'other_generalized', 'grounded')


def generalization_series(row):
    if row.get('decision_source') == 'learned_rule':
        return row.get('driving_rule_class') if row.get('driving_rule_class') in IMPLICATION_CLASSES else 'unknown'
    return 'motor_babbling' if row.get('decision_source') == 'motor_babbling' else 'unknown'


def seed_trials(path):
    """Rows per seed in trial order, or None when the run has no decision provenance."""
    by_seed = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if 'driving_rule_class' not in (reader.fieldnames or ()):
            return None
        for row in reader:
            by_seed.setdefault(int(row['seed']), []).append(row)
    # Bandit logs share the columns but fill them with n/a.
    if not any(row['decision_source'] in ('learned_rule', 'motor_babbling') for rows in by_seed.values() for row in rows):
        return None
    return {seed: sorted(rows, key=lambda r: int(r['trial'])) for seed, rows in by_seed.items()}


def generalization_curves(folder, experiments, window):
    """Per agent, one row per RQ4 experiment: seed-mean decision shares by implication class over all inputs."""
    from adaptive_sorting.analysis.evaluation import rolling_accuracy
    labels = [label for label, _ in next(iter(experiments.values()))]
    records = []
    for label in labels:
        panels = []
        for experiment, paths in experiments.items():
            runs = seed_trials(dict(paths)[label])
            if runs is None:
                break
            rows = list(runs.values())
            shares = {name: np.nanmean([rolling_accuracy(r, window, correct=lambda x, n=name: generalization_series(x) == n)
                                        for r in rows], axis=0) for name in GENERALIZATION_SERIES}
            panels.append((experiment, int(rows[0][0]['expanded_phase_start']) - .5, shares))
        else:
            fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.8 * len(panels)), sharex=True,
                                     squeeze=False, layout='constrained')
            for axis, (experiment, onset, shares) in zip(axes.flat, panels):
                trials = np.arange(1, len(shares['grounded']) + 1)
                shown = [n for n in shares if np.nanmax(np.nan_to_num(shares[n])) > 0]
                axis.stackplot(trials, *[np.nan_to_num(shares[n]) for n in shown],
                               labels=[GENERALIZATION_SERIES[n][0] for n in shown],
                               colors=[GENERALIZATION_SERIES[n][1] for n in shown], alpha=.9)
                axis.axvline(onset, color='#333333', ls='--', lw=1)
                axis.set(ylabel='Share of decisions', xlim=(1, trials[-1]), ylim=(0, 1))
                axis.set_title(f'Experiment {experiment.removeprefix("rq")}', loc='left')
                axis.yaxis.set_major_formatter(PercentFormatter(1))
                records += [dict(agent=label, experiment=experiment, trial=int(t), series=n, share=float(shares[n][i]))
                            for n in shares for i, t in enumerate(trials)]
            axes.flat[-1].set_xlabel('Trial')
            finish_figure(fig, folder / f'generalization_sources_{slug(label)}')
    if records:
        with (folder / 'generalization_sources.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def retained_vs_used(folder, experiments):
    """Retained implications at phase ends against the decisions they drove, per phase; seed means."""
    from adaptive_sorting.analysis.rq4_relational_diagnostics import classify_generalization_rule
    phases = (('known_only', 'known_only'), ('expanded_inputs', 'expanded_inputs'))
    labels = [label for label, _ in next(iter(experiments.values()))]
    rows = []
    for label in labels:
        for experiment, paths in experiments.items():
            path = dict(paths)[label]
            snapshots = path.with_name('relational_rule_snapshots.jsonl')
            runs = seed_trials(path)
            if runs is None or not snapshots.is_file():
                continue
            retained = {}
            with snapshots.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    key = (item['seed'], item['phase'], classify_generalization_rule(item['rule']))
                    retained[key] = retained.get(key, 0) + 1
            for snapshot_phase, trial_phase in phases:
                for name in GENERALIZATION_SERIES:
                    used = [np.mean([generalization_series(r) == name for r in seed_rows if r['phase'] == trial_phase])
                            for seed_rows in runs.values()]
                    counts = [retained.get((seed, snapshot_phase, name), 0) for seed in runs] if name in IMPLICATION_CLASSES else None
                    rows.append(dict(agent=label, experiment=experiment, phase=trial_phase, series=name,
                                     mean_retained_at_phase_end='' if counts is None else float(np.mean(counts)),
                                     retaining_seeds='' if counts is None else sum(c > 0 for c in counts),
                                     decision_share=float(np.mean(used)), seeds=len(runs)))
    if not rows:
        return
    with (folder / 'retained_vs_used.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Retained versus used implications (RQ4)', '',
             'Retained: seed-mean count in memory at the end of the phase (seeds retaining at least one). '
             'Used: seed-mean share of decisions in the phase. Memory is read only at phase ends.', '']
    for label in dict.fromkeys(r['agent'] for r in rows):
        for experiment in dict.fromkeys(r['experiment'] for r in rows if r['agent'] == label):
            lines += [f'## {label}, Experiment {experiment.removeprefix("rq")}', '',
                      '| Source | Retained, end initial | Used, initial | Retained, end expanded | Used, expanded |',
                      '|---|---:|---:|---:|---:|']
            for name, (text, _) in GENERALIZATION_SERIES.items():
                cells = []
                for phase, _ in phases:
                    r = next(r for r in rows if (r['agent'], r['experiment'], r['phase'], r['series']) == (label, experiment, phase, name))
                    cells += ['—' if r['mean_retained_at_phase_end'] == '' else f"{r['mean_retained_at_phase_end']:.2f} ({r['retaining_seeds']}/{r['seeds']})",
                              f"{100 * r['decision_share']:.1f}%"]
                lines.append(f'| {text} | ' + ' | '.join(cells) + ' |')
            lines.append('')
    (folder / 'retained_vs_used.md').write_text('\n'.join(lines))
