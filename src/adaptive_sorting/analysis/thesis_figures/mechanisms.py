"""Mechanism figures: decision sources of ONA (relational) and implication truth values in one traced run."""
import json
import re
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from adaptive_sorting.analysis import plot_diagnostics as diagnostics
from adaptive_sorting.analysis.plot_style import plt
from adaptive_sorting.analysis.thesis_figures import WINDOW, manifest, output_path, printed, source, trace

LABEL = 'ONA (relational)'


def decision_sources() -> None:
    """Decision shares by driving implication family in Experiments 3a, 3b, and 3b with 4800 mixed trials."""
    main, extended = manifest(), manifest(extended=True)
    # source, experiment, output, share of the text width in the thesis
    figures = [(main, 'rq3b', output_path('rq3b/mechanism.png'), .9),
               (main, 'rq3a', output_path('appendix/rq3a_mechanism.png'), .85),
               (extended, 'rq3b', output_path('appendix/rq3b_mechanism_extended.png'), .85)]
    for sources, experiment, output, fraction in figures:
        with tempfile.TemporaryDirectory() as folder, printed(11, fraction):
            diagnostics.mechanism_curves(Path(folder), [(LABEL, source(sources, experiment, LABEL))], WINDOW)
            shutil.copy(Path(folder) / 'rule_mechanism_ona_relational.png', output)


@printed(11)
def generalization_sources() -> None:
    """Decision shares by implication class in Experiments 4a and 4b."""
    sources = manifest()
    experiments = {rq: [(LABEL, source(sources, rq, LABEL))] for rq in ('rq4a', 'rq4b')}
    with tempfile.TemporaryDirectory() as folder:
        diagnostics.generalization_curves(Path(folder), experiments, WINDOW)
        shutil.copy(Path(folder) / 'generalization_sources_ona_relational.png', output_path('rq4/generalization_sources.png'))


@printed(10)
def tip_mechanism() -> None:
    """Expectation of the replaced and the new implication of green around Switch 2 of Experiment 2b, run 1.

    Data are the per-trial memory snapshots of the implication traces of run 1 with the ONA and
    the ONA-TIP binary; the traces must reproduce the main runs action by action. Truth values are
    those ONA prints at the snapshot, i.e. projected to that time for ONA-TIP. Green is shown
    because its adaptation in ONA is typical of the 200 runs; orange in this run has an unusually
    long stage of motor babbling after the replaced implication stops driving decisions.
    """
    color, old_bin, new_bin = 'green', '4', '2'
    switch, start, end = 1401, 1301, 2600
    pattern = re.compile(rf'<\(<\(sphere \* {color}\) --> is> &/ <\(\{{SELF\}} \* bin(\d)\) --> \^place>\) '
                         r'=/> correct_sorting>\. \{([\d.]+) ([\d.]+)\}')
    line_colors = {old_bin: '#D95F59', new_bin: '#4C78A8'}
    sources = manifest()
    figure, axes = plt.subplots(1, 2, figsize=(10, 2.7), sharex=True, sharey=True)
    for axis, label in zip(axes, ('ONA (relational)', 'ONA-TIP (relational)')):
        records = [json.loads(line) for line in trace(label).open()]
        formal = pd.read_csv(source(sources, 'rq2b', label), usecols=['seed', 'trial', 'selected_action'])
        if formal[formal.seed == 1].sort_values('trial').selected_action.tolist() != [r['action'] for r in records]:
            raise ValueError(f'Trace differs from the main run: {label}')
        for bin_ in (old_bin, new_bin):
            trials, expectations = [], []
            for record in records[start - 1:end]:
                truth = next((match for line in record['rules_after_feedback']
                              if (match := pattern.search(line)) and match[1] == bin_), None)
                trials.append(record['trial'])
                # The new implication has no curve before its first success forms it.
                expectations.append(float(truth[3]) * (float(truth[2]) - .5) + .5 if truth else float('nan'))
            axis.plot(trials, expectations, color=line_colors[bin_], linewidth=1.3,
                      label=f'{color} $\\rightarrow$ bin {bin_} ({"replaced" if bin_ == old_bin else "new"})')
        axis.axvline(switch - .5, color='#6C757D', linestyle='--', linewidth=1)
        axis.text(.02, .05, label, transform=axis.transAxes)
        axis.set(ylim=(-.03, 1.03), xlim=(start, end), xlabel='Trial')
        axis.grid(alpha=.25)
    axes[0].set_ylabel('Expectation')
    axes[1].legend(loc='lower right', frameon=False)
    figure.tight_layout()
    figure.savefig(output_path('rq2b/tip_mechanism.png'), dpi=200, bbox_inches='tight')
    plt.close(figure)
