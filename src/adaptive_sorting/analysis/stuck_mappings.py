"""Stuck mappings of ONA (relational) in Experiments 3a and 3b (Section 4.3, Appendices B.2 and B.3).

A stuck mapping is a color under one state of L1 in one run that is answered
correctly on at most 10% of its presentations in the final 300 trials. The command
reports their number and effect in the main runs, how many remain in the runs with a
mixed phase of 4800 trials, and the comparison of the suppression thresholds 0.55 and
0.65 over seeds 1-20.
"""
import argparse
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from adaptive_sorting.analysis.thesis_figures import (
    INPUTS, add_input_arguments, manifest, set_inputs, source, threshold_run,
)

LABEL = 'ONA (relational)'
EXPERIMENTS = ('rq3a', 'rq3b')
FINAL_WINDOW = 300
STUCK_ACCURACY = .1
INITIAL_TRIALS = 200
THRESHOLD_SEEDS = range(1, 21)
COLUMNS = ['seed', 'trial', 'context', 'sphere_color', 'is_correct']


def load(path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=COLUMNS)


def final_trials(frame: pd.DataFrame) -> pd.DataFrame:
    last = frame.groupby('seed').trial.transform('max')
    return frame[frame.trial > last - FINAL_WINDOW]


def mapping_accuracy(frame: pd.DataFrame) -> pd.Series:
    """Final accuracy per run, L1 state (`context`), and color."""
    return final_trials(frame).groupby(['seed', 'context', 'sphere_color']).is_correct.mean()


def stuck(frame: pd.DataFrame) -> set:
    accuracy = mapping_accuracy(frame)
    return set(accuracy[accuracy <= STUCK_ACCURACY].index)


def main_runs(frame: pd.DataFrame) -> dict[str, float]:
    accuracy = mapping_accuracy(frame)
    is_stuck = accuracy <= STUCK_ACCURACY
    final = final_trials(frame)
    errors = final[~final.is_correct].groupby(['seed', 'context', 'sphere_color']).size()
    whole_run = frame.groupby(['seed', 'context', 'sphere_color']).is_correct.sum()
    per_run = pd.DataFrame({'accuracy': final.groupby('seed').is_correct.mean(),
                            'stuck': is_stuck.groupby(level='seed').sum()})
    slope = np.polyfit(per_run.stuck, per_run.accuracy, 1)[0]
    return {
        'mappings': len(accuracy),
        'stuck mappings': int(is_stuck.sum()),
        'stuck share of mappings (%)': 100 * is_stuck.mean(),
        'stuck share of final errors (%)': 100 * errors[errors.index.isin(accuracy[is_stuck].index)].sum() / errors.sum(),
        'stuck mappings never correct in the run': int((whole_run.loc[accuracy[is_stuck].index] == 0).sum()),
        'mappings between 10% and 90% (%)': 100 * ((accuracy > STUCK_ACCURACY) & (accuracy < .9)).mean(),
        'runs with stuck mappings': int((per_run.stuck > 0).sum()),
        'final accuracy of runs without stuck mappings (%)': 100 * per_run.accuracy[per_run.stuck == 0].mean(),
        'final accuracy change per stuck mapping (pp)': 100 * slope,
    }


def longer_runs(main: pd.DataFrame, extended: pd.DataFrame) -> dict[str, int]:
    before, after = stuck(main), stuck(extended)
    return {'stuck in the main runs': len(before), 'remaining': len(before & after),
            'resolved': len(before - after), 'new': len(after - before)}


def threshold_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Stuck mappings summed over runs; accuracies averaged over runs."""
    final = final_trials(frame)

    def averaged(rows: pd.DataFrame) -> float:
        return 100 * rows.groupby('seed').is_correct.mean().mean()

    return {
        'stuck mappings': len(stuck(frame)),
        'initial phase (%)': averaged(frame[frame.trial <= INITIAL_TRIALS]),
        'final, new task (%)': averaged(final[final.context == 'light_on']),
        'final, old task (%)': averaged(final[final.context == 'light_off']),
    }


def rounded(value: float) -> Decimal:
    """One decimal with exact halves rounded up, as in the thesis tables.

    Cutting to nine decimals first removes the binary error of values such as 58.45.
    """
    return Decimal(f'{value:.9f}').quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def report(title: str, values: dict) -> None:
    print(title)
    for name, value in values.items():
        print(f'  {name}: {rounded(value)}' if isinstance(value, float) else f'  {name}: {value}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    args = parser.parse_args()
    set_inputs(args)
    main_sources = manifest()
    extended_sources = manifest(extended=True) if INPUTS.extended_manifest else None
    for experiment in EXPERIMENTS:
        frame = load(source(main_sources, experiment, LABEL))
        report(f'== {experiment}: main runs', main_runs(frame))
        if extended_sources:
            report(f'== {experiment}: runs with a mixed phase of 4800 trials',
                   longer_runs(frame, load(source(extended_sources, experiment, LABEL))))
        if INPUTS.threshold:
            raised = load(threshold_run(experiment))
            if sorted(raised.seed.unique()) != list(THRESHOLD_SEEDS):
                raise SystemExit(f'The threshold run of {experiment} must contain seeds 1-20')
            report(f'== {experiment}: suppression threshold 0.55, seeds 1-20',
                   threshold_summary(frame[frame.seed.isin(THRESHOLD_SEEDS)]))
            report(f'== {experiment}: suppression threshold 0.65, seeds 1-20', threshold_summary(raised))


if __name__ == '__main__':
    main()
