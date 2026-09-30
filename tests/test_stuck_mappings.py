import argparse
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from adaptive_sorting.analysis import stuck_mappings
from adaptive_sorting.analysis.thesis_figures import add_input_arguments, manifest, set_inputs, source, threshold_run


def run_frame(seed: int, accuracy: dict[tuple[str, str], list[bool]]) -> pd.DataFrame:
    """One run whose final trials cycle through the given (context, color) outcomes."""
    rows = []
    trial = 0
    outcomes = {key: iter(values) for key, values in accuracy.items()}
    while outcomes:
        for key in list(outcomes):
            value = next(outcomes[key], None)
            if value is None:
                del outcomes[key]
                continue
            trial += 1
            rows.append({'seed': seed, 'trial': trial, 'context': key[0], 'sphere_color': key[1], 'is_correct': value})
    return pd.DataFrame(rows)


class StuckMappingTests(unittest.TestCase):
    def test_stuck_mappings_use_the_final_window_of_each_run(self):
        # 700 alternating trials: red is correct early in the run but never in the final 300.
        frame = run_frame(1, {('light_off', 'red'): [True] * 200 + [False] * 150,
                              ('light_off', 'blue'): [True] * 350})
        self.assertEqual(stuck_mappings.stuck(frame), {(1, 'light_off', 'red')})

    def test_main_run_summary_counts_errors_on_stuck_mappings(self):
        frame = pd.concat([
            run_frame(1, {('light_off', 'red'): [False] * 20, ('light_on', 'red'): [True] * 20}),
            run_frame(2, {('light_off', 'red'): [True] * 20, ('light_on', 'red'): [True] * 20}),
        ])
        summary = stuck_mappings.main_runs(frame)
        self.assertEqual(summary['mappings'], 4)
        self.assertEqual(summary['stuck mappings'], 1)
        self.assertEqual(summary['stuck share of final errors (%)'], 100)
        self.assertEqual(summary['stuck mappings never correct in the run'], 1)
        self.assertEqual(summary['final accuracy of runs without stuck mappings (%)'], 100)

    def test_longer_runs_report_remaining_resolved_and_new_mappings(self):
        main = run_frame(1, {('light_off', 'red'): [False] * 10, ('light_on', 'blue'): [False] * 10})
        longer = run_frame(1, {('light_off', 'red'): [False] * 10, ('light_on', 'blue'): [True] * 10,
                               ('light_on', 'green'): [False] * 10})
        self.assertEqual(stuck_mappings.longer_runs(main, longer),
                         {'stuck in the main runs': 2, 'remaining': 1, 'resolved': 1, 'new': 1})

    def test_threshold_accuracy_is_averaged_over_runs(self):
        # Run 1 has twice as many final L1-on trials as run 2; averaging over runs weighs them equally.
        frame = pd.concat([run_frame(1, {('light_on', 'red'): [True] * 4}),
                           run_frame(2, {('light_on', 'red'): [False] * 2})])
        self.assertEqual(stuck_mappings.threshold_summary(frame)['final, new task (%)'], 50)


class RoundingTests(unittest.TestCase):
    def test_exact_halves_round_up_despite_binary_error(self):
        self.assertEqual(str(stuck_mappings.rounded(58.45)), '58.5')
        self.assertEqual(str(stuck_mappings.rounded(19.15)), '19.2')
        self.assertEqual(str(stuck_mappings.rounded(99.97)), '100.0')
        self.assertEqual(str(stuck_mappings.rounded(80.725)), '80.7')


class InputTests(unittest.TestCase):
    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        add_input_arguments(parser)
        set_inputs(parser.parse_args(list(argv)))

    def test_manifest_paths_resolve_from_the_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sources.json'
            path.write_text(json.dumps({'rq3a': {'sources': {'ONA (relational)': 'runs/trials.csv'}}}))
            self.parse('--manifest', str(path))
            self.assertEqual(source(manifest(), 'rq3a', 'ONA (relational)'),
                             (Path(directory) / 'runs/trials.csv').resolve())
            with self.assertRaisesRegex(SystemExit, 'no Epsilon-greedy run'):
                source(manifest(), 'rq3a', 'Epsilon-greedy')

    def test_missing_inputs_name_the_required_option(self):
        self.parse()
        with self.assertRaisesRegex(SystemExit, '--manifest'):
            manifest()
        with self.assertRaisesRegex(SystemExit, '--threshold-rq3b'):
            threshold_run('rq3b')


if __name__ == '__main__':
    unittest.main()
