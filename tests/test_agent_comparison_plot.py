from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest

from adaptive_sorting.analysis.plot_agent_comparison import (
    curve_summary,
    plot_rq1,
    plot_rq2b,
    plot_rq4b,
)


class AgentComparisonPlotTests(unittest.TestCase):
    def test_curve_alignment_rejects_missing_trials_and_seeds(self):
        from adaptive_sorting.analysis.plot_style import validate_curves
        from adaptive_sorting.analysis.plot_rq1 import TrialRecord
        valid = [TrialRecord(1, i, True) for i in (1, 2, 3)]
        with self.assertRaisesRegex(ValueError, 'Missing, duplicate'):
            validate_curves({'A': {1: [valid[0], valid[2]]}})
        with self.assertRaisesRegex(ValueError, 'Seed sets differ'):
            validate_curves({'A': {1:valid}, 'B': {2:valid}})
        with self.assertRaisesRegex(ValueError, 'horizons'):
            validate_curves({'A': {1:valid}, 'B': {1:valid[:2]}})

    def test_summary_keeps_missing_agent_positions_visible(self):
        from unittest.mock import patch
        from adaptive_sorting.analysis import summarize_agent_comparison as module
        from adaptive_sorting.analysis.plot_style import plt
        labels = ['A', 'B', 'C']
        data = [module.MetricSummary(label, 'evaluation_phase', metric,
                .5 if label == 'A' else None, .4 if label == 'A' else None,
                .6 if label == 'A' else None, 2 if label == 'A' else 0, 2,
                100 if metric.censored_median else None)
                for label in labels for metric in module.METRICS_BY_RQ['rq1']]
        with patch.object(module, 'finish_figure') as finish:
            module.plot_summary(Path(tempfile.gettempdir()) / 'thesis-table-test.png', 'rq1', [(label,Path()) for label in labels], data)
            fig = finish.call_args.args[0]
            for axis in fig.axes:
                texts = [cell.get_text().get_text() for cell in axis.tables[0].get_celld().values()]
                self.assertEqual(texts.count('NA'), 2)
                self.assertTrue(all(label in texts for label in labels))
            plt.close(fig)

    def test_rq1_comparison_plot_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            trial_paths = []
            for name, results in (
                ("ona.csv", [False, True, True, True]),
                ("epsilon_greedy.csv", [False, False, True, True]),
                ("ucb1.csv", [False, True, True, True]),
                ("sw_ucb.csv", [False, True, False, True]),
            ):
                path = temp_path / name
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["seed", "trial", "is_correct"])
                    writer.writerows(
                        [1, trial, is_correct]
                        for trial, is_correct in enumerate(results, start=1)
                    )
                trial_paths.append(path)

            output = temp_path / "comparison.png"
            plot_rq1(
                [
                    ("ONA", trial_paths[0]),
                    ("Epsilon-greedy", trial_paths[1]),
                    ("Contextual UCB1", trial_paths[2]),
                    ("Contextual SW-UCB", trial_paths[3]),
                ],
                output,
                window=2,
            )

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_curve_summary_preserves_missing_seed_coverage(self) -> None:
        average, lower, upper = curve_summary(
            [[0.0, 0.5, float("nan")], [1.0, 1.0, 0.75]]
        )

        self.assertEqual(average[:2], [0.5, 0.75])
        self.assertEqual(average[2], .75)
        self.assertTrue(math.isnan(lower[2]))
        self.assertTrue(math.isnan(upper[2]))
        self.assertTrue(all(0 <= value <= 1 for value in lower[:2] + upper[:2]))

    def test_curve_summary_keeps_empty_points_and_bootstraps_available_seeds(self):
        nan = float('nan')
        average, lower, upper = curve_summary([[nan, 0, 1], [nan, 1, 1], [nan, nan, 0]])
        self.assertTrue(math.isnan(average[0]))
        self.assertTrue(math.isnan(lower[0]))
        self.assertEqual(average[1], .5)
        self.assertEqual((lower[1], upper[1]), (0, 1))
        self.assertAlmostEqual(average[2], 2/3)

    def test_rq2b_comparison_plot_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trial_paths = []
            for agent, results in (
                ("ONA", [False, True, True, False, True, True]),
                ("Contextual UCB1", [False, False, True, True, True, True]),
            ):
                path = root / f"{agent}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "seed",
                            "trial",
                            "phase_index",
                            "phase_label",
                            "phase_start",
                            "is_correct",
                        ],
                    )
                    writer.writeheader()
                    for trial, correct in enumerate(results, start=1):
                        second_phase = trial > 3
                        writer.writerow(
                            {
                                "seed": 1,
                                "trial": trial,
                                "phase_index": 2 if second_phase else 1,
                                "phase_label": "B" if second_phase else "A",
                                "phase_start": 4 if second_phase else 1,
                                "is_correct": correct,
                            }
                        )
                trial_paths.append((agent, path))

            output = root / "rq2b-comparison.png"
            plot_rq2b(trial_paths, output, window=2)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_rq4b_compares_flat_and_relational_trial_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trial_paths = []
            for agent, results in (
                ("ONA", [True, True, False, True]),
                ("Relational ONA", [False, True, True, True]),
            ):
                path = root / f"{agent}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "seed",
                            "trial",
                            "is_novel",
                            "is_correct",
                            "expanded_phase_start",
                        ],
                    )
                    writer.writeheader()
                    for trial, correct in enumerate(results, start=1):
                        writer.writerow(
                            {
                                "seed": 1,
                                "trial": trial,
                                "is_novel": trial >= 3,
                                "is_correct": correct,
                                "expanded_phase_start": 3,
                            }
                        )
                trial_paths.append((agent, path))

            output = root / "rq4b-comparison.png"
            plot_rq4b(trial_paths, output, window=2)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
