from __future__ import annotations

import csv
import math
from pathlib import Path
import random
import tempfile
import unittest

from adaptive_sorting.agents.sliding_window_ucb_agent import SlidingWindowUCBAgent
from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.analysis.plot_rq2a import (
    TrialRecord,
    load_trials,
    plot_report,
    rolling_accuracy,
)
from adaptive_sorting.analysis.plot_rq2a_summary import (
    load_summary,
    plot_comparison,
)
from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES
from adaptive_sorting.experiments.run_rq2a import (
    run_seed,
)


class RQ2MetricsTests(unittest.TestCase):
    def test_ucb_agents_run_across_local_rule_update(self) -> None:
        factories = {
            "ucb1": lambda actions, agent_seed: UCB1Agent(
                actions, rng=random.Random(agent_seed)
            ),
            "sw_ucb": lambda actions, agent_seed: SlidingWindowUCBAgent(
                actions, rng=random.Random(agent_seed)
            ),
        }
        for name, factory in factories.items():
            with self.subTest(agent=name):
                result, rows = run_seed(
                    change_size=2,
                    seed=1,
                    before_trials=150,
                    after_trials=600,
                    final_window=50,
                    config_path=DEFAULT_TASK_RULES,
                    agent_factory=factory,
                )

                self.assertEqual(len(rows), 750)
                self.assertEqual(
                    {row["phase"] for row in rows},
                    {"before_update", "after_update"},
                )
                self.assertIsNotNone(result["new_task_errors"])
                self.assertGreater(result["old_task_n"], 0)

    def test_trial_window_excludes_old_changed_observations(self) -> None:
        records = [
            TrialRecord(2, 1, 1, True, False, 1),
            TrialRecord(2, 1, 2, True, False, 1),
            TrialRecord(2, 1, 3, False, True, 1),
            TrialRecord(2, 1, 4, False, True, 1),
        ]

        self.assertTrue(
            math.isnan(rolling_accuracy(records, 2, changed=True)[-1])
        )


    def test_new_task_and_old_task_plot_is_generated(self) -> None:
        rows = [
            [2, 1, 1, True, False, 3],
            [2, 1, 2, False, True, 3],
            [2, 1, 3, True, False, 3],
            [2, 1, 4, False, True, 3],
            [2, 2, 1, False, True, 3],
            [2, 2, 2, True, False, 3],
            [2, 2, 3, False, True, 3],
            [2, 2, 4, True, True, 3],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "trials.csv"
            output_path = Path(temp_dir) / "report.png"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "change_size",
                        "seed",
                        "trial",
                        "is_changed",
                        "is_correct",
                        "change_point",
                    ]
                )
                writer.writerows(rows)

            grouped = load_trials(csv_path, change_size=2)
            changed_curve = rolling_accuracy(grouped[1], window=2, changed=True)
            plot_report(grouped, output_path, rolling_window=2)

            self.assertEqual(changed_curve[0], 0.0)
            self.assertEqual(changed_curve[2], 0.0)
            self.assertEqual([changed_curve[1], changed_curve[3]], [0.0, 0.0])
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_change_size_comparison_plot_is_generated(self) -> None:
        rows = [
            [2, 1, 200, 1.0, 1.0, 3, 80, 60, True, 1200],
            [2, 2, "", 1.0, 1.0, 4, 90, 70, True, 1200],
            [3, 1, 250, 0.95, 1.0, 5, 120, 90, True, 1200],
            [3, 2, 270, 1.0, 1.0, 6, 130, "", False, 1200],
            [4, 1, 300, 0.9, 1.0, 7, 160, 110, True, 1200],
            [4, 2, 320, 0.95, 1.0, 8, 170, 120, True, 1200],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "summary.csv"
            output_path = Path(temp_dir) / "comparison.png"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "change_size",
                        "seed",
                        "unused_extra_field",
                        "final_new_task_accuracy",
                        "final_old_task_accuracy",
                        "old_task_errors",
                        "new_task_errors",
                        "trials_to_criterion",
                        "criterion_reached",
                        "criterion_horizon",
                    ]
                )
                writer.writerows(rows)

            grouped = load_summary(csv_path)
            plot_comparison(grouped, output_path)

            self.assertEqual(sorted(grouped), [2, 3, 4])
            self.assertEqual(grouped[2][1]["new_task_errors"], 90)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)


    def test_cli_finishes_with_dictionary_metrics(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from adaptive_sorting.experiments.run_rq2a import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            argv = ['run_rq2a', '--agent', 'ucb1', '--seeds', '1', '--workers', '1',
                    '--change-sizes', '2', '--before-trials', '20', '--after-trials', '20',
                    '--final-window', '10', '--output-dir', str(output)]
            with patch('sys.argv', argv), patch('builtins.print') as printed:
                main()
            self.assertTrue((output / 'trials.csv').is_file())
            self.assertTrue(any(str(call.args[0]).startswith('final_overall_accuracy=')
                                for call in printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
