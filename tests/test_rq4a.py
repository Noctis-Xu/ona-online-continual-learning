from __future__ import annotations

from adaptive_sorting.analysis.rq4_relational_diagnostics import _write_rq4_markdown_report

import csv
import math
from pathlib import Path
import random
import tempfile
import unittest

from adaptive_sorting.agents.sliding_window_ucb_agent import SlidingWindowUCBAgent
from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.analysis.plot_rq4a import (
    TrialRecord,
    load_trials,
    plot_report,
    rolling_accuracy,
)
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    load_task_rules,
)
from adaptive_sorting.experiments.run_rq4a import (
    RQ4SeedDiagnostics,
    classify_generalization_rule,
    observable_bin_colors,
    parse_args,
    run_seed,
    validate_rq4_rules,
)


CONFIG_PATH = DEFAULT_TASK_RULES


class RQ4Tests(unittest.TestCase):
    def test_explicit_binary_is_preserved_for_both_encodings(self) -> None:
        for agent in ("flat_ona", "relational_ona"):
            args = parse_args(["--agent", agent, "--ona-binary", "custom/NAR-c256-t240"])
            self.assertEqual(args.ona_binary, Path("custom/NAR-c256-t240"))

    def test_relational_report_tracks_feedback_free_target_rule_use(self) -> None:
        target_rule = (
            "<(<(sphere * #1) --> is> &/ "
            "<({SELF} * (bin * #1)) --> ^place>) =/> correct_sorting>"
        )
        diagnostics = [
            RQ4SeedDiagnostics(
                seed=1,
                decisions=[
                    {
                        "phase": "known_only",
                        "is_novel": False,
                        "sphere_color": "red",
                        "is_correct": True,
                        "decision_source": "learned_rule",
                        "driving_rule_class": "target_match_generalization",
                    },
                    {
                        "phase": "expanded_inputs",
                        "is_novel": True,
                        "sphere_color": "indigo",
                        "is_correct": True,
                        "decision_source": "learned_rule",
                        "driving_rule_class": "target_match_generalization",
                    },
                    {
                        "phase": "expanded_inputs",
                        "is_novel": True,
                        "sphere_color": "indigo",
                        "is_correct": False,
                        "decision_source": "motor_babbling",
                        "driving_rule_class": "grounded",
                    },
                ],
                snapshots={"known_only": (target_rule,), "expanded_inputs": ()},
            )
        ]
        for trial, row in enumerate(diagnostics[0].decisions, 1):
            row.update(seed=1, trial=trial)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.md"
            _write_rq4_markdown_report(report, diagnostics, {"indigo", "violet"})
            content = report.read_text(encoding="utf-8")

        self.assertIn("indigo | 1/1 (100.0%) | 1/1 (100.0%)", content)
        self.assertIn("retained before expansion: 1/1 seeds", content)

    def test_diagnostic_report_uses_configured_novel_colors(self) -> None:
        diagnostics = [RQ4SeedDiagnostics(seed=1, decisions=[
            dict(seed=1, trial=i, phase='expanded_inputs', is_novel=True,
                 sphere_color=color, is_correct=True, decision_source='motor_babbling',
                 driving_rule_class='grounded')
            for i, color in enumerate(('cyan', 'magenta'), 1)
        ])]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.md'
            _write_rq4_markdown_report(report, diagnostics, {'cyan', 'magenta'})
            content = report.read_text()
        self.assertIn('cyan | 1/1 (100.0%)', content)
        self.assertIn('magenta | 1/1 (100.0%)', content)
        self.assertNotIn('indigo', content)
        self.assertIn('not the early or final evaluation windows', content)

    def test_ucb_agents_run_across_input_expansion(self) -> None:
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
                    seed=1,
                    base_trials=300,
                    expanded_trials=600,

                    final_window=200,
                    config_path=CONFIG_PATH,
                    agent_factory=factory,
                )

                self.assertEqual(len(rows), 900)
                self.assertEqual(len({row["selected_action"] for row in rows}), 7)
                self.assertIsNotNone(result["first_indigo_accuracy"])
                self.assertIsNotNone(result["first_violet_accuracy"])
                self.assertGreater(result["old_task_n"], 0)

    def test_rules_keep_fixed_actions_and_only_add_expected_colors(self) -> None:
        rules = load_task_rules(CONFIG_PATH)

        novel = validate_rq4_rules(rules["novel_input_base"], rules["expanded_inputs"])

        self.assertEqual(novel, {"indigo", "violet"})
        self.assertEqual(
            observable_bin_colors(rules["expanded_inputs"])[-2:],
            (("bin6", "indigo"), ("bin7", "violet")),
        )

    def test_color_target_task_uses_shared_named_actions_without_layout_observations(self):
        from adaptive_sorting.env.sorting_task_env import Observation
        from adaptive_sorting.agents.ona_agent import ONAAgent
        captured = []

        class RecordingBandit(UCB1Agent):
            def select_action(self, observation):
                captured.append(observation)
                return super().select_action(observation)

        result, rows = run_seed(
            seed=1, base_trials=200, expanded_trials=1200,
            final_window=200,
            config_path=CONFIG_PATH,
            agent_factory=lambda actions, seed: RecordingBandit(actions, rng=random.Random(seed)),
            bin_colors_visible=True,
        )
        self.assertEqual(len(rows), 1400)
        self.assertTrue(all(row['correct_action'] == 'place_to_bin_' + row['sphere_color'] for row in rows))
        self.assertTrue(all(row['selected_action'].startswith('place_to_bin_') for row in rows))
        for obs in captured:
            self.assertEqual(obs.key, ONAAgent._state_atom(obs))
        self.assertEqual(Observation('red', 'light_on').key, 'sphere_red_L1_on')
        self.assertEqual(Observation('red', 'light_1_on_light_2_off').key, 'sphere_red_L1_on_L2_off')

    def test_generalized_relational_rules_are_classified_for_diagnostics(self) -> None:
        rule = (
            "<(&/,<(sphere * #1) --> is>,"
            "<(#2 * #1) --> is>,"
            "<({SELF} * #2) --> ^place>) =/> correct_sorting>"
        )

        self.assertEqual(
            classify_generalization_rule(rule),
            "target_match_generalization",
        )
        color_addressed_rule = (
            "<(<(sphere * #1) --> is> &/ "
            "<({SELF} * (bin * #1)) --> ^place>) =/> correct_sorting>"
        )
        self.assertEqual(
            classify_generalization_rule(color_addressed_rule),
            "target_match_generalization",
        )
        self.assertEqual(
            classify_generalization_rule(
                "<(&/,<(sphere * red) --> is>,(^place,bin1)) "
                "=/> correct_sorting>"
            ),
            "grounded",
        )


    def test_trial_window_curves_use_the_same_trial_range(self) -> None:
        records = [
            TrialRecord(1, 1, True, False, 1),
            TrialRecord(1, 2, True, False, 1),
            TrialRecord(1, 3, False, True, 1),
            TrialRecord(1, 4, False, True, 1),
        ]

        self.assertEqual(rolling_accuracy(records, 2, "overall")[-1], 1.0)
        self.assertEqual(rolling_accuracy(records, 2, "known")[-1], 1.0)
        self.assertTrue(math.isnan(rolling_accuracy(records, 2, "novel")[-1]))

    def test_plot_leaves_novel_curve_empty_before_first_novel_input(self) -> None:
        rows = [
            [1, 1, False, False, 4],
            [1, 2, False, True, 4],
            [1, 3, False, True, 4],
            [1, 4, True, False, 4],
            [1, 5, False, True, 4],
            [1, 6, True, True, 4],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "trials.csv"
            output_path = Path(temp_dir) / "report.png"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "seed",
                        "trial",
                        "is_novel",
                        "is_correct",
                        "expanded_phase_start",
                    ]
                )
                writer.writerows(rows)

            grouped = load_trials(csv_path)
            novel_curve = rolling_accuracy(grouped[1], 2, "novel")
            plot_report(grouped, output_path, rolling_window=2)

            self.assertTrue(all(value != value for value in novel_curve[:3]))
            self.assertEqual(novel_curve[-1], 1.0)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
