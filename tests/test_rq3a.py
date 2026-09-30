from __future__ import annotations

import csv
import gzip
from pathlib import Path
import random
import tempfile
import unittest

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.agents.ona_agent import (
    DecisionDiagnostic,
    ONAAgent,
    RELATIONAL_SORTING_ENCODING,
)
from adaptive_sorting.agents.sliding_window_ucb_agent import SlidingWindowUCBAgent
from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.analysis.plot_rq3a import (
    TrialRecord,
    load_trials,
    plot_report,
    rolling_accuracy,
    stable_switch_rates,
    switch_cost_curve,
)
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    Observation,
)
from adaptive_sorting.experiments.run_rq3a import (
    balanced_context_schedule,
    RQ3SeedDiagnostics,
    parse_args,
    run_seed,
)


class RQ3Tests(unittest.TestCase):
    def test_explicit_binary_is_preserved_for_both_encodings(self) -> None:
        for agent in ("flat_ona", "relational_ona"):
            args = parse_args(["--agent", agent, "--ona-binary", "custom/NAR-c256-t240"])
            self.assertEqual(args.ona_binary, Path("custom/NAR-c256-t240"))

    def test_relational_diagnostics_capture_interval_and_phase_end_snapshots(self) -> None:
        class FakeRelationalONA(ONAAgent):
            def __init__(self, actions: tuple[str, ...]) -> None:
                BaseAgent.__init__(self, actions)
                self.interaction_encoding = RELATIONAL_SORTING_ENCODING
                self._last_decision_diagnostic = None

            def select_action(self, observation: Observation) -> str:
                self._last_decision_diagnostic = DecisionDiagnostic(
                    source="learned_rule",
                    expectation=0.8,
                    implication="<(sphere * red) --> is>",
                    rule_family="color",
                )
                return self.action_space[0]

            def update(
                self,
                observation: Observation,
                action: str,
                reward: int,
            ) -> None:
                pass

            def retained_operation_rules(self) -> tuple[str, ...]:
                return ("dt=2 <(<(sphere * red) --> is> &/ <({SELF} * bin1) --> ^place>) =/> correct_sorting>. {1.000000 0.900000}",)

            def close(self) -> None:
                pass

        diagnostics = RQ3SeedDiagnostics(seed=1)
        memory_dir = Path(tempfile.mkdtemp())
        _, rows = run_seed(
            seed=1,
            before_trials=4,
            mixed_trials=8,
            context_block_size=4,
            final_window=4,
            snapshot_interval=5,
            memory_dir=memory_dir,
            config_path=DEFAULT_TASK_RULES,
            agent_factory=lambda actions, _: FakeRelationalONA(tuple(actions)),
            diagnostics=diagnostics,
        )

        self.assertEqual(diagnostics.snapshot_trials, [4, 5, 10, 12])
        with gzip.open(memory_dir / "seed_0001_family_stats.csv.gz", "rt", newline="") as handle:
            self.assertEqual(sorted({int(row["trial"]) for row in csv.DictReader(handle)}), list(range(1, 13)))
        self.assertEqual(len(diagnostics.decisions), 12)
        self.assertTrue(all(row["decision_source"] == "learned_rule" for row in rows))

    def test_ucb_agents_run_across_context_switching(self) -> None:
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
                    before_trials=150,
                    mixed_trials=600,
                    context_block_size=10,
                    final_window=200,
                    config_path=DEFAULT_TASK_RULES,
                    agent_factory=factory,
                )

                self.assertEqual(len(rows), 750)
                self.assertEqual(
                    {row["context"] for row in rows}, {"light_off", "light_on"}
                )
                self.assertIsNotNone(result["new_task_errors"])
                self.assertEqual(result["new_task_n"] + result["old_task_n"], 600)

    def test_trial_windows_align_context_and_switch_cost_samples(self) -> None:
        records = [
            TrialRecord(1, 1, "mixed_contexts", "light_on", True, False, 1),
            TrialRecord(1, 2, "mixed_contexts", "light_off", True, True, 1),
            TrialRecord(1, 3, "mixed_contexts", "light_off", False, True, 1),
            TrialRecord(1, 4, "mixed_contexts", "light_on", True, False, 1),
        ]

        self.assertEqual(rolling_accuracy(records, 2, "light_on")[-1], 0.0)
        trial_cost = switch_cost_curve(records, 3)
        self.assertTrue(all(value != value for value in trial_cost[:2]))
        self.assertEqual(trial_cost[-1], 0.5)

    def test_ona_state_contains_color_and_context(self) -> None:
        state = ONAAgent._state_atom(Observation("red", "light_on"))

        self.assertEqual(state, "sphere_red_L1_on")

    def test_context_schedule_is_balanced_per_block_and_reproducible(self) -> None:
        first = balanced_context_schedule(40, 10, random.Random(3))
        second = balanced_context_schedule(40, 10, random.Random(3))

        self.assertEqual(first, second)
        for start in range(0, 40, 10):
            block = first[start : start + 10]
            self.assertEqual(block.count("light_off"), 5)
            self.assertEqual(block.count("light_on"), 5)


    def test_context_report_is_generated(self) -> None:
        rows = [
            [1, 1, "off_only", "light_off", False, False, 5],
            [1, 2, "off_only", "light_off", False, True, 5],
            [1, 3, "off_only", "light_off", False, True, 5],
            [1, 4, "off_only", "light_off", False, True, 5],
            [1, 5, "mixed_contexts", "light_on", True, False, 5],
            [1, 6, "mixed_contexts", "light_off", True, True, 5],
            [1, 7, "mixed_contexts", "light_off", False, True, 5],
            [1, 8, "mixed_contexts", "light_on", True, True, 5],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "trials.csv"
            output_path = Path(temp_dir) / "report.png"
            switch_output_path = Path(temp_dir) / "switch_cost.png"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "seed",
                        "trial",
                        "phase",
                        "context",
                        "context_switched",
                        "is_correct",
                        "mixed_phase_start",
                    ]
                )
                writer.writerows(rows)

            grouped = load_trials(csv_path)
            on_curve = rolling_accuracy(grouped[1], 2, "light_on")
            switch_rate, stay_rate = stable_switch_rates(grouped, final_window=4)
            cost_curve = switch_cost_curve(grouped[1], 2)
            plot_report(grouped, output_path, rolling_window=2, final_window=4)

            self.assertTrue(on_curve[0] != on_curve[0])
            self.assertEqual(on_curve[-1], 1.0)
            self.assertEqual(switch_rate, 2 / 3)
            self.assertEqual(stay_rate, 1.0)
            self.assertEqual(cost_curve[-1], 0.0)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)
            self.assertTrue(switch_output_path.is_file())
            self.assertGreater(switch_output_path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
