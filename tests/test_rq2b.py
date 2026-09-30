from __future__ import annotations

from pathlib import Path
import random
import tempfile
import unittest

from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.analysis.plot_rq2b import (
    load_trials,
    plot_report,
)
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    load_task_rules,
)
from adaptive_sorting.experiments.result_paths import write_experiment_results
from adaptive_sorting.experiments.run_rq2b import (
    PHASE_RULES,
    changed_colors,
    run_seed,
)


class RQ2bTests(unittest.TestCase):
    def test_each_transition_changes_two_colors(self) -> None:
        rules = load_task_rules(DEFAULT_TASK_RULES)
        changes = [
            changed_colors(rules[before], rules[after])
            for before, after in zip(PHASE_RULES, PHASE_RULES[1:])
        ]
        self.assertTrue(all(len(colors) == 2 for colors in changes))


    def test_one_agent_is_preserved_across_all_phases(self) -> None:
        factory = lambda actions, agent_seed: UCB1Agent(
            actions, rng=random.Random(agent_seed)
        )
        results, rows = run_seed(
            seed=1,
            initial_trials=40,
            phase_trials=40,
            final_window=40,


            config_path=DEFAULT_TASK_RULES,
            agent_factory=factory,
        )
        self.assertEqual(len(results), len(PHASE_RULES))
        self.assertEqual(len(rows), 40 * len(PHASE_RULES))
        self.assertEqual([result["phase_index"] for result in results], list(range(1, 10)))

    def test_report_is_generated(self) -> None:
        factory = lambda actions, agent_seed: UCB1Agent(
            actions, rng=random.Random(agent_seed)
        )
        results, rows = run_seed(
            seed=1,
            initial_trials=40,
            phase_trials=40,
            final_window=40,


            config_path=DEFAULT_TASK_RULES,
            agent_factory=factory,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            trial_path, _ = write_experiment_results(run_dir, results, rows)
            output_path = run_dir / "report.png"
            plot_report(load_trials(trial_path), output_path, "UCB1", rolling_window=10)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
