from __future__ import annotations

import random
import unittest

from adaptive_sorting.agents.sliding_window_ucb_agent import SlidingWindowUCBAgent
from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES
from adaptive_sorting.experiments.run_rq1 import run_seed


class RQ1Tests(unittest.TestCase):
    def test_ucb_agents_learn_initial_mapping(self) -> None:
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
                    trials=300,
                    final_window=50,
                    config_path=DEFAULT_TASK_RULES,
                    agent_factory=factory,
                )

                self.assertEqual(len(rows), 300)
                self.assertGreaterEqual(result["final_overall_accuracy"], 0.9)
                self.assertGreaterEqual(result["final_new_task_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
