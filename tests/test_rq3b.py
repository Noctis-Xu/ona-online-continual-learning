from __future__ import annotations

import random
import unittest

from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES
from adaptive_sorting.experiments.run_rq3a import (
    L2_RANDOM,
    irrelevant_light_schedule,
    run_seed,
)


class RQ3bTests(unittest.TestCase):
    @staticmethod
    def _factory(actions: tuple[str, ...], agent_seed: int) -> UCB1Agent:
        return UCB1Agent(actions, rng=random.Random(agent_seed))

    def test_irrelevant_light_schedule_is_iid_and_reproducible(self) -> None:
        first = irrelevant_light_schedule(100, random.Random(3))
        second = irrelevant_light_schedule(100, random.Random(3))
        different = irrelevant_light_schedule(100, random.Random(4))

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(set(first), {False, True})
        self.assertTrue(any(first[index] == first[index + 1] for index in range(99)))

    def test_rq3b_changes_only_the_paired_l2_presentation(self) -> None:
        common = {
            "seed": 3,
            "before_trials": 20,
            "mixed_trials": 40,
            "context_block_size": 10,
            "final_window": 20,
            "config_path": DEFAULT_TASK_RULES,
            "agent_factory": self._factory,
        }
        _, rq3_rows = run_seed(**common)
        _, rq3b_rows = run_seed(**common, light_2_mode=L2_RANDOM)

        base_schedule = [
            (row["context"], row["sphere_color"]) for row in rq3_rows
        ]
        self.assertEqual(
            base_schedule,
            [(row["context"], row["sphere_color"]) for row in rq3b_rows],
        )
        self.assertTrue(
            all(
                row["joint_context"]
                == f"light_1_{row['light_1']}_light_2_{row['light_2']}"
                for row in rq3b_rows
            )
        )
        self.assertEqual({row["light_2"] for row in rq3b_rows}, {"off", "on"})
        self.assertNotIn("light_2", rq3_rows[0])


if __name__ == "__main__":
    unittest.main()
