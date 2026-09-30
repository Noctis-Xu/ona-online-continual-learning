from __future__ import annotations

import random
import unittest

from adaptive_sorting.agents.sliding_window_ucb_agent import SlidingWindowUCBAgent
from adaptive_sorting.env.sorting_task_env import Observation


class SlidingWindowUCBAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = ("bin1", "bin2")

    def test_default_parameters_use_balanced_exploration(self) -> None:
        agent = SlidingWindowUCBAgent(self.actions)

        self.assertEqual(agent.window_size, 300)
        self.assertEqual(agent.exploration, 0.5)

    def test_parameters_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            SlidingWindowUCBAgent(self.actions, window_size=0)
        with self.assertRaises(ValueError):
            SlidingWindowUCBAgent(self.actions, exploration=-0.1)

    def test_untried_actions_are_selected_before_ucb_scores(self) -> None:
        agent = SlidingWindowUCBAgent(self.actions, rng=random.Random(1))
        observation = Observation("red")

        first = agent.select_action(observation)
        agent.update(observation, first, reward=1)

        self.assertEqual(agent.select_action(observation), next(
            action for action in self.actions if action != first
        ))

    def test_contexts_have_separate_action_values(self) -> None:
        agent = SlidingWindowUCBAgent(
            self.actions, exploration=0.0, rng=random.Random(1)
        )
        red = Observation("red")
        blue = Observation("blue")
        for observation, rewards in ((red, (1, -1)), (blue, (-1, 1))):
            for action, reward in zip(self.actions, rewards):
                agent.update(observation, action, reward)

        self.assertEqual(agent.select_action(red), "bin1")
        self.assertEqual(agent.select_action(blue), "bin2")

    def test_window_expires_records_by_global_trial_order(self) -> None:
        agent = SlidingWindowUCBAgent(self.actions, window_size=3)
        red = Observation("red")
        blue = Observation("blue")
        agent.update(red, "bin1", 1)
        agent.update(blue, "bin1", 1)
        agent.update(blue, "bin2", -1)
        agent.update(blue, "bin1", 1)

        self.assertEqual(len(agent._history), 3)
        self.assertNotIn(red.key, [item.context for item in agent._history])

    def test_tie_breaking_is_reproducible(self) -> None:
        first = SlidingWindowUCBAgent(self.actions, rng=random.Random(7))
        second = SlidingWindowUCBAgent(self.actions, rng=random.Random(7))
        observation = Observation("red")

        self.assertEqual(
            [first.select_action(observation) for _ in range(10)],
            [second.select_action(observation) for _ in range(10)],
        )


if __name__ == "__main__":
    unittest.main()
