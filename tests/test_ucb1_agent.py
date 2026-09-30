from __future__ import annotations

import random
import unittest

from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.env.sorting_task_env import Observation


class UCB1AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = ("bin1", "bin2")

    def test_default_exploration_matches_other_ucb_agents(self) -> None:
        agent = UCB1Agent(self.actions)

        self.assertEqual(agent.exploration, 0.5)

    def test_exploration_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            UCB1Agent(self.actions, exploration=-0.1)

    def test_untried_actions_are_selected_first(self) -> None:
        agent = UCB1Agent(self.actions, rng=random.Random(1))
        observation = Observation("red")

        first = agent.select_action(observation)
        agent.update(observation, first, reward=1)

        self.assertEqual(
            agent.select_action(observation),
            next(action for action in self.actions if action != first),
        )

    def test_contexts_have_separate_reward_estimates(self) -> None:
        agent = UCB1Agent(self.actions, exploration=0.0, rng=random.Random(1))
        red = Observation("red")
        blue = Observation("blue")
        for observation, rewards in ((red, (1, -1)), (blue, (-1, 1))):
            for action, reward in zip(self.actions, rewards):
                agent.update(observation, action, reward)

        self.assertEqual(agent.select_action(red), "bin1")
        self.assertEqual(agent.select_action(blue), "bin2")

    def test_evidence_is_never_forgotten(self) -> None:
        agent = UCB1Agent(self.actions)
        red = Observation("red")
        blue = Observation("blue")
        agent.update(red, "bin1", reward=1)
        for _ in range(20):
            agent.update(blue, "bin1", reward=-1)

        self.assertEqual(agent._counts[red.key]["bin1"], 1)
        self.assertEqual(agent._reward_sums[red.key]["bin1"], 1)

    def test_tie_breaking_is_reproducible(self) -> None:
        first = UCB1Agent(self.actions, rng=random.Random(7))
        second = UCB1Agent(self.actions, rng=random.Random(7))
        observation = Observation("red")

        self.assertEqual(
            [first.select_action(observation) for _ in range(10)],
            [second.select_action(observation) for _ in range(10)],
        )


if __name__ == "__main__":
    unittest.main()
