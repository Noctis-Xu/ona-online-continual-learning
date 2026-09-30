from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from adaptive_sorting.agents.ona_agent import RELATIONAL_SORTING_ENCODING
from adaptive_sorting.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from adaptive_sorting.agents.sliding_window_ucb_agent import SlidingWindowUCBAgent
from adaptive_sorting.agents.ucb1_agent import UCB1Agent
from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.experiments.agent_factory import (
    agent_metadata,
    build_agent_factory,
)


class AgentFactoryTests(unittest.TestCase):
    def test_ona_factory_uses_runtime_parameters(self) -> None:
        args = argparse.Namespace(
            agent="flat_ona",
            ona_binary="unused",
            ona_timeout=5.0,
            ona_startup_timeout=20.0,
            ona_cycles=50,
            ona_anticipation_confidence=0.05,
            ona_decision_threshold=0.6,
        )

        with patch(
            "adaptive_sorting.experiments.agent_factory.binary_provenance",
            return_value={"binary_path": "unused", "binary_sha256": "abc"},
        ):
            metadata = agent_metadata(args)

        self.assertEqual(
            metadata,
            {
                "encoding_version": "compact-symbols-v2",
                "protocol_version": "unified-v3",
                "agent": "flat_ona",
                "binary_path": "unused",
                "binary_sha256": "abc",
                "command_timeout_seconds": 5.0,
                "startup_timeout_seconds": 20.0,
                "inference_cycles": 50,
                "anticipation_confidence": 0.05,
                "decision_threshold": 0.6,
            },
        )

    def test_epsilon_greedy_factory_is_reproducible(self) -> None:
        args = argparse.Namespace(
            agent="epsilon_greedy",
            epsilon=1.0,
            learning_rate=0.2,
        )
        factory = build_agent_factory(args)
        actions = ("place_to_bin1", "place_to_bin2")
        first = factory(actions, 7)
        second = factory(actions, 7)
        observation = Observation("red")

        self.assertIsInstance(first, EpsilonGreedyAgent)
        self.assertEqual(
            [first.select_action(observation) for _ in range(10)],
            [second.select_action(observation) for _ in range(10)],
        )
        self.assertEqual(
            agent_metadata(args),
            {"encoding_version": "compact-symbols-v2", "protocol_version": "unified-v3", "agent": "epsilon_greedy", "epsilon": 1.0, "learning_rate": 0.2},
        )

    def test_relational_ona_factory_enables_relational_encoding(self) -> None:
        args = argparse.Namespace(
            agent="relational_ona",
            ona_binary="unused",
            ona_timeout=5.0,
            ona_startup_timeout=20.0,
            ona_cycles=50,
            ona_anticipation_confidence=0.05,
            ona_decision_threshold=0.6,
        )

        with patch(
            "adaptive_sorting.experiments.agent_factory.ONAAgent"
        ) as ona_agent:
            factory = build_agent_factory(args)
            factory(("place_to_bin1", "place_to_bin2"), 7)

        ona_agent.assert_called_once_with(
            ("place_to_bin1", "place_to_bin2"),
            seed=7,
            binary_path="unused",
            timeout=5.0,
            startup_timeout=20.0,
            inference_cycles=50,
            anticipation_confidence=0.05,
            decision_threshold=0.6,
            interaction_encoding=RELATIONAL_SORTING_ENCODING,
            name="relational_ona",
        )
        with patch(
            "adaptive_sorting.experiments.agent_factory.binary_provenance",
            return_value={"binary_path": "unused", "binary_sha256": "abc"},
        ):
            metadata = agent_metadata(args)

        self.assertEqual(
            metadata,
            {
                "encoding_version": "compact-symbols-v2",
                "protocol_version": "unified-v3",
                "agent": "relational_ona",
                "binary_path": "unused",
                "binary_sha256": "abc",
                "command_timeout_seconds": 5.0,
                "startup_timeout_seconds": 20.0,
                "inference_cycles": 50,
                "anticipation_confidence": 0.05,
                "decision_threshold": 0.6,
                "interaction_encoding": RELATIONAL_SORTING_ENCODING,
            },
        )

    def test_sw_ucb_factory_uses_shared_parameters(self) -> None:
        args = argparse.Namespace(
            agent="sw_ucb",
            ucb_window=120,
            ucb_exploration=0.75,
        )
        factory = build_agent_factory(args)
        actions = ("place_to_bin1", "place_to_bin2")
        first = factory(actions, 7)
        second = factory(actions, 7)
        observation = Observation("red")

        self.assertIsInstance(first, SlidingWindowUCBAgent)
        self.assertEqual(
            [first.select_action(observation) for _ in range(10)],
            [second.select_action(observation) for _ in range(10)],
        )
        self.assertEqual(
            agent_metadata(args),
            {
                "encoding_version": "compact-symbols-v2",
                "protocol_version": "unified-v3",
                "agent": "sw_ucb",
                "ucb_window": 120,
                "ucb_exploration": 0.75,
            },
        )

    def test_ucb1_factory_uses_shared_parameters(self) -> None:
        args = argparse.Namespace(agent="ucb1", ucb1_exploration=0.6)
        factory = build_agent_factory(args)
        actions = ("place_to_bin1", "place_to_bin2")
        first = factory(actions, 7)
        second = factory(actions, 7)
        observation = Observation("red")

        self.assertIsInstance(first, UCB1Agent)
        self.assertEqual(
            [first.select_action(observation) for _ in range(10)],
            [second.select_action(observation) for _ in range(10)],
        )
        self.assertEqual(
            agent_metadata(args),
            {"encoding_version": "compact-symbols-v2", "protocol_version": "unified-v3", "agent": "ucb1", "ucb1_exploration": 0.6},
        )

if __name__ == "__main__":
    unittest.main()
