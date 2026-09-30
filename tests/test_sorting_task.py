from __future__ import annotations

import random
import unittest

from adaptive_sorting.agents.epsilon_greedy_agent import EpsilonGreedyAgent
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    Observation,
    SortingTaskEnv,
    context_light_states,
    load_task_rules,
)


class SortingTaskEnvironmentTests(unittest.TestCase):
    def test_context_labels_resolve_both_light_states(self) -> None:
        self.assertEqual(context_light_states("light_on"), (True, False))
        self.assertEqual(context_light_states("light_off"), (False, False))
        self.assertEqual(
            context_light_states("light_1_off_light_2_on"),
            (False, True),
        )

    def test_initial_rule_uses_five_sphere_to_bin_actions(self) -> None:
        env = SortingTaskEnv(load_task_rules(DEFAULT_TASK_RULES), rng=random.Random(1))

        observation = env.reset()

        self.assertIn(
            observation.sphere_color,
            ("red", "orange", "yellow", "green", "blue"),
        )
        self.assertIsNone(observation.context)
        self.assertEqual(len(env.action_space), 5)
        result = env.step(env.correct_action(observation))
        self.assertTrue(result.is_correct)
        self.assertEqual(result.reward, 1)

    def test_context_mapping_changes_every_color(self) -> None:
        rules = load_task_rules(DEFAULT_TASK_RULES)
        light_off = rules["context_light_off"]
        light_on = rules["context_light_on"]

        self.assertEqual(light_off.context, "light_off")
        self.assertEqual(light_on.context, "light_on")
        self.assertEqual(light_off.mapping.keys(), light_on.mapping.keys())
        self.assertEqual(light_off.action_space, light_on.action_space)
        for sphere_color, off_action in light_off.mapping.items():
            self.assertNotEqual(off_action, light_on.mapping[sphere_color])

    def test_local_update_rules_change_exactly_two_three_and_four_colors(self) -> None:
        rules = load_task_rules(DEFAULT_TASK_RULES)
        initial = rules["initial"]
        expected = {
            "local_update_after": {"red", "blue"},
            "local_update_3_after": {"red", "yellow", "blue"},
            "local_update_4_after": {"red", "orange", "green", "blue"},
        }

        for rule_name, expected_colors in expected.items():
            updated = rules[rule_name]
            changed_colors = {
                sphere_color
                for sphere_color, initial_action in initial.mapping.items()
                if updated.mapping[sphere_color] != initial_action
            }
            self.assertEqual(changed_colors, expected_colors)

    def test_novel_input_rules_keep_seven_actions_and_add_two_colors(self) -> None:
        rules = load_task_rules(DEFAULT_TASK_RULES)
        initial = rules["novel_input_base"]
        expanded = rules["expanded_inputs"]

        self.assertEqual(len(initial.mapping), 5)
        self.assertEqual(len(initial.action_space), 7)
        self.assertEqual(len(expanded.mapping), 7)
        self.assertEqual(expanded.action_space, initial.action_space)
        for sphere_color, action in initial.mapping.items():
            self.assertEqual(expanded.mapping[sphere_color], action)
        self.assertEqual(expanded.mapping["indigo"], "place_to_bin6")
        self.assertEqual(expanded.mapping["violet"], "place_to_bin7")


class EpsilonGreedyAgentTests(unittest.TestCase):
    def test_agent_uses_recent_feedback_for_rule_updates(self) -> None:
        actions = ("place_to_bin1", "place_to_bin2")
        observation = Observation("red")
        agent = EpsilonGreedyAgent(
            actions,
            epsilon=0.0,
            learning_rate=0.5,
            rng=random.Random(9),
        )

        agent.update(observation, "place_to_bin1", 1)
        agent.update(observation, "place_to_bin1", -1)

        self.assertEqual(agent._q_values[observation.key]["place_to_bin1"], -0.25)

if __name__ == "__main__":
    unittest.main()
