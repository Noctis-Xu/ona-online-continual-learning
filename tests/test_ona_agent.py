from __future__ import annotations

import os
import unittest
from unittest.mock import (
    Mock,
    call,
    patch,
)

from adaptive_sorting.agents.ona_agent import (
    _ONAProcess,
    RESPONSE_END,
    FLAT_ENCODING,
    DecisionDiagnostic,
    DEFAULT_ONA_BINARY,
    MOTOR_BABBLING_CHANCE,
    ONAAgent,
    RELATIONAL_SORTING_ENCODING,
    classify_relational_rule,
    sorting_action_argument,
    relation_fact,
)
from adaptive_sorting.env.sorting_task_env import (
    DEFAULT_TASK_RULES,
    Observation,
)
from adaptive_sorting.experiments.run_rq1 import run_seed


ONA_AVAILABLE = DEFAULT_ONA_BINARY.is_file() and os.access(DEFAULT_ONA_BINARY, os.X_OK)


class ONAProcessTests(unittest.TestCase):
    def make_process(self):
        child = Mock()
        child.poll.return_value = None
        with patch('adaptive_sorting.agents.ona_agent.subprocess.Popen', return_value=child) as launch, \
             patch('adaptive_sorting.agents.ona_agent.Thread') as thread:
            process = _ONAProcess(DEFAULT_ONA_BINARY, seed=7, timeout=5.0)
        self.assertEqual(launch.call_args.args[0], [str(DEFAULT_ONA_BINARY), 'shell', '--seed', '7'])
        thread.return_value.start.assert_called_once()
        return process, child

    def test_response_markers_keep_consecutive_commands_separate(self):
        process, child = self.make_process()
        for line in ('first response', RESPONSE_END, 'second response', RESPONSE_END):
            process._output.put(line)
        self.assertEqual(process.send_command('probe.'), ['first response'])
        self.assertEqual(process.send_command('probe?'), ['second response'])
        self.assertEqual(child.stdin.write.call_args_list, [call('probe.\n0\n'), call('probe?\n0\n')])
        self.assertEqual(child.stdin.flush.call_count, 2)
        process.close()

    def test_invalid_commands_do_not_write_to_process(self):
        process, child = self.make_process()
        for command, timeout in (('', None), ('one\ntwo', None), ('probe.', 0)):
            with self.subTest(command=command, timeout=timeout), self.assertRaises(ValueError):
                process.send_command(command, timeout=timeout)
        child.stdin.write.assert_not_called()
        process.close()
        with self.assertRaisesRegex(RuntimeError, 'not running'):
            process.send_command('probe.')

    def test_timeout_and_unexpected_exit_are_explicit(self):
        from queue import Empty
        process, child = self.make_process()
        process._output = Mock()
        process._output.get.side_effect = Empty
        with self.assertRaisesRegex(TimeoutError, 'timed out'):
            process.send_command('probe.', timeout=.01)
        process._output.get.side_effect = None
        process._output.get.return_value = None
        child.poll.side_effect = [None, 9]
        with self.assertRaisesRegex(RuntimeError, 'exited with code 9'):
            process.send_command('probe.')
        child.poll.side_effect = None
        child.poll.return_value = 9
        process.close()

    def test_close_escalates_and_is_idempotent(self):
        import subprocess
        process, child = self.make_process()
        child.wait.side_effect = [subprocess.TimeoutExpired('NAR', 1),
                                 subprocess.TimeoutExpired('NAR', 1), None]
        child.stdin.write.side_effect = BrokenPipeError
        process.close()
        process.close()
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        child.stdin.close.assert_called_once()
        child.stdout.close.assert_called_once()
        process._reader.join.assert_called_once_with(timeout=1.0)
        self.assertFalse(process.is_running)


class ONAAgentTransportTests(unittest.TestCase):
    def test_agent_flow_uses_only_transport_interface(self):
        process = Mock(spec_set=['send_command', 'close', 'is_running'])
        process.send_command.return_value = []
        with patch('adaptive_sorting.agents.ona_agent._ONAProcess', return_value=process), \
             patch('adaptive_sorting.agents.ona_agent.Path.is_file', return_value=True):
            with ONAAgent(('place_to_bin1',), seed=7, inference_cycles=12) as agent:
                self.assertEqual(process.send_command.call_args_list, [
                    call('*volume=0', timeout=20.0),
                    call(f'*motorbabbling={MOTOR_BABBLING_CHANCE}', timeout=5.0),
                    call('*babblingops=1', timeout=5.0),
                    call('*setopname 1 ^place_to_bin1', timeout=5.0),
                ])
                process.send_command.reset_mock()
                process.send_command.side_effect = [[], ['^place_to_bin1 executed with args '], [], []]
                action = agent.select_action(Observation('red'))
                agent.update(Observation('red'), action, 1)
                self.assertEqual(process.send_command.call_args_list, [
                    call('sphere_red. :|:', timeout=5.0),
                    call('correct_sorting! :|:', timeout=5.0),
                    call('correct_sorting. :|: {1.0 0.9}', timeout=5.0),
                    call('12', timeout=5.0),
                ])
        process.close.assert_called_once()

    def test_configuration_failure_closes_transport(self):
        process = Mock(spec_set=['send_command', 'close', 'is_running'])
        process.send_command.side_effect = TimeoutError('startup timed out')
        with patch('adaptive_sorting.agents.ona_agent._ONAProcess', return_value=process), \
             patch('adaptive_sorting.agents.ona_agent.Path.is_file', return_value=True):
            with self.assertRaisesRegex(TimeoutError, 'startup timed out'):
                ONAAgent(('place_to_bin1',), seed=7)
        process.close.assert_called_once()


class ONAAgentDecisionTests(unittest.TestCase):
    def test_sorting_symbols_are_explicit_and_reject_invalid_action_names(self):
        self.assertEqual(relation_fact('sphere', 'red'), '<(sphere * red) --> is>')
        self.assertEqual(sorting_action_argument('place_to_bin12'), 'bin12')
        self.assertEqual(sorting_action_argument('place_to_bin_blue2'), '(bin * blue2)')
        for action in ('place_to_bin0', 'place_to_bin01', 'place_to_bin_',
                       'place_to_bin_Red', 'place_to_bin_red-blue', 'place_to_bin1extra'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                sorting_action_argument(action)

    def test_parameterized_action_lookup_requires_exact_target(self):
        agent = object.__new__(ONAAgent)
        agent.action_space = ('place_to_bin_red', 'place_to_bin_blue')
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        self.assertIsNone(agent._parse_action(['^place executed with args ({SELF} * (bin * redder))']))
        self.assertEqual(agent._parse_action([
            '^other executed with args ({SELF} * (bin * blue))',
            '^place executed with args ({SELF} * (bin * redder))',
            '^place executed with args ({SELF} * (bin * blue))',
        ]), 'place_to_bin_blue')

    def test_relational_rule_family_ignores_condition_order_and_values(self) -> None:
        first = (
            "<(&/,<(L2 * off) --> is>,"
            "<(sphere * red) --> is>,"
            "<(L1 * on) --> is>,(^place,bin1)) =/> "
            "correct_sorting>"
        )
        second = (
            "<(&/,<(L1 * $1) --> is>,"
            "<(sphere * $2) --> is>,(^place,bin2)) =/> "
            "correct_sorting>"
        )

        self.assertEqual(classify_relational_rule(first), "color+L1+L2")
        self.assertEqual(classify_relational_rule(second), "color+L1")

    def test_shared_predicate_does_not_conflate_observation_roles(self) -> None:
        self.assertEqual(classify_relational_rule("<(L1 * on) --> is>"), "L1")
        self.assertEqual(classify_relational_rule("<(bin1 * red) --> is>"), "unclassified")
        self.assertEqual(classify_relational_rule("<(sphere * red) --> island>"), "unclassified")
        self.assertEqual(classify_relational_rule("<({SELF} * L2) --> ^place>"), "unclassified")

    def test_unrecognized_rule_family_is_explicit(self) -> None:
        self.assertEqual(
            classify_relational_rule("<((^place,bin1)) =/> correct_sorting>"),
            "unclassified",
        )

    def test_seed_must_fit_ona_rng(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
            ONAAgent(("place_to_bin1",), seed=-1)

    def test_no_decision_fails_after_one_goal(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.interaction_encoding = FLAT_ENCODING
        agent.send_command = Mock(side_effect=[[], []])

        with self.assertRaisesRegex(RuntimeError, "no operation"):
            agent.select_action(Observation("red"))

        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("sphere_red. :|:"),
                call("correct_sorting! :|:"),
            ],
        )

    def test_shell_configuration_forces_a_babbling_candidate(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.anticipation_confidence = None
        agent.decision_threshold = None
        agent.startup_timeout = 20.0
        agent.interaction_encoding = FLAT_ENCODING
        agent.send_command = Mock(return_value=[])

        agent._configure_shell()

        self.assertIn(
            call(f"*motorbabbling={MOTOR_BABBLING_CHANCE}"),
            agent.send_command.call_args_list,
        )
        self.assertEqual(
            agent.send_command.call_args_list[0],
            call("*volume=0", timeout=20.0),
        )

    def test_shell_configuration_applies_optional_learning_parameters(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.anticipation_confidence = 0.05
        agent.decision_threshold = 0.6
        agent.startup_timeout = 20.0
        agent.interaction_encoding = FLAT_ENCODING
        agent.send_command = Mock(return_value=[])

        agent._configure_shell()

        self.assertIn(
            call("*anticipationconfidence=0.05"),
            agent.send_command.call_args_list,
        )
        self.assertIn(
            call("*decisionthreshold=0.6"),
            agent.send_command.call_args_list,
        )

    def test_state_atom_uses_observed_color(self) -> None:
        self.assertEqual(
            ONAAgent._state_atom(Observation("blue")),
            "sphere_blue",
        )

    def test_context_state_atom_combines_color_and_context(self) -> None:
        self.assertEqual(
            ONAAgent._state_atom(Observation("green", "light_off")),
            "sphere_green_L1_off",
        )

    def test_color_addressed_bin_actions_use_color_targets(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin_red", "place_to_bin_green")
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(return_value=[])

        agent.configure_color_addressed_bin_actions(
            (("bin_red", "red"), ("bin_green", "green"))
        )

        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("*setoparg 1 1 (bin * red)"),
                call("*setoparg 1 2 (bin * green)"),
            ],
        )
        agent.send_command.reset_mock()
        agent.send_command.side_effect = [
            [],
            ["^place executed with args ({SELF} * (bin * green))"],
        ]

        action = agent.select_action(Observation("green"))

        self.assertEqual(action, "place_to_bin_green")
        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("<(sphere * green) --> is>. :|:"),
                call("correct_sorting! :|:"),
            ],
        )

    def test_color_addressed_bin_actions_reject_numbered_actions(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent.send_command = Mock(return_value=[])

        with self.assertRaisesRegex(ValueError, "Color-addressed actions"):
            agent.configure_color_addressed_bin_actions(
                (("bin1", "red"), ("bin2", "red"))
            )

        agent.send_command.assert_not_called()

    def test_context_state_is_sent_as_one_atom(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1",)
        agent.interaction_encoding = FLAT_ENCODING
        agent.send_command = Mock(
            side_effect=[
                [],
                ["^place_to_bin1 executed with args "],
            ]
        )

        action = agent.select_action(Observation("red", "light_on"))

        self.assertEqual(action, "place_to_bin1")
        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("sphere_red_L1_on. :|:"),
                call("correct_sorting! :|:"),
            ],
        )

    def test_relational_encoding_configures_one_parameterized_operation(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.anticipation_confidence = None
        agent.decision_threshold = None
        agent.startup_timeout = 20.0
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(return_value=[])

        agent._configure_shell()

        self.assertEqual(
            agent.send_command.call_args_list[-4:],
            [
                call("*babblingops=1"),
                call("*setopname 1 ^place"),
                call("*setoparg 1 1 bin1"),
                call("*setoparg 1 2 bin2"),
            ],
        )

    def test_relational_encoding_maps_parameterized_operation_to_action(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(
            side_effect=[
                [],
                ["^place executed with args ({SELF} * bin2)"],
            ]
        )

        action = agent.select_action(Observation("red"))

        self.assertEqual(action, "place_to_bin2")
        self.assertEqual(
            agent.last_decision_diagnostic,
            DecisionDiagnostic(source="motor_babbling"),
        )
        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("<(sphere * red) --> is>. :|:"),
                call("correct_sorting! :|:"),
            ],
        )

    def test_relational_learned_rule_records_driving_family(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1",)
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        implication = (
            "<(&/,<(sphere * red) --> is>,"
            "<(L1 * on) --> is>,(^place,bin1)) =/> "
            "correct_sorting>"
        )
        agent.send_command = Mock(
            side_effect=[
                [],
                [],
                [],
                [
                    f"decision expectation=0.800000 implication: {implication}. "
                    "Stamp=[1,2] Truth: frequency=1.000000 confidence=0.900000 "
                    "dt=0.000000 precondition: "
                    "<(L2 * off) --> is>. :|: Stamp=[3]",
                    "^place executed with args ({SELF} * bin1)",
                ],
            ]
        )

        action = agent.select_action(Observation("red", "light_on"))

        self.assertEqual(action, "place_to_bin1")
        self.assertEqual(
            agent.last_decision_diagnostic,
            DecisionDiagnostic(
                source="learned_rule",
                expectation=0.8,
                implication=implication,
                rule_family="color+L1",
            ),
        )

    def test_weak_printed_candidate_does_not_drive_babbling_action(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1",)
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(
            side_effect=[
                [],
                [
                    "decision expectation=0.550000 implication: candidate",
                    "^place executed with args ({SELF} * bin1)",
                ],
            ]
        )

        agent.select_action(Observation("red"))

        self.assertEqual(
            agent.last_decision_diagnostic,
            DecisionDiagnostic(source="motor_babbling"),
        )

    def test_malformed_decision_output_is_not_called_babbling(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1",)
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(
            side_effect=[
                [],
                [
                    "decision expectation=not-a-number implication: candidate",
                    "^place executed with args ({SELF} * bin1)",
                ],
            ]
        )

        agent.select_action(Observation("red"))

        self.assertEqual(
            agent.last_decision_diagnostic,
            DecisionDiagnostic(source="unknown"),
        )

    def test_relational_context_events_are_concurrent(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1", "place_to_bin2")
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(
            side_effect=[
                [],
                [],
                [],
                ["^place executed with args ({SELF} * bin1)"],
            ]
        )

        action = agent.select_action(Observation("red", "light_on"))

        self.assertEqual(action, "place_to_bin1")
        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("<(sphere * red) --> is>. :|:"),
                call("*concurrent"),
                call("<(L1 * on) --> is>. :|:"),
                call("correct_sorting! :|:"),
            ],
        )

    def test_relational_joint_context_includes_both_lights(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1",)
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING
        agent._set_operation_arguments(agent._build_operation_argument_map())
        agent.send_command = Mock(
            side_effect=[
                [],
                [],
                [],
                [],
                [],
                ["^place executed with args ({SELF} * bin1)"],
            ]
        )

        action = agent.select_action(
            Observation("red", "light_1_on_light_2_off")
        )

        self.assertEqual(action, "place_to_bin1")
        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("<(sphere * red) --> is>. :|:"),
                call("*concurrent"),
                call("<(L1 * on) --> is>. :|:"),
                call("*concurrent"),
                call("<(L2 * off) --> is>. :|:"),
                call("correct_sorting! :|:"),
            ],
        )

    def test_relational_encoding_rejects_incompatible_action_names(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("drop",)
        agent.interaction_encoding = RELATIONAL_SORTING_ENCODING

        with self.assertRaisesRegex(ValueError, "Unsupported sorting action"):
            agent._build_operation_argument_map()

    def test_feedback_reports_correct_sorting_as_present_or_absent(self) -> None:
        agent = object.__new__(ONAAgent)
        agent.action_space = ("place_to_bin1",)
        agent.inference_cycles = 100
        agent.send_command = Mock(return_value=[])

        observation = Observation("red")
        agent.update(observation, "place_to_bin1", reward=1)
        agent.update(observation, "place_to_bin1", reward=-1)

        self.assertEqual(
            agent.send_command.call_args_list,
            [
                call("correct_sorting. :|: {1.0 0.9}"),
                call("100"),
                call("correct_sorting. :|: {0.0 0.9}"),
                call("100"),
            ],
        )


@unittest.skipUnless(ONA_AVAILABLE, "Build the vendored NAR binary to run ONA tests.")
class ONAAgentTests(unittest.TestCase):
    def test_runtime_seed_controls_motor_babbling_sequence(self) -> None:
        def action_sequence(seed: int) -> list[str]:
            with ONAAgent(
                ("place_to_bin1", "place_to_bin2", "place_to_bin3"),
                seed=seed,
                inference_cycles=0,
            ) as agent:
                return [
                    agent.select_action(Observation("red")) for _ in range(12)
                ]

        first = action_sequence(7)

        self.assertEqual(first, action_sequence(7))
        self.assertNotEqual(first, action_sequence(8))

    def test_shell_keeps_memory_across_commands_and_closes(self) -> None:
        agent = ONAAgent(
            ("place_to_bin1", "place_to_bin2"),
            seed=7,
            inference_cycles=1,
        )

        with agent:
            operation_config = agent.send_command("*opconfig")
            agent.send_command("communication_probe.")
            query_output = agent.send_command("communication_probe?")

            self.assertTrue(agent.is_running)
            self.assertIn("*setopname 1 ^place_to_bin1", operation_config)
            self.assertTrue(
                any(line.startswith("Answer: communication_probe.") for line in query_output)
            )

            action = agent.select_action(Observation("red"))
            self.assertIn(action, agent.action_space)
            agent.update(Observation("red"), action, reward=1)
            self.assertTrue(agent.is_running)

        self.assertFalse(agent.is_running)

    def test_parameterized_operation_maps_back_to_external_action(self) -> None:
        with ONAAgent(
            ("place_to_bin1", "place_to_bin2"),
            seed=7,
            inference_cycles=1,
            interaction_encoding=RELATIONAL_SORTING_ENCODING,
        ) as agent:
            operation_config = agent.send_command("*opconfig")
            action = agent.select_action(Observation("red"))

        self.assertIn("*setopname 1 ^place", operation_config)
        self.assertIn("*setoparg 1 1 bin1", operation_config)
        self.assertIn("*setoparg 1 2 bin2", operation_config)
        self.assertIn(action, agent.action_space)

    def test_rq1_mapping_is_learned(self) -> None:
        result, rows = run_seed(
            seed=1,
            trials=150,
            final_window=50,
            config_path=DEFAULT_TASK_RULES,
            agent_factory=lambda actions, agent_seed: ONAAgent(
                actions,
                seed=agent_seed,
                binary_path=DEFAULT_ONA_BINARY,
                timeout=5.0,
                inference_cycles=100,
            ),
        )

        self.assertEqual(len(rows), 150)
        self.assertGreater(result["new_task_errors"], 0)
        self.assertGreaterEqual(result["final_new_task_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
