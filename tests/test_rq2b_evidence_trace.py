from __future__ import annotations

from io import StringIO
import json
import unittest

from adaptive_sorting.agents.ona_agent import DEFAULT_ONA_BINARY, ONAAgent
from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.experiments.trace_rq2b_evidence import TracedONAAgent


@unittest.skipUnless(DEFAULT_ONA_BINARY.is_file(), "requires the ONA executable")
class EvidenceTraceTests(unittest.TestCase):
    def test_diagnostics_preserve_actions_truth_and_engine_time(self) -> None:
        output = StringIO()
        actions = ("place_to_bin1", "place_to_bin2")
        with ONAAgent(actions, seed=31) as plain, TracedONAAgent(
            actions, seed=31, trace=output, snapshot_interval=2,
        ) as traced:
            for i in range(12):
                observation = Observation("red" if i % 2 else "blue")
                action = plain.select_action(observation)
                self.assertEqual(action, traced.select_action(observation))
                self.assertEqual(plain.last_decision_diagnostic, traced.last_decision_diagnostic)
                reward = 1 if action == actions[i % 2] else -1
                plain.update(observation, action, reward)
                traced.update(observation, action, reward)
            self.assertEqual(plain.send_command("*stats"), traced.send_command("*stats"))
            self.assertEqual(plain.send_command("*concepts"), traced.send_command("*concepts"))
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(records), 12)
        self.assertEqual(sum("rules_after_feedback" in r for r in records), 6)
        self.assertTrue(records[-1]["rules_after_feedback"])
