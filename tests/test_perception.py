from __future__ import annotations

import unittest

from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.perception import GroundTruthPerception


class GroundTruthPerceptionTests(unittest.TestCase):
    def test_reads_discrete_observation_from_controlled_source(self) -> None:
        expected = Observation("blue", "light_on")
        perception = GroundTruthPerception(lambda: expected)

        self.assertEqual(perception.observe(), expected)

    def test_reads_source_again_for_each_trial(self) -> None:
        observations = iter((Observation("red"), Observation("green")))
        perception = GroundTruthPerception(lambda: next(observations))

        self.assertEqual(perception.observe(), Observation("red"))
        self.assertEqual(perception.observe(), Observation("green"))


if __name__ == "__main__":
    unittest.main()
