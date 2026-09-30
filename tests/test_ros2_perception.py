from __future__ import annotations

import unittest

from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.perception.perception_interface import PerceptionError
from adaptive_sorting.perception.ros2_perception import Ros2Perception


class FakeObservationClient:
    def __init__(self, state: tuple[str, bool, bool]) -> None:
        self.state = state

    def observe(self) -> tuple[str, bool, bool]:
        return self.state

class Ros2PerceptionTests(unittest.TestCase):
    def test_observes_color_without_optional_context(self) -> None:
        perception = Ros2Perception(client=FakeObservationClient(("blue", True, False)))

        self.assertEqual(perception.observe(), Observation("blue"))

    def test_maps_relevant_light_to_context(self) -> None:
        on = Ros2Perception(
            include_context=True,
            client=FakeObservationClient(("red", True, False)),
        )
        off = Ros2Perception(
            include_context=True,
            client=FakeObservationClient(("red", False, True)),
        )

        self.assertEqual(on.observe(), Observation("red", "light_on"))
        self.assertEqual(off.observe(), Observation("red", "light_off"))

    def test_maps_both_lights_to_rq3b_context(self) -> None:
        perception = Ros2Perception(
            include_context=True,
            include_irrelevant_context=True,
            client=FakeObservationClient(("red", False, True)),
        )

        self.assertEqual(
            perception.observe(),
            Observation("red", "light_1_off_light_2_on"),
        )

    def test_irrelevant_context_requires_relevant_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires include_context"):
            Ros2Perception(
                include_irrelevant_context=True,
                client=FakeObservationClient(("red", False, True)),
            )

    def test_translates_transport_errors(self) -> None:
        class FailingClient:
            def observe(self) -> tuple[str, bool, bool]:
                raise RuntimeError("service unavailable")

        perception = Ros2Perception(client=FailingClient())
        with self.assertRaisesRegex(PerceptionError, "service unavailable"):
            perception.observe()


if __name__ == "__main__":
    unittest.main()
