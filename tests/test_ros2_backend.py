from __future__ import annotations

import unittest
from typing import Mapping

from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.execution.execution_backend import ExecutionResult
from adaptive_sorting.execution.ros2_client import SortingSceneConfig
from adaptive_sorting.execution.ros2_backend import Ros2Backend


class FakePrimitiveClient:
    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        self.presented: list[tuple[str, bool, bool]] = []
        self.primitives: list[tuple[str, str]] = []
        self.rewards: list[int] = []
        self.bin_color_labels: list[tuple[str, ...]] = []
        self.statuses: list[tuple[str, str, int, int, int, int, float, int]] = []

    def present(
        self,
        sphere_color: str,
        light_1_on: bool,
        light_2_on: bool,
    ) -> None:
        self.presented.append((sphere_color, light_1_on, light_2_on))

    def execute(self, primitive: str, sphere_color: str) -> ExecutionResult:
        self.primitives.append((primitive, sphere_color))
        if primitive == self.failure:
            return ExecutionResult(
                success=False,
                duration_seconds=0.5,
                planning_duration_seconds=0.1,
                motion_duration_seconds=0.3,
                error="planned failure",
            )
        return ExecutionResult(
            success=True,
            duration_seconds=1.0,
            planning_duration_seconds=0.2,
            motion_duration_seconds=0.7,
            trajectory_cache_hits=1,
        )

    def present_bin_color_labels(self, bin_colors: tuple[str, ...]) -> None:
        self.bin_color_labels.append(bin_colors)

    def get_scene_config(self) -> SortingSceneConfig:
        return SortingSceneConfig(
            bin_count=5,
            context_light_count=0,
            show_bin_color_labels=True,
            color_target_bins=False,
            supported_colors=("red", "orange", "yellow", "green", "blue"),
        )

    def present_feedback(self, reward: int) -> None:
        self.rewards.append(reward)

    def present_experiment_status(
        self,
        research_question: str,
        agent: str,
        trial: int,
        total_trials: int,
        phase: int,
        total_phases: int,
        rolling_accuracy: float,
        last_reward: int,
    ) -> None:
        self.statuses.append(
            (
                research_question,
                agent,
                trial,
                total_trials,
                phase,
                total_phases,
                rolling_accuracy,
                last_reward,
            )
        )

class ColorTargetExecutionTests(unittest.TestCase):
    def test_selected_color_target_is_forwarded_without_correcting_from_sphere(self):
        client = FakePrimitiveClient()
        backend = Ros2Backend(client)
        backend.present(Observation('red'))
        result = backend.execute('place_to_bin_violet')
        self.assertTrue(result.success)
        self.assertEqual(client.primitives, [('pick', 'red'),
                         ('place_to_bin_violet', 'red'), ('return_home', 'red')])


class Ros2BackendTests(unittest.TestCase):
    def test_publishes_compact_experiment_status_each_trial(self) -> None:
        client = FakePrimitiveClient()
        backend = Ros2Backend(client=client)
        backend.configure_experiment_status(
            "rq2a",
            "ONA",
            750,
            rolling_window=2,
            phase_lengths=(1, 749),
        )

        backend.present(Observation("red"))
        backend.present_feedback(1)
        backend.present(Observation("blue"))
        backend.present_feedback(-1)

        self.assertEqual(
            client.statuses[0], ("RQ2A", "ONA", 0, 750, 1, 2, -1.0, 0)
        )
        self.assertEqual(
            client.statuses[-1], ("RQ2A", "ONA", 2, 750, 2, 2, 0.5, -1)
        )

    def test_rejects_phase_lengths_that_do_not_cover_the_run(self) -> None:
        backend = Ros2Backend(client=FakePrimitiveClient())

        with self.assertRaisesRegex(ValueError, "sum to total_trials"):
            backend.configure_experiment_status(
                "rq2a", "ONA", 10, phase_lengths=(4, 5)
            )

    def test_validates_scene_capabilities_before_motion(self) -> None:
        backend = Ros2Backend(client=FakePrimitiveClient())

        config = backend.require_scene_capabilities(5, ("red", "blue"))

        self.assertEqual(config.bin_count, 5)
        with self.assertRaisesRegex(RuntimeError, "requires 7"):
            backend.require_scene_capabilities(7, ("red", "blue"))
        with self.assertRaisesRegex(RuntimeError, "violet"):
            backend.require_scene_capabilities(5, ("red", "violet"))
        with self.assertRaisesRegex(RuntimeError, "requires 2"):
            backend.require_scene_capabilities(
                5,
                ("red", "blue"),
                expected_context_lights=2,
            )
        with self.assertRaisesRegex(RuntimeError, "color-target"):
            backend.require_scene_capabilities(
                5,
                ("red", "blue"),
                expected_color_target_bins=True,
            )

    def test_presents_changed_mapping_only_and_preserves_empty_bins(self) -> None:
        client = FakePrimitiveClient()
        backend = Ros2Backend(client=client)
        mapping: Mapping[str, str] = {
            "red": "place_to_bin1",
            "blue": "place_to_bin3",
        }

        backend.present_mapping(mapping)
        backend.present_mapping(mapping)
        backend.present_mapping({"red": "place_to_bin3", "blue": "place_to_bin1"})

        self.assertEqual(
            client.bin_color_labels,
            [("red", "", "blue"), ("blue", "", "red")],
        )

    def test_fixed_labels_ignore_later_phase_mappings(self) -> None:
        client = FakePrimitiveClient()
        backend = Ros2Backend(client=client)

        backend.fix_bin_color_labels({"red": "place_to_bin1", "violet": "place_to_bin2"})
        backend.present_mapping({"red": "place_to_bin1"})

        self.assertEqual(client.bin_color_labels, [("red", "violet")])

    def test_color_target_mapping_sends_no_annotation(self) -> None:
        client = FakePrimitiveClient()
        backend = Ros2Backend(client=client)

        backend.present_mapping({"red": "place_to_bin_red"})

        self.assertEqual(client.bin_color_labels, [])

    def test_rejects_non_bin_actions_in_mapping(self) -> None:
        backend = Ros2Backend(client=FakePrimitiveClient())

        with self.assertRaisesRegex(ValueError, "Unsupported bin action"):
            backend.present_mapping({"red": "inspect_sphere"})

    def test_presents_joint_context_and_executes_primitive_sequence(self) -> None:
        client = FakePrimitiveClient()
        backend = Ros2Backend(client=client)

        backend.present(Observation("blue", "light_1_on_light_2_off"))
        result = backend.execute("place_to_bin5")
        backend.present_feedback(1)

        self.assertEqual(client.presented, [("blue", True, False)])
        self.assertEqual(
            client.primitives,
            [
                ("pick", "blue"),
                ("place_to_bin5", "blue"),
                ("return_home", "blue"),
            ],
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.duration_seconds, 3.0)
        self.assertAlmostEqual(result.planning_duration_seconds, 0.6)
        self.assertAlmostEqual(result.motion_duration_seconds, 2.1)
        self.assertEqual(result.trajectory_cache_hits, 3)
        self.assertEqual(client.rewards, [1])

    def test_stops_sequence_when_a_primitive_fails(self) -> None:
        client = FakePrimitiveClient(failure="place_to_bin2")
        backend = Ros2Backend(client=client)
        backend.present(Observation("orange"))

        result = backend.execute("place_to_bin2")

        self.assertFalse(result.success)
        self.assertEqual(result.duration_seconds, 1.5)
        self.assertIn("place_to_bin2: planned failure", result.error or "")
        self.assertEqual(
            client.primitives,
            [("pick", "orange"), ("place_to_bin2", "orange")],
        )

    def test_rejects_actions_outside_fixed_action_space(self) -> None:
        client = FakePrimitiveClient()
        backend = Ros2Backend(client=client)
        backend.present(Observation("red", "light_off"))

        result = backend.execute("inspect_sphere")

        self.assertFalse(result.success)
        self.assertIn("unknown action", result.error or "")
        self.assertEqual(client.primitives, [])

    def test_requires_present_before_execute(self) -> None:
        client = FakePrimitiveClient()
        result = Ros2Backend(client=client).execute("place_to_bin1")

        self.assertFalse(result.success)
        self.assertIn("present()", result.error or "")

if __name__ == "__main__":
    unittest.main()
