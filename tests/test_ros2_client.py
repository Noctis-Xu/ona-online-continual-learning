from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_sorting.execution.ros2_client import (
    Ros2SortingClient,
    _RosBindings,
)


class _FakeContext:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeNode:
    def __init__(self) -> None:
        self.services: list[str] = []
        self.destroy_calls = 0

    def create_client(self, _service_type: type, name: str) -> object:
        self.services.append(name)
        return object()

    def destroy_node(self) -> None:
        self.destroy_calls += 1


class _FakeExecutor:
    def __init__(self, *, context: _FakeContext) -> None:
        self.context = context
        self.added_nodes: list[_FakeNode] = []
        self.removed_nodes: list[_FakeNode] = []
        self.shutdown_calls = 0

    def add_node(self, node: _FakeNode) -> None:
        self.added_nodes.append(node)

    def remove_node(self, node: _FakeNode) -> None:
        self.removed_nodes.append(node)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeRclpy:
    context = SimpleNamespace(Context=_FakeContext)
    executors = SimpleNamespace(SingleThreadedExecutor=_FakeExecutor)

    def __init__(self) -> None:
        self.init_contexts: list[_FakeContext] = []
        self.node = _FakeNode()
        self.node_context: _FakeContext | None = None

    def init(self, *, context: _FakeContext) -> None:
        self.init_contexts.append(context)

    def create_node(self, _name: str, *, context: _FakeContext) -> _FakeNode:
        self.node_context = context
        return self.node


class _Service:
    class Request:
        pass


class Ros2SortingClientTests(unittest.TestCase):
    def test_owns_one_context_and_node_and_closes_idempotently(self) -> None:
        rclpy = _FakeRclpy()
        bindings = _RosBindings(
            rclpy=rclpy,
            execute_primitive=_Service,
            get_scene_config=_Service,
            get_observation=_Service,
            present_bin_color_labels=_Service,
            present_experiment_status=_Service,
            present_feedback=_Service,
            present_observation=_Service,
        )

        with patch(
            "adaptive_sorting.execution.ros2_client._load_ros_bindings",
            return_value=bindings,
        ):
            client = Ros2SortingClient(20.0)
            context = rclpy.init_contexts[0]
            client.close()
            client.close()

        self.assertIs(rclpy.node_context, context)
        self.assertEqual(len(rclpy.init_contexts), 1)
        self.assertEqual(len(rclpy.node.services), 7)
        self.assertEqual(client._executor.added_nodes, [rclpy.node])
        self.assertEqual(client._executor.removed_nodes, [rclpy.node])
        self.assertEqual(client._executor.shutdown_calls, 1)
        self.assertEqual(rclpy.node.destroy_calls, 1)
        self.assertEqual(context.shutdown_calls, 1)

    def test_rejects_non_positive_timeout_before_loading_ros(self) -> None:
        with patch(
            "adaptive_sorting.execution.ros2_client._load_ros_bindings"
        ) as load_bindings:
            with self.assertRaisesRegex(ValueError, "positive"):
                Ros2SortingClient(0.0)

        load_bindings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
