"""Shared ROS 2 service client for the embodied sorting experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_sorting.execution.execution_backend import ExecutionResult


@dataclass(frozen=True)
class SortingSceneConfig:
    bin_count: int
    context_light_count: int
    show_bin_color_labels: bool
    color_target_bins: bool
    supported_colors: tuple[str, ...]


@dataclass(frozen=True)
class _RosBindings:
    rclpy: Any
    execute_primitive: type
    get_scene_config: type
    get_observation: type
    present_bin_color_labels: type
    present_experiment_status: type
    present_feedback: type
    present_observation: type


def _load_ros_bindings() -> _RosBindings:
    try:
        import rclpy
        from adaptive_sorting_bringup.srv import (
            ExecutePrimitive,
            GetSortingObservation,
            GetSortingSceneConfig,
            PresentBinColorLabels,
            PresentExperimentStatus,
            PresentFeedback,
            PresentObservation,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 and the thesis_ws overlay must be sourced before creating "
            "Ros2SortingClient."
        ) from exc

    return _RosBindings(
        rclpy=rclpy,
        execute_primitive=ExecutePrimitive,
        get_scene_config=GetSortingSceneConfig,
        get_observation=GetSortingObservation,
        present_bin_color_labels=PresentBinColorLabels,
        present_experiment_status=PresentExperimentStatus,
        present_feedback=PresentFeedback,
        present_observation=PresentObservation,
    )


class Ros2SortingClient:
    """Own one ROS context and node shared by backend and perception adapters."""

    def __init__(self, service_timeout_seconds: float = 10.0) -> None:
        if service_timeout_seconds <= 0:
            raise ValueError("service_timeout_seconds must be positive.")

        bindings = _load_ros_bindings()
        self._rclpy = bindings.rclpy
        self._types = bindings
        self._timeout = service_timeout_seconds
        self._closed = False
        self._context = self._rclpy.context.Context()
        self._rclpy.init(context=self._context)
        try:
            self._node = self._rclpy.create_node(
                "adaptive_sorting_client",
                context=self._context,
            )
            self._executor = self._rclpy.executors.SingleThreadedExecutor(
                context=self._context
            )
            self._executor.add_node(self._node)
        except Exception:
            self._context.shutdown()
            raise

        self._present_client = self._node.create_client(
            bindings.present_observation,
            "present_sorting_observation",
        )
        self._observation_client = self._node.create_client(
            bindings.get_observation,
            "observe_sorting_scene",
        )
        self._execute_client = self._node.create_client(
            bindings.execute_primitive,
            "execute_sorting_primitive",
        )
        self._scene_config_client = self._node.create_client(
            bindings.get_scene_config,
            "get_sorting_scene_config",
        )
        self._feedback_client = self._node.create_client(
            bindings.present_feedback,
            "present_sorting_feedback",
        )
        self._experiment_status_client = self._node.create_client(
            bindings.present_experiment_status,
            "present_experiment_status",
        )
        self._bin_color_labels_client = self._node.create_client(
            bindings.present_bin_color_labels,
            "present_bin_color_labels",
        )

    def present(
        self,
        sphere_color: str,
        light_1_on: bool,
        light_2_on: bool,
    ) -> None:
        request = self._types.present_observation.Request()
        request.sphere_color = sphere_color
        request.light_1_on = light_1_on
        request.light_2_on = light_2_on
        response = self._call(self._present_client, request)
        if not response.success:
            raise RuntimeError(response.error or "failed to present observation")

    def observe(self) -> tuple[str, bool, bool]:
        response = self._call(
            self._observation_client,
            self._types.get_observation.Request(),
        )
        return response.sphere_color, response.light_1_on, response.light_2_on

    def execute(self, primitive: str, sphere_color: str) -> ExecutionResult:
        request = self._types.execute_primitive.Request()
        request.primitive = primitive
        request.sphere_color = sphere_color
        try:
            response = self._call(self._execute_client, request)
        except RuntimeError as exc:
            return ExecutionResult(success=False, error=str(exc))
        return ExecutionResult(
            success=response.success,
            duration_seconds=response.duration_seconds,
            planning_duration_seconds=response.planning_duration_seconds,
            motion_duration_seconds=response.execution_duration_seconds,
            trajectory_cache_hits=int(response.cache_hit),
            error=response.error or None,
        )

    def present_bin_color_labels(self, bin_colors: tuple[str, ...]) -> None:
        request = self._types.present_bin_color_labels.Request()
        request.bin_colors = list(bin_colors)
        response = self._call(self._bin_color_labels_client, request)
        if not response.success:
            raise RuntimeError(response.error or "failed to present bin color labels")

    def get_scene_config(self) -> SortingSceneConfig:
        response = self._call(
            self._scene_config_client,
            self._types.get_scene_config.Request(),
        )
        return SortingSceneConfig(
            bin_count=response.bin_count,
            context_light_count=response.context_light_count,
            show_bin_color_labels=response.show_bin_color_labels,
            color_target_bins=response.color_target_bins,
            supported_colors=tuple(response.supported_colors),
        )

    def present_feedback(self, reward: int) -> None:
        request = self._types.present_feedback.Request()
        request.reward = reward
        response = self._call(self._feedback_client, request)
        if not response.success:
            raise RuntimeError(response.error or "failed to present feedback")

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
        request = self._types.present_experiment_status.Request()
        request.research_question = research_question
        request.agent = agent
        request.trial = trial
        request.total_trials = total_trials
        request.phase = phase
        request.total_phases = total_phases
        request.rolling_accuracy = rolling_accuracy
        request.last_reward = last_reward
        response = self._call(self._experiment_status_client, request)
        if not response.success:
            raise RuntimeError(response.error or "failed to present experiment status")

    def _call(self, client: Any, request: Any) -> Any:
        if self._closed:
            raise RuntimeError("ROS 2 sorting client is closed")
        if not client.wait_for_service(timeout_sec=self._timeout):
            raise RuntimeError(
                f"ROS 2 service unavailable: {client.srv_name}. "
                "Start ./ros2/start-sorting-backend.sh "
                "<rq1|rq2a|rq2b|rq3a|rq3b|rq4a> "
                "in another terminal and keep it running."
            )
        future = client.call_async(request)
        self._executor.spin_until_future_complete(
            future,
            timeout_sec=self._timeout,
        )
        if not future.done():
            raise RuntimeError(f"ROS 2 service timed out: {client.srv_name}")
        exception = future.exception()
        if exception is not None:
            raise RuntimeError(str(exception)) from exception
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.remove_node(self._node)
            self._executor.shutdown()
            self._node.destroy_node()
        finally:
            self._context.shutdown()

    def __enter__(self) -> Ros2SortingClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
