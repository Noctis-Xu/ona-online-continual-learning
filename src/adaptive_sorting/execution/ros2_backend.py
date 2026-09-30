"""ROS 2 implementation of the symbolic execution boundary."""

from __future__ import annotations

import re
from collections import deque
from typing import Mapping, Protocol

from adaptive_sorting.env.sorting_task_env import (
    Action,
    Observation,
    context_light_states,
)
from adaptive_sorting.execution.ros2_client import SortingSceneConfig
from adaptive_sorting.execution.execution_backend import ExecutionResult


# Physical target labels for the default ROS scene, not an ordering of actions.
SCENE_TARGET_COLORS = ("red", "orange", "yellow", "green", "blue", "indigo", "violet")
_PLACE_ACTION = re.compile(r"place_to_bin[1-7]")
_COLOR_PLACE_ACTION = re.compile(r"place_to_bin_([a-z]+)")


class _PrimitiveClient(Protocol):
    def present(
        self,
        sphere_color: str,
        light_1_on: bool,
        light_2_on: bool,
    ) -> None: ...

    def present_bin_color_labels(self, bin_colors: tuple[str, ...]) -> None: ...

    def get_scene_config(self) -> SortingSceneConfig: ...

    def execute(self, primitive: str, sphere_color: str) -> ExecutionResult: ...

    def present_feedback(self, reward: int) -> None: ...

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
    ) -> None: ...

class Ros2Backend:
    """Present the scene and feedback, and expand bin actions into ROS 2 primitives."""

    def __init__(
        self,
        client: _PrimitiveClient,
    ) -> None:
        self._client = client
        self._sphere_color: str | None = None
        self._last_bin_colors: tuple[str, ...] | None = None
        self._bin_labels_fixed = False
        self._research_question: str | None = None
        self._agent_name: str | None = None
        self._total_trials = 0
        self._trial = 0
        self._phase_lengths: tuple[int, ...] = ()
        self._last_reward = 0
        self._recent_results: deque[bool] = deque(maxlen=50)

    def configure_experiment_status(
        self,
        research_question: str,
        agent: str,
        total_trials: int,
        rolling_window: int = 50,
        phase_lengths: tuple[int, ...] | None = None,
    ) -> None:
        if not research_question or not agent:
            raise ValueError("research_question and agent must not be empty.")
        if total_trials <= 0 or rolling_window <= 0:
            raise ValueError("total_trials and rolling_window must be positive.")
        resolved_phase_lengths = phase_lengths or (total_trials,)
        if any(length <= 0 for length in resolved_phase_lengths):
            raise ValueError("phase lengths must be positive.")
        if sum(resolved_phase_lengths) != total_trials:
            raise ValueError("phase lengths must sum to total_trials.")
        self._research_question = research_question.upper()
        self._agent_name = agent
        self._total_trials = total_trials
        self._trial = 0
        self._phase_lengths = resolved_phase_lengths
        self._last_reward = 0
        self._recent_results = deque(maxlen=rolling_window)
        self._publish_status()

    def present_mapping(self, mapping: Mapping[str, Action]) -> None:
        """Show ground-truth labels in RViz without exposing them to perception."""

        if self._bin_labels_fixed:
            return
        if mapping and all(_COLOR_PLACE_ACTION.fullmatch(action) for action in mapping.values()):
            # Color-target bins are drawn by the scene and need no annotation.
            return
        bin_colors = _bin_colors_from_mapping(mapping)
        if bin_colors == self._last_bin_colors:
            return
        self._client.present_bin_color_labels(bin_colors)
        self._last_bin_colors = bin_colors

    def fix_bin_color_labels(self, mapping: Mapping[str, Action]) -> None:
        """Show experiment-wide labels once and ignore later phase mappings."""

        self.present_mapping(mapping)
        self._bin_labels_fixed = True

    def require_scene_capabilities(
        self,
        expected_bins: int,
        required_colors: tuple[str, ...],
        expected_context_lights: int = 0,
        expected_color_target_bins: bool = False,
    ) -> SortingSceneConfig:
        """Fail before motion when the backend cannot run this experiment."""

        config = self._client.get_scene_config()
        if config.bin_count != expected_bins:
            raise RuntimeError(
                f"ROS 2 scene has {config.bin_count} bins; this demo requires "
                f"{expected_bins}. Restart the backend with the matching RQ profile."
            )
        if config.context_light_count != expected_context_lights:
            raise RuntimeError(
                f"ROS 2 scene shows {config.context_light_count} context lights; "
                f"this demo requires {expected_context_lights}. Restart the backend "
                "with the matching RQ profile."
            )
        if config.color_target_bins != expected_color_target_bins:
            required = "color-target" if expected_color_target_bins else "numbered"
            raise RuntimeError(
                f"This demo requires {required} bins. Restart the backend with "
                "the matching RQ profile."
            )
        unsupported = sorted(set(required_colors) - set(config.supported_colors))
        if unsupported:
            raise RuntimeError(
                "ROS 2 scene does not support experiment colors: "
                + ", ".join(unsupported)
            )
        return config

    def present(self, observation: Observation) -> None:
        light_1_on, light_2_on = context_light_states(observation.context)
        self._client.present(
            observation.sphere_color,
            light_1_on,
            light_2_on,
        )
        self._sphere_color = observation.sphere_color
        if self._research_question is not None:
            self._trial += 1
            self._publish_status()

    def execute(self, action: Action) -> ExecutionResult:
        color_target = _COLOR_PLACE_ACTION.fullmatch(action)
        if not (_PLACE_ACTION.fullmatch(action) or
                (color_target and color_target.group(1) in SCENE_TARGET_COLORS)):
            return ExecutionResult(success=False, error=f"unknown action: {action}")
        if self._sphere_color is None:
            return ExecutionResult(
                success=False,
                error="present() must be called before execute()",
            )

        duration = 0.0
        planning_duration = 0.0
        motion_duration = 0.0
        cache_hits = 0
        for primitive in ("pick", action, "return_home"):
            result = self._client.execute(primitive, self._sphere_color)
            duration += result.duration_seconds
            planning_duration += result.planning_duration_seconds
            motion_duration += result.motion_duration_seconds
            cache_hits += result.trajectory_cache_hits
            if not result.success:
                detail = result.error or "unspecified execution failure"
                return ExecutionResult(
                    success=False,
                    duration_seconds=duration,
                    planning_duration_seconds=planning_duration,
                    motion_duration_seconds=motion_duration,
                    trajectory_cache_hits=cache_hits,
                    error=f"{primitive}: {detail}",
                )
        return ExecutionResult(
            success=True,
            duration_seconds=duration,
            planning_duration_seconds=planning_duration,
            motion_duration_seconds=motion_duration,
            trajectory_cache_hits=cache_hits,
        )

    def present_feedback(self, reward: int) -> None:
        self._client.present_feedback(reward)
        if self._research_question is not None:
            self._last_reward = reward
            self._recent_results.append(reward > 0)
            self._publish_status()

    def _publish_status(self) -> None:
        if self._research_question is None or self._agent_name is None:
            return
        rolling_accuracy = (
            sum(self._recent_results) / len(self._recent_results)
            if self._recent_results
            else -1.0
        )
        self._client.present_experiment_status(
            self._research_question,
            self._agent_name,
            self._trial,
            self._total_trials,
            self._current_phase(),
            len(self._phase_lengths),
            rolling_accuracy,
            self._last_reward,
        )

    def _current_phase(self) -> int:
        completed = 0
        for phase, length in enumerate(self._phase_lengths, start=1):
            completed += length
            if self._trial <= completed:
                return phase
        return len(self._phase_lengths)


def _bin_colors_from_mapping(mapping: Mapping[str, Action]) -> tuple[str, ...]:
    colors_by_bin: dict[int, str] = {}
    for sphere_color, action in mapping.items():
        if not _PLACE_ACTION.fullmatch(action):
            raise ValueError(f"Unsupported bin action in task mapping: {action!r}.")
        bin_number = int(action.removeprefix("place_to_bin"))
        if bin_number in colors_by_bin:
            raise ValueError(
                f"Task mapping assigns multiple colors to bin {bin_number}."
            )
        colors_by_bin[bin_number] = sphere_color

    highest_bin = max(colors_by_bin, default=0)
    return tuple(
        colors_by_bin.get(bin_number, "")
        for bin_number in range(1, highest_bin + 1)
    )
