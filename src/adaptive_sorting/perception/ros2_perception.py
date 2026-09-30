"""ROS 2 metadata perception for the controlled sorting scene."""

from __future__ import annotations

from typing import Protocol

from adaptive_sorting.env.sorting_task_env import Observation
from adaptive_sorting.perception.perception_interface import PerceptionError


class _ObservationClient(Protocol):
    def observe(self) -> tuple[str, bool, bool]: ...


class Ros2Perception:
    """Read discrete color and light metadata from the ROS 2 scene."""

    def __init__(
        self,
        include_context: bool = False,
        include_irrelevant_context: bool = False,
        *,
        client: _ObservationClient,
    ) -> None:
        if include_irrelevant_context and not include_context:
            raise ValueError("include_irrelevant_context requires include_context.")
        self._include_context = include_context
        self._include_irrelevant_context = include_irrelevant_context
        self._client = client

    def observe(self) -> Observation:
        try:
            sphere_color, light_1_on, light_2_on = self._client.observe()
        except RuntimeError as exc:
            raise PerceptionError(str(exc)) from exc
        context = None
        if self._include_irrelevant_context:
            context = (
                f"light_1_{'on' if light_1_on else 'off'}_"
                f"light_2_{'on' if light_2_on else 'off'}"
            )
        elif self._include_context:
            context = "light_on" if light_1_on else "light_off"
        return Observation(sphere_color, context)
