"""Minimal interface between a sorting scene and the learning agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from adaptive_sorting.env.sorting_task_env import Observation


class PerceptionInterface(Protocol):
    """Convert the current scene state into one discrete observation."""

    def observe(self) -> Observation:
        """Return the sphere color and optional context visible to the agent."""


class GroundTruthPerception:
    """Read controlled simulation metadata without modeling vision errors."""

    def __init__(self, observation_source: Callable[[], Observation]) -> None:
        self._observation_source = observation_source

    def observe(self) -> Observation:
        return self._observation_source()


class PerceptionError(RuntimeError):
    """Raised when a perception backend cannot produce an observation."""
