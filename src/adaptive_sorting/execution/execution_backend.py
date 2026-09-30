"""Minimal interface between symbolic experiments and an execution backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from adaptive_sorting.env.sorting_task_env import Action, Observation


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of executing a selected symbolic action."""

    success: bool
    duration_seconds: float = 0.0
    planning_duration_seconds: float = 0.0
    motion_duration_seconds: float = 0.0
    trajectory_cache_hits: int = 0
    error: str | None = None


class ExecutionBackend(Protocol):
    """Present trial inputs and feedback, and execute selected symbolic actions."""

    def present(self, observation: Observation) -> None:
        """Display or otherwise prepare the current trial observation."""

    def present_mapping(self, mapping: Mapping[str, Action]) -> None:
        """Show the hidden mapping to human observers, never to perception."""

    def execute(self, action: Action) -> ExecutionResult:
        """Execute one symbolic action and report its outcome."""

    def present_feedback(self, reward: int) -> None:
        """Display task feedback after a successfully executed action."""


class NoOpBackend:
    """Skip scene presentation and report successful execution without motion."""

    def present(self, observation: Observation) -> None:
        pass

    def present_mapping(self, mapping: Mapping[str, Action]) -> None:
        pass

    def execute(self, action: Action) -> ExecutionResult:
        return ExecutionResult(success=True)

    def present_feedback(self, reward: int) -> None:
        pass


class ActionExecutionError(RuntimeError):
    """Raised when a backend cannot complete the selected action."""

    def __init__(self, action: Action, result: ExecutionResult) -> None:
        self.action = action
        self.result = result
        detail = result.error or "unspecified execution failure"
        super().__init__(f"Backend failed to execute {action!r}: {detail}.")
