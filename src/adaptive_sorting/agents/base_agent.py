"""Common interface for sorting-task agents."""

from __future__ import annotations

from typing import Sequence

from adaptive_sorting.env.sorting_task_env import Action, Observation


class BaseAgent:
    """Minimal action-selection interface."""

    def __init__(self, action_space: Sequence[Action], name: str | None = None) -> None:
        actions = tuple(action_space)
        if not actions:
            raise ValueError("Agent action space must not be empty.")
        if len(set(actions)) != len(actions):
            raise ValueError("Agent action space must not contain duplicates.")
        self.action_space = actions
        self.name = name or self.__class__.__name__

    def select_action(self, observation: Observation) -> Action:
        raise NotImplementedError

    def update(self, observation: Observation, action: Action, reward: int) -> None:
        """Stateless agents use this default no-op implementation."""

    def close(self) -> None:
        """Agents without external resources use this default no-op."""

    def __enter__(self) -> "BaseAgent":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
