"""Tabular contextual Sliding-Window UCB baseline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Sequence

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.env.sorting_task_env import Action, Observation, ObservationKey


DEFAULT_UCB_WINDOW = 300
DEFAULT_UCB_EXPLORATION = 0.5


@dataclass(frozen=True)
class _Experience:
    context: ObservationKey
    action: Action
    reward: int


class SlidingWindowUCBAgent(BaseAgent):
    """Contextual UCB using the most recent global trials as finite memory."""

    def __init__(
        self,
        action_space: Sequence[Action],
        window_size: int = DEFAULT_UCB_WINDOW,
        exploration: float = DEFAULT_UCB_EXPLORATION,
        rng: random.Random | None = None,
        name: str = "sw_ucb",
    ) -> None:
        super().__init__(action_space=action_space, name=name)
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if exploration < 0:
            raise ValueError("exploration must be non-negative.")
        self.window_size = window_size
        self.exploration = exploration
        self._rng = rng or random.Random()
        self._history: deque[_Experience] = deque(maxlen=window_size)

    def select_action(self, observation: Observation) -> Action:
        rewards_by_action = {action: [] for action in self.action_space}
        for experience in self._history:
            if experience.context == observation.key:
                rewards_by_action[experience.action].append(experience.reward)

        untried_actions = [
            action for action, rewards in rewards_by_action.items() if not rewards
        ]
        if untried_actions:
            return self._rng.choice(untried_actions)

        context_visits = sum(len(rewards) for rewards in rewards_by_action.values())
        scores: dict[Action, float] = {}
        for action, rewards in rewards_by_action.items():
            average_reward = sum(rewards) / len(rewards)
            uncertainty = self.exploration * math.sqrt(
                2 * math.log(context_visits) / len(rewards)
            )
            scores[action] = average_reward + uncertainty

        best_score = max(scores.values())
        best_actions = [
            action for action, score in scores.items() if score == best_score
        ]
        return self._rng.choice(best_actions)

    def update(self, observation: Observation, action: Action, reward: int) -> None:
        if action not in self.action_space:
            raise ValueError(f"Action {action!r} is outside the agent action space.")
        self._history.append(_Experience(observation.key, action, reward))
