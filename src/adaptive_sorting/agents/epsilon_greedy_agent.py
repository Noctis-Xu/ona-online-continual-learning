"""Tabular epsilon-greedy contextual bandit baseline."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import DefaultDict, Sequence

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.env.sorting_task_env import Action, Observation, ObservationKey

DEFAULT_EPSILON = 0.05
DEFAULT_LEARNING_RATE = 0.2


class EpsilonGreedyAgent(BaseAgent):
    """Epsilon-greedy contextual bandit with incremental value estimates."""

    def __init__(
        self,
        action_space: Sequence[Action],
        epsilon: float = DEFAULT_EPSILON,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        rng: random.Random | None = None,
        name: str = "epsilon_greedy",
    ) -> None:
        super().__init__(action_space=action_space, name=name)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._rng = rng or random.Random()
        self._q_values: DefaultDict[ObservationKey, dict[Action, float]] = defaultdict(
            self._new_action_values
        )

    def select_action(self, observation: Observation) -> Action:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.action_space)

        values = self._q_values[observation.key]
        best_value = max(values.values())
        best_actions = [action for action, value in values.items() if value == best_value]
        return self._rng.choice(best_actions)

    def update(self, observation: Observation, action: Action, reward: int) -> None:
        state = observation.key
        values = self._q_values[state]
        values[action] += self.learning_rate * (reward - values[action])

    def _new_action_values(self) -> dict[Action, float]:
        return {action: 0.0 for action in self.action_space}
