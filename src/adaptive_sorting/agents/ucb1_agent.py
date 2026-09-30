"""Tabular contextual UCB1 baseline."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import DefaultDict, Sequence

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.env.sorting_task_env import Action, Observation, ObservationKey


DEFAULT_UCB1_EXPLORATION = 0.5


class UCB1Agent(BaseAgent):
    """Apply UCB1 independently to each discrete observation context."""

    def __init__(
        self,
        action_space: Sequence[Action],
        exploration: float = DEFAULT_UCB1_EXPLORATION,
        rng: random.Random | None = None,
        name: str = "ucb1",
    ) -> None:
        super().__init__(action_space=action_space, name=name)
        if exploration < 0:
            raise ValueError("exploration must be non-negative.")
        self.exploration = exploration
        self._rng = rng or random.Random()
        self._counts: DefaultDict[ObservationKey, dict[Action, int]] = defaultdict(
            self._new_counts
        )
        self._reward_sums: DefaultDict[ObservationKey, dict[Action, int]] = defaultdict(
            self._new_counts
        )

    def select_action(self, observation: Observation) -> Action:
        counts = self._counts[observation.key]
        untried_actions = [action for action, count in counts.items() if count == 0]
        if untried_actions:
            return self._rng.choice(untried_actions)

        reward_sums = self._reward_sums[observation.key]
        context_visits = sum(counts.values())
        scores: dict[Action, float] = {}
        for action in self.action_space:
            average_reward = reward_sums[action] / counts[action]
            uncertainty = self.exploration * math.sqrt(
                2 * math.log(context_visits) / counts[action]
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
        self._counts[observation.key][action] += 1
        self._reward_sums[observation.key][action] += reward

    def _new_counts(self) -> dict[Action, int]:
        return {action: 0 for action in self.action_space}
