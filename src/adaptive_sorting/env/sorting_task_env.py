"""Symbolic environment for the robotic sorting experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Mapping

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML installed.
    yaml = None


Action = str
ObservationKey = str
DEFAULT_TASK_RULES = Path(__file__).resolve().parents[1] / "configs" / "task_rules.yaml"
LIGHT_CONTEXT_PATTERN = re.compile(r"light_1_(on|off)_light_2_(on|off)")


def context_light_states(context: str | None) -> tuple[bool, bool]:
    """Resolve a symbolic context label to the two observable light states."""

    if context is None or context == "light_off":
        return (False, False)
    if context == "light_on":
        return (True, False)

    match = LIGHT_CONTEXT_PATTERN.fullmatch(context)
    if match is None:
        raise ValueError(f"Unsupported light context label: {context!r}.")
    return (match.group(1) == "on", match.group(2) == "on")


@dataclass(frozen=True)
class Observation:
    """Symbolic scene information visible to an agent."""

    sphere_color: str
    context: str | None = None

    @property
    def key(self) -> ObservationKey:
        parts = [f"sphere_{self.sphere_color}"]
        if self.context is not None:
            states = context_light_states(self.context)
            if self.context in {"light_on", "light_off"}:
                states = states[:1]
            parts.extend(f"L{i}_{'on' if value else 'off'}" for i, value in enumerate(states, 1))
        return "_".join(parts)

    def to_label(self) -> str:
        sphere = f"sphere_{self.sphere_color}"
        return f"{self.context}:{sphere}" if self.context else sphere


@dataclass(frozen=True)
class StepResult:
    """Outcome of one sorting action."""

    observation: Observation
    action: Action
    correct_action: Action
    reward: int
    is_correct: bool
    rule_name: str
    trial: int


@dataclass(frozen=True)
class TaskRule:
    """A hidden color-to-bin mapping used during one task condition."""

    name: str
    mapping: dict[str, Action]
    action_space: tuple[Action, ...]
    context: str | None = None

    def __post_init__(self) -> None:
        if not self.mapping:
            raise ValueError(f"Rule {self.name!r} has no color mapping.")
        if not self.action_space:
            raise ValueError(f"Rule {self.name!r} has no available actions.")
        if len(set(self.action_space)) != len(self.action_space):
            raise ValueError(f"Rule {self.name!r} contains duplicate actions.")

        unavailable = set(self.mapping.values()) - set(self.action_space)
        if unavailable:
            raise ValueError(
                f"Rule {self.name!r} maps to unavailable actions: {sorted(unavailable)}."
            )

    def correct_action_for(self, sphere_color: str) -> Action:
        try:
            return self.mapping[sphere_color]
        except KeyError as exc:
            raise ValueError(
                f"Color {sphere_color!r} is not part of rule {self.name!r}."
            ) from exc


def load_task_rules(path: str | Path) -> dict[str, TaskRule]:
    """Read the fixed task-rule schema used by this project."""

    if yaml is None:
        raise RuntimeError("PyYAML is required to read task rules.")

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, Mapping) or not isinstance(config.get("rules"), Mapping):
        raise ValueError("Task configuration must contain a 'rules' mapping.")

    rules: dict[str, TaskRule] = {}
    for name, data in config["rules"].items():
        if not isinstance(data, Mapping):
            raise ValueError(f"Rule {name!r} must be a mapping.")

        mapping = data.get("mapping")
        actions = data.get("available_actions")
        context = data.get("context")

        if not isinstance(mapping, Mapping):
            raise ValueError(f"Rule {name!r} must define a 'mapping'.")
        if not isinstance(actions, list):
            raise ValueError(f"Rule {name!r} must define 'available_actions' as a list.")
        if not all(
            isinstance(color, str) and isinstance(action, str)
            for color, action in mapping.items()
        ):
            raise ValueError(f"Rule {name!r} mapping keys and values must be strings.")
        if not all(isinstance(action, str) for action in actions):
            raise ValueError(f"Rule {name!r} actions must be strings.")
        if context is not None and not isinstance(context, str):
            raise ValueError(f"Rule {name!r} context must be a string or null.")

        rules[name] = TaskRule(
            name=name,
            mapping=dict(mapping),
            action_space=tuple(actions),
            context=context,
        )

    return rules


class SortingTaskEnv:
    """Sample spheres and score predefined bin-placement actions."""

    def __init__(
        self,
        rules: Mapping[str, TaskRule],
        initial_rule: str = "initial",
        rng: random.Random | None = None,
    ) -> None:
        if not rules:
            raise ValueError("At least one task rule is required.")

        self._rules = dict(rules)
        self._rng = rng or random.Random()
        self._trial = 0
        self._current_rule: TaskRule
        self._current_observation: Observation | None = None
        self.set_rule(initial_rule)

    @property
    def rule_name(self) -> str:
        return self._current_rule.name

    @property
    def action_space(self) -> tuple[Action, ...]:
        return self._current_rule.action_space

    @property
    def current_mapping(self) -> Mapping[str, Action]:
        """Return the hidden rule for human-only scene annotation."""

        return self._current_rule.mapping

    def reset(self) -> Observation:
        self._trial = 0
        self._current_observation = self._sample_observation()
        return self._current_observation

    def set_rule(self, rule_name: str) -> None:
        """Change the hidden rule without resetting the agent."""

        try:
            self._current_rule = self._rules[rule_name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._rules))
            raise ValueError(f"Unknown rule {rule_name!r}. Available: {choices}") from exc
        self._current_observation = None

    def get_observation(self) -> Observation:
        if self._current_observation is None:
            self._current_observation = self._sample_observation()
        return self._current_observation

    def correct_action(self, observation: Observation | None = None) -> Action:
        observation = observation or self.get_observation()
        return self._current_rule.correct_action_for(observation.sphere_color)

    def step(self, action: Action) -> StepResult:
        if action not in self.action_space:
            raise ValueError(f"Action {action!r} is outside {self.action_space}.")

        observation = self.get_observation()
        correct_action = self.correct_action(observation)
        is_correct = action == correct_action
        self._trial += 1

        result = StepResult(
            observation=observation,
            action=action,
            correct_action=correct_action,
            reward=1 if is_correct else -1,
            is_correct=is_correct,
            rule_name=self.rule_name,
            trial=self._trial,
        )
        self._current_observation = self._sample_observation()
        return result

    def _sample_observation(self) -> Observation:
        sphere_color = self._rng.choice(tuple(self._current_rule.mapping))
        return Observation(sphere_color, self._current_rule.context)
