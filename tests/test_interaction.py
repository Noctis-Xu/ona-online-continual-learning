from __future__ import annotations

import random
import unittest
from typing import Mapping

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.env.sorting_task_env import Observation, SortingTaskEnv, TaskRule
from adaptive_sorting.execution.execution_backend import (
    ExecutionResult,
    ActionExecutionError,
)
from adaptive_sorting.experiments.interaction import run_interaction
from adaptive_sorting.perception import PerceptionError


ACTION = "place_to_bin1"


class RecordingAgent(BaseAgent):
    def __init__(self, events: list[str]) -> None:
        super().__init__([ACTION])
        self.events = events
        self.updates: list[tuple[Observation, str, int]] = []

    def select_action(self, observation: Observation) -> str:
        self.events.append("select")
        return ACTION

    def update(self, observation: Observation, action: str, reward: int) -> None:
        self.events.append("update")
        self.updates.append((observation, action, reward))


class RecordingBackend:
    def __init__(self, events: list[str], result: ExecutionResult) -> None:
        self.events = events
        self.result = result
        self.observations: list[Observation] = []
        self.actions: list[str] = []
        self.rewards: list[int] = []
        self.mappings: list[dict[str, str]] = []

    def present_mapping(self, mapping: Mapping[str, str]) -> None:
        self.events.append("mapping")
        self.mappings.append(dict(mapping))

    def present(self, observation: Observation) -> None:
        self.events.append("present")
        self.observations.append(observation)

    def execute(self, action: str) -> ExecutionResult:
        self.events.append("execute")
        self.actions.append(action)
        return self.result

    def present_feedback(self, reward: int) -> None:
        self.events.append("feedback")
        self.rewards.append(reward)


class RecordingPerception:
    def __init__(
        self,
        events: list[str],
        observation: Observation | None = None,
        error: Exception | None = None,
    ) -> None:
        self.events = events
        self.observation = observation
        self.error = error

    def observe(self) -> Observation:
        self.events.append("observe")
        if self.error is not None:
            raise self.error
        if self.observation is None:
            raise AssertionError("test perception has no observation")
        return self.observation


def make_env() -> SortingTaskEnv:
    rule = TaskRule(
        name="test",
        mapping={"red": ACTION},
        action_space=(ACTION,),
    )
    return SortingTaskEnv({"test": rule}, initial_rule="test", rng=random.Random(1))


class InteractionTests(unittest.TestCase):
    def test_interaction_preserves_perception_action_feedback_order(self) -> None:
        events: list[str] = []
        env = make_env()
        observation = env.reset()
        agent = RecordingAgent(events)
        backend = RecordingBackend(events, ExecutionResult(success=True))
        perception = RecordingPerception(events, observation)

        interaction = run_interaction(env, agent, backend, perception)

        self.assertEqual(
            events,
            [
                "mapping",
                "present",
                "observe",
                "select",
                "execute",
                "update",
                "feedback",
            ],
        )
        self.assertEqual(backend.mappings, [{"red": ACTION}])
        self.assertEqual(backend.observations, [observation])
        self.assertEqual(backend.actions, [ACTION])
        self.assertEqual(backend.rewards, [1])
        self.assertEqual(interaction.observation, observation)
        self.assertEqual(interaction.action, ACTION)
        self.assertTrue(interaction.outcome.is_correct)
        self.assertEqual(agent.updates, [(observation, ACTION, 1)])

    def test_execution_failure_does_not_advance_or_update_experiment(self) -> None:
        events: list[str] = []
        env = make_env()
        observation = env.reset()
        agent = RecordingAgent(events)
        backend = RecordingBackend(
            events,
            ExecutionResult(success=False, error="controller unavailable"),
        )
        perception = RecordingPerception(events, observation)

        with self.assertRaisesRegex(ActionExecutionError, "controller unavailable"):
            run_interaction(env, agent, backend, perception)

        self.assertEqual(
            events, ["mapping", "present", "observe", "select", "execute"]
        )
        self.assertEqual(agent.updates, [])
        self.assertEqual(env.get_observation(), observation)
        self.assertEqual(env.step(ACTION).trial, 1)

    def test_perception_failure_stops_before_action_selection(self) -> None:
        events: list[str] = []
        env = make_env()
        observation = env.reset()
        agent = RecordingAgent(events)
        backend = RecordingBackend(events, ExecutionResult(success=True))
        perception = RecordingPerception(
            events,
            error=PerceptionError("camera unavailable"),
        )

        with self.assertRaisesRegex(PerceptionError, "camera unavailable"):
            run_interaction(env, agent, backend, perception)

        self.assertEqual(events, ["mapping", "present", "observe"])
        self.assertEqual(agent.updates, [])
        self.assertEqual(env.get_observation(), observation)

    def test_default_perception_uses_controlled_environment_metadata(self) -> None:
        events: list[str] = []
        env = make_env()
        observation = env.reset()
        agent = RecordingAgent(events)
        backend = RecordingBackend(events, ExecutionResult(success=True))

        interaction = run_interaction(env, agent, backend)

        self.assertEqual(interaction.observation, observation)
        self.assertEqual(
            events,
            ["mapping", "present", "select", "execute", "update", "feedback"],
        )

    def test_presented_observation_can_add_observable_context(self) -> None:
        events: list[str] = []
        env = make_env()
        base_observation = env.reset()
        presented = Observation(
            base_observation.sphere_color,
            "light_1_off_light_2_on",
        )
        agent = RecordingAgent(events)
        backend = RecordingBackend(events, ExecutionResult(success=True))

        interaction = run_interaction(
            env,
            agent,
            backend,
            presented_observation=presented,
        )

        self.assertEqual(backend.observations, [presented])
        self.assertEqual(interaction.observation, presented)
        self.assertEqual(agent.updates, [(presented, ACTION, 1)])

    def test_presented_observation_cannot_change_environment_color(self) -> None:
        events: list[str] = []
        env = make_env()
        env.reset()
        agent = RecordingAgent(events)
        backend = RecordingBackend(events, ExecutionResult(success=True))

        with self.assertRaisesRegex(ValueError, "preserve.*sphere color"):
            run_interaction(
                env,
                agent,
                backend,
                presented_observation=Observation("blue", "light_off"),
            )

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
