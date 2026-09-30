"""Shared perception-action-feedback step for all experiments."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.env.sorting_task_env import (
    Action,
    Observation,
    SortingTaskEnv,
    StepResult,
)
from adaptive_sorting.execution.execution_backend import (
    ExecutionResult,
    ActionExecutionError,
    ExecutionBackend,
)
from adaptive_sorting.perception import GroundTruthPerception, PerceptionInterface


@dataclass(frozen=True)
class InteractionResult:
    """Decision, action execution, and symbolic outcome for one trial."""

    observation: Observation
    action: Action
    execution: ExecutionResult
    outcome: StepResult


def run_interaction(
    env: SortingTaskEnv,
    agent: BaseAgent,
    backend: ExecutionBackend,
    perception: PerceptionInterface | None = None,
    *,
    presented_observation: Observation | None = None,
) -> InteractionResult:
    """Run one trial without mixing execution failures into learning feedback."""

    environment_observation = env.get_observation()
    scene_observation = presented_observation or environment_observation
    if scene_observation.sphere_color != environment_observation.sphere_color:
        raise ValueError(
            "Presented observation must preserve the environment sphere color."
        )
    backend.present_mapping(env.current_mapping)
    backend.present(scene_observation)
    active_perception = perception or GroundTruthPerception(lambda: scene_observation)
    observation = active_perception.observe()
    action = agent.select_action(observation)
    execution = backend.execute(action)
    if not execution.success:
        raise ActionExecutionError(action, execution)

    outcome = env.step(action)
    agent.update(observation, action, outcome.reward)
    backend.present_feedback(outcome.reward)
    return InteractionResult(observation, action, execution, outcome)
