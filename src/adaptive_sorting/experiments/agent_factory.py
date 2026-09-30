"""Create experiment agents from one shared set of command-line options."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Callable, Sequence

from adaptive_sorting.agents.base_agent import BaseAgent
from adaptive_sorting.agents.epsilon_greedy_agent import (
    DEFAULT_EPSILON,
    DEFAULT_LEARNING_RATE,
    EpsilonGreedyAgent,
)
from adaptive_sorting.agents.ona_agent import (
    FLAT_ENCODING,
    DEFAULT_ONA_BINARY,
    DEFAULT_ONA_STARTUP_TIMEOUT,
    DEFAULT_ONA_TIMEOUT,
    ONAAgent,
    RELATIONAL_SORTING_ENCODING,
)
from adaptive_sorting.agents.sliding_window_ucb_agent import (
    DEFAULT_UCB_EXPLORATION,
    DEFAULT_UCB_WINDOW,
    SlidingWindowUCBAgent,
)
from adaptive_sorting.agents.ucb1_agent import DEFAULT_UCB1_EXPLORATION, UCB1Agent
from adaptive_sorting.env.sorting_task_env import Action
from adaptive_sorting.experiments.defaults import ENCODING_VERSION, PROTOCOL_VERSION
from adaptive_sorting.experiments.ona_profile import ONA_C256_T240_MAX_WORKERS, binary_provenance


from adaptive_sorting.naming import ona_label


AgentFactory = Callable[[Sequence[Action], int], BaseAgent]
AGENT_LABELS = {
    "flat_ona": "ONA (flat)",
    "relational_ona": "ONA (relational)",
    "epsilon_greedy": "Epsilon-greedy",
    "ucb1": "Contextual UCB1",
    "sw_ucb": "Contextual SW-UCB",
}
STANDARD_AGENT_CHOICES = ("flat_ona", "epsilon_greedy", "ucb1", "sw_ucb")


def add_agent_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_relational_ona: bool = False,
) -> None:
    choices = STANDARD_AGENT_CHOICES
    if include_relational_ona:
        choices += ("relational_ona",)
    parser.add_argument(
        "--agent",
        choices=choices,
        default="flat_ona",
    )
    parser.add_argument("--ona-binary", type=Path, default=DEFAULT_ONA_BINARY)
    parser.add_argument(
        "--ona-timeout", type=float, default=DEFAULT_ONA_TIMEOUT
    )
    parser.add_argument(
        "--ona-startup-timeout",
        type=float,
        default=DEFAULT_ONA_STARTUP_TIMEOUT,
    )
    parser.add_argument("--ona-cycles", type=int, default=100)
    parser.add_argument("--ona-anticipation-confidence", type=float)
    parser.add_argument("--ona-decision-threshold", type=float)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--ucb1-exploration", type=float, default=DEFAULT_UCB1_EXPLORATION
    )
    parser.add_argument("--ucb-window", type=int, default=DEFAULT_UCB_WINDOW)
    parser.add_argument(
        "--ucb-exploration", type=float, default=DEFAULT_UCB_EXPLORATION
    )


def build_agent_factory(args: argparse.Namespace) -> AgentFactory:
    if args.agent == "epsilon_greedy":
        return lambda actions, agent_seed: EpsilonGreedyAgent(
            actions,
            epsilon=args.epsilon,
            learning_rate=args.learning_rate,
            rng=random.Random(agent_seed),
        )
    if args.agent == "sw_ucb":
        return lambda actions, agent_seed: SlidingWindowUCBAgent(
            actions,
            window_size=args.ucb_window,
            exploration=args.ucb_exploration,
            rng=random.Random(agent_seed),
        )
    if args.agent == "ucb1":
        return lambda actions, agent_seed: UCB1Agent(
            actions,
            exploration=args.ucb1_exploration,
            rng=random.Random(agent_seed),
        )
    interaction_encoding = (
        RELATIONAL_SORTING_ENCODING
        if args.agent == "relational_ona"
        else FLAT_ENCODING
    )
    return lambda actions, agent_seed: ONAAgent(
        actions,
        seed=agent_seed,
        binary_path=args.ona_binary,
        timeout=args.ona_timeout,
        startup_timeout=args.ona_startup_timeout,
        inference_cycles=args.ona_cycles,
        anticipation_confidence=args.ona_anticipation_confidence,
        decision_threshold=args.ona_decision_threshold,
        interaction_encoding=interaction_encoding,
        name=args.agent,
    )


def agent_metadata(args: argparse.Namespace) -> dict[str, object]:
    encoding_metadata = {"encoding_version": ENCODING_VERSION, "protocol_version": PROTOCOL_VERSION}
    if args.agent == "epsilon_greedy":
        return {
            **encoding_metadata,
            "agent": "epsilon_greedy",
            "epsilon": args.epsilon,
            "learning_rate": args.learning_rate,
        }
    if args.agent == "sw_ucb":
        return {
            **encoding_metadata,
            "agent": "sw_ucb",
            "ucb_window": args.ucb_window,
            "ucb_exploration": args.ucb_exploration,
        }
    if args.agent == "ucb1":
        return {
            **encoding_metadata,
            "agent": "ucb1",
            "ucb1_exploration": args.ucb1_exploration,
        }
    metadata = {
        **encoding_metadata,
        "agent": args.agent,
        **binary_provenance(args.ona_binary),
        "command_timeout_seconds": args.ona_timeout,
        "startup_timeout_seconds": args.ona_startup_timeout,
        "inference_cycles": args.ona_cycles,
        "anticipation_confidence": args.ona_anticipation_confidence,
        "decision_threshold": args.ona_decision_threshold,
    }
    if args.agent == "relational_ona":
        metadata["interaction_encoding"] = RELATIONAL_SORTING_ENCODING
    return metadata


def agent_label(agent_name: str, binary_path: Path | None = None) -> str:
    if agent_name not in ("flat_ona", "relational_ona"):
        return AGENT_LABELS[agent_name]
    variant = None
    if binary_path is not None and Path(binary_path).is_file():
        variant = binary_provenance(binary_path).get("variant")
    return ona_label(relational=agent_name == "relational_ona", variant=variant)


def agent_worker_count(args: argparse.Namespace) -> int:
    """Limit every ONA condition to three concurrent processes."""
    return min(args.workers, ONA_C256_T240_MAX_WORKERS) if args.agent in ("flat_ona", "relational_ona") else args.workers
