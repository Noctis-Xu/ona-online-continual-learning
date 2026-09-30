"""Derive independent, reproducible RNG streams for one experiment replicate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence


SEED_SCHEME = "replicate-v1"
_SEED_NAMESPACE = "adaptive-sorting"
_STREAM_NAMES = (
    "environment",
    "context_schedule",
    "distractor_schedule",
    "agent",
)


@dataclass(frozen=True)
class ReplicateSeeds:
    """Concrete RNG seeds derived from one user-facing replicate seed."""

    replicate: int
    environment: int
    context_schedule: int
    distractor_schedule: int
    agent: int


def derive_replicate_seeds(replicate: int) -> ReplicateSeeds:
    """Return stable 32-bit seeds for each independent stochastic component."""

    if replicate < 0:
        raise ValueError("replicate seed must not be negative.")
    derived = {
        stream: _derive_stream_seed(replicate, stream) for stream in _STREAM_NAMES
    }
    return ReplicateSeeds(replicate=replicate, **derived)


def replicate_seed_metadata(
    replicates: Sequence[int],
    *,
    uses_context_schedule: bool = False,
    uses_distractor_schedule: bool = False,
) -> dict[str, object]:
    """Describe the seed scheme and concrete streams used by an experiment."""

    active_streams = ["environment"]
    if uses_context_schedule:
        active_streams.append("context_schedule")
    if uses_distractor_schedule:
        active_streams.append("distractor_schedule")
    active_streams.append("agent")
    entries = []
    for replicate in replicates:
        seeds = derive_replicate_seeds(replicate)
        entry = {
            "seed": seeds.replicate,
            "environment_seed": seeds.environment,
            "context_schedule_seed": seeds.context_schedule,
            "agent_seed": seeds.agent,
        }
        if uses_distractor_schedule:
            entry["distractor_schedule_seed"] = seeds.distractor_schedule
        entries.append(entry)
    return {
        "seeds": list(replicates),
        "seed_scheme": SEED_SCHEME,
        "active_seed_streams": active_streams,
        "replicates": entries,
    }


def _derive_stream_seed(replicate: int, stream: str) -> int:
    material = f"{_SEED_NAMESPACE}/{SEED_SCHEME}/{replicate}/{stream}".encode()
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
