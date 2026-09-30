"""Shared seed scheduling and checkpoint persistence for experiment runners."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from typing import TypeVar

from adaptive_sorting.experiments.defaults import DEFAULT_WORKERS
from adaptive_sorting.experiments.result_paths import write_seed_checkpoint, record_run_failure

JobT = TypeVar('JobT')
ResultT = TypeVar('ResultT')


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_worker_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=positive_integer,
        default=DEFAULT_WORKERS,
        help=f"maximum number of seed workers (default: {DEFAULT_WORKERS})",
    )


def effective_worker_count(workers: int, job_count: int) -> int:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if job_count <= 0:
        raise ValueError("job count must be positive.")
    return min(workers, job_count)


def _run_checkpointed_job(worker, job, directory):
    outcome = worker(job)
    write_seed_checkpoint(directory, job, outcome)
    return outcome


def run_jobs(
    worker: Callable[[JobT], ResultT],
    jobs: Sequence[JobT],
    workers: int,
    *,
    checkpoint_dir: Path | None = None,
) -> list[ResultT]:
    """Run jobs in input order; workers persist completed jobs independently."""
    effective_workers = effective_worker_count(workers, len(jobs))
    directories = [
        checkpoint_dir / "seeds" / f"job_{index:04d}_seed_{job.seed}"
        for index, job in enumerate(jobs, start=1)
    ] if checkpoint_dir is not None else []
    try:
        if effective_workers == 1:
            return [
                _run_checkpointed_job(worker, job, directories[index])
                if checkpoint_dir is not None else worker(job)
                for index, job in enumerate(jobs)
            ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=context) as executor:
            if checkpoint_dir is None:
                return list(executor.map(worker, jobs))
            return list(executor.map(_run_checkpointed_job, [worker] * len(jobs), jobs, directories))
    except BaseException as exc:
        if checkpoint_dir is not None:
            record_run_failure(checkpoint_dir, exc)
        raise


