"""Filesystem helpers for reproducible experiment outputs."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Sequence

from adaptive_sorting.analysis.evaluation import METRICS_VERSION
from adaptive_sorting.experiments.task_config import snapshot_task_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiment_logs"


def create_run_dir(
    results_dir: Path,
    rq: str,
    output_dir: Path | None = None,
) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    parent = results_dir / rq
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = parent / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = parent / f"{run_id}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_metadata(run_dir: Path, metadata: dict[str, object]) -> Path:
    path = run_dir / "metadata.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    content = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": current_git_commit(),
        **existing,
        "status": "complete",
        **metadata,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, default=str)
        handle.write("\n")
    return path


def prepare_run(args, research_question: str, *, metadata=None) -> Path:
    """Record configuration before running any seed, including interrupted runs."""
    from adaptive_sorting.experiments.agent_factory import agent_metadata

    # Validate binary identity before allocating a new run directory.
    identity = agent_metadata(args)
    directory = create_run_dir(args.results_dir, research_question, args.output_dir)
    snapshot = snapshot_task_config(args.config, directory)
    args.config = (directory / snapshot["path"]).resolve()
    write_metadata(directory, {
        "task_config_snapshot": snapshot,
        "status": "running",
        "research_question": research_question,
        "parameters": vars(args),
        "checkpoint_directory": "seeds",
        "metrics_version": METRICS_VERSION,
        **identity,
        **(metadata or {}),
    })
    return directory


def record_run_failure(directory: Path, exc: BaseException) -> None:
    write_metadata(directory, {
        'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
        'error': f'{type(exc).__name__}: {exc}',
    })


@contextmanager
def recorded_run(args, research_question: str, *, metadata=None):
    """Persist identity before execution and mark completion only on success."""
    directory = prepare_run(args, research_question, metadata=metadata)
    try:
        yield directory
    except BaseException as exc:
        record_run_failure(directory, exc)
        raise
    else:
        write_metadata(directory, {'status': 'complete'})


def write_seed_checkpoint(directory: Path, job, outcome) -> None:
    """Persist a completed job before its worker returns to the process pool."""
    directory.mkdir(parents=True, exist_ok=False)
    result, rows, *diagnostics = outcome
    results = result if isinstance(result, list) else [result]
    write_experiment_results(directory, results, rows)
    if diagnostics and diagnostics[0] is not None:
        (directory / "diagnostics.json").write_text(
            json.dumps(asdict(diagnostics[0]), indent=2) + "\n", encoding="utf-8"
        )
    # The completion marker is written last; missing marker means incomplete output.
    job_metadata = asdict(job)
    if hasattr(job, "args"):
        job_metadata["args"] = vars(job.args)
    (directory / "completed.json").write_text(
        json.dumps({"seed": job.seed, "job": job_metadata}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_experiment_results(
    run_dir: Path,
    results: Sequence[object],
    rows: Sequence[dict[str, object]],
) -> tuple[Path, Path]:
    if not results or not rows:
        raise ValueError("Experiment results and trial rows must not be empty.")

    trial_path = run_dir / "trials.csv"
    summary_path = run_dir / "summary.csv"
    with trial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        summary_rows = [dict(result) if isinstance(result, Mapping) else vars(result) for result in results]
        fieldnames = list(dict.fromkeys(key for row in summary_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return trial_path, summary_path
