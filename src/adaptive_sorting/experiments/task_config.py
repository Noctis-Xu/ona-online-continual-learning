"""Preserve and resolve the task configuration used by an experiment."""
from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES


def snapshot_task_config(source: Path, directory: Path) -> dict[str, str]:
    content = Path(source).read_bytes()
    target = directory / 'task_config.yaml'
    target.write_bytes(content)
    return {'path': target.name, 'sha256': hashlib.sha256(content).hexdigest(),
            'original_path': str(Path(source).resolve())}


def resolve_task_config(directory: Path, metadata: dict) -> Path:
    snapshot = metadata.get('task_config_snapshot')
    if snapshot is not None:
        path = directory / snapshot['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != snapshot['sha256']:
            raise ValueError(f'Task configuration snapshot is missing or modified: {path}')
        return path
    path = Path(metadata.get('parameters', {}).get('config', metadata.get('task_config', DEFAULT_TASK_RULES)))
    if not path.is_file():
        raise ValueError(f'Task config unavailable: {path}')
    warnings.warn(f'Run has no task configuration snapshot: {directory}. '
                  f'Using {path}; its identity at experiment time is unverified.',
                  UserWarning, stacklevel=2)
    return path
