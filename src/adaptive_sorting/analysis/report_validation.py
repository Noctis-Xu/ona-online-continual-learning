"""Validate planned runs and fingerprint inputs before reusing analysis results."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from adaptive_sorting.experiments.task_config import resolve_task_config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# Statistical dependencies are deliberately separate from renderers and CLI text.
STATISTICAL_SOURCES = (
    'analysis/evaluation.py', 'analysis/statistics.py', 'analysis/metric_summary.py',
    'analysis/report_validation.py', 'experiments/task_config.py',
    'env/sorting_task_env.py', 'naming.py',
)


def source_code_hashes(root=None):
    root = Path(root) if root is not None else Path(__file__).parents[1]
    def digest(paths):
        value = hashlib.sha256()
        for path in sorted(set(paths)):
            value.update(str(path.relative_to(root)).encode())
            value.update(path.read_bytes())
        return value.hexdigest()
    statistical = [root / name for name in STATISTICAL_SOURCES]
    return digest(statistical), digest([*statistical, *(root / 'analysis').glob('*.py')])


def source_fingerprints(paths):
    fingerprints = {}
    for label, path in paths:
        metadata_path = path.with_name('metadata.json')
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        config = resolve_task_config(path.parent, metadata)
        fingerprints[label] = {
            'trials': file_hash(path),
            'metadata': file_hash(metadata_path) if metadata_path.exists() else None,
            'task_config': file_hash(config),
            'task_config_provenance': 'snapshot' if metadata.get('task_config_snapshot') else 'unverified',
        }
    return fingerprints


def validate_cache(previous, paths, config, fingerprints, code_hash, samples, seed):
    expected = {**config.metadata(), 'bootstrap_samples': samples, 'bootstrap_seed': seed,
                'binary_proportion_interval': 'wilson', 'paired_interval': 'seed_bootstrap',
                'sources': {label: str(path.resolve()) for label, path in paths},
                'source_fingerprints': fingerprints, 'analysis_source_sha256': code_hash}
    mismatches = [key for key, value in expected.items() if previous.get(key) != value]
    if mismatches:
        raise ValueError('Cached metrics are stale or unverifiable: ' + ', '.join(mismatches)
                         + '. Recompute this protocol without cache reuse.')


def phase_plan(rq, metadata):
    """Return the recorded plan; never infer a complete run from its observed tail."""
    p = {**metadata, **metadata.get('parameters', {})}
    if p.get('status') not in (None, 'complete'):
        raise ValueError('Run metadata does not declare completion.')
    if rq == 'rq1':
        return [(None, int(p['trials']))] if 'trials' in p else None
    if rq == 'rq2b':
        keys = ('initial_trials', 'phase_trials', 'phase_labels')
        if not all(k in p for k in keys):
            return None
        return [(label, int(p['initial_trials'] if i == 0 else p['phase_trials']))
                for i, label in enumerate(p['phase_labels'])]
    definitions = {
        'rq2a': ('before_trials', 'after_trials', 'before_update', 'after_update'),
        'rq3a': ('before_trials', 'mixed_trials', 'off_only', 'mixed_contexts'),
        'rq3b': ('before_trials', 'mixed_trials', 'off_only', 'mixed_contexts'),
        'rq4a': ('base_trials', 'expanded_trials', 'known_only', 'expanded_inputs'),
        'rq4b': ('base_trials', 'expanded_trials', 'known_only', 'expanded_inputs'),
    }
    before, after, first, second = definitions[rq]
    return [(first, int(p[before])), (second, int(p[after]))] if before in p and after in p else None


def validate_job(rq, rows, plan):
    if plan is None:
        return
    if len(rows) != sum(length for _, length in plan):
        raise ValueError('Trial count disagrees with the recorded phase plan.')
    offset = 0
    for index, (label, length) in enumerate(plan, 1):
        for local, row in enumerate(rows[offset:offset + length], 1):
            if int(row['trial']) != offset + local or int(row.get('phase_trial', local)) != local:
                raise ValueError('Trial or phase_trial sequence disagrees with the phase plan.')
            if rq == 'rq2b':
                if int(row['phase_index']) != index or row['phase_label'] != label:
                    raise ValueError('Recurring phase index or label disagrees with the phase plan.')
            elif label is not None and row['phase'] != label:
                raise ValueError('Phase boundaries disagree with the recorded phase plan.')
        offset += length


def validate_jobs(rq, jobs, metadata):
    p = {**metadata, **metadata.get('parameters', {})}
    if 'seeds' not in p:
        return
    conditions = p.get('change_sizes') if rq == 'rq2a' else [0]
    if conditions is None:
        return
    expected = {(int(seed), int(condition)) for seed in p['seeds'] for condition in conditions}
    if Counter(jobs) != Counter(expected):
        raise ValueError('Seed/condition jobs disagree with the recorded experiment plan.')
