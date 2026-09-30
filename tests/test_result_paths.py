from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from adaptive_sorting.experiments.result_paths import (
    create_run_dir,
    write_experiment_results,
    write_metadata,
)


@dataclass(frozen=True)
class ExampleResult:
    seed: int
    score: float


class ResultPathTests(unittest.TestCase):
    def test_shared_writer_saves_trials_and_dataclass_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            trial_path, summary_path = write_experiment_results(
                run_dir,
                [ExampleResult(seed=1, score=0.75)],
                [{"seed": 1, "reward": 1}],
            )

            with trial_path.open(newline="", encoding="utf-8") as handle:
                trial = next(csv.DictReader(handle))
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = next(csv.DictReader(handle))

            self.assertEqual(trial, {"seed": "1", "reward": "1"})
            self.assertEqual(summary, {"seed": "1", "score": "0.75"})

    def test_run_directories_are_unique_and_include_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            first = create_run_dir(results_dir, "rq2a")
            second = create_run_dir(results_dir, "rq2a")
            metadata_path = write_metadata(
                first,
                {
                    "research_question": "rq2a",
                    "agent": "flat_ona",
                    "seeds": [1, 2],
                },
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, results_dir / "rq2a")
            self.assertEqual(metadata["research_question"], "rq2a")
            self.assertEqual(metadata["agent"], "flat_ona")
            self.assertEqual(metadata["seeds"], [1, 2])
            self.assertIn("created_at", metadata)
            self.assertIn("git_commit", metadata)

    def test_explicit_output_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "rq1" / "run" / "flat_ona"

            run_dir = create_run_dir(Path(temp_dir), "rq1", output_dir)

            self.assertEqual(run_dir, output_dir)
            self.assertTrue(output_dir.is_dir())
            with self.assertRaises(FileExistsError):
                create_run_dir(Path(temp_dir), "rq1", output_dir)


    def test_prepared_run_uses_verified_snapshot_after_source_changes(self):
        from argparse import Namespace
        from unittest.mock import patch
        from adaptive_sorting.experiments.result_paths import prepare_run
        from adaptive_sorting.experiments.task_config import resolve_task_config
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'rules.yaml'
            source.write_text('original configuration')
            args = Namespace(config=source, results_dir=root, output_dir=None)
            with patch('adaptive_sorting.experiments.agent_factory.agent_metadata', return_value={}):
                directory = prepare_run(args, 'rq1')
            metadata = json.loads((directory / 'metadata.json').read_text())
            source.write_text('changed configuration')
            snapshot = resolve_task_config(directory, metadata)
            self.assertEqual(args.config, snapshot.resolve())
            self.assertEqual(snapshot.read_text(), 'original configuration')
            moved = root / 'moved'
            directory.rename(moved)
            self.assertEqual(resolve_task_config(moved, metadata).read_text(), 'original configuration')
            (moved / 'task_config.yaml').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'missing or modified'):
                resolve_task_config(moved, metadata)

    def test_run_without_snapshot_is_explicitly_unverified(self):
        from adaptive_sorting.experiments.task_config import resolve_task_config
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rules.yaml'
            path.write_text('rules')
            with self.assertWarnsRegex(UserWarning, 'unverified'):
                self.assertEqual(resolve_task_config(path.parent, {'task_config': str(path)}), path)


if __name__ == "__main__":
    unittest.main()
