from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adaptive_sorting.execution.execution_backend import NoOpBackend
from adaptive_sorting.execution.ros2_experiment_demo import (
    _phase_lengths,
    _scene_requirements,
    parse_args,
    run_demo,
    save_demo,
    demo_record,
)


class Ros2ExperimentDemoTests(unittest.TestCase):
    def test_demo_snapshots_before_execution_and_records_failures(self) -> None:
        from unittest.mock import patch
        from adaptive_sorting.env.sorting_task_env import DEFAULT_TASK_RULES
        from adaptive_sorting.experiments.task_config import resolve_task_config
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'original.yaml'
            source.write_bytes(DEFAULT_TASK_RULES.read_bytes())
            args = parse_args(['rq1', '--agent', 'ucb1', '--trials', '5',
                               '--final-window', '3', '--config', str(source),
                               '--output-dir', str(root / 'run')])
            with self.assertRaisesRegex(RuntimeError, 'plot failed'):
                with demo_record(args) as run_dir:
                    metadata = json.loads((run_dir / 'metadata.json').read_text())
                    self.assertEqual(metadata['status'], 'running')
                    self.assertEqual(metadata['seeds'], [args.seed])
                    source.write_text('invalid replacement')
                    self.assertEqual(resolve_task_config(run_dir, metadata).resolve(), args.config)
                    result, rows = run_demo(args, NoOpBackend(), None)
                    with patch('adaptive_sorting.execution.ros2_experiment_demo.plot_rq1.plot_report',
                               side_effect=RuntimeError('plot failed')):
                        save_demo(args, result, rows, run_dir)
            metadata = json.loads((root/'run/metadata.json').read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertIn('plot failed', metadata['error'])
            self.assertTrue((root/'run/trials.csv').exists())

    def test_ros_startup_failure_and_interruption_leave_run_identity(self) -> None:
        from unittest.mock import patch
        from adaptive_sorting.execution.ros2_experiment_demo import main
        for error, status in ((RuntimeError('backend unavailable'), 'failed'),
                              (KeyboardInterrupt(), 'interrupted')):
            with self.subTest(status=status), TemporaryDirectory() as directory:
                args = parse_args(['rq1', '--agent', 'ucb1', '--output-dir', str(Path(directory)/'run')])
                with patch('adaptive_sorting.execution.ros2_experiment_demo.parse_args', return_value=args), \
                     patch('adaptive_sorting.execution.ros2_experiment_demo.Ros2SortingClient', side_effect=error):
                    with self.assertRaises(type(error)):
                        main()
                metadata = json.loads((Path(directory)/'run/metadata.json').read_text())
                self.assertEqual(metadata['status'], status)
                self.assertEqual(metadata['backend'], 'ros2_moveit')
                self.assertIn('task_config_snapshot', metadata)

    def test_every_experiment_accepts_relational_ona(self) -> None:
        for rq in ("rq1", "rq2a", "rq2b", "rq3a", "rq3b", "rq4a", "rq4b"):
            with self.subTest(rq=rq):
                self.assertEqual(
                    parse_args([rq, "--agent", "relational_ona"]).agent,
                    "relational_ona",
                )

    def test_derives_scene_requirements_from_each_experiment_action_space(self) -> None:
        expected = {
            "rq1": (5, 0),
            "rq2a": (5, 0),
            "rq2b": (5, 0),
            "rq3a": (5, 1),
            "rq3b": (5, 2),
            "rq4a": (7, 0),
        }

        for rq, (bin_count, light_count) in expected.items():
            with self.subTest(rq=rq):
                requirements = _scene_requirements(parse_args([rq]))

                self.assertEqual(requirements.bin_count, bin_count)
                self.assertEqual(len(requirements.colors), bin_count)
                self.assertEqual(requirements.context_light_count, light_count)
                self.assertFalse(requirements.color_target_bins)

        requirements = _scene_requirements(parse_args(["rq4b"]))
        self.assertEqual(requirements.bin_count, 7)
        self.assertTrue(requirements.color_target_bins)

    def test_rejects_non_contiguous_ros_bin_action_space(self) -> None:
        with TemporaryDirectory() as directory:
            config = Path(directory) / "rules.yaml"
            config.write_text(
                """
rules:
  initial:
    context: null
    available_actions: [place_to_bin1, place_to_bin3]
    mapping:
      red: place_to_bin1
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "contiguous"):
                _scene_requirements(
                    parse_args(["rq1", "--config", str(config)])
                )

    def test_phase_lengths_match_each_rq_schedule(self) -> None:
        cases = {
            "rq1": (1400,),
            "rq2a": (200, 1200),
            "rq2b": (200,) + (1200,) * 8,
            "rq3a": (200, 1200),
            "rq3b": (200, 1200),
            "rq4a": (200, 1200),
        }

        for rq, expected in cases.items():
            with self.subTest(rq=rq):
                self.assertEqual(_phase_lengths(parse_args([rq])), expected)

    def test_each_rq_reuses_existing_single_seed_experiment(self) -> None:
        cases = (
            (
                ["rq1", "--agent", "ucb1", "--trials", "5", "--final-window", "3"],
                5,
            ),
            (
                [
                    "rq2a",
                    "--agent",
                    "ucb1",
                    "--before-trials",
                    "6",
                    "--after-trials",
                    "6",
                    "--final-window",
                    "3",
                ],
                12,
            ),
            (
                [
                    "rq2b",
                    "--agent",
                    "ucb1",
                    "--initial-trials",
                    "5",
                    "--phase-trials",
                    "5",
                    "--final-window",
                    "5",
                ],
                45,
            ),
            (
                [
                    "rq3a",
                    "--agent",
                    "ucb1",
                    "--before-trials",
                    "6",
                    "--mixed-trials",
                    "10",
                    "--context-block-size",
                    "10",
                    "--final-window",
                    "5",
                ],
                16,
            ),
            (
                [
                    "rq3b",
                    "--agent",
                    "ucb1",
                    "--before-trials",
                    "6",
                    "--mixed-trials",
                    "10",
                    "--context-block-size",
                    "10",
                    "--final-window",
                    "5",
                ],
                16,
            ),
            (
                [
                    "rq4a",
                    "--agent",
                    "ucb1",
                    "--base-trials",
                    "6",
                    "--expanded-trials",
                    "8",
                    "--final-window",
                    "4",
                ],
                14,
            ),
        )

        for argv, expected_trials in cases:
            with self.subTest(rq=argv[0]):
                args = parse_args(argv)
                result, rows = run_demo(args, NoOpBackend(), None)

                self.assertEqual(len(rows), expected_trials)
                if args.rq == "rq2b":
                    self.assertEqual(len(result), 9)
                    self.assertTrue(all(phase["seed"] == 1 for phase in result))
                else:
                    self.assertEqual(result["seed"], 1)
                self.assertTrue(all(row["execution_success"] for row in rows))
                self.assertTrue(
                    all(
                        {
                            "planning_duration_seconds",
                            "motion_duration_seconds",
                            "trajectory_cache_hits",
                        }.issubset(row)
                        for row in rows
                    )
                )

    def test_rejects_invalid_rq3_context_schedule_before_ros_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            parse_args(
                [
                    "rq3a",
                    "--mixed-trials",
                    "11",
                    "--final-window",
                    "10",
                ]
            )

    def test_saves_auditable_ros2_results_and_figure(self) -> None:
        args = parse_args(
            ["rq1", "--agent", "ucb1", "--trials", "5", "--final-window", "3"]
        )
        with TemporaryDirectory() as directory:
            args.output_dir = Path(directory) / "rq1_ros2"
            with demo_record(args) as run_dir:
                result, rows = run_demo(args, NoOpBackend(), None)
                save_demo(args, result, rows, run_dir)

            self.assertTrue((run_dir / "trials.csv").is_file())
            self.assertTrue((run_dir / "summary.csv").is_file())
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "complete")
            self.assertIn("task_config_snapshot", metadata)
            self.assertEqual(metadata["seed_scheme"], "replicate-v1")
            self.assertEqual(metadata["active_seed_streams"], ["environment", "agent"])
            self.assertEqual(metadata["replicates"][0]["seed"], args.seed)
            self.assertTrue((run_dir / "report.png").is_file())

    def test_saves_rq2b_phase_results_and_figure(self) -> None:
        args = parse_args(
            [
                "rq2b",
                "--agent",
                "ucb1",
                "--initial-trials",
                "5",
                "--phase-trials",
                "5",
                "--final-window",
                "5",
            ]
        )
        with TemporaryDirectory() as directory:
            args.output_dir = Path(directory) / "rq2b_ros2"
            with demo_record(args) as run_dir:
                result, rows = run_demo(args, NoOpBackend(), None)
                save_demo(args, result, rows, run_dir)

            self.assertTrue((run_dir / "trials.csv").is_file())
            self.assertTrue((run_dir / "summary.csv").is_file())
            self.assertTrue((run_dir / "metadata.json").is_file())
            self.assertTrue((run_dir / "report.png").is_file())

    def test_saves_rq3b_distractor_schedule_metadata(self) -> None:
        args = parse_args(
            [
                "rq3b",
                "--agent",
                "ucb1",
                "--before-trials",
                "6",
                "--mixed-trials",
                "10",
                "--context-block-size",
                "10",
                "--final-window",
                "5",
            ]
        )
        with TemporaryDirectory() as directory:
            args.output_dir = Path(directory) / "rq3b_ros2"
            with demo_record(args) as run_dir:
                result, rows = run_demo(args, NoOpBackend(), None)
                save_demo(args, result, rows, run_dir)
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["research_question"], "rq3b")
            self.assertEqual(
                metadata["active_seed_streams"],
                [
                    "environment",
                    "context_schedule",
                    "distractor_schedule",
                    "agent",
                ],
            )
            schedule = metadata["distractor_schedules"][0]
            self.assertEqual(len(schedule["states"]), 16)
            self.assertEqual(
                schedule["light_2_on_trials"] + schedule["light_2_off_trials"],
                16,
            )

if __name__ == "__main__":
    unittest.main()
