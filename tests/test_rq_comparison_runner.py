from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_sorting.experiments.runner import effective_worker_count, run_jobs
from adaptive_sorting.experiments.run_rq_comparison import (
    PLOT_MODULE,
    RQ_RUNNERS,
    SUMMARY_MODULE,
    main,
    parse_args,
    run_rq,
    trial_counts,
)


def identify_process(value: int) -> tuple[int, int]:
    return value, os.getpid()


def fail_on_two(value: int) -> int:
    if value == 2:
        raise RuntimeError("seed failed")
    return value


class ComparisonArgumentTests(unittest.TestCase):
    def test_parses_multiple_targets_agent_and_seed_count(self) -> None:
        args = parse_args(
            ["rq1", "rq3b", "ucb1", "--seeds", "12"]
        )

        self.assertEqual(args.targets, ["rq1", "rq3b"])
        self.assertEqual(args.agents, ["ucb1"])
        self.assertEqual(args.seed_count, 12)

    def test_parses_explicit_worker_count(self) -> None:
        args = parse_args(["rq1", "--workers", "7"])

        self.assertEqual(args.workers, 7)

    def test_relational_ona_supports_every_research_question(self) -> None:
        args = parse_args(["all", "relational_ona"])

        self.assertEqual(args.agents, ["relational_ona"])

    def test_trial_counts_require_two_positive_integers(self) -> None:
        self.assertEqual(trial_counts("150/600"), (150, 600))
        invalid_values = (
            "300",
            "150/",
            "/600",
            "0/600",
            "150/0",
            "150/600/900",
        )
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                trial_counts(value)

    def test_parses_trial_counts(self) -> None:
        args = parse_args(["rq2b", "ucb1", "--trials", "200/300"])

        self.assertEqual(args.trials, (200, 300))

    def test_rejects_invalid_trial_counts_before_running_all(self) -> None:
        args = parse_args(["all", "--trials", "300/600"])
        self.assertEqual(args.trials, (300, 600))

        with self.assertRaises(SystemExit):
            parse_args(["all", "--trials", "199/600"])

    def test_rq2b_requires_final_window_in_post_change_phases(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["rq2b", "ucb1", "--trials", "150/200"])

    def test_rq3_trial_validation_requires_the_final_window(self) -> None:
        for agent in ("ucb1", "relational_ona"):
            args = parse_args(["rq3a", agent, "--trials", "50/300"])
            self.assertEqual(args.trials, (50, 300))

        with self.assertRaises(SystemExit):
            parse_args(["rq3a", "relational_ona", "--trials", "49/300"])
        with self.assertRaises(SystemExit):
            parse_args(["rq3a", "relational_ona", "--trials", "50/290"])
        with self.assertRaises(SystemExit):
            parse_args(["rq3a", "ucb1", "--trials", "50/201"])


class SerialComparisonRunnerTests(unittest.TestCase):
    def test_forwards_trial_counts_using_each_rq_interface(self) -> None:
        commands: dict[str, list[str]] = {}

        def execute(command: list[str]) -> int:
            module = command[command.index("-m") + 1]
            rq = next(rq for rq, runner in RQ_RUNNERS.items() if runner == module)
            commands[rq] = command
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "trials.csv").touch()
            return 0

        expected_arguments = {
            "rq1": ("--trials", "200", None, None),
            "rq2a": ("--before-trials", "200", "--after-trials", "600"),
            "rq2b": ("--initial-trials", "200", "--phase-trials", "600"),
            "rq3a": ("--before-trials", "200", "--mixed-trials", "600"),
            "rq3b": ("--before-trials", "200", "--mixed-trials", "600"),
            "rq4a": ("--base-trials", "200", "--expanded-trials", "600"),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            for rq in RQ_RUNNERS:
                run_rq(
                    rq,
                    agents=("ucb1",),
                    seeds=(1,),
                    results_dir=Path(temp_dir),
                    trials=(200, 600),
                    execute=execute,
                )

        for rq, (initial_arg, initial, subsequent_arg, subsequent) in (
            expected_arguments.items()
        ):
            command = commands[rq]
            self.assertEqual(command[command.index(initial_arg) + 1], initial)
            if subsequent_arg is None:
                self.assertNotIn("600", command)
            else:
                self.assertEqual(
                    command[command.index(subsequent_arg) + 1], subsequent
                )

    def test_reports_agent_elapsed_time(self) -> None:
        output = StringIO()

        def execute(command: list[str]) -> int:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "trials.csv").touch()
            return 0

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "adaptive_sorting.experiments.run_rq_comparison.perf_counter",
            side_effect=[10.0, 12.5],
        ), redirect_stdout(output):
            run_rq(
                "rq1",
                agents=("ucb1",),
                seeds=(1,),
                results_dir=Path(temp_dir),
                execute=execute,
            )

        self.assertIn(
            "timing=rq1/ucb1 status=completed elapsed_seconds=2.50",
            output.getvalue(),
        )

    def test_reports_total_elapsed_time(self) -> None:
        output = StringIO()

        with patch(
            "adaptive_sorting.experiments.run_rq_comparison.perf_counter",
            side_effect=[20.0, 27.25],
        ), patch(
            "adaptive_sorting.experiments.run_rq_comparison.run_rq"
        ), redirect_stdout(output):
            main(["rq1", "ucb1", "--seeds", "1"])

        self.assertIn(
            "timing=total status=completed elapsed_seconds=7.25",
            output.getvalue(),
        )

    def test_runs_default_agents_serially_before_comparison_reports(self) -> None:
        commands: list[list[str]] = []

        def execute(command: list[str]) -> int:
            commands.append(command)
            module = command[command.index("-m") + 1]
            if module == RQ_RUNNERS["rq1"]:
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir()
                (output_dir / "trials.csv").touch()
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = run_rq(
                "rq1",
                agents=(),
                seeds=(1, 2),
                results_dir=Path(temp_dir),
                execute=execute,
            )

        modules = [command[command.index("-m") + 1] for command in commands]
        self.assertEqual(
            modules,
            [RQ_RUNNERS["rq1"]] * 5 + [PLOT_MODULE, SUMMARY_MODULE],
        )
        self.assertEqual(run_dir.parent.name, "rq1")
        first_experiment = commands[0]
        self.assertEqual(
            first_experiment[
                first_experiment.index("--seeds")
                + 1 : first_experiment.index("--workers")
            ],
            ["1", "2"],
        )

    def test_rq3_forwards_ona_options_and_caps_ona_workers(self) -> None:
        commands: list[list[str]] = []

        def execute(command: list[str]) -> int:
            commands.append(command)
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "trials.csv").touch()
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            run_rq(
                "rq3a",
                agents=("relational_ona",),
                seeds=(1,),
                results_dir=Path(temp_dir),
                ona_binary=Path("special/NAR-c256-t240"),
                ona_timeout=8.0,
                ona_startup_timeout=30.0,
                trials=(200, 900),
                execute=execute,
            )

        command = commands[0]
        self.assertEqual(
            command[command.index("--ona-binary") + 1],
            "special/NAR-c256-t240",
        )
        self.assertEqual(command[command.index("--workers") + 1], "3")
        self.assertEqual(command[command.index("--ona-timeout") + 1], "8.0")
        self.assertEqual(
            command[command.index("--ona-startup-timeout") + 1], "30.0"
        )
        self.assertEqual(command[command.index("--before-trials") + 1], "200")
        self.assertEqual(command[command.index("--mixed-trials") + 1], "900")

    def test_rq4_forwards_ona_binary_and_caps_ona_workers(self) -> None:
        commands: list[list[str]] = []

        def execute(command: list[str]) -> int:
            commands.append(command)
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "trials.csv").touch()
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            run_rq(
                "rq4a",
                agents=("relational_ona",),
                seeds=(1,),
                results_dir=Path(temp_dir),
                ona_binary=Path("special/NAR-c256-t240"),
                execute=execute,
            )

        command = commands[0]
        self.assertEqual(
            command[command.index("--ona-binary") + 1],
            "special/NAR-c256-t240",
        )
        self.assertEqual(command[command.index("--workers") + 1], "3")

    def test_single_agent_skips_comparison_reports(self) -> None:
        commands: list[list[str]] = []

        def execute(command: list[str]) -> int:
            commands.append(command)
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir()
            (output_dir / "trials.csv").touch()
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            run_rq(
                "rq4a",
                agents=("relational_ona",),
                seeds=(1,),
                results_dir=Path(temp_dir),
                execute=execute,
            )

        self.assertEqual(len(commands), 1)
        self.assertEqual(
            commands[0][commands[0].index("--agent") + 1],
            "relational_ona",
        )

    def test_stops_when_an_experiment_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                run_rq(
                    "rq2a",
                    agents=("flat_ona", "ucb1"),
                    seeds=(1,),
                    results_dir=Path(temp_dir),
                    execute=lambda command: 7,
                )


class SeedJobExecutionTests(unittest.TestCase):
    def test_parallel_jobs_are_isolated_and_returned_in_input_order(self) -> None:
        parent_pid = os.getpid()

        outcomes = run_jobs(identify_process, [3, 1, 2], workers=2)

        self.assertEqual([value for value, _ in outcomes], [3, 1, 2])
        self.assertTrue(all(pid != parent_pid for _, pid in outcomes))

    def test_effective_workers_are_capped_by_job_count(self) -> None:
        self.assertEqual(effective_worker_count(4, 2), 2)
        with self.assertRaisesRegex(ValueError, "job count"):
            effective_worker_count(4, 0)

    def test_worker_failure_is_propagated_without_partial_results(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "seed failed"):
            run_jobs(fail_on_two, [1, 2, 3], workers=2)


class ComparisonMainTests(unittest.TestCase):
    def test_main_runs_each_target_with_sequential_seeds(self) -> None:
        with patch("adaptive_sorting.experiments.run_rq_comparison.run_rq") as dispatch:
            main(["rq4b", "flat_ona", "relational_ona", "--seeds", "10"])
        self.assertEqual(dispatch.call_args.args[0], "rq4b")
        self.assertEqual(dispatch.call_args.args[1], ["flat_ona", "relational_ona"])
        self.assertEqual(dispatch.call_args.args[2], tuple(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
