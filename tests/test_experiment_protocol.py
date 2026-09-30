from dataclasses import dataclass
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_sorting.experiments import run_rq1, run_rq2a, run_rq2b, run_rq3a, run_rq4a, run_rq4b
from adaptive_sorting.experiments.run_rq_comparison import parse_args
from adaptive_sorting.experiments.runner import run_jobs
from adaptive_sorting.agents.ona_agent import DEFAULT_ONA_BINARY
from adaptive_sorting.experiments.agent_factory import agent_worker_count
from adaptive_sorting.experiments.result_paths import DEFAULT_RESULTS_DIR
from adaptive_sorting.analysis.trial_ticks import set_trial_ticks


@dataclass
class CheckpointJob:
    seed: int
    fail: bool = False


@dataclass
class CheckpointSummary:
    seed: int
    accuracy: float


@dataclass
class CheckpointDiagnostics:
    seed: int
    decisions: list
    snapshots: dict


def checkpoint_worker(job):
    if job.fail:
        raise RuntimeError("deliberate failure")
    return (
        CheckpointSummary(job.seed, 1.0),
        [{"seed": job.seed, "trial": 1, "is_correct": True}],
        CheckpointDiagnostics(job.seed, [{"source": "learned_rule"}], {1: ["raw rule"]}),
    )


class UnifiedProtocolTests(unittest.TestCase):
    def test_direct_runners_share_defaults(self):
        for module, first, first_trials, second in (
            (run_rq1, "trials", 1400, None),
            (run_rq2a, "before_trials", 200, "after_trials"),
            (run_rq2b, "initial_trials", 200, "phase_trials"),
            (run_rq3a, "before_trials", 200, "mixed_trials"),
            (run_rq4a, "base_trials", 200, "expanded_trials"),
        ):
            with self.subTest(module=module.__name__), patch("sys.argv", [module.__name__]):
                args = module.parse_args()
                self.assertEqual(args.seeds, list(range(1, 11)))
                self.assertEqual(getattr(args, first), first_trials)
                if second:
                    self.assertEqual(getattr(args, second), 1200)
                self.assertEqual(args.ona_binary, DEFAULT_ONA_BINARY)
                self.assertEqual(args.results_dir, DEFAULT_RESULTS_DIR)
        self.assertEqual(DEFAULT_ONA_BINARY.name, "NAR-c256-t240")
        self.assertEqual(DEFAULT_RESULTS_DIR.name, "experiment_logs")
        self.assertEqual(parse_args(["all"]).seed_count, 10)

    def test_ona_concurrency_cap_applies_to_both_encodings(self):
        for agent in ("flat_ona", "relational_ona", "ucb1"):
            args = run_rq3a.parse_args(["--agent", agent, "--workers", "10"])
            self.assertEqual(agent_worker_count(args), 10 if agent == "ucb1" else 3)

    def test_rq4_entry_points_select_the_new_visibility_conditions(self):
        for module, visible, name in ((run_rq4a, False, "rq4a"), (run_rq4b, True, "rq4b")):
            with patch("sys.argv", [name]), patch.object(module, "run_experiment") as run:
                module.main()
                self.assertEqual(run.call_args.kwargs["bin_colors_visible"], visible)
                self.assertEqual(run.call_args.kwargs["research_question"], name)

    def test_completed_seed_keeps_full_data_when_another_job_fails(self):
        for workers, jobs, completed_index in (
            (1, [CheckpointJob(1), CheckpointJob(2, True)], 1),
            (2, [CheckpointJob(1, True), CheckpointJob(2)], 2),
        ):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                    run_jobs(checkpoint_worker, jobs, workers, checkpoint_dir=directory)
                checkpoint = directory / "seeds" / f"job_{completed_index:04d}_seed_{completed_index}"
                self.assertTrue((checkpoint / "completed.json").is_file())
                with (checkpoint / "trials.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]["trial"], "1")
                self.assertTrue((checkpoint / "summary.csv").is_file())
                diagnostics = json.loads((checkpoint / "diagnostics.json").read_text())
                self.assertEqual(diagnostics["snapshots"]["1"], ["raw rule"])
                self.assertEqual(diagnostics["decisions"][0]["source"], "learned_rule")
                self.assertEqual(json.loads((directory / "metadata.json").read_text())["status"], "failed")

    def test_trial_axes_use_100_steps_and_sparse_recurring_labels(self):
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots()
        try:
            set_trial_ticks(axis, 1400)
            ticks = axis.xaxis.get_major_locator().tick_values(0, 1400)
            self.assertEqual(ticks[1] - ticks[0], 100)
            set_trial_ticks(axis, 9800)
            ticks = axis.xaxis.get_major_locator().tick_values(0, 9800)
            self.assertEqual(ticks[1] - ticks[0], 1000)
            minor = axis.xaxis.get_minor_locator().tick_values(0, 9800)
            self.assertEqual(minor[1] - minor[0], 100)
        finally:
            plt.close(figure)
