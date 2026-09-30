from __future__ import annotations

import csv
import gzip
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from adaptive_sorting.analysis.evaluation import EvaluationConfig
from adaptive_sorting.analysis.rq3_relational_diagnostics import (
    SeedMemoryWriter,
    family_statistics,
    rule_expectation,
    seed_memory_paths,
    snapshot_trials,
    write_relational_diagnostics,
)


class RQ3RelationalDiagnosticsTests(unittest.TestCase):
    def test_snapshot_trials_cover_interval_and_phase_ends(self) -> None:
        self.assertEqual(snapshot_trials(200, 1400, 500), (200, 500, 1000, 1400))
        with self.assertRaises(ValueError):
            snapshot_trials(200, 1400, 0)

    def test_family_statistics_summarize_truth_values(self) -> None:
        self.assertAlmostEqual(rule_expectation("dt=2 <a =/> b>. {1.000000 0.800000}")[0], 0.9)
        rows = {r["rule_family"]: r for r in family_statistics(1, 7, (
            ("color", "r1. {1.000000 0.800000}"), ("color", "r2. {0.500000 0.600000}"), ("L1", "r3. {0.000000 0.500000}")))}
        self.assertEqual(rows["color"]["count"], 2)
        self.assertAlmostEqual(rows["color"]["max_expectation"], 0.9)
        self.assertAlmostEqual(rows["color"]["mean_expectation"], 0.7)
        self.assertAlmostEqual(rows["L1"]["mean_expectation"], 0.25)
        self.assertEqual(rows["L2"]["count"], 0)
        self.assertEqual(rows["L2"]["max_expectation"], "")

    def test_streamed_memory_files_feed_inventory_and_final_attribution(self) -> None:
        snapshots = {2: (("color", "c. {1.0 0.5}"),), 4: (("color+L1", "p. {1.0 0.9}"),), 6: (("color+L1+L2", "t. {1.0 0.9}"),)}
        rows = [
            self._row(1, "off_only", 1, "learned_rule", "color", True),
            self._row(2, "off_only", 2, "motor_babbling", "", False),
            self._row(3, "mixed_contexts", 1, "learned_rule", "L1", True),
            self._row(4, "mixed_contexts", 2, "learned_rule", "color+L1+L2", False),
            self._row(5, "mixed_contexts", 3, "unknown", "", False),
            self._row(6, "mixed_contexts", 4, "learned_rule", "color+L1", True),
        ]

        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with SeedMemoryWriter(run_dir / "memory", 1) as memory:
                for trial, rules in snapshots.items():
                    memory.record(trial, rules, full=True)
            diagnostics = [SimpleNamespace(
                seed=1,
                decisions=[{"seed": 1, "trial": trial, "implication": "rule"} for trial in range(1, 7)],
                snapshot_trials=[2, 4, 6],
            )]
            paths = write_relational_diagnostics(run_dir, diagnostics, rows, config=EvaluationConfig(2), snapshot_interval=2)
            with paths["decision_attribution"].open(newline="", encoding="utf-8") as handle:
                attribution = list(csv.DictReader(handle))
            with paths["retained_rule_inventory"].open(newline="", encoding="utf-8") as handle:
                inventory = list(csv.DictReader(handle))
            snapshot_path, _ = seed_memory_paths(run_dir / "memory", 1)
            with gzip.open(snapshot_path, "rt", encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 3)
            report = paths["relational_diagnostics_report"].read_text(encoding="utf-8")
            diagnostics[0].snapshot_trials = [2, 4]
            with self.assertRaisesRegex(ValueError, "snapshots"):
                write_relational_diagnostics(run_dir, diagnostics, rows, config=EvaluationConfig(2), snapshot_interval=2)

        self.assertEqual(len(inventory), 3 * 8)
        pair = next(row for row in inventory if row["trial"] == "4" and row["rule_family"] == "color+L1")
        self.assertEqual((pair["mean_count"], pair["mean_max_expectation"]), ("1.0", "0.95"))
        self.assertEqual({row["stage"] for row in attribution}, {"final_mixed"})
        relevant = next(row for row in attribution if row["decision_category"] == "relevant_only")
        self.assertEqual(relevant["trial_count"], "1")
        self.assertEqual(relevant["conditional_accuracy"], "1.0")
        self.assertEqual(sum(int(row["trial_count"]) for row in attribution), 2)
        self.assertIn("color+L1", report)
        self.assertIn("trial share / conditional accuracy", report)

    @staticmethod
    def _row(
        trial: int,
        phase: str,
        phase_trial: int,
        source: str,
        family: str,
        correct: bool,
    ) -> dict[str, object]:
        return {
            "seed": 1,
            "trial": trial,
            "phase": phase,
            "phase_trial": phase_trial,
            "mixed_phase_start": 3,
            "mixed_phase_trials": 4,
            "decision_source": source,
            "driving_rule_family": family,
            "is_correct": correct,
        }


if __name__ == "__main__":
    unittest.main()
