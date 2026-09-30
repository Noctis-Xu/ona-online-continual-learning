from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from adaptive_sorting.analysis.plot_rq1 import load_trials, plot_report


class RQ1PlotTests(unittest.TestCase):
    def test_learning_curve_png_is_generated(self) -> None:
        rows = [
            [1, 1, "red", False],
            [1, 2, "blue", True],
            [1, 3, "red", True],
            [1, 4, "blue", True],
            [2, 1, "red", False],
            [2, 2, "blue", False],
            [2, 3, "red", True],
            [2, 4, "blue", True],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "trials.csv"
            output_path = Path(temp_dir) / "report.png"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["seed", "trial", "sphere_color", "is_correct"]
                )
                writer.writerows(rows)

            grouped = load_trials(csv_path)
            plot_report(
                grouped,
                output_path,
                rolling_window=2,
            )

            self.assertEqual(len(grouped), 2)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 10_000)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
