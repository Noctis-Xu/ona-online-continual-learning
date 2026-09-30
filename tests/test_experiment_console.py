from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

from adaptive_sorting.experiments.console import run_cli


class ExperimentConsoleTests(unittest.TestCase):
    def test_runtime_errors_are_highlighted_in_red(self) -> None:
        output = StringIO()

        def fail() -> None:
            raise RuntimeError("ONA stopped")

        with redirect_stderr(output), self.assertRaises(RuntimeError):
            run_cli(fail)

        self.assertIn("\033[1;31mERROR\033[0m: ONA stopped", output.getvalue())


if __name__ == "__main__":
    unittest.main()
