"""Shell-level tests of the ONA-TIP executable (ONA_TIP_BINARY or the default build)."""

import os
from pathlib import Path
import re
import subprocess
import unittest

from adaptive_sorting.experiments.ona_profile import ONA_SOURCE_DIR

TIP_BINARY = Path(os.environ.get("ONA_TIP_BINARY", ONA_SOURCE_DIR / "NAR-tip-c256-t240"))


@unittest.skipUnless(TIP_BINARY.is_file(), "requires an ONA-TIP build")
class ONATIPTests(unittest.TestCase):
    def run_shell(self, commands):
        result = subprocess.run(
            [str(TIP_BINARY.resolve()), "shell"],
            input="\n".join(["*volume=0", *commands, "quit", ""]),
            capture_output=True, text=True, check=True, timeout=30,
        )
        return [(float(f), float(c)) for f, c in re.findall(
            r"^Answer: .*frequency=([0-9.]+), confidence=([0-9.]+)$", result.stdout, re.M
        )]

    def test_projection_halves_confidence_and_queries_do_not_refresh_it(self):
        half_life = 10000
        for rule in ("<signal =/> outcome>", "<(signal &/ ^respond) =/> outcome>"):
            with self.subTest(rule=rule):
                before, first, repeated, later = self.run_shell([
                    f"{rule}. {{0.8 0.9}}", f"{rule}?", str(half_life),
                    f"{rule}?", f"{rule}?", str(half_life), f"{rule}?",
                ])
                self.assertEqual(first, repeated)
                self.assertEqual(before[0], first[0])
                self.assertAlmostEqual(first[1] / before[1], 0.5, places=5)
                self.assertAlmostEqual(later[1] / before[1], 0.25, places=5)

    def test_revision_uses_projected_evidence(self):
        half_life = 10000
        rule = "<(signal &/ ^respond) =/> outcome>"
        [(frequency, confidence)] = self.run_shell([
            f"{rule}. {{1 0.9}}", str(half_life), f"{rule}. {{0 0.9}}", f"{rule}?",
        ])
        self.assertAlmostEqual(frequency, 1 / 12, delta=0.001)
        self.assertGreater(confidence, 0.89)

    def test_declarative_beliefs_are_outside_rule_projection(self):
        half_life = 10000
        for rule in ("<signal --> category>", "<signal ==> outcome>"):
            with self.subTest(rule=rule):
                before, after = self.run_shell([
                    f"{rule}. {{0.8 0.9}}", f"{rule}?", str(half_life), f"{rule}?",
                ])
                self.assertEqual(before, after)
