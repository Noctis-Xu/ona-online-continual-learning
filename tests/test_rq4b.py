from __future__ import annotations

import unittest

from adaptive_sorting.experiments.run_rq4a import build_parser


class RQ4bTests(unittest.TestCase):
    def test_shared_parser_exposes_the_complete_agent_suite(self) -> None:
        parser = build_parser()

        for agent in (
            "flat_ona",
            "relational_ona",
            "epsilon_greedy",
            "ucb1",
            "sw_ucb",
        ):
            with self.subTest(agent=agent):
                self.assertEqual(parser.parse_args(["--agent", agent]).agent, agent)


if __name__ == "__main__":
    unittest.main()
