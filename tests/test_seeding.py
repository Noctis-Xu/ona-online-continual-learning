from __future__ import annotations

import unittest

from adaptive_sorting.experiments.seeding import (
    SEED_SCHEME,
    derive_replicate_seeds,
    replicate_seed_metadata,
)


class ReplicateSeedingTests(unittest.TestCase):
    def test_scheme_keeps_known_seed_mapping_stable(self) -> None:
        seeds = derive_replicate_seeds(1)

        self.assertEqual(seeds.environment, 1_112_293_818)
        self.assertEqual(seeds.context_schedule, 4_066_426_003)
        self.assertEqual(seeds.distractor_schedule, 650_560_432)
        self.assertEqual(seeds.agent, 920_433_132)

    def test_derivation_is_reproducible_and_separates_streams(self) -> None:
        first = derive_replicate_seeds(7)
        second = derive_replicate_seeds(7)

        self.assertEqual(first, second)
        self.assertEqual(
            len(
                {
                    first.environment,
                    first.context_schedule,
                    first.distractor_schedule,
                    first.agent,
                }
            ),
            4,
        )

    def test_different_replicates_derive_different_streams(self) -> None:
        first = derive_replicate_seeds(7)
        second = derive_replicate_seeds(8)

        self.assertNotEqual(first.environment, second.environment)
        self.assertNotEqual(first.context_schedule, second.context_schedule)
        self.assertNotEqual(first.distractor_schedule, second.distractor_schedule)
        self.assertNotEqual(first.agent, second.agent)

    def test_derived_seeds_fit_ona_unsigned_32_bit_rng(self) -> None:
        seeds = derive_replicate_seeds(2**63)

        for value in (
            seeds.environment,
            seeds.context_schedule,
            seeds.distractor_schedule,
            seeds.agent,
        ):
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 2**32 - 1)

    def test_negative_replicate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            derive_replicate_seeds(-1)

    def test_metadata_records_concrete_seeds_and_active_streams(self) -> None:
        metadata = replicate_seed_metadata([1], uses_context_schedule=True)
        seeds = derive_replicate_seeds(1)

        self.assertEqual(metadata["seeds"], [1])
        self.assertEqual(metadata["seed_scheme"], SEED_SCHEME)
        self.assertEqual(
            metadata["active_seed_streams"],
            ["environment", "context_schedule", "agent"],
        )
        self.assertEqual(
            metadata["replicates"],
            [
                {
                    "seed": 1,
                    "environment_seed": seeds.environment,
                    "context_schedule_seed": seeds.context_schedule,
                    "agent_seed": seeds.agent,
                }
            ],
        )

    def test_metadata_marks_context_schedule_inactive_when_unused(self) -> None:
        metadata = replicate_seed_metadata([1])

        self.assertEqual(
            metadata["active_seed_streams"],
            ["environment", "agent"],
        )

    def test_metadata_records_distractor_schedule_only_when_active(self) -> None:
        metadata = replicate_seed_metadata(
            [1],
            uses_context_schedule=True,
            uses_distractor_schedule=True,
        )
        seeds = derive_replicate_seeds(1)

        self.assertEqual(
            metadata["active_seed_streams"],
            ["environment", "context_schedule", "distractor_schedule", "agent"],
        )
        self.assertEqual(
            metadata["replicates"][0]["distractor_schedule_seed"],
            seeds.distractor_schedule,
        )


if __name__ == "__main__":
    unittest.main()
