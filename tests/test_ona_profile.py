from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adaptive_sorting.experiments import ona_profile

from adaptive_sorting.experiments.ona_profile import (
    binary_provenance,
    profile_manifest_path,
    replace_config_macro,
)


class ONAProfileTests(unittest.TestCase):
    def test_source_hash_ignores_build_outputs_but_tracks_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "third_party/OpenNARS-for-Applications/src/Config.h"
            source.parent.mkdir(parents=True)
            source.write_text("#define TABLE_SIZE 120\n")
            names = [str(source.relative_to(root))]
            with patch.object(ona_profile, "PROJECT_ROOT", root), patch.object(
                ona_profile, "_run"
            ) as run:
                run.return_value.stdout = "\0".join(names)
                original = ona_profile._tracked_source_hash()
                for name in ("NAR-tip-c256-t240", "NAR-tip-c256-t240.profile.json"):
                    output = source.parent.parent / name
                    output.write_bytes(b"build output")
                    names.append(str(output.relative_to(root)))
                run.return_value.stdout = "\0".join(names)
                self.assertEqual(original, ona_profile._tracked_source_hash())
                source.write_text("#define TABLE_SIZE 240\n")
                self.assertNotEqual(original, ona_profile._tracked_source_hash())

    def test_replaces_one_integer_macro(self) -> None:
        config = "#define TABLE_SIZE 120\n#define OTHER 4\n"

        self.assertEqual(
            replace_config_macro(config, "TABLE_SIZE", 240),
            "#define TABLE_SIZE 240\n#define OTHER 4\n",
        )

    def test_rejects_missing_or_duplicate_macro(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0"):
            replace_config_macro("#define OTHER 1\n", "TABLE_SIZE", 240)
        with self.assertRaisesRegex(ValueError, "found 2"):
            replace_config_macro(
                "#define TABLE_SIZE 120\n#define TABLE_SIZE 60\n",
                "TABLE_SIZE",
                240,
            )

    def test_custom_binary_records_hash_and_unknown_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "NAR"
            binary.write_bytes(b"custom binary")

            metadata = binary_provenance(binary)

        self.assertEqual(
            metadata["binary_sha256"],
            hashlib.sha256(b"custom binary").hexdigest(),
        )
        self.assertIsNone(metadata["build_profile"])
        self.assertIsNone(metadata["compile_time_macros"])

    def test_profile_manifest_is_loaded_after_hash_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "NAR"
            binary.write_bytes(b"profile binary")
            digest = hashlib.sha256(b"profile binary").hexdigest()
            profile_manifest_path(binary).write_text(
                json.dumps(
                    {
                        "profile": "test-profile",
                        "compile_time_macros": {
                            "COMPOUND_TERM_SIZE_MAX": 256,
                            "TABLE_SIZE": 240,
                        },
                        "binary": {"sha256": digest},
                    }
                ),
                encoding="utf-8",
            )

            metadata = binary_provenance(binary)

        self.assertEqual(metadata["build_profile"], "test-profile")
        self.assertEqual(
            metadata["compile_time_macros"],
            {"COMPOUND_TERM_SIZE_MAX": 256, "TABLE_SIZE": 240},
        )

    def test_profile_manifest_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "NAR"
            binary.write_bytes(b"changed")
            profile_manifest_path(binary).write_text(
                json.dumps({"binary": {"sha256": "stale"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                binary_provenance(binary)


if __name__ == "__main__":
    unittest.main()
