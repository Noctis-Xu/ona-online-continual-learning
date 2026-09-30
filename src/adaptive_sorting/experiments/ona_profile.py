"""Build and identify reproducible ONA binary profiles."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ONA_SOURCE_DIR = PROJECT_ROOT / "third_party" / "OpenNARS-for-Applications"
BUILD_STAGING_ROOT = PROJECT_ROOT / "tmp" / "ona-build"
ONA_VARIANTS = ("ona", "ona-tip")
TIP_HALF_LIFE_CYCLES = 10000
ONA_C256_T240_COMPOUND_TERM_SIZE = 256
ONA_C256_T240_TABLE_SIZE = 240
ONA_C256_T240_BUILD_ARGUMENTS = ("-mcmodel=large",)
ONA_C256_T240_MAX_WORKERS = 3
PROFILE_MANIFEST_SUFFIX = ".profile.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_manifest_path(binary_path: str | Path) -> Path:
    path = Path(binary_path)
    return path.with_name(path.name + PROFILE_MANIFEST_SUFFIX)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def replace_config_macro(content: str, name: str, value: int) -> str:
    """Replace exactly one integer macro definition in an ONA Config.h."""

    pattern = re.compile(rf"^#define\s+{re.escape(name)}\s+\d+.*$", re.MULTILINE)
    updated, replacements = pattern.subn(f"#define {name} {value}", content)
    if replacements != 1:
        raise ValueError(f"Expected exactly one {name} definition; found {replacements}.")
    return updated


def _tracked_source_hash() -> str:
    listed = _run(
        ["git", "ls-files", "-z", "third_party/OpenNARS-for-Applications"],
        cwd=PROJECT_ROOT,
    ).stdout
    digest = hashlib.sha256()
    for relative_name in sorted(name for name in listed.split("\0") if name):
        path = PROJECT_ROOT / relative_name
        # Relocated build outputs are not inputs to the source identity.
        if path.name == "NAR" or path.name.startswith("NAR-") or path.name in (
            "NAR.profile.json", "RuleTable.c"
        ):
            continue
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_profile(
    *,
    profile_name: str | None = None,
    compound_term_size: int = ONA_C256_T240_COMPOUND_TERM_SIZE,
    table_size: int = ONA_C256_T240_TABLE_SIZE,
    output_path: Path | None = None,
    variant: str = "ona",
    build_arguments: Sequence[str] = ONA_C256_T240_BUILD_ARGUMENTS,
    force: bool = False,
) -> Path:
    """Build ONA in an isolated copy and return the resulting binary path."""

    if compound_term_size <= 0 or table_size <= 0:
        raise ValueError("ONA profile capacities must be positive.")
    if variant not in ONA_VARIANTS:
        raise ValueError(f"Unknown ONA variant: {variant}")
    if any("TEMPORAL_IMPLICATION_PROJECTION" in arg for arg in build_arguments):
        raise ValueError("Select temporal implication projection with --variant.")
    tip = variant == "ona-tip"
    build_arguments = (*build_arguments, f"-DTEMPORAL_IMPLICATION_PROJECTION={int(tip)}")
    capacities = f"c{compound_term_size}-t{table_size}"
    if profile_name is None:
        profile_name = f"{variant}-{capacities}"
    if output_path is None:
        output_path = ONA_SOURCE_DIR / f"NAR{'-tip' if tip else ''}-{capacities}"
    destination = output_path.resolve()
    destination_manifest = profile_manifest_path(destination)
    existing_outputs = [
        path for path in (destination, destination_manifest) if path.exists()
    ]
    if existing_outputs and not force:
        raise FileExistsError(
            f"Profile output already exists: {existing_outputs[0]}. "
            "Pass --force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    BUILD_STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{profile_name}-", dir=BUILD_STAGING_ROOT
    ) as raw:
        staging = Path(raw) / "source"
        shutil.copytree(
            ONA_SOURCE_DIR,
            staging,
            ignore=shutil.ignore_patterns(
                ".git", "NAR", "NAR-*", "RuleTable.c"
            ),
        )
        config_path = staging / "src" / "Config.h"
        config = config_path.read_text(encoding="utf-8")
        config = replace_config_macro(
            config, "COMPOUND_TERM_SIZE_MAX", compound_term_size
        )
        config = replace_config_macro(config, "TABLE_SIZE", table_size)
        config_path.write_text(config, encoding="utf-8")

        # The upstream script removes these files before enabling `set -e`.
        (staging / "NAR").touch()
        (staging / "src" / "RuleTable.c").touch()
        try:
            _run(["sh", "build.sh", *build_arguments], cwd=staging)
        except subprocess.CalledProcessError as exc:
            details = "\n".join(
                part.strip() for part in (exc.stdout, exc.stderr) if part.strip()
            )
            raise RuntimeError(f"ONA profile build failed:\n{details}") from exc
        compiler_version = _run(["gcc", "--version"], cwd=staging).stdout.splitlines()[0]
        git_commit = _run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).stdout.strip()

        binary_path = Path(raw) / destination.name
        shutil.copy2(staging / "NAR", binary_path)
        manifest = {
            "schema_version": 1,
            "profile": profile_name,
            "variant": variant,
            "tip_half_life_cycles": TIP_HALF_LIFE_CYCLES if tip else None,
            "compile_time_macros": {
                "COMPOUND_TERM_SIZE_MAX": compound_term_size,
                "TABLE_SIZE": table_size,
            },
            "source": {
                "path": str(ONA_SOURCE_DIR.resolve()),
                "project_git_commit": git_commit,
                "tracked_tree_sha256": _tracked_source_hash(),
            },
            "compiler": compiler_version,
            "build_command": ["sh", "build.sh", *build_arguments],
            "build_arguments": list(build_arguments),
            "build_script_sha256": sha256_file(staging / "build.sh"),
            "binary": {
                "path": str(destination),
                "sha256": sha256_file(binary_path),
            },
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        temporary_manifest = profile_manifest_path(binary_path)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        binary_path.replace(destination)
        temporary_manifest.replace(destination_manifest)

    return destination


def binary_provenance(binary_path: str | Path) -> dict[str, object]:
    """Return verified binary identity and optional build-manifest details."""

    path = Path(binary_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONA binary not found: {path}")
    digest = sha256_file(path)
    manifest_path = profile_manifest_path(path)
    if not manifest_path.is_file():
        return {
            "binary_path": str(path),
            "binary_sha256": digest,
            "build_profile": None,
            "compile_time_macros": None,
            "compiler": None,
            "build_command": None,
            "build_arguments": None,
            "profile_manifest": None,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_binary = manifest.get("binary")
    if not isinstance(recorded_binary, dict):
        raise ValueError(f"Invalid ONA profile manifest: {manifest_path}")
    recorded_hash = recorded_binary.get("sha256")
    if recorded_hash != digest:
        raise ValueError(
            f"ONA profile manifest hash mismatch for {path}: "
            f"recorded {recorded_hash!r}, actual {digest!r}."
        )
    return {
        "binary_path": str(path),
        "binary_sha256": digest,
        "build_profile": manifest.get("profile"),
        "variant": manifest.get("variant"),
        "tip_half_life_cycles": manifest.get("tip_half_life_cycles"),
        "compile_time_macros": manifest.get("compile_time_macros"),
        "compiler": manifest.get("compiler"),
        "build_command": manifest.get("build_command"),
        "build_arguments": manifest.get("build_arguments"),
        "build_script_sha256": manifest.get("build_script_sha256"),
        "profile_manifest": str(manifest_path.resolve()),
        "source": manifest.get("source"),
        "built_at": manifest.get("built_at"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ONA_VARIANTS, default="ona")
    parser.add_argument("--profile", help="defaults to VARIANT-cSIZE-tSIZE")
    parser.add_argument(
        "--compound-term-size",
        type=int,
        default=ONA_C256_T240_COMPOUND_TERM_SIZE,
    )
    parser.add_argument(
        "--table-size", type=int, default=ONA_C256_T240_TABLE_SIZE
    )
    parser.add_argument("--output", type=Path, help="defaults to a distinct path for the selected variant and capacities")
    parser.add_argument(
        "--build-argument",
        action="append",
        dest="build_arguments",
        help="extra compiler argument passed through the upstream build script",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    binary = build_profile(
        profile_name=args.profile,
        variant=args.variant,
        compound_term_size=args.compound_term_size,
        table_size=args.table_size,
        output_path=args.output,
        build_arguments=(
            args.build_arguments
            if args.build_arguments is not None
            else ONA_C256_T240_BUILD_ARGUMENTS
        ),
        force=args.force,
    )
    print(f"binary={binary}")
    print(f"manifest={profile_manifest_path(binary)}")


if __name__ == "__main__":
    main()
