# Upstream Source

- Repository: <https://github.com/opennars/OpenNARS-for-Applications>
- Release: `v0.9.3`
- Commit: `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a`

## Local modifications

- `NAR shell --seed N` sets the pseudorandom seed of a run (`src/main.c`).
- ONA-TIP, ONA with temporal implication projection, is the compile-time option
  `TEMPORAL_IMPLICATION_PROJECTION` (`src/Config.h`). It is off by default, so the
  default build keeps the original ONA behavior. When on, reading a temporal
  implication returns a copy whose confidence is projected to the current time
  with a half-life of 10,000 NAR cycles. Decisions, predictions, and revision use
  the projected values. Unit tests are in `src/unit_tests/TIProjection_Test.h`.

Build both engines with `python -m adaptive_sorting.experiments.ona_profile`
(see the repository README).
