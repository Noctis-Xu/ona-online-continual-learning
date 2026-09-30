"""Shared defaults; the ten default seeds are a quick check, the thesis uses 200."""

DEFAULT_SEED_COUNT = 10
DEFAULT_SEEDS = tuple(range(1, DEFAULT_SEED_COUNT + 1))
DEFAULT_RQ1_TRIALS = 1400
DEFAULT_INITIAL_TRIALS = 200
DEFAULT_SUBSEQUENT_TRIALS = 1200
DEFAULT_WORKERS = 3

PROTOCOL_VERSION = "unified-v3"
ENCODING_VERSION = "compact-symbols-v2"
