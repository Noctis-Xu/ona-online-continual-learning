"""Experiment IDs and controller labels."""

EXPERIMENT_IDS = ("rq1", "rq2a", "rq2b", "rq3a", "rq3b", "rq4a", "rq4b")


def experiment_id(value: str) -> str:
    if value not in EXPERIMENT_IDS:
        raise ValueError(f"Unknown experiment: {value}")
    return value


def ona_label(*, relational: bool = False, variant: str | None = None) -> str:
    engine = "ONA-TIP" if variant == "ona-tip" else "ONA"
    return f"{engine} ({'relational' if relational else 'flat'})"
