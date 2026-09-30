"""Consistent trial-axis spacing without changing statistical windows."""

from math import ceil

from matplotlib.ticker import MultipleLocator


def set_trial_ticks(axis, trial_count: int, *, max_labels: int | None = None) -> None:
    spacing = 100 if trial_count <= 2000 else ceil(trial_count / 1000) * 100
    if max_labels is not None:
        if max_labels < 1:
            raise ValueError("max_labels must be positive")
        spacing = max(spacing, ceil(trial_count / max_labels / 100) * 100)
    axis.xaxis.set_major_locator(MultipleLocator(spacing))
    if spacing > 100:
        axis.xaxis.set_minor_locator(MultipleLocator(100))
