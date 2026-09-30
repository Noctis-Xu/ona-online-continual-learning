"""Wilson intervals for binary seed outcomes; seed bootstrap for other statistics."""
from __future__ import annotations
from functools import lru_cache
from statistics import NormalDist
import math
import numpy as np

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


@lru_cache(maxsize=8)
def bootstrap_weights(n: int, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=samples) / n


def bootstrap_interval(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None
    average = float(values.mean())
    if len(values) == 1:
        return average, None, None
    estimates = np.einsum('ij,j->i', bootstrap_weights(len(values), samples, seed), values)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return average, float(lower), float(upper)



def bootstrap_median_interval(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Median with a seed-bootstrap percentile interval; censored values enter as large ranks."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None
    median = float(np.median(values))
    if len(values) == 1:
        return median, None, None
    indices = np.random.default_rng(seed).integers(0, len(values), size=(samples, len(values)))
    lower, upper = np.quantile(np.median(values[indices], axis=1), [0.025, 0.975])
    return median, float(lower), float(upper)


def wilson_interval(values):
    """95% Wilson score interval for one binary observation per independent seed."""
    values = list(values)
    if not values:
        return None, None, None
    if any(value not in (0, 1) for value in values):
        raise ValueError('Wilson intervals require binary seed outcomes.')
    n = len(values)
    p = sum(values) / n
    z = NormalDist().inv_cdf(.975)
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return p, max(0., center-half), min(1., center+half)


def curve_summary(curves):
    if not curves or len({len(c) for c in curves}) != 1:
        raise ValueError("Seed curves must have the same nonempty trial horizon.")
    values = np.asarray(curves, dtype=float)
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    average = np.divide(np.where(valid, values, 0).sum(axis=0), counts,
                        out=np.full(values.shape[1], np.nan), where=counts > 0)
    lower = np.full(len(average), np.nan)
    upper = lower.copy()
    # Group columns with identical available seeds to bootstrap independent runs.
    masks, inverse = np.unique(valid.T, axis=0, return_inverse=True)
    for index, mask in enumerate(masks):
        n = int(mask.sum())
        if n < 2:
            continue
        columns = np.flatnonzero(inverse == index)
        weights = bootstrap_weights(n, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
        for start in range(0, len(columns), 128):
            selected = columns[start:start+128]
            estimates = np.einsum('ij,jk->ik', weights, values[np.ix_(mask, selected)])
            lower[selected], upper[selected] = np.quantile(estimates, [.025, .975], axis=0)
    return average.tolist(), lower.tolist(), upper.tolist()
