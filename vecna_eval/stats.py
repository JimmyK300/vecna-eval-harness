"""Paired bootstrap over queries."""
from __future__ import annotations

import random


def paired_bootstrap(deltas: list[float], resamples: int = 10_000, seed: int = 82,
                     alpha: float = 0.05) -> dict:
    """Percentile interval for the mean of per-query paired differences."""
    n = len(deltas)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "resamples": 0}
    rng = random.Random(seed)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low = means[int((alpha / 2) * resamples)]
    high = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return {"mean": round(sum(deltas) / n, 6), "low": round(low, 6), "high": round(high, 6),
            "resamples": resamples, "seed": seed}
