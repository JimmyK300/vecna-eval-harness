"""Gold-conditioned oracle over saved arms.

For each query and cutoff the oracle credits the best arm *using the answer
key*. It is an upper bound on what any selection or routing policy over those
arms could reach. It is not the score of an implementable method.
"""
from __future__ import annotations

from .scoring import KS


def oracle(ranks_by_arm: dict[str, dict[str, int | None]], ks=KS) -> dict:
    arms = list(ranks_by_arm)
    queries = sorted(next(iter(ranks_by_arm.values()))) if arms else []
    best = {}
    for qid in queries:
        observed = [ranks_by_arm[arm][qid] for arm in arms if ranks_by_arm[arm][qid] is not None]
        best[qid] = min(observed) if observed else None
    out = {"arms": arms, "n": len(queries)}
    for k in ks:
        out[f"recall_at_{k}"] = sum(1 for r in best.values() if r is not None and r <= k)
    out["never_found_at_max_k"] = sorted(q for q, r in best.items() if r is None or r > max(ks))
    return out
