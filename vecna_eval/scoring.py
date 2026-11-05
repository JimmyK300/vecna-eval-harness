"""Distinct-video Recall@k and MRR@k.

A query succeeds at k when any accepted video appears among the first k
distinct videos. Excluded queries are dropped from both numerator and
denominator; queries with empty rankings stay in the denominator.
"""
from __future__ import annotations

KS = (1, 5, 10, 20)
MRR_CUTOFF = 20


def first_correct_rank(videos: list[str], accepted: set[str]) -> int | None:
    for rank, video in enumerate(videos, 1):
        if video in accepted:
            return rank
    return None


def scored_queries(truth: dict) -> list[str]:
    return sorted(qid for qid, row in truth.items() if not row["excluded"])


def summarize(ranks: dict[str, int | None], ks=KS, mrr_cutoff=MRR_CUTOFF) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in ks:
        out[f"recall_at_{k}"] = sum(1 for r in ranks.values() if r is not None and r <= k)
    reciprocal = [1.0 / r if r is not None and r <= mrr_cutoff else 0.0 for r in ranks.values()]
    out[f"mrr_at_{mrr_cutoff}"] = round(sum(reciprocal) / n, 6) if n else 0.0
    return out


def rescues_and_regressions(base: dict, alt: dict, k: int) -> tuple[list[str], list[str]]:
    hit = lambda r: r is not None and r <= k  # noqa: E731
    rescues = sorted(q for q in base if not hit(base[q]) and hit(alt[q]))
    regressions = sorted(q for q in base if hit(base[q]) and not hit(alt[q]))
    return rescues, regressions
