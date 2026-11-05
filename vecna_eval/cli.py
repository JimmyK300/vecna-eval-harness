"""Command line: run every fusion arm, score it, and compare against a control.

    python -m vecna_eval --rankings data/synthetic/provider_rankings.jsonl \
                         --truth data/synthetic/ground_truth.jsonl \
                         --out runs/synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import fusion, oracle, scoring, stats
from .io import ContractError, load_rankings, load_truth, write_json


def evaluate(rankings: dict, truth: dict, config: dict, control: str = "current",
             baseline_provider: str | None = "qwen", resamples: int = 10_000) -> dict:
    config = fusion.validate_config(config)
    queries = scoring.scored_queries(truth)
    missing = [q for q in queries if q not in rankings]
    ranks_by_arm: dict[str, dict[str, int | None]] = {}
    for arm in fusion.ARMS:
        ranks_by_arm[arm] = {
            q: scoring.first_correct_rank(fusion.fuse(rankings.get(q, {}), arm, config), truth[q]["accepted"])
            for q in queries
        }
    if baseline_provider:
        name = f"{baseline_provider}_only"
        ranks_by_arm[name] = {
            q: scoring.first_correct_rank(
                fusion.single_provider(rankings.get(q, {}), baseline_provider, config["output_frame_k"]),
                truth[q]["accepted"])
            for q in queries
        }
    summary = {arm: scoring.summarize(r) for arm, r in ranks_by_arm.items()}

    comparisons = {}
    base = ranks_by_arm[control]
    for arm, alt in ranks_by_arm.items():
        if arm == control:
            continue
        rescues, regressions = scoring.rescues_and_regressions(base, alt, 20)
        hit = lambda r: 1.0 if r is not None and r <= 20 else 0.0  # noqa: E731
        rr = lambda r: 1.0 / r if r is not None and r <= 20 else 0.0  # noqa: E731
        comparisons[arm] = {
            "vs": control,
            "r20_rescues": rescues,
            "r20_regressions": regressions,
            "r20_delta": stats.paired_bootstrap([hit(alt[q]) - hit(base[q]) for q in queries], resamples),
            "mrr20_delta": stats.paired_bootstrap([rr(alt[q]) - rr(base[q]) for q in queries], resamples),
        }
    return {
        "config": config,
        "scored_queries": len(queries),
        "excluded_queries": sorted(q for q, row in truth.items() if row["excluded"]),
        "queries_without_rankings": missing,
        "summary": summary,
        "comparisons": comparisons,
        "oracle_over_fusion_arms": oracle.oracle({a: ranks_by_arm[a] for a in fusion.ARMS}),
        "per_query_first_correct_rank": {q: {a: ranks_by_arm[a][q] for a in ranks_by_arm} for q in queries},
    }


def render_table(result: dict) -> str:
    rows = ["| Arm | R@1 | R@5 | R@10 | R@20 | MRR@20 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, s in result["summary"].items():
        n = s["n"]
        rows.append(f"| {arm} | {s['recall_at_1']}/{n} | {s['recall_at_5']}/{n} | "
                    f"{s['recall_at_10']}/{n} | {s['recall_at_20']}/{n} | {s['mrr_at_20']:.6f} |")
    o = result["oracle_over_fusion_arms"]
    rows.append(f"| oracle (upper bound) | {o['recall_at_1']}/{o['n']} | {o['recall_at_5']}/{o['n']} | "
                f"{o['recall_at_10']}/{o['n']} | {o['recall_at_20']}/{o['n']} | n/a |")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vecna_eval", description=__doc__.splitlines()[0])
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="JSON overriding fusion defaults")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--control", default="current", choices=fusion.ARMS)
    parser.add_argument("--baseline-provider", default="qwen")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        result = evaluate(load_rankings(args.rankings), load_truth(args.truth), config,
                          args.control, args.baseline_provider or None, args.resamples)
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(args.out / "result.json", result)
    table = render_table(result)
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    print(table, end="")
    return 0
