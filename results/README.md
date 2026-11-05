# Results

These four results come from the VECNA evaluation work for AI Challenge HCMC 2026. Each number below is copied from the file named under **Source**. Nothing here was re-run for this repository. The contest videos, query text and ground truth cannot be redistributed, so the per-query evidence is referenced by query ID only (`p0_q02` = round P0, query 2).

| File | Experiment | Headline |
|---|---|---|
| [fusion_sweep_113.md](fusion_sweep_113.md) | Five fusion rules over identical provider lists | The current nested-max fusion leads: R@20 87/113, MRR@20 0.529989 |
| [reranker_audit_113.md](reranker_audit_113.md) | Qwen3-VL-Reranker-2B over the Qwen top-20 | R@20 68 → 76/113, but R@1 41 → 40, at 28.219 s/query |
| [oracle_saved_runs_113.md](oracle_saved_runs_113.md) | Gold-conditioned union of 153 saved (run, arm) pairs | Upper bound: 108/113 video @20, 87/113 complete evidence @20 |
| [clause_decomposition_19.md](clause_decomposition_19.md) | Clause-level BM25 + RRF vs the raw query | Rejected: R@20 13/19 → 9/19 |

All machine-readable values are in [results.json](results.json).

**Cohort.** There are 115 query identities across rounds P0 to P3. 113 are scoreable, because `p0_q15` and `p3_q09` are excluded. P0 to P2 use historical benchmark truth. P3 truth is provisional, made by the team rather than the organizers. A single shared-cohort benchmark like this reuses the same queries for every decision, so treat the paired intervals as descriptive.
