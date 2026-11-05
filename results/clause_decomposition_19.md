# Clause decomposition: rejected (19 queries)

**Hypothesis:** long Vietnamese queries stack many visual constraints, so a high-information clause gets diluted. The idea was to search the raw query and each existing atomic clause separately over the semantic index (BM25), collapse each result list to its best video, and fuse the lists with RRF (k = 60). Duplicate normalized views get one vote.

**Setup:** this used the 19 validated queries that have semantic-index coverage. The experiment was run once, with no parameter sweep, label change or index edit.

| Mode | MRR | R@1 | R@5 | R@10 | R@20 | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Raw query (baseline) | 0.421487 | 7/19 | 10/19 | 11/19 | **13/19** | 5 |
| Atomic-clause RRF | 0.275518 | 4/19 | 7/19 | 8/19 | **9/19** | 30 |

Clause RRF improved 3 queries, regressed 12 and left 4 unchanged. One regression moved a target from rank 10 to rank 137.

**Why it failed (from the experiment's own analysis):** the idea did work where it was supposed to. For a query whose exact topic was buried under other constraints, the isolated clause matched the stored record and jumped up the list. But the regressions were systematic. With equal voting, generic or visually specific clauses pull in many unrelated records. Splitting a query also destroys the useful conjunction in queries that were already succeeding, so an exact-topic hit can be outvoted by generic clauses. Some baseline misses could not be fixed by any query rewriting, because the requested visual detail was simply absent from the stored semantic text. That is a coverage problem, not a query problem.

With only 19 queries (14 TKIS, 3 TRAKE, 2 QA), tuning clause weights was not justified. The evidence rejects this unweighted multi-view change. It does not show that all retrieval fusion is harmful. Another variant concatenated the clauses back onto the raw query. It left R@20 at 13/19 and dropped R@5 from 10/19 to 9/19.

**Source:** `evaluation/queries/benchmark/semantic-index-bm25-atomic-rrf-v1/README.md` on branch `codex/semantic-retrieval-high-recall` (commit `13fadfa`) of the private `official-dataset-control` history. The concatenated variant is in `evaluation/queries/benchmark/semantic-index-bm25-v0/README.md` on `main`. R@k counts were converted from fractions of 19.
