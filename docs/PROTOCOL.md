# Evaluation protocol

These are the rules every VECNA retrieval experiment followed. The harness in `vecna_eval/` implements the scoring and fusion parts.

## Scoring contract

1. **Cohort.** There are 115 canonical query identities. Two (`p0_q15`, `p3_q09`) are excluded from numerator and denominator. Queries with empty rankings stay in the denominator.
2. **Unit.** The unit is the distinct video. A frame list is truncated to the output budget (100 frames) first, and then collapsed to the first frame of each video. Truncating before collapsing matters: a video whose only frame sits at position 101 is a miss.
3. **Metrics.** R@k for k ∈ {1, 5, 10, 20}, counted as successes out of 113, never rounded percentages. MRR is cut off at 20.
4. **Stricter surfaces are reported separately, never mixed.** These are video identity at raw frame positions, and the frozen range/event contract. For TRAKE, success needs *all* required events, not any one.
5. **Ground truth is scoring-only.** Fusion, reranking and query rewriting never read it. Anything that uses the answer key, such as an oracle or per-query best arm, is labeled an upper bound.

## Comparison rules

- **Freeze inputs, then vary one thing.** Provider lists are captured once (top 100 per provider) and replayed. Every arm sees identical hits, candidate identity and tie order.
- **Pre-declare the selection order.** The order is video R@20, then MRR@20, then R@1, then a fixed arm order. There is no per-query winner, weight search or temperature tuning on the evaluation cohort.
- **Report per-query movement.** Every comparison lists R@20 rescues and regressions by query ID, along with a paired bootstrap interval (10,000 resamples, seed 82).
- **One hypothesis per run.** A rejected idea is written up, and its temporary code is removed instead of being tuned on the same queries.

## Experiment archive rule

Each experiment is archived in a versioned, non-overwriting folder (`<slug>-vN/`) containing:

- the runnable code and tests, and the configuration;
- the inputs, or immutable paths plus SHA-256 hashes;
- the outputs, plus a `README` and `PROVENANCE` recording the source branch and commit, dirty state, the scoring/join contract, exclusions, reproduction commands and input/output hashes;
- whether inference, retrieval, corpus or index semantics changed.

An experiment is never called reproducible without its required inputs or hash-bound paths.
