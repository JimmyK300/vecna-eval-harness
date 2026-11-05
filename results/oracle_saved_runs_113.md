# Saved-run oracle (113 queries)

**Question:** if we could always pick the best saved experiment arm for each query, how far could retrieval go? And which queries does *nothing* we tried retrieve?

**Setup:** this is a retrospective analysis over 23 complete per-query result files: modality/fusion runs, compound-query retrieval runs, compound packets and the historical reranker. Together they hold **153 (run, arm) combinations**. These are not 153 independent models, because control arms repeat. For each query and cutoff the oracle credits the best arm **using the answer key**. `vecna_eval/oracle.py` implements the same rule.

| Saved-arm union | Accepted video @20 | Accepted video @100 | Complete evidence @20 | Complete evidence @100 |
|---|---:|---:|---:|---:|
| Qwen full control | 84/113 | 91/113 | 61/113 | 78/113 |
| Six matched modality/fusion arms | 97/113 | 101/113 | 74/113 | 81/113 |
| Compound initial arms | 98/113 | 106/113 | 70/113 | 82/113 |
| Compound packet arms | 105/113 | 110/113 | 79/113 | 90/113 |
| **All comparable saved arms** | **108/113** | **111/113** | **87/113** | **94/113** |

- *Accepted video* means a gold video appears among the distinct videos. *Complete evidence* means every required frame range or event is represented. A correct video alone does not mean the answer-bearing moment was found.
- 5 queries miss the accepted video in the top 20 of every arm. 2 of them (`p3_q11`, `p3_q24`) are never seen even in the top 100.
- 19 queries never get complete required evidence in the top 100 of any arm.

**Reading:** for 21 queries, at least one saved arm finds the video in its top 20 while the best single fusion run does not (108 vs 87, see [fusion_sweep_113.md](fusion_sweep_113.md)). That is headroom for *selection or routing*, but only measured with the answer key. The remaining 5 are missed at @20 by every arm. Three of them surface at best ranks 24–37. The other two never appear in any top 100, which makes them *candidate-generation* failures. The arms were captured at different times, so the union is diagnostic, not a controlled comparison. The two 87/113 figures are different metrics: video R@20 for the fusion, complete evidence @20 for the oracle.

**This is an upper bound, not a result.** No implementable router reached it.

**Source:** `evaluation/saved-run-oracle-20260923-v1/README.md` in the full-text retirement archive of `official-dataset-control` (`Official-Dataset-2026-09-28-v1`, dated 2026-09-23). That folder was not pushed to GitHub, so this result has no public git source.
