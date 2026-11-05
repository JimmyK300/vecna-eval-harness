# Reranker audit (113 queries)

**Question:** should Qwen3-VL-Reranker-2B, which reorders the Qwen-only top-20 candidates, be integrated unconditionally?

**Setup:** the saved historical Qwen-only rankings and reranker outputs were rescored with a frozen dependency-free scorer. Video truth is phase-aware, and TRAKE success requires all required events. The same 113-query cohort is used, with `p0_q15` and `p3_q09` excluded. The reranker cannot add videos that are not already in the Qwen top-20.

| Metric | Qwen only | + Reranker |
|---|---:|---:|
| R@1 | 41/113 | 40/113 |
| R@5 | 56/113 | 61/113 |
| R@10 | 65/113 | 64/113 |
| **R@20** | **68/113** | **76/113** |
| MRR@20 | 0.424340 | 0.432021 |

- 9 R@20 rescues and 1 regression (`p2_q14`). The paired bootstrap (10,000 resamples, seed 82) gives R@20 +1.77 to +12.39 pp and MRR −0.0659 to +0.0851.
- **Latency:** active reranking calls average **28.219 s per query** (p50 28.049 s, p95 30.742 s) on an AMD RX 6900 XT via DirectML, with one known CPU fallback op. Cold load and warm-up could not be separated.
- The artifact named `top100` actually keeps 20 reranked candidates per query, and all 2,280 are traceable to baseline candidates.

**Decision:** keep the reranker experimental or operator-triggered. It improves R@20, but R@1 and R@10 do not improve, and each active call costs about 28 s.

**Sources:**
- Accuracy: `experiments/vecna-reranker-fusion-v1-20260914/README.md` in the private `JimmyK300/official-dataset-control` (commit `7093153`). It is reproduced independently in the Vecna report below.
- Latency and bootstrap: [`JimmyK300/Vecna@37b3210`](https://github.com/JimmyK300/Vecna/blob/37b321048432ccba5182a66bd39b1ca73c547537/benchmark-results/astra-retrieval-rd-v1/REPORT.md) at `benchmark-results/astra-retrieval-rd-v1/REPORT.md` (section C).
