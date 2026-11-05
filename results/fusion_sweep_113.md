# Fusion sweep (113 queries)

**Question:** given fixed provider lists, does a standard normalization (RRF, min–max, robust-z, softmax) beat the system's existing nested-max fusion?

**Setup:** all 115 queries were captured once, with 100 frame hits each from Qwen3-VL (visual), SigLIP (visual), OCR sparse/BM25 and ASR sparse/BM25. Dense text providers were disabled. Every arm uses identical hits, the same 100-frame output budget and the same exported tie order. The winner was chosen by a frozen ordering (video R@20, then MRR@20, then R@1). There was no weight or temperature search. `vecna_eval/fusion.py` re-implements these five rules.

| Arm | R@1 | R@5 | R@10 | R@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| **Current (nested max, 0.5/0.25/0.25)** | 49/113 | 77/113 | 82/113 | **87/113** | **0.529989** |
| RRF, k = 60 | 36/113 | 68/113 | 76/113 | 85/113 | 0.433591 |
| Min–max sum | 36/113 | 67/113 | 79/113 | 84/113 | 0.429601 |
| Robust-z / sigmoid sum | 36/113 | 61/113 | 69/113 | 81/113 | 0.411475 |
| Softmax sum, T = 1 | 16/113 | 62/113 | 80/113 | 86/113 | 0.304353 |
| Matched Qwen only | 47/113 | 71/113 | 79/113 | 84/113 | 0.507495 |

Paired against the current fusion (10,000-resample bootstrap):

| Alternative | R@20 rescues / regressions | R@20 Δ (pp) [95% CI] | MRR Δ [95% CI] |
|---|---:|---|---|
| RRF | 4 / 6 | −1.77 [−7.08, +3.54] | −0.0964 [−0.1400, −0.0546] |
| Min–max | 5 / 8 | −2.65 [−8.85, +3.54] | −0.1004 [−0.1501, −0.0510] |
| Robust-z | 3 / 9 | −5.31 [−11.50, +0.88] | −0.1185 [−0.1718, −0.0692] |
| Softmax | 2 / 3 | −0.88 [−4.42, +2.65] | −0.2256 [−0.2893, −0.1618] |

Against matched Qwen only, the current fusion rescues 3 queries at R@20 (`p0_q02`, `p1_q04`, `p2_q21`) and loses none. The interval is +2.65 pp [0.00, +6.19].

**Reading:** every alternative loses MRR with an interval that excludes zero. None wins R@20. Fusion's gain over Qwen alone is small: 3 rescues on 113 queries.

**Source:** [`JimmyK300/Vecna@37b3210`](https://github.com/JimmyK300/Vecna/blob/37b321048432ccba5182a66bd39b1ca73c547537/benchmark-results/astra-retrieval-rd-v1/REPORT.md) at `benchmark-results/astra-retrieval-rd-v1/REPORT.md` (section B), plus `outputs/fusion/FINDINGS.md`. Branch: `btl/issue-82-failure-ledger-and-proposal`.
