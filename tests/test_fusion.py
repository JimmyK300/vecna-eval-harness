"""Unit tests for the fusion rules and scoring contract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vecna_eval import fusion, scoring  # noqa: E402
from vecna_eval.io import ContractError  # noqa: E402

CFG = fusion.validate_config({})


class TransformTests(unittest.TestCase):
    def test_rrf_uses_rank_not_score(self):
        out = fusion.transform([("a#1", 100.0), ("b#1", 0.001)], "rrf")
        self.assertAlmostEqual(out["a#1"], 1 / 61)
        self.assertAlmostEqual(out["b#1"], 1 / 62)

    def test_minmax_equal_scores_map_to_one(self):
        self.assertEqual(fusion.transform([("a#1", 2.0), ("b#1", 2.0)], "minmax"), {"a#1": 1.0, "b#1": 1.0})

    def test_softmax_sums_to_one(self):
        out = fusion.transform([("a#1", 3.0), ("b#1", 1.0), ("c#1", 0.0)], "softmax")
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_robust_z_is_bounded(self):
        out = fusion.transform([(f"v{i}#1", float(i)) for i in range(10)] + [("x#1", 1e9)], "robust_z")
        self.assertTrue(all(0.0 < v < 1.0 for v in out.values()))


class CurrentControlTests(unittest.TestCase):
    def test_visual_scores_are_summed_before_scaling(self):
        providers = {"qwen": [("a#1", 0.4), ("b#1", 0.3)], "siglip": [("b#1", 0.3)]}
        order = fusion.candidate_order(providers, ["qwen", "siglip"])
        scores = fusion.current_scores(providers, order, CFG)
        # b: 0.3 + 0.3 = 0.6 is the visual max -> 0.5 * 1.0; a: 0.5 * 0.4 / 0.6
        self.assertAlmostEqual(scores["b#1"], 0.5)
        self.assertAlmostEqual(scores["a#1"], 0.5 * 0.4 / 0.6)

    def test_text_only_hit_gets_text_weight(self):
        providers = {"qwen": [("a#1", 1.0)], "siglip": [], "ocr_sparse": [("b#1", 12.0)]}
        order = fusion.candidate_order(providers, ["qwen", "siglip", "ocr_sparse"])
        scores = fusion.current_scores(providers, order, CFG)
        self.assertAlmostEqual(scores["a#1"], 0.5)
        self.assertAlmostEqual(scores["b#1"], 0.25)


class CollapseAndScoringTests(unittest.TestCase):
    def test_ties_keep_first_appearance_and_videos_collapse(self):
        providers = {"qwen": [("v1#10", 1.0), ("v1#20", 1.0), ("v2#5", 1.0)], "siglip": []}
        self.assertEqual(fusion.fuse(providers, "current", CFG), ["v1", "v2"])

    def test_frame_budget_truncates_before_collapse(self):
        cfg = fusion.validate_config({"output_frame_k": 2})
        providers = {"qwen": [("v1#1", 0.9), ("v1#2", 0.8), ("v2#1", 0.7)], "siglip": []}
        self.assertEqual(fusion.fuse(providers, "current", cfg), ["v1"])

    def test_recall_and_mrr(self):
        s = scoring.summarize({"q1": 1, "q2": 4, "q3": None, "q4": 25})
        self.assertEqual((s["recall_at_1"], s["recall_at_5"], s["recall_at_20"]), (1, 2, 2))
        self.assertAlmostEqual(s["mrr_at_20"], round((1 + 0.25) / 4, 6))

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ContractError):
            fusion.validate_config({"ocr_weight": 0.7, "asr_weight": 0.5})


if __name__ == "__main__":
    unittest.main()
