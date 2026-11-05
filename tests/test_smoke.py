"""End-to-end smoke test on the synthetic benchmark (stdlib unittest; pytest also works)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vecna_eval import fusion  # noqa: E402
from vecna_eval.cli import evaluate  # noqa: E402
from vecna_eval.io import load_rankings, load_truth  # noqa: E402

DATA = ROOT / "data" / "synthetic"


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rankings = load_rankings(DATA / "provider_rankings.jsonl")
        cls.truth = load_truth(DATA / "ground_truth.jsonl")
        cls.result = evaluate(cls.rankings, cls.truth, {}, resamples=500)

    def test_every_arm_scored_on_same_cohort(self):
        summary = self.result["summary"]
        self.assertEqual(set(fusion.ARMS) | {"qwen_only"}, set(summary))
        for arm, s in summary.items():
            self.assertEqual(s["n"], 39, arm)  # 40 queries, 1 excluded
            self.assertLessEqual(s["recall_at_1"], s["recall_at_5"])
            self.assertLessEqual(s["recall_at_5"], s["recall_at_10"])
            self.assertLessEqual(s["recall_at_10"], s["recall_at_20"])
            self.assertTrue(0.0 <= s["mrr_at_20"] <= 1.0)

    def test_oracle_is_an_upper_bound(self):
        o = self.result["oracle_over_fusion_arms"]
        for arm in fusion.ARMS:
            for k in (1, 5, 10, 20):
                self.assertGreaterEqual(o[f"recall_at_{k}"], self.result["summary"][arm][f"recall_at_{k}"])

    def test_fusion_beats_chance_on_synthetic_data(self):
        self.assertGreater(self.result["summary"]["current"]["recall_at_20"], 20)

    def test_deterministic(self):
        again = evaluate(self.rankings, self.truth, {}, resamples=500)
        self.assertEqual(json.dumps(self.result, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "vecna_eval", "--rankings", str(DATA / "provider_rankings.jsonl"),
                 "--truth", str(DATA / "ground_truth.jsonl"), "--out", tmp, "--resamples", "200"],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("| current |", proc.stdout)
            result = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["excluded_queries"], ["q040"])


if __name__ == "__main__":
    unittest.main()
