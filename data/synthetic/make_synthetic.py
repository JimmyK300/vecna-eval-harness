"""Generate the synthetic benchmark used by the smoke test.

Everything here is invented: 40 queries (q001..q040) over 60 fake videos
(vid_000..vid_059). No contest video, frame, query text or label is used.
Each provider sees the gold video with its own reliability so that the
fusion arms disagree in realistic ways (some queries are visual-only, some
text-only, a few are missed by every provider).

    python data/synthetic/make_synthetic.py   # rewrites the two .jsonl files
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 2026
N_QUERIES = 40
N_VIDEOS = 60
DEPTH = 50
# probability that a provider surfaces the gold video, per query "kind"
RELIABILITY = {
    "visual": {"qwen": 0.85, "siglip": 0.7, "ocr_sparse": 0.15, "asr_sparse": 0.15},
    "text":   {"qwen": 0.3, "siglip": 0.2, "ocr_sparse": 0.8, "asr_sparse": 0.6},
    "mixed":  {"qwen": 0.6, "siglip": 0.5, "ocr_sparse": 0.5, "asr_sparse": 0.4},
    "hard":   {"qwen": 0.05, "siglip": 0.05, "ocr_sparse": 0.05, "asr_sparse": 0.05},
}
SCORE_RANGE = {"qwen": (0.15, 0.45), "siglip": (0.05, 0.35), "ocr_sparse": (1.0, 18.0), "asr_sparse": (1.0, 14.0)}


def frame(rng: random.Random, video: int) -> str:
    return f"vid_{video:03d}#{rng.randrange(0, 30000):06d}"


def main() -> None:
    rng = random.Random(SEED)
    here = Path(__file__).resolve().parent
    kinds = ["visual"] * 14 + ["text"] * 12 + ["mixed"] * 10 + ["hard"] * 4
    rng.shuffle(kinds)
    rankings, truth = [], []
    for i, kind in enumerate(kinds, 1):
        qid = f"q{i:03d}"
        gold = rng.randrange(N_VIDEOS)
        providers = {}
        for name, (low, high) in SCORE_RANGE.items():
            hits = {}
            while len(hits) < DEPTH:
                video = rng.randrange(N_VIDEOS)
                if video != gold:
                    hits[frame(rng, video)] = rng.uniform(low, high)
            if rng.random() < RELIABILITY[kind][name]:
                for _ in range(rng.randint(1, 3)):  # several gold frames per list
                    hits[frame(rng, gold)] = rng.uniform(low + 0.6 * (high - low), high * 1.05)
            ordered = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:DEPTH]
            providers[name] = [{"frame_id": f, "score": round(s, 6)} for f, s in ordered]
        rankings.append({"query_id": qid, "kind": kind, "providers": providers})
        truth.append({"query_id": qid, "accepted_videos": [f"vid_{gold:03d}"], "excluded": False})
    truth[-1]["excluded"] = True  # exercise the exclusion rule
    for name, rows in (("provider_rankings.jsonl", rankings), ("ground_truth.jsonl", truth)):
        with open(here / name, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
