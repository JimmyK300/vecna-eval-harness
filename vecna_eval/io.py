"""Input loading and validation.

Provider rankings (one JSON object per line)::

    {"query_id": "q001",
     "providers": {"qwen":       [{"frame_id": "vid_007#000120", "score": 0.81}, ...],
                   "siglip":     [...],
                   "ocr_sparse": [...],
                   "asr_sparse": [...]}}

Ground truth (one JSON object per line)::

    {"query_id": "q001", "accepted_videos": ["vid_007"], "excluded": false}

A frame id is ``<video_id>#<frame_number>``. Lists are in provider rank order
(best first). Ground truth is read only by scoring and the oracle, never by
fusion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when an input breaks the documented format."""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{path}:{number}: invalid JSON ({exc})") from exc
    return rows


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def frame_video(frame_id: str) -> str:
    if not isinstance(frame_id, str) or frame_id.count("#") != 1:
        raise ContractError(f"frame_id must look like 'video#frame': {frame_id!r}")
    video, frame = frame_id.split("#")
    if not video or not frame.isdigit():
        raise ContractError(f"invalid frame_id: {frame_id!r}")
    return video


def load_rankings(path: str | Path) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Return ``{query_id: {provider: [(frame_id, score), ...]}}``."""
    out: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for row in read_jsonl(path):
        qid = row.get("query_id")
        if not isinstance(qid, str) or qid in out:
            raise ContractError(f"missing or duplicate query_id: {qid!r}")
        providers = row.get("providers")
        if not isinstance(providers, dict) or not providers:
            raise ContractError(f"{qid}: 'providers' must be a non-empty object")
        parsed: dict[str, list[tuple[str, float]]] = {}
        for name, hits in providers.items():
            seen = set()
            items = []
            for hit in hits:
                fid, score = hit.get("frame_id"), hit.get("score")
                frame_video(fid)
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ContractError(f"{qid}/{name}: non-finite score for {fid}")
                if fid in seen:
                    raise ContractError(f"{qid}/{name}: duplicate frame {fid}")
                seen.add(fid)
                items.append((fid, float(score)))
            parsed[name] = items
        out[qid] = parsed
    return out


def load_truth(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return ``{query_id: {"accepted": set[str], "excluded": bool}}``."""
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        qid = row.get("query_id")
        if not isinstance(qid, str) or qid in out:
            raise ContractError(f"missing or duplicate query_id in truth: {qid!r}")
        accepted = row.get("accepted_videos", [])
        if not isinstance(accepted, list):
            raise ContractError(f"{qid}: accepted_videos must be a list")
        out[qid] = {"accepted": set(accepted), "excluded": bool(row.get("excluded", False))}
    return out
