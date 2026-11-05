"""Score-fusion arms over fixed provider frame lists.

Every arm sees identical inputs: the same provider hits, the same candidate
set and the same tie order. Only the normalization/combination rule changes,
so differences between arms are caused by the rule, not by retrieval.

Arms
----
``current``   Nested max scaling used by the VECNA searcher: visual provider
              scores are summed then divided by their joint max; each text
              modality is max-scaled, blended sparse/dense by ``alpha`` and
              max-scaled again; the three modalities are combined with outer
              weights (default visual 0.5, OCR 0.25, ASR 0.25).
``rrf``       Reciprocal Rank Fusion, 1 / (k + rank), k = 60.
``minmax``    Per-provider min-max to [0, 1] (all-equal lists map to 1).
``robust_z``  Per-provider median/MAD z-score (MAD x 1.4826), clipped to
              +-6, passed through a sigmoid.
``softmax``   Per-provider softmax with temperature 1.

The four alternative arms weight each provider by its *nominal* share of the
outer weights, so effective coefficients differ from ``current``. This is a
frozen-configuration comparison, not a weight calibration.

The frame list is truncated to ``output_frame_k`` and then collapsed to the
first occurrence of each video (distinct-video ranking).
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable

from .io import ContractError, frame_video

ARMS = ("current", "rrf", "minmax", "robust_z", "softmax")
RRF_K = 60
MAD_FACTOR = 1.4826
ROBUST_CLIP = 6.0
SOFTMAX_TEMPERATURE = 1.0

DEFAULT_CONFIG = {
    "visual_providers": ["qwen", "siglip"],
    "ocr_weight": 0.25,
    "asr_weight": 0.25,
    "ocr_alpha": 0.0,  # 0 = sparse only, 1 = dense only
    "asr_alpha": 0.0,
    "output_frame_k": 100,
    "output_video_k": 100,
}
TEXT_PROVIDERS = ("ocr_sparse", "ocr_dense", "asr_sparse", "asr_dense")


def validate_config(config: dict) -> dict:
    merged = {**DEFAULT_CONFIG, **config}
    for key in ("ocr_weight", "asr_weight", "ocr_alpha", "asr_alpha"):
        value = merged[key]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ContractError(f"{key} must be within [0, 1]")
    if merged["ocr_weight"] + merged["asr_weight"] > 1:
        raise ContractError("ocr_weight + asr_weight must not exceed 1")
    for key in ("output_frame_k", "output_video_k"):
        if not isinstance(merged[key], int) or merged[key] <= 0:
            raise ContractError(f"{key} must be a positive integer")
    if not merged["visual_providers"]:
        raise ContractError("at least one visual provider is required")
    return merged


def nominal_weights(config: dict) -> dict[str, float]:
    visual = 1.0 - config["ocr_weight"] - config["asr_weight"]
    weights = {name: visual / len(config["visual_providers"]) for name in config["visual_providers"]}
    weights.update({
        "ocr_sparse": config["ocr_weight"] * (1.0 - config["ocr_alpha"]),
        "ocr_dense": config["ocr_weight"] * config["ocr_alpha"],
        "asr_sparse": config["asr_weight"] * (1.0 - config["asr_alpha"]),
        "asr_dense": config["asr_weight"] * config["asr_alpha"],
    })
    return weights


def candidate_order(providers: dict[str, list[tuple[str, float]]], names: Iterable[str]) -> list[str]:
    """Deterministic tie order: first appearance, scanning providers in ``names`` order."""
    order, seen = [], set()
    for name in names:
        for fid, _ in providers.get(name, []):
            if fid not in seen:
                seen.add(fid)
                order.append(fid)
    return order


def maxscale(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {k: v / maximum if maximum > 0 else 0.0 for k, v in values.items()}


def transform(hits: list[tuple[str, float]], arm: str) -> dict[str, float]:
    """Normalize one provider's ranked list for an alternative arm."""
    if not hits:
        return {}
    keys = [fid for fid, _ in hits]
    values = [score for _, score in hits]
    if arm == "rrf":
        out = [1.0 / (RRF_K + rank) for rank in range(1, len(values) + 1)]
    elif arm == "minmax":
        low, spread = min(values), max(values) - min(values)
        out = [(v - low) / spread for v in values] if spread else [1.0] * len(values)
    elif arm == "robust_z":
        center = statistics.median(values)
        scale = MAD_FACTOR * statistics.median(abs(v - center) for v in values)
        if scale == 0.0:
            scale = max(abs(v - center) for v in values)
        z = [(v - center) / scale if scale else 0.0 for v in values]
        out = [1.0 / (1.0 + math.exp(-max(-ROBUST_CLIP, min(ROBUST_CLIP, x)))) for x in z]
    elif arm == "softmax":
        top = max(values)
        mass = [math.exp((v - top) / SOFTMAX_TEMPERATURE) for v in values]
        total = sum(mass)
        out = [m / total for m in mass]
    else:
        raise ContractError(f"unknown arm {arm!r}")
    return dict(zip(keys, out))


def current_scores(providers: dict, order: list[str], config: dict) -> dict[str, float]:
    maps = {name: dict(hits) for name, hits in providers.items()}
    visual: dict[str, float] = {}
    for name in config["visual_providers"]:
        for fid, score in maps.get(name, {}).items():
            visual[fid] = visual.get(fid, 0.0) + score
    visual_norm = maxscale(visual)
    text_norm = {}
    for modality in ("ocr", "asr"):
        sparse = maxscale(maps.get(modality + "_sparse", {}))
        dense = maxscale(maps.get(modality + "_dense", {}))
        alpha = config[modality + "_alpha"]
        merged = {}
        for fid in order:
            if fid in sparse or fid in dense:
                merged[fid] = alpha * dense.get(fid, 0.0) + (1.0 - alpha) * sparse.get(fid, 0.0)
        text_norm[modality] = maxscale(merged)
    visual_weight = 1.0 - config["ocr_weight"] - config["asr_weight"]
    return {
        fid: visual_weight * visual_norm.get(fid, 0.0)
        + config["ocr_weight"] * text_norm["ocr"].get(fid, 0.0)
        + config["asr_weight"] * text_norm["asr"].get(fid, 0.0)
        for fid in order
    }


def fuse(providers: dict[str, list[tuple[str, float]]], arm: str, config: dict) -> list[str]:
    """Return the distinct-video ranking produced by ``arm`` for one query."""
    names = list(config["visual_providers"]) + [p for p in TEXT_PROVIDERS if p in providers]
    order = candidate_order(providers, names)
    if arm == "current":
        scores = current_scores(providers, order, config)
    else:
        weights = nominal_weights(config)
        scores = dict.fromkeys(order, 0.0)
        for name in names:
            weight = weights.get(name, 0.0)
            if weight:
                for fid, value in transform(providers.get(name, []), arm).items():
                    scores[fid] += weight * value
    ranked = sorted(order, key=lambda fid: scores[fid], reverse=True)  # stable => tie order kept
    videos, seen = [], set()
    for fid in ranked[: config["output_frame_k"]]:
        video = frame_video(fid)
        if video not in seen:
            seen.add(video)
            videos.append(video)
            if len(videos) >= config["output_video_k"]:
                break
    return videos


def single_provider(providers: dict[str, list[tuple[str, float]]], name: str, k: int) -> list[str]:
    """Distinct-video ranking from one provider's list, for matched baselines."""
    videos, seen = [], set()
    for fid, _ in providers.get(name, [])[:k]:
        video = frame_video(fid)
        if video not in seen:
            seen.add(video)
            videos.append(video)
    return videos
