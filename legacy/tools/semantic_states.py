import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def temporal_weights(timestamps, max_weight=0.75):
    """
    Approximate how much real video time each sampled frame represents.

    This prevents extra I-frames from getting excessive influence just
    because several frames happen to exist close together.
    """
    n = len(timestamps)

    if n == 1:
        return np.ones(1, dtype=np.float32)

    w = np.empty(n, dtype=np.float32)

    w[0] = timestamps[1] - timestamps[0]
    w[-1] = timestamps[-1] - timestamps[-2]

    if n > 2:
        w[1:-1] = (timestamps[2:] - timestamps[:-2]) / 2.0

    return np.clip(w, 1e-4, max_weight)


def weighted_window_means(embeddings, timestamps, window_sec):
    """
    For every t:

        LEFT  = mean embeddings from [t-window, t)
        RIGHT = mean embeddings from (t, t+window]

    Uses temporal weights and prefix sums.
    """

    weights = temporal_weights(timestamps)

    weighted = embeddings * weights[:, None]

    prefix_e = np.vstack([
        np.zeros((1, embeddings.shape[1]), dtype=np.float32),
        np.cumsum(weighted, axis=0, dtype=np.float32),
    ])

    prefix_w = np.concatenate([
        np.zeros(1, dtype=np.float32),
        np.cumsum(weights, dtype=np.float32),
    ])

    left_start = np.searchsorted(
        timestamps,
        timestamps - window_sec,
        side="left",
    )

    left_end = np.arange(len(timestamps))

    right_start = np.arange(len(timestamps)) + 1

    right_end = np.searchsorted(
        timestamps,
        timestamps + window_sec,
        side="right",
    )

    left_sum = prefix_e[left_end] - prefix_e[left_start]
    right_sum = prefix_e[right_end] - prefix_e[right_start]

    left_weight = prefix_w[left_end] - prefix_w[left_start]
    right_weight = prefix_w[right_end] - prefix_w[right_start]

    left_mean = left_sum / np.maximum(left_weight[:, None], 1e-8)
    right_mean = right_sum / np.maximum(right_weight[:, None], 1e-8)

    valid = (left_weight > 0.5) & (right_weight > 0.5)

    return left_mean, right_mean, valid


def cosine_change(left, right, valid):
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)

    denom = np.maximum(left_norm * right_norm, 1e-8)

    similarity = np.sum(left * right, axis=1) / denom
    score = 1.0 - similarity

    score[~valid] = 0.0

    return score


def smooth_time_series(values, timestamps, radius_sec=0.75):
    """
    Timestamp-aware moving average.
    """

    prefix = np.concatenate([[0.0], np.cumsum(values)])

    starts = np.searchsorted(
        timestamps,
        timestamps - radius_sec,
        side="left",
    )

    ends = np.searchsorted(
        timestamps,
        timestamps + radius_sec,
        side="right",
    )

    sums = prefix[ends] - prefix[starts]
    counts = np.maximum(ends - starts, 1)

    return sums / counts


def adaptive_threshold(scores):
    """
    Robust threshold. No model-specific cosine threshold needs to be
    guessed for the first experiment.
    """
    positive = scores[scores > 0]

    if len(positive) == 0:
        return np.inf

    median = np.median(positive)
    mad = np.median(np.abs(positive - median))

    robust = median + 2.5 * 1.4826 * mad
    percentile = np.quantile(positive, 0.90)

    return max(robust, percentile)


def find_boundaries(
    scores,
    timestamps,
    threshold,
    min_state_sec=3.0,
):
    """
    Find local maxima above threshold.

    Higher peaks win first, then minimum state duration prevents
    lots of tiny adjacent states.
    """

    candidates = []

    for i in range(1, len(scores) - 1):
        if (
            scores[i] >= threshold
            and scores[i] >= scores[i - 1]
            and scores[i] >= scores[i + 1]
        ):
            candidates.append(i)

    # Strongest peaks first
    candidates.sort(key=lambda i: scores[i], reverse=True)

    selected = []

    for idx in candidates:
        t = timestamps[idx]

        if all(
            abs(t - timestamps[other]) >= min_state_sec
            for other in selected
        ):
            selected.append(idx)

    selected.sort()

    return selected


def representative_frame(
    embeddings,
    timestamps,
    start_t,
    end_t,
):
    indices = np.where(
        (timestamps >= start_t)
        & (timestamps < end_t)
    )[0]

    if len(indices) == 0:
        return None

    segment = embeddings[indices]

    centroid = segment.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-8)

    similarities = segment @ centroid

    best_local = np.argmax(similarities)

    return int(indices[best_local])


def build_states(
    embeddings,
    timestamps,
    boundary_indices,
    scores,
):
    video_start = float(timestamps[0])
    video_end = float(timestamps[-1])

    boundary_times = [
        float(timestamps[i])
        for i in boundary_indices
    ]

    edges = [video_start] + boundary_times + [video_end]

    states = []

    for state_id in range(len(edges) - 1):
        start = edges[state_id]
        end = edges[state_id + 1]

        rep = representative_frame(
            embeddings,
            timestamps,
            start,
            end,
        )

        state = {
            "state": state_id,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "representative_index": rep,
            "representative_timestamp": (
                float(timestamps[rep])
                if rep is not None
                else None
            ),
        }

        if state_id > 0:
            boundary_idx = boundary_indices[state_id - 1]
            state["incoming_boundary_score"] = float(
                scores[boundary_idx]
            )

        states.append(state)

    return states


def plot_scores(
    timestamps,
    raw_scores,
    smooth_scores,
    boundaries,
    threshold,
    output,
):
    plt.figure(figsize=(15, 5))

    plt.plot(
        timestamps,
        raw_scores,
        alpha=0.35,
        label="raw",
    )

    plt.plot(
        timestamps,
        smooth_scores,
        label="smoothed",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label=f"threshold {threshold:.3f}",
    )

    for idx in boundaries:
        plt.axvline(
            timestamps[idx],
            linestyle="--",
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Semantic state change")
    plt.title("Semantic state boundaries")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output, dpi=160)
    plt.close()


def format_clock(seconds):
    if seconds is None:
        return "n/a"
    total = max(float(seconds), 0.0)
    minutes = int(total // 60)
    rem = total - 60 * minutes
    return f"{minutes:02d}:{rem:04.1f}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        required=True,
    )

    parser.add_argument(
        "--timestamps",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="state_output",
    )

    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="Seconds on each side of candidate boundary",
    )

    parser.add_argument(
        "--smooth",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--min-state",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings).astype(np.float32)
    timestamps = np.load(args.timestamps).astype(np.float64)

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be [N, D], got {embeddings.shape}"
        )

    if timestamps.ndim != 1:
        raise ValueError(
            f"timestamps must be [N], got {timestamps.shape}"
        )

    if len(embeddings) != len(timestamps):
        raise ValueError(
            "embeddings and timestamps have different lengths"
        )

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    embeddings = embeddings[order]

    embeddings = normalize(embeddings)

    print(
        f"Loaded {len(embeddings):,} samples, "
        f"{embeddings.shape[1]} dimensions"
    )

    print(
        f"Video duration: "
        f"{timestamps[-1] - timestamps[0]:.1f}s"
    )

    left, right, valid = weighted_window_means(
        embeddings,
        timestamps,
        args.window,
    )

    raw_scores = cosine_change(
        left,
        right,
        valid,
    )

    smooth_scores = smooth_time_series(
        raw_scores,
        timestamps,
        args.smooth,
    )

    threshold = (
        args.threshold
        if args.threshold is not None
        else adaptive_threshold(smooth_scores)
    )

    boundaries = find_boundaries(
        smooth_scores,
        timestamps,
        threshold,
        args.min_state,
    )

    states = build_states(
        embeddings,
        timestamps,
        boundaries,
        smooth_scores,
    )

    np.savetxt(
        output / "change_scores.csv",
        np.column_stack([
            timestamps,
            raw_scores,
            smooth_scores,
        ]),
        delimiter=",",
        header="timestamp,raw_change,smoothed_change",
        comments="",
    )

    with open(
        output / "states.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            states,
            f,
            indent=2,
        )

    plot_scores(
        timestamps,
        raw_scores,
        smooth_scores,
        boundaries,
        threshold,
        output / "change_scores.png",
    )

    print()
    print(f"Threshold: {threshold:.4f}")
    print(f"Found {len(states)} states")
    print()

    for state in states:
        if state["state"] > 0:
            print(
                f"              boundary score = "
                f"{state['incoming_boundary_score']:.3f}"
            )
            print("                       ↓")

        print(f"STATE {state['state']}")
        print(
            f"{format_clock(state['start_sec'])} "
            f"→ {format_clock(state['end_sec'])}"
        )
        print(
            "representative frame: "
            f"{format_clock(state['representative_timestamp'])}"
        )
        print()

        rep = state["representative_timestamp"]
        rep_text = f"{rep:7.1f}s" if rep is not None else "    n/a"
        print(
            f"STATE {state['state']:02d}: "
            f"{state['start_sec']:7.1f}s "
            f"→ {state['end_sec']:7.1f}s "
            f"({state['duration_sec']:6.1f}s) | "
            f"representative {rep_text}"
        )
        print()


if __name__ == "__main__":
    main()
