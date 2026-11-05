"""Stack per-frame Official-Dataset features into [N, D] + [N] arrays."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = (
    REPO_ROOT / "features_L21-L30_branch-feats-siglip" / "features"
)
DEFAULT_VIDEOS = REPO_ROOT / "videos"
DEFAULT_FEATURE_NAME = "image_siglip_so400m-384.npy"


def probe_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key)
        if not raw or raw == "0/0":
            continue
        num, den = raw.split("/")
        value = float(num) / float(den)
        if value > 0:
            return value
    raise RuntimeError(f"could not read fps from {video_path}")


def find_video(video_id: str) -> Path:
    matches = sorted(DEFAULT_VIDEOS.glob(f"**/{video_id}.mp4"))
    if not matches:
        raise FileNotFoundError(f"no video file for {video_id}")
    return matches[0]


def stack_video(
    video_id: str,
    feature_name: str = DEFAULT_FEATURE_NAME,
    fps: float | None = None,
) -> dict:
    feature_dir = DEFAULT_FEATURES / video_id
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"no feature directory for {video_id}")

    video_path = find_video(video_id)
    resolved_fps = float(fps) if fps is not None else probe_fps(video_path)

    frame_dirs = sorted(
        path
        for path in feature_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not frame_dirs:
        raise RuntimeError(f"no numbered frame folders in {feature_dir}")

    embeddings = []
    timestamps = []
    frame_ids = []
    missing = []

    expected_dim = None

    for frame_dir in frame_dirs:
        feature_path = frame_dir / feature_name
        if not feature_path.is_file():
            missing.append(frame_dir.name)
            continue

        vector = np.load(feature_path)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)

        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        elif vector.shape[0] != expected_dim:
            raise ValueError(
                f"{feature_path} has dim {vector.shape[0]}, "
                f"expected {expected_dim}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(f"non-finite values in {feature_path}")

        frame_id = int(frame_dir.name)
        embeddings.append(vector)
        timestamps.append(frame_id / resolved_fps)
        frame_ids.append(frame_id)

    if not embeddings:
        raise RuntimeError(
            f"no {feature_name} files found under {feature_dir}"
        )

    return {
        "video_id": video_id,
        "feature_name": feature_name,
        "video_path": str(video_path),
        "fps": resolved_fps,
        "embeddings": np.stack(embeddings, axis=0),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "missing_frames": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="e.g. L21_V008")
    parser.add_argument(
        "--feature",
        default=DEFAULT_FEATURE_NAME,
        help="per-frame npy filename",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    packed = stack_video(args.video, args.feature, args.fps)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    feature_stem = Path(packed["feature_name"]).stem.replace(".", "-")
    stem = f"{packed['video_id']}_{feature_stem}"
    np.save(output / f"{stem}_embeddings.npy", packed["embeddings"])
    np.save(output / f"{packed['video_id']}_timestamps.npy", packed["timestamps"])
    np.save(output / f"{packed['video_id']}_frame_ids.npy", packed["frame_ids"])

    manifest = {
        "video_id": packed["video_id"],
        "feature_name": packed["feature_name"],
        "video_path": packed["video_path"],
        "fps": packed["fps"],
        "n_samples": int(packed["embeddings"].shape[0]),
        "dim": int(packed["embeddings"].shape[1]),
        "start_sec": float(packed["timestamps"][0]),
        "end_sec": float(packed["timestamps"][-1]),
        "median_gap_sec": float(np.median(np.diff(packed["timestamps"])))
        if len(packed["timestamps"]) > 1
        else None,
        "missing_frames": packed["missing_frames"],
        "embeddings": f"{stem}_embeddings.npy",
        "timestamps": f"{stem}_timestamps.npy",
        "frame_ids": f"{stem}_frame_ids.npy",
        "note": (
            "Existing Official-Dataset features are I-frame + max-gap "
            "samples, not a new 2-FPS extract. Temporal weights still "
            "downweight clustered extra I-frames."
        ),
    }

    with open(output / f"{stem}_stack_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Stacked {manifest['n_samples']:,} x {manifest['dim']} "
        f"from {args.feature}"
    )
    print(f"fps={manifest['fps']:.4f}")
    print(
        f"span {manifest['start_sec']:.1f}s → {manifest['end_sec']:.1f}s "
        f"(median gap {manifest['median_gap_sec']:.3f}s)"
    )
    print(f"wrote {output / (stem + '_embeddings.npy')}")
    print(f"wrote {output / (stem + '_timestamps.npy')}")
    if packed["missing_frames"]:
        print(f"missing feature files: {len(packed['missing_frames'])}")


if __name__ == "__main__":
    main()
