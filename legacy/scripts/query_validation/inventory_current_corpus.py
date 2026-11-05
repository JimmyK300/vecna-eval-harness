"""Create a read-only inventory of the current local media corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MEDIA_RE = re.compile(r"^(?P<level>L(?:2[1-9]|30|26[A-E]))_V(?P<number>\d+)$")


def probe(path: Path) -> dict[str, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        stream = json.loads(result.stdout).get("streams", [{}])[0]
        return {
            "duration_seconds": stream.get("duration", ""),
            "frame_count": stream.get("nb_frames", ""),
            "probe_error": "",
        }
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
        return {"duration_seconds": "", "frame_count": "", "probe_error": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.media_root.rglob("*.mp4"))
    jobs = []
    for path in paths:
        match = MEDIA_RE.match(path.stem)
        if not match:
            continue
        jobs.append((match.group("level"), path.stem, path))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        probes = list(pool.map(lambda item: probe(item[2]), jobs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "media_id",
                "level_batch",
                "video_path",
                "duration_seconds",
                "frame_count",
                "availability",
                "probe_error",
            ],
        )
        writer.writeheader()
        for (level, media_id, path), probe_result in zip(jobs, probes):
            writer.writerow(
                {
                    "media_id": media_id,
                    "level_batch": level,
                    "video_path": str(path),
                    "availability": "available" if path.is_file() else "missing",
                    **probe_result,
                }
            )


if __name__ == "__main__":
    main()
