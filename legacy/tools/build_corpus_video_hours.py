#!/usr/bin/env python3
"""Measure source-video durations and summarize hours by corpus folder.

The source videos are treated as read-only.  The output directory contains
only derived, reproducible metadata and evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


MICROSECOND = Decimal("0.000001")


def q(value: Decimal) -> Decimal:
    return value.quantize(MICROSECOND, rounding=ROUND_HALF_UP)


def format_hhmmss(seconds: Decimal) -> str:
    whole = int(seconds.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def run_probe(ffprobe: str, path: Path) -> tuple[Decimal | None, str | None]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip().replace("\r", " ").replace("\n", " ")
        return None, error or f"ffprobe exited with {result.returncode}"
    raw = result.stdout.strip()
    try:
        value = Decimal(raw)
    except Exception:
        return None, f"invalid duration output: {raw!r}"
    if not value.is_finite() or value < 0:
        return None, f"invalid duration value: {raw!r}"
    return q(value), None


def tool_version(ffprobe: str) -> str:
    result = subprocess.run([ffprobe, "-version"], capture_output=True, text=True)
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0].strip() if first_line else "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=Path("videos"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("derived/metadata/corpus-video-hours-v1"),
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    if not video_root.is_dir():
        raise SystemExit(f"video root does not exist or is not a directory: {video_root}")

    videos = sorted(
        (path for path in video_root.rglob("*.mp4") if path.is_file()),
        key=lambda path: path.relative_to(video_root).as_posix().lower(),
    )
    if not videos:
        raise SystemExit(f"no MP4 files found below {video_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, object]] = []
    corpus_totals: dict[str, dict[str, object]] = {}
    failures = 0
    ffprobe_version = tool_version(args.ffprobe)

    for path in videos:
        relative = path.relative_to(video_root)
        parts = relative.parts
        corpus = parts[0] if len(parts) > 1 else "(root)"
        media_id = path.stem
        duration, error = run_probe(args.ffprobe, path)
        status = "ok" if duration is not None else "probe_error"
        if error:
            failures += 1
        record = {
            "media_id": media_id,
            "corpus": corpus,
            "relative_video_path": relative.as_posix(),
            "duration_seconds": f"{duration:.6f}" if duration is not None else "",
            "video_hours": f"{(duration / Decimal(3600)):.9f}" if duration is not None else "",
            "duration_hhmmss": format_hhmmss(duration) if duration is not None else "",
            "file_size_bytes": path.stat().st_size,
            "probe_status": status,
            "probe_error": error or "",
        }
        inventory_rows.append(record)

        total = corpus_totals.setdefault(corpus, {"video_count": 0, "duration_seconds": Decimal(0)})
        total["video_count"] = int(total["video_count"]) + 1
        if duration is not None:
            total["duration_seconds"] = Decimal(total["duration_seconds"]) + duration

    inventory_path = output_dir / "video_durations.csv"
    inventory_fields = [
        "media_id",
        "corpus",
        "relative_video_path",
        "duration_seconds",
        "video_hours",
        "duration_hhmmss",
        "file_size_bytes",
        "probe_status",
        "probe_error",
    ]
    write_csv(inventory_path, inventory_fields, inventory_rows)

    summary_rows: list[dict[str, object]] = []
    for corpus in sorted(corpus_totals):
        data = corpus_totals[corpus]
        seconds = q(Decimal(data["duration_seconds"]))
        summary_rows.append(
            {
                "row_type": "corpus_folder",
                "corpus": corpus,
                "video_count": data["video_count"],
                "duration_seconds": f"{seconds:.6f}",
                "video_hours": f"{(seconds / Decimal(3600)):.9f}",
                "duration_hhmmss": format_hhmmss(seconds),
                "notes": "Source folder total",
            }
        )

    l26_rows = [row for row in summary_rows if row["corpus"] in {"L26A", "L26B", "L26C", "L26D", "L26E"}]
    l26_seconds = q(sum((Decimal(row["duration_seconds"]) for row in l26_rows), Decimal(0)))
    l26_count = sum(int(row["video_count"]) for row in l26_rows)
    summary_rows.append(
        {
            "row_type": "rollup",
            "corpus": "L26",
            "video_count": l26_count,
            "duration_seconds": f"{l26_seconds:.6f}",
            "video_hours": f"{(l26_seconds / Decimal(3600)):.9f}",
            "duration_hhmmss": format_hhmmss(l26_seconds),
            "notes": "Rollup of L26A-L26E; excluded from grand total to avoid double counting",
        }
    )

    total_seconds = q(sum((Decimal(row["duration_seconds"]) for row in summary_rows if row["row_type"] == "corpus_folder"), Decimal(0)))
    total_count = sum(int(row["video_count"]) for row in summary_rows if row["row_type"] == "corpus_folder")
    summary_rows.append(
        {
            "row_type": "grand_total",
            "corpus": "TOTAL",
            "video_count": total_count,
            "duration_seconds": f"{total_seconds:.6f}",
            "video_hours": f"{(total_seconds / Decimal(3600)):.9f}",
            "duration_hhmmss": format_hhmmss(total_seconds),
            "notes": "Sum of corpus_folder rows only",
        }
    )

    summary_path = output_dir / "corpus_video_hours.csv"
    summary_fields = [
        "row_type",
        "corpus",
        "video_count",
        "duration_seconds",
        "video_hours",
        "duration_hhmmss",
        "notes",
    ]
    write_csv(summary_path, summary_fields, summary_rows)

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    corpus_rows = [row for row in summary_rows if row["row_type"] == "corpus_folder"]
    table_lines = [
        "| Corpus folder | Videos | Seconds | Video hours | HH:MM:SS |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in corpus_rows:
        table_lines.append(
            f"| `{row['corpus']}` | {row['video_count']} | {row['duration_seconds']} | {row['video_hours']} | {row['duration_hhmmss']} |"
        )
    table_lines.append(
        f"| **TOTAL** | **{total_count}** | **{total_seconds:.6f}** | **{(total_seconds / Decimal(3600)):.9f}** | **{format_hhmmss(total_seconds)}** |"
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Corpus video-hour evidence v1\n\n"
        "This package records the measured duration of every source MP4 in the local `videos/` tree and sums those durations by source folder. The source videos were read only.\n\n"
        "## Result\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        "`L26` is represented by five source folders (`L26A`-`L26E`). The summary CSV also contains an `L26` rollup, but that rollup is excluded from `TOTAL` to prevent double counting.\n\n"
        "## Evidence files\n\n"
        "- `video_durations.csv`: one row per MP4, including the measured duration, source-folder corpus, relative path, file size, and probe status.\n"
        "- `corpus_video_hours.csv`: per-folder totals, the `L26A`-`L26E` rollup, and the grand total.\n"
        "- `manifest.json`: source scope, probe details, counts, totals, and SHA-256 hashes of the CSV and README evidence.\n"
        "- `../../../tools/build_corpus_video_hours.py`: reproducible generator.\n\n"
        "## Method\n\n"
        f"- Observed at UTC: `{created_at}`.\n"
        f"- Source root: `{video_root}` (portable relative root: `videos/`).\n"
        f"- Each `*.mp4` was enumerated recursively and probed with `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1`.\n"
        "- Hours are duration seconds divided by 3,600. Stored seconds are rounded to six decimal places and stored hours to nine decimal places; totals are summed from the unrounded probe values represented at six-decimal precision.\n"
        f"- Probe result: `{len(videos) - failures}/{len(videos)}` successful; failures: `{failures}`.\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact": "corpus-video-hours-v1",
        "created_at_utc": created_at,
        "source_video_root": str(video_root),
        "source_video_root_relative": "videos",
        "probe_command": [
            args.ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            "<video-path>",
        ],
        "ffprobe_version": ffprobe_version,
        "video_count": len(videos),
        "probe_success_count": len(videos) - failures,
        "probe_failure_count": failures,
        "total_duration_seconds": f"{total_seconds:.6f}",
        "total_video_hours": f"{(total_seconds / Decimal(3600)):.9f}",
        "total_duration_hhmmss": format_hhmmss(total_seconds),
        "evidence_files": {
            "inventory": {"path": inventory_path.name, "sha256": sha256(inventory_path)},
            "summary": {"path": summary_path.name, "sha256": sha256(summary_path)},
            "readme": {"path": readme_path.name, "sha256": sha256(readme_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "video_count": len(videos),
        "probe_failures": failures,
        "total_duration_seconds": f"{total_seconds:.6f}",
        "total_video_hours": f"{(total_seconds / Decimal(3600)):.9f}",
        "summary_path": str(summary_path),
        "inventory_path": str(inventory_path),
        "manifest_path": str(manifest_path),
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
