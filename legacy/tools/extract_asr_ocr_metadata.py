#!/usr/bin/env python3
"""Decode keyframe-owned ASR/OCR .npy strings into derived metadata.

The official feature tree stores WhisperX/Tesseract text as 0-d numpy strings
in ``asr.npy`` / ``ocr.npy``. Those files are not searchable as text. This
tool projects them into per-video JSON/TXT plus corpus JSONL using the same
decode and consecutive-identical grouping as Vecna
(``aic51.packages.analyse.features.text_embedding._load_frame_text`` and
``/api/video/transcript``).

Dense embedding files (``asr_dense.npy``, ``ocr_dense.npy``) are left as-is.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO_ROOT / "features_L21-L30_branch-feats-siglip" / "features"
DEFAULT_VIDEOS = REPO_ROOT / "videos"
DEFAULT_OUTPUT = REPO_ROOT / "derived" / "metadata" / "asr-ocr-text-v1"
DEFAULT_FPS = 25.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame_text(path: Path) -> str:
    """Decode a Vecna ASR/OCR .npy payload to a Python string."""
    if not path.exists():
        return ""
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray):
        if payload.size == 0:
            return ""
        value = payload.reshape(-1)[0]
    else:
        value = payload
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    text = str(value)
    if text in {"nan", "NaN"}:
        return ""
    return text


def group_consecutive(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse adjacent identical nonempty texts. Empty frames are skipped."""
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for entry in entries:
        text = entry["text"].strip()
        if not text:
            continue
        if current is None:
            current = {
                "start_frame": entry["frame_id"],
                "end_frame": entry["frame_id"],
                "start_frame_idx": entry["frame_idx"],
                "end_frame_idx": entry["frame_idx"],
                "start_seconds": entry["time_seconds"],
                "end_seconds": entry["time_seconds"],
                "frame_count": 1,
                "text": text,
            }
        elif current["text"] == text:
            current["end_frame"] = entry["frame_id"]
            current["end_frame_idx"] = entry["frame_idx"]
            current["end_seconds"] = entry["time_seconds"]
            current["frame_count"] += 1
        else:
            segments.append(current)
            current = {
                "start_frame": entry["frame_id"],
                "end_frame": entry["frame_id"],
                "start_frame_idx": entry["frame_idx"],
                "end_frame_idx": entry["frame_idx"],
                "start_seconds": entry["time_seconds"],
                "end_seconds": entry["time_seconds"],
                "frame_count": 1,
                "text": text,
            }
    if current is not None:
        segments.append(current)
    return segments


def format_clock(seconds: float) -> str:
    return f"{seconds:08.3f}"


def transcript_txt(segments: list[dict[str, Any]]) -> str:
    lines = []
    for segment in segments:
        start = format_clock(float(segment["start_seconds"]))
        end = format_clock(float(segment["end_seconds"]))
        lines.append(f"[{start} -> {end}] {segment['text']}")
    return "\n".join(lines) + ("\n" if lines else "")


def parse_r_frame_rate(raw: str) -> tuple[str, float, int]:
    value = raw.strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        fps = float(numerator) / float(denominator)
    else:
        fps = float(value)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid r_frame_rate: {raw!r}")
    return value, fps, int(round(fps))


def probe_fps(ffprobe: str, video_path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip().replace("\n", " ")
        raise RuntimeError(error or f"ffprobe exited with {result.returncode}")
    raw = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    r_frame_rate, fps, rounded = parse_r_frame_rate(raw)
    return {
        "r_frame_rate": r_frame_rate,
        "fps": fps,
        "rounded_fps": rounded,
        "probe_status": "ok",
        "probe_error": "",
        "video_path": str(video_path),
    }


def list_frame_dirs(video_dir: Path) -> list[Path]:
    frames = [path for path in video_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    frames.sort(key=lambda path: int(path.name))
    return frames


def extract_channel(
    frame_dirs: list[Path],
    filename: str,
    rounded_fps: int,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    present = 0
    missing = 0
    decode_errors = 0
    empty = 0
    nonempty_frames: list[dict[str, Any]] = []
    for frame_dir in frame_dirs:
        frame_id = frame_dir.name
        frame_idx = int(frame_id)
        time_seconds = frame_idx / rounded_fps if rounded_fps else 0.0
        path = frame_dir / filename
        if not path.exists():
            missing += 1
            entries.append(
                {
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "time_seconds": time_seconds,
                    "text": "",
                    "status": "missing",
                }
            )
            continue
        present += 1
        try:
            text = load_frame_text(path)
        except Exception as exc:  # noqa: BLE001 - keep the corpus job going
            decode_errors += 1
            entries.append(
                {
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "time_seconds": time_seconds,
                    "text": "",
                    "status": f"decode_error:{type(exc).__name__}",
                }
            )
            continue
        text = text.strip()
        record = {
            "frame_id": frame_id,
            "frame_idx": frame_idx,
            "time_seconds": round(time_seconds, 6),
            "text": text,
            "status": "ok",
        }
        entries.append(record)
        if text:
            nonempty_frames.append(
                {
                    "frame_id": frame_id,
                    "frame_idx": frame_idx,
                    "time_seconds": record["time_seconds"],
                    "text": text,
                }
            )
        else:
            empty += 1
    segments = group_consecutive(entries)
    return {
        "present": present,
        "missing": missing,
        "decode_errors": decode_errors,
        "empty": empty,
        "nonempty": len(nonempty_frames),
        "segment_count": len(segments),
        "char_count": sum(len(item["text"]) for item in segments),
        "frames": nonempty_frames,
        "segments": segments,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def process_video(
    video_id: str,
    video_dir: Path,
    video_path: Path | None,
    ffprobe: str,
    output_dir: Path,
    features_root_label: str,
) -> dict[str, Any]:
    frame_dirs = list_frame_dirs(video_dir)
    if video_path is not None:
        try:
            timing = probe_fps(ffprobe, video_path)
        except Exception as exc:  # noqa: BLE001
            timing = {
                "r_frame_rate": "",
                "fps": DEFAULT_FPS,
                "rounded_fps": int(DEFAULT_FPS),
                "probe_status": "fallback_default_fps",
                "probe_error": str(exc),
                "video_path": str(video_path),
            }
    else:
        timing = {
            "r_frame_rate": "",
            "fps": DEFAULT_FPS,
            "rounded_fps": int(DEFAULT_FPS),
            "probe_status": "missing_video_fallback_default_fps",
            "probe_error": "source mp4 not found",
            "video_path": "",
        }

    rounded_fps = int(timing["rounded_fps"])
    asr = extract_channel(frame_dirs, "asr.npy", rounded_fps)
    ocr = extract_channel(frame_dirs, "ocr.npy", rounded_fps)

    shared_timing = {
        "r_frame_rate": timing["r_frame_rate"],
        "fps": timing["fps"],
        "rounded_fps": rounded_fps,
        "timestamp_formula": "frame_idx / rounded_fps",
        "timestamp_note": (
            "Inclusive keyframe times. ASR text is the WhisperX segment "
            "projected onto this keyframe, not a native word interval."
        ),
        "probe_status": timing["probe_status"],
        "probe_error": timing["probe_error"],
        "source_video_path": timing["video_path"],
    }

    asr_payload = {
        "video_id": video_id,
        "kind": "asr",
        "source": {
            "features_root": features_root_label,
            "video_dir": str(video_dir),
            "feature_file": "asr.npy",
            "provider": "whisperx_projected_to_keyframes",
        },
        "timing": shared_timing,
        "counts": {
            "frame_dirs": len(frame_dirs),
            "npy_present": asr["present"],
            "npy_missing": asr["missing"],
            "decode_errors": asr["decode_errors"],
            "empty": asr["empty"],
            "nonempty_frames": asr["nonempty"],
            "segments": asr["segment_count"],
            "nonempty_chars": asr["char_count"],
        },
        "segments": asr["segments"],
        "frames": asr["frames"],
    }
    ocr_payload = {
        "video_id": video_id,
        "kind": "ocr",
        "source": {
            "features_root": features_root_label,
            "video_dir": str(video_dir),
            "feature_file": "ocr.npy",
            "provider": "tesseract_normalized_frame_text",
        },
        "timing": shared_timing,
        "counts": {
            "frame_dirs": len(frame_dirs),
            "npy_present": ocr["present"],
            "npy_missing": ocr["missing"],
            "decode_errors": ocr["decode_errors"],
            "empty": ocr["empty"],
            "nonempty_frames": ocr["nonempty"],
            "spans": ocr["segment_count"],
            "nonempty_chars": ocr["char_count"],
        },
        "spans": ocr["segments"],
        "frames": ocr["frames"],
    }

    write_json(output_dir / "asr" / "by-video" / f"{video_id}.json", asr_payload)
    write_text(output_dir / "asr" / "by-video" / f"{video_id}.txt", transcript_txt(asr["segments"]))
    write_json(output_dir / "ocr" / "by-video" / f"{video_id}.json", ocr_payload)
    write_text(output_dir / "ocr" / "by-video" / f"{video_id}.txt", transcript_txt(ocr["segments"]))

    return {
        "video_id": video_id,
        "corpus": video_id.split("_", 1)[0],
        "frame_dirs": len(frame_dirs),
        "rounded_fps": rounded_fps,
        "r_frame_rate": timing["r_frame_rate"],
        "fps": timing["fps"],
        "fps_status": timing["probe_status"],
        "asr_npy_present": asr["present"],
        "asr_npy_missing": asr["missing"],
        "asr_decode_errors": asr["decode_errors"],
        "asr_nonempty_frames": asr["nonempty"],
        "asr_segments": asr["segment_count"],
        "asr_chars": asr["char_count"],
        "ocr_npy_present": ocr["present"],
        "ocr_npy_missing": ocr["missing"],
        "ocr_decode_errors": ocr["decode_errors"],
        "ocr_nonempty_frames": ocr["nonempty"],
        "ocr_spans": ocr["segment_count"],
        "ocr_chars": ocr["char_count"],
        "asr_segments_payload": [
            {"video_id": video_id, **segment} for segment in asr["segments"]
        ],
        "ocr_spans_payload": [
            {"video_id": video_id, **segment} for segment in ocr["segments"]
        ],
        "ocr_frames_payload": [
            {"video_id": video_id, **frame} for frame in ocr["frames"]
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def map_videos(video_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in video_root.rglob("*.mp4"):
        if path.is_file():
            mapping[path.stem] = path
    return mapping


def render_readme(
    *,
    created_at: str,
    features_root: Path,
    video_root: Path,
    output_dir: Path,
    video_count: int,
    frame_dirs: int,
    asr_present: int,
    asr_segments: int,
    ocr_present: int,
    ocr_nonempty: int,
    ocr_spans: int,
    fps_ok: int,
    table_lines: list[str],
) -> str:
    return f"""# ASR / OCR text metadata v1

This package decodes the official keyframe-owned `asr.npy` and `ocr.npy` strings into readable derived metadata. The `.npy` files remain the original feature store. Dense embeddings (`asr_dense.npy`, `ocr_dense.npy`) are not decoded here.

## Result

| Item | Count |
|---|---:|
| Videos | {video_count} |
| Keyframe folders | {frame_dirs} |
| ASR `.npy` present | {asr_present} |
| ASR grouped segments | {asr_segments} |
| OCR `.npy` present | {ocr_present} |
| OCR nonempty frames | {ocr_nonempty} |
| OCR grouped spans | {ocr_spans} |
| FPS probes ok | {fps_ok}/{video_count} |

### Coverage by corpus

{chr(10).join(table_lines)}

## Evidence files

- `coverage.csv`: one row per video, with frame counts, nonempty rates, segment/span counts, and FPS probe status.
- `asr/by-video/<video_id>.json`: grouped ASR segments plus every nonempty keyframe projection.
- `asr/by-video/<video_id>.txt`: human-readable `[start -> end] text` transcript.
- `asr/segments.jsonl`: corpus-wide ASR segments.
- `ocr/by-video/<video_id>.json`: grouped OCR spans plus every nonempty keyframe string.
- `ocr/by-video/<video_id>.txt`: human-readable OCR spans.
- `ocr/spans.jsonl`: corpus-wide grouped OCR spans.
- `ocr/frames.jsonl`: corpus-wide nonempty OCR frames.
- `manifest.json`: source roots, method, counts, and SHA-256 hashes of the inventory files.
- `../../../tools/extract_asr_ocr_metadata.py`: reproducible generator.

## Method

- Observed at UTC: `{created_at}`.
- Feature root: `{features_root}` (portable relative root: `features_L21-L30_branch-feats-siglip/features`).
- Video root: `{video_root}` (portable relative root: `videos/`).
- Output root: `{output_dir}`.
- Text decode matches Vecna: `np.load(..., allow_pickle=True)` then `str(array.reshape(-1)[0])`.
- Consecutive identical nonempty strings are collapsed into one span. Empty frames are skipped and do not break a run of identical text. This matches Vecna `/api/video/transcript`.
- Timestamps use `frame_idx / rounded_fps`, where `rounded_fps` is `round(r_frame_rate)` from `ffprobe`. This matches the FPS rounding used when Vecna projected WhisperX onto keyframes.
- Interval endpoints are inclusive keyframe times, not native WhisperX word boundaries.
- ASR source is WhisperX text already projected onto keyframes. OCR source is Tesseract normalized frame text (`strip + lower + collapse whitespace`). Bounding boxes and native confidences are not present in the `.npy` files.
- These texts are evidence, not ground truth. OCR is noisy.

## Non-goals

- Does not re-run WhisperX or Tesseract.
- Does not decode `asr_dense.npy` / `ocr_dense.npy` (BGE-M3 vectors).
- Does not sentence-split ASR (Vecna's UI does that as a display convenience and fabricates sub-span times).
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Limit to one or more video IDs. Repeatable.",
    )
    args = parser.parse_args()

    features_root = args.features_root.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    if not features_root.is_dir():
        raise SystemExit(f"features root does not exist: {features_root}")

    selected = set(args.video)
    video_dirs = sorted(
        (path for path in features_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if selected:
        video_dirs = [path for path in video_dirs if path.name in selected]
        missing = selected.difference(path.name for path in video_dirs)
        if missing:
            raise SystemExit(f"requested videos not found in features root: {sorted(missing)}")
    if not video_dirs:
        raise SystemExit(f"no video feature folders found in {features_root}")

    video_map = map_videos(video_root) if video_root.is_dir() else {}
    features_root_label = "features_L21-L30_branch-feats-siglip/features"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1 or len(video_dirs) == 1:
        for video_dir in video_dirs:
            results.append(
                process_video(
                    video_dir.name,
                    video_dir,
                    video_map.get(video_dir.name),
                    args.ffprobe,
                    output_dir,
                    features_root_label,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    process_video,
                    video_dir.name,
                    video_dir,
                    video_map.get(video_dir.name),
                    args.ffprobe,
                    output_dir,
                    features_root_label,
                ): video_dir.name
                for video_dir in video_dirs
            }
            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(f"failed on {video_id}: {exc}") from exc

    results.sort(key=lambda row: row["video_id"])

    coverage_fields = [
        "video_id",
        "corpus",
        "frame_dirs",
        "rounded_fps",
        "r_frame_rate",
        "fps",
        "fps_status",
        "asr_npy_present",
        "asr_npy_missing",
        "asr_decode_errors",
        "asr_nonempty_frames",
        "asr_segments",
        "asr_chars",
        "ocr_npy_present",
        "ocr_npy_missing",
        "ocr_decode_errors",
        "ocr_nonempty_frames",
        "ocr_spans",
        "ocr_chars",
    ]
    coverage_rows = [{key: row[key] for key in coverage_fields} for row in results]
    for row in coverage_rows:
        row["fps"] = f"{float(row['fps']):.6f}" if row["fps"] != "" else ""

    asr_segments: list[dict[str, Any]] = []
    ocr_spans: list[dict[str, Any]] = []
    ocr_frames: list[dict[str, Any]] = []
    for row in results:
        asr_segments.extend(row.pop("asr_segments_payload"))
        ocr_spans.extend(row.pop("ocr_spans_payload"))
        ocr_frames.extend(row.pop("ocr_frames_payload"))

    coverage_path = output_dir / "coverage.csv"
    write_csv(coverage_path, coverage_fields, coverage_rows)
    write_jsonl(output_dir / "asr" / "segments.jsonl", asr_segments)
    write_jsonl(output_dir / "ocr" / "spans.jsonl", ocr_spans)
    write_jsonl(output_dir / "ocr" / "frames.jsonl", ocr_frames)

    frame_dirs_total = sum(int(row["frame_dirs"]) for row in coverage_rows)
    asr_present = sum(int(row["asr_npy_present"]) for row in coverage_rows)
    asr_segment_count = sum(int(row["asr_segments"]) for row in coverage_rows)
    ocr_present = sum(int(row["ocr_npy_present"]) for row in coverage_rows)
    ocr_nonempty = sum(int(row["ocr_nonempty_frames"]) for row in coverage_rows)
    ocr_span_count = sum(int(row["ocr_spans"]) for row in coverage_rows)
    fps_ok = sum(1 for row in coverage_rows if row["fps_status"] == "ok")

    by_corpus: dict[str, dict[str, int]] = {}
    for row in coverage_rows:
        bucket = by_corpus.setdefault(
            str(row["corpus"]),
            {
                "videos": 0,
                "frames": 0,
                "asr_segments": 0,
                "ocr_nonempty": 0,
            },
        )
        bucket["videos"] += 1
        bucket["frames"] += int(row["frame_dirs"])
        bucket["asr_segments"] += int(row["asr_segments"])
        bucket["ocr_nonempty"] += int(row["ocr_nonempty_frames"])

    table_lines = [
        "| Corpus | Videos | Frames | ASR segments | OCR nonempty frames |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus in sorted(by_corpus):
        bucket = by_corpus[corpus]
        table_lines.append(
            f"| `{corpus}` | {bucket['videos']} | {bucket['frames']} | {bucket['asr_segments']} | {bucket['ocr_nonempty']} |"
        )
    table_lines.append(
        f"| **TOTAL** | **{len(coverage_rows)}** | **{frame_dirs_total}** | **{asr_segment_count}** | **{ocr_nonempty}** |"
    )

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        render_readme(
            created_at=created_at,
            features_root=features_root,
            video_root=video_root,
            output_dir=output_dir,
            video_count=len(coverage_rows),
            frame_dirs=frame_dirs_total,
            asr_present=asr_present,
            asr_segments=asr_segment_count,
            ocr_present=ocr_present,
            ocr_nonempty=ocr_nonempty,
            ocr_spans=ocr_span_count,
            fps_ok=fps_ok,
            table_lines=table_lines,
        ),
        encoding="utf-8",
        newline="\n",
    )

    catalog = {
        "artifact": "asr-ocr-text-v1",
        "created_at_utc": created_at,
        "video_count": len(coverage_rows),
        "videos": [row["video_id"] for row in coverage_rows],
        "by_corpus": by_corpus,
    }
    catalog_path = output_dir / "catalog.json"
    write_json(catalog_path, catalog)

    manifest = {
        "artifact": "asr-ocr-text-v1",
        "created_at_utc": created_at,
        "source_features_root": str(features_root),
        "source_features_root_relative": "features_L21-L30_branch-feats-siglip/features",
        "source_video_root": str(video_root),
        "source_video_root_relative": "videos",
        "generator": "tools/extract_asr_ocr_metadata.py",
        "decode": "numpy allow_pickle; str(reshape(-1)[0]); strip",
        "grouping": "consecutive identical nonempty strings; empty frames skipped",
        "timestamp_formula": "frame_idx / round(r_frame_rate)",
        "video_count": len(coverage_rows),
        "frame_dir_count": frame_dirs_total,
        "asr_npy_present": asr_present,
        "asr_segment_count": asr_segment_count,
        "ocr_npy_present": ocr_present,
        "ocr_nonempty_frame_count": ocr_nonempty,
        "ocr_span_count": ocr_span_count,
        "fps_probe_ok_count": fps_ok,
        "evidence_files": {
            "coverage": {"path": coverage_path.name, "sha256": sha256_file(coverage_path)},
            "catalog": {"path": catalog_path.name, "sha256": sha256_file(catalog_path)},
            "readme": {"path": readme_path.name, "sha256": sha256_file(readme_path)},
            "asr_segments": {
                "path": "asr/segments.jsonl",
                "sha256": sha256_file(output_dir / "asr" / "segments.jsonl"),
            },
            "ocr_spans": {
                "path": "ocr/spans.jsonl",
                "sha256": sha256_file(output_dir / "ocr" / "spans.jsonl"),
            },
            "ocr_frames": {
                "path": "ocr/frames.jsonl",
                "sha256": sha256_file(output_dir / "ocr" / "frames.jsonl"),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "video_count": len(coverage_rows),
                "frame_dirs": frame_dirs_total,
                "asr_segments": asr_segment_count,
                "ocr_nonempty_frames": ocr_nonempty,
                "ocr_spans": ocr_span_count,
                "fps_ok": fps_ok,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
