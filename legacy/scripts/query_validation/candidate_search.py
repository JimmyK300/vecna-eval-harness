"""Read-only lexical candidate search over existing OCR/ASR/metadata artifacts.

The feature store uses scalar Unicode .npy files for OCR and ASR. This module
reads those files with the NumPy file header format from the standard library;
it never writes to or regenerates the feature store.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
from pathlib import Path


MEDIA_RE = re.compile(r"(?P<media>L(?:2[1-9]|30|26[A-E])_V\d{3})")
TEXT_SURFACES = {"ocr.npy": "OCR", "asr.npy": "ASR"}


def scalar_npy_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        return ""
    major = raw[6]
    if major == 1:
        header_size = struct.unpack("<H", raw[8:10])[0]
        data_start = 10 + header_size
    else:
        header_size = struct.unpack("<I", raw[8:12])[0]
        data_start = 12 + header_size
    header = raw[10:data_start] if major == 1 else raw[12:data_start]
    match = re.search(rb"'descr': '(<U)(\d+)'", header)
    if not match:
        return ""
    width = int(match.group(2))
    data = raw[data_start : data_start + width * 4]
    try:
        return data.decode("utf-32-le", errors="ignore").rstrip("\x00")
    except UnicodeDecodeError:
        return ""


def term_hits(text: str, terms: list[str]) -> list[str]:
    folded = text.casefold()
    return [term for term in terms if term.casefold() in folded]


def search_feature_text(root: Path, terms: list[str], limit: int):
    per_media: dict[tuple[str, str], dict] = {}
    for media_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        media_id = media_dir.name
        if not MEDIA_RE.fullmatch(media_id):
            continue
        for path in media_dir.rglob("*.npy"):
            modality = TEXT_SURFACES.get(path.name)
            if not modality:
                continue
            raw = path.read_bytes()
            encoded_terms = {
                term: (term.encode("utf-32-le"), term.casefold().encode("utf-32-le"))
                for term in terms
            }
            hits = [
                term
                for term, variants in encoded_terms.items()
                if any(variant in raw for variant in variants)
            ]
            if not hits:
                continue
            text = scalar_npy_text(path)
            key = (media_id, modality)
            record = per_media.setdefault(
                key,
                {
                    "media_id": media_id,
                    "evidence_modality": modality,
                    "matching_text": text,
                    "timestamp_or_frame": path.parent.name,
                    "source_artifact": str(path),
                    "score_or_rank": len(hits),
                    "matched_terms": set(),
                },
            )
            record["matched_terms"].update(hits)
            record["score_or_rank"] = max(record["score_or_rank"], len(record["matched_terms"]))
    rows = list(per_media.values())
    rows.sort(key=lambda row: (-row["score_or_rank"], row["media_id"], row["evidence_modality"]))
    for row in rows[:limit]:
        row["matched_terms"] = ";".join(sorted(row["matched_terms"]))
        yield row


def search_named_metadata(root: Path, terms: list[str], limit: int):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".srt", ".txt", ".csv", ".jsonl", ".md"}:
            continue
        match = MEDIA_RE.search(path.name)
        if not match:
            match = MEDIA_RE.search(str(path))
        if not match:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = term_hits(text, terms)
        if hits:
            rows.append(
                {
                    "media_id": match.group("media"),
                    "evidence_modality": "metadata",
                    "matching_text": " ".join(text.split())[:500],
                    "timestamp_or_frame": "available in artifact" if "-->" in text else "",
                    "source_artifact": str(path),
                    "score_or_rank": len(hits),
                    "matched_terms": ";".join(hits),
                }
            )
    rows.sort(key=lambda row: (-row["score_or_rank"], row["media_id"]))
    yield from rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = list(search_feature_text(args.feature_root, args.query, args.limit))
    if args.metadata_root:
        rows.extend(search_named_metadata(args.metadata_root, args.query, args.limit))
    fields = [
        "media_id",
        "evidence_modality",
        "matching_text",
        "timestamp_or_frame",
        "source_artifact",
        "score_or_rank",
        "matched_terms",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
