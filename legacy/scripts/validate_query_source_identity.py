#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 2
EXPECTED = {
    "p0": {"round": "P0", "count": 24, "prefix": "query-p0-"},
    "p1": {"round": "P1", "count": 25, "prefix": "query-p1-"},
    "p2": {"round": "P2", "count": 30, "prefix": "query-p2-"},
    "p3": {"round": "P3", "count": 36, "prefix": "query-p3-"},
}
TOTAL = sum(item["count"] for item in EXPECTED.values())
QUERY_RE = re.compile(r"^### `query-(p[0-3])-(\d+)-(kis|qa|trake)`\s*$", re.MULTILINE)


class IdentityError(ValueError):
    pass


def parse_frontmatter(text: str, *, path: Path) -> dict[str, str | int]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise IdentityError(f"{path}: missing query-source frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise IdentityError(f"{path}: unterminated query-source frontmatter") from exc

    metadata: dict[str, str | int] = {}
    for line_number, line in enumerate(lines[1:end], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or ":" not in line:
            raise IdentityError(f"{path}:{line_number}: canonical frontmatter must be flat key: value")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise IdentityError(f"{path}:{line_number}: invalid frontmatter entry")
        metadata[key] = int(value) if value.isdigit() else value
    return metadata


def validate_source(path: Path, source_id: str) -> int:
    spec = EXPECTED[source_id]
    text = path.read_text(encoding="utf-8")
    metadata = parse_frontmatter(text, path=path)

    required = {
        "query_source_schema": SCHEMA_VERSION,
        "canonical_source_id": source_id,
        "canonical_round": spec["round"],
        "question_count": spec["count"],
        "source_scope": "current",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise IdentityError(f"{path}: {key} must be {expected!r}, got {metadata.get(key)!r}")

    matches = QUERY_RE.findall(text)
    if len(matches) != spec["count"]:
        raise IdentityError(
            f"{path}: expected {spec['count']} canonical query headings, found {len(matches)}"
        )
    wrong = [f"query-{round_id}-{number}-{kind}" for round_id, number, kind in matches if round_id != source_id]
    if wrong:
        raise IdentityError(f"{path}: wrong canonical query prefix: {wrong[0]}")

    numbers = [int(number) for _, number, _ in matches]
    if len(numbers) != len(set(numbers)):
        raise IdentityError(f"{path}: duplicate canonical query number")
    return len(matches)


def validate_manifest(current_root: Path) -> None:
    path = current_root / "canonical-rounds.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IdentityError(f"{path}: invalid or missing canonical manifest") from exc

    if payload.get("schema_version") != 1:
        raise IdentityError(f"{path}: schema_version must be 1")
    if payload.get("authority") != "Official-Queries/current-rounds":
        raise IdentityError(f"{path}: invalid authority")
    if payload.get("total_raw_queries") != TOTAL:
        raise IdentityError(f"{path}: total_raw_queries must be {TOTAL}")

    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 4:
        raise IdentityError(f"{path}: rounds must contain exactly four entries")
    by_id = {item.get("canonical_source_id"): item for item in rounds if isinstance(item, dict)}
    if set(by_id) != set(EXPECTED):
        raise IdentityError(f"{path}: canonical source IDs must be exactly {sorted(EXPECTED)}")
    for source_id, spec in EXPECTED.items():
        item = by_id[source_id]
        expected_path = f"Official-Queries/current-rounds/{source_id}.md"
        expected = {
            "canonical_round": spec["round"],
            "question_count": spec["count"],
            "query_prefix": spec["prefix"],
            "path": expected_path,
        }
        for key, value in expected.items():
            if item.get(key) != value:
                raise IdentityError(f"{path}: {source_id}.{key} must be {value!r}")


def validate_repository(repo_root: Path) -> dict[str, int]:
    current_root = repo_root / "Official-Queries" / "current-rounds"
    if not current_root.is_dir():
        raise IdentityError(f"missing canonical source directory: {current_root}")

    expected_md = {"README.md", *(f"{source_id}.md" for source_id in EXPECTED)}
    actual_md = {path.name for path in current_root.glob("*.md")}
    if actual_md != expected_md:
        extra = sorted(actual_md - expected_md)
        missing = sorted(expected_md - actual_md)
        raise IdentityError(f"{current_root}: canonical Markdown set mismatch; extra={extra}, missing={missing}")

    validate_manifest(current_root)
    counts = {source_id: validate_source(current_root / f"{source_id}.md", source_id) for source_id in EXPECTED}
    total = sum(counts.values())
    if total != TOTAL:
        raise IdentityError(f"canonical query total must be {TOTAL}, found {total}")
    return {"canonical_sources": 4, "canonical_queries": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate canonical P0/P1/P2/P3 query sources.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    try:
        counts = validate_repository(args.root.resolve())
    except (IdentityError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {counts['canonical_sources']} canonical sources; {counts['canonical_queries']} canonical queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
