#!/usr/bin/env python3
"""Deterministic slicing, normalization, merging, and auditing for the L23 AGY test."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "gemini-3.7-flash-high"
ACTION_FIELDS = (
    "actors", "actions", "objects", "attributes", "spatial_relationships",
    "temporal_relationships", "visual_observations", "spoken_facts",
    "visible_text", "interpretations", "uncertain_inferences",
)
SEGMENT_TYPES = {
    "meaningful_action", "stable_visual_state", "replay", "interview",
    "transition", "static_graphic", "nonsemantic_interval",
}


class PipelineError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise PipelineError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"expected one JSON object in {path}")
    return value


def extract_json_object(text: str, worker_role: str) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(text.lstrip("\ufeff"))
        if isinstance(value, dict) and value.get("worker_role") == worker_role:
            return value, "exact"
    except Exception:
        pass
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except Exception:
            continue
        if isinstance(value, dict) and value.get("worker_role") == worker_role:
            candidates.append(value)
    if not candidates:
        raise PipelineError(f"no parseable {worker_role} object in worker output")
    return candidates[-1], "extracted_last_matching_object"


def parse_worker_outputs(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    failures = 0
    for path in sorted(input_dir.glob(args.pattern)):
        role = args.role or ("video_worker" if path.stem.startswith(args.video_id + "_video") else "scene_worker")
        try:
            value, mode = extract_json_object(path.read_text(encoding="utf-8-sig"), role)
            if value.get("status") != "success":
                raise PipelineError(f"worker status is {value.get('status')}: {value.get('failure_reason')}")
            target_name = (args.video_id + "_video.json") if role == "video_worker" else path.name.replace(".stdout.txt", ".json")
            target = output_dir / target_name
            write_json(target, value)
            report.append({"source": str(path.resolve()), "parsed": str(target.resolve()), "mode": mode, "status": "success"})
        except Exception as exc:
            failures += 1
            report.append({"source": str(path.resolve()), "status": "failure", "reason": str(exc)})
    write_json(Path(args.report), {"video_id": args.video_id, "failures": failures, "workers": report})
    print(json.dumps({"video_id": args.video_id, "workers": len(report), "failures": failures}))
    if failures:
        raise PipelineError(f"{failures} worker outputs could not be parsed")


def apply_adjudication(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    changed = 0
    with Path(args.input).open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") == args.record_id:
                for actor in row.get("actors", []):
                    if actor.get("raw_name_heard") == args.old_raw_name:
                        actor["raw_name_heard"] = args.new_raw_name
                        changed += 1
                row["adjudication"] = {
                    "worker_id": args.audit_worker_id,
                    "resolution": args.resolution,
                    "change": f"raw_name_heard: {args.old_raw_name} -> {args.new_raw_name}",
                }
            rows.append(row)
    if changed != 1:
        raise PipelineError(f"expected exactly one adjudication change, made {changed}")
    write_jsonl(Path(args.output), rows)
    print(json.dumps({"record_id": args.record_id, "changed": changed}))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def verify_worker(obj: dict[str, Any], expected_role: str) -> None:
    if obj.get("worker_role") != expected_role:
        raise PipelineError(f"expected worker_role={expected_role!r}, got {obj.get('worker_role')!r}")
    if obj.get("actual_model_identifier") != MODEL_ID:
        raise PipelineError(f"model mismatch: {obj.get('actual_model_identifier')!r}")
    if obj.get("status") != "success":
        raise PipelineError(f"worker failed: {obj.get('failure_reason')!r}")
    if obj.get("direct_video_evidence_observed") is not True:
        raise PipelineError("worker did not attest direct video evidence")
    if obj.get("audio_evidence_observed") is not True:
        raise PipelineError("worker did not attest audio evidence")


def validate_video_boundaries(obj: dict[str, Any], video_id: str, duration: float) -> list[dict[str, Any]]:
    verify_worker(obj, "video_worker")
    if obj.get("video_id") != video_id:
        raise PipelineError(f"video ID mismatch: {obj.get('video_id')!r}")
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise PipelineError("video worker returned no scenes")
    previous_end = 0.0
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            raise PipelineError(f"scene {index} is not an object")
        scene_id = str(scene.get("scene_id", ""))
        if not re.fullmatch(re.escape(video_id) + r"_E\d+_S\d+", scene_id):
            raise PipelineError(f"invalid scene ID: {scene_id!r}")
        if scene_id in seen:
            raise PipelineError(f"duplicate scene ID: {scene_id}")
        seen.add(scene_id)
        start = float(scene["source_start"])
        end = float(scene["source_end"])
        if start < -0.05 or end <= start or end > duration + 0.25:
            raise PipelineError(f"scene {scene_id} has invalid bounds {start}-{end}")
        if abs(start - previous_end) > 0.25:
            raise PipelineError(f"scene gap/overlap before {scene_id}: previous={previous_end}, start={start}")
        previous_end = end
        normalized.append(scene)
    if abs(previous_end - duration) > 0.35:
        raise PipelineError(f"scene timeline ends at {previous_end}, video ends at {duration}")
    return normalized


def make_slices(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    output = read_json(Path(args.video_output))
    scenes = validate_video_boundaries(output, args.video_id, args.duration)
    if args.limit:
        scenes = scenes[: args.limit]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments: list[dict[str, Any]] = []
    for scene in scenes:
        source_start = max(0.0, float(scene["source_start"]))
        source_end = min(args.duration, float(scene["source_end"]))
        actual_start = max(0.0, source_start - args.overlap)
        actual_end = min(args.duration, source_end + args.overlap)
        target = out_dir / f"{scene['scene_id']}.mp4"
        assignment = {
            "source_video_id": args.video_id,
            "event_id": scene["event_id"],
            "scene_id": scene["scene_id"],
            "source_start": round(source_start, 6),
            "source_end": round(source_end, 6),
            "slice_actual_start": round(actual_start, 6),
            "slice_actual_end": round(actual_end, 6),
            "slice_path": str(target),
            "boundary_reason": scene.get("boundary_reason", ""),
        }
        if target.exists() and not args.overwrite:
            assignment["slice_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest().upper()
            assignments.append(assignment)
            continue
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{actual_start:.6f}", "-i", str(source),
            "-t", f"{actual_end - actual_start:.6f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise PipelineError(f"ffmpeg failed for {scene['scene_id']}: {completed.stderr.strip()}")
        assignment["slice_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest().upper()
        assignments.append(assignment)
    write_json(Path(args.assignments), assignments)
    print(json.dumps({"slices_created": len(assignments), "assignments": str(Path(args.assignments).resolve())}))


def render_scene_prompts(args: argparse.Namespace) -> None:
    template = Path(args.template).read_text(encoding="utf-8-sig")
    assignments = json.loads(Path(args.assignments).read_text(encoding="utf-8-sig"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    for assignment in assignments:
        if args.only and assignment["scene_id"] not in args.only:
            continue
        text = template
        replacements = {
            "{{ABS_SLICE_PATH}}": str(Path(assignment["slice_path"]).resolve()),
            "{{ASSIGNMENT_JSON}}": json.dumps(assignment, indent=2, ensure_ascii=False),
            "{{VIDEO_ID}}": assignment["source_video_id"],
            "{{SCENE_ID}}": assignment["scene_id"],
        }
        for needle, value in replacements.items():
            text = text.replace(needle, value)
        target = out_dir / f"{assignment['scene_id']}.scene_worker.md"
        target.write_bytes(text.encode("utf-8"))
        rendered.append(str(target.resolve()))
    print(json.dumps({"prompts_rendered": len(rendered), "paths": rendered}))


def render_video_prompts(args: argparse.Namespace) -> None:
    template = Path(args.template).read_text(encoding="utf-8-sig")
    manifest = read_json(Path(args.manifest))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    root = Path(args.workspace).resolve()
    for video in manifest["videos"]:
        if args.only and video["id"] not in args.only:
            continue
        text = template
        replacements = {
            "{{VIDEO_ID}}": video["id"],
            "{{ABS_VIDEO_PATH}}": str((root / video["path"]).resolve()),
            "{{DURATION_SECONDS}}": str(video["duration_seconds"]),
        }
        for needle, value in replacements.items():
            text = text.replace(needle, value)
        target = out_dir / f"{video['id']}.video_worker.md"
        target.write_bytes(text.encode("utf-8"))
        rendered.append(str(target.resolve()))
    print(json.dumps({"prompts_rendered": len(rendered), "paths": rendered}))


def clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def normalize(args: argparse.Namespace) -> None:
    video_obj = read_json(Path(args.video_output))
    scenes = validate_video_boundaries(video_obj, args.video_id, args.duration)
    assignments_value = json.loads(Path(args.assignments).read_text(encoding="utf-8-sig"))
    assignments = {item["scene_id"]: item for item in assignments_value}
    scene_output_dir = Path(args.scene_output_dir)
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = [video_obj]

    event_map = {item["event_id"]: item for item in video_obj.get("major_events", [])}
    for event_id, event in event_map.items():
        rows.append({
            "id": event_id,
            "type": "event",
            "parent_id": args.video_id,
            "start": float(event["start"]),
            "end": float(event["end"]),
            "summary": event.get("summary", ""),
            "visual_observations": event.get("visual_observations", []),
            "spoken_facts": event.get("spoken_facts", []),
            "visible_text": event.get("visible_text", []),
            "interpretations": event.get("interpretations", []),
            "uncertain_inferences": event.get("uncertain_inferences", []),
        })

    action_count = 0
    for scene in scenes:
        scene_id = scene["scene_id"]
        if scene_id not in assignments:
            raise PipelineError(f"missing assignment for {scene_id}")
        assignment = assignments[scene_id]
        output_path = scene_output_dir / f"{scene_id}.json"
        if not output_path.exists():
            output_path = scene_output_dir / f"{scene_id}.stdout.txt"
        scene_obj = read_json(output_path)
        verify_worker(scene_obj, "scene_worker")
        if scene_obj.get("scene_id") != scene_id:
            raise PipelineError(f"scene output ID mismatch in {output_path}")
        raw_rows.append(scene_obj)
        rows.append({
            "id": scene_id,
            "type": "scene",
            "parent_id": scene["event_id"],
            "start": float(scene["source_start"]),
            "end": float(scene["source_end"]),
            "boundary_reason": scene.get("boundary_reason", ""),
            "summary": scene.get("summary", ""),
            "assignment": assignment,
        })
        actions = scene_obj.get("actions_local")
        if not isinstance(actions, list) or not actions:
            raise PipelineError(f"no actions_local in {output_path}")
        previous_end = float(scene["source_start"])
        for ordinal, action in enumerate(actions, 1):
            segment_type = action.get("segment_type")
            if segment_type not in SEGMENT_TYPES:
                raise PipelineError(f"invalid segment_type in {scene_id}: {segment_type!r}")
            source_start = float(assignment["slice_actual_start"]) + float(action["local_start"])
            source_end = float(assignment["slice_actual_start"]) + float(action["local_end"])
            source_start = clip(source_start, float(scene["source_start"]), float(scene["source_end"]))
            source_end = clip(source_end, float(scene["source_start"]), float(scene["source_end"]))
            if source_end <= source_start:
                continue
            if abs(source_start - previous_end) > 0.30:
                raise PipelineError(f"action gap/overlap in {scene_id}: previous={previous_end}, start={source_start}")
            previous_end = source_end
            action_count += 1
            row: dict[str, Any] = {
                "id": f"{scene_id}_A{ordinal:03d}",
                "type": "action",
                "parent_id": scene_id,
                "segment_type": segment_type,
                "start": round(source_start, 6),
                "end": round(source_end, 6),
            }
            for field in ACTION_FIELDS:
                row[field] = action.get(field, [])
            row["retrieval_text"] = action.get("retrieval_text", "")
            row["confidence"] = action.get("confidence", "low")
            if action.get("long_record_justification"):
                row["long_record_justification"] = action["long_record_justification"]
            rows.append(row)
        if abs(previous_end - float(scene["source_end"])) > 0.30:
            raise PipelineError(f"action timeline does not reach scene end in {scene_id}")

    video_row = {
        "id": args.video_id,
        "type": "video",
        "parent_id": None,
        "start": 0.0,
        "end": args.duration,
        "topic": video_obj.get("topic", ""),
        "source_path": args.source_path,
        "direct_video_evidence_observed": True,
        "audio_evidence_observed": True,
    }
    rows.insert(0, video_row)
    write_jsonl(Path(args.raw_jsonl), raw_rows)
    write_jsonl(Path(args.normalized_jsonl), rows)
    print(json.dumps({"records": len(rows), "actions": action_count}))


def validate_scene_probe(args: argparse.Namespace) -> None:
    assignment_list = json.loads(Path(args.assignments).read_text(encoding="utf-8-sig"))
    if len(assignment_list) != 1:
        raise PipelineError("capability probe requires exactly one assignment")
    assignment = assignment_list[0]
    scene = read_json(Path(args.scene_output))
    verify_worker(scene, "scene_worker")
    if scene.get("scene_id") != assignment["scene_id"]:
        raise PipelineError("probe scene ID does not match assignment")
    converted: list[dict[str, Any]] = []
    previous = float(assignment["source_start"])
    warnings: list[str] = []
    for ordinal, action in enumerate(scene.get("actions_local", []), 1):
        local_start = float(action["local_start"])
        local_end = float(action["local_end"])
        start = clip(float(assignment["slice_actual_start"]) + local_start,
                     float(assignment["source_start"]), float(assignment["source_end"]))
        end = clip(float(assignment["slice_actual_start"]) + local_end,
                   float(assignment["source_start"]), float(assignment["source_end"]))
        if end <= start:
            raise PipelineError(f"probe action {ordinal} becomes empty after timestamp conversion")
        if abs(start - previous) > 0.30:
            raise PipelineError(f"probe timeline discontinuity before action {ordinal}: {previous} vs {start}")
        length = end - start
        limit = 15.0 if action.get("segment_type") == "meaningful_action" else 30.0
        if length > limit + 0.01 and not action.get("long_record_justification"):
            raise PipelineError(f"probe action {ordinal} exceeds {limit}s without justification")
        row = dict(action)
        row.pop("local_start", None)
        row.pop("local_end", None)
        row.update({
            "id": f"{assignment['scene_id']}_A{ordinal:03d}",
            "type": "action",
            "parent_id": assignment["scene_id"],
            "start": round(start, 6),
            "end": round(end, 6),
        })
        converted.append(row)
        previous = end
    if not converted:
        raise PipelineError("probe returned no actions")
    if abs(previous - float(assignment["source_end"])) > 0.30:
        raise PipelineError("probe actions do not cover the assigned interval")
    if all(row.get("confidence") == "high" for row in converted):
        warnings.append("uniformly high confidence")
    write_jsonl(Path(args.converted_output), converted)
    result = {
        "status": "pass",
        "model_identifier": MODEL_ID,
        "direct_video_evidence_observed": True,
        "audio_evidence_observed": True,
        "scene_id": assignment["scene_id"],
        "assignment": assignment,
        "action_count": len(converted),
        "converted_start": converted[0]["start"],
        "converted_end": converted[-1]["end"],
        "warnings": warnings,
    }
    write_json(Path(args.report), result)
    print(json.dumps(result, ensure_ascii=False))


def find_gaps(actions: list[dict[str, Any]], duration: float, tolerance: float = 0.05) -> tuple[list[list[float]], list[list[float]]]:
    gaps: list[list[float]] = []
    overlaps: list[list[float]] = []
    cursor = 0.0
    for row in sorted(actions, key=lambda item: (float(item["start"]), float(item["end"]))):
        start, end = float(row["start"]), float(row["end"])
        if start > cursor + tolerance:
            gaps.append([round(cursor, 6), round(start, 6)])
        elif start < cursor - tolerance:
            overlaps.append([round(start, 6), round(cursor, 6)])
        cursor = max(cursor, end)
    if cursor < duration - tolerance:
        gaps.append([round(cursor, 6), round(duration, 6)])
    return gaps, overlaps


def audit(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    with Path(args.normalized_jsonl).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("not an object")
                rows.append(value)
            except Exception as exc:
                errors.append(f"invalid JSONL line {line_number}: {exc}")
    ids = {row.get("id") for row in rows}
    if len(ids) != len(rows):
        errors.append("duplicate record IDs")
    for row in rows:
        if row.get("type") != "video" and row.get("parent_id") not in ids:
            errors.append(f"missing parent for {row.get('id')}")
        if float(row.get("start", -1)) < 0 or float(row.get("end", math.inf)) > args.duration + 0.01:
            errors.append(f"timestamp out of bounds: {row.get('id')}")
        if float(row.get("end", 0)) <= float(row.get("start", 0)):
            errors.append(f"non-positive duration: {row.get('id')}")
    actions = [row for row in rows if row.get("type") == "action"]
    gaps, overlaps = find_gaps(actions, args.duration)
    if gaps:
        errors.append(f"uncovered intervals: {gaps}")
    if overlaps:
        warnings.append(f"overlapping intervals: {overlaps}")
    over_limit: list[dict[str, Any]] = []
    duplicates: list[list[str]] = []
    unsupported_measurements: list[str] = []
    commentary_mislabeled: list[str] = []
    numeric_claim = re.compile(r"\b\d+(?:\.\d+)?\s*(?:km/h|kph|mph|seconds?|secs?|meters?|metres?|m|%)\b", re.I)
    for index, row in enumerate(actions):
        length = float(row["end"]) - float(row["start"])
        limit = 15.0 if row.get("segment_type") == "meaningful_action" else 30.0
        if length > limit + 0.01 and not row.get("long_record_justification"):
            over_limit.append({"id": row["id"], "duration": round(length, 3), "limit": limit})
        text = " ".join(str(x) for f in ACTION_FIELDS for x in row.get(f, []))
        if numeric_claim.search(text) and not row.get("visible_text") and not row.get("spoken_facts"):
            unsupported_measurements.append(row["id"])
        visual_text = " ".join(str(x) for x in row.get("visual_observations", [])).casefold()
        if re.search(r"\b(commentator|commentary|spoken|heard|says|states)\b", visual_text):
            commentary_mislabeled.append(row["id"])
        if index:
            prev = actions[index - 1]
            if row.get("retrieval_text", "").strip().casefold() == prev.get("retrieval_text", "").strip().casefold():
                duplicates.append([prev["id"], row["id"]])
    if over_limit:
        errors.append(f"records exceeding duration limits: {over_limit}")
    if duplicates:
        warnings.append(f"duplicate neighboring records: {duplicates}")
    if unsupported_measurements:
        warnings.append(f"possibly unsupported exact measurements: {unsupported_measurements}")
    if commentary_mislabeled:
        warnings.append(f"possible commentary mislabeled as visual evidence: {commentary_mislabeled}")
    confidence_values = [row.get("confidence") for row in actions]
    if confidence_values and all(value == "high" for value in confidence_values):
        warnings.append("uniformly high confidence")
    unresolved = sorted({
        actor.get("raw_name_heard") or actor.get("canonical_name") or "(unnamed)"
        for row in actions for actor in row.get("actors", [])
        if isinstance(actor, dict) and actor.get("canonicalization_source") == "unresolved"
    })
    durations = [float(row["end"]) - float(row["start"]) for row in actions]
    adjudications = [
        {"record_id": row["id"], **row["adjudication"]}
        for row in actions if isinstance(row.get("adjudication"), dict)
    ]
    report = {
        "video_id": args.video_id,
        "valid_jsonl": not any(item.startswith("invalid JSONL") for item in errors),
        "record_counts": {
            "events": sum(row.get("type") == "event" for row in rows),
            "scenes": sum(row.get("type") == "scene" for row in rows),
            "actions": len(actions),
        },
        "average_action_duration": round(sum(durations) / len(durations), 3) if durations else None,
        "maximum_action_duration": round(max(durations), 3) if durations else None,
        "uncovered_intervals": gaps,
        "timeline_overlaps": overlaps,
        "contradiction_count": len(adjudications),
        "contradictions": adjudications,
        "unresolved_identities": unresolved,
        "checks": {
            "parent_child_references": not any("parent" in item for item in errors),
            "timestamps_within_duration": not any("timestamp" in item for item in errors),
            "duration_limits": not over_limit,
            "duplicate_neighbors": not duplicates,
            "uniform_confidence": not (confidence_values and all(value == "high" for value in confidence_values)),
            "commentary_mislabeled_as_visual": not commentary_mislabeled,
            "telemetry_consistency": not unsupported_measurements,
            "cross_video_spelling": "checked at final merge",
        },
        "errors": errors,
        "warnings": warnings,
        "audit_verdict": "pass" if not errors else "fail",
    }
    write_json(Path(args.audit_output), report)
    print(json.dumps(report))
    if errors:
        raise PipelineError("audit failed")


def merge_final(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    spellings: dict[str, set[str]] = {}
    for source in args.inputs:
        with Path(source).open(encoding="utf-8-sig") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(row)
                if row.get("type") == "action":
                    for actor in row.get("actors", []):
                        if isinstance(actor, dict) and actor.get("canonical_name"):
                            name = str(actor["canonical_name"])
                            spellings.setdefault(name.casefold(), set()).add(name)
    inconsistent = {key: sorted(values) for key, values in spellings.items() if len(values) > 1}
    if inconsistent:
        raise PipelineError(f"cross-video spelling inconsistencies: {inconsistent}")
    write_jsonl(Path(args.output), rows)
    print(json.dumps({"records": len(rows), "output": str(Path(args.output).resolve())}))


def normalize_canonical_spellings(args: argparse.Namespace) -> None:
    per_file: dict[Path, list[dict[str, Any]]] = {}
    counts: dict[str, collections.Counter[str]] = {}
    for source_text in args.inputs:
        source = Path(source_text)
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line]
        per_file[source] = rows
        for row in rows:
            if row.get("type") != "action":
                continue
            for actor in row.get("actors", []):
                if isinstance(actor, dict) and actor.get("canonical_name"):
                    value = str(actor["canonical_name"])
                    counts.setdefault(value.casefold(), collections.Counter())[value] += 1
    choices = {
        key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for key, counter in counts.items()
    }
    changes: list[dict[str, str]] = []
    for source, rows in per_file.items():
        for row in rows:
            if row.get("type") != "action":
                continue
            for actor in row.get("actors", []):
                if isinstance(actor, dict) and actor.get("canonical_name"):
                    old = str(actor["canonical_name"])
                    new = choices[old.casefold()]
                    if old != new:
                        actor["canonical_name"] = new
                        changes.append({"record_id": row["id"], "old": old, "new": new})
        write_jsonl(source, rows)
    write_json(Path(args.report), {"policy": "most_frequent_then_lexicographic", "changes": changes})
    print(json.dumps({"changes": len(changes), "report": str(Path(args.report).resolve())}))


def snapshot_archived_prompts(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment).resolve()
    current_template = (experiment / "prompts" / "scene_worker.md").read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    current_policy = (
        "The slice MP4 is your only evidence. Do not read the full-video worker output, transcripts, ASR/OCR files, cached JSON, filenames as semantic evidence, web search, or external knowledge. "
        "Do not call terminal, command, web, or sub-agent tools. You may and must use AGY's native multimodal file-opening capability to ingest the explicitly attached MP4 path; that is not a fallback source. "
        "If the slice cannot be ingested, frames cannot be inspected, audio cannot be heard, or the selected model is not gemini-3.7-flash-high, return a failure object with the exact reason and no semantic guesses."
    )
    old_policies = [
        (
            "The slice MP4 is your only evidence. Do not read the full-video worker output, transcripts, ASR/OCR files, cached JSON, filenames as semantic evidence, web search, or external knowledge. "
            "If the slice cannot be ingested, frames cannot be inspected, audio cannot be heard, or the selected model is not gemini-3.7-flash-high, return a failure object with the exact reason and no semantic guesses."
        ),
        (
            "The slice MP4 is your only evidence. Do not read the full-video worker output, transcripts, ASR/OCR files, cached JSON, filenames as semantic evidence, web search, or external knowledge. "
            "Do not call terminal, command, file-reading, web, or sub-agent tools; inspect only the already-attached MP4 through your native multimodal capability. "
            "If the slice cannot be ingested, frames cannot be inspected, audio cannot be heard, or the selected model is not gemini-3.7-flash-high, return a failure object with the exact reason and no semantic guesses."
        ),
        current_policy,
    ]
    assignments: dict[str, dict[str, Any]] = {}
    for path in (experiment / "logs").glob("L23_V*.assignments.json"):
        for item in json.loads(path.read_text(encoding="utf-8-sig")):
            assignments[item["scene_id"]] = item
    archive_dir = experiment / "prompts" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures = 0
    for log_path in sorted((experiment / "logs").glob("*.invocation.json.attempt*")):
        record = read_json(log_path)
        worker_id = record["worker_id"]
        assignment = assignments.get(worker_id)
        if not assignment:
            failures += 1
            results.append({"log": str(log_path), "status": "failure", "reason": "assignment not found"})
            continue
        matched: bytes | None = None
        for policy in old_policies:
            template = current_template.replace(current_policy, policy)
            rendered = template
            replacements = {
                "{{ABS_SLICE_PATH}}": str(Path(assignment["slice_path"]).resolve()),
                "{{ASSIGNMENT_JSON}}": json.dumps(assignment, indent=2, ensure_ascii=False),
                "{{VIDEO_ID}}": assignment["source_video_id"],
                "{{SCENE_ID}}": assignment["scene_id"],
            }
            for needle, value in replacements.items():
                rendered = rendered.replace(needle, value)
            for candidate in (rendered.encode("utf-8"), rendered.replace("\n", "\r\n").encode("utf-8")):
                if hashlib.sha256(candidate).hexdigest().upper() == record["prompt_sha256"]:
                    matched = candidate
                    break
            if matched is not None:
                break
        if matched is None:
            failures += 1
            results.append({"log": str(log_path), "status": "failure", "reason": "hash not reconstructed"})
            continue
        snapshot = archive_dir / f"{record['prompt_sha256']}.md"
        snapshot.write_bytes(matched)
        record["original_rendered_prompt_path"] = record["rendered_prompt_path"]
        record["rendered_prompt_path"] = str(snapshot.resolve())
        write_json(log_path, record)
        results.append({"log": str(log_path), "status": "success", "snapshot": str(snapshot), "hash": record["prompt_sha256"]})
    write_json(Path(args.report), {"failures": failures, "archived_invocations": results})
    print(json.dumps({"archived_invocations": len(results), "failures": failures}))
    if failures:
        raise PipelineError("could not reconstruct every archived prompt")


def final_check(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment).resolve()
    videos = ("L23_V014", "L23_V015", "L23_V023")
    required = [experiment / "manifest.json", experiment / "report.md", experiment / "l23_V014_V015_V023_semantic_index.jsonl"]
    for video in videos:
        required.extend([
            experiment / "raw" / f"{video}.jsonl",
            experiment / "normalized" / f"{video}.jsonl",
            experiment / "audits" / f"{video}.audit.json",
            experiment / "logs" / f"{video}.assignments.json",
        ])
    errors = [f"missing required file: {path}" for path in required if not path.exists()]
    line_counts: dict[str, int] = {}
    for path in [p for p in required if p.suffix == ".jsonl" and p.exists()]:
        count = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("not an object")
            except Exception as exc:
                errors.append(f"invalid JSONL {path}:{line_number}: {exc}")
            count += 1
        line_counts[str(path.relative_to(experiment))] = count
    expected_merged = sum(
        sum(1 for line in (experiment / "normalized" / f"{video}.jsonl").read_text(encoding="utf-8-sig").splitlines() if line)
        for video in videos
    )
    actual_merged = line_counts.get("l23_V014_V015_V023_semantic_index.jsonl", -1)
    if actual_merged != expected_merged:
        errors.append(f"merged line count {actual_merged} != normalized total {expected_merged}")
    for video in videos:
        audit_path = experiment / "audits" / f"{video}.audit.json"
        if audit_path.exists() and read_json(audit_path).get("audit_verdict") != "pass":
            errors.append(f"audit verdict is not pass: {video}")
        assignments_path = experiment / "logs" / f"{video}.assignments.json"
        if assignments_path.exists():
            for assignment in json.loads(assignments_path.read_text(encoding="utf-8-sig")):
                slice_path = Path(assignment["slice_path"])
                if not slice_path.exists():
                    errors.append(f"missing slice: {slice_path}")
                elif hashlib.sha256(slice_path.read_bytes()).hexdigest().upper() != assignment["slice_sha256"]:
                    errors.append(f"slice hash mismatch: {slice_path}")
    required_log_fields = {
        "requested_model", "actual_model_identifier", "worker_role", "source_mp4_path",
        "assigned_interval", "rendered_prompt_path", "prompt_sha256", "start_time", "end_time",
        "exit_code", "stdout_path", "stderr_path", "retry_count",
        "direct_video_evidence_observed", "audio_evidence_observed", "fallback_source_used",
    }
    invocation_logs = list((experiment / "logs").glob("*.invocation.json")) + list((experiment / "logs").glob("*.invocation.json.attempt*"))
    for path in invocation_logs:
        record = read_json(path)
        missing = required_log_fields - set(record)
        if missing:
            errors.append(f"invocation log missing fields {sorted(missing)}: {path}")
        prompt_path = Path(record["rendered_prompt_path"])
        if not prompt_path.exists():
            errors.append(f"invocation prompt missing: {prompt_path}")
        elif hashlib.sha256(prompt_path.read_bytes()).hexdigest().upper() != record["prompt_sha256"]:
            errors.append(f"invocation prompt hash mismatch: {path}")
        if record.get("actual_model_identifier") != MODEL_ID:
            errors.append(f"model substitution in {path}")
        if record.get("fallback_source_used") is not False:
            errors.append(f"fallback source used in {path}")
    report = {
        "status": "pass" if not errors else "fail",
        "expected_files_exist": not any(item.startswith("missing required") for item in errors),
        "jsonl_line_counts": line_counts,
        "merged_line_count_matches": actual_merged == expected_merged,
        "slice_count": sum(len(json.loads((experiment / "logs" / f"{video}.assignments.json").read_text(encoding="utf-8-sig"))) for video in videos),
        "invocation_log_count": len(invocation_logs),
        "model_identifier": MODEL_ID,
        "fallback_source_used": False,
        "errors": errors,
    }
    write_json(Path(args.output), report)
    print(json.dumps(report))
    if errors:
        raise PipelineError("final experiment check failed")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    slicing = sub.add_parser("make-slices")
    slicing.add_argument("--video-output", required=True)
    slicing.add_argument("--video-id", required=True)
    slicing.add_argument("--source", required=True)
    slicing.add_argument("--duration", required=True, type=float)
    slicing.add_argument("--out-dir", required=True)
    slicing.add_argument("--assignments", required=True)
    slicing.add_argument("--overlap", type=float, default=2.0)
    slicing.add_argument("--limit", type=int)
    slicing.add_argument("--overwrite", action="store_true")
    slicing.set_defaults(func=make_slices)

    parsing = sub.add_parser("parse-worker-outputs")
    parsing.add_argument("--input-dir", required=True)
    parsing.add_argument("--output-dir", required=True)
    parsing.add_argument("--video-id", required=True)
    parsing.add_argument("--pattern", default="*.stdout.txt")
    parsing.add_argument("--report", required=True)
    parsing.add_argument("--role", choices=["video_worker", "scene_worker", "targeted_audit"])
    parsing.set_defaults(func=parse_worker_outputs)

    adjudication = sub.add_parser("apply-adjudication")
    adjudication.add_argument("--input", required=True)
    adjudication.add_argument("--output", required=True)
    adjudication.add_argument("--record-id", required=True)
    adjudication.add_argument("--old-raw-name", required=True)
    adjudication.add_argument("--new-raw-name", required=True)
    adjudication.add_argument("--audit-worker-id", required=True)
    adjudication.add_argument("--resolution", required=True)
    adjudication.set_defaults(func=apply_adjudication)

    rendering = sub.add_parser("render-scene-prompts")
    rendering.add_argument("--template", required=True)
    rendering.add_argument("--assignments", required=True)
    rendering.add_argument("--out-dir", required=True)
    rendering.add_argument("--only", nargs="*")
    rendering.set_defaults(func=render_scene_prompts)

    video_rendering = sub.add_parser("render-video-prompts")
    video_rendering.add_argument("--template", required=True)
    video_rendering.add_argument("--manifest", required=True)
    video_rendering.add_argument("--workspace", required=True)
    video_rendering.add_argument("--out-dir", required=True)
    video_rendering.add_argument("--only", nargs="*")
    video_rendering.set_defaults(func=render_video_prompts)

    normal = sub.add_parser("normalize")
    normal.add_argument("--video-output", required=True)
    normal.add_argument("--video-id", required=True)
    normal.add_argument("--duration", required=True, type=float)
    normal.add_argument("--source-path", required=True)
    normal.add_argument("--assignments", required=True)
    normal.add_argument("--scene-output-dir", required=True)
    normal.add_argument("--raw-jsonl", required=True)
    normal.add_argument("--normalized-jsonl", required=True)
    normal.set_defaults(func=normalize)

    probe = sub.add_parser("validate-scene-probe")
    probe.add_argument("--assignments", required=True)
    probe.add_argument("--scene-output", required=True)
    probe.add_argument("--converted-output", required=True)
    probe.add_argument("--report", required=True)
    probe.set_defaults(func=validate_scene_probe)

    auditing = sub.add_parser("audit")
    auditing.add_argument("--normalized-jsonl", required=True)
    auditing.add_argument("--video-id", required=True)
    auditing.add_argument("--duration", required=True, type=float)
    auditing.add_argument("--audit-output", required=True)
    auditing.set_defaults(func=audit)

    merging = sub.add_parser("merge-final")
    merging.add_argument("--inputs", nargs="+", required=True)
    merging.add_argument("--output", required=True)
    merging.set_defaults(func=merge_final)

    spelling = sub.add_parser("normalize-canonical-spellings")
    spelling.add_argument("--inputs", nargs="+", required=True)
    spelling.add_argument("--report", required=True)
    spelling.set_defaults(func=normalize_canonical_spellings)

    snapshots = sub.add_parser("snapshot-archived-prompts")
    snapshots.add_argument("--experiment", required=True)
    snapshots.add_argument("--report", required=True)
    snapshots.set_defaults(func=snapshot_archived_prompts)

    checking = sub.add_parser("final-check")
    checking.add_argument("--experiment", required=True)
    checking.add_argument("--output", required=True)
    checking.set_defaults(func=final_check)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
