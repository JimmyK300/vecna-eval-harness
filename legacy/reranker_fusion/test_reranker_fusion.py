import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from evaluate_reranker_fusion import historical_trake_score, join_historical  # noqa: E402


def test_historical_trake_requires_all_events_not_any_event():
    truth = {
        "accepted_video_id": "V1",
        "trake_event_truth": [
            {"event_index": 1, "proxy_window": {"start_frame": 100, "end_frame": 110}},
            {"event_index": 2, "proxy_window": {"start_frame": 200, "end_frame": 210}},
        ],
    }
    only_first = [{"rank": 1, "video_id": "V1", "frame_id": "V1#105", "time_line": []}]
    score = historical_trake_score(only_first, truth)
    assert score["event_first_ranks"] == [1, None]
    assert score["all_events_covered"] == {"R@1": False, "R@5": False, "R@10": False, "R@20": False}
    assert score["first_all_events_rank"] is None
    assert score["reciprocal_rank"] == 0.0


def test_historical_trake_all_events_success_uses_latest_required_event_rank():
    truth = {
        "accepted_video_id": "V1",
        "trake_event_truth": [
            {"event_index": 1, "proxy_window": {"start_frame": 100, "end_frame": 110}},
            {"event_index": 2, "proxy_window": {"start_frame": 200, "end_frame": 210}},
        ],
    }
    both = [
        {"rank": 1, "video_id": "V1", "frame_id": "V1#105", "time_line": []},
        {"rank": 5, "video_id": "V1", "frame_id": "V1#205", "time_line": []},
    ]
    score = historical_trake_score(both, truth)
    assert score["event_first_ranks"] == [1, 5]
    assert score["all_events_covered"]["R@1"] is False
    assert score["all_events_covered"]["R@5"] is True
    assert score["first_all_events_rank"] == 5
    assert score["reciprocal_rank"] == 0.2


def test_historical_join_keeps_operational_phase_when_raw_ids_collide():
    p0 = {"operational_phase": "P0", "raw_query_id": "p1-1", "accepted_video_id": "P0_VIDEO"}
    p1 = {"operational_phase": "P1", "raw_query_id": "p1-1", "accepted_video_id": "P1_VIDEO"}
    records = {(row["operational_phase"], row["raw_query_id"]): row for row in (p0, p1)}
    p0_row = {"query_id": "p0_q01", "canonical_source_key": "query-p0-1-kis"}
    p1_row = {"query_id": "p1_q01", "canonical_source_key": "query-p1-1-kis"}
    assert join_historical(p0_row, records)["accepted_video_id"] == "P0_VIDEO"
    assert join_historical(p1_row, records)["accepted_video_id"] == "P1_VIDEO"
