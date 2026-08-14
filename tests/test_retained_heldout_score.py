from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation import retained_heldout_score as heldout


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-late-import-heldout-pool-v1.json"
)


def _spec() -> dict:
    return {
        "diagnostic_id": "heldout-test",
        "spec_identity": "s" * 64,
    }


def _record(
    *,
    start_index: int,
    candidate_id: str,
    candidate_color: str,
    score: float | None,
) -> dict:
    rules_terminal = score is not None
    return {
        "match_key": f"start-{start_index:03d}:{candidate_color}",
        "candidate_id": candidate_id,
        "candidate_color": candidate_color,
        "start_id": f"start-{start_index:03d}",
        "phase": "placement",
        "ongoing_after_post_start_logical_ply_108": False,
        "post_start_ply_108_snapshot": None,
        "post_start_logical_plies": (
            20 if rules_terminal else heldout.MAX_POST_START_LOGICAL_PLIES
        ),
        "total_logical_plies": (
            32 if rules_terminal else 12 + heldout.MAX_POST_START_LOGICAL_PLIES
        ),
        "termination_class": (
            "rules_terminal" if rules_terminal else "safety_cap_incomplete"
        ),
        "outcome_reason": (
            "drawFiftyMove" if rules_terminal else "safety_cap_incomplete"
        ),
        "candidate_score": score,
        "candidate_malom": {
            "candidate_turns": 10,
            "queryable_turns": 10,
            "unqueryable_turns": 0,
            "preserving_turns": 10,
            "one_step_downgrade_turns": 0,
            "two_step_downgrade_turns": 0,
        },
        "history_process": {
            "start": {
                "no_capture_count": 0,
                "repetition_current_count": 0,
                "repetition_history_length": 12,
            },
            "horizon": None,
            "final": {
                "no_capture_count": 100 if rules_terminal else 20,
                "repetition_current_count": 1,
                "repetition_history_length": 32,
            },
        },
    }


def _complete_records(
    *,
    v3_score: float = 0.5,
    v4_score: float = 0.5,
) -> list[dict]:
    records = []
    for start_index in range(heldout.EXPECTED_STARTS):
        for color in ("W", "B"):
            records.extend(
                [
                    _record(
                        start_index=start_index,
                        candidate_id=heldout.EXPECTED_CANDIDATES[0],
                        candidate_color=color,
                        score=v3_score,
                    ),
                    _record(
                        start_index=start_index,
                        candidate_id=heldout.EXPECTED_CANDIDATES[1],
                        candidate_color=color,
                        score=v4_score,
                    ),
                ]
            )
    return records


def test_frozen_pool_selects_exact_high_precision_prefix() -> None:
    payload = json.loads(POOL.read_text(encoding="utf-8"))
    records = heldout.load_corpus_records(payload)

    assert len(records) == heldout.EXPECTED_STARTS == 253
    assert records[0]["start_id"] == "late-import-heldout-001"
    assert records[-1]["start_id"] == "late-import-heldout-253"
    assert heldout.canonical_sha256(
        [record["record_identity"] for record in records]
    ) == (heldout.EXPECTED_PREFIX_RECORDS_IDENTITY)


def test_schedule_is_adjacent_by_candidate_inside_start_and_colour() -> None:
    payload = json.loads(POOL.read_text(encoding="utf-8"))
    schedule = heldout.build_schedule(heldout.load_corpus_records(payload))

    assert len(schedule) == heldout.EXPECTED_GAMES == 1012
    for unit_index in range(heldout.EXPECTED_MATCHED_COLOUR_UNITS):
        v3, v4 = schedule[unit_index * 2 : unit_index * 2 + 2]
        assert v3["match_key"] == v4["match_key"]
        assert v3["candidate_id"] == heldout.EXPECTED_CANDIDATES[0]
        assert v4["candidate_id"] == heldout.EXPECTED_CANDIDATES[1]


def test_primary_score_clusters_both_colours_inside_each_start() -> None:
    report = heldout.summarize_records(
        _spec(),
        _complete_records(v3_score=0.5, v4_score=1.0),
        "f" * 64,
    )
    primary = report["paired"]["primary_start_clustered_score_v4_minus_v3"]

    assert primary["support"] == 253
    assert primary["mean"] == 0.5
    assert primary["interval"] == [0.5, 0.5]
    assert primary["half_width"] == 0.0
    assert primary["decision"] == "v4_higher_fixed_heldout_score"


def test_interval_crossing_zero_is_inconclusive_not_equivalence() -> None:
    records = _complete_records()
    records[1]["candidate_score"] = 1.0
    records[5]["candidate_score"] = 0.0
    report = heldout.summarize_records(_spec(), records, "f" * 64)
    primary = report["paired"]["primary_start_clustered_score_v4_minus_v3"]

    assert primary["half_width"] < heldout.MAX_PRIMARY_HALF_WIDTH
    assert primary["interval"][0] < 0 < primary["interval"][1]
    assert primary["decision"] == "inconclusive"
    assert report["claim_boundary"]["equivalence_claim"] is False


def test_safety_cap_invalidates_directional_primary() -> None:
    records = _complete_records(v3_score=0.5, v4_score=1.0)
    records[-1] = _record(
        start_index=heldout.EXPECTED_STARTS - 1,
        candidate_id=heldout.EXPECTED_CANDIDATES[1],
        candidate_color="B",
        score=None,
    )
    report = heldout.summarize_records(_spec(), records, "f" * 64)
    primary = report["paired"]["primary_start_clustered_score_v4_minus_v3"]

    assert primary["support"] == 252
    assert primary["all_rules_terminal"] is False
    assert primary["decision"] == "inconclusive_incomplete_safety_cap"


def test_wrong_prefix_size_fails_closed() -> None:
    payload = json.loads(POOL.read_text(encoding="utf-8"))
    payload["nested_precision_prefixes"][2]["target_starts"] = 252
    with pytest.raises(Exception, match="pool_identity differs|profiles|prefix"):
        heldout.load_corpus_records(payload)
