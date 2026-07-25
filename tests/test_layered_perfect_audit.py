from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from learned_ai.evaluation.layered_perfect_audit import (
    LayeredPerfectAuditError,
    _uniform_index,
    _validate_candidate_pool,
    build_layered_perfect_audit,
    load_source_overlap_index,
)
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    PerfectCandidateData,
    SanmillDataQuerySession,
)
from learned_ai.evaluation.sanmill_uci import inspect_sanmill_installation
from learned_ai.training.run_contract import canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"


def _candidate(
    stable_index: int,
    action: str,
    *,
    category: str = "draw",
    wdl: int = 0,
    steps: int = 1,
) -> DataQueryCandidate:
    return DataQueryCandidate(
        logical_move_id="perfect:" + f"{stable_index + 1:064x}",
        source_group_id=None,
        stable_index=stable_index,
        source_rank=None,
        raw_notation=None,
        mapped_notation=action,
        full_turn_actions=(action,),
        remaining_actions=(action,),
        contains_removal=False,
        removal_action=None,
        logical_ply_delta=1,
        turn_prefix_complete=True,
        perfect=PerfectCandidateData(
            category=category,
            wdl=wdl,
            steps=steps,
            mode="strict_steps",
        ),
        human=None,
    )


def test_strictsteps_draw_candidates_may_have_different_steps() -> None:
    records, tie = _validate_candidate_pool(
        (
            _candidate(0, "a1", steps=1),
            _candidate(1, "a4", steps=9),
        )
    )

    assert len(records) == 2
    assert tie == {
        "category": "draw",
        "wdl": 0,
        "step_policy": "draw_steps_not_ranked",
        "step_values": [1, 9],
        "candidate_count": 2,
        "multiple_tied_best": True,
    }


def test_strictsteps_decisive_candidates_must_share_optimal_steps() -> None:
    with pytest.raises(
        LayeredPerfectAuditError,
        match="non-tied outcomes",
    ):
        _validate_candidate_pool(
            (
                _candidate(0, "a1", category="win", wdl=1, steps=3),
                _candidate(1, "a4", category="win", wdl=1, steps=5),
            )
        )


def test_candidate_pool_fails_closed_on_order_or_completion_drift() -> None:
    with pytest.raises(LayeredPerfectAuditError, match="lexicographic"):
        _validate_candidate_pool(
            (
                _candidate(0, "a4"),
                _candidate(1, "a1"),
            )
        )

    incomplete = replace(
        _candidate(0, "a1"),
        turn_prefix_complete=False,
    )
    with pytest.raises(LayeredPerfectAuditError, match="complete"):
        _validate_candidate_pool((incomplete,))


def test_uniform_selection_is_stable_and_bound_to_the_candidate_pool() -> None:
    first = _uniform_index(
        17,
        route_id="perfect-audit-route-007",
        seed=49,
        logical_ply=5,
        candidate_pool_identity="a" * 64,
    )
    repeated = _uniform_index(
        17,
        route_id="perfect-audit-route-007",
        seed=49,
        logical_ply=5,
        candidate_pool_identity="a" * 64,
    )
    changed = _uniform_index(
        17,
        route_id="perfect-audit-route-007",
        seed=49,
        logical_ply=5,
        candidate_pool_identity="b" * 64,
    )

    assert first == repeated
    assert first[1]["candidate_pool_identity"] == "a" * 64
    assert first[1]["draw_sha256"] != changed[1]["draw_sha256"]


@pytest.mark.skipif(
    not _LOCAL_PATHS.is_file(),
    reason="requires the ignored Sanmill and Perfect DB path registry",
)
def test_local_perfect_routes_are_byte_stable_across_fresh_processes() -> None:
    config = json.loads(_LOCAL_PATHS.read_text(encoding="utf-8"))
    installation = inspect_sanmill_installation(_LOCAL_PATHS)
    database = Path(config["malom_db_path"])
    ledger = Path(config["human_db_prefix12_history_ledger_path"])
    if not ledger.is_absolute():
        ledger = _ROOT / ledger
    overlap = load_source_overlap_index(
        book_audit_path=(
            _ROOT
            / "docs/evidence/"
            "sanmill-layered-book-source-audit-2026-07-25.json"
        ),
        human_audit_path=(
            _ROOT
            / "docs/evidence/"
            "sanmill-layered-human-source-audit-2026-07-25.json"
        ),
        human_ledger_path=ledger,
    )

    encoded = []
    for _ in range(2):
        with SanmillDataQuerySession(installation, timeout=300.0) as session:
            audit = build_layered_perfect_audit(
                session,
                installation,
                database_path=database,
                generator_commit="0" * 40,
                overlap=overlap,
                route_count=2,
                base_seed=42,
                fresh_processes=2,
            )
            encoded.append(canonical_json_bytes(audit))

    assert encoded[0] == encoded[1]
