from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.layered_core_selection import derive_human_core
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
EXPERIMENTS = ROOT / "docs" / "experiments"
DECISION = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
)
DECISION_DOC = DECISION.with_suffix(".md")
BOOK = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
)
HUMAN_AUDIT = (
    EVIDENCE / "sanmill-layered-human-source-audit-2026-07-25.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_record(source: dict) -> dict:
    return {
        "history_identity": source["source_history_id"],
        "logical_ply_count": source["logical_ply_count"],
        "logical_plies_by_side": source["logical_plies_by_side"],
        "action_tokens": source["action_tokens"],
        "logical_turns": source["logical_turns"],
        "distinct_game_count": source["distinct_game_count"],
        "occurrence_count": source["occurrence_count"],
        "results": source["results"],
        "side_roles": source["side_roles"],
        "final": source["final"],
    }


def test_frozen_human_core_is_exactly_rederived_from_portable_window() -> None:
    decision = _load(DECISION)
    selection = decision["selection"]
    audit = _load(HUMAN_AUDIT)
    book = _load(BOOK)
    header = {
        "schema_version": selection["ledger"]["schema_version"],
        "logical_ply_count": 12,
        "logical_plies_by_side": [6, 6],
        "ordering": selection["selection_rule"],
        "record_count": selection["ledger"]["history_count"],
    }
    records = [
        _ledger_record(item["source_record"])
        for item in selection["candidate_window"]
    ]

    derived = derive_human_core(
        human_audit=audit,
        book_selection=book["selection"],
        ledger_header=header,
        ledger_records=records,
    )

    assert derived == selection
    assert selection["candidate_window_identity"] == canonical_sha256(
        selection["candidate_window"]
    )
    assert selection["membership_identity"] == canonical_sha256(
        selection["members"]
    )


def test_human_core_uses_frequency_order_with_structural_deduplication() -> None:
    selection = _load(DECISION)["selection"]
    members = selection["members"]
    skipped = [
        item
        for item in selection["candidate_window"]
        if item["disposition"] == "skipped_duplicate"
    ]

    assert [item["ledger_rank"] for item in members] == [
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        13,
        16,
        17,
        18,
        19,
        20,
        21,
        23,
        24,
        25,
        27,
        28,
        30,
        31,
    ]
    assert [item["ledger_rank"] for item in skipped] == [
        1,
        2,
        3,
        9,
        12,
        14,
        15,
        22,
        26,
        29,
    ]
    assert all(item["collision_dimensions"] == ["ring16"] for item in skipped)
    assert selection["summary"] == {
        "member_count": 21,
        "last_selected_ledger_rank": 31,
        "minimum_distinct_game_count": 16,
        "skipped_before_quota": 10,
    }


def test_book_and_human_members_are_structurally_disjoint() -> None:
    book = _load(BOOK)["selection"]["members"]
    human = _load(DECISION)["selection"]["members"]
    combined = book + human

    assert len(combined) == 43
    assert len({tuple(item["action_tokens"]) for item in combined}) == 43
    assert len({item["final"]["nmm_fen"] for item in combined}) == 43
    assert len(
        {item["final"]["ring16_canonical_fen"] for item in combined}
    ) == 43


def test_human_core_is_bound_but_not_executable_authority() -> None:
    decision = _load(DECISION)
    audit = _load(HUMAN_AUDIT)
    selection = decision["selection"]

    assert selection["ledger"] == audit["raw_game_source"]["history_ledger"]
    assert decision["status"] == "human_membership_frozen_perfect_pending"
    assert decision["candidate_loaded"] is False
    assert decision["games_played"] == 0
    assert decision["fallback"] == "none"
    assert all(
        item["execution_record_status"] == "full_sanmill_replay_pending"
        for item in selection["members"]
    )
    assert decision["decision"] == {
        "human_db_membership_frozen": True,
        "human_execution_records_frozen": False,
        "perfect_db_membership_frozen": False,
        "final_64_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_human_core_document_links_resolve() -> None:
    document = DECISION_DOC.read_text(encoding="utf-8")
    targets = [DECISION.name, BOOK.name]

    for target in targets:
        assert f"({target})" in document
        assert (DECISION_DOC.parent / target).is_file()
