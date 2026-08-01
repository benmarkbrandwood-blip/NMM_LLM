from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.layered_core_selection import derive_perfect_core
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
EXPERIMENTS = ROOT / "docs" / "experiments"
DECISION = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.json"
)
DECISION_DOC = DECISION.with_suffix(".md")
BOOK = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
)
HUMAN = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
)
PERFECT_AUDIT = (
    EVIDENCE / "sanmill-layered-perfect-source-audit-2026-07-25.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_perfect_core_is_exactly_rederived() -> None:
    decision = _load(DECISION)
    derived = derive_perfect_core(
        perfect_audit=_load(PERFECT_AUDIT),
        book_selection=_load(BOOK)["selection"],
        human_selection=_load(HUMAN)["selection"],
    )

    assert decision["selection"] == derived
    assert derived["candidate_window_identity"] == canonical_sha256(
        derived["candidate_window"]
    )
    assert derived["membership_identity"] == canonical_sha256(
        derived["members"]
    )


def test_perfect_core_uses_first_21_fixed_routes_without_duplicates() -> None:
    selection = _load(DECISION)["selection"]
    members = selection["members"]

    assert [item["route_id"] for item in members] == [
        f"perfect-audit-route-{index:03d}" for index in range(21)
    ]
    assert [item["route_seed"] for item in members] == list(range(42, 63))
    assert selection["summary"] == {
        "member_count": 21,
        "last_selected_audit_route_index": 20,
        "skipped_before_quota": 0,
        "tied_best_step_count": 240,
        "single_best_step_count": 12,
    }
    assert len({tuple(item["action_tokens"]) for item in members}) == 21
    assert len({item["final"]["nmm_fen"] for item in members}) == 21
    assert len(
        {item["final"]["ring16_canonical_fen"] for item in members}
    ) == 21


def test_all_64_source_members_are_structurally_disjoint() -> None:
    members = (
        _load(BOOK)["selection"]["members"]
        + _load(HUMAN)["selection"]["members"]
        + _load(DECISION)["selection"]["members"]
    )

    assert len(members) == 64
    assert len({tuple(item["action_tokens"]) for item in members}) == 64
    assert len({item["final"]["nmm_fen"] for item in members}) == 64
    assert len(
        {item["final"]["ring16_canonical_fen"] for item in members}
    ) == 64


def test_perfect_core_preserves_strictsteps_and_claim_boundaries() -> None:
    decision = _load(DECISION)
    members = decision["selection"]["members"]

    assert all(item["theory_summary"]["selected_wdl"] == 0 for item in members)
    assert all(
        item["theory_summary"]["selected_category"] == "draw"
        for item in members
    )
    assert all(
        item["execution_record_status"] == "frozen_source_prefix_available"
        for item in members
    )
    assert decision["candidate_loaded"] is False
    assert decision["games_played"] == 0
    assert decision["fallback"] == "none"
    assert decision["decision"] == {
        "perfect_db_membership_frozen": True,
        "all_source_membership_frozen": True,
        "human_execution_records_frozen": False,
        "final_execution_corpus_frozen": False,
        "review_package_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_perfect_core_document_links_resolve() -> None:
    document = DECISION_DOC.read_text(encoding="utf-8")
    targets = [DECISION.name, BOOK.name, HUMAN.name]

    for target in targets:
        assert f"({target})" in document
        assert (DECISION_DOC.parent / target).is_file()
