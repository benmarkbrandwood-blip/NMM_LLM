from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation.layered_core_selection import (
    LayeredCoreSelectionError,
    derive_book_core,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
EXPERIMENTS = ROOT / "docs" / "experiments"
DECISION = (
    EXPERIMENTS / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
)
DECISION_DOC = DECISION.with_suffix(".md")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive() -> dict:
    return derive_book_core(
        sanmill_book_audit=_load(
            EVIDENCE / "sanmill-layered-book-source-audit-2026-07-25.json"
        ),
        expert_book_audit=_load(
            EVIDENCE
            / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
        ),
        expert_coverage=_load(
            EXPERIMENTS
            / "sanmill-layered-expert-book-coverage-decision-2026-08-01.json"
        ),
        expert_shortlist=_load(
            EXPERIMENTS
            / "sanmill-layered-expert-book-shortlist-proposal-2026-07-31.json"
        ),
    )


def test_frozen_book_core_is_exactly_rederived() -> None:
    decision = _load(DECISION)
    derived = _derive()

    assert decision["selection"] == derived
    assert derived["membership_identity"] == canonical_sha256(
        derived["members"]
    )
    assert derived["allocation"] == {
        "book_total": 22,
        "maintainer_expert_curated_play": 15,
        "named_book_variation": 7,
    }


def test_book_core_covers_parent_breadth_and_declared_families() -> None:
    members = _derive()["members"]
    expert = [
        item
        for item in members
        if item["source_subtype"] == "maintainer_expert_curated_play"
    ]
    named = [
        item
        for item in members
        if item["source_subtype"] == "named_book_variation"
    ]

    assert [item["family"] for item in expert[:14]] == [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
        "P07",
        "P08",
        "P09",
        "P10",
        "P11",
        "P12",
        "P13",
        "P14",
    ]
    assert expert[14]["source_member_id"] == "expert-book-play-008"
    assert {item["family"] for item in named} == {
        "Early Game",
        "Man-to-Man Marking",
        "Black Diamond",
        "Mill Rush",
        "Battle Lines",
        "Z Mill",
        "novel",
    }


def test_book_core_has_22_distinct_histories_fens_and_orbits() -> None:
    members = _derive()["members"]

    assert len(members) == 22
    assert len({item["source_history_id"] for item in members}) == 22
    assert len({item["final"]["nmm_fen"] for item in members}) == 22
    assert len(
        {item["final"]["ring16_canonical_fen"] for item in members}
    ) == 22
    assert all(item["logical_ply_count"] == 12 for item in members)
    assert all(item["logical_plies_by_side"] == [6, 6] for item in members)


def test_book_core_decision_does_not_authorize_execution() -> None:
    decision = _load(DECISION)

    assert decision["status"] == "book_membership_frozen_other_strata_pending"
    assert decision["candidate_loaded"] is False
    assert decision["games_played"] == 0
    assert decision["fallback"] == "none"
    assert decision["decision"] == {
        "book_subtype_allocation_frozen": True,
        "book_membership_frozen": True,
        "human_db_membership_frozen": False,
        "perfect_db_membership_frozen": False,
        "final_64_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_book_core_fails_closed_when_a_parent_primary_drifts() -> None:
    shortlist = _load(
        EXPERIMENTS
        / "sanmill-layered-expert-book-shortlist-proposal-2026-07-31.json"
    )
    shortlist["breadth_first_parent_primaries"][0]["variation_id"] = (
        "missing"
    )

    with pytest.raises(LayeredCoreSelectionError):
        derive_book_core(
            sanmill_book_audit=_load(
                EVIDENCE
                / "sanmill-layered-book-source-audit-2026-07-25.json"
            ),
            expert_book_audit=_load(
                EVIDENCE
                / "sanmill-layered-expert-book-reviewed-source-audit-"
                "2026-07-26.json"
            ),
            expert_coverage=_load(
                EXPERIMENTS
                / "sanmill-layered-expert-book-coverage-decision-"
                "2026-08-01.json"
            ),
            expert_shortlist=shortlist,
        )


def test_book_core_document_links_resolve() -> None:
    document = DECISION_DOC.read_text(encoding="utf-8")
    targets = [
        DECISION.name,
        "sanmill-layered-expert-book-coverage-decision-2026-08-01.md",
    ]

    for target in targets:
        assert f"({target})" in document
        assert (DECISION_DOC.parent / target).is_file()
