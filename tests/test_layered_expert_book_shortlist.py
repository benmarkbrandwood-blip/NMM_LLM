from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
)
PROPOSAL = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-layered-expert-book-shortlist-proposal-2026-07-31.json"
)
PROPOSAL_DOC = PROPOSAL.with_suffix(".md")


def _load() -> tuple[dict, dict]:
    return (
        json.loads(AUDIT.read_text(encoding="utf-8")),
        json.loads(PROPOSAL.read_text(encoding="utf-8")),
    )


def test_shortlist_is_bound_and_explicitly_non_executable() -> None:
    audit, proposal = _load()

    assert proposal["source_audit_identity"] == audit["audit_identity"]
    assert proposal["source_record_count"] == len(audit["records"]) == 36
    assert proposal["candidate_loaded"] is False
    assert proposal["games_played"] == 0
    assert proposal["fallback"] == "none"
    assert proposal["status"] == "proposal_for_expert_correction_not_frozen"
    assert proposal["decision"] == {
        "book_subtype_allocation_frozen": False,
        "book_membership_frozen": False,
        "final_64_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_breadth_layer_has_one_unique_endpoint_per_parent_group() -> None:
    audit, proposal = _load()
    records = {item["variation_id"]: item for item in audit["records"]}
    primaries = proposal["breadth_first_parent_primaries"]

    assert [item["review_id"] for item in primaries] == [
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
    selected = [records[item["variation_id"]] for item in primaries]
    assert all(
        source["source_row"] == proposed["source_row"]
        for source, proposed in zip(selected, primaries, strict=True)
    )
    assert len({item["exact_history_sha256"] for item in selected}) == 14
    assert len(
        {item["prefix_record"]["final"]["nmm_fen"] for item in selected}
    ) == 14
    assert len(
        {
            item["prefix_record"]["final"]["ring16_canonical_fen"]
            for item in selected
        }
    ) == 14


def test_p03_proposal_covers_every_child_once() -> None:
    audit, proposal = _load()
    records = {item["variation_id"]: item for item in audit["records"]}
    members = [
        member
        for family in proposal["p03_extended_family_proposals"]
        for member in family["members"]
    ]

    assert sorted(member["child"] for member in members) == list(range(1, 17))
    assert len({member["child"] for member in members}) == 16
    assert {
        member["source_row"] for member in members
    } == {3, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28}
    for member in members:
        source = records[member["variation_id"]]
        assert source["source_row"] == member["source_row"]

    closed_z = proposal["p03_extended_family_proposals"][0]
    assert closed_z["primary"]["variation_id"] == "expert-book-play-003"
    dispositions = {
        item["variation_id"]: item["disposition"]
        for item in closed_z["members"]
    }
    assert dispositions == {
        "expert-book-play-003": "must_keep_primary",
        "expert-book-play-014": "same_plan_endpoint_transposition",
        "expert-book-play-020": "exact_history_duplicate",
    }

    primary = records["expert-book-play-003"]
    transposition = records["expert-book-play-014"]
    duplicate = records["expert-book-play-020"]
    assert primary["exact_history_sha256"] != transposition[
        "exact_history_sha256"
    ]
    assert transposition["exact_history_sha256"] == duplicate[
        "exact_history_sha256"
    ]
    assert primary["prefix_record"]["final"]["nmm_fen"] == transposition[
        "prefix_record"
    ]["final"]["nmm_fen"]


def test_every_shortlist_reference_resolves_to_the_frozen_audit() -> None:
    audit, proposal = _load()
    records = {item["variation_id"]: item for item in audit["records"]}

    referenced = {
        item["variation_id"]
        for item in proposal["breadth_first_parent_primaries"]
    }
    p03_primary_ids = []
    for family in proposal["p03_extended_family_proposals"]:
        primary = family["primary"]
        assert records[primary["variation_id"]]["source_row"] == primary[
            "source_row"
        ]
        member_ids = {item["variation_id"] for item in family["members"]}
        assert primary["variation_id"] in member_ids
        p03_primary_ids.append(primary["variation_id"])
        referenced.update(member_ids)

    for group in proposal["other_multi_child_proposals"]:
        referenced.add(group["primary_variation_id"])
        referenced.update(group["additional_variation_ids"])

    referenced.update(proposal["recommended_extra_order"])
    assert referenced <= records.keys()
    assert len(proposal["recommended_extra_order"]) == 5
    assert set(proposal["recommended_extra_order"]) == set(
        p03_primary_ids[1:]
    )


def test_unconfirmed_primary_tiebreaks_match_frozen_overlap_evidence() -> None:
    audit, proposal = _load()
    records = {item["variation_id"]: item for item in audit["records"]}
    exact_support = {
        reference["variation_id"]: match["distinct_game_count"]
        for match in audit["overlap"]["with_human_db"][
            "exact_history_support"
        ]["matches"]
        for reference in match["source_references"]
    }

    assert exact_support["expert-book-play-005"] == 4
    assert "expert-book-play-004" not in exact_support
    assert exact_support["expert-book-play-007"] == 2
    assert "expert-book-play-006" not in exact_support
    assert exact_support["expert-book-play-008"] == 9

    assert records["expert-book-play-024"]["overlap"]["human_db"][
        "ring16_orbit"
    ] is True
    assert records["expert-book-play-025"]["overlap"]["human_db"][
        "ring16_orbit"
    ] is False
    assert records["expert-book-play-026"]["overlap"]["sanmill_book"][
        "ring16_orbit"
    ] is True
    assert records["expert-book-play-027"]["overlap"]["sanmill_book"][
        "ring16_orbit"
    ] is False

    unconfirmed = {
        item["review_id"]: item
        for item in proposal["other_multi_child_proposals"]
        if not item["expert_confirmed_primary"]
    }
    assert unconfirmed["P04"]["primary_variation_id"] == (
        "expert-book-play-005"
    )
    assert unconfirmed["P05"]["primary_variation_id"] == (
        "expert-book-play-007"
    )
    assert unconfirmed["P08"]["primary_variation_id"] == (
        "expert-book-play-025"
    )
    assert unconfirmed["P09"]["primary_variation_id"] == (
        "expert-book-play-027"
    )


def test_capacity_analysis_does_not_freeze_the_book_allocation() -> None:
    _, proposal = _load()
    capacity = proposal["capacity_analysis"]

    assert capacity["expert_parent_breadth_candidates"] == 14
    assert capacity["sanmill_declared_family_minimum"] == 7
    assert capacity["base_slots_consumed"] == 21
    assert capacity["book_slots_proposed"] == 22
    assert capacity["slots_remaining_before_cross_source_deduplication"] == 1
    assert capacity["additional_p03_family_primaries_proposed"] == 5
    assert proposal["decision"]["book_subtype_allocation_frozen"] is False


def test_shortlist_review_document_embeds_resolvable_review_assets() -> None:
    document = PROPOSAL_DOC.read_text(encoding="utf-8")
    relative_targets = [
        PROPOSAL.name,
        (
            "assets/sanmill-layered-expert-book-parent-review-"
            "reviewed-source-2026-07-26/parent-overview.png"
        ),
        *[
            (
                "assets/sanmill-layered-expert-book-parent-review-"
                "reviewed-source-2026-07-26/child-overviews/"
                f"{review_id}.png"
            )
            for review_id in ["P03", "P04", "P05", "P08", "P09", "P13-A"]
        ],
    ]

    for target in relative_targets:
        assert f"({target})" in document
        assert (PROPOSAL_DOC.parent / target).is_file()
