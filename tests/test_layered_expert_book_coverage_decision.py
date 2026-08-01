from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
)
DECISION = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-layered-expert-book-coverage-decision-2026-08-01.json"
)
DECISION_DOC = DECISION.with_suffix(".md")


def _load() -> tuple[dict, dict]:
    return (
        json.loads(AUDIT.read_text(encoding="utf-8")),
        json.loads(DECISION.read_text(encoding="utf-8")),
    )


def test_coverage_decision_is_bound_and_non_executable() -> None:
    audit, decision = _load()

    assert decision["source_audit_identity"] == audit["audit_identity"]
    assert decision["source_record_count"] == len(audit["records"]) == 36
    assert decision["status"] == (
        "expert_coverage_frozen_core_membership_not_frozen"
    )
    assert decision["candidate_loaded"] is False
    assert decision["games_played"] == 0
    assert decision["fallback"] == "none"
    assert decision["expert_scope"] == {
        "selection_unit": "unique_twelve_ply_final_placement_pattern",
        "different_route_same_placement_requires_separate_slot": False,
        "all_source_records_retained_as_provenance": True,
        "technical_arrangement_delegated": True,
        "evidence": (
            "../evidence/"
            "maintainer-book-opening-plays-semantic-review-2026-07-26.md"
        ),
    }


def test_catalog_is_exactly_one_representative_per_ring16_pattern() -> None:
    audit, decision = _load()
    records = {item["variation_id"]: item for item in audit["records"]}
    catalog = decision["coverage_catalog"]

    assert decision["catalog_identity"] == canonical_sha256(catalog)
    assert [item["coverage_id"] for item in catalog] == [
        f"expert-pattern-{index:03d}" for index in range(1, 34)
    ]
    assert len(catalog) == 33
    assert len({item["variation_id"] for item in catalog}) == 33
    assert len({item["ring16_canonical_fen"] for item in catalog}) == 33

    audit_patterns = {
        item["prefix_record"]["final"]["ring16_canonical_fen"]
        for item in audit["records"]
    }
    assert {item["ring16_canonical_fen"] for item in catalog} == (
        audit_patterns
    )

    for item in catalog:
        source = records[item["variation_id"]]
        assert item["source_row"] == source["source_row"]
        assert item["ring16_canonical_fen"] == source["prefix_record"][
            "final"
        ]["ring16_canonical_fen"]


def test_technical_pattern_policy_matches_the_frozen_audit_counts() -> None:
    audit, decision = _load()
    policy = decision["technical_policy"]

    assert policy["pattern_key"] == (
        "prefix_record.final.ring16_canonical_fen"
    )
    assert policy["symmetry_group"] == "D4"
    assert policy["raw_record_count"] == len(audit["records"]) == 36
    assert policy["unique_exact_history_count"] == audit["summary"][
        "unique_exact_history_count"
    ] == 35
    assert policy["unique_exact_final_fen_count"] == audit["summary"][
        "unique_exact_final_fen_count"
    ] == 34
    assert policy["unique_ring16_pattern_count"] == audit["summary"][
        "unique_ring16_final_orbit_count"
    ] == 33
    assert policy["representative_precedence"] == [
        "explicit_expert_primary",
        "exact_human_db_distinct_game_support_descending",
        "typed_source_evidence_before_visual_interpretation",
        "source_row_then_variation_id",
    ]


def test_catalog_and_alternates_partition_all_source_records() -> None:
    audit, decision = _load()
    source_ids = {item["variation_id"] for item in audit["records"]}
    representative_ids = {
        item["variation_id"] for item in decision["coverage_catalog"]
    }
    collapsed = decision["collapsed_pattern_groups"]

    assert len(collapsed) == 1
    group = collapsed[0]
    assert group["representative_variation_id"] == "expert-book-play-003"
    assert group["representative_source_row"] == 3
    alternate_ids = {
        item["variation_id"] for item in group["alternate_records"]
    }
    assert alternate_ids == {
        "expert-book-play-014",
        "expert-book-play-016",
        "expert-book-play-020",
    }
    assert representative_ids.isdisjoint(alternate_ids)
    assert representative_ids | alternate_ids == source_ids


def test_collapsed_p03_relationships_match_the_frozen_audit() -> None:
    audit, decision = _load()
    records = {item["variation_id"]: item for item in audit["records"]}
    identifiers = [
        "expert-book-play-003",
        "expert-book-play-014",
        "expert-book-play-016",
        "expert-book-play-020",
    ]
    selected = [records[item] for item in identifiers]

    assert len(
        {
            item["prefix_record"]["final"]["ring16_canonical_fen"]
            for item in selected
        }
    ) == 1
    assert records["expert-book-play-003"]["prefix_record"]["final"][
        "nmm_fen"
    ] == records["expert-book-play-014"]["prefix_record"]["final"][
        "nmm_fen"
    ]
    assert records["expert-book-play-016"]["prefix_record"]["final"][
        "nmm_fen"
    ] != records["expert-book-play-003"]["prefix_record"]["final"][
        "nmm_fen"
    ]
    assert records["expert-book-play-014"]["exact_history_sha256"] == (
        records["expert-book-play-020"]["exact_history_sha256"]
    )
    assert decision["coverage_catalog"][3]["variation_id"] == (
        "expert-book-play-003"
    )


def test_catalog_retains_every_review_family_without_claiming_core_membership() -> None:
    _, decision = _load()
    counts = Counter(item["review_id"] for item in decision["coverage_catalog"])

    assert counts == {
        "P01": 2,
        "P02": 1,
        "P03": 13,
        "P04": 2,
        "P05": 2,
        "P06": 1,
        "P07": 1,
        "P08": 2,
        "P09": 2,
        "P10": 1,
        "P11": 1,
        "P12": 1,
        "P13-A": 2,
        "P14": 1,
        "P13-B": 1,
    }
    architecture = decision["suite_architecture"]
    assert architecture["expert_coverage_catalog"] == {
        "prefix_membership_frozen": True,
        "prefix_count": 33,
        "purpose": (
            "Preserve one representative for every audited D4-unique "
            "expert placement pattern."
        ),
    }
    assert architecture["expert_book_diagnostic_suite"][
        "prefix_membership_frozen"
    ] is True
    assert architecture["expert_book_diagnostic_suite"][
        "execution_contract_frozen"
    ] is False
    assert architecture["balanced_core_64"]["composition_proposal"] == {
        "book": 22,
        "human_db": 21,
        "perfect_db": 21,
    }
    assert architecture["balanced_core_64"]["composition_frozen"] is False
    assert architecture["balanced_core_64"]["membership_frozen"] is False


def test_decision_flags_do_not_authorize_core_evaluation_or_training() -> None:
    _, decision = _load()

    assert decision["decision"] == {
        "expert_coverage_membership_frozen": True,
        "expert_diagnostic_execution_contract_frozen": False,
        "balanced_core_composition_frozen": False,
        "balanced_core_membership_frozen": False,
        "final_64_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_decision_document_links_resolve() -> None:
    document = DECISION_DOC.read_text(encoding="utf-8")
    targets = [
        DECISION.name,
        (
            "../evidence/"
            "maintainer-book-opening-plays-semantic-review-2026-07-26.md"
        ),
        (
            "assets/sanmill-layered-expert-book-parent-review-"
            "reviewed-source-2026-07-26/parent-overview.png"
        ),
        (
            "assets/sanmill-layered-expert-book-parent-review-"
            "reviewed-source-2026-07-26/child-overviews/P03.png"
        ),
    ]

    for target in targets:
        assert f"({target})" in document
        assert (DECISION_DOC.parent / target).is_file()
