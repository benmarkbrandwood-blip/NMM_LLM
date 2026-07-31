from __future__ import annotations

import json
from pathlib import Path

from tools.render_layered_expert_book_review import (
    build_review_model,
    verify_review_assets,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-layered-expert-book-source-audit-2026-07-26.json"
)
REVIEWED_AUDIT = (
    ROOT
    / "docs"
    / "evidence"
    / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
)
ASSETS = (
    ROOT
    / "docs"
    / "experiments"
    / "assets"
    / "sanmill-layered-expert-book-parent-review-2026-07-26"
)
REVIEWED_ASSETS = (
    ROOT
    / "docs"
    / "experiments"
    / "assets"
    / "sanmill-layered-expert-book-parent-review-reviewed-source-2026-07-26"
)


def test_review_model_preserves_parent_and_child_boundaries() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    model = build_review_model(audit)

    assert model["parent_group_count"] == 14
    assert len(model["parent_variants"]) == 15
    assert [item["review_id"] for item in model["child_comparisons"]] == [
        "P01",
        "P03",
        "P04",
        "P05",
        "P08",
        "P09",
        "P13-A",
    ]
    assert sum(
        len(item["records"]) for item in model["child_comparisons"]
    ) == 28


def test_p13_keeps_two_exact_parents_separate() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    model = build_review_model(audit)
    variants = {
        item["review_id"]: item for item in model["parent_variants"]
    }
    comparisons = {
        item["review_id"]: item for item in model["child_comparisons"]
    }

    assert variants["P13-A"]["source_rows"] == [32, 33]
    assert variants["P13-B"]["source_rows"] == [35]
    assert [item["source_row"] for item in comparisons["P13-A"]["records"]] == [
        32,
        33,
    ]
    assert "P13-B" not in comparisons


def test_frozen_review_assets_match_source_and_renderer() -> None:
    assert verify_review_assets(AUDIT, ASSETS) == {
        "parent_groups": 14,
        "parent_variants": 15,
        "child_comparisons": 7,
        "child_panels": 28,
        "assets": 51,
    }


def test_reviewed_model_keeps_row_18_and_row_19_distinct() -> None:
    audit = json.loads(REVIEWED_AUDIT.read_text(encoding="utf-8"))

    model = build_review_model(audit)
    comparisons = {
        item["review_id"]: item for item in model["child_comparisons"]
    }
    p03 = {
        item["source_row"]: item for item in comparisons["P03"]["records"]
    }

    assert p03[18]["continuation"] == "c4 d7 e3 d1"
    assert p03[19]["continuation"] == "c4 d5 e3 d1"
    assert p03[18]["exact_history_sha256"] != (
        p03[19]["exact_history_sha256"]
    )


def test_p03_primary_and_sixth_child_are_endpoint_transpositions() -> None:
    audit = json.loads(REVIEWED_AUDIT.read_text(encoding="utf-8"))
    records = {
        (item["source_row"], item["variation_id"]): item
        for item in audit["records"]
    }

    primary = records[(3, "expert-book-play-003")]
    sixth = records[(14, "expert-book-play-014")]
    duplicate = records[(20, "expert-book-play-020")]

    assert primary["resolved_logical_turns"][8:] == [
        ["b2"],
        ["c5"],
        ["c4"],
        ["e5"],
    ]
    assert sixth["resolved_logical_turns"][8:] == [
        ["c4"],
        ["e5"],
        ["b2"],
        ["c5"],
    ]
    assert primary["exact_history_sha256"] != sixth["exact_history_sha256"]
    assert sixth["exact_history_sha256"] == duplicate["exact_history_sha256"]
    assert primary["prefix_record"]["final"]["nmm_fen"] == (
        sixth["prefix_record"]["final"]["nmm_fen"]
    )
    assert primary["prefix_record"]["final"]["ring16_canonical_fen"] == (
        sixth["prefix_record"]["final"]["ring16_canonical_fen"]
    )


def test_frozen_reviewed_assets_match_source_and_renderer() -> None:
    manifest = json.loads(
        (REVIEWED_ASSETS / "manifest.json").read_text(encoding="utf-8")
    )

    assert verify_review_assets(REVIEWED_AUDIT, REVIEWED_ASSETS) == {
        "parent_groups": 14,
        "parent_variants": 15,
        "child_comparisons": 7,
        "child_panels": 28,
        "assets": 51,
    }
    assert manifest["manifest_identity"] == (
        "1349107cc616b44a7017af2db1734f04a267c286be5155f449fed93c552bf568"
    )
    assert manifest["rendered_positions"] == {
        "count": 43,
        "ordered_fens_sha256": (
            "329ba864219a6e0d1898b87a2d84586e8bd3a480dbd320602a487b42fdee76c3"
        ),
    }
