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
ASSETS = (
    ROOT
    / "docs"
    / "experiments"
    / "assets"
    / "sanmill-layered-expert-book-parent-review-2026-07-26"
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
