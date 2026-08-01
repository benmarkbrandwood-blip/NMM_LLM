from __future__ import annotations

import json
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
DECISION = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.json"
)
DECISION_DOC = DECISION.with_suffix(".md")
EXPERT_COVERAGE = (
    EXPERIMENTS
    / "sanmill-layered-expert-book-coverage-decision-2026-08-01.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_composition_decision_freezes_counts_only() -> None:
    decision = _load(DECISION)
    composition = decision["composition"]

    assert decision["schema_version"] == (
        "nmm.layered-opening-prefix-composition-decision.v1"
    )
    assert decision["status"] == "composition_frozen_membership_pending"
    assert composition["total_prefixes"] == 64
    assert composition["strata"] == [
        {"stratum": "book", "count": 22},
        {"stratum": "human_db", "count": 21},
        {"stratum": "perfect_db", "count": 21},
    ]
    assert decision["composition_identity"] == canonical_sha256(composition)

    assert decision["decision"] == {
        "composition_frozen": True,
        "book_subtype_allocation_frozen": False,
        "membership_frozen": False,
        "final_64_frozen": False,
        "review_package_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_composition_decision_preserves_source_and_claim_boundaries() -> None:
    decision = _load(DECISION)
    expert = _load(EXPERT_COVERAGE)

    assert decision["candidate_loaded"] is False
    assert decision["games_played"] == 0
    assert decision["fallback"] == "none"
    assert decision["composition"]["report_each_stratum_separately"] is True
    assert decision["composition"]["movement_flying_corpus_replaced"] is False
    assert decision["selection_gates"][
        "candidate_conditioned_selection_allowed"
    ] is False
    assert decision["inputs"]["expert_coverage_catalog_identity"] == (
        expert["catalog_identity"]
    )


def test_composition_document_links_resolve() -> None:
    document = DECISION_DOC.read_text(encoding="utf-8")
    targets = [
        DECISION.name,
        "sanmill-layered-expert-book-coverage-decision-2026-08-01.md",
    ]

    for target in targets:
        assert f"({target})" in document
        assert (DECISION_DOC.parent / target).is_file()
