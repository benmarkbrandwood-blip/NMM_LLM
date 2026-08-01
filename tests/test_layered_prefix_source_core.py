from __future__ import annotations

import json
from pathlib import Path

from learned_ai.evaluation.layered_core_selection import (
    build_layered_source_core,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
SOURCE_CORE = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
)
SOURCE_CORE_DOC = SOURCE_CORE.with_suffix(".md")
COMPOSITION = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.json"
)
BOOK = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
)
HUMAN = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
)
PERFECT = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_source_core_is_exactly_rederived_from_three_frozen_strata() -> None:
    decision = _load(SOURCE_CORE)
    derived = build_layered_source_core(
        composition_decision=_load(COMPOSITION),
        book_decision=_load(BOOK),
        human_decision=_load(HUMAN),
        perfect_decision=_load(PERFECT),
    )

    assert decision["source_core"] == derived
    assert derived["source_inputs_identity"] == canonical_sha256(
        derived["source_inputs"]
    )
    assert derived["source_membership_identity"] == canonical_sha256(
        derived["records"]
    )


def test_source_core_has_64_unique_structures_and_accepted_counts() -> None:
    core = _load(SOURCE_CORE)["source_core"]
    records = core["records"]

    assert core["composition"] == {
        "total": 64,
        "book": 22,
        "human_db": 21,
        "perfect_db": 21,
    }
    assert [item["source_core_id"] for item in records] == [
        f"source-core-{index:03d}" for index in range(1, 65)
    ]
    assert len({tuple(item["action_tokens"]) for item in records}) == 64
    assert len({item["final"]["nmm_fen"] for item in records}) == 64
    assert len(
        {item["final"]["ring16_canonical_fen"] for item in records}
    ) == 64
    assert core["summary"] == {
        "record_count": 64,
        "unique_exact_history_count": 64,
        "unique_final_fen_count": 64,
        "unique_ring16_count": 64,
        "side_to_move": "white",
        "logical_ply_count": 12,
        "logical_plies_by_side": [6, 6],
        "execution_record_status_counts": {
            "frozen_source_prefix_available": 43,
            "full_sanmill_replay_pending": 21,
        },
    }


def test_source_core_freezes_membership_not_execution() -> None:
    decision = _load(SOURCE_CORE)

    assert decision["status"] == (
        "source_membership_frozen_execution_replay_pending"
    )
    assert decision["candidate_loaded"] is False
    assert decision["games_played"] == 0
    assert decision["fallback"] == "none"
    assert decision["decision"] == {
        "source_membership_manifest_frozen": True,
        "human_execution_records_frozen": False,
        "final_execution_corpus_frozen": False,
        "review_package_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }


def test_source_core_document_links_resolve() -> None:
    document = SOURCE_CORE_DOC.read_text(encoding="utf-8")
    targets = [
        SOURCE_CORE.name,
        COMPOSITION.name,
        BOOK.name,
        HUMAN.name,
        PERFECT.name,
    ]

    for target in targets:
        assert f"({target})" in document
        assert (SOURCE_CORE_DOC.parent / target).is_file()
