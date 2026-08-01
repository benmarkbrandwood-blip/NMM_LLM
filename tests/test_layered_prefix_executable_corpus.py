from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.layered_executable_corpus import (
    LayeredExecutableCorpusError,
    build_layered_executable_corpus,
    verify_layered_executable_corpus,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
EVIDENCE = ROOT / "docs" / "evidence"
CORPUS = (
    EXPERIMENTS
    / "sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json"
)
CORPUS_DOC = CORPUS.with_suffix(".md")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> dict[str, dict]:
    return {
        "composition_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.json"
        ),
        "book_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json"
        ),
        "human_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json"
        ),
        "perfect_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.json"
        ),
        "source_core_decision": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json"
        ),
        "sanmill_book_audit": _load(
            EVIDENCE / "sanmill-layered-book-source-audit-2026-07-25.json"
        ),
        "expert_book_audit": _load(
            EVIDENCE
            / "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json"
        ),
        "human_audit": _load(
            EVIDENCE / "sanmill-layered-human-source-audit-2026-07-25.json"
        ),
        "perfect_audit": _load(
            EVIDENCE / "sanmill-layered-perfect-source-audit-2026-07-25.json"
        ),
        "human_execution": _load(
            EXPERIMENTS
            / "sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.json"
        ),
        "runtime_decision": _load(
            EXPERIMENTS
            / "sanmill-prefix12-human-replay-runtime-2026-08-01.json"
        ),
    }


def test_frozen_executable_corpus_verifies_from_tracked_inputs() -> None:
    summary = verify_layered_executable_corpus(_load(CORPUS), **_inputs())

    assert summary["record_count"] == 64
    assert summary["stratum_counts"] == {
        "book": 22,
        "human_db": 21,
        "perfect_db": 21,
    }
    assert summary["total_logical_ply_count"] == 768
    assert summary["compound_turn_count"] == 39
    assert summary["unique_prefix_identity_count"] == 64
    assert summary["unique_final_history_identity_count"] == 64
    assert sorted(
        record["record_count"]
        for record in summary["sanmill_runtime_records"]
    ) == [21, 43]


def test_generator_exactly_reproduces_frozen_executable_corpus() -> None:
    assert build_layered_executable_corpus(**_inputs()) == _load(CORPUS)


def test_executable_corpus_order_and_bindings_are_frozen() -> None:
    records = _load(CORPUS)["corpus"]["records"]
    assert [record["corpus_id"] for record in records] == [
        f"layered-prefix-v2-{index:03d}" for index in range(1, 65)
    ]
    assert [record["source_core_id"] for record in records] == [
        f"source-core-{index:03d}" for index in range(1, 65)
    ]
    assert [record["stratum"] for record in records] == (
        ["book"] * 22 + ["human_db"] * 21 + ["perfect_db"] * 21
    )


def test_executable_corpus_rejects_rehashed_binding_drift() -> None:
    payload = copy.deepcopy(_load(CORPUS))
    record = payload["corpus"]["records"][0]
    record["source_core_id"] = "source-core-999"
    record_body = dict(record)
    record_body.pop("record_identity")
    record["record_identity"] = canonical_sha256(record_body)
    payload["corpus"]["records_identity"] = canonical_sha256(
        payload["corpus"]["records"]
    )
    identity_body = {
        "input_identities": payload["input_identities"],
        "input_identities_identity": payload["input_identities_identity"],
        "corpus": payload["corpus"],
    }
    payload["executable_corpus_identity"] = canonical_sha256(identity_body)

    with pytest.raises(
        LayeredExecutableCorpusError,
        match="executable corpus drifted",
    ):
        verify_layered_executable_corpus(payload, **_inputs())


def test_executable_corpus_document_links_frozen_inputs() -> None:
    document = CORPUS_DOC.read_text(encoding="utf-8")
    links = (
        f"({CORPUS.name})",
        "(sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json)",
        "(sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.json)",
        "(../evidence/sanmill-layered-book-source-audit-2026-07-25.json)",
        (
            "(../evidence/"
            "sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json)"
        ),
        "(../evidence/sanmill-layered-perfect-source-audit-2026-07-25.json)",
    )
    for link in links:
        assert link in document
