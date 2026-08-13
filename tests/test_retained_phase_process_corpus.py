"""Tests for the candidate-blind retained phase-process corpus."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.retained_phase_process_corpus import (
    EXPECTED_STRICT_EXCLUSIONS,
    RetainedPhaseProcessCorpusError,
    select_candidate_blind_entries,
    validate_retained_phase_process_corpus,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
PRIOR = ROOT / (
    "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
CORPUS = ROOT / (
    "docs/experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh(payload: dict) -> None:
    for record in payload["records"]:
        body = {
            key: value for key, value in record.items() if key != "record_identity"
        }
        record["record_identity"] = canonical_sha256(body)
    payload["records_identity"] = canonical_sha256(payload["records"])
    body = {key: value for key, value in payload.items() if key != "corpus_identity"}
    payload["corpus_identity"] = canonical_sha256(body)


def test_selection_uses_all_unused_identity_histories() -> None:
    selected = select_candidate_blind_entries(_load(PHASE), _load(PRIOR))
    prior = {record["source_entry_index"] for record in _load(PRIOR)["records"]}

    assert len(selected) == 42
    assert not ({entry["index"] for entry in selected} & prior)
    assert all(entry["sources"][0]["color_transform"] == "identity" for entry in selected)


def test_committed_corpus_is_replayable_exposure_free_and_source_bound() -> None:
    payload = _load(CORPUS)
    result = validate_retained_phase_process_corpus(payload)

    assert result["record_count"] == 39
    assert result["phase_counts"] == {
        "flying": 7,
        "movement": 14,
        "placement": 18,
    }
    assert payload["selection_contract"][
        "strict_replay_excluded_source_entry_indices"
    ] == list(EXPECTED_STRICT_EXCLUSIONS)
    assert payload["exposure_audit"]["summary"] == {
        "human_db_d4_exposed_count": 0,
        "record_count": 39,
        "specialist_db_d4_exposed_count": {
            "retained-v3-refresh50": 0,
            "retained-v4-no-refresh": 0,
        },
        "strict_independence_count": 39,
    }
    assert payload["claim_boundaries"]["games_played"] == 0
    assert payload["claim_boundaries"]["candidate_loaded"] is False


def test_history_tampering_fails_even_with_refreshed_record_identities() -> None:
    payload = copy.deepcopy(_load(CORPUS))
    payload["records"][0]["logical_turns"][0] = ["a1"]
    payload["records"][0]["action_history"][0] = "a1"
    _refresh(payload)

    with pytest.raises(RetainedPhaseProcessCorpusError, match="legal moves"):
        validate_retained_phase_process_corpus(payload)


def test_excluded_strict_terminal_history_cannot_be_reintroduced() -> None:
    payload = copy.deepcopy(_load(CORPUS))
    payload["records"][0]["source_entry_index"] = EXPECTED_STRICT_EXCLUSIONS[0]
    _refresh(payload)

    with pytest.raises(RetainedPhaseProcessCorpusError, match="excluded source"):
        validate_retained_phase_process_corpus(payload)
