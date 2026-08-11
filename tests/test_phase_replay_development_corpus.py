"""Tests for the candidate-blind replayable development corpus."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.phase_replay_development_corpus import (
    CORPUS_ID,
    PhaseReplayCorpusError,
    group_action_tokens,
    replay_record_into_sanmill_game,
    select_replayable_phase_entries,
    validate_phase_replay_development_corpus,
    validate_phase_replay_sanmill_audit,
)
from learned_ai.training.run_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
PHASE_CORPUS = (
    ROOT / "docs/experiments/dev-v4-phase-covered-corpus-v1.json"
)
REPLAY_CORPUS = (
    ROOT / "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
SANMILL_AUDIT = ROOT / (
    "docs/evidence/"
    "phase-replay-development-corpus-sanmill-audit-2026-08-11.json"
)


def _phase_payload() -> dict:
    return json.loads(PHASE_CORPUS.read_text(encoding="utf-8"))


def _replay_payload() -> dict:
    return json.loads(REPLAY_CORPUS.read_text(encoding="utf-8"))


def _refresh_identities(payload: dict) -> None:
    for record in payload["records"]:
        body = {
            key: value for key, value in record.items() if key != "record_identity"
        }
        record["record_identity"] = canonical_sha256(body)
    body = {key: value for key, value in payload.items() if key != "corpus_identity"}
    payload["corpus_identity"] = canonical_sha256(body)


def test_candidate_blind_selection_is_exact_and_phase_covered() -> None:
    selected = select_replayable_phase_entries(_phase_payload())

    assert [entry["index"] for entry in selected] == [
        1,
        10,
        17,
        19,
        23,
        36,
        39,
        41,
        45,
        48,
        58,
        63,
    ]
    assert all(entry["sources"][0]["color_transform"] == "identity" for entry in selected)


def test_compulsory_removal_is_one_logical_turn() -> None:
    assert group_action_tokens(["d6", "d2", "f4", "xb4", "c3"]) == [
        ["d6"],
        ["d2"],
        ["f4", "xb4"],
        ["c3"],
    ]
    with pytest.raises(PhaseReplayCorpusError, match="orphan removal"):
        group_action_tokens(["xd6"])


def test_committed_replay_corpus_is_legal_and_source_bound() -> None:
    payload = _replay_payload()
    report = validate_phase_replay_development_corpus(
        payload,
        phase_corpus=_phase_payload(),
    )

    assert payload["corpus_id"] == CORPUS_ID
    assert report["record_count"] == 12
    assert report["phase_counts"] == {
        "placement": 4,
        "movement": 4,
        "flying": 4,
    }


def test_shared_strict_replay_entry_reaches_record_state() -> None:
    record = _replay_payload()["records"][0]

    class FakeGame:
        def __init__(self) -> None:
            self.board = None
            self.state = type("State", (), {"terminal": False})()

        def apply_nmm_move(self, board, move) -> None:
            assert self.board is None or board == self.board
            self.board = board.apply_move(move)

        def assert_current_board(self, board) -> None:
            assert board == self.board

    game = FakeGame()
    board = replay_record_into_sanmill_game(record, game)

    assert board.to_fen_string() == record["fen"]


def test_history_tampering_fails_even_with_refreshed_identities() -> None:
    payload = copy.deepcopy(_replay_payload())
    payload["records"][0]["logical_turns"][0] = ["a1"]
    payload["records"][0]["action_history"][0] = "a1"
    _refresh_identities(payload)

    with pytest.raises(PhaseReplayCorpusError, match="logical ply"):
        validate_phase_replay_development_corpus(payload)


def test_source_projection_tampering_fails_closed() -> None:
    payload = copy.deepcopy(_replay_payload())
    payload["records"][0]["malom_wdl_for_side_to_move"] = "D"
    _refresh_identities(payload)

    with pytest.raises(PhaseReplayCorpusError, match="source phase entry projection"):
        validate_phase_replay_development_corpus(
            payload,
            phase_corpus=_phase_payload(),
        )


def test_committed_strict_sanmill_audit_is_bound_to_corpus() -> None:
    corpus = _replay_payload()
    report = json.loads(SANMILL_AUDIT.read_text(encoding="utf-8"))

    result = validate_phase_replay_sanmill_audit(report, corpus=corpus)

    assert result["record_count"] == 12
    assert result["fresh_process_count"] == 24


def test_strict_sanmill_audit_tampering_fails_closed() -> None:
    corpus = _replay_payload()
    report = json.loads(SANMILL_AUDIT.read_text(encoding="utf-8"))
    report["records"][0]["terminal"] = True
    body = {key: value for key, value in report.items() if key != "audit_identity"}
    report["audit_identity"] = canonical_sha256(body)

    with pytest.raises(PhaseReplayCorpusError, match="observation differs"):
        validate_phase_replay_sanmill_audit(report, corpus=corpus)
