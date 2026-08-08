from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.evaluation.heldout_exposure import (
    HeldoutExposureError,
    build_exposure_audit,
    classify_exposure,
    validate_executable_corpus,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json"
)
CORPUS_IDENTITY = (
    "417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9"
)
RECORDS_IDENTITY = (
    "e8a1828cb1d7e0e86c686d934e87934c6c12e6a8cf7610974ed8035937ab8cff"
)


class _Human:
    def __init__(self, exposed_fens: set[str]) -> None:
        self.exposed_fens = exposed_fens

    def query_position(self, board):
        if board.to_fen_string() in self.exposed_fens:
            return SimpleNamespace(total_games=7)
        return None


class _Specialist:
    def __init__(self, exposed_fens: set[str]) -> None:
        self.exposed_fens = exposed_fens

    def query_wdl_evidence(self, board, min_samples=0):
        assert min_samples == 0
        if board.to_fen_string() in self.exposed_fens:
            return SimpleNamespace(
                empirical_counts=(1, 2, 3),
                theoretical_wdl=SimpleNamespace(value="D"),
            )
        return None


def _payload() -> dict:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_frozen_executable_corpus_identity_is_accepted() -> None:
    records = validate_executable_corpus(
        _payload(),
        expected_corpus_identity=CORPUS_IDENTITY,
        expected_records_identity=RECORDS_IDENTITY,
    )
    assert len(records) == 64


def test_executable_corpus_identity_drift_is_rejected() -> None:
    payload = copy.deepcopy(_payload())
    payload["corpus"]["records"][0]["stratum"] = "changed"
    with pytest.raises(HeldoutExposureError, match="records have drifted"):
        validate_executable_corpus(
            payload,
            expected_corpus_identity=CORPUS_IDENTITY,
            expected_records_identity=RECORDS_IDENTITY,
        )


def test_exposure_classification_separates_operational_and_strict_sets() -> None:
    records = validate_executable_corpus(
        _payload(),
        expected_corpus_identity=CORPUS_IDENTITY,
        expected_records_identity=RECORDS_IDENTITY,
    )[:3]
    fens = [record["execution_record"]["final"]["nmm_fen"] for record in records]
    classified = classify_exposure(
        records,
        human_db=_Human({fens[0]}),
        specialist_db=_Specialist({fens[1]}),
    )
    assert [item["strict_independence_subset"] for item in classified] == [
        False,
        False,
        True,
    ]
    assert classified[0]["human_db_games"] == 7
    assert classified[1]["specialist_db_empirical_samples"] == 6
    assert classified[1]["specialist_db_has_theoretical_label"] is True

    audit = build_exposure_audit(
        records,
        human_db=_Human({fens[0]}),
        specialist_db=_Specialist({fens[1]}),
        corpus_identity="corpus",
        records_identity="records",
        human_db_identity="human",
        specialist_db_identity="specialist",
    )
    assert audit["summary"]["strict_independence_count"] == 1
    assert audit["strict_independence_source_core_ids"] == [
        records[2]["source_core_id"]
    ]
    assert audit["candidate_loaded"] is False
    assert audit["games_played"] == 0
