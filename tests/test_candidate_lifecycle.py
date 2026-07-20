from __future__ import annotations

import json
from pathlib import Path

import pytest

import learned_ai.delivery.candidate_lifecycle as lifecycle
from learned_ai.delivery.candidate_lifecycle import CandidateLifecycleError


def test_candidate_state_machine_preserves_inconclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(lifecycle, "verify_model_bundle", lambda path: {"bundle_identity": "a" * 64, "model_identity": "b" * 64})
    lifecycle.register_candidate(tmp_path / "registry", "candidate-1", bundle)
    lifecycle.transition_candidate(tmp_path / "registry", "candidate-1", "validating")
    lifecycle.transition_candidate(tmp_path / "registry", "candidate-1", "evaluating")
    monkeypatch.setattr(lifecycle, "load_evaluation_spec", lambda path: type("Spec", (), {"candidate_bundle": "a" * 64, "spec_identity": "c" * 64})())
    monkeypatch.setattr(lifecycle, "recompute_evaluation", lambda spec, records: {"decision": "inconclusive", "result_identity": "d" * 64, "records_sha256": "e" * 64})

    result = lifecycle.decide_candidate(tmp_path / "registry", "candidate-1", "spec", "records", tmp_path / "accepted")

    assert result["status"] == "inconclusive"
    assert not (tmp_path / "accepted").exists()
    assert lifecycle.candidate_status(tmp_path / "registry", "candidate-1")["status"] == "inconclusive"


def test_candidate_rejects_invalid_transition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(lifecycle, "verify_model_bundle", lambda path: {"bundle_identity": "a" * 64, "model_identity": "b" * 64})
    lifecycle.register_candidate(tmp_path / "registry", "candidate-1", bundle)

    with pytest.raises(CandidateLifecycleError, match="invalid candidate transition"):
        lifecycle.transition_candidate(tmp_path / "registry", "candidate-1", "evaluating")


def test_accepted_decision_atomically_copies_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "weights.pt").write_bytes(b"weights")
    monkeypatch.setattr(lifecycle, "verify_model_bundle", lambda path: {"bundle_identity": "a" * 64, "model_identity": "b" * 64})
    lifecycle.register_candidate(tmp_path / "registry", "candidate-1", bundle)
    lifecycle.transition_candidate(tmp_path / "registry", "candidate-1", "validating")
    lifecycle.transition_candidate(tmp_path / "registry", "candidate-1", "evaluating")
    monkeypatch.setattr(lifecycle, "load_evaluation_spec", lambda path: type("Spec", (), {"candidate_bundle": "a" * 64, "spec_identity": "c" * 64})())
    monkeypatch.setattr(lifecycle, "recompute_evaluation", lambda spec, records: {"decision": "accepted", "result_identity": "d" * 64, "records_sha256": "e" * 64})

    result = lifecycle.decide_candidate(tmp_path / "registry", "candidate-1", "spec", "records", tmp_path / "accepted")

    accepted = tmp_path / "accepted" / ("a" * 64)
    assert result["status"] == "accepted"
    assert (accepted / "weights.pt").read_bytes() == b"weights"
    acceptance = json.loads((accepted / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["result_identity"] == "d" * 64


def test_quarantine_requires_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(lifecycle, "verify_model_bundle", lambda path: {"bundle_identity": "a" * 64, "model_identity": "b" * 64})
    lifecycle.register_candidate(tmp_path / "registry", "candidate-1", bundle)

    with pytest.raises(CandidateLifecycleError, match="explicit reason"):
        lifecycle.transition_candidate(tmp_path / "registry", "candidate-1", "quarantined")
