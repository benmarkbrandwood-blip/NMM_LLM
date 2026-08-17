from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    append_resource_checkpoint,
    load_resource_checkpoints,
)
from learned_ai.evaluation.sanmill_trained_model_boundary_registry import (
    BoundaryCoverageRecorder,
    BoundaryRegistryError,
    REGISTRY_SCHEMA,
    audit_boundary_registry,
    canonical_sha256,
    load_boundary_registry,
    reflected_code_identity,
    reflected_signature,
    verify_rehearsal_coverage,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB


_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY = (
    _ROOT
    / "docs/experiments/"
    "sanmill-trained-model-baseline-boundary-registry-v1.json"
)


def _covered_mapping() -> dict[str, bool]:
    return {"covered": True}


def _second_mapping() -> dict[str, bool]:
    return {"second": True}


def _preflight_mapping() -> dict[str, bool]:
    return {"preflight": True}


def _row(
    boundary_id: str,
    function: object,
    *,
    classification: str,
) -> dict[str, object]:
    stage = {
        "rehearsal-required": ("rehearsal", "profile-event"),
        "preflight-required": ("preflight", "explicit-canary"),
    }[classification]
    return {
        "boundary_id": boundary_id,
        "module": __name__,
        "callable": f"{__name__}:{function.__name__}",
        "callable_kind": "callable",
        "role": "canary",
        "classification": classification,
        "required_stage": stage[0],
        "dynamic_required": True,
        "evidence_mode": stage[1],
        "reason": "Focused registry discrimination fixture.",
        "result_shape_validator": "mapping",
        "resource_semantics": "none",
        "signature": reflected_signature(function),
        "code_identity": reflected_code_identity(function, _ROOT),
    }


def _registry(*rows: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": "focused-test-registry",
        "frozen_at_utc": "2026-08-17T00:00:00Z",
        "classification_principle": {"fixture": True},
        "boundaries": list(rows),
    }
    value["registry_identity"] = canonical_sha256(value)
    return value


def test_registry_rejects_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _file_sha256 = load_boundary_registry(_REGISTRY)

    def incompatible_query_all_moves(self: object, board: object) -> list[dict]:
        del self, board
        return []

    monkeypatch.setattr(
        ExternalSolvedDB,
        "query_all_moves",
        incompatible_query_all_moves,
    )
    report = audit_boundary_registry(registry, repository_root=_ROOT)
    failures = {
        row["boundary_id"]
        for row in report["signature_checks"]
        if row["passed"] is False
    }
    assert failures == {"solved-db.query-all-moves"}
    assert report["passed"] is False


def test_registry_rejects_an_unregistered_public_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _file_sha256 = load_boundary_registry(_REGISTRY)

    def query_batch(self: object, boards: list[object]) -> list[object]:
        del self
        return boards

    monkeypatch.setattr(ExternalSolvedDB, "query_batch", query_batch, raising=False)
    report = audit_boundary_registry(registry, repository_root=_ROOT)
    assert "public method or property surface differs" in report["mismatches"]
    assert report["passed"] is False


def test_new_unexecuted_rehearsal_boundary_fails_dynamic_gate(
    tmp_path: Path,
) -> None:
    registry = _registry(
        _row(
            "fixture.covered",
            _covered_mapping,
            classification="rehearsal-required",
        ),
        _row(
            "fixture.new-unexecuted",
            _second_mapping,
            classification="rehearsal-required",
        ),
    )
    ledger = tmp_path / "coverage.jsonl"
    with BoundaryCoverageRecorder(
        registry,
        ledger,
        formal_result_eligibility=False,
    ):
        _covered_mapping()

    with pytest.raises(BoundaryRegistryError, match="new-unexecuted"):
        verify_rehearsal_coverage(ledger, registry)


def test_preflight_only_event_cannot_satisfy_rehearsal_coverage(
    tmp_path: Path,
) -> None:
    registry = _registry(
        _row(
            "fixture.required-rehearsal",
            _covered_mapping,
            classification="rehearsal-required",
        ),
        _row(
            "fixture.preflight-only",
            _preflight_mapping,
            classification="preflight-required",
        ),
    )
    ledger = tmp_path / "coverage.jsonl"
    with BoundaryCoverageRecorder(
        registry,
        ledger,
        formal_result_eligibility=False,
    ):
        _preflight_mapping()

    with pytest.raises(BoundaryRegistryError, match="required-rehearsal"):
        verify_rehearsal_coverage(ledger, registry)


def test_profile_event_maps_to_the_registered_real_code_object(
    tmp_path: Path,
) -> None:
    row = _row(
        "fixture.real-code",
        _covered_mapping,
        classification="rehearsal-required",
    )
    registry = _registry(row)
    ledger = tmp_path / "coverage.jsonl"
    with BoundaryCoverageRecorder(
        registry,
        ledger,
        formal_result_eligibility=False,
    ):
        assert _covered_mapping() == {"covered": True}

    recovered = verify_rehearsal_coverage(ledger, registry)
    assert recovered["passed"] is True
    assert recovered["events"][0]["boundary_id"] == "fixture.real-code"
    assert recovered["events"][0]["code_identity"] == row["code_identity"]


def test_crash_keeps_exact_completed_game_resources(tmp_path: Path) -> None:
    record = {"ordinal": 0, "game_id": "crash-fixture"}
    before = {
        "engine_single_step_searches": 248,
        "malom_read_only_queries": 6792,
        "active_seconds": 114.31433940000716,
    }
    after = {
        "engine_single_step_searches": 249,
        "malom_read_only_queries": 6807,
        "active_seconds": 114.75,
    }
    journal = tmp_path / "resource-checkpoints.jsonl"
    append_resource_checkpoint(
        journal,
        completion_index=0,
        complete_games_before=27,
        game_record=record,
        resources_before=before,
        resources_after=after,
        previous_checkpoint_sha256=None,
    )

    recovered = load_resource_checkpoints(
        journal,
        expected_baseline=before,
        complete_games_before=27,
    )
    assert recovered["checkpoint_count"] == 1
    assert recovered["last_resources"] == after
    assert not (tmp_path / "games.jsonl").exists()
    assert json.loads(journal.read_text(encoding="ascii").splitlines()[0])
