"""Focused tests for the schedule-isolation analysis-only recovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_sha256
from scripts import run_target_refresh_schedule_isolation_analysis_recovery as recovery


def _plan() -> dict:
    body = {
        "schema_version": recovery.PLAN_SCHEMA,
        "status": "designed_unlaunched_needs_authorization",
        "resource_envelope": {
            "candidate_models_loaded": True,
            "checkpoint_writes": 0,
            "database_writes": 0,
            "maximum_active_wall_hours": 5.5,
            "no_update_development_games": 288,
            "optimizer_updates": 0,
            "training_games": 0,
        },
        "parent_attempt": {
            "contract": {"path": "contract.json"},
            "readiness": {"path": "readiness.json"},
        },
        "control_files": {
            "authorization": "out/authorization.json",
            "readiness": "out/readiness.json",
        },
        "local_inputs": {"paths_config": "data/training_paths.local.json"},
        "outputs": {
            "development_ledger": "out/ledger.jsonl",
            "development_result": "out/result.json",
        },
        "claim_boundary": "development evidence only",
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def test_recovery_plan_requires_canonical_identity(
    tmp_path: Path,
) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(
        __import__("json").dumps(plan),
        encoding="utf-8",
    )

    assert recovery.load_recovery_plan(path)["plan_identity"] == plan[
        "plan_identity"
    ]
    plan["claim_boundary"] = "changed"
    path.write_text(__import__("json").dumps(plan), encoding="utf-8")
    with pytest.raises(recovery.AnalysisRecoveryError, match="identity differs"):
        recovery.load_recovery_plan(path)


def test_authorization_is_exactly_plan_and_readiness_bound() -> None:
    plan = _plan()
    authorization = recovery.build_authorization(
        plan=plan,
        readiness_identity="b" * 64,
        authorized_at_utc="2026-08-11T12:00:00Z",
        decision_note="one bounded analysis-only recovery",
    )

    assert authorization["resource_envelope"]["training_games"] == 0
    assert authorization["resource_envelope"]["no_update_development_games"] == 288
    assert "training-retry-or-resume" in authorization["prohibited_operations"]
    body = dict(authorization)
    identity = body.pop("authorization_identity")
    assert identity == canonical_sha256(body)


def test_reporter_command_is_cpu_analysis_only(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    monkeypatch.setattr(recovery, "ROOT", Path("repo"))
    monkeypatch.setattr(recovery.sys, "executable", "python")

    command = recovery._reporter_command(plan)

    assert command[0] == "python"
    assert "report_target_refresh_schedule_isolation_diagnostic.py" in command[1]
    assert command[-2:] == ["--device", "cpu"]
    assert "train_s_gen_v2.py" not in " ".join(command)


def test_resource_envelope_cannot_expand(tmp_path: Path) -> None:
    plan = _plan()
    plan["resource_envelope"]["training_games"] = 1
    body = dict(plan)
    body.pop("plan_identity")
    plan["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "plan.json"
    path.write_text(__import__("json").dumps(plan), encoding="utf-8")

    with pytest.raises(recovery.AnalysisRecoveryError, match="resource envelope"):
        recovery.load_recovery_plan(path)


def test_control_paths_cannot_be_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    monkeypatch.setattr(recovery, "ROOT", tmp_path)

    with pytest.raises(recovery.AnalysisRecoveryError, match="path differs"):
        recovery._require_control_path(
            plan,
            name="readiness",
            observed=tmp_path / "out/elsewhere.json",
        )
