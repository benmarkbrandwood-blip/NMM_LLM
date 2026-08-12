"""Focused tests for mature target-refresh analysis-only recovery."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from learned_ai.training.run_contract import canonical_sha256
from scripts import run_target_refresh_mature_fork_analysis_recovery as recovery


def _plan() -> dict:
    body = {
        "schema_version": recovery.PLAN_SCHEMA,
        "status": "designed_unlaunched_needs_authorization",
        "resource_envelope": {
            "candidate_models_loaded": True,
            "checkpoint_writes": 0,
            "database_writes": 0,
            "maximum_active_wall_hours": 3.5,
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
        "local_inputs": {
            "paths_config": "data/training_paths.local.json",
            "malom_manifest": "data/manifests/malom-sector-corrected-v1.json",
        },
        "outputs": {
            "development_ledger": "out/ledger.jsonl",
            "development_result": "out/result.json",
        },
        "claim_boundary": "development evidence only",
    }
    return {**body, "plan_identity": canonical_sha256(body)}


def test_recovery_plan_requires_canonical_identity(tmp_path: Path) -> None:
    plan = _plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    assert recovery.load_recovery_plan(path)["plan_identity"] == plan[
        "plan_identity"
    ]
    plan["claim_boundary"] = "changed"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(
        recovery.MatureRefreshAnalysisRecoveryError,
        match="identity differs",
    ):
        recovery.load_recovery_plan(path)


def test_repository_recovery_plan_freezes_published_implementation() -> None:
    plan = recovery.load_recovery_plan(recovery.DEFAULT_PLAN)

    assert plan["plan_identity"] == (
        "70fb522b863ceb583b393697a11894540ce3ab5c5764b6aa8e892ebb7cc451e6"
    )
    assert plan["completed_artifact_identity"] == (
        "c186b7b02e2b012e6e899fe48429a388def6f94a640118faac273fb3a60fe49d"
    )
    commit = plan["analysis_implementation"]["minimum_commit"]
    for name in ("publisher", "runner"):
        record = plan["analysis_implementation"][name]
        frozen = subprocess.check_output(
            ["git", "show", f"{commit}:{record['path']}"],
            cwd=recovery.ROOT,
        )
        assert hashlib.sha256(frozen).hexdigest() == record["sha256"]


def test_successor_recovery_plan_is_isolated_and_hash_bound() -> None:
    path = (
        recovery.ROOT
        / "docs/experiments/"
        "sanmill-target-refresh-mature-fork-analysis-recovery-v2.json"
    )
    plan = recovery.load_recovery_plan(path)

    assert plan["plan_identity"] == (
        "32158846cb3e3903589663465d6217ed546442eee617e0aa5fe94defe45feb25"
    )
    assert "analysis-recovery-v2" in plan["control_files"]["readiness"]
    assert "analysis-recovery-v2" in plan["outputs"]["failure"]
    commit = subprocess.check_output(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            path.relative_to(recovery.ROOT).as_posix(),
        ],
        cwd=recovery.ROOT,
        text=True,
    ).strip()
    for name in ("publisher", "runner"):
        record = plan["analysis_implementation"][name]
        frozen = subprocess.check_output(
            ["git", "show", f"{commit}:{record['path']}"],
            cwd=recovery.ROOT,
        )
        assert hashlib.sha256(frozen).hexdigest() == record["sha256"]


def test_authorization_is_exactly_plan_and_readiness_bound() -> None:
    plan = _plan()
    authorization = recovery.build_authorization(
        plan=plan,
        readiness_identity="b" * 64,
        authorized_at_utc="2026-08-12T12:00:00Z",
        decision_note="one bounded analysis-only recovery",
    )

    assert authorization["resource_envelope"]["training_games"] == 0
    assert authorization["resource_envelope"]["no_update_development_games"] == 288
    assert "training-retry-resume-or-recovery" in authorization[
        "prohibited_operations"
    ]
    body = dict(authorization)
    identity = body.pop("authorization_identity")
    assert identity == canonical_sha256(body)


def test_reporter_command_is_cpu_analysis_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setattr(recovery, "ROOT", Path("repo"))
    monkeypatch.setattr(recovery.sys, "executable", "python")

    command = recovery._reporter_command(plan)

    assert command[0] == "python"
    assert "report_target_refresh_mature_fork_diagnostic.py" in command[1]
    assert "--allow-published-analysis-descendant" in command
    assert command[command.index("--device") + 1] == "cpu"
    assert "train_s_gen_v2.py" not in " ".join(command)


def test_resource_envelope_cannot_expand(tmp_path: Path) -> None:
    plan = _plan()
    plan["resource_envelope"]["training_games"] = 1
    body = dict(plan)
    body.pop("plan_identity")
    plan["plan_identity"] = canonical_sha256(body)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(
        recovery.MatureRefreshAnalysisRecoveryError,
        match="resource envelope",
    ):
        recovery.load_recovery_plan(path)


def test_control_paths_cannot_be_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    monkeypatch.setattr(recovery, "ROOT", tmp_path)

    with pytest.raises(
        recovery.MatureRefreshAnalysisRecoveryError,
        match="path differs",
    ):
        recovery._require_control_path(
            plan,
            name="readiness",
            observed=tmp_path / "out/elsewhere.json",
        )


def test_completed_artifact_manifest_binds_analysis_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    control = tmp_path / "out/arm"
    segment = control / "segments/segment-0001"
    database = tmp_path / "data/arm.sqlite"
    paths = {
        "plan": control / "plan.json",
        "authorization": control / "authorization.json",
        "controller_events": control / "controller-events.jsonl",
        "initial_branch": control / "initial-mature-target-refresh-fork.pt",
        "train_log": segment / "train_log.jsonl",
        "update_log": segment / "update_log.jsonl",
        "policy_health": segment / "policy-health.json",
        "transition_4096": segment / "transition-00004096.pt",
        "transition_8192": segment / "transition-00008192.pt",
        "latest": segment / "latest.pt",
    }
    for index, path in enumerate(paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"database")
    contract = {
        "arms": [
            {
                "seed": 67,
                "condition": "refresh-mature",
                "arm_id": "arm",
                "control_dir": "out/arm",
                "specialist_db": "data/arm.sqlite",
            }
        ]
    }
    result = {
        "plan_sha256": "managed-plan",
        "authorization_sha256": recovery._sha256_file(paths["authorization"]),
        "controller_events_sha256": recovery._sha256_file(
            paths["controller_events"]
        ),
    }
    failure = {
        "completed_steps": [
            {
                "kind": "run-arm",
                "launch_order": 1,
                "seed": 67,
                "condition": "refresh-mature",
                "result": result,
            }
        ]
    }
    monkeypatch.setattr(
        recovery.reporter,
        "load_managed_plan",
        lambda path: SimpleNamespace(plan_sha256="managed-plan"),
    )

    manifest = recovery.build_completed_artifact_manifest(
        contract=contract,
        failure=failure,
    )

    files = manifest["arms"][0]["files"]
    assert files["transition_4096"]["sha256"] == recovery._sha256_file(
        paths["transition_4096"]
    )
    assert files["transition_8192"]["sha256"] == recovery._sha256_file(
        paths["transition_8192"]
    )
    assert files["latest"]["sha256"] == recovery._sha256_file(paths["latest"])


def test_completed_artifact_manifest_rejects_database_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    control = tmp_path / "out/arm"
    segment = control / "segments/segment-0001"
    required = (
        control / "plan.json",
        control / "authorization.json",
        control / "controller-events.jsonl",
        control / "initial-mature-target-refresh-fork.pt",
        segment / "train_log.jsonl",
        segment / "update_log.jsonl",
        segment / "policy-health.json",
        segment / "transition-00004096.pt",
        segment / "transition-00008192.pt",
        segment / "latest.pt",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"value")
    database = tmp_path / "data/arm.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_bytes(b"database")
    Path(str(database) + "-wal").write_bytes(b"sidecar")
    sha = recovery._sha256_file
    contract = {
        "arms": [
            {
                "seed": 67,
                "condition": "refresh-mature",
                "arm_id": "arm",
                "control_dir": "out/arm",
                "specialist_db": "data/arm.sqlite",
            }
        ]
    }
    failure = {
        "completed_steps": [
            {
                "kind": "run-arm",
                "launch_order": 1,
                "seed": 67,
                "condition": "refresh-mature",
                "result": {
                    "plan_sha256": "managed-plan",
                    "authorization_sha256": sha(control / "authorization.json"),
                    "controller_events_sha256": sha(
                        control / "controller-events.jsonl"
                    ),
                },
            }
        ]
    }
    monkeypatch.setattr(
        recovery.reporter,
        "load_managed_plan",
        lambda path: SimpleNamespace(plan_sha256="managed-plan"),
    )

    with pytest.raises(
        recovery.MatureRefreshAnalysisRecoveryError,
        match="database has sidecars",
    ):
        recovery.build_completed_artifact_manifest(
            contract=contract,
            failure=failure,
        )


def test_candidate_semantics_audit_covers_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = {
        "sources": [{"seed": seed} for seed in (67, 68, 69)],
        "arms": [
            {"seed": seed, "condition": condition}
            for seed in (67, 68, 69)
            for condition in ("refresh-mature", "stale-control")
        ],
    }
    readiness = {
        "seeds": [
            {"seed": seed, "arms": [{"condition": "refresh-mature"}]}
            for seed in (67, 68, 69)
        ]
    }
    monkeypatch.setattr(
        recovery.reporter,
        "_load_common_fork",
        lambda **kwargs: (object(), {"checkpoint_id": f"fork-{kwargs['source']['seed']}"}),
    )

    calls: list[tuple[int, int]] = []

    def load_pair(**kwargs):
        calls.append((kwargs["seed"], kwargs["boundary"]))
        return {}, {
            "refresh-mature": {"checkpoint_id": "refresh"},
            "stale-control": {"checkpoint_id": "stale"},
        }

    monkeypatch.setattr(recovery.reporter, "_load_candidate_pair", load_pair)

    audit = recovery._audit_candidate_semantics(
        contract=contract,
        readiness=readiness,
    )

    assert calls == [
        (seed, boundary)
        for seed in (67, 68, 69)
        for boundary in recovery.reporter.POLICY_BOUNDARIES
    ]
    assert len(audit["candidate_pairs"]) == 6
    body = dict(audit)
    identity = body.pop("audit_identity")
    assert identity == canonical_sha256(body)
