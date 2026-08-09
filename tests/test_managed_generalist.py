"""Tests for the bounded, product-authorized Generalist supervisor."""

from __future__ import annotations

import json
import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from learned_ai.training import managed_generalist as managed
from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    CheckpointPayload,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from learned_ai.training.managed_generalist import (
    ManagedContractError,
    ManagedPlan,
    PolicyHealthGate,
    authorize_plan,
    build_segment_command,
    load_managed_plan,
    managed_status,
    publish_managed_plan,
    recover_failed_segment,
    recover_interrupted_segment,
    run_next_segment,
    verify_managed_launch,
)


def _managed_checkpoint_payload(game_count: int) -> CheckpointPayload:
    return CheckpointPayload(
        model_state={},
        optimizer_state=None,
        scheduler_state=None,
        scaler_state=None,
        rng_state=capture_rng_state(),
        trainer_state={
            "game_count": game_count,
            "batch_count": game_count,
            "update_count": 0,
            "difficulty": 1,
            "temperature": 0.8,
            "rolling_metrics": {},
            "curriculum": {},
            "target_network": {},
            "recovery_state": {},
            "model_config": {},
        },
        data_state={
            "cursor": {"completed_games": game_count},
            "consumed_snapshots": [],
            "cache": {},
            "buckets": {},
            "mutable_assets": {
                "specialist_db": {"sha256": "d" * 64}
            },
        },
    )


def _plan(tmp_path: Path) -> ManagedPlan:
    paths_config = tmp_path / "training_paths.local.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    paths_config_sha256 = hashlib.sha256(paths_config.read_bytes()).hexdigest()
    return ManagedPlan(
        plan_id="managed-v4-test",
        created_at_utc="2026-07-20T12:00:00Z",
        objective="corrected-v4-single-gpu-baseline",
        experiment_id="dev-v4-managed-baseline-v1",
        git_commit="a" * 40,
        control_dir=str((tmp_path / "control").resolve()),
        paths_config=str(paths_config.resolve()),
        paths_config_sha256=paths_config_sha256,
        resume_config_sha256="c" * 64,
        max_games=500,
        segment_games=100,
        max_wall_hours=12.0,
        common_trainer_args=(
            "--experiment-id",
            "dev-v4-managed-baseline-v1",
            "--max-games",
            "500",
            "--heuristic-node-budget",
            "500000",
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--no-s1a-warmstart",
            "--no-imitation-mix",
        ),
        allow_safe_exact_resume=True,
        publication_allowed=False,
        promotion_allowed=False,
    )


def _policy_health_plan(tmp_path: Path) -> ManagedPlan:
    plan = _plan(tmp_path)
    corpus = tmp_path / "fixed-corpus.json"
    corpus.write_text('{"entries": []}\n', encoding="utf-8")
    audit_script = tmp_path / "audit.py"
    audit_script.write_text("# test audit identity\n", encoding="utf-8")
    specialist_db = tmp_path / "specialist.sqlite"
    specialist_db.write_bytes(b"test-specialist-db")
    gate = PolicyHealthGate(
        corpus_path=str(corpus.resolve()),
        corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        audit_script_path=str(audit_script.resolve()),
        audit_script_sha256=hashlib.sha256(audit_script.read_bytes()).hexdigest(),
        exact_critical_states=29,
        required_direct_preserving_rate=1.0,
        min_candidate_preserving_rate=0.50,
        min_candidate_logit_margin=-0.10,
    )
    return replace(
        plan,
        common_trainer_args=(
            *plan.common_trainer_args,
            "--seed",
            "42",
            "--temp-start",
            "0.90",
            "--specialist-db",
            str(specialist_db.resolve()),
        ),
        policy_health=gate,
    )


def _publish_health_report(
    command: list[str],
    plan: ManagedPlan,
    checkpoint: Path,
    *,
    candidate_rate: float = 0.75,
    candidate_margin: float = 0.05,
) -> Path:
    assert plan.policy_health is not None
    output = Path(command[command.index("--output") + 1])
    specialist_db = Path(command[command.index("--specialist-db") + 1])
    core = {
        "schema_version": "nmm.generalist-policy-health.v1",
        "identities": {
            "git_commit": plan.git_commit,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest(),
            "run_id": "managed-v4-test-segment-0001",
            "experiment_id": plan.experiment_id,
            "corpus": plan.policy_health.corpus_path,
            "corpus_sha256": plan.policy_health.corpus_sha256,
            "paths_config_sha256": plan.paths_config_sha256,
            "specialist_db": str(specialist_db.resolve()),
            "specialist_db_sha256": hashlib.sha256(
                specialist_db.read_bytes()
            ).hexdigest(),
        },
        "checkpoint_state": {"game_count": 100},
        "fixed_state_diagnostic": {
            "direct_lookahead_signal": {
                "critical_states": 29,
                "argmax_value_preserving_rate": 1.0,
            },
            "candidate": {
                "metrics": {
                    "all": {
                        "critical_states": 29,
                        "critical_argmax_value_preserving_rate": candidate_rate,
                        "critical_mean_preserving_minus_downgrading_logit": (
                            candidate_margin
                        ),
                    }
                }
            },
        },
    }
    report = {**core, "evidence_id": managed.canonical_sha256(core)}
    output.write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def test_plan_is_exclusive_and_tamper_evident(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"

    publish_managed_plan(plan_path, plan)

    assert load_managed_plan(plan_path) == plan
    with pytest.raises(FileExistsError):
        publish_managed_plan(plan_path, plan)

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["max_games"] = 501
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManagedContractError, match="plan hash"):
        load_managed_plan(plan_path)


def test_policy_health_plan_round_trip_preserves_legacy_plan_shape(
    tmp_path: Path,
) -> None:
    legacy = _plan(tmp_path)
    assert "policy_health" not in legacy.to_dict()
    assert "completion_game_bound" not in legacy.to_dict()
    health = _policy_health_plan(tmp_path)

    restored = ManagedPlan.from_dict(health.to_dict())

    assert restored == health
    assert restored.policy_health is not None
    assert restored.policy_health.exact_critical_states == 29


def test_completion_bound_round_trip_keeps_larger_schedule_horizon(
    tmp_path: Path,
) -> None:
    plan = replace(
        _plan(tmp_path),
        max_games=5000,
        completion_game_bound=100,
        common_trainer_args=(
            "--experiment-id",
            "dev-v4-managed-baseline-v1",
            "--max-games",
            "5000",
            "--heuristic-node-budget",
            "500000",
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--no-s1a-warmstart",
            "--no-imitation-mix",
        ),
    )

    restored = ManagedPlan.from_dict(plan.to_dict())
    command = build_segment_command(
        restored,
        plan_path=tmp_path / "control" / "plan.json",
        authorization_path=tmp_path / "control" / "authorization.json",
        segment_index=1,
        previous_checkpoint=None,
        previous_run_id=None,
        previous_completed_games=0,
        python_executable="python",
    )

    assert restored.game_bound == 100
    assert restored.max_games == 5000
    assert restored.to_dict()["completion_game_bound"] == 100
    assert command[command.index("--segment-stop-game") + 1] == "100"
    assert command[command.index("--max-games") + 1] == "5000"


def test_authorization_is_separate_and_bound_to_exact_plan(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)

    before = managed_status(plan_path, authorization_path)
    assert before["state"] == "awaiting_product_authorization"
    assert before["needs_product_decision"] is True

    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Run within the frozen resource envelope.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )

    after = managed_status(plan_path, authorization_path)
    assert after["state"] == "ready_to_run"
    assert after["needs_product_decision"] is False


def test_segment_commands_only_allow_fresh_then_exact_resume(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"

    first = build_segment_command(
        plan,
        plan_path=plan_path,
        authorization_path=authorization_path,
        segment_index=1,
        previous_checkpoint=None,
        previous_run_id=None,
        previous_completed_games=0,
        python_executable="python",
    )
    assert first[first.index("--start-mode") + 1] == "fresh"
    assert "--resume" not in first
    assert first[first.index("--segment-stop-game") + 1] == "100"

    second = build_segment_command(
        plan,
        plan_path=plan_path,
        authorization_path=authorization_path,
        segment_index=2,
        previous_checkpoint=tmp_path / "segment-0001" / "latest.pt",
        previous_run_id="managed-v4-test-segment-0001",
        previous_completed_games=100,
        python_executable="python",
    )
    assert second[second.index("--start-mode") + 1] == "exact-resume"
    assert second[second.index("--parent-run-id") + 1] == (
        "managed-v4-test-segment-0001"
    )
    assert second[second.index("--segment-stop-game") + 1] == "200"


def test_launch_verification_rejects_wrong_semantics(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )

    with pytest.raises(ManagedContractError, match="training semantics"):
        verify_managed_launch(
            plan_path,
            authorization_path,
            git_commit=plan.git_commit,
            resume_config_sha256="d" * 64,
            out_dir=Path(plan.control_dir) / "segments" / "segment-0001",
            run_id="managed-v4-test-segment-0001",
            segment_games=plan.segment_games,
            segment_stop_game=plan.segment_games,
            start_mode="fresh",
            resume="",
            parent_run_id=None,
            experiment_id=plan.experiment_id,
        )


def test_launch_verification_accepts_exact_authorized_segment(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )

    verified = verify_managed_launch(
        plan_path,
        authorization_path,
        git_commit=plan.git_commit,
        resume_config_sha256=plan.resume_config_sha256,
        out_dir=Path(plan.control_dir) / "segments" / "segment-0001",
        run_id="managed-v4-test-segment-0001",
        segment_games=plan.segment_games,
        segment_stop_game=plan.segment_games,
        start_mode="fresh",
        resume="",
        parent_run_id=None,
        experiment_id=plan.experiment_id,
    )

    assert verified == plan


def test_supervisor_never_runs_without_product_authorization(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    calls: list[list[str]] = []

    def unexpected_runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(ManagedContractError, match="authorization"):
        run_next_segment(
            plan_path,
            authorization_path,
            runner=unexpected_runner,
            python_executable="python",
        )

    assert calls == []


def test_supervisor_runs_one_bounded_segment_and_publishes_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    monkeypatch.setattr(managed, "_git_state", lambda _root: (plan.git_commit, False))
    checkpoint = Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
    monkeypatch.setattr(
        managed,
        "_inspect_completed_segment",
        lambda *_args, **_kwargs: (100, checkpoint),
    )
    calls: list[tuple[list[str], dict]] = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    status = run_next_segment(
        plan_path,
        authorization_path,
        runner=runner,
        python_executable="python",
    )

    assert len(calls) == 1
    command, options = calls[0]
    assert command[:3] == ["python", "scripts/train_s_gen_v2.py", "--launch"]
    assert options["check"] is False
    assert options["timeout"] <= plan.max_wall_hours * 3600
    assert status["state"] == "ready_to_run"
    assert status["progress"]["completed_games"] == 100


def test_supervisor_stops_at_completion_bound_not_schedule_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = replace(
        _plan(tmp_path),
        max_games=5000,
        completion_game_bound=100,
        common_trainer_args=(
            "--experiment-id",
            "dev-v4-managed-baseline-v1",
            "--max-games",
            "5000",
            "--heuristic-node-budget",
            "500000",
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--no-s1a-warmstart",
            "--no-imitation-mix",
        ),
    )
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approve one bounded comparison segment.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    monkeypatch.setattr(managed, "_git_state", lambda _root: (plan.git_commit, False))
    checkpoint = Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
    monkeypatch.setattr(
        managed,
        "_inspect_completed_segment",
        lambda *_args, **_kwargs: (100, checkpoint),
    )
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    status = run_next_segment(
        plan_path,
        authorization_path,
        runner=runner,
        python_executable="python",
    )
    second_status = run_next_segment(
        plan_path,
        authorization_path,
        runner=runner,
        python_executable="python",
    )

    assert len(calls) == 1
    assert calls[0][calls[0].index("--segment-stop-game") + 1] == "100"
    assert calls[0][calls[0].index("--max-games") + 1] == "5000"
    assert status["state"] == "completed"
    assert status["progress"] == {
        "completed_games": 100,
        "max_games": 100,
        "schedule_max_games": 5000,
        "completed_segments": 1,
        "elapsed_hours": 0.0,
        "max_wall_hours": 12.0,
    }
    assert second_status["state"] == "completed"


def test_supervisor_requires_passing_policy_health_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _policy_health_plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    monkeypatch.setattr(managed, "_git_state", lambda _root: (plan.git_commit, False))
    checkpoint = Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
    monkeypatch.setattr(
        managed,
        "_inspect_completed_segment",
        lambda *_args, **_kwargs: (100, checkpoint),
    )
    trainer_calls: list[list[str]] = []
    health_calls: list[list[str]] = []

    def trainer_runner(command, **_kwargs):
        trainer_calls.append(command)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    def health_runner(command, **_kwargs):
        health_calls.append(command)
        _publish_health_report(command, plan, checkpoint)
        return subprocess.CompletedProcess(command, 0)

    status = run_next_segment(
        plan_path,
        authorization_path,
        runner=trainer_runner,
        health_runner=health_runner,
        python_executable="python",
    )

    assert len(trainer_calls) == 1
    assert len(health_calls) == 1
    assert status["progress"]["completed_games"] == 100
    completed = managed._completed_segment_events(plan)[-1]
    assert completed.details["policy_health"]["passed"] is True
    assert (
        completed.details["policy_health"]["metrics"]
        ["candidate_value_preserving_rate"]
        == 0.75
    )


def test_policy_health_threshold_failure_quarantines_and_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _policy_health_plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    monkeypatch.setattr(managed, "_git_state", lambda _root: (plan.git_commit, False))
    checkpoint = Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
    monkeypatch.setattr(
        managed,
        "_inspect_completed_segment",
        lambda *_args, **_kwargs: (100, checkpoint),
    )
    trainer_calls = 0

    def trainer_runner(command, **_kwargs):
        nonlocal trainer_calls
        trainer_calls += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    def health_runner(command, **_kwargs):
        _publish_health_report(
            command,
            plan,
            checkpoint,
            candidate_rate=0.49,
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(ManagedContractError, match="threshold"):
        run_next_segment(
            plan_path,
            authorization_path,
            runner=trainer_runner,
            health_runner=health_runner,
            python_executable="python",
        )

    events = managed.load_run_events(
        Path(plan.control_dir) / managed.CONTROLLER_LEDGER_NAME
    )
    assert events[-1].status == "quarantined"
    assert events[-1].reason_code == "policy_health_threshold_failed"
    assert events[-1].details["policy_health"]["failures"] == (
        "candidate_value_preserving_rate",
    )
    with pytest.raises(ManagedContractError, match="Agent review"):
        run_next_segment(
            plan_path,
            authorization_path,
            runner=trainer_runner,
            health_runner=health_runner,
            python_executable="python",
        )
    assert trainer_calls == 1


@pytest.mark.parametrize("report_mode", ("missing", "malformed"))
def test_invalid_policy_health_report_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_mode: str,
) -> None:
    plan = _policy_health_plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    monkeypatch.setattr(managed, "_git_state", lambda _root: (plan.git_commit, False))
    checkpoint = Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
    monkeypatch.setattr(
        managed,
        "_inspect_completed_segment",
        lambda *_args, **_kwargs: (100, checkpoint),
    )

    def trainer_runner(command, **_kwargs):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    def health_runner(command, **_kwargs):
        if report_mode == "malformed":
            output = Path(command[command.index("--output") + 1])
            output.write_text("{not-json\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(ManagedContractError, match="evidence is invalid"):
        run_next_segment(
            plan_path,
            authorization_path,
            runner=trainer_runner,
            health_runner=health_runner,
            python_executable="python",
        )

    events = managed.load_run_events(
        Path(plan.control_dir) / managed.CONTROLLER_LEDGER_NAME
    )
    assert events[-1].reason_code == "policy_health_audit_failed"
    assert managed_status(plan_path, authorization_path)["state"] == (
        "stopped_for_agent_review"
    )


def test_supervisor_never_removes_a_lock_it_does_not_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    monkeypatch.setattr(managed, "_git_state", lambda _root: (plan.git_commit, False))
    lock = Path(plan.control_dir) / managed.CONTROLLER_LOCK_NAME
    lock.write_text("pid=123\n", encoding="ascii")

    with pytest.raises(ManagedContractError, match="another supervisor"):
        run_next_segment(
            plan_path,
            authorization_path,
            runner=lambda *_args, **_kwargs: pytest.fail("runner was called"),
            python_executable="python",
        )

    assert lock.read_text(encoding="ascii") == "pid=123\n"


def test_stale_lock_is_cleared_when_pid_is_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    Path(plan.control_dir).mkdir(parents=True, exist_ok=True)
    lock = Path(plan.control_dir) / managed.CONTROLLER_LOCK_NAME
    lock.write_text("pid=424242\n", encoding="ascii")
    monkeypatch.setattr(managed, "_pid_is_running", lambda _pid: False)

    assert managed._clear_stale_controller_lock(plan) is True
    assert not lock.exists()


@pytest.mark.parametrize(
    "reason_code", ("host_reboot", "verified_implementation_repair")
)
def test_verify_accepts_pending_recovery_resume(
    tmp_path: Path,
    reason_code: str,
) -> None:
    plan = _plan(tmp_path)
    plan_path = tmp_path / "control" / "plan.json"
    authorization_path = tmp_path / "control" / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    recovery = Path(plan.control_dir) / "recovery" / "segment-0002.pt"
    recovery.parent.mkdir(parents=True, exist_ok=True)
    recovery.write_bytes(b"placeholder")
    managed._append_controller_event(
        plan,
        status="completed",
        event_type="managed_segment_completed",
        details={
            "segment_index": 1,
            "run_id": "managed-v4-test-segment-0001",
            "completed_games": 100,
            "checkpoint": str(
                Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
            ),
            "elapsed_seconds": 1.0,
        },
    )
    managed._append_controller_event(
        plan,
        status="interrupted",
        event_type="managed_segment_interrupted",
        reason_code=reason_code,
        details={
            "segment_index": 2,
            "recovery_checkpoint": str(recovery.resolve()),
            "parent_run_id": "managed-v4-test-segment-0001",
        },
    )

    verified = verify_managed_launch(
        plan_path,
        authorization_path,
        git_commit=plan.git_commit,
        resume_config_sha256=plan.resume_config_sha256,
        out_dir=Path(plan.control_dir) / "segments" / "segment-0002",
        run_id="managed-v4-test-segment-0002",
        segment_games=plan.segment_games,
        segment_stop_game=200,
        start_mode="exact-resume",
        resume=str(recovery.resolve()),
        parent_run_id="managed-v4-test-segment-0001",
        experiment_id=plan.experiment_id,
    )
    assert verified == plan
    assert managed._pending_recovery_for_segment(plan, 2) is not None


def test_technical_recovery_evidence_is_commit_and_failure_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    managed._append_controller_event(
        plan,
        status="failed",
        event_type="managed_segment_failed",
        reason_code="trainer_exit_nonzero",
        details={"segment_index": 1},
    )
    failed = managed.load_run_events(
        Path(plan.control_dir) / managed.CONTROLLER_LEDGER_NAME
    )[-1]
    repair_commit = "b" * 40
    runtime_commit = "c" * 40
    evidence = tmp_path / "repair.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": managed.TECHNICAL_RECOVERY_EVIDENCE_SCHEMA,
                "plan_sha256": plan.plan_sha256,
                "failed_event_sha256": failed.event_sha256,
                "failed_segment_index": 1,
                "failure_code": "zero-expansion-search",
                "source_commit": plan.git_commit,
                "tested_repair_commit": repair_commit,
                "reproduction": {"result": "reproduced-and-fixed"},
                "verification": ["focused regression passed"],
                "claim_boundary": "technical recovery only",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(managed, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(managed, "_git_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        managed.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )

    result = managed._load_technical_recovery_evidence(
        evidence,
        plan=plan,
        failed_event=failed,
        runtime_commit=runtime_commit,
    )

    assert result["tested_repair_commit"] == repair_commit
    assert result["failure_code"] == "zero-expansion-search"
    assert result["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["failed_event_sha256"] = "d" * 64
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ManagedContractError, match="another failure"):
        managed._load_technical_recovery_evidence(
            evidence,
            plan=plan,
            failed_event=failed,
            runtime_commit=runtime_commit,
        )


def test_recovery_checkpoint_rebinds_database_and_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    source = tmp_path / "source.pt"
    destination = tmp_path / "recovery.pt"
    descriptor = CheckpointDescriptor(
        checkpoint_id="managed-v4-test-segment-0001:checkpoint:1",
        run_id="managed-v4-test-segment-0001",
        experiment_id=plan.experiment_id,
        parent_checkpoint_id=None,
        role="latest",
        save_reason="periodic",
        created_at_utc="2026-07-20T12:10:00Z",
        config_sha256=plan.resume_config_sha256,
        feature_schema_version="test",
        label_schema_version="sector-corrected-v1",
        database_schema_versions={"specialist_db": "sector-corrected-v1"},
        asset_identities={
            "malom_tablebase": "m" * 64,
            "human_db": "h" * 64,
            "specialist_db": "d" * 64,
        },
        implementation={"experiment_digest": "sha256:" + "a" * 64},
    )
    save_checkpoint(
        source,
        descriptor,
        _managed_checkpoint_payload(50),
        previous_copies=0,
    )
    rebound_digest = "sha256:" + "b" * 64
    monkeypatch.setattr(
        managed,
        "_recovery_experiment_digest",
        lambda *_args, **_kwargs: rebound_digest,
    )
    specialist_identity = {
        "sha256": "1" * 64,
        "size": 123,
        "label_version": "sector-corrected-v1",
        "malom_label_count": 27,
    }

    managed._write_recovery_checkpoint(
        source,
        destination,
        specialist_identity=specialist_identity,
        plan=plan,
        runtime_commit="b" * 40,
        recovery_reason="verified-implementation-repair",
    )

    recovered = load_checkpoint(destination, map_location="cpu")
    assert recovered.descriptor.checkpoint_id.endswith(
        ":verified-implementation-repair-recovery"
    )
    assert (
        recovered.descriptor.save_reason
        == "interrupted-verified-implementation-repair-recovery"
    )
    assert recovered.descriptor.asset_identities["specialist_db"] == "1" * 64
    assert recovered.descriptor.implementation["experiment_digest"] == rebound_digest
    assert (
        recovered.payload.data_state["mutable_assets"]["specialist_db"]
        == specialist_identity
    )


def test_early_interruption_recovers_from_completed_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    plan_path = Path(plan.control_dir) / "plan.json"
    authorization_path = Path(plan.control_dir) / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    parent = Path(plan.control_dir) / "segments" / "segment-0001" / "latest.pt"
    descriptor = CheckpointDescriptor(
        checkpoint_id="managed-v4-test-segment-0001:checkpoint:1",
        run_id="managed-v4-test-segment-0001",
        experiment_id=plan.experiment_id,
        parent_checkpoint_id=None,
        role="latest",
        save_reason="segment-complete",
        created_at_utc="2026-07-20T12:10:00Z",
        config_sha256=plan.resume_config_sha256,
        feature_schema_version="test",
        label_schema_version="sector-corrected-v1",
        database_schema_versions={"specialist_db": "sector-corrected-v1"},
        asset_identities={"specialist_db": "d" * 64},
        implementation={"trainer": "test"},
    )
    save_checkpoint(
        parent,
        descriptor,
        _managed_checkpoint_payload(100),
        previous_copies=0,
    )
    managed._append_controller_event(
        plan,
        status="completed",
        event_type="managed_segment_completed",
        details={
            "segment_index": 1,
            "run_id": "managed-v4-test-segment-0001",
            "completed_games": 100,
            "checkpoint": str(parent.resolve()),
            "elapsed_seconds": 1.0,
        },
    )
    interrupted = Path(plan.control_dir) / "segments" / "segment-0002"
    interrupted.mkdir(parents=True)
    (interrupted / "run-events.jsonl").write_text(
        "partial segment evidence\n", encoding="utf-8"
    )
    managed._append_controller_event(
        plan,
        status="running",
        event_type="managed_segment_started",
        details={
            "segment_index": 2,
            "run_id": "managed-v4-test-segment-0002",
            "resume_checkpoint": str(parent.resolve()),
            "recovery": False,
        },
    )
    specialist = tmp_path / "specialist.sqlite"
    specialist.write_bytes(b"specialist evidence")
    monkeypatch.setattr(
        managed,
        "_assert_managed_git_state",
        lambda *_args, **_kwargs: "b" * 40,
    )
    monkeypatch.setattr(
        managed, "_specialist_db_path_for_plan", lambda _plan: specialist
    )
    monkeypatch.setattr(
        managed,
        "_live_specialist_identity",
        lambda _path: {
            "sha256": "1" * 64,
            "size": specialist.stat().st_size,
            "label_version": "sector-corrected-v1",
            "malom_label_count": 0,
            "wal_log_pages": 0,
            "wal_checkpointed_pages": 0,
        },
    )
    observed_source: list[Path] = []

    def write_recovery(source: Path, destination: Path, **_kwargs) -> Path:
        observed_source.append(source.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination.resolve()

    monkeypatch.setattr(managed, "_write_recovery_checkpoint", write_recovery)

    result = recover_interrupted_segment(plan_path, authorization_path)

    assert observed_source == [parent.resolve()]
    assert result["recovery"]["resume_game_count"] == 100
    assert result["recovery"]["checkpoint_origin"] == "previous_completed_boundary"
    assert not interrupted.exists()
    assert Path(result["recovery"]["incomplete_output"]).is_dir()
    events = managed.load_run_events(
        Path(plan.control_dir) / managed.CONTROLLER_LEDGER_NAME
    )
    assert events[-1].reason_code == "host_reboot"


def test_failed_segment_recovery_quarantines_and_records_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    plan_path = Path(plan.control_dir) / "plan.json"
    authorization_path = Path(plan.control_dir) / "authorization.json"
    publish_managed_plan(plan_path, plan)
    authorize_plan(
        plan_path,
        authorization_path,
        authorized_by="product-owner",
        decision_note="Approved.",
        authorized_at_utc="2026-07-20T12:05:00Z",
    )
    segment = Path(plan.control_dir) / "segments" / "segment-0001"
    checkpoint = segment / "latest.pt"
    descriptor = CheckpointDescriptor(
        checkpoint_id="managed-v4-test-segment-0001:checkpoint:1",
        run_id="managed-v4-test-segment-0001",
        experiment_id=plan.experiment_id,
        parent_checkpoint_id=None,
        role="latest",
        save_reason="periodic",
        created_at_utc="2026-07-20T12:10:00Z",
        config_sha256=plan.resume_config_sha256,
        feature_schema_version="test",
        label_schema_version="sector-corrected-v1",
        database_schema_versions={"specialist_db": "sector-corrected-v1"},
        asset_identities={"specialist_db": "d" * 64},
        implementation={"trainer": "test"},
    )
    save_checkpoint(
        checkpoint,
        descriptor,
        _managed_checkpoint_payload(50),
        previous_copies=0,
    )
    specialist = tmp_path / "specialist.sqlite"
    specialist.write_bytes(b"specialist evidence")
    managed._append_controller_event(
        plan,
        status="failed",
        event_type="managed_segment_failed",
        reason_code="trainer_exit_nonzero",
        details={"segment_index": 1},
    )
    runtime_commit = "b" * 40
    repair = {
        "path": str((tmp_path / "repair.json").resolve()),
        "sha256": "e" * 64,
        "failure_code": "zero-expansion-search",
        "tested_repair_commit": "f" * 40,
    }
    monkeypatch.setattr(
        managed,
        "_assert_managed_git_state",
        lambda *_args, **_kwargs: runtime_commit,
    )
    monkeypatch.setattr(
        managed,
        "_load_technical_recovery_evidence",
        lambda *_args, **_kwargs: repair,
    )
    monkeypatch.setattr(
        managed, "_specialist_db_path_for_plan", lambda _plan: specialist
    )
    monkeypatch.setattr(
        managed,
        "_live_specialist_identity",
        lambda _path: {
            "sha256": "1" * 64,
            "size": specialist.stat().st_size,
            "label_version": "sector-corrected-v1",
            "malom_label_count": 0,
            "wal_log_pages": 0,
            "wal_checkpointed_pages": 0,
        },
    )

    def write_recovery(source: Path, destination: Path, **_kwargs) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination.resolve()

    monkeypatch.setattr(managed, "_write_recovery_checkpoint", write_recovery)

    result = recover_failed_segment(
        plan_path,
        authorization_path,
        technical_evidence_path=tmp_path / "repair.json",
    )

    recovery = Path(result["recovery"]["recovery_checkpoint"])
    assert recovery.is_file()
    assert not segment.exists()
    assert result["recovery"]["resume_game_count"] == 50
    assert result["recovery"]["runtime_commit"] == runtime_commit
    assert result["recovery"]["technical_repair"] == repair
    events = managed.load_run_events(
        Path(plan.control_dir) / managed.CONTROLLER_LEDGER_NAME
    )
    assert events[-1].reason_code == "verified_implementation_repair"
