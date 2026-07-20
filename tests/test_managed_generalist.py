"""Tests for the bounded, product-authorized Generalist supervisor."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

from learned_ai.training import managed_generalist as managed
from learned_ai.training.managed_generalist import (
    ManagedContractError,
    ManagedPlan,
    authorize_plan,
    build_segment_command,
    load_managed_plan,
    managed_status,
    publish_managed_plan,
    run_next_segment,
    verify_managed_launch,
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
        python_executable="python",
    )
    assert first[first.index("--start-mode") + 1] == "fresh"
    assert "--resume" not in first

    second = build_segment_command(
        plan,
        plan_path=plan_path,
        authorization_path=authorization_path,
        segment_index=2,
        previous_checkpoint=tmp_path / "segment-0001" / "latest.pt",
        previous_run_id="managed-v4-test-segment-0001",
        python_executable="python",
    )
    assert second[second.index("--start-mode") + 1] == "exact-resume"
    assert second[second.index("--parent-run-id") + 1] == (
        "managed-v4-test-segment-0001"
    )


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


def test_prepare_common_args_can_isolate_specialist_db(tmp_path: Path) -> None:
    from argparse import Namespace
    from scripts.manage_generalist_run import _common_trainer_args

    specialist_db = tmp_path / "specialist_db.smoke.sqlite"
    args = Namespace(
        experiment_id="dev-v4-managed-smoke-rl-update-v1",
        max_games=16,
        heuristic_node_budget=500_000,
        specialist_db=str(specialist_db),
    )
    common = _common_trainer_args(args, tmp_path / "training_paths.local.json")
    assert "--specialist-db" in common
    assert common[common.index("--specialist-db") + 1] == str(specialist_db.resolve())
    assert "--no-imitation-mix" in common
    assert common[common.index("--heuristic-node-budget") + 1] == "500000"
