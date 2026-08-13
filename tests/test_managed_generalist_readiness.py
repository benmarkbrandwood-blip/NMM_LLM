"""Focused tests for generic managed Generalist readiness evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.training.managed_generalist import (
    ManagedInitialResume,
    ManagedPlan,
    publish_managed_plan,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation import managed_generalist_readiness as readiness


SOURCE_COMMIT = "a" * 40
REVIEWED_MAIN = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path) -> tuple[ManagedPlan, Path]:
    paths_config = tmp_path / "data/training_paths.local.json"
    paths_config.parent.mkdir(parents=True)
    paths_config.write_text("{}\n", encoding="utf-8")
    database_path = tmp_path / "data/fresh.sqlite"
    database = SpecialistDB(str(database_path))
    database.close()
    plan = ManagedPlan(
        plan_id="managed-readiness-test",
        created_at_utc="2026-08-13T00:00:00Z",
        objective="test one exact readiness bundle",
        experiment_id="dev-v4-readiness-test",
        git_commit=SOURCE_COMMIT,
        control_dir=str((tmp_path / "control").resolve()),
        paths_config=str(paths_config.resolve()),
        paths_config_sha256=_sha256(paths_config),
        resume_config_sha256="c" * 64,
        max_games=500,
        segment_games=100,
        max_wall_hours=2.0,
        common_trainer_args=(
            "--experiment-id",
            "dev-v4-readiness-test",
            "--max-games",
            "500",
            "--specialist-db",
            str(database_path.resolve()),
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
    plan_path = Path(plan.control_dir) / "plan.json"
    publish_managed_plan(plan_path, plan)
    return plan, plan_path


def _preflight(plan: ManagedPlan) -> dict[str, object]:
    out_dir = Path(plan.control_dir) / "segments/segment-0001"
    return {
        "schema_version": "nmm.generalist-preflight.v1",
        "mode": "long-run",
        "verdict": "needs_decision",
        "errors": [],
        "unresolved_decisions": [readiness.PRODUCT_AUTHORIZATION_DECISION],
        "resume_config_sha256": plan.resume_config_sha256,
        "config_sha256": "d" * 64,
        "experimentDigest": "sha256:" + "e" * 64,
        "git": {"commit": SOURCE_COMMIT, "dirty": False},
        "resolved_config": {
            "experiment_id": plan.experiment_id,
            "run_id": f"{plan.plan_id}-segment-0001",
            "segment_games": plan.segment_games,
            "segment_stop_game": 100,
            "start_mode": "fresh",
            "resume": "",
            "auto_resume_best": False,
            "out_dir": str(out_dir),
        },
        "checks": {
            "output": {
                "exists": False,
                "isolated": True,
                "kind": "run_directory",
            },
            "human_db": {
                "identity": "f" * 64,
                "quick_check": "ok",
                "trust": "empirical_frequencies_and_outcomes",
                "malom_columns_policy": "masked_historical_labels",
            },
            "malom": {
                "identity": "1" * 64,
                "component_count": 512,
                "size_bytes": 123,
                "manifest_schema": "nmm.dataset-manifest.v1",
            },
            "ruleset": {
                "id": "nmm-training-core",
                "version": 2,
                "semanticDigest": "sha256:" + "2" * 64,
                "documentDigest": "sha256:" + "3" * 64,
            },
            "sanmill_training": {
                "identity": "4" * 64,
                "commit": "5" * 40,
                "checkout_head": "5" * 40,
                "binary_sha256": "6" * 64,
                "binary_size": 1234,
                "probe": {
                    "observation_sha256": "7" * 64,
                    "first_turn": {
                        "state": {
                            "strict_referee_identity": {
                                "format": "SANMILL-STRICT-REFEREE-RULES/1",
                                "semanticDigest": "sha256:" + "8" * 64,
                            }
                        }
                    },
                },
            },
        },
        "mifSuite": {
            "tag": "mif-suite-1.0",
            "suiteJcsSha256": "sha256:" + "9" * 64,
        },
        "path_sources": {"specialist_db": "cli"},
    }


def test_first_segment_command_is_the_real_fresh_contract(
    tmp_path: Path,
) -> None:
    plan, _plan_path = _plan(tmp_path)

    command = readiness.build_first_segment_preflight_command(
        plan,
        root=tmp_path,
        python_executable="python-under-test",
    )

    assert command[:4] == [
        "python-under-test",
        str(tmp_path / "scripts/train_s_gen_v2.py"),
        "--preflight",
        "long-run",
    ]
    assert command[command.index("--run-id") + 1] == (
        "managed-readiness-test-segment-0001"
    )
    assert command[command.index("--segment-stop-game") + 1] == "100"
    assert command[-2:] == ["--start-mode", "fresh"]
    assert "--resume" not in command
    assert "--auto-resume-best" not in command
    assert "--managed-authorization" not in command


def test_first_segment_command_supports_explicit_initial_resume(
    tmp_path: Path,
) -> None:
    plan, _plan_path = _plan(tmp_path)
    initial = ManagedInitialResume(
        checkpoint_path=str(tmp_path / "parent.pt"),
        checkpoint_sha256="d" * 64,
        checkpoint_id="parent:latest:1",
        checkpoint_role="latest",
        parent_run_id="parent-segment-0001",
        completed_games=50,
    )
    resumed = replace(
        plan,
        initial_resume=initial,
        completion_game_bound=200,
    )

    command = readiness.build_first_segment_preflight_command(
        resumed,
        root=tmp_path,
        python_executable="python-under-test",
    )

    assert command[command.index("--segment-stop-game") + 1] == "150"
    assert command[-6:] == [
        "--start-mode",
        "exact-resume",
        "--resume",
        initial.checkpoint_path,
        "--parent-run-id",
        initial.parent_run_id,
    ]


def test_specialist_db_audit_rejects_any_sidecar(tmp_path: Path) -> None:
    plan, _plan_path = _plan(tmp_path)
    database = readiness._specialist_db_path(plan)
    sidecar = Path(f"{database}-shm")
    sidecar.write_bytes(b"sidecar")

    with pytest.raises(
        readiness.ManagedReadinessError,
        match="SQLite sidecars",
    ):
        readiness.inspect_empty_specialist_db(tmp_path, plan)


def test_preflight_rejects_a_non_authority_decision(tmp_path: Path) -> None:
    plan, _plan_path = _plan(tmp_path)
    report = _preflight(plan)
    report["unresolved_decisions"] = ["choose a learning rate"]

    with pytest.raises(
        readiness.ManagedReadinessError,
        match="beyond launch authority",
    ):
        readiness._validate_preflight(
            report,
            plan=plan,
            source_commit=SOURCE_COMMIT,
        )


def test_generate_persists_raw_report_command_and_canonical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, plan_path = _plan(tmp_path)
    document = tmp_path / "docs/experiment.md"
    document.parent.mkdir()
    document.write_text("# Frozen test\n", encoding="utf-8")
    preflight = _preflight(plan)
    raw_preflight = json.dumps(preflight, indent=2).encode("utf-8") + b"\n"

    monkeypatch.setattr(readiness, "_assert_tracked", lambda *_args: None)
    monkeypatch.setattr(readiness, "_assert_ignored", lambda *_args: None)
    monkeypatch.setattr(
        readiness,
        "inspect_published_source",
        lambda *_args, **_kwargs: {
            "branch": "dev",
            "head": SOURCE_COMMIT,
            "origin_dev": SOURCE_COMMIT,
            "origin_main_reviewed": REVIEWED_MAIN,
            "tracked_worktree_clean": True,
            "git_diff_check": "passed",
        },
    )

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout=raw_preflight,
            stderr=b"one expected warning\n",
        )

    report = readiness.generate_readiness(
        root=tmp_path,
        plan_path=plan_path,
        experiment_document=document,
        reviewed_main=REVIEWED_MAIN,
        python_executable="python-under-test",
        runner=runner,
    )

    control = Path(plan.control_dir)
    command_path = control / "first-segment-preflight-command.json"
    preflight_path = control / "first-segment-preflight.json"
    readiness_path = control / "technical-readiness.json"
    assert preflight_path.read_bytes() == raw_preflight
    assert _sha256(preflight_path) == report["first_segment"][
        "raw_preflight_artifact"
    ]["sha256"]
    command_record = json.loads(command_path.read_text(encoding="utf-8"))
    assert command_record["preflight_argv"] == report["first_segment"][
        "preflight_command"
    ]
    assert command_record["launch_argv"] == report["first_segment"][
        "launch_command"
    ]
    assert "--launch" in command_record["launch_argv"]
    assert "--managed-plan" in command_record["launch_argv"]
    assert "--managed-authorization" in command_record["launch_argv"]
    body = {
        key: value
        for key, value in report.items()
        if key != "readiness_identity"
    }
    assert report["readiness_identity"] == canonical_sha256(body)
    assert readiness_path.read_bytes() == canonical_json_bytes(report)
    verified = readiness.verify_persisted_readiness(
        root=tmp_path,
        readiness_path=readiness_path,
    )
    assert verified == report


def test_verifier_rejects_a_changed_raw_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, plan_path = _plan(tmp_path)
    document = tmp_path / "docs/experiment.md"
    document.parent.mkdir()
    document.write_text("# Frozen test\n", encoding="utf-8")
    raw_preflight = json.dumps(_preflight(plan)).encode("utf-8")
    monkeypatch.setattr(readiness, "_assert_tracked", lambda *_args: None)
    monkeypatch.setattr(readiness, "_assert_ignored", lambda *_args: None)
    monkeypatch.setattr(
        readiness,
        "inspect_published_source",
        lambda *_args, **_kwargs: {
            "branch": "dev",
            "head": SOURCE_COMMIT,
            "origin_dev": SOURCE_COMMIT,
            "origin_main_reviewed": REVIEWED_MAIN,
            "tracked_worktree_clean": True,
            "git_diff_check": "passed",
        },
    )
    readiness.generate_readiness(
        root=tmp_path,
        plan_path=plan_path,
        experiment_document=document,
        reviewed_main=REVIEWED_MAIN,
        runner=lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 2, stdout=raw_preflight, stderr=b""
        ),
    )
    preflight_path = Path(plan.control_dir) / "first-segment-preflight.json"
    preflight_path.write_bytes(preflight_path.read_bytes() + b" ")

    with pytest.raises(
        readiness.ManagedReadinessError,
        match="raw_preflight_artifact SHA-256 differs",
    ):
        readiness.verify_persisted_readiness(
            root=tmp_path,
            readiness_path=Path(plan.control_dir) / "technical-readiness.json",
        )
