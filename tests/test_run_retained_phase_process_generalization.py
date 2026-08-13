"""Tests for the fail-closed retained phase-process controller."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_retained_phase_process_generalization as runner
from learned_ai.evaluation.retained_phase_process_generalization import (
    EXPECTED_GAMES,
    RetainedPhaseProcessError,
)


def _plan() -> dict:
    return {
        "diagnostic_id": "phase-process-test",
        "plan_identity": "p" * 64,
        "workload": {
            "games": EXPECTED_GAMES,
            "unique_starts": 39,
            "max_active_hours": 2.0,
        },
        "claim_boundary": {"held_out": False},
    }


def _paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        plan=tmp_path / "plan.json",
        authorization=tmp_path / "authorization.json",
    )


def _patch_technical_gates(monkeypatch) -> None:
    repository = {
        "branch": "dev",
        "head": "h" * 40,
        "tree": "t" * 40,
        "upstream_commit": "h" * 40,
        "published": True,
        "tracked_worktree": "clean",
        "implementation_commit_is_ancestor": True,
        "plan_commit": "c" * 40,
        "post_plan_runtime_files_unchanged": True,
        "post_plan_status_documents_only": True,
    }
    monkeypatch.setattr(
        runner,
        "_repository_record",
        lambda plan, paths: dict(repository),
    )
    monkeypatch.setattr(
        runner,
        "_output_record",
        lambda paths, resume: {"mode": "resume" if resume else "fresh"},
    )
    monkeypatch.setattr(
        runner,
        "_input_record",
        lambda plan, paths: {"snapshot_identity": "s" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_corpus_record",
        lambda plan, paths: (
            {"records": 39, "games": EXPECTED_GAMES},
            [{} for _ in range(39)],
        ),
    )
    monkeypatch.setattr(
        runner,
        "_candidate_record",
        lambda plan, paths: {"v3": "ok", "v4": "ok"},
    )
    monkeypatch.setattr(
        runner,
        "_sanmill_record",
        lambda plan, paths: ({"runtime": "ok"}, object()),
    )
    monkeypatch.setattr(runner, "_competing_processes", lambda: [])
    monkeypatch.setattr(runner, "_test_record", lambda: {"tests": "pass"})


def _patch_repository_commands(monkeypatch, *, changed: str) -> None:
    monkeypatch.setattr(
        runner,
        "_base_repository_record",
        lambda plan, paths: {
            "branch": "dev",
            "head": "h" * 40,
            "tree": "t" * 40,
            "upstream_commit": "h" * 40,
            "published": True,
            "tracked_worktree": "clean",
            "implementation_commit_is_ancestor": True,
        },
    )
    monkeypatch.setattr(
        runner,
        "_git",
        lambda *arguments: (
            "c" * 40 if arguments[0] == "log" else changed
        ),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )


def test_repository_record_allows_only_post_plan_status_documents(
    monkeypatch,
) -> None:
    changed = "\n".join(sorted(runner.POST_PLAN_STATUS_DOCUMENTS))
    _patch_repository_commands(monkeypatch, changed=changed)
    paths = SimpleNamespace(plan=runner.DEFAULT_PLAN)

    observed = runner._repository_record(_plan(), paths)

    assert observed["plan_commit"] == "c" * 40
    assert observed["post_plan_runtime_files_unchanged"] is True
    assert observed["post_plan_status_documents_only"] is True


def test_repository_record_rejects_post_plan_runtime_change(monkeypatch) -> None:
    _patch_repository_commands(
        monkeypatch,
        changed="scripts/run_retained_phase_process_generalization.py",
    )
    paths = SimpleNamespace(plan=runner.DEFAULT_PLAN)

    with pytest.raises(
        RetainedPhaseProcessError,
        match="runtime-affecting files changed after the frozen plan",
    ):
        runner._repository_record(_plan(), paths)


def test_source_readiness_ignores_post_plan_status_commit_identity(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_technical_gates(monkeypatch)
    plan = _plan()
    paths = _paths(tmp_path)
    paths.plan.write_bytes(b"{}")
    first = runner.build_readiness_report(
        plan,
        paths,
        resume=False,
        run_tests=True,
        audit_histories=False,
    )

    monkeypatch.setattr(
        runner,
        "_repository_record",
        lambda plan, paths: {
            "branch": "dev",
            "head": "n" * 40,
            "tree": "u" * 40,
            "upstream_commit": "n" * 40,
            "published": True,
            "tracked_worktree": "clean",
            "implementation_commit_is_ancestor": True,
            "plan_commit": "c" * 40,
            "post_plan_runtime_files_unchanged": True,
            "post_plan_status_documents_only": True,
        },
    )
    second = runner.build_readiness_report(
        plan,
        paths,
        resume=False,
        run_tests=True,
        audit_histories=False,
    )

    assert second["source_readiness_identity"] == first[
        "source_readiness_identity"
    ]
    assert second["readiness_identity"] != first["readiness_identity"]


def test_source_readiness_is_stable_when_authorization_is_added(
    tmp_path,
    monkeypatch,
) -> None:
    _patch_technical_gates(monkeypatch)
    plan = _plan()
    paths = _paths(tmp_path)
    paths.plan.write_bytes(b"{}")
    absent = runner.build_readiness_report(
        plan,
        paths,
        resume=False,
        run_tests=True,
        audit_histories=False,
    )
    assert absent["ready"] is False
    assert absent["verdict"] == "needs_decision"
    assert absent["gates"][-1]["gate"] == "authorization"
    assert absent["gates"][-1]["result"] == "fail"

    monkeypatch.setattr(
        runner,
        "_load_authorization",
        lambda plan, paths, expected_source_readiness_identity: {
            "authorization_identity": "a" * 64,
            "source_readiness_identity": expected_source_readiness_identity,
        },
    )
    authorized = runner.build_readiness_report(
        plan,
        paths,
        resume=False,
        run_tests=True,
        audit_histories=False,
    )
    assert authorized["ready"] is True
    assert authorized["verdict"] == "ready_for_evaluation"
    assert authorized["source_readiness_identity"] == absent[
        "source_readiness_identity"
    ]


def test_authorization_builder_binds_bounds_and_all_prohibitions(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b"{}")
    authorization = runner.build_authorization(
        plan=_plan(),
        plan_path=plan_path,
        plan_commit="c" * 40,
        source_readiness_identity="1" * 64,
        authority_text_sha256="a" * 64,
    )
    assert authorization["source_readiness_identity"] == "1" * 64
    assert authorization["grant"] == {
        "plan_identity": "p" * 64,
        "plan_file_sha256": hashlib.sha256(b"{}").hexdigest(),
        "plan_commit": "c" * 40,
        "games": 156,
        "max_active_hours": 2.0,
        "host_interruption_exact_resume_only": True,
        "same_spec_exact_resume": True,
        "automatic_retry": False,
        "semantic_failure_recovery": False,
        "expansion": False,
        "training": False,
        "updates": False,
        "held_out_strength_claim": False,
        "promotion": False,
        "publication": False,
        "release": False,
    }


def test_launch_readiness_requires_every_gate_once() -> None:
    names = {
        "repository",
        "plan",
        "outputs",
        "inputs",
        "corpus",
        "candidates",
        "sanmill",
        "strict_history_replay",
        "process_ownership",
        "tests",
        "authorization",
    }
    readiness = {
        "ready": True,
        "verdict": "ready_for_evaluation",
        "gates": [{"gate": name, "result": "pass"} for name in names],
    }
    runner.require_launch_ready(readiness, resume=False)
    readiness["gates"].pop()
    with pytest.raises(RetainedPhaseProcessError, match="skipped or duplicated"):
        runner.require_launch_ready(readiness, resume=False)


def test_resume_readiness_adds_exact_continuity_gate() -> None:
    names = {
        "repository",
        "plan",
        "outputs",
        "inputs",
        "corpus",
        "candidates",
        "sanmill",
        "strict_history_replay",
        "process_ownership",
        "tests",
        "authorization",
        "resume_continuity",
    }
    readiness = {
        "ready": True,
        "verdict": "ready_for_evaluation",
        "gates": [{"gate": name, "result": "pass"} for name in names],
    }
    runner.require_launch_ready(readiness, resume=True)


def test_cli_cannot_launch_without_flag_or_with_skipped_gates(capsys) -> None:
    assert runner.main(["run"]) == 2
    assert "explicit --launch" in capsys.readouterr().err
    assert runner.main(["--skip-tests", "run", "--launch"]) == 2
    assert "cannot skip tests" in capsys.readouterr().err
    assert runner.main(["--skip-history-audit", "run", "--launch"]) == 2
    assert "cannot skip tests" in capsys.readouterr().err
