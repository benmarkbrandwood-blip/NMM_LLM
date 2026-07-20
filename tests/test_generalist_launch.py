"""Tests for the contract-backed Generalist v2 CLI launch lifecycle."""

from __future__ import annotations

import pytest

from scripts import train_s_gen_v2 as trainer


def _launch_arguments() -> list[str]:
    return [
        "--launch",
        "smoke",
        "--run-id",
        "run-001",
        "--no-sentinel",
        "--no-value-net",
        "--no-gap-net",
        "--max-games",
        "1",
        "--batch-games",
        "1",
    ]


def _install_launch_mocks(monkeypatch, *, run_effect=None) -> list[str]:
    statuses: list[str] = []
    monkeypatch.setattr(trainer, "_configure_paths", lambda _args: {})
    monkeypatch.setattr(
        trainer,
        "run_generalist_preflight",
        lambda *_args, **_kwargs: {
            "mode": "smoke",
            "verdict": "ready_for_smoke",
            "checks": {"checkpoint": None},
        },
    )
    monkeypatch.setattr(
        trainer, "build_generalist_run_manifest", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        trainer, "publish_initial_run_contract", lambda *_args, **_kwargs: None
    )

    def append_event(*_args, **kwargs):
        statuses.append(kwargs["status"])

    monkeypatch.setattr(trainer, "append_run_lifecycle_event", append_event)

    def run(*_args, **_kwargs):
        if run_effect is not None:
            raise run_effect

    monkeypatch.setattr(trainer, "run", run)
    return statuses


def test_launch_records_running_then_completed(monkeypatch, capsys) -> None:
    statuses = _install_launch_mocks(monkeypatch)

    exit_code = trainer.main(_launch_arguments())

    assert exit_code == 0
    assert statuses == ["running", "completed"]
    assert '"verdict": "ready_for_smoke"' in capsys.readouterr().out


def test_launch_records_failed_before_propagating_training_error(
    monkeypatch,
) -> None:
    statuses = _install_launch_mocks(
        monkeypatch, run_effect=RuntimeError("simulated failure")
    )

    with pytest.raises(RuntimeError, match="simulated failure"):
        trainer.main(_launch_arguments())

    assert statuses == ["running", "failed"]
