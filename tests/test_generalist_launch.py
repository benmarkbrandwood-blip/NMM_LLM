"""Tests for the contract-backed Generalist v2 CLI launch lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.training.generalist_preflight import LoadedTrainingSettings
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


def test_launch_closes_runtime_specialist_db_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_launch_mocks(monkeypatch)
    database_path = tmp_path / "runtime-specialist.sqlite"
    retained: list[SpecialistDB] = []

    def run(args, **_kwargs) -> None:
        database = SpecialistDB(database_path)
        database.checkpoint_identity()
        retained.append(database)
        setattr(args, "_runtime_specialist_db", database)
        assert Path(f"{database_path}-wal").exists()
        assert Path(f"{database_path}-shm").exists()

    monkeypatch.setattr(trainer, "run", run)

    assert trainer.main(_launch_arguments()) == 0
    assert retained
    assert not Path(f"{database_path}-wal").exists()
    assert not Path(f"{database_path}-shm").exists()


def test_launch_closes_runtime_specialist_db_after_training_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_launch_mocks(monkeypatch)
    database_path = tmp_path / "runtime-specialist.sqlite"
    retained: list[SpecialistDB] = []

    def run(args, **_kwargs) -> None:
        database = SpecialistDB(database_path)
        database.checkpoint_identity()
        retained.append(database)
        setattr(args, "_runtime_specialist_db", database)
        raise RuntimeError("simulated failure after database open")

    monkeypatch.setattr(trainer, "run", run)

    with pytest.raises(RuntimeError, match="simulated failure"):
        trainer.main(_launch_arguments())
    assert retained
    assert not Path(f"{database_path}-wal").exists()
    assert not Path(f"{database_path}-shm").exists()


def test_launch_records_failed_before_propagating_training_error(
    monkeypatch,
) -> None:
    statuses = _install_launch_mocks(
        monkeypatch, run_effect=RuntimeError("simulated failure")
    )

    with pytest.raises(RuntimeError, match="simulated failure"):
        trainer.main(_launch_arguments())

    assert statuses == ["running", "failed"]


def test_long_run_preflight_accepts_ready_for_long_run(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(trainer, "_configure_paths", lambda _args: {})
    monkeypatch.setattr(
        trainer,
        "run_generalist_preflight",
        lambda *_args, **_kwargs: {
            "mode": "long-run",
            "verdict": "ready_for_long_run",
            "checks": {"checkpoint": None},
        },
    )

    exit_code = trainer.main(
        [
            "--preflight",
            "long-run",
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--max-games",
            "1",
            "--batch-games",
            "1",
        ]
    )

    assert exit_code == 0
    assert '"verdict": "ready_for_long_run"' in capsys.readouterr().out


def test_preflight_reserves_stdout_for_one_json_document(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        trainer,
        "load_training_settings",
        lambda *_args, **_kwargs: LoadedTrainingSettings(
            {}, {}, Path("machine-local.json")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "configure_generalist_paths",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        trainer,
        "run_generalist_preflight",
        lambda *_args, **_kwargs: {
            "mode": "long-run",
            "verdict": "ready_for_long_run",
            "checks": {"checkpoint": None},
        },
    )

    exit_code = trainer.main(
        [
            "--preflight",
            "long-run",
            "--no-sentinel",
            "--no-value-net",
            "--no-gap-net",
            "--max-games",
            "1",
            "--batch-games",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["verdict"] == "ready_for_long_run"
    assert "[s_gen_v2] Path config: machine-local.json" in captured.err
