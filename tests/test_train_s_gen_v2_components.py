"""Focused tests for fail-closed Generalist runtime components."""

from __future__ import annotations

import pytest

from scripts import train_s_gen_v2 as trainer


def test_explicitly_disabled_component_never_calls_loader(tmp_path) -> None:
    def unexpected_loader(path):
        raise AssertionError("disabled component must not be loaded")

    result = trainer._load_runtime_component(
        label="Sentinel",
        path=tmp_path / "missing.pt",
        disabled=True,
        expected_kind="file",
        loader=unexpected_loader,
    )

    assert result is None


def test_required_component_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="required HumanDB path"):
        trainer._load_runtime_component(
            label="HumanDB",
            path=tmp_path / "missing.sqlite",
            disabled=False,
            expected_kind="file",
            loader=lambda path: object(),
        )


def test_required_component_rejects_loader_failure(tmp_path) -> None:
    path = tmp_path / "value.pt"
    path.touch()

    with pytest.raises(RuntimeError, match="ValueNet load failed"):
        trainer._load_runtime_component(
            label="ValueNet",
            path=path,
            disabled=False,
            expected_kind="file",
            loader=lambda path: (_ for _ in ()).throw(ValueError("bad weights")),
        )


def test_required_component_rejects_unready_instance(tmp_path) -> None:
    path = tmp_path / "malom"
    path.mkdir()

    with pytest.raises(RuntimeError, match="Malom DB is not ready"):
        trainer._load_runtime_component(
            label="Malom DB",
            path=path,
            disabled=False,
            expected_kind="directory",
            loader=lambda path: object(),
            ready=lambda component: False,
        )


def test_specialist_evidence_write_failure_propagates() -> None:
    class BrokenSpecialistDB:
        def record_game(self, *args, **kwargs):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        trainer._persist_rollout_evidence(
            specialist_db=BrokenSpecialistDB(),
            malom_db=None,
            learner_boards=[object()],
            learner_result_boards=[],
            outcome=trainer.LOSS_REWARD,
            learner_moves_notation=["d2"],
            learner_color="W",
        )
