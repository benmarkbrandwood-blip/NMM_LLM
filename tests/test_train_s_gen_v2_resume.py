"""Regression tests for Generalist v2 checkpoint selection."""

from __future__ import annotations

from argparse import Namespace

from scripts import train_s_gen_v2 as trainer


def _args(*, out_dir, resume="", auto_resume_best=True) -> Namespace:
    return Namespace(
        resume=resume,
        auto_resume_best=auto_resume_best,
        out_dir=str(out_dir),
    )


def test_explicit_resume_takes_precedence_over_output_best(tmp_path):
    explicit = tmp_path / "explicit.pt"
    explicit.touch()
    out_dir = tmp_path / "configured-output"
    out_dir.mkdir()
    (out_dir / "best.pt").touch()

    path, source = trainer._choose_resume_path(
        _args(out_dir=out_dir, resume=str(explicit))
    )

    assert path == explicit
    assert source == "explicit_resume"


def test_auto_resume_uses_best_from_configured_output_directory(
    tmp_path, monkeypatch
):
    legacy_best = (
        tmp_path
        / "learned_ai"
        / "checkpoints"
        / "scaffolded"
        / "s_gen_v2"
        / "best.pt"
    )
    legacy_best.parent.mkdir(parents=True)
    legacy_best.touch()
    monkeypatch.setattr(trainer, "_ROOT", tmp_path)

    out_dir = tmp_path / "configured-output"
    out_dir.mkdir()
    configured_best = out_dir / "best.pt"
    configured_best.touch()

    path, source = trainer._choose_resume_path(_args(out_dir=out_dir))

    assert path == configured_best
    assert source == "s_gen_v2_best"


def test_auto_resume_does_not_fall_back_to_legacy_output_directory(
    tmp_path, monkeypatch
):
    legacy_best = (
        tmp_path
        / "learned_ai"
        / "checkpoints"
        / "scaffolded"
        / "s_gen_v2"
        / "best.pt"
    )
    legacy_best.parent.mkdir(parents=True)
    legacy_best.touch()
    monkeypatch.setattr(trainer, "_ROOT", tmp_path)

    path, source = trainer._choose_resume_path(
        _args(out_dir=tmp_path / "empty-configured-output")
    )

    assert path is None
    assert source == "scratch"
