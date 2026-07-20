"""Regression tests for explicit controls used by managed Generalist runs."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts import train_s_gen_v2 as trainer


def test_no_imitation_mix_never_reads_the_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "imitation.npz"
    dataset.write_bytes(b"must not be opened")
    args = Namespace(no_imitation_mix=True, s1a_data=str(dataset))

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("disabled imitation data was read")

    monkeypatch.setattr(trainer.np, "load", unexpected_load)

    assert trainer._load_imitation_mix_data(args) is None


def test_imitation_mix_load_failure_is_not_silently_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "imitation.npz"
    dataset.write_bytes(b"invalid")
    args = Namespace(no_imitation_mix=False, s1a_data=str(dataset))

    def invalid_load(*_args, **_kwargs):
        raise ValueError("invalid imitation fixture")

    monkeypatch.setattr(trainer.np, "load", invalid_load)

    with pytest.raises(RuntimeError, match="required imitation mixing dataset"):
        trainer._load_imitation_mix_data(args)


def test_parser_exposes_explicit_imitation_mix_disable() -> None:
    args = trainer._build_argument_parser().parse_args(["--no-imitation-mix"])

    assert args.no_imitation_mix is True
