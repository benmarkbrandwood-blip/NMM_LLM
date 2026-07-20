"""Tests for Generalist v2 component-path configuration."""

from __future__ import annotations

from argparse import Namespace

from scripts import train_s_gen_v2 as trainer


def test_disable_flags_override_configured_legacy_model_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trainer, "_load_settings", lambda _path: {})
    args = Namespace(
        paths_config=None,
        out_dir=str(tmp_path / "out"),
        sentinel=str(tmp_path / "sentinel.pt"),
        malom=str(tmp_path / "malom"),
        value_net=str(tmp_path / "value-net.npz"),
        gap_net=str(tmp_path / "gap-net.npz"),
        human_db=str(tmp_path / "human.sqlite"),
        specialist_db=str(tmp_path / "specialist.sqlite"),
        no_sentinel=True,
        no_value_net=True,
        no_gap_net=True,
    )

    trainer._configure_paths(args)

    assert args.sentinel == ""
    assert args.value_net == ""
    assert args.gap_net == ""
