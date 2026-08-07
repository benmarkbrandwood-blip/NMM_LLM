"""Focused tests for explicit Generalist opening-forcing controls."""

from __future__ import annotations

import json

import pytest

from learned_ai.training.generalist_preflight import PreflightConfigurationError
from learned_ai.training.generalist_preflight import _probe_opening_sources
from scripts import train_s_gen_v2 as trainer


def test_disabled_opening_forcing_does_not_read_sources(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        ["--preflight", "smoke", "--no-opening-forcing"]
    )

    assert trainer._resolve_opening_lines(args, root=tmp_path) == []


def test_enabled_opening_forcing_requires_explicit_source(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(["--preflight", "smoke"])

    with pytest.raises(PreflightConfigurationError, match="opening_source"):
        trainer.validate_generalist_configuration(args)


def test_strict_opening_loader_rejects_malformed_source(tmp_path) -> None:
    openings = tmp_path / "data" / "openings"
    openings.mkdir(parents=True)
    (openings / "book_openings.json").write_text(
        json.dumps([{"line_moves": "d2 d6"}]),
        encoding="utf-8",
    )
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--opening-source",
            "book",
            "--opening-force-probability",
            "0.5",
        ]
    )

    with pytest.raises(RuntimeError, match="line_moves must be a list"):
        trainer._resolve_opening_lines(args, root=tmp_path)

    report = _probe_opening_sources(args, root=tmp_path)
    assert "line_moves must be a list" in report["error"]


def test_disabled_opening_probe_never_requires_repository_assets(tmp_path) -> None:
    args = trainer._build_argument_parser().parse_args(
        ["--preflight", "smoke", "--no-opening-forcing"]
    )

    report = _probe_opening_sources(args, root=tmp_path)

    assert report == {"enabled": False, "sources": []}


def test_opening_forcing_control_changes_resume_semantics(tmp_path) -> None:
    disabled = trainer._build_argument_parser().parse_args(
        ["--preflight", "smoke", "--no-opening-forcing"]
    )
    enabled = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "smoke",
            "--opening-source",
            "book",
            "--opening-force-probability",
            "0.5",
        ]
    )

    assert trainer.resume_config_sha256(disabled) != trainer.resume_config_sha256(
        enabled
    )
