"""Regression tests for explicit controls used by managed Generalist runs."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import train_s_gen_v2 as trainer
from game.board import BoardState
from game.rules import get_all_legal_moves


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


def test_parser_exposes_fixed_heuristic_node_budget() -> None:
    args = trainer._build_argument_parser().parse_args(
        ["--heuristic-node-budget", "25000"]
    )

    assert args.heuristic_node_budget == 25_000


def test_game_ai_rejects_mixed_time_and_node_budgets() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer._GA(
            override_time_budget=1.0,
            override_node_budget=25_000,
        )


def test_game_ai_passes_fixed_node_budget_to_native_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai import native_core

    captured: dict[str, object] = {}

    class FakeTtHandle:
        pass

    def fake_search(*_args, **kwargs):
        captured.update(kwargs)
        return 25_000, 4, [(None, 0, None, 10)]

    fake_module = SimpleNamespace(
        RustTtHandle=FakeTtHandle,
        py_search_root_scored=fake_search,
    )
    monkeypatch.setattr(native_core, "RUST_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "nmm_core", fake_module)
    ai = trainer._GA(
        color="W",
        difficulty=3,
        override_node_budget=25_000,
    )
    board = BoardState.new_game()

    move = ai._choose_rust_scored(
        board,
        max_depth=19,
        moves=get_all_legal_moves(board),
        time_limit_ms=1,
        node_limit=25_000,
    )

    assert move == {"from": None, "to": "a7", "capture": None}
    assert captured["node_limit"] == 25_000
    assert ai._nodes == 25_000
