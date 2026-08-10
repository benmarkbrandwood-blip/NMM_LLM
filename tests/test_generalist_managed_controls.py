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


def test_parser_exposes_explicit_sanmill_resource_schedule() -> None:
    args = trainer._build_argument_parser().parse_args(
        [
            "--referee-engine",
            "sanmill",
            "--opponent-engine",
            "sanmill",
            "--sanmill-runtime",
            "runtime",
            "--sanmill-node-ladder",
            "1000,5000",
            "--sanmill-stage-games",
            "10,20",
            "--sanmill-search-depth",
            "7",
            "--curriculum-advance-policy",
            "fixed-resource",
            "--no-recovery",
        ]
    )

    assert args.referee_engine == "sanmill"
    assert args.opponent_engine == "sanmill"
    assert args.sanmill_node_ladder == (1_000, 5_000)
    assert args.sanmill_stage_games == (10, 20)
    assert args.sanmill_search_depth == 7
    assert args.curriculum_advance_policy == "fixed-resource"
    assert args.no_recovery is True


def test_parser_exposes_explicit_learning_rate_mode() -> None:
    args = trainer._build_argument_parser().parse_args(
        ["--lr-adaptation-mode", "fixed"]
    )

    assert args.lr_adaptation_mode == "fixed"


@pytest.mark.parametrize(
    ("game_count", "expected_level"),
    [
        (0, 1),
        (499, 1),
        (500, 2),
        (999, 2),
        (1_000, 3),
        (1_499, 3),
        (1_500, 4),
        (2_499, 4),
        (2_500, 5),
        (4_999, 5),
    ],
)
def test_fixed_resource_schedule_uses_global_game_count(
    game_count: int,
    expected_level: int,
) -> None:
    stage_games = (500, 500, 500, 1_000, 2_500)

    assert trainer._fixed_resource_level(game_count, stage_games) == expected_level


def test_fixed_resource_schedule_selects_exact_resume_boundary() -> None:
    """A checkpoint cursor at 500 must select level two after process restart."""
    checkpoint_completed_games = 500

    assert trainer._fixed_resource_transition_level(
        1,
        checkpoint_completed_games,
        (500, 500, 500, 1_000, 2_500),
    ) == 2


def test_fixed_resource_schedule_rejects_corrupt_resumed_level() -> None:
    with pytest.raises(RuntimeError, match="persisted curriculum level"):
        trainer._fixed_resource_transition_level(
            5,
            500,
            (500, 500, 500, 1_000, 2_500),
        )


def test_fixed_resource_schedule_never_samples_a_lower_level() -> None:
    class NoSamplingExpected:
        def random(self) -> float:
            raise AssertionError("fixed-resource policy attempted random blending")

        def randint(self, _start: int, _stop: int) -> int:
            raise AssertionError("fixed-resource policy attempted a lower level")

    assert trainer._opponent_resource_level(
        5,
        "fixed-resource",
        NoSamplingExpected(),
    ) == 5


@pytest.mark.parametrize("game_count", [-1, 5_000])
def test_fixed_resource_schedule_rejects_out_of_range_game_count(
    game_count: int,
) -> None:
    with pytest.raises(ValueError, match="outside the resource schedule"):
        trainer._fixed_resource_level(
            game_count,
            (500, 500, 500, 1_000, 2_500),
        )


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
    legal = get_all_legal_moves(board)

    move = ai._choose_rust_scored(
        board,
        max_depth=19,
        moves=legal,
        time_limit_ms=1,
        node_limit=25_000,
    )

    assert move == {"from": None, "to": "a7", "capture": None}
    assert captured["node_limit"] == 25_000
    assert captured["fast_eval"] is True
    assert captured["root_moves"]
    assert len(captured["root_moves"]) == len(legal)
    assert ai._nodes == 25_000


def test_fixed_node_search_returns_move_when_qsearch_exhausts_budget() -> None:
    """Regression for the managed smoke stop on a mid-placement position.

    Placement qsearch is Sanmill-aligned (stand-pat). Fixed-node search must
    still return a legal move for training opponents without burning the
    entire budget inside the first root move.
    """
    from ai.native_core import RUST_AVAILABLE

    if not RUST_AVAILABLE:
        pytest.skip("nmm_core is required for fixed-node search")

    board = BoardState.from_fen_string("...W.....W.B...........B|W|2|2")
    legal = get_all_legal_moves(board)
    assert legal
    ai = trainer._GA(
        color="W",
        difficulty=1,
        override_node_budget=500_000,
    )

    move = ai.choose_move(board)

    assert move in legal
    assert 0 < ai._nodes <= 500_000


def test_fixed_node_search_keeps_mandatory_block_candidate() -> None:
    """Python mandatory-block allowlists must still intersect native scores."""
    from ai.native_core import RUST_AVAILABLE

    if not RUST_AVAILABLE:
        pytest.skip("nmm_core is required for fixed-node search")

    board = BoardState.from_fen_string("BB.W.....W..............|W|2|2")
    ai = trainer._GA(
        color="W",
        difficulty=1,
        override_node_budget=500_000,
    )

    move = ai.choose_move(board)

    assert move == {"from": None, "to": "g7", "capture": None}
    assert 0 < ai._nodes <= 500_000


def test_fixed_node_native_root_restrict_scores_only_allowlist() -> None:
    """Native must score the caller allowlist, not the full legal set."""
    from ai.native_core import RUST_AVAILABLE
    from ai import native_core as nc
    import nmm_core as rc

    if not RUST_AVAILABLE:
        pytest.skip("nmm_core is required for fixed-node search")

    board = BoardState.from_fen_string("BB.W.....W..............|W|2|2")
    white, black, wp, bp, stm = nc.board_to_bits(board)
    # Only g7 (index 2)
    root_moves = [(None, 2, None)]
    nodes, depth, raw = rc.py_search_root_scored(
        white,
        black,
        wp,
        bp,
        stm,
        19,
        1,
        node_limit=25_000,
        root_moves=root_moves,
        threads=1,
        fast_eval=True,
    )
    assert nodes > 0
    assert len(raw) == 1
    assert raw[0][0] is None and raw[0][1] == 2
