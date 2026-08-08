"""Focused evidence for the fresh Sanmill-refereed training contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.evaluation.sanmill_uci import SanmillBridgeError
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    TRAINING_SANMILL_COMMIT,
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
    probe_sanmill_training_runtime,
)
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from scripts import train_s_gen_v2 as trainer


_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PATHS = _ROOT / "data" / "training_paths.local.json"


def _training_checkout() -> Path:
    if not _LOCAL_PATHS.is_file():
        pytest.skip("requires the ignored Sanmill training path registry")
    config = json.loads(_LOCAL_PATHS.read_text(encoding="utf-8"))
    value = config.get("sanmill_training_checkout")
    if not isinstance(value, str) or not value:
        pytest.skip("sanmill_training_checkout is not configured")
    checkout = Path(value)
    if not checkout.is_absolute():
        checkout = _ROOT / checkout
    return checkout.resolve(strict=False)


def _placement(board: BoardState, to: str, capture: str | None = None) -> dict:
    candidates = [
        move
        for move in get_all_legal_moves(board)
        if move.get("from") is None
        and move.get("to") == to
        and move.get("capture") == capture
    ]
    assert len(candidates) == 1
    return candidates[0]


def test_training_runtime_identity_and_cross_process_search_are_fixed() -> None:
    report = probe_sanmill_training_runtime(
        _training_checkout(),
        node_budget=1_000,
        seed=42,
    )

    assert report["commit"] == TRAINING_SANMILL_COMMIT
    assert report["strict_referee"]["semanticDigest"] == (
        TRAINING_REFEREE_SEMANTIC_DIGEST
    )
    assert report["probe"]["fresh_processes"] == 2
    assert report["probe"]["deterministic"] is True
    assert len(report["identity"]) == 64
    assert report["strict_options"]["StrictFailurePolicy"] == "true"
    assert report["strict_options"]["Shuffling"] == "false"
    assert report["strict_options"]["UsePerfectDatabase"] == "false"


def test_training_referee_replays_complete_logical_turn_with_capture() -> None:
    installation = inspect_sanmill_training_installation(_training_checkout())
    board = BoardState.new_game()
    line = (
        ("d6", None),
        ("d2", None),
        ("f4", None),
        ("b4", None),
        ("g7", None),
        ("d7", None),
        ("g4", None),
        ("e4", None),
        ("g1", "e4"),
    )

    with SanmillTrainingGame(installation, seed=42) as game:
        for logical_ply, (to, capture) in enumerate(line, start=1):
            move = _placement(board, to, capture)
            turn = game.apply_nmm_move(board, move)
            board = board.apply_move(move)
            game.assert_current_board(board)
            assert turn.state.logical_ply_count == logical_ply

        assert turn.actions == ("g1", "xe4")
        assert turn.state.action_token_count == 10
        assert turn.state.logical_plies_by_side == (5, 4)
        assert turn.state.strict_referee_identity is not None
        assert turn.state.strict_referee_identity.semantic_digest == (
            TRAINING_REFEREE_SEMANTIC_DIGEST
        )


def test_training_referee_rejects_illegal_local_move_without_fallback() -> None:
    installation = inspect_sanmill_training_installation(_training_checkout())
    with SanmillTrainingGame(installation, seed=42) as game:
        with pytest.raises(SanmillBridgeError, match="illegal training move"):
            game.apply_nmm_move(
                BoardState.new_game(),
                {"from": None, "to": "not-a-point", "capture": None},
            )


def test_training_rollout_uses_sanmill_for_search_and_every_referee_ply() -> None:
    installation = inspect_sanmill_training_installation(_training_checkout())
    model = ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        policy_hidden=(256, 128),
    )

    with SanmillTrainingGame(installation, seed=42) as game:
        opponent = trainer.SanmillTrainingOpponent(game, node_budget=1_000)
        result = trainer._rollout(
            model=model,
            device=torch.device("cpu"),
            start_board=BoardState.new_game(),
            learner_color="B",
            opponent=opponent,
            opp_color="W",
            sentinel=None,
            value_net=None,
            temperature=0.9,
            max_ply=8,
            record_branches=False,
            branch_every=0,
            retry_ply=0,
            torch_generator=trainer._game_torch_generator(42),
            sanmill_game=game,
        )

        assert result.termination_reason == "max-ply-truncation"
        assert result.opponent_search_calls == 4
        assert result.opponent_search_nodes > 0
        assert game.state.logical_ply_count == 8
        assert game.state.action_token_count == 8
