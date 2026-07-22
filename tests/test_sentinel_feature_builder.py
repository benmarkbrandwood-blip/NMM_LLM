"""Tests for the current move-level Sentinel feature contract."""

from __future__ import annotations

import numpy as np

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.sentinel.feature_builder import (
    BOARD_CTX_DIM,
    COUNTERFACTUAL_DIM,
    FEATURE_DIM,
    MOVE_DIM,
    board_context_features,
    build_move_features,
    counterfactual_features,
)
from learned_ai.sentinel.labels import dtm_quality


def _opening_position():
    board = BoardState.new_game()
    moves = get_all_legal_moves(board)
    return board, moves


def test_output_shape_matches_move_level_contract():
    board, moves = _opening_position()
    features = build_move_features(
        board,
        moves[0],
        "W",
        {"n_legal": len(moves)},
    )

    assert isinstance(features, np.ndarray)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    assert FEATURE_DIM == 58
    assert BOARD_CTX_DIM == 20
    assert MOVE_DIM == 20
    assert COUNTERFACTUAL_DIM == 18


def test_board_prefix_matches_public_context_builder():
    board, moves = _opening_position()
    features = build_move_features(board, moves[0], "W")

    assert np.array_equal(
        features[:BOARD_CTX_DIM],
        board_context_features(board, "W"),
    )


def test_candidates_share_the_same_board_context():
    board, moves = _opening_position()
    first = build_move_features(board, moves[0], "W")
    second = build_move_features(board, moves[1], "W")

    assert np.array_equal(
        first[:BOARD_CTX_DIM],
        second[:BOARD_CTX_DIM],
    )
    assert not np.array_equal(
        first[BOARD_CTX_DIM:BOARD_CTX_DIM + MOVE_DIM],
        second[BOARD_CTX_DIM:BOARD_CTX_DIM + MOVE_DIM],
    )


def test_opening_move_and_candidate_context_are_encoded():
    board, moves = _opening_position()
    rank = 3
    score = 0.75
    features = build_move_features(
        board,
        moves[0],
        "W",
        {
            "heuristic_rank": rank,
            "n_legal": len(moves),
            "heuristic_score_norm": score,
        },
    )
    move_block = features[BOARD_CTX_DIM:BOARD_CTX_DIM + MOVE_DIM]
    context = features[BOARD_CTX_DIM + MOVE_DIM:]

    assert move_block[2] == 1.0  # placement
    assert context[0] == len(moves) / 24.0
    assert context[6] == rank / len(moves)
    assert context[7] == score
    assert np.all(context[1:6] == 0.0)
    assert np.all(context[8:] == 0.0)


def test_solved_move_context_populates_wdl_and_dtm_slots():
    board, moves = _opening_position()
    selected = moves[0]
    all_moves = [
        {"move": selected, "wdl": "win", "dtm": 3},
        {"move": moves[1], "wdl": "loss", "dtm": 7},
    ]

    context = counterfactual_features(selected, all_moves)

    assert context.shape == (COUNTERFACTUAL_DIM,)
    assert context[1] == 0.5  # fraction winning
    assert context[2] == 0.5  # fraction losing
    assert context[4] == 1.0  # best available WDL
    assert context[5] == 0.0  # worst available WDL
    assert context[8] == 1.0  # winning move available
    assert context[9] == 1.0  # losing move available
    assert context[10] == 1.0  # selected move is a win
    assert context[13] == 1.0  # selected WDL is known
    assert context[14] == 1.0  # solved context is available
    assert context[15] == 0.0  # no better WDL move exists
    assert context[16] == dtm_quality("win", 3)
    assert context[17] == dtm_quality("win", 3)


def test_finite_features_for_every_opening_candidate():
    board, moves = _opening_position()
    rows = np.stack(
        [
            build_move_features(
                board,
                move,
                "W",
                {
                    "heuristic_rank": rank,
                    "n_legal": len(moves),
                    "heuristic_score_norm": rank / max(len(moves) - 1, 1),
                },
            )
            for rank, move in enumerate(moves)
        ]
    )

    assert rows.shape == (len(moves), FEATURE_DIM)
    assert np.isfinite(rows).all()
