from __future__ import annotations

import pytest
import torch

from game.board import BoardState
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.scaffolded_encoder import (
    encode_position_with_lookahead,
)
from learned_ai.models.training_rollout_heuristic import (
    training_rollout_evaluate,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB


class _BrokenTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def policy_logits(self, _features):
        raise RuntimeError("target failed")


class _BrokenHumanDB:
    def query_all_frequencies(self, _board):
        raise RuntimeError("human db failed")


class _BrokenSpecialistDB:
    def query_wdl(self, _board, *, min_samples):
        raise RuntimeError(f"specialist db failed at {min_samples}")


class _BrokenMalom:
    def query(self, _board):
        raise RuntimeError("malom failed")


class _BrokenLookahead:
    feat_dim = 72

    def score_moves_matrix(self, _board, _encoded, _player):
        raise RuntimeError("lookahead failed")


def _advisor(**overrides) -> LookaheadAdvisor:
    values = {
        "sentinel": None,
        "evaluate_fn": training_rollout_evaluate,
        "use_sentinel": False,
        "ply_depth": 12,
        "sim_ply_depth": 2,
        "strict": True,
    }
    values.update(overrides)
    return LookaheadAdvisor(**values)


def test_strict_lookahead_propagates_frozen_target_failure() -> None:
    advisor = _advisor(frozen_model=_BrokenTarget(), frozen_device="cpu")

    with pytest.raises(RuntimeError, match="target failed"):
        advisor._frozen_model_best_move(BoardState.new_game(), "W")


def test_strict_lookahead_propagates_human_db_failure() -> None:
    advisor = _advisor(human_db=_BrokenHumanDB())

    with pytest.raises(RuntimeError, match="human db failed"):
        advisor._human_db_best_move(BoardState.new_game(), "W")


def test_strict_encoder_propagates_specialist_db_failure() -> None:
    with pytest.raises(RuntimeError, match="specialist db failed"):
        encode_position_with_lookahead(
            BoardState.new_game(),
            "W",
            lookahead_advisor=None,
            specialist_db=_BrokenSpecialistDB(),
            sdb_min_samples=3,
            strict=True,
        )


def test_strict_encoder_propagates_lookahead_failure() -> None:
    with pytest.raises(RuntimeError, match="lookahead failed"):
        encode_position_with_lookahead(
            BoardState.new_game(),
            "W",
            lookahead_advisor=_BrokenLookahead(),
            strict=True,
        )


def test_strict_lookahead_propagates_malom_failure() -> None:
    advisor = _advisor(endgame_db=_BrokenMalom())
    encoded = encode_position_with_lookahead(
        BoardState.new_game(), "W", lookahead_advisor=None
    )
    assert encoded is not None

    with pytest.raises(RuntimeError, match="malom failed"):
        advisor.score_moves_matrix(
            BoardState.new_game(),
            encoded,
            "W",
            moves_subset=encoded.legal_moves[:1],
        )


def test_external_solved_db_strict_lookup_propagates_decoder_failure() -> None:
    database = ExternalSolvedDB(enabled=False, strict=True)

    class _BrokenDecoder:
        def query(self, _board):
            raise RuntimeError("decoder failed")

    database._malom = _BrokenDecoder()

    with pytest.raises(RuntimeError, match="decoder failed"):
        database._lookup(BoardState.new_game())


def test_default_encoder_keeps_legacy_best_effort_fallback() -> None:
    encoded = encode_position_with_lookahead(
        BoardState.new_game(),
        "W",
        lookahead_advisor=_BrokenLookahead(),
        specialist_db=_BrokenSpecialistDB(),
        sdb_min_samples=3,
    )

    assert encoded is not None
    assert encoded.feat_matrix.shape[1] == 134
