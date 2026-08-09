"""Truthful mill/Malom cross-tab diagnostics for Generalist v2."""

from __future__ import annotations

from collections import Counter, deque

import pytest
import torch

from scripts import train_s_gen_v2 as trainer


def _step(
    *,
    phase: str,
    mills_formed: int,
    mill_bonus: float,
    malom_quality: float | None,
) -> trainer.StepDiag:
    reward = trainer.RewardBreakdown(
        total=mill_bonus,
        mill_formed=mill_bonus,
    )
    return trainer.StepDiag(
        reward=reward,
        legal_moves=1,
        chosen_idx=0,
        chosen_prob=1.0,
        entropy=0.0,
        top1_prob=1.0,
        sentinel_mean=0.0,
        sentinel_chosen=0.0,
        h_before=0.0,
        h_after=0.0,
        h_delta=0.0,
        vn_before=0.0,
        vn_after=0.0,
        vn_delta=0.0,
        malom_chosen_wdl="n/a",
        malom_chosen_dtm=malom_quality,
        was_top1_policy=1,
        was_top1_heuristic=0,
        board_phase=phase,
        mills_formed=mills_formed,
        mill_bonus_awarded=mill_bonus,
        malom_quality=malom_quality,
    )


def test_game_diag_separates_formed_mills_rewards_and_malom_downgrades() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    steps = [
        _step(
            phase="place",
            mills_formed=1,
            mill_bonus=0.0,
            malom_quality=-1.0,
        ),
        _step(
            phase="move",
            mills_formed=1,
            mill_bonus=trainer.MILL_BONUS,
            malom_quality=0.0,
        ),
        _step(
            phase="fly",
            mills_formed=0,
            mill_bonus=0.0,
            malom_quality=-2.0,
        ),
        _step(
            phase="place",
            mills_formed=0,
            mill_bonus=0.0,
            malom_quality=None,
        ),
    ]
    result = trainer.RolloutResult(
        trajectory=[],
        step_diags=steps,
        outcome=trainer.LOSS_REWARD,
        ply=4,
        termination_reason="material",
        branch_candidates=[],
    )

    diag = trainer._build_game_diag(
        game_id="game:mill-observability",
        game_count=1,
        difficulty=1,
        learner_color="W",
        temperature=0.9,
        result=result,
        best_win_rate=0.0,
        win_history=deque(),
        last_update_pl=None,
        last_update_vl=None,
        last_update_ent=None,
        opt=optimizer,
        temp_frozen=False,
        source_ckpt="",
        game_type="vs_sanmill",
        phase_bucket="main",
        is_branch=False,
        branch_ply_start=0,
        target_age=0,
        bucket_counts=Counter(),
    )

    assert diag.formed_mill_count == 2
    assert diag.formed_mill_move_count == 2
    assert diag.mill_bonus_awarded_total == pytest.approx(trainer.MILL_BONUS)
    assert diag.reward_mill_bonus_mean == pytest.approx(
        trainer.MILL_BONUS / 4
    )
    assert diag.malom_known_move_rate == pytest.approx(0.75)
    assert diag.malom_preserving_move_rate == pytest.approx(1 / 3)
    assert diag.malom_downgrade_move_rate == pytest.approx(2 / 3)
    assert diag.formed_mill_malom_downgrade_count == 1
    assert diag.formed_mill_malom_downgrade_rate == pytest.approx(0.5)
    assert diag.formed_mill_malom_unknown_count == 0
    assert diag.formed_mill_malom_downgrade_place == 1
    assert diag.formed_mill_malom_downgrade_move == 0
    assert diag.formed_mill_malom_downgrade_fly == 0
    # Retain the historical field as a documented compatibility alias.
    assert diag.malom_win_move_rate == diag.malom_preserving_move_rate


def test_game_diag_reports_unknown_malom_on_a_mill_separately() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    result = trainer.RolloutResult(
        trajectory=[],
        step_diags=[
            _step(
                phase="place",
                mills_formed=1,
                mill_bonus=trainer.MILL_BONUS,
                malom_quality=None,
            )
        ],
        outcome=trainer.DRAW_LONG,
        ply=1,
        termination_reason="max-ply-truncation",
        branch_candidates=[],
    )

    diag = trainer._build_game_diag(
        game_id="game:legacy-unknown",
        game_count=1,
        difficulty=1,
        learner_color="W",
        temperature=0.9,
        result=result,
        best_win_rate=0.0,
        win_history=deque(),
        last_update_pl=None,
        last_update_vl=None,
        last_update_ent=None,
        opt=optimizer,
        temp_frozen=False,
        source_ckpt="",
        game_type="vs_frozen",
        phase_bucket="main",
        is_branch=False,
        branch_ply_start=0,
        target_age=0,
        bucket_counts=Counter(),
    )

    assert diag.formed_mill_malom_unknown_count == 1
    assert diag.formed_mill_malom_downgrade_count == 0
    assert diag.formed_mill_malom_downgrade_rate == 0.0
