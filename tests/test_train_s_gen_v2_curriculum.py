"""Focused tests for Generalist curriculum evidence routing."""

from __future__ import annotations

from collections import deque
import copy

import torch

from scripts import train_s_gen_v2 as trainer


def _histories() -> tuple[deque[float], deque[float], deque[float]]:
    return deque(), deque(), deque()


def test_primary_reference_game_feeds_advancement_history() -> None:
    all_games, heuristic_games, advancement_games = _histories()

    added = trainer._record_curriculum_outcome(
        trainer.WIN_REWARD,
        win_history=all_games,
        win_history_heuristic=heuristic_games,
        level_heuristic_history=advancement_games,
        is_full_diff=True,
        is_advance_reference=True,
    )

    assert added is True
    assert list(all_games) == [1.0]
    assert list(heuristic_games) == [1.0]
    assert list(advancement_games) == [1.0]


def test_confirmation_or_retry_does_not_feed_advancement_history() -> None:
    all_games, heuristic_games, advancement_games = _histories()

    added = trainer._record_curriculum_outcome(
        trainer.DRAW_SHORT,
        win_history=all_games,
        win_history_heuristic=heuristic_games,
        level_heuristic_history=advancement_games,
        is_full_diff=True,
        is_advance_reference=False,
    )

    assert added is False
    assert list(all_games) == [0.5]
    assert list(heuristic_games) == [0.5]
    assert list(advancement_games) == []


def test_frozen_or_branch_game_only_feeds_all_game_history() -> None:
    all_games, heuristic_games, advancement_games = _histories()

    added = trainer._record_curriculum_outcome(
        trainer.LOSS_REWARD,
        win_history=all_games,
        win_history_heuristic=heuristic_games,
        level_heuristic_history=advancement_games,
        is_full_diff=False,
        is_advance_reference=False,
    )

    assert added is False
    assert list(all_games) == [0.0]
    assert list(heuristic_games) == []
    assert list(advancement_games) == []


def test_advancement_preserves_model_and_optimizer_continuity() -> None:
    model = trainer.ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        policy_hidden=(8,),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    frozen = trainer.FrozenModelOpponent(model, torch.device("cpu"))
    for parameter in frozen._model.parameters():
        parameter.data.zero_()
    histories = (deque([1.0]), deque([1.0]), deque([1.0]))

    returned = trainer._complete_curriculum_transition(
        model=model,
        optimizer=optimizer,
        frozen_opponent=frozen,
        histories=histories,
    )

    assert returned is optimizer
    for name, value in model.state_dict().items():
        assert torch.equal(value, model_before[name])
        assert torch.equal(frozen._model.state_dict()[name], value)
    assert optimizer.state_dict()["param_groups"] == optimizer_before["param_groups"]
    for parameter_id, state in optimizer.state_dict()["state"].items():
        for key, value in state.items():
            expected = optimizer_before["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected)
            else:
                assert value == expected
    assert all(not history for history in histories)
