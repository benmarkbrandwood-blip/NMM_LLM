"""Focused tests for atomic in-training recovery state."""

from __future__ import annotations

import copy

import pytest
import torch

from scripts import train_s_gen_v2 as trainer


def _model() -> trainer.ScaffoldedPolicyNet:
    return trainer.ScaffoldedPolicyNet(
        move_feat_dim=trainer.MOVE_FEAT_DIM_WITH_LOOKAHEAD,
        value_input_dim=trainer.VALUE_INPUT_DIM_WITH_HISTORY,
        policy_hidden=(8,),
    )


def _prime(model, optimizer, scale: float) -> None:
    loss = sum((parameter * scale).square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_recovery_restores_matching_model_and_optimizer_together() -> None:
    saved_model = _model()
    saved_optimizer = torch.optim.Adam(saved_model.parameters(), lr=3e-4)
    _prime(saved_model, saved_optimizer, 0.5)

    current_model = _model()
    current_optimizer = torch.optim.Adam(current_model.parameters(), lr=9e-4)
    _prime(current_model, current_optimizer, 1.5)

    trainer._restore_recovery_training_state(
        model=current_model,
        optimizer=current_optimizer,
        model_state=copy.deepcopy(saved_model.state_dict()),
        optimizer_state=copy.deepcopy(saved_optimizer.state_dict()),
    )

    _assert_nested_equal(current_model.state_dict(), saved_model.state_dict())
    _assert_nested_equal(
        current_optimizer.state_dict(), saved_optimizer.state_dict()
    )


def test_recovery_failure_is_atomic() -> None:
    current_model = _model()
    current_optimizer = torch.optim.Adam(current_model.parameters(), lr=9e-4)
    _prime(current_model, current_optimizer, 1.5)
    model_before = copy.deepcopy(current_model.state_dict())
    optimizer_before = copy.deepcopy(current_optimizer.state_dict())

    incompatible = copy.deepcopy(current_model.state_dict())
    first_key = next(iter(incompatible))
    incompatible[first_key] = torch.zeros(1)

    with pytest.raises(RuntimeError, match="incompatible recovery model state"):
        trainer._restore_recovery_training_state(
            model=current_model,
            optimizer=current_optimizer,
            model_state=incompatible,
            optimizer_state=copy.deepcopy(optimizer_before),
        )

    _assert_nested_equal(current_model.state_dict(), model_before)
    _assert_nested_equal(current_optimizer.state_dict(), optimizer_before)


def test_recovery_requires_optimizer_state() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    with pytest.raises(RuntimeError, match="no optimizer state"):
        trainer._restore_recovery_training_state(
            model=model,
            optimizer=optimizer,
            model_state=copy.deepcopy(model.state_dict()),
            optimizer_state=None,
        )


def test_incompatible_recovery_optimizer_is_atomic() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    _prime(model, optimizer, 1.0)
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    incompatible_optimizer = copy.deepcopy(optimizer_before)
    incompatible_optimizer["param_groups"][0]["params"] = []

    with pytest.raises(RuntimeError, match="incompatible recovery optimizer state"):
        trainer._restore_recovery_training_state(
            model=model,
            optimizer=optimizer,
            model_state=copy.deepcopy(model_before),
            optimizer_state=incompatible_optimizer,
        )

    _assert_nested_equal(model.state_dict(), model_before)
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)
