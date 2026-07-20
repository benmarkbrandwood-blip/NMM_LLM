"""Tests for deterministic ScaffoldedPolicyNet canaries."""

from __future__ import annotations

import copy

import pytest
import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.validation.model_canary import (
    ModelCanary,
    capture_model_canary,
    verify_model_canary,
)


def _model() -> ScaffoldedPolicyNet:
    torch.manual_seed(42)
    return ScaffoldedPolicyNet(
        move_feat_dim=12,
        value_input_dim=9,
        policy_hidden=(8,),
        value_hidden=(7,),
    )


def test_model_canary_round_trip_and_single_batch_parity() -> None:
    model = _model()

    canary = capture_model_canary(model)

    assert ModelCanary.from_dict(canary.to_dict()) == canary
    assert len(canary.identity) == 64
    assert verify_model_canary(copy.deepcopy(model), canary) == {
        "policy_max_abs": 0.0,
        "value_max_abs": 0.0,
    }


def test_model_canary_rejects_modified_weights() -> None:
    model = _model()
    canary = capture_model_canary(model)
    with torch.no_grad():
        next(model.parameters()).add_(0.1)

    with pytest.raises(RuntimeError, match="policy canary mismatch"):
        verify_model_canary(model, canary)
