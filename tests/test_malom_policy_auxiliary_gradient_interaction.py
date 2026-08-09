from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
)
from learned_ai.validation.malom_policy_auxiliary_gradient_interaction import (
    MalomPolicyAuxiliaryGradientInteractionError,
    audit_malom_policy_auxiliary_gradient_interaction,
)


class _FixedPolicyValue(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))
        self.value_bias = torch.nn.Parameter(torch.tensor(0.0))

    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        assert features.shape == (3, 4)
        return self.policy

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_bias.expand(features.shape[0])


def _steps() -> list[ScaffoldedStep]:
    phases = (0, 0, 0, 0, 1, 1, 3, 3)
    steps: list[ScaffoldedStep] = []
    for phase in phases:
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[phase] = 1.0
        steps.append(
            ScaffoldedStep(
                move_features=np.tile(one_hot, (3, 1)),
                value_input=np.zeros(1, dtype=np.float32),
                chosen_idx=2,
                log_prob_old=-1.0,
                reward=1.0,
                next_move_features=np.tile(one_hot, (3, 1)),
                next_value_input=np.zeros(1, dtype=np.float32),
                done=True,
                behaviour_temperature=0.9,
                malom_preserving_mask=np.asarray([True, True, False]),
            )
        )
    return steps


def _adam(model: torch.nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=5e-5)


def test_audit_measures_components_without_mutating_sources() -> None:
    model = _FixedPolicyValue()
    optimizer = _adam(model)
    steps = _steps()
    before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    expected = copy.deepcopy(model)
    expected_optimizer = _adam(expected)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    scaffolded_a2c_update(
        expected,
        expected_optimizer,
        steps,
        torch.device("cpu"),
        malom_policy_aux_coef=2.0,
    )

    report = audit_malom_policy_auxiliary_gradient_interaction(
        model,
        optimizer,
        steps,
        coefficient=2.0,
        device=torch.device("cpu"),
        expected_treatment_model=expected,
    )

    assert report["support"]["steps"] == 8
    assert report["support"]["labelled_by_phase"] == {
        "placement": 4,
        "movement": 2,
        "flying": 2,
    }
    assert report["support"]["informative_by_phase"] == {
        "placement": 4,
        "movement": 2,
        "flying": 2,
    }
    assert report["objectives"]["auxiliary"]["raw_gradient_l2"] > 0.0
    assert report["gradients"]["joint_pre_clip_l2"] > 0.0
    assert (
        report["gradients"]["auxiliary_to_ordinary_gradient_l2_ratio"]
        > 0.0
    )
    assert (
        report["adam_step"]["treatment_minus_baseline_preserving_mass"]
        >= 0.0
    )
    assert report["adam_step"]["persisted_treatment_replay_difference"] == {
        "l2": 0.0,
        "max_abs": 0.0,
    }
    assert report["original_model_unchanged"] is True
    assert report["original_optimizer_unchanged"] is True
    assert all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    assert optimizer.state_dict() == optimizer_before


def test_audit_rejects_missing_or_all_safe_labels() -> None:
    model = _FixedPolicyValue()
    optimizer = _adam(model)
    missing = _steps()
    missing[0].malom_preserving_mask = None
    with pytest.raises(
        MalomPolicyAuxiliaryGradientInteractionError,
        match="missing Malom preserving mask",
    ):
        audit_malom_policy_auxiliary_gradient_interaction(
            model,
            optimizer,
            missing,
            coefficient=0.1,
            device=torch.device("cpu"),
        )

    all_safe = _steps()
    for step in all_safe:
        step.malom_preserving_mask = np.ones(3, dtype=np.bool_)
    with pytest.raises(
        MalomPolicyAuxiliaryGradientInteractionError,
        match="no informative preserving set",
    ):
        audit_malom_policy_auxiliary_gradient_interaction(
            model,
            optimizer,
            all_safe,
            coefficient=0.1,
            device=torch.device("cpu"),
        )


def test_audit_rejects_invalid_phase_encoding() -> None:
    model = _FixedPolicyValue()
    optimizer = _adam(model)
    steps = _steps()
    steps[0].move_features[:, :4] = 0.0
    with pytest.raises(
        MalomPolicyAuxiliaryGradientInteractionError,
        match="invalid phase one-hot",
    ):
        audit_malom_policy_auxiliary_gradient_interaction(
            model,
            optimizer,
            steps,
            coefficient=0.1,
            device=torch.device("cpu"),
        )
