from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
)
from learned_ai.validation.malom_policy_auxiliary_gradient_interaction import (
    MalomPolicyAuxiliaryGradientInteractionError,
    audit_malom_policy_auxiliary_gradient_interaction,
    audit_malom_policy_auxiliary_normalized_target_response,
    measure_malom_policy_auxiliary_batch_gradients,
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
    assert report["gradients"]["auxiliary_to_ordinary_gradient_l2_ratio"] > 0.0
    assert (
        report["gradients"]["auxiliary_to_ordinary_policy_head_gradient_l2_ratio"] > 0.0
    )
    assert report["gradients"]["ordinary_policy_head_gradient_l2"] > 0.0
    assert report["adam_step"]["treatment_minus_baseline_preserving_mass"] >= 0.0
    assert report["adam_step"]["baseline_to_treatment_policy_kl"]["mean"] >= 0.0
    assert (
        report["adam_step"]["informative_batch_policy_after_treatment"][
            "informative_steps"
        ]
        == 8
    )
    assert report["adam_step"]["treatment_minus_baseline_entropy"] <= 0.0
    assert report["adam_step"]["persisted_treatment_replay_difference"] == {
        "raw": {"l2": 0.0, "max_abs": 0.0},
        "functionally_relevant": {"l2": 0.0, "max_abs": 0.0},
        "softmax_invariant_parameter_names": [],
    }
    assert report["original_model_unchanged"] is True
    assert report["original_optimizer_unchanged"] is True
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
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


def test_persisted_replay_separates_shared_policy_bias_invariance() -> None:
    torch.manual_seed(7)
    model = ScaffoldedPolicyNet(
        move_feat_dim=4,
        value_input_dim=1,
        policy_hidden=(4,),
        value_hidden=(4,),
    )
    optimizer = _adam(model)
    steps = _steps()
    expected = copy.deepcopy(model)
    expected_optimizer = _adam(expected)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    scaffolded_a2c_update(
        expected,
        expected_optimizer,
        steps,
        torch.device("cpu"),
        malom_policy_aux_coef=0.1,
    )
    with torch.no_grad():
        expected.policy_mlp[2].bias.add_(0.125)

    report = audit_malom_policy_auxiliary_gradient_interaction(
        model,
        optimizer,
        steps,
        coefficient=0.1,
        device=torch.device("cpu"),
        expected_treatment_model=expected,
    )
    difference = report["adam_step"]["persisted_treatment_replay_difference"]
    assert difference["raw"]["max_abs"] == pytest.approx(0.125)
    assert difference["functionally_relevant"]["max_abs"] == 0.0
    assert difference["softmax_invariant_parameter_names"] == ["policy_mlp.2.bias"]


def test_batch_measurement_derives_ratios_without_an_optimizer_or_mutation() -> None:
    model = _FixedPolicyValue()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = copy.deepcopy(model.state_dict())

    report = measure_malom_policy_auxiliary_batch_gradients(
        model,
        _steps(),
        device=torch.device("cpu"),
        target_policy_head_ratios=(0.25, 0.5, 1.0),
    )

    assert report["support"]["informative_steps"] == 8
    assert report["ordinary_policy_head_gradient_l2"] > 0.0
    assert report["raw_auxiliary_gradient_l2"] > 0.0
    assert [item["status"] for item in report["candidate_scales"]] == [
        "measured",
        "measured",
        "measured",
    ]
    ordinary = report["ordinary_policy_head_gradient_l2"]
    for item in report["candidate_scales"]:
        assert item["applied_auxiliary_gradient_l2"] == pytest.approx(
            ordinary * item["target_policy_head_ratio"]
        )
    assert report["optimizer_constructed"] is False
    assert report["optimizer_steps"] == 0
    assert report["backward_calls"] == 0
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )


def test_batch_measurement_reports_an_all_safe_batch_without_fabricating_scale() -> (
    None
):
    model = _FixedPolicyValue()
    steps = _steps()
    for step in steps:
        step.malom_preserving_mask = np.ones(3, dtype=np.bool_)

    report = measure_malom_policy_auxiliary_batch_gradients(
        model,
        steps,
        device=torch.device("cpu"),
        target_policy_head_ratios=(0.5,),
    )

    assert report["support"]["informative_steps"] == 0
    assert report["raw_auxiliary_gradient_l2"] == 0.0
    assert report["raw_auxiliary_to_ordinary_policy_head_cosine"] is None
    assert report["candidate_scales"] == [
        {
            "target_policy_head_ratio": 0.5,
            "status": "no_informative_steps",
            "effective_coefficient": None,
        }
    ]


@pytest.mark.parametrize(
    "ratios, message",
    [
        ((), "must be finite and positive"),
        ((0.0,), "must be finite and positive"),
        ((0.5, 0.25), "must be unique and increasing"),
        ((0.5, 0.5), "must be unique and increasing"),
    ],
)
def test_batch_measurement_rejects_invalid_target_ratios(
    ratios: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(
        MalomPolicyAuxiliaryGradientInteractionError,
        match=message,
    ):
        measure_malom_policy_auxiliary_batch_gradients(
            _FixedPolicyValue(),
            _steps(),
            device=torch.device("cpu"),
            target_policy_head_ratios=ratios,
        )


def test_normalized_target_response_replays_production_and_preserves_sources() -> None:
    model = _FixedPolicyValue()
    optimizer = _adam(model)
    steps = _steps()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    measurement = measure_malom_policy_auxiliary_batch_gradients(
        model,
        steps,
        device=torch.device("cpu"),
        target_policy_head_ratios=(0.25,),
    )
    expected_coefficient = measurement["candidate_scales"][0]["effective_coefficient"]
    expected = copy.deepcopy(model)
    expected_optimizer = _adam(expected)
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    scaffolded_a2c_update(
        expected,
        expected_optimizer,
        steps,
        torch.device("cpu"),
        malom_policy_aux_mode="policy-head-normalized",
        malom_policy_aux_target_ratio=0.25,
        malom_policy_aux_coef_cap=10.0,
    )

    report = audit_malom_policy_auxiliary_normalized_target_response(
        model,
        optimizer,
        steps,
        target_policy_head_ratios=(0.25, 0.5, 1.0),
        coefficient_cap=10.0,
        denominator_floor=1e-12,
        device=torch.device("cpu"),
        expected_treatment_model=expected,
        expected_treatment_target_ratio=0.25,
    )

    assert report["original_model_unchanged"] is True
    assert report["original_optimizer_unchanged"] is True
    assert [item["target_policy_head_ratio"] for item in report["responses"]] == [
        0.25,
        0.5,
        1.0,
    ]
    assert report["responses"][0]["effective_coefficient"] == pytest.approx(
        expected_coefficient
    )
    assert report["responses"][0]["realized_policy_head_ratio"] == pytest.approx(0.25)
    assert report["responses"][1]["realized_policy_head_ratio"] == pytest.approx(0.5)
    assert report["responses"][2]["realized_policy_head_ratio"] == pytest.approx(1.0)
    replay = report["responses"][0]["audit"]["adam_step"][
        "persisted_treatment_replay_difference"
    ]
    assert replay["functionally_relevant"] == {"l2": 0.0, "max_abs": 0.0}
    assert all(
        torch.equal(model_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert optimizer.state_dict() == optimizer_before


def test_normalized_target_response_reports_coefficient_cap() -> None:
    model = _FixedPolicyValue()
    report = audit_malom_policy_auxiliary_normalized_target_response(
        model,
        _adam(model),
        _steps(),
        target_policy_head_ratios=(0.25, 0.5, 1.0),
        coefficient_cap=1e-6,
        denominator_floor=1e-12,
        device=torch.device("cpu"),
    )

    assert all(item["coefficient_capped"] for item in report["responses"])
    assert all(
        item["effective_coefficient"] == pytest.approx(1e-6)
        for item in report["responses"]
    )
    assert (
        len({item["realized_policy_head_ratio"] for item in report["responses"]}) == 1
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"coefficient_cap": 0.0}, "coefficient cap must be positive"),
        ({"denominator_floor": 0.0}, "denominator floor must be positive"),
        (
            {"expected_treatment_model": _FixedPolicyValue()},
            "must be provided together",
        ),
        (
            {
                "expected_treatment_model": _FixedPolicyValue(),
                "expected_treatment_target_ratio": 0.75,
            },
            "is not in the candidate set",
        ),
    ],
)
def test_normalized_target_response_rejects_invalid_contract(
    kwargs: dict[str, object],
    message: str,
) -> None:
    model = _FixedPolicyValue()
    defaults: dict[str, object] = {
        "target_policy_head_ratios": (0.25, 0.5, 1.0),
        "coefficient_cap": 0.25,
        "denominator_floor": 1e-12,
        "device": torch.device("cpu"),
    }
    defaults.update(kwargs)
    with pytest.raises(
        MalomPolicyAuxiliaryGradientInteractionError,
        match=message,
    ):
        audit_malom_policy_auxiliary_normalized_target_response(
            model,
            _adam(model),
            _steps(),
            **defaults,
        )
