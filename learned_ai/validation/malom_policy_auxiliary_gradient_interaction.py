"""No-update gradient interaction audit for production-shaped A2C batches."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learned_ai.training.scaffolded_a2c import (
    GRAD_CLIP,
    MIN_UPDATE_STEPS,
    VALUE_COEF,
    NonFiniteTrainingError,
    ScaffoldedStep,
    _behaviour_temperature,
    _bootstrap_sign,
    _malom_preserving_mask,
    malom_preserving_set_loss,
    scaffolded_a2c_update,
)


class MalomPolicyAuxiliaryGradientInteractionError(ValueError):
    """Raised when a gradient audit input cannot be interpreted safely."""


def _require_finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            f"{label} must be finite"
        )
    return result


def _model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _hash_state_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_state_value(digest, key)
            _hash_state_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_state_value(digest, item)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(repr(value).encode("utf-8"))


def _optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_state_value(digest, optimizer.state_dict())
    return digest.hexdigest()


def _phase(step: ScaffoldedStep) -> str:
    features = np.asarray(step.move_features)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] < 4:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "move features do not contain a phase one-hot"
        )
    one_hots = features[:, :4]
    reference = one_hots[0]
    if not np.allclose(one_hots, reference, rtol=0.0, atol=0.0):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "legal actions disagree on the state phase"
        )
    if not np.array_equal(reference, np.eye(4, dtype=reference.dtype)[np.argmax(reference)]):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "move features contain an invalid phase one-hot"
        )
    index = int(np.argmax(reference))
    return ("placement", "movement", "movement", "flying")[index]


def _objective_tensors(
    model: nn.Module,
    steps: Sequence[ScaffoldedStep],
    *,
    device: torch.device,
    gamma: float,
    require_informative: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if len(steps) < MIN_UPDATE_STEPS:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient audit batch is smaller than the production minimum"
        )
    gamma = _require_finite(gamma, label="gamma")
    if not 0.0 <= gamma <= 1.0:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gamma must be between zero and one"
        )

    all_vi = torch.as_tensor(
        np.stack([step.value_input for step in steps]),
        dtype=torch.float32,
        device=device,
    )
    all_nvi = torch.as_tensor(
        np.stack([step.next_value_input for step in steps]),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.as_tensor(
        [step.reward for step in steps],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.as_tensor(
        [float(step.done) for step in steps],
        dtype=torch.float32,
        device=device,
    )
    signs = torch.as_tensor(
        [_bootstrap_sign(step) for step in steps],
        dtype=torch.float32,
        device=device,
    )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (all_vi, all_nvi, rewards, dones, signs)
    ):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient audit batch contains a non-finite tensor"
        )

    model.train()
    with torch.no_grad():
        next_value = model.value(all_nvi) * signs
        next_value = next_value * (1.0 - dones)
    targets = rewards + gamma * next_value
    current_value = model.value(all_vi)
    advantages = (targets - current_value).detach()
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

    policy_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    auxiliary_terms: list[torch.Tensor] = []
    labelled_by_phase = {phase: 0 for phase in ("placement", "movement", "flying")}
    informative_by_phase = dict(labelled_by_phase)
    temperatures: list[float] = []
    preserving_masses: list[float] = []

    for index, step in enumerate(steps):
        features = torch.as_tensor(
            step.move_features,
            dtype=torch.float32,
            device=device,
        )
        logits = model.policy_logits(features)
        temperature = _behaviour_temperature(step)
        temperatures.append(temperature)
        log_probabilities = F.log_softmax(logits / temperature, dim=-1)
        if not 0 <= int(step.chosen_idx) < len(log_probabilities):
            raise MalomPolicyAuxiliaryGradientInteractionError(
                "chosen action index is outside the legal action set"
            )
        policy_terms.append(
            -log_probabilities[int(step.chosen_idx)] * advantages[index]
        )
        probabilities = log_probabilities.exp()
        entropy_terms.append(-(probabilities * log_probabilities).sum())

        try:
            preserving = _malom_preserving_mask(
                step,
                legal_move_count=len(log_probabilities),
                device=device,
            )
        except NonFiniteTrainingError as exc:
            raise MalomPolicyAuxiliaryGradientInteractionError(str(exc)) from exc
        phase = _phase(step)
        labelled_by_phase[phase] += 1
        preserving_mass = probabilities[preserving].sum()
        preserving_masses.append(float(preserving_mass.detach().item()))
        if not bool(preserving.all()):
            informative_by_phase[phase] += 1
            auxiliary_terms.append(
                malom_preserving_set_loss(log_probabilities, preserving)
            )

    if require_informative and not auxiliary_terms:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient audit batch has no informative preserving set"
        )

    policy_loss = torch.stack(policy_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    value_loss = F.mse_loss(current_value, targets.detach())
    auxiliary_loss = (
        torch.stack(auxiliary_terms).mean()
        if auxiliary_terms
        else policy_loss * 0.0
    )
    objectives = {
        "policy": policy_loss,
        "entropy": entropy,
        "value": value_loss,
        "auxiliary": auxiliary_loss,
    }
    if not all(bool(torch.isfinite(value).all()) for value in objectives.values()):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient audit objective is non-finite"
        )
    support = {
        "steps": len(steps),
        "labelled_by_phase": labelled_by_phase,
        "informative_by_phase": informative_by_phase,
        "informative_steps": len(auxiliary_terms),
        "temperature_min": min(temperatures),
        "temperature_max": max(temperatures),
        "mean_preserving_mass": sum(preserving_masses) / len(preserving_masses),
        "terminal_steps": sum(bool(step.done) for step in steps),
        "reward_min": min(float(step.reward) for step in steps),
        "reward_max": max(float(step.reward) for step in steps),
    }
    return objectives, support


def _require_target_ratios(values: Sequence[float]) -> tuple[float, ...]:
    ratios = tuple(
        _require_finite(value, label="target policy-head ratio")
        for value in values
    )
    if not ratios or any(value <= 0.0 for value in ratios):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "target policy-head ratios must be finite and positive"
        )
    if tuple(sorted(set(ratios))) != ratios:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "target policy-head ratios must be unique and increasing"
        )
    return ratios


def _gradients(
    objective: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    raw = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    result = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, raw, strict=True)
    )
    if not all(bool(torch.isfinite(value).all()) for value in result):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient audit produced a non-finite gradient"
        )
    return result


def _dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    return float(
        sum(
            (
                first.detach().double() * second.detach().double()
            ).sum().item()
            for first, second in zip(left, right, strict=True)
        )
    )


def _norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, _dot(values, values)))


def _scaled(
    values: Sequence[torch.Tensor], scale: float
) -> tuple[torch.Tensor, ...]:
    return tuple(value * scale for value in values)


def _sum_gradients(
    gradients: Sequence[Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        sum((group[index] for group in gradients), torch.zeros_like(gradients[0][index]))
        for index in range(len(gradients[0]))
    )


def _cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    denominator = _norm(left) * _norm(right)
    return _dot(left, right) / denominator if denominator > 0.0 else 0.0


def measure_malom_policy_auxiliary_batch_gradients(
    model: nn.Module,
    steps: Sequence[ScaffoldedStep],
    *,
    device: torch.device,
    target_policy_head_ratios: Sequence[float],
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = VALUE_COEF,
    denominator_floor: float = 1e-12,
) -> dict[str, Any]:
    """Measure candidate auxiliary scales without an optimiser or update.

    The production policy, entropy, value and exact-WDL preserving-set
    objectives are evaluated on one production-shaped batch.  Candidate
    coefficients are derived only as diagnostics: no coefficient is selected,
    no ``backward`` or optimiser method is called, and the model's parameters,
    mode and ``requires_grad`` flags are restored before returning.
    """
    ratios = _require_target_ratios(target_policy_head_ratios)
    entropy_coef = _require_finite(
        entropy_coef,
        label="entropy coefficient",
    )
    value_coef = _require_finite(value_coef, label="value coefficient")
    denominator_floor = _require_finite(
        denominator_floor,
        label="denominator floor",
    )
    if entropy_coef < 0.0 or value_coef < 0.0:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "loss coefficients must be non-negative"
        )
    if denominator_floor <= 0.0:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "denominator floor must be positive"
        )
    if any(
        isinstance(module, nn.Dropout) and module.p > 0.0
        for module in model.modules()
    ):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient measurement requires dropout-free model semantics"
        )

    model_before = _model_digest(model)
    was_training = model.training
    parameters = tuple(model.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(True)
        objectives, support = _objective_tensors(
            model,
            steps,
            device=device,
            gamma=gamma,
            require_informative=False,
        )
        raw_gradients: dict[str, tuple[torch.Tensor, ...]] = {}
        objective_names = tuple(objectives)
        for index, name in enumerate(objective_names):
            raw_gradients[name] = _gradients(
                objectives[name],
                parameters,
                retain_graph=index < len(objective_names) - 1,
            )

        applied_policy = raw_gradients["policy"]
        applied_entropy = _scaled(raw_gradients["entropy"], -entropy_coef)
        applied_value = _scaled(raw_gradients["value"], value_coef)
        ordinary_policy = _sum_gradients(
            [applied_policy, applied_entropy]
        )
        ordinary_full = _sum_gradients(
            [ordinary_policy, applied_value]
        )
        raw_auxiliary = raw_gradients["auxiliary"]
        ordinary_policy_norm = _norm(ordinary_policy)
        raw_auxiliary_norm = _norm(raw_auxiliary)
        informative = int(support["informative_steps"])

        candidates: list[dict[str, Any]] = []
        for target in ratios:
            if informative == 0:
                candidates.append(
                    {
                        "target_policy_head_ratio": target,
                        "status": "no_informative_steps",
                        "effective_coefficient": None,
                    }
                )
                continue
            if ordinary_policy_norm <= denominator_floor:
                candidates.append(
                    {
                        "target_policy_head_ratio": target,
                        "status": "ordinary_policy_gradient_below_floor",
                        "effective_coefficient": None,
                    }
                )
                continue
            if raw_auxiliary_norm <= denominator_floor:
                candidates.append(
                    {
                        "target_policy_head_ratio": target,
                        "status": "auxiliary_gradient_below_floor",
                        "effective_coefficient": None,
                    }
                )
                continue
            coefficient = (
                target * ordinary_policy_norm / raw_auxiliary_norm
            )
            applied_auxiliary = _scaled(raw_auxiliary, coefficient)
            joint_policy = _sum_gradients(
                [ordinary_policy, applied_auxiliary]
            )
            candidates.append(
                {
                    "target_policy_head_ratio": target,
                    "status": "measured",
                    "effective_coefficient": coefficient,
                    "applied_auxiliary_gradient_l2": _norm(
                        applied_auxiliary
                    ),
                    "joint_policy_head_gradient_l2": _norm(joint_policy),
                    "auxiliary_to_ordinary_policy_head_cosine": _cosine(
                        applied_auxiliary,
                        ordinary_policy,
                    ),
                }
            )

        return {
            "support": support,
            "objectives": {
                name: {
                    "objective_value": float(value.detach().item()),
                    "raw_gradient_l2": _norm(raw_gradients[name]),
                }
                for name, value in objectives.items()
            },
            "ordinary_policy_head_gradient_l2": ordinary_policy_norm,
            "ordinary_full_gradient_l2": _norm(ordinary_full),
            "raw_auxiliary_gradient_l2": raw_auxiliary_norm,
            "raw_auxiliary_to_ordinary_policy_head_cosine": (
                _cosine(raw_auxiliary, ordinary_policy)
                if informative > 0
                else None
            ),
            "denominator_floor": denominator_floor,
            "candidate_scales": candidates,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "interpretation": (
                "read-only per-batch gradient measurement; candidate scales "
                "are diagnostics, not a selected training rule"
            ),
        }
    finally:
        model.train(was_training)
        for parameter, required in zip(
            parameters,
            requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(required)
        if _model_digest(model) != model_before:
            raise MalomPolicyAuxiliaryGradientInteractionError(
                "gradient measurement changed model parameters"
            )


def _clone_adam(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Module, torch.optim.Adam]:
    if not isinstance(optimizer, torch.optim.Adam):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient interaction audit currently requires Adam"
        )
    if len(optimizer.param_groups) != 1:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient interaction audit requires one Adam parameter group"
        )
    candidate = copy.deepcopy(model)
    group = optimizer.param_groups[0]
    candidate_optimizer = torch.optim.Adam(
        candidate.parameters(),
        lr=float(group["lr"]),
        betas=tuple(group["betas"]),
        eps=float(group["eps"]),
        weight_decay=float(group["weight_decay"]),
        amsgrad=bool(group["amsgrad"]),
    )
    candidate_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return candidate, candidate_optimizer


def _softmax_invariant_policy_bias_names(model: nn.Module) -> tuple[str, ...]:
    policy_mlp = getattr(model, "policy_mlp", None)
    if not isinstance(policy_mlp, nn.Sequential):
        return ()
    linear_modules = [
        (name, module)
        for name, module in policy_mlp.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    if not linear_modules or linear_modules[-1][1].out_features != 1:
        return ()
    name, module = linear_modules[-1]
    if module.bias is None:
        return ()
    return (f"policy_mlp.{name}.bias",)


def _parameter_distance(
    left: nn.Module,
    right: nn.Module,
    *,
    excluded: Sequence[str] = (),
) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    if set(left_parameters) != set(right_parameters):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "model parameter names differ"
        )
    excluded_set = set(excluded)
    if not excluded_set <= set(left_parameters):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "excluded parameter name is not present"
        )
    for name in sorted(left_parameters):
        if name in excluded_set:
            continue
        difference = (
            left_parameters[name].detach().double()
            - right_parameters[name].detach().double()
        )
        squared += float(difference.square().sum().item())
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max().item()))
    return {"l2": math.sqrt(squared), "max_abs": maximum}


def _batch_preserving_mass(
    model: nn.Module,
    steps: Sequence[ScaffoldedStep],
    *,
    device: torch.device,
) -> float:
    masses: list[float] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for step in steps:
                mask = np.asarray(step.malom_preserving_mask)
                if mask.dtype != np.bool_ or mask.ndim != 1 or mask.all():
                    continue
                features = torch.as_tensor(
                    step.move_features,
                    dtype=torch.float32,
                    device=device,
                )
                preserving = torch.as_tensor(mask, dtype=torch.bool, device=device)
                probabilities = torch.softmax(
                    model.policy_logits(features) / _behaviour_temperature(step),
                    dim=-1,
                )
                masses.append(float(probabilities[preserving].sum().item()))
    finally:
        model.train(was_training)
    if not masses:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "batch has no informative preserving probability"
        )
    return sum(masses) / len(masses)


def audit_malom_policy_auxiliary_gradient_interaction(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: Sequence[ScaffoldedStep],
    *,
    coefficient: float,
    device: torch.device,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    expected_treatment_model: nn.Module | None = None,
) -> dict[str, Any]:
    """Measure objective gradients and disposable Adam steps without mutation."""
    coefficient = _require_finite(coefficient, label="coefficient")
    entropy_coef = _require_finite(entropy_coef, label="entropy coefficient")
    value_coef = _require_finite(value_coef, label="value coefficient")
    grad_clip = _require_finite(grad_clip, label="gradient clip")
    if coefficient <= 0.0 or entropy_coef < 0.0 or value_coef < 0.0:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "coefficient must be positive and loss coefficients non-negative"
        )
    if grad_clip <= 0.0:
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient clip must be positive"
        )
    if any(
        isinstance(module, nn.Dropout) and module.p > 0.0
        for module in model.modules()
    ):
        raise MalomPolicyAuxiliaryGradientInteractionError(
            "gradient interaction audit requires dropout-free model semantics"
        )

    model_before = _model_digest(model)
    optimizer_before = _optimizer_digest(optimizer)
    was_training = model.training
    objectives, support = _objective_tensors(
        model,
        steps,
        device=device,
        gamma=gamma,
    )
    parameters = tuple(model.parameters())
    raw_gradients: dict[str, tuple[torch.Tensor, ...]] = {}
    objective_names = tuple(objectives)
    for index, name in enumerate(objective_names):
        raw_gradients[name] = _gradients(
            objectives[name],
            parameters,
            retain_graph=index < len(objective_names) - 1,
        )

    applied_scales = {
        "policy": 1.0,
        "entropy": -entropy_coef,
        "value": value_coef,
        "auxiliary": coefficient,
    }
    applied = {
        name: _scaled(raw_gradients[name], scale)
        for name, scale in applied_scales.items()
    }
    ordinary = _sum_gradients(
        [applied["policy"], applied["entropy"], applied["value"]]
    )
    ordinary_policy = _sum_gradients(
        [applied["policy"], applied["entropy"]]
    )
    joint_policy = _sum_gradients(
        [ordinary_policy, applied["auxiliary"]]
    )
    joint = _sum_gradients([ordinary, applied["auxiliary"]])
    joint_norm = _norm(joint)
    ordinary_norm = _norm(ordinary)
    auxiliary_norm = _norm(applied["auxiliary"])
    ordinary_policy_norm = _norm(ordinary_policy)
    joint_policy_norm = _norm(joint_policy)
    clip_scale = min(1.0, grad_clip / (joint_norm + 1e-6))
    denominator = joint_norm * joint_norm

    component_reports: dict[str, dict[str, Any]] = {}
    for name in objective_names:
        component_reports[name] = {
            "objective_value": float(objectives[name].detach().item()),
            "raw_gradient_l2": _norm(raw_gradients[name]),
            "applied_scale": applied_scales[name],
            "applied_gradient_l2": _norm(applied[name]),
            "projection_fraction_of_joint": (
                _dot(applied[name], joint) / denominator
                if denominator > 0.0
                else 0.0
            ),
            "projection_fraction_of_policy_joint": (
                _dot(applied[name], joint_policy)
                / (joint_policy_norm * joint_policy_norm)
                if name != "value" and joint_policy_norm > 0.0
                else None
            ),
        }

    pairwise: dict[str, float] = {}
    for left_index, left in enumerate(objective_names):
        for right in objective_names[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = _cosine(
                applied[left], applied[right]
            )

    baseline_model, baseline_optimizer = _clone_adam(model, optimizer)
    treatment_model, treatment_optimizer = _clone_adam(model, optimizer)
    before_mass = _batch_preserving_mass(model, steps, device=device)
    baseline_losses = scaffolded_a2c_update(
        baseline_model,
        baseline_optimizer,
        list(steps),
        device,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        grad_clip=grad_clip,
        malom_policy_aux_coef=0.0,
    )
    treatment_diagnostics: dict[str, float | int] = {}
    treatment_losses = scaffolded_a2c_update(
        treatment_model,
        treatment_optimizer,
        list(steps),
        device,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        grad_clip=grad_clip,
        malom_policy_aux_coef=coefficient,
        diagnostics=treatment_diagnostics,
    )
    baseline_mass = _batch_preserving_mass(
        baseline_model,
        steps,
        device=device,
    )
    treatment_mass = _batch_preserving_mass(
        treatment_model,
        steps,
        device=device,
    )
    model.train(was_training)

    current_lr = float(optimizer.param_groups[0]["lr"])
    adam_step = {
        "learning_rate": current_lr,
        "baseline_reported_losses": list(baseline_losses),
        "treatment_reported_losses": list(treatment_losses),
        "treatment_auxiliary_diagnostics": treatment_diagnostics,
        "baseline_parameter_delta": _parameter_distance(
            baseline_model, model
        ),
        "treatment_parameter_delta": _parameter_distance(
            treatment_model, model
        ),
        "treatment_minus_baseline_parameter_delta": _parameter_distance(
            treatment_model, baseline_model
        ),
        "informative_batch_preserving_mass_before": before_mass,
        "informative_batch_preserving_mass_after_baseline": baseline_mass,
        "informative_batch_preserving_mass_after_treatment": treatment_mass,
        "treatment_minus_baseline_preserving_mass": (
            treatment_mass - baseline_mass
        ),
    }
    if expected_treatment_model is not None:
        invariant_names = _softmax_invariant_policy_bias_names(treatment_model)
        adam_step["persisted_treatment_replay_difference"] = (
            {
                "raw": _parameter_distance(
                    treatment_model,
                    expected_treatment_model,
                ),
                "functionally_relevant": _parameter_distance(
                    treatment_model,
                    expected_treatment_model,
                    excluded=invariant_names,
                ),
                "softmax_invariant_parameter_names": list(invariant_names),
            }
        )

    return {
        "support": support,
        "objectives": component_reports,
        "gradients": {
            "pairwise_applied_cosine": pairwise,
            "ordinary_gradient_l2": ordinary_norm,
            "ordinary_policy_head_gradient_l2": ordinary_policy_norm,
            "auxiliary_applied_gradient_l2": auxiliary_norm,
            "auxiliary_to_ordinary_gradient_l2_ratio": (
                auxiliary_norm / ordinary_norm if ordinary_norm > 0.0 else None
            ),
            "auxiliary_to_ordinary_cosine": _cosine(
                applied["auxiliary"], ordinary
            ),
            "auxiliary_to_ordinary_policy_head_gradient_l2_ratio": (
                auxiliary_norm / ordinary_policy_norm
                if ordinary_policy_norm > 0.0
                else None
            ),
            "auxiliary_to_ordinary_policy_head_cosine": _cosine(
                applied["auxiliary"], ordinary_policy
            ),
            "joint_policy_head_pre_clip_l2": joint_policy_norm,
            "joint_pre_clip_l2": joint_norm,
            "clip_threshold": grad_clip,
            "clip_scale": clip_scale,
            "joint_post_clip_l2": joint_norm * clip_scale,
        },
        "adam_step": adam_step,
        "original_model_unchanged": _model_digest(model) == model_before,
        "original_optimizer_unchanged": (
            _optimizer_digest(optimizer) == optimizer_before
        ),
        "interpretation": (
            "read-only audit of one persisted pre-update batch; not a training "
            "run, validation curve, strength result, or normalization decision"
        ),
    }
