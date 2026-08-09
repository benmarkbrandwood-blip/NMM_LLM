"""Pure in-memory diagnostics for exact-WDL preserving-set supervision."""

from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from learned_ai.training.scaffolded_a2c import malom_preserving_set_loss


@dataclass(frozen=True)
class MalomPolicyAuxiliaryProbeState:
    """One fixed production-route feature matrix and exact preserving set."""

    phase: str
    features: np.ndarray
    preserving_mask: np.ndarray


def _validated_states(
    states: Iterable[MalomPolicyAuxiliaryProbeState],
) -> list[MalomPolicyAuxiliaryProbeState]:
    state_list = list(states)
    if not state_list:
        raise ValueError("policy auxiliary probe requires at least one state")
    for state in state_list:
        features = np.asarray(state.features)
        preserving = np.asarray(state.preserving_mask)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("probe features must be a non-empty matrix")
        if preserving.dtype != np.bool_:
            raise ValueError("probe preserving mask must be boolean")
        if preserving.ndim != 1 or preserving.shape[0] != features.shape[0]:
            raise ValueError("probe preserving mask does not align with features")
        if not bool(preserving.any()):
            raise ValueError("probe preserving set is empty")
        if not isinstance(state.phase, str) or not state.phase:
            raise ValueError("probe phase must be a non-empty string")
    return state_list


def _model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_measurements(
    model: torch.nn.Module,
    states: Sequence[MalomPolicyAuxiliaryProbeState],
    *,
    temperature: float,
    device: torch.device,
) -> list[dict[str, float | int | str | bool]]:
    measurements: list[dict[str, float | int | str | bool]] = []
    was_training = model.training
    model.eval()
    try:
        for state in states:
            features = torch.as_tensor(
                state.features,
                dtype=torch.float32,
                device=device,
            )
            preserving = torch.as_tensor(
                state.preserving_mask,
                dtype=torch.bool,
                device=device,
            )
            with torch.no_grad():
                logits = model.policy_logits(features)
                if logits.shape != preserving.shape:
                    raise ValueError("probe policy logits do not align with actions")
                log_probs = F.log_softmax(logits / temperature, dim=-1)
                loss = malom_preserving_set_loss(log_probs, preserving)
                probabilities = log_probs.exp()
                preserving_mass = probabilities[preserving].sum()
                entropy = -(probabilities * log_probs).sum()
            measurements.append(
                {
                    "phase": state.phase,
                    "informative": not bool(preserving.all()),
                    "preserving_probability": float(preserving_mass.item()),
                    "loss": float(loss.item()),
                    "entropy": float(entropy.item()),
                }
            )
    finally:
        model.train(was_training)
    return measurements


def summarize_preserving_policy(
    model: torch.nn.Module,
    states: Iterable[MalomPolicyAuxiliaryProbeState],
    *,
    temperature: float,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Summarize preserving probability without mutating model parameters."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    state_list = _validated_states(states)
    rows = _state_measurements(
        model,
        state_list,
        temperature=temperature,
        device=device,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped[str(row["phase"])].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(grouped.items()):
        informative = [row for row in group_rows if row["informative"]]
        all_safe = [row for row in group_rows if not row["informative"]]

        def mean(key: str, selected: Sequence[dict[str, Any]]) -> float | None:
            if not selected:
                return None
            return float(sum(float(row[key]) for row in selected) / len(selected))

        summary[group] = {
            "states": len(group_rows),
            "informative_states": len(informative),
            "all_safe_states": len(all_safe),
            "mean_preserving_probability": mean(
                "preserving_probability",
                group_rows,
            ),
            "mean_informative_preserving_probability": mean(
                "preserving_probability",
                informative,
            ),
            "mean_all_safe_preserving_probability": mean(
                "preserving_probability",
                all_safe,
            ),
            "mean_informative_loss": mean("loss", informative),
            "mean_entropy": mean("entropy", group_rows),
        }
    return summary


def _mean_auxiliary_loss(
    model: torch.nn.Module,
    states: Sequence[MalomPolicyAuxiliaryProbeState],
    *,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for state in states:
        preserving = torch.as_tensor(
            state.preserving_mask,
            dtype=torch.bool,
            device=device,
        )
        if bool(preserving.all()):
            continue
        features = torch.as_tensor(
            state.features,
            dtype=torch.float32,
            device=device,
        )
        logits = model.policy_logits(features)
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        terms.append(malom_preserving_set_loss(log_probs, preserving))
    if not terms:
        raise ValueError("policy auxiliary probe found no informative state")
    return torch.stack(terms).mean()


def _gradient_summary(model: torch.nn.Module) -> dict[str, float | bool | int]:
    squared_norm = 0.0
    parameters_with_gradient = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        parameters_with_gradient += 1
        finite = finite and bool(torch.isfinite(parameter.grad).all())
        squared_norm += float(parameter.grad.detach().double().square().sum().item())
    return {
        "finite": finite,
        "parameters_with_gradient": parameters_with_gradient,
        "l2_norm": math.sqrt(squared_norm),
    }


def run_in_memory_auxiliary_probe(
    model: torch.nn.Module,
    states: Iterable[MalomPolicyAuxiliaryProbeState],
    *,
    temperature: float,
    device: torch.device,
    coefficients: Sequence[float],
    step_size: float,
) -> dict[str, Any]:
    """Measure gradients and isolated SGD directions on disposable model copies."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not coefficients or any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in coefficients
    ):
        raise ValueError("coefficient candidates must be finite and positive")
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    state_list = _validated_states(states)
    original_digest = _model_digest(model)
    baseline = summarize_preserving_policy(
        model,
        state_list,
        temperature=temperature,
        device=device,
    )

    gradient_model = copy.deepcopy(model).to(device)
    gradient_model.train()
    gradient_model.zero_grad(set_to_none=True)
    auxiliary_loss = _mean_auxiliary_loss(
        gradient_model,
        state_list,
        temperature=temperature,
        device=device,
    )
    auxiliary_loss.backward()
    gradient = _gradient_summary(gradient_model)
    if not gradient["finite"] or gradient["parameters_with_gradient"] == 0:
        raise ValueError("policy auxiliary gradient is missing or non-finite")

    trials: list[dict[str, Any]] = []
    before_all = baseline["all"]
    before_informative = float(
        before_all["mean_informative_preserving_probability"]
    )
    before_all_safe = before_all["mean_all_safe_preserving_probability"]
    for raw_coefficient in coefficients:
        coefficient = float(raw_coefficient)
        candidate = copy.deepcopy(model).to(device)
        candidate.train()
        candidate.zero_grad(set_to_none=True)
        loss = _mean_auxiliary_loss(
            candidate,
            state_list,
            temperature=temperature,
            device=device,
        )
        (coefficient * loss).backward()
        candidate_gradient = _gradient_summary(candidate)
        if not candidate_gradient["finite"]:
            raise ValueError("scaled policy auxiliary gradient is non-finite")
        with torch.no_grad():
            for parameter in candidate.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-step_size)
        after = summarize_preserving_policy(
            candidate,
            state_list,
            temperature=temperature,
            device=device,
        )
        after_all = after["all"]
        after_informative = float(
            after_all["mean_informative_preserving_probability"]
        )
        after_all_safe = after_all["mean_all_safe_preserving_probability"]
        all_safe_delta = (
            0.0
            if before_all_safe is None and after_all_safe is None
            else abs(float(after_all_safe) - float(before_all_safe))
        )
        trials.append(
            {
                "coefficient": coefficient,
                "scaled_gradient_l2_norm": candidate_gradient["l2_norm"],
                "informative_preserving_probability_delta": (
                    after_informative - before_informative
                ),
                "all_safe_max_probability_delta": all_safe_delta,
                "entropy_delta": (
                    float(after_all["mean_entropy"])
                    - float(before_all["mean_entropy"])
                ),
                "after": after,
            }
        )

    return {
        "baseline": baseline,
        "auxiliary_loss": float(auxiliary_loss.detach().item()),
        "gradient": gradient,
        "coefficient_trials": trials,
        "original_model_unchanged": _model_digest(model) == original_digest,
        "interpretation": (
            "isolated in-memory SGD direction only; not an Adam trajectory, "
            "training result, validation result, or strength claim"
        ),
    }
