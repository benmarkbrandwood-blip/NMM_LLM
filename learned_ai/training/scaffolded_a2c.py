"""learned_ai/training/scaffolded_a2c.py — A2C update for ScaffoldedPolicyNet.

The scaffolded model operates on variable-length move sets, so this update
function differs from a2c.py in two ways:

1. Policy loss: computed per-step (each step has a different k legal moves),
   accumulated into a single loss, then averaged.  No phase routing.

2. Value loss: computed in batch (value_input is always VALUE_INPUT_DIM floats),
   enabling efficient bootstrapping with one batched forward pass.

Both losses flow through a single backward() call so shared module parameters
(none here — policy and value heads are separate) would still receive correct
gradients.

ScaffoldedStep dataclass fields:
  move_features   (k, 62) np.ndarray  — features for current position's legal moves
  value_input     (23,)  np.ndarray   — board-level features for value head
  chosen_idx      int                 — which legal move was selected
  log_prob_old    float               — log P at collection time (for PPO ratio, unused in A2C)
  reward          float               — per-move shaped reward
  next_move_features  (k', 62) np.ndarray
  next_value_input    (23,)  np.ndarray
  done            bool
  bootstrap_perspective str  — whether next_value_input is same/opponent view
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
GRAD_CLIP    = 1.0
MIN_UPDATE_STEPS = 8
MALOM_POLICY_AUX_MODES = ("fixed", "policy-head-normalized")
DEFAULT_MALOM_POLICY_AUX_TARGET_RATIO = 0.25
DEFAULT_MALOM_POLICY_AUX_COEF_CAP = 0.25
DEFAULT_MALOM_POLICY_AUX_DENOMINATOR_FLOOR = 1e-12


class NonFiniteTrainingError(RuntimeError):
    """Stop training when an update can no longer preserve finite state."""


@dataclass
class ScaffoldedStep:
    """One learner-turn step for the scaffolded A2C update."""

    move_features:      np.ndarray   # (k, 62)
    value_input:        np.ndarray   # (23,)
    chosen_idx:         int
    log_prob_old:       float
    reward:             float
    next_move_features: np.ndarray   # (k', 62) — for optional bootstrapping
    next_value_input:   np.ndarray   # (23,)
    done:               bool
    behaviour_temperature: float = 1.0
    bootstrap_perspective: str = "same"
    malom_preserving_mask: Optional[np.ndarray] = None


def _behaviour_temperature(step: ScaffoldedStep) -> float:
    """Return the recorded collection temperature or reject old pending data."""
    if "behaviour_temperature" not in vars(step):
        raise NonFiniteTrainingError(
            "A2C: pending step predates behaviour-temperature tracking"
        )
    try:
        value = float(step.behaviour_temperature)
    except (TypeError, ValueError) as exc:
        raise NonFiniteTrainingError(
            "A2C: invalid behaviour_temperature value"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise NonFiniteTrainingError(
            f"A2C: invalid behaviour_temperature={value!r}"
        )
    return value


def _bootstrap_sign(step: ScaffoldedStep) -> float:
    """Map the recorded successor perspective onto the current mover."""
    if "bootstrap_perspective" not in vars(step):
        raise NonFiniteTrainingError(
            "A2C: pending step predates bootstrap-perspective tracking"
        )
    perspective = step.bootstrap_perspective
    if perspective == "same":
        return 1.0
    if perspective == "opponent":
        return -1.0
    raise NonFiniteTrainingError(
        f"A2C: invalid bootstrap_perspective={perspective!r}"
    )


def _require_finite(tensor: torch.Tensor, *, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise NonFiniteTrainingError(f"A2C: non-finite {label}")


def _malom_preserving_mask(
    step: ScaffoldedStep,
    *,
    legal_move_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return an exact per-action preserving mask or fail closed."""
    raw_mask = getattr(step, "malom_preserving_mask", None)
    if raw_mask is None:
        raise NonFiniteTrainingError("A2C: missing Malom preserving mask")
    mask = np.asarray(raw_mask)
    if mask.ndim != 1 or len(mask) != legal_move_count:
        raise NonFiniteTrainingError(
            "A2C: Malom preserving mask length does not match legal moves"
        )
    if mask.dtype != np.bool_:
        raise NonFiniteTrainingError("A2C: Malom preserving mask must be boolean")
    if not bool(mask.any()):
        raise NonFiniteTrainingError("A2C: Malom label set has no preserving action")
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def malom_preserving_set_loss(
    log_probs: torch.Tensor,
    preserving_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``-log P(preserving set)`` for one legal action set."""
    if log_probs.ndim != 1 or preserving_mask.ndim != 1:
        raise NonFiniteTrainingError("A2C: preserving-set inputs must be vectors")
    if log_probs.shape != preserving_mask.shape:
        raise NonFiniteTrainingError("A2C: preserving-set shape mismatch")
    if preserving_mask.dtype != torch.bool:
        raise NonFiniteTrainingError("A2C: preserving-set mask must be boolean")
    if not bool(preserving_mask.any()):
        raise NonFiniteTrainingError("A2C: preserving set is empty")
    _require_finite(log_probs, label="preserving-set log probabilities")
    return -torch.logsumexp(log_probs[preserving_mask], dim=0)


def malom_policy_auxiliary_enabled(
    *,
    mode: str,
    fixed_coefficient: float,
) -> bool:
    """Return whether exact-WDL action labels are required for this update."""
    return mode == "policy-head-normalized" or fixed_coefficient > 0.0


def _detached_gradients(
    objective: torch.Tensor,
    parameters: Sequence[nn.Parameter],
) -> tuple[torch.Tensor, ...]:
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    detached = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, gradients, strict=True)
    )
    if not all(bool(torch.isfinite(gradient).all()) for gradient in detached):
        raise NonFiniteTrainingError(
            "A2C: non-finite Malom policy auxiliary scale gradient"
        )
    return detached


def _gradient_dot(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> float:
    return float(
        sum(
            (first.double() * second.double()).sum().item()
            for first, second in zip(left, right, strict=True)
        )
    )


def _gradient_l2(gradients: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(0.0, _gradient_dot(gradients, gradients)))


def _gradient_cosine(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
) -> float | None:
    denominator = _gradient_l2(left) * _gradient_l2(right)
    if denominator == 0.0:
        return None
    return _gradient_dot(left, right) / denominator


def _step_phase(step: ScaffoldedStep) -> str:
    """Return the production phase encoded by the first four move features."""
    features = np.asarray(step.move_features)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] < 4:
        return "unknown"
    one_hots = features[:, :4]
    reference = one_hots[0]
    if not np.allclose(one_hots, reference, rtol=0.0, atol=0.0):
        return "unknown"
    expected = np.eye(4, dtype=reference.dtype)[int(np.argmax(reference))]
    if not np.array_equal(reference, expected):
        return "unknown"
    return ("placement", "movement", "movement", "flying")[
        int(np.argmax(reference))
    ]


def scaffolded_a2c_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: List[ScaffoldedStep],
    device: torch.device,
    gamma: float = 0.99,
    entropy_coef: float = ENTROPY_COEF,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    min_batch: int = MIN_UPDATE_STEPS,
    malom_policy_aux_coef: float = 0.0,
    malom_policy_aux_mode: str = "fixed",
    malom_policy_aux_target_ratio: float = DEFAULT_MALOM_POLICY_AUX_TARGET_RATIO,
    malom_policy_aux_coef_cap: float = DEFAULT_MALOM_POLICY_AUX_COEF_CAP,
    malom_policy_aux_denominator_floor: float = (
        DEFAULT_MALOM_POLICY_AUX_DENOMINATOR_FLOOR
    ),
    diagnostics: Optional[dict[str, Any]] = None,
) -> tuple[float, float, float]:
    """One A2C gradient update over a batch of ScaffoldedSteps.

    Returns (policy_loss, value_loss, entropy) as Python floats.
    Returns (0, 0, 0) if batch is too small.
    """
    try:
        malom_policy_aux_coef = float(malom_policy_aux_coef)
    except (TypeError, ValueError) as exc:
        raise NonFiniteTrainingError(
            "A2C: invalid Malom policy auxiliary coefficient"
        ) from exc
    if not math.isfinite(malom_policy_aux_coef) or malom_policy_aux_coef < 0.0:
        raise NonFiniteTrainingError(
            "A2C: Malom policy auxiliary coefficient must be finite and non-negative"
        )
    if malom_policy_aux_mode not in MALOM_POLICY_AUX_MODES:
        raise NonFiniteTrainingError(
            f"A2C: unsupported Malom policy auxiliary mode={malom_policy_aux_mode!r}"
        )
    try:
        malom_policy_aux_target_ratio = float(malom_policy_aux_target_ratio)
        malom_policy_aux_coef_cap = float(malom_policy_aux_coef_cap)
        malom_policy_aux_denominator_floor = float(
            malom_policy_aux_denominator_floor
        )
    except (TypeError, ValueError) as exc:
        raise NonFiniteTrainingError(
            "A2C: invalid Malom policy auxiliary normalization setting"
        ) from exc
    if (
        not math.isfinite(malom_policy_aux_target_ratio)
        or malom_policy_aux_target_ratio <= 0.0
        or not math.isfinite(malom_policy_aux_coef_cap)
        or malom_policy_aux_coef_cap <= 0.0
        or not math.isfinite(malom_policy_aux_denominator_floor)
        or malom_policy_aux_denominator_floor <= 0.0
    ):
        raise NonFiniteTrainingError(
            "A2C: Malom policy auxiliary normalization settings must be "
            "finite and positive"
        )
    if (
        malom_policy_aux_mode == "policy-head-normalized"
        and malom_policy_aux_coef != 0.0
    ):
        raise NonFiniteTrainingError(
            "A2C: normalized Malom policy auxiliary mode requires fixed "
            "coefficient zero"
        )

    auxiliary_enabled = malom_policy_auxiliary_enabled(
        mode=malom_policy_aux_mode,
        fixed_coefficient=malom_policy_aux_coef,
    )

    if len(steps) < min_batch:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "malom_policy_aux_loss": 0.0,
                    "malom_policy_aux_informative_steps": 0,
                    "malom_policy_aux_labelled_steps": 0,
                    "malom_policy_aux_mean_preserving_mass": 0.0,
                    "malom_policy_aux_mode": malom_policy_aux_mode,
                    "malom_policy_aux_effective_coef": 0.0,
                    "malom_policy_aux_scale_status": "undersized_batch",
                }
            )
        return 0.0, 0.0, 0.0

    # ── Batch value inputs (fixed size — easy to stack) ────────────────────────
    all_vi  = torch.tensor(
        np.stack([s.value_input      for s in steps]), dtype=torch.float32
    ).to(device)  # (B, 23)
    all_nvi = torch.tensor(
        np.stack([s.next_value_input for s in steps]), dtype=torch.float32
    ).to(device)  # (B, 23)
    rewards = torch.tensor(
        [s.reward for s in steps], dtype=torch.float32, device=device
    )                                                    # (B,)
    dones   = torch.tensor(
        [float(s.done) for s in steps], dtype=torch.float32, device=device
    )                                                    # (B,)
    bootstrap_signs = torch.tensor(
        [_bootstrap_sign(s) for s in steps],
        dtype=torch.float32,
        device=device,
    )                                                    # (B,)

    model.train()

    # ── Bootstrap: V(next_state) — no gradient ────────────────────────────────
    with torch.no_grad():
        v_next = model.value(all_nvi) * bootstrap_signs # (B,)
        v_next = v_next * (1.0 - dones)

    td_targets = rewards + gamma * v_next               # (B,)

    # ── Current value (with gradient) ─────────────────────────────────────────
    v_curr = model.value(all_vi)                        # (B,)

    # ── Advantage (detached for policy gradient) ───────────────────────────────
    advantages = (td_targets - v_curr).detach()         # (B,)
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── Policy loss + entropy (per-step, variable k) ───────────────────────────
    policy_terms:  list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    malom_aux_terms: list[torch.Tensor] = []
    malom_preserving_masses: list[float] = []
    labelled_by_phase: dict[str, int] = {}
    informative_by_phase: dict[str, int] = {}

    for i, step in enumerate(steps):
        feat   = torch.tensor(step.move_features, dtype=torch.float32).to(device)
        logits = model.policy_logits(feat)              # (k,)
        scaled = logits / _behaviour_temperature(step)
        log_probs = F.log_softmax(scaled, dim=-1)       # (k,)
        policy_terms.append(-log_probs[step.chosen_idx] * advantages[i])
        probs = log_probs.exp()
        entropy_terms.append(-(probs * log_probs).sum())
        if auxiliary_enabled:
            preserving = _malom_preserving_mask(
                step,
                legal_move_count=len(log_probs),
                device=device,
            )
            auxiliary_loss = malom_preserving_set_loss(log_probs, preserving)
            log_preserving_mass = -auxiliary_loss
            malom_preserving_masses.append(
                float(log_preserving_mass.detach().exp().item())
            )
            phase = _step_phase(step)
            labelled_by_phase[phase] = labelled_by_phase.get(phase, 0) + 1
            if not bool(preserving.all()):
                malom_aux_terms.append(auxiliary_loss)
                informative_by_phase[phase] = (
                    informative_by_phase.get(phase, 0) + 1
                )

    policy_loss  = torch.stack(policy_terms).mean()
    entropy_loss = torch.stack(entropy_terms).mean()
    value_loss   = F.mse_loss(v_curr, td_targets.detach())
    malom_aux_loss = (
        torch.stack(malom_aux_terms).mean()
        if malom_aux_terms
        else policy_loss.new_zeros(())
    )

    effective_aux_coefficient = malom_policy_aux_coef
    scale_diagnostics: dict[str, Any] = {
        "malom_policy_aux_mode": malom_policy_aux_mode,
        "malom_policy_aux_effective_coef": effective_aux_coefficient,
        "malom_policy_aux_scale_status": (
            "fixed" if auxiliary_enabled else "disabled"
        ),
    }
    if malom_policy_aux_mode == "policy-head-normalized":
        parameters = tuple(model.parameters())
        ordinary_policy_objective = policy_loss - entropy_coef * entropy_loss
        ordinary_gradients = _detached_gradients(
            ordinary_policy_objective,
            parameters,
        )
        ordinary_norm = _gradient_l2(ordinary_gradients)
        raw_auxiliary_norm = 0.0
        applied_auxiliary_norm = 0.0
        cosine = None
        capped = False
        if not malom_aux_terms:
            effective_aux_coefficient = 0.0
            scale_status = "no_informative_steps"
        else:
            auxiliary_gradients = _detached_gradients(
                malom_aux_loss,
                parameters,
            )
            raw_auxiliary_norm = _gradient_l2(auxiliary_gradients)
            cosine = _gradient_cosine(auxiliary_gradients, ordinary_gradients)
            if raw_auxiliary_norm <= malom_policy_aux_denominator_floor:
                raise NonFiniteTrainingError(
                    "A2C: informative Malom auxiliary gradient is below the "
                    "normalization denominator floor"
                )
            if ordinary_norm <= malom_policy_aux_denominator_floor:
                effective_aux_coefficient = 0.0
                scale_status = "ordinary_policy_gradient_below_floor"
            else:
                uncapped = (
                    malom_policy_aux_target_ratio
                    * ordinary_norm
                    / raw_auxiliary_norm
                )
                effective_aux_coefficient = min(
                    uncapped,
                    malom_policy_aux_coef_cap,
                )
                capped = effective_aux_coefficient < uncapped
                scale_status = "capped" if capped else "normalized"
                applied_auxiliary_norm = (
                    effective_aux_coefficient * raw_auxiliary_norm
                )
        scale_diagnostics = {
            "malom_policy_aux_mode": malom_policy_aux_mode,
            "malom_policy_aux_scale_status": scale_status,
            "malom_policy_aux_target_policy_head_ratio": (
                malom_policy_aux_target_ratio
            ),
            "malom_policy_aux_coef_cap": malom_policy_aux_coef_cap,
            "malom_policy_aux_denominator_floor": (
                malom_policy_aux_denominator_floor
            ),
            "malom_policy_aux_effective_coef": effective_aux_coefficient,
            "malom_policy_aux_coefficient_capped": capped,
            "malom_policy_aux_ordinary_policy_head_gradient_l2": ordinary_norm,
            "malom_policy_aux_raw_auxiliary_gradient_l2": raw_auxiliary_norm,
            "malom_policy_aux_applied_auxiliary_gradient_l2": (
                applied_auxiliary_norm
            ),
            "malom_policy_aux_applied_to_ordinary_policy_head_ratio": (
                applied_auxiliary_norm / ordinary_norm
                if ordinary_norm > malom_policy_aux_denominator_floor
                else 0.0
            ),
            "malom_policy_auxiliary_to_ordinary_policy_head_cosine": cosine,
        }

    total_loss = (
        policy_loss
        - entropy_coef * entropy_loss
        + value_coef * value_loss
        + effective_aux_coefficient * malom_aux_loss
    )
    _require_finite(total_loss, label="total loss")

    optimizer.zero_grad()
    total_loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            _require_finite(parameter.grad, label=f"gradient in {name}")
    try:
        nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise NonFiniteTrainingError("A2C: non-finite gradient norm") from exc
    optimizer.step()
    for name, parameter in model.named_parameters():
        _require_finite(parameter, label=f"parameter {name} after optimizer step")

    if diagnostics is not None:
        diagnostics.update(
            {
                "malom_policy_aux_loss": float(malom_aux_loss.item()),
                "malom_policy_aux_informative_steps": len(malom_aux_terms),
                "malom_policy_aux_labelled_steps": len(malom_preserving_masses),
                "malom_policy_aux_mean_preserving_mass": (
                    float(sum(malom_preserving_masses) / len(malom_preserving_masses))
                    if malom_preserving_masses
                    else 0.0
                ),
                "malom_policy_aux_labelled_by_phase": labelled_by_phase,
                "malom_policy_aux_informative_by_phase": informative_by_phase,
                **scale_diagnostics,
            }
        )

    return (
        float(policy_loss.item()),
        float(value_loss.item()),
        float(entropy_loss.item()),
    )


def scaffolded_ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: List[ScaffoldedStep],
    device: torch.device,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    epochs: int = 4,
    entropy_coef: float = ENTROPY_COEF,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    min_batch: int = MIN_UPDATE_STEPS,
) -> tuple[float, float, float]:
    """PPO clipped surrogate update over ScaffoldedSteps.

    Returns (policy_loss, value_loss, entropy) averaged over epochs.
    """
    if len(steps) < min_batch:
        return 0.0, 0.0, 0.0

    # Pre-compute TD targets (no grad needed)
    all_vi  = torch.tensor(
        np.stack([s.value_input      for s in steps]), dtype=torch.float32
    ).to(device)
    all_nvi = torch.tensor(
        np.stack([s.next_value_input for s in steps]), dtype=torch.float32
    ).to(device)
    rewards = torch.tensor(
        [s.reward for s in steps], dtype=torch.float32, device=device
    )
    dones   = torch.tensor(
        [float(s.done) for s in steps], dtype=torch.float32, device=device
    )
    bootstrap_signs = torch.tensor(
        [_bootstrap_sign(s) for s in steps],
        dtype=torch.float32,
        device=device,
    )
    log_probs_old = torch.tensor(
        [s.log_prob_old for s in steps], dtype=torch.float32, device=device
    )

    with torch.no_grad():
        v_next = model.value(all_nvi) * bootstrap_signs * (1.0 - dones)
        td_targets = (rewards + gamma * v_next).detach()
        with torch.no_grad():
            v0    = model.value(all_vi)
        advantages = td_targets - v0
        if advantages.std() > 1e-3:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pl_acc, vl_acc, ent_acc = 0.0, 0.0, 0.0

    model.train()
    for _ in range(epochs):
        v_curr = model.value(all_vi)
        policy_terms:  list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []

        for i, step in enumerate(steps):
            feat      = torch.tensor(step.move_features, dtype=torch.float32).to(device)
            logits    = model.policy_logits(feat)
            log_probs = F.log_softmax(logits, dim=-1)
            lp        = log_probs[step.chosen_idx]
            ratio     = torch.exp(lp - log_probs_old[i])
            adv       = advantages[i]
            surr1     = ratio * adv
            surr2     = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_terms.append(-torch.min(surr1, surr2))
            probs = log_probs.exp()
            entropy_terms.append(-(probs * log_probs).sum())

        policy_loss  = torch.stack(policy_terms).mean()
        entropy_loss = torch.stack(entropy_terms).mean()
        value_loss   = F.mse_loss(v_curr, td_targets)
        total_loss   = policy_loss - entropy_coef * entropy_loss + value_coef * value_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        pl_acc  += float(policy_loss.item())
        vl_acc  += float(value_loss.item())
        ent_acc += float(entropy_loss.item())

    return pl_acc / epochs, vl_acc / epochs, ent_acc / epochs
