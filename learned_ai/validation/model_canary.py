"""Deterministic single/batch canaries for ScaffoldedPolicyNet artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.training.run_contract import ContractValidationError, canonical_sha256


MODEL_CANARY_SCHEMA = "nmm.scaffolded-model-canary.v1"


@dataclass(frozen=True)
class ModelCanary:
    """Expected outputs for deterministic synthetic model inputs."""

    model_config_sha256: str
    policy_logits: tuple[float, ...]
    value_outputs: tuple[float, ...]

    _FIELDS: ClassVar[set[str]] = {
        "schema_version",
        "model_config_sha256",
        "policy_logits",
        "value_outputs",
    }

    def __post_init__(self) -> None:
        if len(self.model_config_sha256) != 64:
            raise ContractValidationError("model canary config identity is invalid")
        for field in ("policy_logits", "value_outputs"):
            values = tuple(float(item) for item in getattr(self, field))
            if not values or not all(math.isfinite(item) for item in values):
                raise ContractValidationError(
                    f"model canary {field} must contain finite values"
                )
            object.__setattr__(self, field, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_CANARY_SCHEMA,
            "model_config_sha256": self.model_config_sha256,
            "policy_logits": list(self.policy_logits),
            "value_outputs": list(self.value_outputs),
        }

    @property
    def identity(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelCanary:
        if not isinstance(value, dict) or set(value) != cls._FIELDS:
            raise ContractValidationError("model canary fields are invalid")
        if value["schema_version"] != MODEL_CANARY_SCHEMA:
            raise ContractValidationError("unsupported model canary schema")
        return cls(
            model_config_sha256=value["model_config_sha256"],
            policy_logits=tuple(value["policy_logits"]),
            value_outputs=tuple(value["value_outputs"]),
        )


def _inputs(
    model: ScaffoldedPolicyNet, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    policy = torch.linspace(
        -0.75,
        0.75,
        steps=7 * model.move_feat_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(7, model.move_feat_dim)
    value = torch.linspace(
        -0.5,
        0.5,
        steps=3 * model.value_input_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(3, model.value_input_dim)
    return policy, value


def capture_model_canary(
    model: ScaffoldedPolicyNet,
    *,
    device: str | torch.device = "cpu",
) -> ModelCanary:
    """Capture deterministic finite outputs and assert single/batch parity."""
    target = torch.device(device)
    original_device = next(model.parameters()).device
    model = model.to(target)
    model.eval()
    try:
        policy_input, value_input = _inputs(model, target)
        with torch.no_grad():
            policy_batch = model.policy_logits(policy_input)
            policy_single = model.policy_logits(policy_input[0])
            value_batch = model.value(value_input)
            value_single = model.value(value_input[0])
        if not torch.allclose(
            policy_single.reshape(1), policy_batch[:1], atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError("policy single/batch canary mismatch")
        if not torch.allclose(
            value_single.reshape(1), value_batch[:1], atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError("value single/batch canary mismatch")
        if not torch.isfinite(policy_batch).all() or not torch.isfinite(
            value_batch
        ).all():
            raise RuntimeError("model canary produced non-finite outputs")
        return ModelCanary(
            model_config_sha256=canonical_sha256(model.get_config()),
            policy_logits=tuple(float(item) for item in policy_batch.cpu()),
            value_outputs=tuple(float(item) for item in value_batch.cpu()),
        )
    finally:
        model.to(original_device)


def verify_model_canary(
    model: ScaffoldedPolicyNet,
    expected: ModelCanary,
    *,
    device: str | torch.device = "cpu",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, float]:
    """Verify model semantics and return maximum absolute output differences."""
    if canonical_sha256(model.get_config()) != expected.model_config_sha256:
        raise RuntimeError("model canary configuration mismatch")
    observed = capture_model_canary(model, device=device)
    policy_observed = torch.tensor(observed.policy_logits)
    policy_expected = torch.tensor(expected.policy_logits)
    value_observed = torch.tensor(observed.value_outputs)
    value_expected = torch.tensor(expected.value_outputs)
    if not torch.allclose(policy_observed, policy_expected, atol=atol, rtol=rtol):
        raise RuntimeError("model policy canary mismatch")
    if not torch.allclose(value_observed, value_expected, atol=atol, rtol=rtol):
        raise RuntimeError("model value canary mismatch")
    return {
        "policy_max_abs": float((policy_observed - policy_expected).abs().max()),
        "value_max_abs": float((value_observed - value_expected).abs().max()),
    }
