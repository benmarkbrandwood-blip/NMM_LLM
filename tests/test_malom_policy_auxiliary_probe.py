"""Pure in-memory gradient diagnostics for preserving-set supervision."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from learned_ai.validation.malom_policy_auxiliary_probe import (
    MalomPolicyAuxiliaryProbeState,
    run_in_memory_auxiliary_probe,
    summarize_preserving_policy,
)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy[: features.shape[0]]


def _states() -> list[MalomPolicyAuxiliaryProbeState]:
    return [
        MalomPolicyAuxiliaryProbeState(
            phase="place",
            features=np.zeros((3, 1), dtype=np.float32),
            preserving_mask=np.asarray([True, True, False]),
        ),
        MalomPolicyAuxiliaryProbeState(
            phase="move",
            features=np.zeros((2, 1), dtype=np.float32),
            preserving_mask=np.asarray([True, True]),
        ),
    ]


def test_summary_separates_informative_and_all_safe_states() -> None:
    summary = summarize_preserving_policy(
        _Policy(),
        _states(),
        temperature=1.0,
        device=torch.device("cpu"),
    )

    assert summary["all"]["states"] == 2
    assert summary["all"]["informative_states"] == 1
    assert summary["all"]["all_safe_states"] == 1
    assert summary["all"]["mean_preserving_probability"] == pytest.approx(5 / 6)
    assert summary["all"]["mean_informative_loss"] == pytest.approx(-np.log(2 / 3))
    assert summary["move"]["mean_preserving_probability"] == 1.0
    assert summary["move"]["mean_informative_loss"] is None


def test_in_memory_probe_has_finite_gradient_and_improves_safe_mass() -> None:
    model = _Policy()
    before = model.policy.detach().clone()

    report = run_in_memory_auxiliary_probe(
        model,
        _states(),
        temperature=1.0,
        device=torch.device("cpu"),
        coefficients=(0.05, 0.1),
        step_size=0.25,
    )

    assert torch.equal(model.policy.detach(), before)
    assert report["original_model_unchanged"] is True
    assert report["gradient"]["finite"] is True
    assert report["gradient"]["l2_norm"] > 0.0
    assert report["gradient_alignment"]["directional_derivative"] > 0.0
    assert report["gradient_alignment"]["descent_cosine"] > 0.0
    for trial in report["coefficient_trials"]:
        assert trial["predicted_informative_preserving_probability_delta"] > 0.0
        assert trial["realized_informative_preserving_probability_delta"] >= 0.0
        assert trial["all_safe_max_probability_delta"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("temperature", "coefficients", "step_size", "match"),
    [
        (0.0, (0.1,), 0.1, "temperature"),
        (1.0, (), 0.1, "coefficient"),
        (1.0, (-0.1,), 0.1, "coefficient"),
        (1.0, (0.1,), 0.0, "step_size"),
    ],
)
def test_probe_rejects_invalid_controls(
    temperature: float,
    coefficients: tuple[float, ...],
    step_size: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_in_memory_auxiliary_probe(
            _Policy(),
            _states(),
            temperature=temperature,
            device=torch.device("cpu"),
            coefficients=coefficients,
            step_size=step_size,
        )
