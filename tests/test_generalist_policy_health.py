"""Focused tests for fixed-state Generalist policy-health summaries."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from learned_ai.validation.generalist_policy_health import (
    PolicyHealthState,
    summarize_direct_lookahead_signal,
    summarize_policy_health,
)


class _FirstFeaturePolicy(torch.nn.Module):
    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


def _states() -> list[PolicyHealthState]:
    return [
        PolicyHealthState(
            phase="movement",
            features=np.asarray([[2.0, 3.0], [0.0, 1.0]], dtype=np.float32),
            malom_qualities=np.asarray([0.0, -1.0], dtype=np.float64),
            heuristic_top1_idx=0,
        ),
        PolicyHealthState(
            phase="flying",
            features=np.asarray([[0.0, 4.0], [1.0, 2.0]], dtype=np.float32),
            malom_qualities=np.asarray([0.0, -2.0], dtype=np.float64),
            heuristic_top1_idx=1,
        ),
    ]


def test_summary_exposes_preserving_and_downgrading_direction() -> None:
    summary = summarize_policy_health(
        _FirstFeaturePolicy(),
        _states(),
        temperature=1.0,
        device=torch.device("cpu"),
    )

    all_metrics = summary["metrics"]["all"]
    assert summary["argmax_malom_quality_counts"] == {"-2": 1, "0": 1}
    assert all_metrics["critical_states"] == 2
    assert all_metrics["critical_argmax_value_preserving_rate"] == 0.5
    assert all_metrics[
        "critical_mean_preserving_minus_downgrading_logit"
    ] == pytest.approx(0.5)
    assert all_metrics["mean_argmax_malom_quality"] == -1.0
    assert 0.0 < all_metrics[
        "critical_value_preserving_probability_mass_scheduled"
    ] < 1.0
    assert math.isfinite(all_metrics["mean_entropy_scheduled"])


def test_direct_signal_audit_is_independent_of_policy_logits() -> None:
    audit = summarize_direct_lookahead_signal(_states(), signal_column=1)

    assert audit == {
        "candidate_rows": 4,
        "critical_states": 2,
        "argmax_value_preserving_rate": 1.0,
        "mean_preserving_minus_downgrading_signal": 2.0,
    }


def test_positive_malom_quality_is_rejected() -> None:
    state = PolicyHealthState(
        phase="placement",
        features=np.asarray([[0.0], [1.0]], dtype=np.float32),
        malom_qualities=np.asarray([0.0, 1.0], dtype=np.float64),
        heuristic_top1_idx=0,
    )

    with pytest.raises(ValueError, match="cannot be positive"):
        summarize_policy_health(
            _FirstFeaturePolicy(),
            [state],
            temperature=1.0,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_policy_health(
            _FirstFeaturePolicy(),
            _states(),
            temperature=temperature,
            device=torch.device("cpu"),
        )
