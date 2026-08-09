"""Focused tests for normalized target-response result decisions."""

from __future__ import annotations

import copy

import pytest

from learned_ai.evaluation.malom_policy_auxiliary_normalized_target_response_result import (
    NormalizedTargetResponseResultError,
    decide_normalized_target_response,
)


def _plan() -> dict:
    return {
        "schema_version": (
            "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-plan.v1"
        ),
        "method": {
            "target_policy_head_ratios": [0.25, 0.5, 1.0],
            "logged_scalar_tolerance": 1e-5,
            "production_replay_parameter_tolerance": 1e-6,
        },
        "decision_rule": {"monotonic_tolerance": 1e-9},
    }


def _response(target: float, delta: float, *, capped: bool = False) -> dict:
    return {
        "target_policy_head_ratio": target,
        "effective_coefficient": target / 10.0,
        "coefficient_capped": capped,
        "realized_policy_head_ratio": target,
        "audit": {
            "adam_step": {
                "treatment_minus_baseline_preserving_mass": delta,
                "treatment_minus_baseline_entropy": -1e-6,
                "baseline_to_treatment_policy_kl": {
                    "mean": 1e-7,
                    "max": 2e-7,
                },
            }
        },
    }


def _arm(seed: int, deltas: tuple[float, float, float]) -> dict:
    responses = [
        _response(target, delta)
        for target, delta in zip((0.25, 0.5, 1.0), deltas, strict=True)
    ]
    responses[0]["audit"]["adam_step"]["persisted_treatment_replay_difference"] = {
        "functionally_relevant": {"max_abs": 1e-8}
    }
    return {
        "seed": seed,
        "arm_id": f"seed{seed}-normalized-r025",
        "persisted_target_replay_residuals": {
            "policy_loss": 1e-7,
            "value_loss": 0.0,
        },
        "audit": {"responses": responses},
    }


def _audit() -> dict:
    return {
        "schema_version": (
            "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-result.v1"
        ),
        "mutation_checks": {
            "input_files_unchanged": True,
            "tracked_worktree_clean_after": True,
        },
        "arms": [
            _arm(55, (1e-6, 2e-6, 4e-6)),
            _arm(56, (2e-6, 3e-6, 5e-6)),
            _arm(57, (1e-6, 1.5e-6, 3e-6)),
        ],
    }


def test_monotonic_bounded_result_is_only_a_calibration_candidate() -> None:
    result = decide_normalized_target_response(_plan(), _audit())

    assert result["escalation_candidate"] is True
    assert result["verdict"] == (
        "eligible_to_prepare_independent_seed_learning_calibration"
    )
    assert all(result["checks"].values())
    assert result["median_preserving_mass_delta_by_target"] == {
        "0.25": 1e-6,
        "0.5": 2e-6,
        "1.0": 4e-6,
    }
    assert "no target selection" in result["claim_boundary"]


def test_one_non_monotonic_seed_stops_gradient_ratio_escalation() -> None:
    audit = _audit()
    audit["arms"][1] = _arm(56, (2e-6, 1e-6, -1e-6))

    result = decide_normalized_target_response(_plan(), audit)

    assert result["escalation_candidate"] is False
    assert result["verdict"] == "stop_gradient_ratio_escalation"
    assert result["checks"]["all_seeds_monotonic"] is False
    assert result["per_seed"][1]["monotonic"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda audit: audit["arms"].reverse(),
        lambda audit: audit["arms"][0]["audit"]["responses"].reverse(),
        lambda audit: audit["arms"][0]["audit"]["responses"][0]["audit"][
            "adam_step"
        ].update({"treatment_minus_baseline_entropy": float("nan")}),
    ],
)
def test_invalid_evidence_fails_closed(mutation) -> None:
    audit = copy.deepcopy(_audit())
    mutation(audit)

    with pytest.raises(NormalizedTargetResponseResultError):
        decide_normalized_target_response(_plan(), audit)
