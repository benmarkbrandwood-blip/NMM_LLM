"""Decision logic for the normalized Malom auxiliary target-response audit."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-plan.v1"
AUDIT_SCHEMA = "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-result.v1"


class NormalizedTargetResponseResultError(ValueError):
    """Raised when target-response evidence cannot be interpreted safely."""


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizedTargetResponseResultError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise NormalizedTargetResponseResultError(f"{field} is non-finite")
    return result


def _ordered_responses(
    arm: Mapping[str, Any],
    *,
    targets: Sequence[float],
) -> list[Mapping[str, Any]]:
    audit = arm.get("audit")
    if not isinstance(audit, Mapping):
        raise NormalizedTargetResponseResultError("arm audit is missing")
    responses = audit.get("responses")
    if not isinstance(responses, list):
        raise NormalizedTargetResponseResultError("arm responses are missing")
    observed = [
        _finite(item.get("target_policy_head_ratio"), field="target ratio")
        for item in responses
        if isinstance(item, Mapping)
    ]
    if observed != list(targets) or len(responses) != len(targets):
        raise NormalizedTargetResponseResultError("target response order differs")
    return responses


def decide_normalized_target_response(
    plan: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen target-response screen to one immutable raw audit."""
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise NormalizedTargetResponseResultError("plan schema differs")
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise NormalizedTargetResponseResultError("audit schema differs")
    decision_rule = plan.get("decision_rule")
    method = plan.get("method")
    if not isinstance(decision_rule, Mapping) or not isinstance(method, Mapping):
        raise NormalizedTargetResponseResultError("frozen decision inputs are missing")
    targets = tuple(
        _finite(value, field="target ratio")
        for value in method.get("target_policy_head_ratios", [])
    )
    if targets != (0.25, 0.5, 1.0):
        raise NormalizedTargetResponseResultError("target ratio contract differs")
    monotonic_tolerance = _finite(
        decision_rule.get("monotonic_tolerance"),
        field="monotonic tolerance",
    )
    scalar_tolerance = _finite(
        method.get("logged_scalar_tolerance"),
        field="logged scalar tolerance",
    )
    parameter_tolerance = _finite(
        method.get("production_replay_parameter_tolerance"),
        field="replay parameter tolerance",
    )
    if min(monotonic_tolerance, scalar_tolerance, parameter_tolerance) < 0.0:
        raise NormalizedTargetResponseResultError("tolerance must be non-negative")

    mutation = audit.get("mutation_checks")
    mutation_safe = isinstance(mutation, Mapping) and all(
        mutation.get(field) is True
        for field in ("input_files_unchanged", "tracked_worktree_clean_after")
    )
    arms = audit.get("arms")
    expected_arms = [
        (55, "seed55-normalized-r025"),
        (56, "seed56-normalized-r025"),
        (57, "seed57-normalized-r025"),
    ]
    if (
        not isinstance(arms, list)
        or [
            (arm.get("seed"), arm.get("arm_id"))
            for arm in arms
            if isinstance(arm, Mapping)
        ]
        != expected_arms
    ):
        raise NormalizedTargetResponseResultError("audit arm cohort differs")

    per_seed: list[dict[str, Any]] = []
    deltas_by_target: dict[float, list[float]] = {target: [] for target in targets}
    all_replays_pass = True
    all_bounded = True
    all_monotonic = True
    target_half_uncapped = True
    maximum_kl = 0.0
    maximum_abs_entropy_change = 0.0
    for arm in arms:
        responses = _ordered_responses(arm, targets=targets)
        residuals = arm.get("persisted_target_replay_residuals")
        if not isinstance(residuals, Mapping):
            raise NormalizedTargetResponseResultError("replay residuals are missing")
        scalar_replay_passed = all(
            abs(_finite(value, field=f"replay residual {name}")) <= scalar_tolerance
            for name, value in residuals.items()
        )
        replay_difference = (
            responses[0]
            .get("audit", {})
            .get("adam_step", {})
            .get("persisted_treatment_replay_difference")
        )
        if not isinstance(replay_difference, Mapping) or not isinstance(
            replay_difference.get("functionally_relevant"), Mapping
        ):
            raise NormalizedTargetResponseResultError(
                "persisted replay difference is missing"
            )
        parameter_replay_passed = (
            _finite(
                replay_difference["functionally_relevant"].get("max_abs"),
                field="functional replay max abs",
            )
            <= parameter_tolerance
        )
        replay_passed = scalar_replay_passed and parameter_replay_passed
        all_replays_pass = all_replays_pass and replay_passed

        response_rows: list[dict[str, Any]] = []
        deltas: list[float] = []
        seed_bounded = True
        for response in responses:
            target = _finite(
                response.get("target_policy_head_ratio"),
                field="target ratio",
            )
            adam = response.get("audit", {}).get("adam_step")
            if not isinstance(adam, Mapping):
                raise NormalizedTargetResponseResultError("Adam response is missing")
            delta = _finite(
                adam.get("treatment_minus_baseline_preserving_mass"),
                field="preserving-mass delta",
            )
            entropy_delta = _finite(
                adam.get("treatment_minus_baseline_entropy"),
                field="entropy delta",
            )
            kl = adam.get("baseline_to_treatment_policy_kl")
            if not isinstance(kl, Mapping):
                raise NormalizedTargetResponseResultError("policy KL is missing")
            kl_mean = _finite(kl.get("mean"), field="mean policy KL")
            kl_max = _finite(kl.get("max"), field="maximum policy KL")
            if kl_mean < 0.0 or kl_max < 0.0:
                raise NormalizedTargetResponseResultError("policy KL is negative")
            bounded = kl_mean <= 0.0001 and abs(entropy_delta) <= 0.01
            seed_bounded = seed_bounded and bounded
            maximum_kl = max(maximum_kl, kl_mean, kl_max)
            maximum_abs_entropy_change = max(
                maximum_abs_entropy_change,
                abs(entropy_delta),
            )
            deltas.append(delta)
            deltas_by_target[target].append(delta)
            response_rows.append(
                {
                    "target_policy_head_ratio": target,
                    "effective_coefficient": _finite(
                        response.get("effective_coefficient"),
                        field="effective coefficient",
                    ),
                    "coefficient_capped": response.get("coefficient_capped") is True,
                    "realized_policy_head_ratio": _finite(
                        response.get("realized_policy_head_ratio"),
                        field="realized policy-head ratio",
                    ),
                    "preserving_mass_delta": delta,
                    "entropy_delta": entropy_delta,
                    "policy_kl_mean": kl_mean,
                    "policy_kl_max": kl_max,
                    "bounded": bounded,
                }
            )
        monotonic = all(
            right + monotonic_tolerance >= left
            for left, right in zip(deltas, deltas[1:])
        )
        half_uncapped = responses[1].get("coefficient_capped") is False
        all_bounded = all_bounded and seed_bounded
        all_monotonic = all_monotonic and monotonic
        target_half_uncapped = target_half_uncapped and half_uncapped
        per_seed.append(
            {
                "seed": int(arm["seed"]),
                "replay_passed": replay_passed,
                "bounded": seed_bounded,
                "monotonic": monotonic,
                "target_0_50_uncapped": half_uncapped,
                "responses": response_rows,
            }
        )

    medians = {
        str(target): statistics.median(deltas_by_target[target]) for target in targets
    }
    target_one_exceeds_quarter = medians["1.0"] > medians["0.25"]
    escalation_candidate = all(
        (
            mutation_safe,
            all_replays_pass,
            all_bounded,
            all_monotonic,
            target_half_uncapped,
            target_one_exceeds_quarter,
        )
    )
    return {
        "checks": {
            "source_mutation_safe": mutation_safe,
            "persisted_target_replay_passed": all_replays_pass,
            "bounded_response": all_bounded,
            "all_seeds_monotonic": all_monotonic,
            "target_0_50_uncapped_all_seeds": target_half_uncapped,
            "target_1_00_median_exceeds_target_0_25": (target_one_exceeds_quarter),
        },
        "median_preserving_mass_delta_by_target": medians,
        "maximum_policy_kl": maximum_kl,
        "maximum_absolute_entropy_change": maximum_abs_entropy_change,
        "per_seed": per_seed,
        "escalation_candidate": escalation_candidate,
        "verdict": (
            "eligible_to_prepare_independent_seed_learning_calibration"
            if escalation_candidate
            else "stop_gradient_ratio_escalation"
        ),
        "claim_boundary": (
            "optimizer-mechanism decision only; no target selection for retained "
            "training, strength claim, promotion, or launch authority"
        ),
    }


__all__ = [
    "NormalizedTargetResponseResultError",
    "decide_normalized_target_response",
]
