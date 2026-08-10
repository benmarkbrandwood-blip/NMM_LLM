"""Decision rules for the equal-transition target-refresh diagnostic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from learned_ai.evaluation.common_anchor_policy_distribution import (
    DEFAULT_DIVERGENCE_THRESHOLDS,
    PRIMARY_TEMPERATURE,
)


RESULT_SCHEMA = "nmm.target-refresh-equal-transition-result.v1"
EXPECTED_SEEDS = (64, 65, 66)
EXPECTED_BOUNDARIES = (1024, 2048, 4096, 8192)
EXPECTED_CONDITIONS = ("refresh-once", "no-refresh")
MATERIAL_CONFIRMATION_BOUNDARY = 4096
FINAL_BOUNDARY = 8192
PHASES = ("placement", "movement", "flying")


class TargetRefreshEqualTransitionResultError(RuntimeError):
    """Raised when result inputs cannot support the frozen decision rule."""


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TargetRefreshEqualTransitionResultError(
            f"{field} must be finite"
        ) from exc
    if not math.isfinite(result):
        raise TargetRefreshEqualTransitionResultError(
            f"{field} must be finite"
        )
    return result


def _summary_observation(summary: Mapping[str, Any]) -> dict[str, Any]:
    temperature_key = f"temperature_{PRIMARY_TEMPERATURE:g}"
    try:
        phase_metrics = {
            phase: summary[phase]["distributions"][temperature_key]
            for phase in PHASES
        }
        all_metrics = summary["all"]["distributions"][temperature_key]
        top1_agreement = summary["all"]["top1_agreement_rate"]
    except (KeyError, TypeError) as exc:
        raise TargetRefreshEqualTransitionResultError(
            "policy summary is incomplete"
        ) from exc

    phase_js = {
        phase: _finite(
            metrics["mean_jensen_shannon_nats"],
            field=f"{phase} mean Jensen-Shannon",
        )
        for phase, metrics in phase_metrics.items()
    }
    phase_tv = {
        phase: _finite(
            metrics["mean_total_variation"],
            field=f"{phase} mean total variation",
        )
        for phase, metrics in phase_metrics.items()
    }
    phase_abs_malom = {
        phase: _finite(
            metrics["mean_abs_malom_preserving_probability_mass_delta"],
            field=f"{phase} mean absolute Malom mass delta",
        )
        for phase, metrics in phase_metrics.items()
    }
    phase_signed_malom = {
        phase: _finite(
            metrics["mean_no_refresh_minus_refresh_malom_preserving_mass"],
            field=f"{phase} signed Malom mass delta",
        )
        for phase, metrics in phase_metrics.items()
    }
    return {
        "max_phase_mean_jensen_shannon_nats": max(phase_js.values()),
        "max_phase_mean_total_variation": max(phase_tv.values()),
        "max_phase_mean_abs_malom_preserving_mass_delta": max(
            phase_abs_malom.values()
        ),
        "all_mean_jensen_shannon_nats": _finite(
            all_metrics["mean_jensen_shannon_nats"],
            field="all mean Jensen-Shannon",
        ),
        "all_mean_total_variation": _finite(
            all_metrics["mean_total_variation"],
            field="all mean total variation",
        ),
        "all_top1_disagreement_rate": 1.0
        - _finite(top1_agreement, field="all top-1 agreement"),
        "phase_signed_malom_preserving_mass_delta": phase_signed_malom,
        "phase_abs_malom_preserving_mass_delta": phase_abs_malom,
    }


def _near_identical(
    observed: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> bool:
    return (
        observed["max_phase_mean_jensen_shannon_nats"]
        <= thresholds["near_identical_max_phase_mean_js_nats"]
        and observed["max_phase_mean_total_variation"]
        <= thresholds["near_identical_max_phase_mean_total_variation"]
        and observed["max_phase_mean_abs_malom_preserving_mass_delta"]
        <= thresholds[
            "near_identical_max_phase_mean_abs_malom_preserving_mass_delta"
        ]
    )


def _material_triggers(
    observed: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> set[str]:
    triggers: set[str] = set()
    if (
        observed["all_mean_jensen_shannon_nats"]
        >= thresholds["material_min_all_mean_js_nats"]
    ):
        triggers.add("all_mean_jensen_shannon")
    if (
        observed["all_mean_total_variation"]
        >= thresholds["material_min_all_mean_total_variation"]
    ):
        triggers.add("all_mean_total_variation")
    for phase in PHASES:
        if (
            observed["phase_abs_malom_preserving_mass_delta"][phase]
            >= thresholds[
                "material_min_phase_mean_abs_malom_preserving_mass_delta"
            ]
        ):
            triggers.add(f"malom_preserving_mass:{phase}")
    return triggers


def _persistent_triggers(
    early: Mapping[str, Any],
    final: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> list[str]:
    shared = _material_triggers(early, thresholds) & _material_triggers(
        final, thresholds
    )
    persistent: list[str] = []
    for trigger in sorted(shared):
        if not trigger.startswith("malom_preserving_mass:"):
            persistent.append(trigger)
            continue
        phase = trigger.split(":", 1)[1]
        early_signed = early["phase_signed_malom_preserving_mass_delta"][phase]
        final_signed = final["phase_signed_malom_preserving_mass_delta"][phase]
        if early_signed != 0.0 and final_signed != 0.0 and (
            (early_signed > 0.0) == (final_signed > 0.0)
        ):
            persistent.append(trigger)
    return persistent


def classify_transition_policy_divergence(
    by_seed_boundary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    thresholds: Mapping[str, float] = DEFAULT_DIVERGENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Apply the predeclared three-seed, two-boundary persistence gate."""
    expected_seed_keys = {str(seed) for seed in EXPECTED_SEEDS}
    if set(by_seed_boundary) != expected_seed_keys:
        raise TargetRefreshEqualTransitionResultError(
            "result must contain exactly seeds 64, 65 and 66"
        )
    required_thresholds = set(DEFAULT_DIVERGENCE_THRESHOLDS)
    if set(thresholds) != required_thresholds:
        raise TargetRefreshEqualTransitionResultError(
            "divergence threshold fields differ"
        )
    normalized_thresholds = {
        key: _finite(value, field=f"threshold {key}")
        for key, value in thresholds.items()
    }

    seed_audits: dict[str, Any] = {}
    for seed in sorted(by_seed_boundary, key=int):
        boundaries = by_seed_boundary[seed]
        if set(boundaries) != {str(value) for value in EXPECTED_BOUNDARIES}:
            raise TargetRefreshEqualTransitionResultError(
                f"seed {seed} transition boundaries differ"
            )
        observations = {
            boundary: _summary_observation(boundaries[str(boundary)])
            for boundary in EXPECTED_BOUNDARIES
        }
        early = observations[MATERIAL_CONFIRMATION_BOUNDARY]
        final = observations[FINAL_BOUNDARY]
        persistent = _persistent_triggers(
            early,
            final,
            normalized_thresholds,
        )
        final_triggers = sorted(
            _material_triggers(final, normalized_thresholds)
        )
        seed_audits[seed] = {
            "by_transition_boundary": {
                str(boundary): {
                    "observed": observations[boundary],
                    "near_identical": _near_identical(
                        observations[boundary], normalized_thresholds
                    ),
                    "material_triggers": sorted(
                        _material_triggers(
                            observations[boundary], normalized_thresholds
                        )
                    ),
                }
                for boundary in EXPECTED_BOUNDARIES
            },
            "persistent_material_triggers_4096_to_8192": persistent,
            "materially_diverged_with_persistence": bool(persistent),
            "material_only_at_final_boundary": bool(final_triggers)
            and not persistent,
        }

    final_all_near = all(
        audit["by_transition_boundary"][str(FINAL_BOUNDARY)][
            "near_identical"
        ]
        for audit in seed_audits.values()
    )
    persistent_material_seeds = [
        seed
        for seed, audit in seed_audits.items()
        if audit["materially_diverged_with_persistence"]
    ]
    final_only_material_seeds = [
        seed
        for seed, audit in seed_audits.items()
        if audit["material_only_at_final_boundary"]
    ]

    if final_all_near:
        classification = "near_identical"
        next_design = "stop_without_automatic_extension"
    elif len(persistent_material_seeds) >= 2:
        classification = "materially_diverged"
        next_design = "non_flooring_multi_start_no_update_outcome_measurement"
    elif len(final_only_material_seeds) >= 2:
        classification = "inconclusive_late_onset"
        next_design = "no_automatic_extension_or_outcome_claim"
    else:
        classification = "inconclusive"
        next_design = "resolve_seed_phase_or_metric_disagreement"

    return {
        "classification": classification,
        "primary_temperature": PRIMARY_TEMPERATURE,
        "thresholds": normalized_thresholds,
        "material_confirmation_boundary": MATERIAL_CONFIRMATION_BOUNDARY,
        "final_boundary": FINAL_BOUNDARY,
        "minimum_persistent_material_seeds": 2,
        "persistent_material_seeds": persistent_material_seeds,
        "final_only_material_seeds": final_only_material_seeds,
        "by_seed": seed_audits,
        "next_design": next_design,
        "top1_is_interpretive_not_a_standalone_gate": True,
    }


__all__ = [
    "EXPECTED_BOUNDARIES",
    "EXPECTED_CONDITIONS",
    "EXPECTED_SEEDS",
    "FINAL_BOUNDARY",
    "MATERIAL_CONFIRMATION_BOUNDARY",
    "RESULT_SCHEMA",
    "TargetRefreshEqualTransitionResultError",
    "classify_transition_policy_divergence",
]
