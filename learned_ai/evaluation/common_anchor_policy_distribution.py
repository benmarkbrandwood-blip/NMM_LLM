"""Compare two policies on one immutable, common-anchor feature corpus."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PRIMARY_TEMPERATURE = 0.2
TEMPERATURES = (1.0, PRIMARY_TEMPERATURE)
RANK_TIE_TOLERANCE = 1e-7

DEFAULT_DIVERGENCE_THRESHOLDS: dict[str, float] = {
    "near_identical_max_phase_mean_js_nats": 5e-4,
    "near_identical_max_phase_mean_total_variation": 0.02,
    "near_identical_max_phase_mean_abs_malom_preserving_mass_delta": 0.02,
    "material_min_all_mean_js_nats": 5e-3,
    "material_min_all_mean_total_variation": 0.05,
    "material_min_phase_mean_abs_malom_preserving_mass_delta": 0.05,
}


class CommonAnchorPolicyDistributionError(RuntimeError):
    """Raised when a paired policy comparison is not well defined."""


def _finite_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise CommonAnchorPolicyDistributionError(
            f"{name} must be a non-empty vector"
        )
    if not np.isfinite(vector).all():
        raise CommonAnchorPolicyDistributionError(
            f"{name} contains non-finite values"
        )
    return vector


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise CommonAnchorPolicyDistributionError(
            "temperature must be finite and positive"
        )
    scaled = logits / temperature
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(scaled)
    denominator = float(np.sum(weights))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise CommonAnchorPolicyDistributionError(
            "softmax denominator is not finite and positive"
        )
    probabilities = weights / denominator
    if not np.isfinite(probabilities).all():
        raise CommonAnchorPolicyDistributionError(
            "softmax produced non-finite probabilities"
        )
    return probabilities


def _rank_positions(
    logits: np.ndarray,
    action_keys: Sequence[str],
) -> np.ndarray:
    order = sorted(
        range(logits.size),
        key=lambda index: (-float(logits[index]), action_keys[index]),
    )
    ranks = np.empty(logits.size, dtype=np.int64)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def _ranking_metrics(
    left: np.ndarray,
    right: np.ndarray,
    action_keys: Sequence[str],
) -> dict[str, Any]:
    left_ranks = _rank_positions(left, action_keys)
    right_ranks = _rank_positions(right, action_keys)
    denominator = max(left.size - 1, 1)
    normalized_displacements = (
        np.abs(left_ranks - right_ranks).astype(np.float64) / denominator
    )

    comparable_pairs = 0
    discordant_pairs = 0
    tie_status_changes = 0
    for first in range(left.size):
        for second in range(first + 1, left.size):
            left_delta = float(left[first] - left[second])
            right_delta = float(right[first] - right[second])
            left_tied = abs(left_delta) <= RANK_TIE_TOLERANCE
            right_tied = abs(right_delta) <= RANK_TIE_TOLERANCE
            if left_tied != right_tied:
                tie_status_changes += 1
            if left_tied or right_tied:
                continue
            comparable_pairs += 1
            if (left_delta > 0.0) != (right_delta > 0.0):
                discordant_pairs += 1

    return {
        "refresh_ranks": left_ranks.tolist(),
        "no_refresh_ranks": right_ranks.tolist(),
        "mean_normalized_rank_displacement": float(
            np.mean(normalized_displacements)
        ),
        "max_normalized_rank_displacement": float(
            np.max(normalized_displacements)
        ),
        "comparable_action_pairs": comparable_pairs,
        "discordant_action_pairs": discordant_pairs,
        "discordant_pair_rate": (
            discordant_pairs / comparable_pairs if comparable_pairs else 0.0
        ),
        "tie_status_change_pairs": tie_status_changes,
    }


def _distribution_metrics(
    refresh: np.ndarray,
    no_refresh: np.ndarray,
) -> dict[str, float]:
    midpoint = 0.5 * (refresh + no_refresh)
    kl_refresh_to_no_refresh = float(
        np.sum(refresh * (np.log(refresh) - np.log(no_refresh)))
    )
    kl_no_refresh_to_refresh = float(
        np.sum(no_refresh * (np.log(no_refresh) - np.log(refresh)))
    )
    js = 0.5 * float(
        np.sum(refresh * (np.log(refresh) - np.log(midpoint)))
        + np.sum(no_refresh * (np.log(no_refresh) - np.log(midpoint)))
    )
    return {
        "kl_refresh_to_no_refresh_nats": kl_refresh_to_no_refresh,
        "kl_no_refresh_to_refresh_nats": kl_no_refresh_to_refresh,
        "jensen_shannon_nats": js,
        "total_variation": 0.5 * float(np.sum(np.abs(refresh - no_refresh))),
        "maximum_action_probability_delta": float(
            np.max(np.abs(refresh - no_refresh))
        ),
    }


def _malom_metrics(
    refresh: np.ndarray,
    no_refresh: np.ndarray,
    qualities: np.ndarray,
) -> dict[str, Any]:
    known = np.isfinite(qualities)
    preserving = known & np.isclose(qualities, 0.0)
    downgrading = known & (qualities < 0.0)

    def condition(probabilities: np.ndarray) -> dict[str, float | None]:
        known_mass = float(np.sum(probabilities[known]))
        preserving_mass = float(np.sum(probabilities[preserving]))
        downgrading_mass = float(np.sum(probabilities[downgrading]))
        return {
            "known_probability_mass": known_mass,
            "preserving_probability_mass": preserving_mass,
            "downgrading_probability_mass": downgrading_mass,
            "preserving_probability_given_known": (
                preserving_mass / known_mass if known_mass > 0.0 else None
            ),
        }

    refresh_metrics = condition(refresh)
    no_refresh_metrics = condition(no_refresh)
    return {
        "known_actions": int(np.sum(known)),
        "preserving_actions": int(np.sum(preserving)),
        "downgrading_actions": int(np.sum(downgrading)),
        "critical": bool(np.any(preserving) and np.any(downgrading)),
        "refresh": refresh_metrics,
        "no_refresh": no_refresh_metrics,
        "no_refresh_minus_refresh_preserving_mass": float(
            no_refresh_metrics["preserving_probability_mass"]
            - refresh_metrics["preserving_probability_mass"]
        ),
    }


def compare_action_logits(
    *,
    refresh_logits: Sequence[float],
    no_refresh_logits: Sequence[float],
    action_keys: Sequence[str],
    malom_qualities: Sequence[float],
) -> dict[str, Any]:
    """Compare complete legal-action distributions for one fixed position."""
    refresh = _finite_vector(refresh_logits, name="refresh_logits")
    no_refresh = _finite_vector(no_refresh_logits, name="no_refresh_logits")
    qualities = np.asarray(malom_qualities, dtype=np.float64)
    if refresh.shape != no_refresh.shape:
        raise CommonAnchorPolicyDistributionError(
            "paired policy logits have different shapes"
        )
    if len(action_keys) != refresh.size or len(set(action_keys)) != refresh.size:
        raise CommonAnchorPolicyDistributionError(
            "action keys must uniquely identify every legal action"
        )
    if qualities.shape != refresh.shape:
        raise CommonAnchorPolicyDistributionError(
            "Malom qualities do not align with legal actions"
        )
    known = np.isfinite(qualities)
    if np.any(qualities[known] > 0.0):
        raise CommonAnchorPolicyDistributionError(
            "Malom quality cannot be positive"
        )

    refresh_top1 = int(np.argmax(refresh))
    no_refresh_top1 = int(np.argmax(no_refresh))
    refresh_sorted = np.sort(refresh)
    no_refresh_sorted = np.sort(no_refresh)
    ranking = _ranking_metrics(refresh, no_refresh, action_keys)
    distributions: dict[str, Any] = {}
    action_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for temperature in TEMPERATURES:
        key = f"temperature_{temperature:g}"
        refresh_probabilities = _softmax(refresh, temperature)
        no_refresh_probabilities = _softmax(no_refresh, temperature)
        action_probabilities[key] = (
            refresh_probabilities,
            no_refresh_probabilities,
        )
        distributions[key] = {
            **_distribution_metrics(
                refresh_probabilities,
                no_refresh_probabilities,
            ),
            "malom": _malom_metrics(
                refresh_probabilities,
                no_refresh_probabilities,
                qualities,
            ),
        }

    actions = []
    for index, action_key in enumerate(action_keys):
        action = {
            "action_key": action_key,
            "malom_quality": (
                float(qualities[index]) if known[index] else None
            ),
            "refresh_logit": float(refresh[index]),
            "no_refresh_logit": float(no_refresh[index]),
            "refresh_rank": int(ranking["refresh_ranks"][index]),
            "no_refresh_rank": int(ranking["no_refresh_ranks"][index]),
            "rank_change_no_refresh_minus_refresh": int(
                ranking["no_refresh_ranks"][index]
                - ranking["refresh_ranks"][index]
            ),
        }
        for key, (refresh_probabilities, no_refresh_probabilities) in (
            action_probabilities.items()
        ):
            action[key] = {
                "refresh_probability": float(refresh_probabilities[index]),
                "no_refresh_probability": float(no_refresh_probabilities[index]),
            }
        actions.append(action)

    ranking.pop("refresh_ranks")
    ranking.pop("no_refresh_ranks")
    return {
        "legal_actions": refresh.size,
        "refresh_top1_action_key": action_keys[refresh_top1],
        "no_refresh_top1_action_key": action_keys[no_refresh_top1],
        "top1_agreement": refresh_top1 == no_refresh_top1,
        "refresh_top1_margin": float(
            refresh_sorted[-1] - refresh_sorted[-2]
            if refresh.size > 1
            else math.inf
        ),
        "no_refresh_top1_margin": float(
            no_refresh_sorted[-1] - no_refresh_sorted[-2]
            if no_refresh.size > 1
            else math.inf
        ),
        "ranking": ranking,
        "distributions": distributions,
        "actions": actions,
    }


def _mean(records: Sequence[Mapping[str, Any]], getter) -> float:
    return float(sum(float(getter(record)) for record in records) / len(records))


def summarize_state_comparisons(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate equal-weight state comparisons by phase and overall."""
    if not records:
        raise CommonAnchorPolicyDistributionError(
            "at least one state comparison is required"
        )
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(records)}
    for phase in ("placement", "movement", "flying"):
        phase_records = [record for record in records if record["phase"] == phase]
        if not phase_records:
            raise CommonAnchorPolicyDistributionError(
                f"fixed corpus has no {phase} states"
            )
        groups[phase] = phase_records

    summary: dict[str, Any] = {}
    for group, items in groups.items():
        group_summary: dict[str, Any] = {
            "states": len(items),
            "legal_actions": sum(int(item["comparison"]["legal_actions"]) for item in items),
            "top1_agreement_rate": _mean(
                items, lambda item: item["comparison"]["top1_agreement"]
            ),
            "top1_changed_states": sum(
                not bool(item["comparison"]["top1_agreement"])
                for item in items
            ),
            "mean_normalized_rank_displacement": _mean(
                items,
                lambda item: item["comparison"]["ranking"]
                ["mean_normalized_rank_displacement"],
            ),
            "mean_discordant_pair_rate": _mean(
                items,
                lambda item: item["comparison"]["ranking"]
                ["discordant_pair_rate"],
            ),
        }
        distributions: dict[str, Any] = {}
        for temperature in TEMPERATURES:
            key = f"temperature_{temperature:g}"
            distributions[key] = {
                "mean_kl_refresh_to_no_refresh_nats": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["kl_refresh_to_no_refresh_nats"],
                ),
                "mean_kl_no_refresh_to_refresh_nats": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["kl_no_refresh_to_refresh_nats"],
                ),
                "mean_jensen_shannon_nats": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["jensen_shannon_nats"],
                ),
                "max_jensen_shannon_nats": max(
                    float(item["comparison"]["distributions"][key]
                          ["jensen_shannon_nats"])
                    for item in items
                ),
                "mean_total_variation": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["total_variation"],
                ),
                "max_total_variation": max(
                    float(item["comparison"]["distributions"][key]
                          ["total_variation"])
                    for item in items
                ),
                "mean_refresh_malom_preserving_probability_mass": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["malom"]["refresh"]["preserving_probability_mass"],
                ),
                "mean_no_refresh_malom_preserving_probability_mass": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["malom"]["no_refresh"]["preserving_probability_mass"],
                ),
                "mean_abs_malom_preserving_probability_mass_delta": _mean(
                    items,
                    lambda item: abs(
                        item["comparison"]["distributions"][key]["malom"]
                        ["no_refresh_minus_refresh_preserving_mass"]
                    ),
                ),
                "mean_no_refresh_minus_refresh_malom_preserving_mass": _mean(
                    items,
                    lambda item: item["comparison"]["distributions"][key]
                    ["malom"]["no_refresh_minus_refresh_preserving_mass"],
                ),
            }
        group_summary["distributions"] = distributions
        summary[group] = group_summary
    return summary


def classify_final_policy_divergence(
    final_by_seed: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float] = DEFAULT_DIVERGENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Apply predeclared effect-size gates to both final seed comparisons."""
    if set(final_by_seed) != {"64", "65"}:
        raise CommonAnchorPolicyDistributionError(
            "final comparison must contain seeds 64 and 65"
        )
    temperature_key = f"temperature_{PRIMARY_TEMPERATURE:g}"
    seed_audits: dict[str, Any] = {}
    for seed, summary in sorted(final_by_seed.items()):
        phase_metrics = [
            summary[phase]["distributions"][temperature_key]
            for phase in ("placement", "movement", "flying")
        ]
        all_metrics = summary["all"]["distributions"][temperature_key]
        observed = {
            "max_phase_mean_jensen_shannon_nats": max(
                item["mean_jensen_shannon_nats"] for item in phase_metrics
            ),
            "max_phase_mean_total_variation": max(
                item["mean_total_variation"] for item in phase_metrics
            ),
            "max_phase_mean_abs_malom_preserving_mass_delta": max(
                item["mean_abs_malom_preserving_probability_mass_delta"]
                for item in phase_metrics
            ),
            "all_mean_jensen_shannon_nats": all_metrics[
                "mean_jensen_shannon_nats"
            ],
            "all_mean_total_variation": all_metrics["mean_total_variation"],
            "all_top1_disagreement_rate": 1.0
            - summary["all"]["top1_agreement_rate"],
        }
        near_identical = (
            observed["max_phase_mean_jensen_shannon_nats"]
            <= thresholds["near_identical_max_phase_mean_js_nats"]
            and observed["max_phase_mean_total_variation"]
            <= thresholds[
                "near_identical_max_phase_mean_total_variation"
            ]
            and observed[
                "max_phase_mean_abs_malom_preserving_mass_delta"
            ]
            <= thresholds[
                "near_identical_max_phase_mean_abs_malom_preserving_mass_delta"
            ]
        )
        materially_diverged = (
            observed["all_mean_jensen_shannon_nats"]
            >= thresholds["material_min_all_mean_js_nats"]
            or observed["all_mean_total_variation"]
            >= thresholds["material_min_all_mean_total_variation"]
            or observed[
                "max_phase_mean_abs_malom_preserving_mass_delta"
            ]
            >= thresholds[
                "material_min_phase_mean_abs_malom_preserving_mass_delta"
            ]
        )
        seed_audits[seed] = {
            "observed": observed,
            "near_identical": near_identical,
            "materially_diverged": materially_diverged,
        }

    if all(item["near_identical"] for item in seed_audits.values()):
        classification = "near_identical"
        next_design = "longer_equal_transition_paired_diagnostic"
    elif all(item["materially_diverged"] for item in seed_audits.values()):
        classification = "materially_diverged"
        next_design = "non_flooring_multi_start_outcome_measurement"
    else:
        classification = "inconclusive"
        next_design = "resolve_seed_or_metric_disagreement_before_more_games"
    return {
        "classification": classification,
        "primary_temperature": PRIMARY_TEMPERATURE,
        "thresholds": dict(thresholds),
        "by_seed": seed_audits,
        "next_design": next_design,
        "top1_is_interpretive_not_a_standalone_gate": True,
    }
