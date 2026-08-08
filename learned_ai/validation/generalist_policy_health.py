"""Pure policy-health summaries for fixed, diagnostic Mill positions."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyHealthState:
    """One fixed position after production-route feature construction."""

    phase: str
    features: np.ndarray
    malom_qualities: np.ndarray
    heuristic_top1_idx: int


def _validated_state(state: PolicyHealthState) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(state.features, dtype=np.float32)
    qualities = np.asarray(state.malom_qualities, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("policy-health features must be a non-empty matrix")
    if qualities.ndim != 1 or qualities.shape[0] != features.shape[0]:
        raise ValueError("Malom qualities must align with feature rows")
    if not 0 <= state.heuristic_top1_idx < features.shape[0]:
        raise ValueError("heuristic_top1_idx is outside the legal move set")
    known = np.isfinite(qualities)
    if np.any(qualities[known] > 0.0):
        raise ValueError("Malom move quality cannot be positive")
    return features, qualities


def summarize_policy_health(
    model: torch.nn.Module,
    states: Iterable[PolicyHealthState],
    *,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    """Summarize policy direction without mutating a model or data source."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    state_list = list(states)
    if not state_list:
        raise ValueError("policy-health audit requires at least one state")

    aggregates: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    argmax_qualities: Counter[str] = Counter()

    model.eval()
    for state in state_list:
        features, qualities = _validated_state(state)
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits_tensor = model.policy_logits(feature_tensor)
            if logits_tensor.shape != (features.shape[0],):
                raise ValueError("policy logits do not align with legal moves")
            if not torch.isfinite(logits_tensor).all():
                raise ValueError("policy-health audit observed non-finite logits")
            probabilities = torch.softmax(logits_tensor, dim=-1).cpu().numpy()
            scheduled = torch.softmax(
                logits_tensor / temperature, dim=-1
            ).cpu().numpy()
            logits = logits_tensor.cpu().numpy()

        top_index = int(np.argmax(probabilities))
        known = np.isfinite(qualities)
        preserving = known & np.isclose(qualities, 0.0)
        downgrading = known & (qualities < 0.0)
        critical = bool(np.any(preserving) and np.any(downgrading))
        if known[top_index]:
            argmax_qualities[str(int(qualities[top_index]))] += 1
        else:
            argmax_qualities["unknown"] += 1

        for group in ("all", state.phase):
            bucket = aggregates[group]
            bucket["states"] += 1.0
            bucket["critical_states"] += float(critical)
            bucket["entropy_temp1_sum"] += float(
                -np.sum(probabilities * np.log(np.clip(probabilities, 1e-30, None)))
            )
            bucket["entropy_scheduled_sum"] += float(
                -np.sum(scheduled * np.log(np.clip(scheduled, 1e-30, None)))
            )
            bucket["heuristic_agreement_sum"] += float(
                top_index == state.heuristic_top1_idx
            )
            bucket["known_mass_sum"] += float(np.sum(scheduled[known]))
            if known[top_index]:
                bucket["known_argmax_states"] += 1.0
                bucket["argmax_quality_sum"] += float(qualities[top_index])
            if critical:
                bucket["argmax_preserving_sum"] += float(preserving[top_index])
                bucket["best_logit_margin_sum"] += float(
                    np.max(logits[preserving]) - np.max(logits[downgrading])
                )
                known_mass_temp1 = max(float(np.sum(probabilities[known])), 1e-30)
                known_mass_scheduled = max(float(np.sum(scheduled[known])), 1e-30)
                bucket["preserving_mass_temp1_sum"] += float(
                    np.sum(probabilities[preserving]) / known_mass_temp1
                )
                bucket["preserving_mass_scheduled_sum"] += float(
                    np.sum(scheduled[preserving]) / known_mass_scheduled
                )
                bucket["expected_quality_temp1_sum"] += float(
                    np.sum(probabilities[known] * qualities[known])
                    / known_mass_temp1
                )
                bucket["expected_quality_scheduled_sum"] += float(
                    np.sum(scheduled[known] * qualities[known])
                    / known_mass_scheduled
                )

    metrics: dict[str, dict[str, Any]] = {}
    for group, bucket in sorted(aggregates.items()):
        states_count = int(bucket["states"])
        critical_count = int(bucket["critical_states"])
        known_argmax_count = int(bucket["known_argmax_states"])

        def critical_mean(key: str) -> float | None:
            if critical_count == 0:
                return None
            return bucket[key] / critical_count

        metrics[group] = {
            "states": states_count,
            "critical_states": critical_count,
            "known_argmax_states": known_argmax_count,
            "mean_entropy_temp1": bucket["entropy_temp1_sum"] / states_count,
            "mean_entropy_scheduled": (
                bucket["entropy_scheduled_sum"] / states_count
            ),
            "heuristic_top1_agreement": (
                bucket["heuristic_agreement_sum"] / states_count
            ),
            "mean_scheduled_known_malom_probability_mass": (
                bucket["known_mass_sum"] / states_count
            ),
            "mean_argmax_malom_quality": (
                bucket["argmax_quality_sum"] / known_argmax_count
                if known_argmax_count
                else None
            ),
            "critical_argmax_value_preserving_rate": critical_mean(
                "argmax_preserving_sum"
            ),
            "critical_mean_preserving_minus_downgrading_logit": critical_mean(
                "best_logit_margin_sum"
            ),
            "critical_value_preserving_probability_mass_temp1": critical_mean(
                "preserving_mass_temp1_sum"
            ),
            "critical_value_preserving_probability_mass_scheduled": critical_mean(
                "preserving_mass_scheduled_sum"
            ),
            "critical_expected_malom_quality_temp1": critical_mean(
                "expected_quality_temp1_sum"
            ),
            "critical_expected_malom_quality_scheduled": critical_mean(
                "expected_quality_scheduled_sum"
            ),
        }

    return {
        "argmax_malom_quality_counts": dict(sorted(argmax_qualities.items())),
        "metrics": metrics,
    }


def summarize_direct_lookahead_signal(
    states: Iterable[PolicyHealthState],
    *,
    signal_column: int,
) -> dict[str, Any]:
    """Verify the direct lookahead signal's direction on critical states."""
    critical_count = 0
    preserving_count = 0
    margin_sum = 0.0
    candidate_rows = 0
    for state in states:
        features, qualities = _validated_state(state)
        if not 0 <= signal_column < features.shape[1]:
            raise ValueError("lookahead signal column is outside the feature matrix")
        known = np.isfinite(qualities)
        preserving = known & np.isclose(qualities, 0.0)
        downgrading = known & (qualities < 0.0)
        candidate_rows += features.shape[0]
        if not (np.any(preserving) and np.any(downgrading)):
            continue
        signal = features[:, signal_column]
        critical_count += 1
        preserving_count += int(preserving[int(np.argmax(signal))])
        margin_sum += float(
            np.max(signal[preserving]) - np.max(signal[downgrading])
        )
    if critical_count == 0:
        raise ValueError("lookahead signal audit found no critical states")
    return {
        "candidate_rows": candidate_rows,
        "critical_states": critical_count,
        "argmax_value_preserving_rate": preserving_count / critical_count,
        "mean_preserving_minus_downgrading_signal": margin_sum / critical_count,
    }
