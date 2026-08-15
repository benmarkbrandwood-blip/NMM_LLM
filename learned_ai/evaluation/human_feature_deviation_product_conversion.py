"""Exploratory conversion from human-choice prediction to product upper bounds.

This module never constructs a gameplay policy.  It reuses sealed out-of-fold
conditional-logit coefficients and permits only positional ``A_pos`` Malom
queries over the frozen research-exploration sample.  Its policy quantities
are predictive plug-ins, not causal inducement effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ai.malom_db import MalomDB
from game.rules import get_all_legal_moves, terminal_wdl
from learned_ai.evaluation.human_f0h0_feasibility import (
    F0D0Boundary,
    canonical_sha256,
)
from learned_ai.evaluation.human_feature_deviation_design_round import (
    V2_FEATURE_NAMES,
    _query_inventory,
    extended_action_feature_scores,
)
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    ChoiceObservation,
    EstimatorAccess,
    NumericalContract,
    _choice_probabilities,
    _move_key,
    _stable_probabilities,
    _weighted_quantile,
    extract_exploration_observations,
    player_cluster_bootstrap,
)


PLAN_SCHEMA = "nmm.human-feature-deviation-product-conversion-derivation.v1"
PLAN_V2_SCHEMA = "nmm.human-feature-deviation-product-conversion-derivation.v2"
RESULT_SCHEMA = "nmm.human-feature-deviation-product-conversion-result.v1"


class ConversionError(RuntimeError):
    """Raised when a conversion requirement cannot be established safely."""


def load_conversion_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and verify the frozen derivation contract."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid conversion plan: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_SCHEMA:
        raise ConversionError("conversion plan schema differs")
    identity = value.get("derivation_identity")
    body = dict(value)
    body.pop("derivation_identity", None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ConversionError("conversion plan identity differs")
    if value.get("not_a_new_research_preregistration") is not True:
        raise ConversionError("conversion plan incorrectly creates a research question")
    return value, hashlib.sha256(raw).hexdigest()


def load_readiness_result(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the sealed result that owns the reusable fold parameters."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid readiness result: {source}") from exc
    identity = value.get("result_identity") if isinstance(value, dict) else None
    body = dict(value) if isinstance(value, dict) else {}
    body.pop("result_identity", None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ConversionError("readiness result identity differs")
    return value, hashlib.sha256(raw).hexdigest()


def load_effective_conversion_plan(
    path: str | Path, *, inherited_v1_path: str | Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve the immutable v2 precision correction over frozen v1."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConversionError(f"invalid conversion v2 plan: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_V2_SCHEMA:
        raise ConversionError("conversion v2 plan schema differs")
    identity = value.get("derivation_identity")
    body = dict(value)
    body.pop("derivation_identity", None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise ConversionError("conversion v2 identity differs")
    inherited, inherited_sha = load_conversion_plan(inherited_v1_path)
    disposition = value.get("v1_disposition")
    if not isinstance(disposition, Mapping) or disposition != {
        "derivation_identity": inherited["derivation_identity"],
        "plan_file_sha256": inherited_sha,
        "remains_immutable": True,
        "status": "superseded_for_execution_by_exact_result_transcription_correction",
        "failure": (
            "three rounded narrative log-loss values were entered as exact "
            "reproduction targets and differed from the sealed machine result by "
            "about 1e-11, beyond the frozen 1e-12 tolerance"
        ),
        "new_hypothetical_successor_queries_before_failure": 0,
        "protected_content_reads_before_failure": 0,
    }:
        raise ConversionError("conversion v2 disposition differs")
    inheritance = value.get("frozen_inheritance")
    sections = inheritance.get("sections") if isinstance(inheritance, Mapping) else None
    if (
        not isinstance(sections, list)
        or any(section not in inherited for section in sections)
        or inheritance.get("source_derivation_identity")
        != inherited["derivation_identity"]
        or inheritance.get("unchanged_except_declared_correction") is not True
    ):
        raise ConversionError("conversion v2 inheritance differs")
    effective = deepcopy(inherited)
    correction = value["exact_reproduction_correction"]
    required = effective["frozen_estimator_reuse"]["required_reproduction"]
    for key in (
        "geometry_average_unique_player_log_loss",
        "full_average_unique_player_log_loss",
        "paired_log_loss_improvement",
    ):
        required[key] = correction[key]
    if (
        correction.get("other_required_reproduction_fields_unchanged") is not True
        or correction.get("tolerance_unchanged") != required["tolerance"]
    ):
        raise ConversionError("conversion v2 correction scope differs")
    effective["schema_version"] = PLAN_V2_SCHEMA
    effective["status"] = value["status"]
    effective["derivation_identity"] = identity
    effective["v1_derivation_identity"] = inherited["derivation_identity"]
    return effective, {
        "v2_derivation_identity": identity,
        "v2_plan_file_sha256": hashlib.sha256(raw).hexdigest(),
        "v1_derivation_identity": inherited["derivation_identity"],
        "v1_plan_file_sha256": inherited_sha,
    }


def tier_loss_outcomes(parent_tier: str) -> frozenset[str]:
    """Return strict WDL losses from the specified actor perspective."""
    if parent_tier == "W":
        return frozenset({"D", "L"})
    if parent_tier == "D":
        return frozenset({"L"})
    if parent_tier == "L":
        return frozenset()
    raise ConversionError(f"invalid positional tier: {parent_tier!r}")


def _opponent_tier(learner_tier: str) -> str:
    try:
        return {"W": "L", "D": "D", "L": "W"}[learner_tier]
    except KeyError as exc:
        raise ConversionError(f"invalid learner tier: {learner_tier!r}") from exc


def normalized_safe_weights(
    probabilities: np.ndarray, safe_mask: np.ndarray
) -> np.ndarray:
    """Restrict a complete choice distribution to A_pos and renormalize."""
    values = np.asarray(probabilities, dtype=np.float64)
    mask = np.asarray(safe_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape or not np.all(np.isfinite(values)):
        raise ConversionError("safe reference inputs differ")
    selected = values[mask]
    total = float(np.sum(selected, dtype=np.float64))
    if selected.size == 0 or not math.isfinite(total) or total <= 0.0:
        raise ConversionError("safe reference mass is empty")
    result = selected / total
    if not np.all(np.isfinite(result)) or not math.isclose(
        float(np.sum(result)), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ConversionError("safe reference weights do not normalize")
    return result


def uplift_against_reference(
    action_risks: np.ndarray, reference_weights: np.ndarray
) -> dict[str, float | int]:
    """Compute a predictive argmax uplift against one stochastic reference."""
    risks = np.asarray(action_risks, dtype=np.float64)
    weights = np.asarray(reference_weights, dtype=np.float64)
    if (
        risks.ndim != 1
        or risks.size == 0
        or weights.shape != risks.shape
        or not np.all(np.isfinite(risks))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1e-12)
    ):
        raise ConversionError("uplift inputs differ")
    argmax = int(np.argmax(risks))
    maximum = float(risks[argmax])
    reference = float(np.dot(weights, risks))
    uplift = maximum - reference
    if uplift < -1e-12:
        raise ConversionError("argmax uplift is negative")
    if risks.size == 1:
        uplift = 0.0
    return {
        "maximum": maximum,
        "reference": reference,
        "uplift": max(0.0, uplift),
        "argmax_index": argmax,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    nonnegative = array >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-array[nonnegative]))
    exponentials = np.exp(array[~nonnegative])
    result[~nonnegative] = exponentials / (1.0 + exponentials)
    return result


def _equal_player_weights(players: Sequence[str]) -> np.ndarray:
    keys = [str(player) for player in players]
    counts = Counter(keys)
    if not counts:
        raise ConversionError("player weights are empty")
    return np.asarray(
        [1.0 / (len(counts) * counts[player]) for player in keys],
        dtype=np.float64,
    )


def fit_logistic_calibrator(
    probabilities: np.ndarray,
    events: np.ndarray,
    players: Sequence[str],
    *,
    maximum_iterations: int = 100,
    gradient_tolerance: float = 1e-10,
) -> dict[str, float | int | bool]:
    """Fit event ~ intercept + slope*logit(p) with equal-player weights.

    Exact-zero risk rows must have zero events and are left at exact zero by
    the application function.  No pseudo-count or prior is introduced.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(events, dtype=np.float64)
    player_array = np.asarray([str(player) for player in players], dtype=object)
    if (
        p.ndim != 1
        or y.shape != p.shape
        or player_array.shape != p.shape
        or p.size < 3
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(y))
        or np.any((p < 0.0) | (p > 1.0))
        or np.any((y != 0.0) & (y != 1.0))
    ):
        raise ConversionError("calibration rows are invalid")
    zero = p == 0.0
    if np.any(y[zero] != 0.0):
        raise ConversionError("exact-zero risk has an observed event")
    keep = ~zero
    if int(np.sum(keep)) < 3 or len(set(player_array[keep])) < 2:
        raise ConversionError("positive-risk calibration support is insufficient")
    clipped = np.clip(p[keep], 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack((np.ones(logits.size), logits))
    weights = _equal_player_weights(player_array[keep].tolist())
    beta = np.asarray([0.0, 1.0], dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        eta = design @ candidate
        return float(
            np.sum(weights * (np.logaddexp(0.0, eta) - y[keep] * eta))
        )

    current = objective(beta)
    converged = False
    gradient_norm = math.inf
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        fitted = _sigmoid(design @ beta)
        gradient = design.T @ (weights * (fitted - y[keep]))
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= gradient_tolerance:
            converged = True
            break
        curvature = weights * fitted * (1.0 - fitted)
        information = design.T @ (design * curvature[:, None])
        try:
            direction = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError as exc:
            raise ConversionError("calibration information is singular") from exc
        step = 1.0
        accepted = False
        for _ in range(40):
            candidate = beta - step * direction
            value = objective(candidate)
            if math.isfinite(value) and value < current:
                beta = candidate
                current = value
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if gradient_norm <= 1e-8:
                converged = True
                break
            raise ConversionError("calibration line search failed")
    if not converged:
        fitted = _sigmoid(design @ beta)
        gradient = design.T @ (weights * (fitted - y[keep]))
        gradient_norm = float(np.max(np.abs(gradient)))
        converged = gradient_norm <= 1e-8
    if not converged or not np.all(np.isfinite(beta)) or beta[1] <= 0.0:
        raise ConversionError("calibration did not produce a positive finite slope")
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "objective": current,
        "gradient_infinity_norm": gradient_norm,
        "iterations": iterations,
        "converged": True,
        "rows": int(p.size),
        "positive_risk_rows": int(np.sum(keep)),
        "players": len(set(player_array.tolist())),
        "zero_risk_rows": int(np.sum(zero)),
        "zero_events_not_smoothed": True,
    }


def apply_logistic_calibrator(
    probabilities: np.ndarray, intercept: float, slope: float
) -> np.ndarray:
    """Apply calibration while preserving exact structural zero risk."""
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        not np.all(np.isfinite(p))
        or np.any((p < 0.0) | (p > 1.0))
        or not math.isfinite(intercept)
        or not math.isfinite(slope)
        or slope <= 0.0
    ):
        raise ConversionError("calibration application inputs are invalid")
    result = np.zeros_like(p)
    positive = p > 0.0
    clipped = np.clip(p[positive], 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    result[positive] = _sigmoid(intercept + slope * logits)
    if not np.all(np.isfinite(result)):
        raise ConversionError("calibrated risk is nonfinite")
    return result


def jensen_shannon_divergence(
    left: Sequence[float], right: Sequence[float]
) -> float:
    """Return natural-log Jensen-Shannon divergence for two mass vectors."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if (
        p.ndim != 1
        or q.shape != p.shape
        or p.size == 0
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(q))
        or np.any(p < 0.0)
        or np.any(q < 0.0)
        or float(np.sum(p)) <= 0.0
        or float(np.sum(q)) <= 0.0
    ):
        raise ConversionError("Jensen-Shannon inputs are invalid")
    p = p / np.sum(p)
    q = q / np.sum(q)
    midpoint = 0.5 * (p + q)

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        positive = source > 0.0
        return float(np.sum(source[positive] * np.log(source[positive] / target[positive])))

    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def product_scenario_thresholds(
    *,
    score_points_per_100: Sequence[float],
    mean_parent_d_opportunities_per_side_game: float,
) -> list[dict[str, Any]]:
    """Invert the perfect-redemption union bound into necessary uplifts."""
    opportunities = float(mean_parent_d_opportunities_per_side_game)
    if not math.isfinite(opportunities) or opportunities <= 0.0:
        raise ConversionError("parent-D opportunity rate is invalid")
    rows: list[dict[str, Any]] = []
    for points in score_points_per_100:
        value = float(points)
        if not math.isfinite(value) or value <= 0.0 or value >= 50.0:
            raise ConversionError("product score scenario is invalid")
        gain = value / 100.0
        rows.append(
            {
                "additional_score_points_per_100_games": value,
                "score_gain_per_game": gain,
                "necessary_single_step_uplift": 2.0 * gain / opportunities,
                "necessary_within_state_risk_spread": 2.0 * gain / opportunities,
                "D_discrimination_equivalent": None,
                "log_loss_equivalent": None,
                "perfect_redemption_upper_bound": True,
                "necessary_not_sufficient": True,
            }
        )
    return rows


def _fold_parameters(result: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, dict[str, np.ndarray]] = {}
    for report in result["analysis"]["folds"]:
        fold = int(report["fold"])
        rows[fold] = {
            "mean": np.asarray(report["feature_mean"], dtype=np.float64),
            "scale": np.asarray(report["feature_scale"], dtype=np.float64),
            "geometry": np.asarray(
                report["geometry_fit"]["coefficients"], dtype=np.float64
            ),
            "full": np.asarray(report["full_fit"]["coefficients"], dtype=np.float64),
        }
    if sorted(rows) != [0, 1, 2, 3, 4]:
        raise ConversionError("readiness fold parameters differ")
    return rows


def _probabilities_from_features(
    features: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    *,
    specification: str,
    contract: NumericalContract,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    columns = 3 if specification == "geometry" else 10
    coefficients = parameters[specification]
    standardized = (
        matrix[:, :columns]
        - np.asarray(parameters["mean"], dtype=np.float64)[:columns]
    ) / np.asarray(parameters["scale"], dtype=np.float64)[:columns]
    probabilities, _ = _stable_probabilities(
        standardized @ coefficients,
        floor=contract.reporting_floor,
        ceiling=contract.reporting_ceiling,
    )
    return probabilities


def _current_predictions(
    observations: Sequence[ChoiceObservation],
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce the sealed OOF result from persisted parameters, without fit."""
    parameters = _fold_parameters(result)
    contract = NumericalContract.from_plan(result["frozen_contract"])
    player_geometry: defaultdict[str, list[float]] = defaultdict(list)
    player_full: defaultdict[str, list[float]] = defaultdict(list)
    calibration: list[dict[str, Any]] = []
    for observation in observations:
        fold = observation.fold
        geometry = _choice_probabilities(
            observation,
            columns=tuple(range(3)),
            mean=parameters[fold]["mean"][:3],
            scale=parameters[fold]["scale"][:3],
            coefficients=parameters[fold]["geometry"],
            contract=contract,
        )
        full = _choice_probabilities(
            observation,
            columns=tuple(range(10)),
            mean=parameters[fold]["mean"],
            scale=parameters[fold]["scale"],
            coefficients=parameters[fold]["full"],
            contract=contract,
        )
        if observation.is_degenerate:
            continue
        geometry_loss = -math.log(float(geometry[observation.chosen_index]))
        full_loss = -math.log(float(full[observation.chosen_index]))
        player_geometry[observation.player_key].append(geometry_loss)
        player_full[observation.player_key].append(full_loss)
        if observation.parent_tier == "D":
            risk = sum(
                float(probability)
                for probability, outcome in zip(
                    full, observation.action_outcomes, strict=True
                )
                if outcome == "L"
            )
            event = int(observation.action_outcomes[observation.chosen_index] == "L")
            calibration.append(
                {
                    "player": observation.player_key,
                    "fold": observation.fold,
                    "risk": risk,
                    "event": event,
                }
            )
    geometry_means = {
        player: float(np.mean(values)) for player, values in player_geometry.items()
    }
    full_means = {player: float(np.mean(values)) for player, values in player_full.items()}
    if set(geometry_means) != set(full_means):
        raise ConversionError("reproduced player support differs")
    reproduced = {
        "geometry_average_unique_player_log_loss": float(
            np.mean(list(geometry_means.values()))
        ),
        "full_average_unique_player_log_loss": float(
            np.mean(list(full_means.values()))
        ),
        "paired_log_loss_improvement": float(
            np.mean(
                [geometry_means[player] - full_means[player] for player in full_means]
            )
        ),
        "parent_D_decisions": len(calibration),
        "D_to_L_events": sum(int(row["event"]) for row in calibration),
    }
    required = plan["frozen_estimator_reuse"]["required_reproduction"]
    tolerance = float(required["tolerance"])
    for key, actual in reproduced.items():
        expected = required[key]
        if isinstance(expected, int):
            if actual != expected:
                raise ConversionError(f"sealed reproduction differs for {key}")
        elif not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
        ):
            raise ConversionError(f"sealed reproduction differs for {key}")
    reproduced["passed"] = True
    reproduced["refit_performed"] = False
    return calibration, reproduced


def _fit_cross_calibrators(
    rows: Sequence[Mapping[str, Any]], folds: int
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    calibrators: dict[int, dict[str, Any]] = {}
    for fold in range(folds):
        training = [row for row in rows if int(row["fold"]) != fold]
        calibrators[fold] = fit_logistic_calibrator(
            np.asarray([row["risk"] for row in training], dtype=np.float64),
            np.asarray([row["event"] for row in training], dtype=np.float64),
            [str(row["player"]) for row in training],
        )
    global_fit = fit_logistic_calibrator(
        np.asarray([row["risk"] for row in rows], dtype=np.float64),
        np.asarray([row["event"] for row in rows], dtype=np.float64),
        [str(row["player"]) for row in rows],
    )
    return calibrators, global_fit


def _calibration_report(
    rows: Sequence[Mapping[str, Any]], global_fit: Mapping[str, Any]
) -> dict[str, Any]:
    risks = np.asarray([row["risk"] for row in rows], dtype=np.float64)
    events = np.asarray([row["event"] for row in rows], dtype=np.float64)
    players = [str(row["player"]) for row in rows]
    weights = _equal_player_weights(players)
    boundaries = [
        _weighted_quantile(risks, weights, quantile)
        for quantile in np.linspace(0.1, 0.9, 9)
    ]
    assignments = np.searchsorted(np.asarray(boundaries), risks, side="right")
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(10):
        mask = assignments == index
        mass = float(np.sum(weights[mask]))
        if not np.any(mask):
            bins.append(
                {
                    "bin": index,
                    "weight_mass": 0.0,
                    "decisions": 0,
                    "players": 0,
                    "mean_predicted": None,
                    "observed_rate": None,
                }
            )
            continue
        predicted = float(np.average(risks[mask], weights=weights[mask]))
        observed = float(np.average(events[mask], weights=weights[mask]))
        ece += mass * abs(predicted - observed)
        bins.append(
            {
                "bin": index,
                "weight_mass": mass,
                "decisions": int(np.sum(mask)),
                "players": len({players[row] for row in np.flatnonzero(mask)}),
                "mean_predicted": predicted,
                "observed_rate": observed,
            }
        )
    return {
        "rows": len(rows),
        "players": len(set(players)),
        "events": int(np.sum(events)),
        "risk_minimum": float(np.min(risks)),
        "risk_maximum": float(np.max(risks)),
        "risk_boundaries": boundaries,
        "bins": bins,
        "expected_calibration_error": ece,
        "Brier": float(np.sum(weights * (risks - events) ** 2)),
        "calibration_intercept": float(global_fit["intercept"]),
        "calibration_slope": float(global_fit["slope"]),
        "zero_risk_events": int(np.sum(events[risks == 0.0])),
        "zero_events_not_smoothed": True,
    }


def _weighted_distribution(values: np.ndarray, players: Sequence[str]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ConversionError("distribution values are empty or nonfinite")
    keys = [str(player) for player in players]
    if len(keys) != array.size:
        raise ConversionError("distribution player support differs")
    weights = _equal_player_weights(keys)
    quantiles = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    by_player: defaultdict[str, list[float]] = defaultdict(list)
    for player, value in zip(keys, array, strict=True):
        by_player[player].append(float(value))
    player_means = np.asarray(
        [float(np.mean(by_player[player])) for player in sorted(by_player)]
    )
    return {
        "decisions": int(array.size),
        "players": len(by_player),
        "natural_decision_mean": float(np.mean(array)),
        "average_unique_player_mean": float(np.mean(player_means)),
        "natural_decision_quantiles": {
            str(quantile): float(np.quantile(array, quantile, method="linear"))
            for quantile in quantiles
        },
        "equal_player_weight_quantiles": {
            str(quantile): _weighted_quantile(array, weights, quantile)
            for quantile in quantiles
        },
        "zero_fraction": float(np.mean(array == 0.0)),
    }


def _bootstrap_uplift(
    values: np.ndarray,
    players: Sequence[str],
    *,
    seed: str,
    replicates: int,
) -> dict[str, Any]:
    by_player: defaultdict[str, list[float]] = defaultdict(list)
    for player, value in zip(players, values, strict=True):
        by_player[str(player)].append(float(value))
    player_means = {
        player: float(np.mean(rows)) for player, rows in sorted(by_player.items())
    }
    return player_cluster_bootstrap(
        player_means,
        replicates=replicates,
        seed=seed,
        statistic="mean_and_sd",
    )


def _cardinality_label(value: int) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9-plus"


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _successor_response_risks(
    board: Any,
    *,
    learner_tier: str,
    parameters: Mapping[str, np.ndarray],
    contract: NumericalContract,
    database: MalomDB,
) -> tuple[float, float, str, int, int]:
    if terminal_wdl(board) is not None:
        return 0.0, 0.0, _opponent_tier(learner_tier), 0, 0
    parent_tier, inventory, queries = _query_inventory(board, database)
    expected_tier = _opponent_tier(learner_tier)
    if parent_tier != expected_tier:
        raise ConversionError("safe successor opponent tier differs")
    ordered = sorted(inventory, key=lambda row: _move_key(row[0]))
    features = np.asarray(
        [
            [extended_action_feature_scores(board, move)[name] for name in V2_FEATURE_NAMES]
            for move, _value in ordered
        ],
        dtype=np.float64,
    )
    outcomes = [value.outcome for _move, value in ordered]
    full = _probabilities_from_features(
        features, parameters, specification="full", contract=contract
    )
    geometry = _probabilities_from_features(
        features, parameters, specification="geometry", contract=contract
    )
    losses = tier_loss_outcomes(parent_tier)
    full_risk = sum(
        float(probability)
        for probability, outcome in zip(full, outcomes, strict=True)
        if outcome in losses
    )
    geometry_risk = sum(
        float(probability)
        for probability, outcome in zip(geometry, outcomes, strict=True)
        if outcome in losses
    )
    return full_risk, geometry_risk, parent_tier, queries, len(ordered)


def derive_product_conversion(
    *,
    repository_root: str | Path,
    boundary: F0D0Boundary,
    official_membership: Mapping[str, Any],
    research_split: Mapping[str, Any],
    structure: Mapping[str, Any],
    readiness_plan: Mapping[str, Any],
    conversion_plan: Mapping[str, Any],
    readiness_result: Mapping[str, Any],
    database: MalomDB,
) -> dict[str, Any]:
    """Execute the frozen exploratory conversion derivation without refit."""
    started = time.perf_counter()
    observations, extraction = extract_exploration_observations(
        repository_root=repository_root,
        boundary=boundary,
        official_membership=official_membership,
        research_split=research_split,
        structure=structure,
        plan=readiness_plan,
        database=database,
    )
    required = conversion_plan["frozen_estimator_reuse"]["required_reproduction"]
    if extraction["covered_decisions"] != required["covered_decisions"]:
        raise ConversionError("exploration decision count differs")
    if extraction["queries"] != required["malom_queries_for_original_choice_inventory"]:
        raise ConversionError("original Malom query count differs")
    calibration_rows, reproduction = _current_predictions(
        observations, readiness_result, conversion_plan
    )
    calibrators, global_calibrator = _fit_cross_calibrators(calibration_rows, 5)
    calibration = _calibration_report(calibration_rows, global_calibrator)

    parameters = _fold_parameters(readiness_result)
    contract = NumericalContract.from_plan(readiness_result["frozen_contract"])
    sample_rows = structure["structure"]["sample_games"]
    sample_ids = [str(row["session_id"]) for row in sample_rows]
    access = EstimatorAccess.from_memberships(
        official_membership,
        research_split,
        allowed_sessions=sample_ids,
    )
    records = {record.session_id: record for record in boundary.records}
    observation_index = 0
    successor_queries = 0
    terminal_safe_successors = 0
    successor_response_choice_sets = 0
    successor_response_actions = 0
    columns: defaultdict[str, list[Any]] = defaultdict(list)
    references = ("uniform_A_pos", "geometry_A_pos", "human_frequency_A_pos")
    side_keys = {
        f"{row['session_id']}:{color}"
        for row in sample_rows
        for color in ("W", "B")
    }
    side_sums_raw = {name: {key: 0.0 for key in side_keys} for name in references}
    side_sums_corrected = {
        name: {key: 0.0 for key in side_keys} for name in references
    }
    side_d_counts = {key: 0 for key in side_keys}

    d_counts = Counter(
        observation.player_key
        for observation in observations
        if observation.parent_tier == "D"
    )
    d_players = len(d_counts)
    risk_boundaries = np.asarray(calibration["risk_boundaries"], dtype=np.float64)
    observed_minimum = float(calibration["risk_minimum"])
    observed_maximum = float(calibration["risk_maximum"])
    support_bins = calibration["bins"]
    shift = {
        name: {
            "selected_mass": np.zeros(10, dtype=np.float64),
            "reference_mass": np.zeros(10, dtype=np.float64),
            "selected_below": 0.0,
            "selected_above": 0.0,
            "reference_below": 0.0,
            "reference_above": 0.0,
        }
        for name in references
    }

    root = Path(repository_root)
    maximum_queries = int(conversion_plan["execution_boundary"]["maximum_malom_queries"])
    maximum_seconds = float(
        conversion_plan["execution_boundary"]["maximum_active_seconds"]
    )
    for sample_row in sample_rows:
        session_id = str(sample_row["session_id"])
        decisions = access.load_decisions(root, records[session_id], boundary)
        for decision_index, decision in enumerate(decisions):
            if observation_index >= len(observations):
                raise ConversionError("second replay exceeds observation count")
            observation = observations[observation_index]
            observation_index += 1
            if (
                observation.game_id != decision.game_id
                or observation.decision_index != decision_index
                or observation.player_key != decision.actor_player_key
                or observation.fold != int(sample_row["fold"])
            ):
                raise ConversionError("second replay observation alignment differs")
            fold = observation.fold
            geometry_current = _choice_probabilities(
                observation,
                columns=tuple(range(3)),
                mean=parameters[fold]["mean"][:3],
                scale=parameters[fold]["scale"][:3],
                coefficients=parameters[fold]["geometry"],
                contract=contract,
            )
            full_current = _choice_probabilities(
                observation,
                columns=tuple(range(10)),
                mean=parameters[fold]["mean"],
                scale=parameters[fold]["scale"],
                coefficients=parameters[fold]["full"],
                contract=contract,
            )
            ordered_moves = sorted(
                (dict(move) for move in get_all_legal_moves(decision.board)),
                key=_move_key,
            )
            if len(ordered_moves) != len(observation.action_outcomes):
                raise ConversionError("second replay action inventory differs")
            safe_mask = np.asarray(
                [
                    outcome == observation.parent_tier
                    for outcome in observation.action_outcomes
                ],
                dtype=bool,
            )
            safe_moves = [
                move for move, safe in zip(ordered_moves, safe_mask, strict=True) if safe
            ]
            if not safe_moves:
                raise ConversionError("A_pos is empty during conversion")
            uniform = np.full(len(safe_moves), 1.0 / len(safe_moves))
            reference_weights = {
                "uniform_A_pos": uniform,
                "geometry_A_pos": normalized_safe_weights(
                    geometry_current, safe_mask
                ),
                "human_frequency_A_pos": normalized_safe_weights(
                    full_current, safe_mask
                ),
            }
            full_risks: list[float] = []
            geometry_risks: list[float] = []
            for move in safe_moves:
                after = decision.board.apply_move(move)
                terminal = terminal_wdl(after) is not None
                if terminal:
                    terminal_safe_successors += 1
                full_risk, geometry_risk, _tier, queries, responses = (
                    _successor_response_risks(
                        after,
                        learner_tier=observation.parent_tier,
                        parameters=parameters[fold],
                        contract=contract,
                        database=database,
                    )
                )
                successor_queries += queries
                successor_response_choice_sets += int(not terminal)
                successor_response_actions += responses
                full_risks.append(full_risk)
                geometry_risks.append(geometry_risk)
                total_queries = extraction["queries"] + successor_queries
                if total_queries > maximum_queries:
                    raise ConversionError("conversion Malom query budget exceeded")
                if time.perf_counter() - started > maximum_seconds:
                    raise ConversionError("conversion active-time budget exceeded")
            full_array = np.asarray(full_risks, dtype=np.float64)
            geometry_array = np.asarray(geometry_risks, dtype=np.float64)
            if observation.parent_tier == "D":
                primary_raw = full_array
                calibration_fit = calibrators[fold]
                primary_corrected = apply_logistic_calibrator(
                    primary_raw,
                    float(calibration_fit["intercept"]),
                    float(calibration_fit["slope"]),
                )
            else:
                primary_raw = np.zeros_like(full_array)
                primary_corrected = np.zeros_like(full_array)
            full_argmax = int(np.argmax(primary_raw))
            geometry_argmax = int(np.argmax(geometry_array))
            side_key = f"{observation.game_id}:{observation.color}"
            columns["player"].append(observation.player_key)
            columns["side_key"].append(side_key)
            columns["phase"].append(observation.phase)
            columns["tier"].append(observation.parent_tier)
            columns["cardinality"].append(len(safe_moves))
            if observation.is_degenerate:
                columns["log_loss_improvement"].append(math.nan)
            else:
                columns["log_loss_improvement"].append(
                    -math.log(float(geometry_current[observation.chosen_index]))
                    + math.log(float(full_current[observation.chosen_index]))
                )
            columns["argmax_agreement"].append(
                int(full_argmax == geometry_argmax)
                if observation.parent_tier == "D"
                else -1
            )
            columns["geometry_argmax_regret"].append(
                float(np.max(primary_raw) - primary_raw[geometry_argmax])
                if observation.parent_tier == "D"
                else math.nan
            )
            columns["primary_risk_span"].append(
                float(np.max(primary_raw) - np.min(primary_raw))
            )
            for name, weights in reference_weights.items():
                raw = uplift_against_reference(primary_raw, weights)
                corrected = uplift_against_reference(primary_corrected, weights)
                secondary = uplift_against_reference(full_array, weights)
                columns[f"raw:{name}"].append(float(raw["uplift"]))
                columns[f"corrected:{name}"].append(float(corrected["uplift"]))
                columns[f"secondary:{name}"].append(float(secondary["uplift"]))
                side_sums_raw[name][side_key] += float(raw["uplift"])
                side_sums_corrected[name][side_key] += float(corrected["uplift"])
            if observation.parent_tier == "D":
                side_d_counts[side_key] += 1
                state_weight = 1.0 / (d_players * d_counts[observation.player_key])
                selected_bin = int(
                    np.searchsorted(risk_boundaries, primary_raw[full_argmax], side="right")
                )
                for name, weights in reference_weights.items():
                    shift[name]["selected_mass"][selected_bin] += state_weight
                    selected_risk = float(primary_raw[full_argmax])
                    shift[name]["selected_below"] += state_weight * (
                        selected_risk < observed_minimum
                    )
                    shift[name]["selected_above"] += state_weight * (
                        selected_risk > observed_maximum
                    )
                    for risk, action_weight in zip(primary_raw, weights, strict=True):
                        risk_bin = int(
                            np.searchsorted(risk_boundaries, risk, side="right")
                        )
                        mass = state_weight * float(action_weight)
                        shift[name]["reference_mass"][risk_bin] += mass
                        shift[name]["reference_below"] += mass * (
                            risk < observed_minimum
                        )
                        shift[name]["reference_above"] += mass * (
                            risk > observed_maximum
                        )
    if observation_index != len(observations):
        raise ConversionError("second replay did not consume all observations")
    if access.denied:
        raise ConversionError("protected conversion access attempt occurred")

    players = np.asarray(columns["player"], dtype=object)
    phases = np.asarray(columns["phase"], dtype=object)
    tiers = np.asarray(columns["tier"], dtype=object)
    cardinalities = np.asarray(columns["cardinality"], dtype=np.int64)
    uplift_reports: dict[str, Any] = {}
    bootstrap_contract = conversion_plan["calibration_and_winner_correction"][
        "bootstrap"
    ]
    for name in references:
        raw = np.asarray(columns[f"raw:{name}"], dtype=np.float64)
        corrected = np.asarray(columns[f"corrected:{name}"], dtype=np.float64)
        secondary = np.asarray(columns[f"secondary:{name}"], dtype=np.float64)
        report: dict[str, Any] = {
            "primary_D_to_L": {
                "all_decisions_raw": _weighted_distribution(raw, players),
                "all_decisions_corrected": _weighted_distribution(corrected, players),
                "parent_D_raw": _weighted_distribution(raw[tiers == "D"], players[tiers == "D"]),
                "parent_D_corrected": _weighted_distribution(
                    corrected[tiers == "D"], players[tiers == "D"]
                ),
                "raw_player_bootstrap": _bootstrap_uplift(
                    raw,
                    players,
                    seed=f"{bootstrap_contract['seed']}:{name}:raw",
                    replicates=int(bootstrap_contract["replicates"]),
                ),
                "corrected_player_bootstrap": _bootstrap_uplift(
                    corrected,
                    players,
                    seed=f"{bootstrap_contract['seed']}:{name}:corrected",
                    replicates=int(bootstrap_contract["replicates"]),
                ),
                "by_phase": {},
                "by_learner_parent_tier": {},
                "by_A_pos_cardinality": {},
            },
            "secondary_all_tier_loss_raw": _weighted_distribution(secondary, players),
        }
        for phase in sorted(set(phases)):
            mask = phases == phase
            report["primary_D_to_L"]["by_phase"][str(phase)] = {
                "raw": _weighted_distribution(raw[mask], players[mask]),
                "corrected": _weighted_distribution(corrected[mask], players[mask]),
            }
        for tier in ("W", "D", "L"):
            mask = tiers == tier
            report["primary_D_to_L"]["by_learner_parent_tier"][tier] = {
                "raw": _weighted_distribution(raw[mask], players[mask]),
                "corrected": _weighted_distribution(corrected[mask], players[mask]),
            }
        labels = np.asarray([_cardinality_label(int(v)) for v in cardinalities])
        for label in ("1", "2", "3-4", "5-8", "9-plus"):
            mask = labels == label
            report["primary_D_to_L"]["by_A_pos_cardinality"][label] = {
                "raw": _weighted_distribution(raw[mask], players[mask]),
                "corrected": _weighted_distribution(corrected[mask], players[mask]),
            }
        raw_mean = report["primary_D_to_L"]["parent_D_raw"][
            "average_unique_player_mean"
        ]
        corrected_mean = report["primary_D_to_L"]["parent_D_corrected"][
            "average_unique_player_mean"
        ]
        report["winner_correction"] = {
            "raw_parent_D_mean": raw_mean,
            "corrected_parent_D_mean": corrected_mean,
            "absolute_attenuation": raw_mean - corrected_mean,
            "retained_fraction": (
                corrected_mean / raw_mean if raw_mean > 0.0 else None
            ),
            "does_not_correct_counterfactual_action_specific_ranking_error": True,
        }
        uplift_reports[name] = report

    low_support_bins = {
        int(row["bin"])
        for row in support_bins
        if int(row["players"]) < 100 or int(row["decisions"]) < 1000
    }
    shift_reports: dict[str, Any] = {}
    for name in references:
        selected = shift[name]["selected_mass"]
        reference = shift[name]["reference_mass"]
        selected /= np.sum(selected)
        reference /= np.sum(reference)
        ratios = [
            float(selected[index] / reference[index])
            if reference[index] > 0.0
            else None
            for index in range(10)
        ]
        shift_reports[name] = {
            "selected_mass": selected.tolist(),
            "reference_mass": reference.tolist(),
            "selected_to_reference_mass_ratio": ratios,
            "Jensen_Shannon_divergence_nats": jensen_shannon_divergence(
                selected, reference
            ),
            "low_support_bins": sorted(low_support_bins),
            "selected_low_support_mass": float(
                sum(selected[index] for index in low_support_bins)
            ),
            "reference_low_support_mass": float(
                sum(reference[index] for index in low_support_bins)
            ),
            "selected_below_observed_risk_range": float(
                shift[name]["selected_below"]
            ),
            "selected_above_observed_risk_range": float(
                shift[name]["selected_above"]
            ),
            "reference_below_observed_risk_range": float(
                shift[name]["reference_below"]
            ),
            "reference_above_observed_risk_range": float(
                shift[name]["reference_above"]
            ),
        }

    log_loss = np.asarray(columns["log_loss_improvement"], dtype=np.float64)
    agreement = np.asarray(columns["argmax_agreement"], dtype=np.int64)
    regret = np.asarray(columns["geometry_argmax_regret"], dtype=np.float64)
    relation_mask = (tiers == "D") & np.isfinite(log_loss) & np.isfinite(regret)
    relation_players = players[relation_mask]
    player_pairs: defaultdict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for player, loss, value, same in zip(
        relation_players,
        log_loss[relation_mask],
        regret[relation_mask],
        agreement[relation_mask],
        strict=True,
    ):
        player_pairs[str(player)].append((float(loss), float(value), int(same)))
    player_loss = np.asarray(
        [np.mean([row[0] for row in player_pairs[player]]) for player in sorted(player_pairs)]
    )
    player_regret = np.asarray(
        [np.mean([row[1] for row in player_pairs[player]]) for player in sorted(player_pairs)]
    )
    player_agreement = np.asarray(
        [np.mean([row[2] for row in player_pairs[player]]) for player in sorted(player_pairs)]
    )
    same_mask = relation_mask & (agreement == 1)
    different_mask = relation_mask & (agreement == 0)

    def unique_player_mean(values: np.ndarray, keys: np.ndarray) -> float | None:
        if values.size == 0:
            return None
        groups: defaultdict[str, list[float]] = defaultdict(list)
        for key, value in zip(keys, values, strict=True):
            groups[str(key)].append(float(value))
        return float(np.mean([np.mean(rows) for rows in groups.values()]))

    relationship = {
        "parent_D_evaluable_decisions": int(np.sum(relation_mask)),
        "players": len(player_pairs),
        "full_vs_geometry_argmax_agreement_average_unique_player": float(
            np.mean(player_agreement)
        ),
        "full_risk_regret_of_geometry_argmax_average_unique_player": float(
            np.mean(player_regret)
        ),
        "player_level_Pearson_log_loss_vs_argmax_regret": _correlation(
            player_loss, player_regret
        ),
        "player_level_Spearman_log_loss_vs_argmax_regret": _correlation(
            _rankdata(player_loss), _rankdata(player_regret)
        ),
        "log_loss_improvement_when_argmax_same": unique_player_mean(
            log_loss[same_mask], players[same_mask]
        ),
        "log_loss_improvement_when_argmax_differs": unique_player_mean(
            log_loss[different_mask], players[different_mask]
        ),
        "log_loss_improvement_is_not_an_argmax_value_estimand": True,
        "empirical_proportionality_accepted": False,
    }

    side_count = len(side_keys)
    mean_d_opportunities = sum(side_d_counts.values()) / side_count
    scenarios = product_scenario_thresholds(
        score_points_per_100=conversion_plan["product_scenarios"][
            "additional_score_points_per_100_games"
        ],
        mean_parent_d_opportunities_per_side_game=mean_d_opportunities,
    )
    product: dict[str, Any] = {
        "side_games": side_count,
        "mean_parent_D_opportunities_per_side_game": mean_d_opportunities,
        "scenario_thresholds": scenarios,
        "references": {},
        "redemption_rate_identified": False,
        "multi_step_dependence_identified": False,
        "log_loss_equivalent_identified": False,
        "D_discrimination_equivalent_identified": False,
    }
    for name in references:
        raw_union = np.asarray(
            [min(1.0, side_sums_raw[name][key]) for key in sorted(side_keys)]
        )
        corrected_union = np.asarray(
            [min(1.0, side_sums_corrected[name][key]) for key in sorted(side_keys)]
        )
        product["references"][name] = {
            "raw_observed_path_perfect_redemption_score_points_per_100_upper": float(
                50.0 * np.mean(raw_union)
            ),
            "corrected_observed_path_perfect_redemption_score_points_per_100_upper": float(
                50.0 * np.mean(corrected_union)
            ),
            "not_a_causal_or_policy_rollout_estimate": True,
        }

    elapsed = time.perf_counter() - started
    total_queries = extraction["queries"] + successor_queries
    if total_queries > maximum_queries or elapsed > maximum_seconds:
        raise ConversionError("conversion completed outside frozen budget")
    access_audit = {
        "successful": {
            "research-exploration:original_OOF_reproduction": int(
                extraction["access_audit"]["successful"][
                    "research-exploration:raw_replay_and_decisions"
                ]
            ),
            "research-exploration:hypothetical_successor_replay": int(
                access.successful[("research-exploration", "raw_replay_and_decisions")]
            ),
        },
        "denied": {},
        "research_confirmation_content_reads": 0,
        "official_selection_content_reads": 0,
        "official_confirmation_content_reads": 0,
        "official_final_test_content_reads": 0,
        "source_pool_2eb04f54_reads_or_consumption": 0,
        "human_db_reads": 0,
        "database_writes": 0,
        "games_searches_strategy_models_or_training": 0,
    }
    return {
        "sealed_OOF_reproduction": reproduction,
        "query_accounting": {
            "original_choice_inventory_queries": extraction["queries"],
            "hypothetical_successor_queries": successor_queries,
            "total_Malom_queries": total_queries,
            "elapsed_seconds": elapsed,
            "terminal_safe_successors": terminal_safe_successors,
            "successor_response_choice_sets": successor_response_choice_sets,
            "successor_response_actions": successor_response_actions,
        },
        "sample": {
            "games": len(sample_rows),
            "decisions": len(observations),
            "players": len(set(players)),
            "A_pos_cardinality_one_decisions": int(np.sum(cardinalities == 1)),
            "A_pos_cardinality_one_fraction": float(np.mean(cardinalities == 1)),
            "A_pos_cardinality_greater_than_one_fraction": float(
                np.mean(cardinalities > 1)
            ),
        },
        "calibration": {
            "observed_OOF_parent_D": calibration,
            "fold_external_calibrators": {
                str(fold): value for fold, value in calibrators.items()
            },
        },
        "uplift_distributions": uplift_reports,
        "policy_shift_support": shift_reports,
        "log_loss_argmax_relationship": relationship,
        "product_upper_bound": product,
        "access_audit": access_audit,
    }


__all__ = [
    "PLAN_SCHEMA",
    "PLAN_V2_SCHEMA",
    "RESULT_SCHEMA",
    "ConversionError",
    "apply_logistic_calibrator",
    "derive_product_conversion",
    "fit_logistic_calibrator",
    "jensen_shannon_divergence",
    "load_conversion_plan",
    "load_effective_conversion_plan",
    "load_readiness_result",
    "normalized_safe_weights",
    "product_scenario_thresholds",
    "tier_loss_outcomes",
    "uplift_against_reference",
]
