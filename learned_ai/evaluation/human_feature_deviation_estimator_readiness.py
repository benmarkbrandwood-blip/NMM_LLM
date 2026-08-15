"""Exploration-only readiness calibration for human choice estimation.

The estimator in this module is a research statistic.  It is not a gameplay
policy and is never wired into inference or training.  Its oracle labels are
position-only ``A_pos`` labels; no function here constructs ``A_allow``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import networkx as nx
import numpy as np

from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    concentration,
)


PLAN_SCHEMA = "nmm.human-feature-deviation-estimator-readiness-plan.v1"
PLAN_V2_SCHEMA = "nmm.human-feature-deviation-estimator-readiness-plan.v2"
STRUCTURE_SCHEMA = "nmm.human-feature-deviation-estimator-crossfit.v1"
RESULT_SCHEMA = "nmm.human-feature-deviation-estimator-readiness-result.v1"


class EstimatorReadinessError(RuntimeError):
    """Raised when a frozen requirement cannot be satisfied."""


def _seed_integer(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big")


def _hash_rank(seed: str, value: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return digest, value


def load_readiness_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the immutable pre-result readiness contract."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EstimatorReadinessError(f"invalid readiness plan: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_SCHEMA:
        raise EstimatorReadinessError("readiness plan schema differs")
    identity = value.get("plan_identity")
    body = dict(value)
    body.pop("plan_identity", None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise EstimatorReadinessError("readiness plan identity differs")
    return value, hashlib.sha256(raw).hexdigest()


def load_effective_readiness_plan(
    path: str | Path,
    *,
    inherited_v1_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load v2 and resolve its byte-bound inheritance from frozen v1."""
    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EstimatorReadinessError(f"invalid v2 readiness plan: {source}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_V2_SCHEMA:
        raise EstimatorReadinessError("v2 readiness plan schema differs")
    identity = value.get("plan_identity")
    body = dict(value)
    body.pop("plan_identity", None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise EstimatorReadinessError("v2 readiness plan identity differs")
    inherited, inherited_sha = load_readiness_plan(inherited_v1_path)
    disposition = value.get("v1_disposition")
    if not isinstance(disposition, Mapping) or disposition != {
        "plan_identity": inherited["plan_identity"],
        "plan_file_sha256": inherited_sha,
        "remains_immutable": True,
        "status": "superseded_for_future_execution_by_structural_fold_correction",
        "failure": (
            "the outcome-blind greedy affinity assignment left only 286 players "
            "in same-fold sample games, below the frozen 800-player gate"
        ),
        "new_human_actions_features_malom_labels_or_outcomes_read_before_failure": 0,
    }:
        raise EstimatorReadinessError("v2 disposition does not bind frozen v1")
    inheritance = value.get("frozen_inheritance")
    sections = inheritance.get("sections") if isinstance(inheritance, Mapping) else None
    if not isinstance(sections, list) or any(
        name not in inherited for name in sections
    ):
        raise EstimatorReadinessError("v2 inherited sections differ")
    effective = {name: inherited[name] for name in sections}
    effective.update(
        {
            "schema_version": PLAN_V2_SCHEMA,
            "plan_identity": identity,
            "cross_fit_contract": value["cross_fit_contract"],
            "v1_plan_identity": inherited["plan_identity"],
            "v1_plan_file_sha256": inherited_sha,
            "confirmation_execution_authorized": False,
        }
    )
    return effective, {
        "v2_plan_identity": identity,
        "v2_plan_file_sha256": hashlib.sha256(raw).hexdigest(),
        "v1_plan_identity": inherited["plan_identity"],
        "v1_plan_file_sha256": inherited_sha,
    }


@dataclass(frozen=True)
class NumericalContract:
    """The frozen optimizer and numerical failure thresholds."""

    ridge_lambda: float = 0.01
    reporting_floor: float = 1e-12
    reporting_ceiling: float = 0.999999999999
    maximum_iterations: int = 100
    history_size: int = 10
    gradient_tolerance: float = 1e-7
    relative_objective_tolerance: float = 1e-10
    relative_objective_consecutive: int = 3
    secondary_gradient_tolerance: float = 1e-6
    initial_step: float = 1.0
    armijo_c1: float = 1e-4
    backtracking_factor: float = 0.5
    maximum_line_search_steps: int = 25
    maximum_absolute_coefficient: float = 20.0
    maximum_information_condition: float = 1e12
    minimum_information_eigenvalue: float = 1e-10
    near_separation_probability: float = 0.9999999999
    maximum_near_separation_fraction: float = 0.01

    @classmethod
    def from_plan(cls, plan: Mapping[str, Any]) -> "NumericalContract":
        numeric = plan["numerical_contract"]
        optimizer = numeric["optimizer"]
        probability = numeric["probability_handling"]
        separation = numeric["finite_and_separation_checks"]
        return cls(
            ridge_lambda=float(numeric["regularization"]["lambda"]),
            reporting_floor=float(probability["reporting_floor"]),
            reporting_ceiling=float(probability["reporting_ceiling"]),
            maximum_iterations=int(optimizer["maximum_iterations"]),
            history_size=int(optimizer["history_size"]),
            gradient_tolerance=float(optimizer["gradient_infinity_tolerance"]),
            relative_objective_tolerance=float(
                optimizer["relative_objective_tolerance"]
            ),
            relative_objective_consecutive=int(
                optimizer["relative_objective_consecutive_iterations"]
            ),
            secondary_gradient_tolerance=float(
                optimizer["secondary_gradient_tolerance_for_objective_stop"]
            ),
            initial_step=float(optimizer["initial_step"]),
            armijo_c1=float(optimizer["armijo_c1"]),
            backtracking_factor=float(optimizer["backtracking_factor"]),
            maximum_line_search_steps=int(optimizer["maximum_line_search_steps"]),
            maximum_absolute_coefficient=float(
                separation["maximum_absolute_coefficient"]
            ),
            maximum_information_condition=float(
                separation["maximum_observed_information_condition_number"]
            ),
            minimum_information_eigenvalue=float(
                separation["minimum_observed_information_eigenvalue"]
            ),
            near_separation_probability=float(
                separation["near_separation_chosen_probability"]
            ),
            maximum_near_separation_fraction=float(
                separation[
                    "maximum_near_separation_choice_fraction_when_"
                    "coefficient_norm_exceeds_10"
                ]
            ),
        )

    @classmethod
    def for_tests(cls, **changes: Any) -> "NumericalContract":
        return replace(cls(), **changes)


@dataclass(frozen=True)
class ChoiceObservation:
    """One complete legal human choice set and its observed action."""

    player_key: str
    game_id: str
    decision_index: int
    fold: int
    features: np.ndarray
    chosen_index: int
    parent_tier: str
    action_outcomes: tuple[str, ...]
    phase: str
    color: str

    def __post_init__(self) -> None:
        rows = np.asarray(self.features)
        if rows.ndim != 2 or rows.shape[0] == 0:
            raise EstimatorReadinessError("choice feature matrix is empty")
        if self.chosen_index < 0 or self.chosen_index >= rows.shape[0]:
            raise EstimatorReadinessError("chosen index is outside choice set")
        if len(self.action_outcomes) != rows.shape[0]:
            raise EstimatorReadinessError("action outcome count differs")
        if not np.all(np.isfinite(rows)):
            raise EstimatorReadinessError("choice feature matrix is nonfinite")

    @property
    def is_degenerate(self) -> bool:
        return self.features.shape[0] == 1


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    objective: float
    gradient_infinity_norm: float
    iterations: int
    converged: bool
    convergence_reason: str


def _move_key(move: Mapping[str, Any]) -> tuple[str, str, str]:
    destination = move.get("to")
    if not isinstance(destination, str) or not destination:
        raise EstimatorReadinessError("choice move destination is missing")
    source = move.get("from")
    capture = move.get("capture")
    if source is not None and not isinstance(source, str):
        raise EstimatorReadinessError("choice move source is invalid")
    if capture is not None and not isinstance(capture, str):
        raise EstimatorReadinessError("choice move capture is invalid")
    return str(source or ""), destination, str(capture or "")


def canonicalize_choice_inventory(
    moves: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, float]],
    observed_move: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    maximum_actions: int,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Validate and lexically order a complete atomic action inventory."""
    if not moves:
        raise EstimatorReadinessError("choice inventory is empty")
    if len(moves) > maximum_actions:
        raise EstimatorReadinessError("choice inventory exceeds maximum actions")
    if len(moves) != len(feature_rows):
        raise EstimatorReadinessError("choice feature row count differs")
    normalized: list[tuple[tuple[str, str, str], dict[str, Any], list[float]]] = []
    for move, row in zip(moves, feature_rows, strict=True):
        key = _move_key(move)
        values: list[float] = []
        if tuple(row) != tuple(feature_names):
            raise EstimatorReadinessError("choice feature fields differ")
        for name in feature_names:
            value = float(row[name])
            if not math.isfinite(value):
                raise EstimatorReadinessError("choice feature is nonfinite")
            values.append(value)
        normalized.append(
            (
                key,
                {
                    "from": move.get("from"),
                    "to": move.get("to"),
                    "capture": move.get("capture"),
                },
                values,
            )
        )
    keys = [row[0] for row in normalized]
    if len(set(keys)) != len(keys):
        raise EstimatorReadinessError("duplicate normalized move key")
    observed_key = _move_key(observed_move)
    observed_rows = [index for index, key in enumerate(keys) if key == observed_key]
    if len(observed_rows) != 1:
        raise EstimatorReadinessError("observed choice is absent or ambiguous")
    ordered = sorted(normalized, key=lambda row: row[0])
    ordered_keys = [row[0] for row in ordered]
    chosen_index = ordered_keys.index(observed_key)
    return (
        [row[1] for row in ordered],
        np.asarray([row[2] for row in ordered], dtype=np.float64),
        chosen_index,
    )


def _stable_probabilities(
    scores: np.ndarray,
    *,
    floor: float,
    ceiling: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise EstimatorReadinessError("softmax scores are empty or nonfinite")
    maximum = float(np.max(values))
    shifted = values - maximum
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0:
        raise EstimatorReadinessError("softmax denominator is invalid")
    probabilities = exponentials / denominator
    reported = np.clip(probabilities, floor, ceiling)
    if not np.all(np.isfinite(reported)):
        raise EstimatorReadinessError("softmax probability is nonfinite")
    return reported, maximum + math.log(denominator)


def standardize_from_training_fold(
    choices: Sequence[ChoiceObservation],
    *,
    columns: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not choices:
        raise EstimatorReadinessError("training fold is empty")
    selected = tuple(int(value) for value in columns)
    rows = np.vstack([choice.features[:, selected] for choice in choices])
    if rows.size == 0 or not np.all(np.isfinite(rows)):
        raise EstimatorReadinessError("training standardization rows are invalid")
    mean = np.mean(rows, axis=0, dtype=np.float64)
    scale = np.std(rows, axis=0, ddof=0, dtype=np.float64)
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale < 1e-12)
    ):
        raise EstimatorReadinessError("training feature scale is zero or nonfinite")
    return mean, scale


def _player_choice_weights(
    choices: Sequence[ChoiceObservation],
) -> dict[str, float]:
    counts = Counter(choice.player_key for choice in choices)
    if not counts:
        raise EstimatorReadinessError("fit has no actor players")
    players = len(counts)
    return {player: 1.0 / (players * count) for player, count in counts.items()}


def _objective_and_gradient(
    coefficients: np.ndarray,
    choices: Sequence[ChoiceObservation],
    *,
    columns: Sequence[int],
    mean: np.ndarray,
    scale: np.ndarray,
    ridge_lambda: float,
) -> tuple[float, np.ndarray]:
    beta = np.asarray(coefficients, dtype=np.float64)
    selected = tuple(int(value) for value in columns)
    if beta.shape != (len(selected),):
        raise EstimatorReadinessError("coefficient dimension differs")
    if not np.all(np.isfinite(beta)):
        raise EstimatorReadinessError("coefficients are nonfinite")
    weights = _player_choice_weights(choices)
    objective = 0.0
    gradient = np.zeros_like(beta)
    for choice in choices:
        if choice.is_degenerate:
            raise EstimatorReadinessError("degenerate choice reached objective")
        matrix = (choice.features[:, selected] - mean) / scale
        scores = matrix @ beta
        maximum = float(np.max(scores))
        exponentials = np.exp(scores - maximum)
        denominator = float(np.sum(exponentials, dtype=np.float64))
        if denominator <= 0 or not math.isfinite(denominator):
            raise EstimatorReadinessError("objective softmax is invalid")
        probabilities = exponentials / denominator
        log_sum = maximum + math.log(denominator)
        weight = weights[choice.player_key]
        objective += weight * (log_sum - float(scores[choice.chosen_index]))
        gradient += weight * (matrix.T @ probabilities - matrix[choice.chosen_index])
    objective += 0.5 * ridge_lambda * float(beta @ beta)
    gradient += ridge_lambda * beta
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise EstimatorReadinessError("objective or gradient is nonfinite")
    return objective, gradient


def _lbfgs_direction(
    gradient: np.ndarray,
    history: Sequence[tuple[np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    vector = gradient.copy()
    alphas: list[float] = []
    for step, change, inverse_curvature in reversed(history):
        alpha = inverse_curvature * float(step @ vector)
        alphas.append(alpha)
        vector -= alpha * change
    if history:
        last_step, last_change, _inverse = history[-1]
        denominator = float(last_change @ last_change)
        scale = float(last_step @ last_change) / denominator if denominator else 1.0
        vector *= max(scale, 1e-12)
    for (step, change, inverse_curvature), alpha in zip(
        history, reversed(alphas), strict=True
    ):
        beta = inverse_curvature * float(change @ vector)
        vector += step * (alpha - beta)
    return -vector


def fit_conditional_logit(
    choices: Sequence[ChoiceObservation],
    *,
    columns: Sequence[int],
    mean: np.ndarray,
    scale: np.ndarray,
    contract: NumericalContract,
) -> FitResult:
    """Fit the frozen convex conditional logit with deterministic L-BFGS."""
    beta = np.zeros(len(tuple(columns)), dtype=np.float64)
    objective, gradient = _objective_and_gradient(
        beta,
        choices,
        columns=columns,
        mean=mean,
        scale=scale,
        ridge_lambda=contract.ridge_lambda,
    )
    gradient_norm = float(np.max(np.abs(gradient)))
    if gradient_norm <= contract.gradient_tolerance:
        return FitResult(beta, objective, gradient_norm, 0, True, "gradient")
    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    stable_iterations = 0
    non_descent_restarts = 0
    for iteration in range(1, contract.maximum_iterations + 1):
        direction = _lbfgs_direction(gradient, history)
        directional = float(gradient @ direction)
        if not math.isfinite(directional) or directional >= 0:
            if non_descent_restarts:
                raise EstimatorReadinessError("repeated non-descent direction")
            history.clear()
            direction = -gradient
            directional = -float(gradient @ gradient)
            non_descent_restarts += 1
        step_size = contract.initial_step
        accepted: tuple[np.ndarray, float, np.ndarray] | None = None
        for _line_step in range(contract.maximum_line_search_steps):
            trial = beta + step_size * direction
            trial_objective, trial_gradient = _objective_and_gradient(
                trial,
                choices,
                columns=columns,
                mean=mean,
                scale=scale,
                ridge_lambda=contract.ridge_lambda,
            )
            if trial_objective <= (
                objective + contract.armijo_c1 * step_size * directional
            ):
                accepted = trial, trial_objective, trial_gradient
                break
            step_size *= contract.backtracking_factor
        if accepted is None:
            raise EstimatorReadinessError("line search failed")
        next_beta, next_objective, next_gradient = accepted
        step = next_beta - beta
        change = next_gradient - gradient
        curvature = float(step @ change)
        if math.isfinite(curvature) and curvature > 1e-12:
            history.append((step, change, 1.0 / curvature))
            if len(history) > contract.history_size:
                history.pop(0)
        relative = abs(objective - next_objective) / max(1.0, abs(objective))
        stable_iterations = (
            stable_iterations + 1
            if relative <= contract.relative_objective_tolerance
            else 0
        )
        beta, objective, gradient = next_beta, next_objective, next_gradient
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= contract.gradient_tolerance:
            return FitResult(
                beta, objective, gradient_norm, iteration, True, "gradient"
            )
        if (
            stable_iterations >= contract.relative_objective_consecutive
            and gradient_norm <= contract.secondary_gradient_tolerance
        ):
            return FitResult(
                beta,
                objective,
                gradient_norm,
                iteration,
                True,
                "relative_objective_and_secondary_gradient",
            )
    raise EstimatorReadinessError("optimizer did not converge")


def diagnose_separation(
    *,
    coefficients: np.ndarray,
    information: np.ndarray,
    chosen_probabilities: np.ndarray,
    contract: NumericalContract,
) -> dict[str, float]:
    beta = np.asarray(coefficients, dtype=np.float64)
    matrix = np.asarray(information, dtype=np.float64)
    probabilities = np.asarray(chosen_probabilities, dtype=np.float64)
    if (
        not np.all(np.isfinite(beta))
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(probabilities))
    ):
        raise EstimatorReadinessError("finite separation diagnostic failed")
    maximum = float(np.max(np.abs(beta))) if beta.size else 0.0
    if maximum > contract.maximum_absolute_coefficient:
        raise EstimatorReadinessError("coefficient separation threshold exceeded")
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(np.min(eigenvalues))
    maximum_eigenvalue = float(np.max(eigenvalues))
    condition = maximum_eigenvalue / minimum if minimum > 0 else math.inf
    if minimum < contract.minimum_information_eigenvalue:
        raise EstimatorReadinessError("observed information eigenvalue too small")
    if condition > contract.maximum_information_condition:
        raise EstimatorReadinessError("observed information condition too large")
    near_fraction = float(
        np.mean(probabilities >= contract.near_separation_probability)
    )
    if np.linalg.norm(beta) > 10.0 and (
        near_fraction > contract.maximum_near_separation_fraction
    ):
        raise EstimatorReadinessError("near-complete separation diagnostic triggered")
    return {
        "maximum_absolute_coefficient": maximum,
        "minimum_information_eigenvalue": minimum,
        "information_condition_number": condition,
        "near_separation_choice_fraction": near_fraction,
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    observations = np.asarray(values, dtype=np.float64)
    masses = np.asarray(weights, dtype=np.float64)
    if (
        observations.ndim != 1
        or observations.size == 0
        or observations.shape != masses.shape
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
        or float(np.sum(masses)) <= 0
        or quantile < 0
        or quantile > 1
    ):
        raise EstimatorReadinessError("weighted quantile input is invalid")
    order = np.argsort(observations, kind="stable")
    sorted_values = observations[order]
    cumulative = np.cumsum(masses[order], dtype=np.float64)
    target = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EstimatorReadinessError("bootstrap distribution is empty")
    return float(np.quantile(np.asarray(values), probability, method="linear"))


def player_cluster_bootstrap(
    values: Mapping[str, float],
    *,
    replicates: int,
    seed: str,
    statistic: str,
) -> dict[str, Any]:
    """Bootstrap scalar player clusters without effect-creating priors."""
    if statistic != "mean_and_sd":
        raise EstimatorReadinessError("unsupported bootstrap statistic")
    players = sorted(values)
    if len(players) < 2 or replicates <= 0:
        raise EstimatorReadinessError("bootstrap support is insufficient")
    observed = np.asarray([float(values[player]) for player in players])
    if not np.all(np.isfinite(observed)):
        raise EstimatorReadinessError("bootstrap values are nonfinite")
    rng = np.random.default_rng(_seed_integer(seed))
    means: list[float] = []
    deviations: list[float] = []
    for _ in range(replicates):
        sample = observed[rng.integers(0, observed.size, size=observed.size)]
        means.append(float(np.mean(sample)))
        deviations.append(float(np.std(sample, ddof=1)))
    return {
        "players": len(players),
        "point_mean": float(np.mean(observed)),
        "point_sd": float(np.std(observed, ddof=1)),
        "mean_interval": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "sd_interval": [
            _percentile(deviations, 0.025),
            _percentile(deviations, 0.975),
        ],
        "replicates": replicates,
        "seed": seed,
        "zero_events_not_smoothed": bool(np.all(observed == 0.0)),
    }


def _structural_decisions(
    games: Sequence[tuple[str, str, str, int]],
) -> Counter[str]:
    result: Counter[str] = Counter()
    for _session, white, black, moves in games:
        result[white] += (int(moves) + 1) // 2
        result[black] += int(moves) // 2
    return result


def build_crossfit_structure(
    *,
    assigned_players: Sequence[str],
    games: Sequence[tuple[str, str, str, int]],
    folds: int,
    capacities: Sequence[int],
    fold_seed: str,
    sample_seed: str,
    maximum_games_per_fold: int,
) -> dict[str, Any]:
    """Build outcome-blind player folds and a player-covering game sample."""
    players = sorted(str(player) for player in assigned_players)
    if len(players) != len(set(players)) or sum(capacities) != len(players):
        raise EstimatorReadinessError("fold player or capacity contract differs")
    if len(capacities) != folds or any(capacity <= 0 for capacity in capacities):
        raise EstimatorReadinessError("fold capacities are invalid")
    player_set = set(players)
    game_rows = [
        (str(session), str(white), str(black), int(moves))
        for session, white, black, moves in games
    ]
    if len({row[0] for row in game_rows}) != len(game_rows):
        raise EstimatorReadinessError("crossfit game IDs are duplicated")
    if any(
        white not in player_set or black not in player_set
        for _session, white, black, _moves in game_rows
    ):
        raise EstimatorReadinessError("crossfit game has unassigned player")
    decisions = _structural_decisions(game_rows)
    edge_weights: dict[str, Counter[str]] = defaultdict(Counter)
    for _session, white, black, _moves in game_rows:
        if white != black:
            edge_weights[white][black] += 1
            edge_weights[black][white] += 1
    ordered_players = sorted(
        players,
        key=lambda player: (
            -decisions[player],
            _hash_rank(fold_seed, player),
        ),
    )
    fold_players: list[list[str]] = [[] for _ in range(folds)]
    fold_decisions = [0] * folds
    player_fold: dict[str, int] = {}
    for player in ordered_players:
        choices: list[tuple[int, int, int, tuple[bytes, str], int]] = []
        for fold in range(folds):
            if len(fold_players[fold]) >= capacities[fold]:
                continue
            internal_weight = sum(
                weight
                for opponent, weight in edge_weights[player].items()
                if player_fold.get(opponent) == fold
            )
            choices.append(
                (
                    -internal_weight,
                    len(fold_players[fold]),
                    fold_decisions[fold],
                    _hash_rank(f"{fold_seed}:{player}", str(fold)),
                    fold,
                )
            )
        if not choices:
            raise EstimatorReadinessError("no fold capacity remains")
        fold = min(choices)[-1]
        fold_players[fold].append(player)
        fold_decisions[fold] += decisions[player]
        player_fold[player] = fold
    if sorted(len(group) for group in fold_players) != sorted(capacities):
        raise EstimatorReadinessError("fold capacities were not filled")

    same_fold: list[tuple[str, str, str, int, int]] = []
    cross_fold: list[tuple[str, str, str, int]] = []
    for session, white, black, moves in game_rows:
        if player_fold[white] == player_fold[black]:
            same_fold.append((session, white, black, moves, player_fold[white]))
        else:
            cross_fold.append((session, white, black, moves))

    sampled: list[tuple[str, str, str, int, int]] = []
    for fold in range(folds):
        eligible = [row for row in same_fold if row[4] == fold]
        remaining = {row[0]: row for row in eligible}
        coverable = {player for row in eligible for player in row[1:3]}
        uncovered = set(coverable)
        selected: list[tuple[str, str, str, int, int]] = []
        while uncovered and len(selected) < maximum_games_per_fold:
            ranked = sorted(
                remaining.values(),
                key=lambda row: (
                    -int(row[1] in uncovered) - int(row[2] in uncovered),
                    _hash_rank(f"{sample_seed}:{fold}", row[0]),
                ),
            )
            if not ranked:
                break
            chosen = ranked[0]
            new_coverage = {chosen[1], chosen[2]} & uncovered
            if not new_coverage:
                break
            selected.append(chosen)
            remaining.pop(chosen[0])
            uncovered -= new_coverage
        fill = sorted(
            remaining.values(),
            key=lambda row: _hash_rank(f"{sample_seed}:{fold}", row[0]),
        )
        selected.extend(fill[: max(0, maximum_games_per_fold - len(selected))])
        sampled.extend(selected)
    sampled.sort(key=lambda row: (row[4], row[0]))
    sample_players = {player for row in sampled for player in row[1:3]}
    return {
        "player_fold": dict(sorted(player_fold.items())),
        "fold_players": [sorted(group) for group in fold_players],
        "fold_player_counts": [len(group) for group in fold_players],
        "fold_structural_decisions": fold_decisions,
        "same_fold_games": len(same_fold),
        "cross_fold_games": len(cross_fold),
        "cross_fold_game_fraction": len(cross_fold) / len(game_rows),
        "sample_games": [
            {
                "session_id": row[0],
                "white": row[1],
                "black": row[2],
                "move_count": row[3],
                "fold": row[4],
            }
            for row in sampled
        ],
        "sample_game_count": len(sampled),
        "sample_decisions": sum(row[3] for row in sampled),
        "sample_players": len(sample_players),
        "sample_session_identity": canonical_sha256([row[0] for row in sampled]),
        "cross_fold_session_identity": canonical_sha256(
            sorted(row[0] for row in cross_fold)
        ),
    }


def build_community_crossfit_structure(
    *,
    assigned_players: Sequence[str],
    games: Sequence[tuple[str, str, str, int]],
    folds: int,
    community_resolution: float,
    community_seed: int,
    sample_seed: str,
    maximum_games_per_fold: int,
) -> dict[str, Any]:
    """Build the corrected whole-community, outcome-blind five-fold split."""
    players = sorted(str(player) for player in assigned_players)
    player_set = set(players)
    if len(players) != len(player_set) or folds <= 1:
        raise EstimatorReadinessError("community fold player contract differs")
    game_rows = [
        (str(session), str(white), str(black), int(moves))
        for session, white, black, moves in games
    ]
    if len({row[0] for row in game_rows}) != len(game_rows):
        raise EstimatorReadinessError("community fold games are duplicated")
    if any(
        white not in player_set or black not in player_set
        for _session, white, black, _moves in game_rows
    ):
        raise EstimatorReadinessError("community fold game has unknown player")
    player_decisions = _structural_decisions(game_rows)
    graph = nx.Graph()
    graph.add_nodes_from(players)
    for _session, white, black, _moves in game_rows:
        if white == black:
            continue
        previous = graph.get_edge_data(white, black, {}).get("games", 0)
        graph.add_edge(white, black, games=int(previous) + 1)
    communities = list(
        nx.community.louvain_communities(
            graph,
            weight="games",
            resolution=community_resolution,
            seed=community_seed,
        )
    )
    if not communities or set().union(*communities) != player_set:
        raise EstimatorReadinessError("Louvain communities do not cover players")
    target_players = len(players) / folds
    target_decisions = sum(player_decisions.values()) / folds
    fold_players: list[set[str]] = [set() for _ in range(folds)]
    fold_decisions = [0] * folds
    ordered_communities = sorted(
        communities,
        key=lambda group: (
            -sum(player_decisions[player] for player in group),
            -len(group),
            canonical_sha256(sorted(group)),
        ),
    )
    community_rows: list[dict[str, Any]] = []
    for community in ordered_communities:
        count = len(community)
        decision_mass = sum(player_decisions[player] for player in community)
        candidates: list[tuple[float, float, int]] = []
        for fold in range(folds):
            player_ratio = (len(fold_players[fold]) + count) / target_players
            decision_ratio = (fold_decisions[fold] + decision_mass) / target_decisions
            candidates.append(
                (
                    max(player_ratio, decision_ratio),
                    abs(player_ratio - 1.0) + abs(decision_ratio - 1.0),
                    fold,
                )
            )
        fold = min(candidates)[-1]
        fold_players[fold].update(community)
        fold_decisions[fold] += decision_mass
        community_rows.append(
            {
                "identity": canonical_sha256(sorted(community)),
                "players": count,
                "structural_decisions": decision_mass,
                "fold": fold,
            }
        )
    player_fold = {
        player: fold for fold, group in enumerate(fold_players) for player in group
    }
    if set(player_fold) != player_set:
        raise EstimatorReadinessError("community folds do not cover players")
    same_fold: list[tuple[str, str, str, int, int]] = []
    cross_fold: list[tuple[str, str, str, int]] = []
    for session, white, black, moves in game_rows:
        if player_fold[white] == player_fold[black]:
            same_fold.append((session, white, black, moves, player_fold[white]))
        else:
            cross_fold.append((session, white, black, moves))

    sampled: list[tuple[str, str, str, int, int]] = []
    for fold in range(folds):
        eligible = [row for row in same_fold if row[4] == fold]
        remaining = {row[0]: row for row in eligible}
        uncovered = {player for row in eligible for player in row[1:3]}
        selected: list[tuple[str, str, str, int, int]] = []
        while uncovered and len(selected) < maximum_games_per_fold:
            ranked = sorted(
                remaining.values(),
                key=lambda row: (
                    -int(row[1] in uncovered) - int(row[2] in uncovered),
                    _hash_rank(f"{sample_seed}:{fold}", row[0]),
                ),
            )
            if not ranked:
                break
            chosen = ranked[0]
            new_coverage = {chosen[1], chosen[2]} & uncovered
            if not new_coverage:
                break
            selected.append(chosen)
            remaining.pop(chosen[0])
            uncovered -= new_coverage
        fill = sorted(
            remaining.values(),
            key=lambda row: _hash_rank(f"{sample_seed}:{fold}", row[0]),
        )
        selected.extend(fill[: max(0, maximum_games_per_fold - len(selected))])
        sampled.extend(selected)
    sampled.sort(key=lambda row: (row[4], row[0]))

    fold_metrics: list[dict[str, Any]] = []
    all_sample_players: set[str] = set()
    for fold in range(folds):
        rows = [row for row in sampled if row[4] == fold]
        counts = _structural_decisions(
            [(row[0], row[1], row[2], row[3]) for row in rows]
        )
        all_sample_players.update(counts)
        fold_metrics.append(
            {
                "fold": fold,
                "assigned_players": len(fold_players[fold]),
                "participating_players": len(counts),
                "games": len(rows),
                "decisions": sum(counts.values()),
                "player_decision_concentration": concentration(list(counts.values())),
                "session_identity": canonical_sha256([row[0] for row in rows]),
                "player_identity": canonical_sha256(sorted(counts)),
            }
        )
    return {
        "algorithm": {
            "method": "whole-community Louvain balanced assignment",
            "community_resolution": community_resolution,
            "community_seed": community_seed,
            "community_weight": "games",
            "communities": len(communities),
            "modularity": nx.community.modularity(
                graph,
                communities,
                weight="games",
                resolution=community_resolution,
            ),
            "community_player_sizes": sorted(
                (len(group) for group in communities), reverse=True
            ),
            "community_rows": community_rows,
        },
        "player_fold": dict(sorted(player_fold.items())),
        "fold_player_counts": [len(group) for group in fold_players],
        "fold_structural_decisions": fold_decisions,
        "same_fold_games": len(same_fold),
        "cross_fold_games": len(cross_fold),
        "cross_fold_game_fraction": len(cross_fold) / len(game_rows),
        "cross_fold_session_identity": canonical_sha256(
            sorted(row[0] for row in cross_fold)
        ),
        "sample_games": [
            {
                "session_id": row[0],
                "white": row[1],
                "black": row[2],
                "move_count": row[3],
                "fold": row[4],
            }
            for row in sampled
        ],
        "sample_game_count": len(sampled),
        "sample_decisions": sum(row[3] for row in sampled),
        "sample_players": len(all_sample_players),
        "sample_session_identity": canonical_sha256([row[0] for row in sampled]),
        "fold_metrics": fold_metrics,
    }


@dataclass
class EstimatorAccess:
    """Explicit allowlist for the frozen expanded exploration sample."""

    official_partition_by_session: Mapping[str, str]
    research_partition_by_session: Mapping[str, str]
    allowed_sessions: frozenset[str]
    successful: Counter[tuple[str, str]] = field(default_factory=Counter)
    denied: Counter[tuple[str, str]] = field(default_factory=Counter)

    def assert_allowed(self, session_id: str, *, access_kind: str) -> None:
        official = self.official_partition_by_session.get(session_id, "outside")
        research = self.research_partition_by_session.get(session_id, "outside")
        if (
            official != "train"
            or research != "research-exploration"
            or session_id not in self.allowed_sessions
        ):
            label = f"{official}:{research}"
            self.denied[(label, access_kind)] += 1
            raise EstimatorReadinessError(
                f"estimator exploration access denied for {label}: {session_id}"
            )

    def derive(
        self,
        session_id: str,
        *,
        access_kind: str,
        producer: Callable[[], Any],
    ) -> Any:
        self.assert_allowed(session_id, access_kind=access_kind)
        value = producer()
        self.successful[("research-exploration", access_kind)] += 1
        return value


__all__ = [
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "STRUCTURE_SCHEMA",
    "ChoiceObservation",
    "EstimatorAccess",
    "EstimatorReadinessError",
    "FitResult",
    "NumericalContract",
    "_objective_and_gradient",
    "_stable_probabilities",
    "_weighted_quantile",
    "build_community_crossfit_structure",
    "build_crossfit_structure",
    "canonicalize_choice_inventory",
    "diagnose_separation",
    "fit_conditional_logit",
    "load_effective_readiness_plan",
    "load_readiness_plan",
    "player_cluster_bootstrap",
    "standardize_from_training_fold",
]
