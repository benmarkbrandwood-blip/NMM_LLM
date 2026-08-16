"""Post-hoc transfer test between Sanmill inducibility and OOF human risk.

The implementation cannot launch Sanmill, play games, fit an estimator, or
load policy weights.  It joins sealed state/action outcomes to the frozen
cross-fit coefficients and, only after a separate plan is sealed, may query
the corrected Malom tablebase to reconstruct successor response risks.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ai.malom_db import MalomDB
from game.board import BoardState
from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    NumericalContract,
)
from learned_ai.evaluation.human_feature_deviation_product_conversion import (
    _fold_parameters,
    _successor_response_risks,
)


AUDIT_SCHEMA = "nmm.sanmill-human-transfer-coverage-audit.v1"
PLAN_SCHEMA = "nmm.sanmill-human-transfer-plan.v1"
RESULT_SCHEMA = "nmm.sanmill-human-transfer-result.v1"
PRIMARY_BUDGET = 100_000
PHASES = ("placement", "movement", "flying")


class TransferError(RuntimeError):
    """Raised when a sealed boundary or required value differs."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one file without changing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(
    path: str | Path,
    *,
    identity_field: str,
    schema: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and independently verify one canonical sealed JSON object."""
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransferError(f"cannot load sealed JSON: {source}") from exc
    if not isinstance(value, dict):
        raise TransferError(f"sealed JSON is not an object: {source}")
    if schema is not None and value.get("schema_version") != schema:
        raise TransferError(f"sealed JSON schema differs: {source}")
    identity = value.get(identity_field)
    if not isinstance(identity, str) or len(identity) != 64:
        raise TransferError(f"sealed identity is absent: {source}")
    body = dict(value)
    body.pop(identity_field)
    if canonical_sha256(body) != identity:
        raise TransferError(f"sealed identity differs: {source}")
    return value, hashlib.sha256(raw).hexdigest()


def _move_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    move = row.get("move", row)
    if not isinstance(move, Mapping):
        raise TransferError("action move is malformed")
    return (
        str(move.get("from") or ""),
        str(move.get("to") or ""),
        str(move.get("capture") or ""),
    )


def audit_coverage(
    *,
    pool: Mapping[str, Any],
    crossfit: Mapping[str, Any],
    readiness: Mapping[str, Any],
    conversion: Mapping[str, Any],
    main_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit OOF availability without reading any outcome or risk value."""
    states = pool.get("states")
    sample = crossfit.get("structure", {}).get("sample_games")
    player_fold = crossfit.get("structure", {}).get("player_fold")
    if not isinstance(states, list) or len(states) != 360:
        raise TransferError("main state pool is not the frozen 360-state pool")
    if not isinstance(sample, list) or len(sample) != 6_400:
        raise TransferError("cross-fit sample is not the frozen 6,400 games")
    if not isinstance(player_fold, Mapping):
        raise TransferError("cross-fit player-fold map is absent")
    sample_by_session = {str(row["session_id"]): row for row in sample}
    if len(sample_by_session) != len(sample):
        raise TransferError("cross-fit sample sessions are duplicated")
    reports = readiness.get("analysis", {}).get("folds")
    if not isinstance(reports, list) or len(reports) != 5:
        raise TransferError("readiness fold reports are absent")
    parameter_folds = {int(row["fold"]) for row in reports}
    if parameter_folds != set(range(5)):
        raise TransferError("readiness fold parameters are incomplete")
    for report in reports:
        if (
            len(report.get("feature_mean", [])) != 10
            or len(report.get("feature_scale", [])) != 10
            or len(report.get("full_fit", {}).get("coefficients", [])) != 10
            or len(report.get("geometry_fit", {}).get("coefficients", [])) != 3
        ):
            raise TransferError("readiness coefficient dimensions differ")

    result_state_ids = {
        str(row["state_id"])
        for row in main_result.get("analysis", {}).get("measurements", [])
    }
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    fold_counts: Counter[int] = Counter()
    cardinalities: list[int] = []
    for state in states:
        reasons: list[str] = []
        session_id = str(state.get("session_id"))
        sample_row = sample_by_session.get(session_id)
        if sample_row is None:
            reasons.append("session_absent_from_frozen_crossfit_sample")
            actor = None
            opponent = None
            fold = None
        else:
            side = str(state.get("side_to_move"))
            if side == "W":
                actor = str(sample_row["white"])
                opponent = str(sample_row["black"])
            elif side == "B":
                actor = str(sample_row["black"])
                opponent = str(sample_row["white"])
            else:
                actor = None
                opponent = None
                reasons.append("invalid_side_to_move")
            fold = int(sample_row["fold"])
            if actor is not None and player_fold.get(actor) != fold:
                reasons.append("actor_not_held_out_in_session_fold")
            if opponent is not None and player_fold.get(opponent) != fold:
                reasons.append("opponent_not_in_same_frozen_fold")
            if fold not in parameter_folds:
                reasons.append("fold_parameters_absent")
        actions = state.get("a_pos")
        if not isinstance(actions, list) or not actions:
            reasons.append("a_pos_absent")
        elif any(
            not isinstance(action.get("successor_fen"), str)
            or not action.get("successor_fen")
            for action in actions
        ):
            reasons.append("successor_fen_absent")
        if str(state.get("state_id")) not in result_state_ids:
            reasons.append("main_measurement_state_absent")
        for reason in reasons:
            reason_counts[reason] += 1
        available = not reasons
        if available:
            phase_counts[str(state["phase"])] += 1
            fold_counts[int(fold)] += 1
            cardinalities.append(int(state["a_pos_cardinality"]))
        rows.append(
            {
                "state_id": str(state.get("state_id")),
                "session_id": session_id,
                "logical_ply": int(state.get("logical_ply", -1)),
                "phase": str(state.get("phase")),
                "a_pos_cardinality": int(state.get("a_pos_cardinality", 0)),
                "actor_player_key": actor,
                "opponent_player_key": opponent,
                "oof_fold": fold,
                "available": available,
                "failure_reasons": reasons,
            }
        )
    available_count = sum(bool(row["available"]) for row in rows)
    persisted = conversion.get("analysis", {}).get("action_predictions")
    if persisted is not None:
        raise TransferError("unexpected persisted action-prediction surface")
    return {
        "state_rows": rows,
        "coverage": {
            "states": len(rows),
            "available_states": available_count,
            "unavailable_states": len(rows) - available_count,
            "fraction": available_count / len(rows),
            "failure_reasons": dict(sorted(reason_counts.items())),
            "available_by_phase": dict(sorted(phase_counts.items())),
            "available_by_fold": {
                str(key): value for key, value in sorted(fold_counts.items())
            },
            "available_a_pos_cardinality": _distribution(cardinalities),
        },
        "selection_bias_audit": {
            "required": available_count != len(rows),
            "result": (
                "not_applicable_full_coverage"
                if available_count == len(rows)
                else "required_before_analysis"
            ),
        },
        "prediction_material": {
            "fold_parameters_persisted": True,
            "per_successor_predictions_persisted": False,
            "conversion_query_accounting_persisted": conversion.get("analysis", {})
            .get("query_accounting", {}),
            "exact_recomputation_required": True,
            "estimator_refit_required": False,
        },
        "scope_explanation": {
            "cross_fold_discard_precedes_pool_selection": True,
            "crossfit_sample_same_fold_games": int(
                crossfit["structure"]["same_fold_games"]
            ),
            "crossfit_frozen_sample_games": len(sample),
            "main_pool_source_games_checked_individually": len(rows),
        },
    }


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": int(np.min(array)),
        "median": float(np.median(array)),
        "maximum": int(np.max(array)),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        raise TransferError("cannot take a percentile of an empty sample")
    position = probability * (array.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(array[lower])
    weight = position - lower
    return float(array[lower] * (1.0 - weight) + array[upper] * weight)


def _bootstrap(
    rows: Sequence[Mapping[str, float]],
    *,
    seed: str,
    repetitions: int,
) -> dict[str, Any]:
    if not rows:
        raise TransferError("bootstrap rows are empty")
    matrix = np.asarray(
        [[row["selected"], row["b"], row["o"]] for row in rows],
        dtype=np.float64,
    )
    a = float(np.mean(matrix[:, 0]))
    b = float(np.mean(matrix[:, 1]))
    o = float(np.mean(matrix[:, 2]))
    if o <= b:
        raise TransferError("oracle headroom is nonpositive")
    point = np.asarray([a, a - b, (a - b) / (o - b)], dtype=np.float64)
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    )
    a_distribution = np.empty(repetitions, dtype=np.float64)
    gain_distribution = np.empty(repetitions, dtype=np.float64)
    transfer_distribution: list[float] = []
    for index in range(repetitions):
        sample = matrix[rng.integers(0, len(matrix), len(matrix))]
        sample_a = float(np.mean(sample[:, 0]))
        sample_b = float(np.mean(sample[:, 1]))
        sample_o = float(np.mean(sample[:, 2]))
        a_distribution[index] = sample_a
        gain_distribution[index] = sample_a - sample_b
        if sample_o > sample_b:
            transfer_distribution.append(
                (sample_a - sample_b) / (sample_o - sample_b)
            )
    if not transfer_distribution:
        raise TransferError("all bootstrap samples have nonpositive oracle headroom")
    distributions = {
        "A": a_distribution,
        "A_minus_b": gain_distribution,
        "transfer": np.asarray(transfer_distribution, dtype=np.float64),
    }
    points = {"A": point[0], "A_minus_b": point[1], "transfer": point[2]}
    return {
        "states": len(rows),
        "b": b,
        "o": o,
        "o_minus_b": o - b,
        **{
            name: {
                "point": float(points[name]),
                "percentile_95": {
                    "lower": _percentile(distribution, 0.025),
                    "upper": _percentile(distribution, 0.975),
                },
            }
            for name, distribution in distributions.items()
        },
        "bootstrap": {
            "repetitions": repetitions,
            "seed": seed,
            "transfer_defined_repetitions": len(transfer_distribution),
            "transfer_undefined_nonpositive_headroom_repetitions": (
                repetitions - len(transfer_distribution)
            ),
        },
    }


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0.0) + 0.5 * (comparisons == 0.0)))


def _auc_report(
    rows: Sequence[Mapping[str, Any]], *, seed: str, repetitions: int
) -> dict[str, Any]:
    values = np.asarray(
        [float(row["auc"]) for row in rows if row.get("auc") is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "eligible_states": 0,
            "abstained_states": len(rows),
            "mean": None,
            "percentile_95": None,
        }
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    )
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = float(
            np.mean(values[rng.integers(0, values.size, values.size)])
        )
    return {
        "eligible_states": int(values.size),
        "abstained_states": len(rows) - int(values.size),
        "mean": float(np.mean(values)),
        "percentile_95": {
            "lower": _percentile(means, 0.025),
            "upper": _percentile(means, 0.975),
        },
        "bootstrap": {"repetitions": repetitions, "seed": seed},
    }


def _action_outcomes(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, tuple[str, str, str]], dict[int, bool]]:
    outcomes: dict[tuple[str, tuple[str, str, str]], dict[int, bool]] = defaultdict(dict)
    for row in measurements:
        if row.get("abstained"):
            raise TransferError("main result contains an abstained cell")
        key = (str(row["state_id"]), _move_key(row["safe_action"]))
        budget = int(row["node_budget"])
        if budget in outcomes[key]:
            raise TransferError("main result action-budget cell is duplicated")
        outcomes[key][budget] = row.get("downgrade_transition") is not None
    return outcomes


def _category_flags(values: Mapping[int, bool]) -> tuple[bool, bool]:
    budgets = (1_000, 100_000, 500_000)
    if set(values) != set(budgets):
        raise TransferError("main action does not have all frozen node budgets")
    flags = [bool(values[budget]) for budget in budgets]
    return all(flags), any(flags) and not all(flags)


def analyze_transfer(
    *,
    plan: Mapping[str, Any],
    pool: Mapping[str, Any],
    main_result: Mapping[str, Any],
    correction: Mapping[str, Any],
    readiness: Mapping[str, Any],
    crossfit: Mapping[str, Any],
    database: MalomDB,
) -> dict[str, Any]:
    """Execute the one frozen transfer estimator without engine access."""
    if plan.get("status") != "frozen_before_transfer_outcome_calculation":
        raise TransferError("transfer plan is not frozen")
    gate = plan["coverage_gate"]
    states = pool["states"]
    if len(states) < int(gate["minimum_states"]):
        raise TransferError("coverage gate failed")
    if len(states) / int(gate["reference_states"]) < float(
        gate["minimum_fraction"]
    ):
        raise TransferError("coverage fraction gate failed")
    if correction.get("primary_decision_reproduced_unchanged", {}).get(
        "decision"
    ) != "mechanism_gate_passed":
        raise TransferError("corrected main decision differs")

    sample_by_session = {
        str(row["session_id"]): row
        for row in crossfit["structure"]["sample_games"]
    }
    parameters = _fold_parameters(readiness)
    contract = NumericalContract.from_plan(readiness["frozen_contract"])
    outcomes = _action_outcomes(main_result["analysis"]["measurements"])
    query_limit = int(plan["resource_envelope"]["maximum_malom_queries"])
    time_limit = float(plan["resource_envelope"]["maximum_active_seconds"])
    repetitions = int(plan["interval"]["repetitions"])
    base_seed = str(plan["interval"]["seed"])
    started = time.perf_counter()
    query_count = 0
    response_sets = 0
    response_actions = 0
    terminal_successors = 0
    state_rows: list[dict[str, Any]] = []

    for state in states:
        session_id = str(state["session_id"])
        try:
            fold = int(sample_by_session[session_id]["fold"])
        except KeyError as exc:
            raise TransferError("state lost frozen OOF fold coverage") from exc
        action_rows = sorted(state["a_pos"], key=_move_key)
        full_risks: list[float] = []
        geometry_risks: list[float] = []
        primary_events: list[bool] = []
        invariant: list[bool] = []
        sensitive: list[bool] = []
        for action in action_rows:
            key = (str(state["state_id"]), _move_key(action))
            if key not in outcomes:
                raise TransferError("state action is absent from main result")
            budget_flags = outcomes[key]
            primary_events.append(bool(budget_flags[PRIMARY_BUDGET]))
            inv, sens = _category_flags(budget_flags)
            invariant.append(inv)
            sensitive.append(sens)
            board = BoardState.from_fen_string(str(action["successor_fen"]))
            full, geometry, _tier, queries, responses = _successor_response_risks(
                board,
                learner_tier=str(state["learner_parent_tier"]),
                parameters=parameters[fold],
                contract=contract,
                database=database,
            )
            full_risks.append(float(full))
            geometry_risks.append(float(geometry))
            query_count += int(queries)
            response_sets += int(responses > 0)
            response_actions += int(responses)
            terminal_successors += int(responses == 0)
            if query_count > query_limit:
                raise TransferError("transfer Malom query ceiling exceeded")
            if time.perf_counter() - started > time_limit:
                raise TransferError("transfer active-time ceiling exceeded")
        events = np.asarray(primary_events, dtype=bool)
        full_array = np.asarray(full_risks, dtype=np.float64)
        geometry_array = np.asarray(geometry_risks, dtype=np.float64)
        if not np.all(np.isfinite(full_array)) or not np.all(
            np.isfinite(geometry_array)
        ):
            raise TransferError("human-risk prediction is nonfinite")
        full_choice = int(np.argmax(full_array))
        geometry_choice = int(np.argmax(geometry_array))
        inv_array = np.asarray(invariant, dtype=bool)
        sens_array = np.asarray(sensitive, dtype=bool)
        auc_full = None
        auc_geometry = None
        if np.any(events) and np.any(~events):
            auc_full = _auc(full_array[events], full_array[~events])
            auc_geometry = _auc(geometry_array[events], geometry_array[~events])
        state_rows.append(
            {
                "state_id": str(state["state_id"]),
                "session_id": session_id,
                "logical_ply": int(state["logical_ply"]),
                "phase": str(state["phase"]),
                "fold": fold,
                "a_pos_cardinality": len(action_rows),
                "b": float(np.mean(events)),
                "o": float(np.any(events)),
                "full": {
                    "selected_index": full_choice,
                    "selected_action": action_rows[full_choice]["move"],
                    "selected": float(events[full_choice]),
                    "selected_invariant": bool(inv_array[full_choice]),
                    "selected_sensitive": bool(sens_array[full_choice]),
                    "maximum_risk": float(full_array[full_choice]),
                    "argmax_tie_count": int(np.sum(full_array == full_array[full_choice])),
                    "auc": auc_full,
                },
                "geometry": {
                    "selected_index": geometry_choice,
                    "selected_action": action_rows[geometry_choice]["move"],
                    "selected": float(events[geometry_choice]),
                    "selected_invariant": bool(inv_array[geometry_choice]),
                    "selected_sensitive": bool(sens_array[geometry_choice]),
                    "maximum_risk": float(geometry_array[geometry_choice]),
                    "argmax_tie_count": int(
                        np.sum(geometry_array == geometry_array[geometry_choice])
                    ),
                    "auc": auc_geometry,
                },
                "action_rows": [
                    {
                        "move": action["move"],
                        "engine_downgrade_100000": bool(event),
                        "budget_invariant": bool(inv),
                        "budget_sensitive": bool(sens),
                        "full_risk": float(full_risk),
                        "geometry_risk": float(geometry_risk),
                    }
                    for action, event, inv, sens, full_risk, geometry_risk in zip(
                        action_rows,
                        events,
                        inv_array,
                        sens_array,
                        full_array,
                        geometry_array,
                        strict=True,
                    )
                ],
            }
        )

    analyses: dict[str, Any] = {}
    for specification in ("full", "geometry"):
        metric_rows = [
            {"selected": row[specification]["selected"], "b": row["b"], "o": row["o"]}
            for row in state_rows
        ]
        report = _bootstrap(
            metric_rows,
            seed=f"{base_seed}:{specification}:primary",
            repetitions=repetitions,
        )
        report["tie_states"] = sum(
            row[specification]["argmax_tie_count"] > 1 for row in state_rows
        )
        report["mean_argmax_tie_count"] = float(
            np.mean([row[specification]["argmax_tie_count"] for row in state_rows])
        )
        report["within_state_AUC"] = _auc_report(
            [row[specification] for row in state_rows],
            seed=f"{base_seed}:{specification}:auc",
            repetitions=repetitions,
        )
        report["by_phase"] = {
            phase: _bootstrap(
                [
                    {
                        "selected": row[specification]["selected"],
                        "b": row["b"],
                        "o": row["o"],
                    }
                    for row in state_rows
                    if row["phase"] == phase
                ],
                seed=f"{base_seed}:{specification}:phase:{phase}",
                repetitions=repetitions,
            )
            for phase in PHASES
        }
        report["by_A_pos_cardinality"] = {}
        bins = {
            "1": lambda value: value == 1,
            "2": lambda value: value == 2,
            "3-4": lambda value: 3 <= value <= 4,
            "5-8": lambda value: 5 <= value <= 8,
            "9-plus": lambda value: value >= 9,
        }
        for label, predicate in bins.items():
            subset = [
                {
                    "selected": row[specification]["selected"],
                    "b": row["b"],
                    "o": row["o"],
                }
                for row in state_rows
                if predicate(int(row["a_pos_cardinality"]))
            ]
            report["by_A_pos_cardinality"][label] = (
                _bootstrap(
                    subset,
                    seed=f"{base_seed}:{specification}:cardinality:{label}",
                    repetitions=repetitions,
                )
                if subset and any(row["o"] > row["b"] for row in subset)
                else {"states": len(subset), "estimable": False}
            )
        report["budget_type"] = {}
        for category in ("invariant", "sensitive"):
            selected = np.asarray(
                [row[specification][f"selected_{category}"] for row in state_rows],
                dtype=np.float64,
            )
            uniform = np.asarray(
                [
                    np.mean([action[f"budget_{category}"] for action in row["action_rows"]])
                    for row in state_rows
                ],
                dtype=np.float64,
            )
            oracle = np.asarray(
                [
                    any(action[f"budget_{category}"] for action in row["action_rows"])
                    for row in state_rows
                ],
                dtype=np.float64,
            )
            report["budget_type"][category] = _bootstrap(
                [
                    {"selected": s, "b": b, "o": o}
                    for s, b, o in zip(selected, uniform, oracle, strict=True)
                ],
                seed=f"{base_seed}:{specification}:budget:{category}",
                repetitions=repetitions,
            )
        analyses[specification] = report

    paired = np.asarray(
        [row["full"]["selected"] - row["geometry"]["selected"] for row in state_rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"{base_seed}:paired".encode()).digest()[:8], "big")
    )
    paired_boot = [
        float(np.mean(paired[rng.integers(0, len(paired), len(paired))]))
        for _ in range(repetitions)
    ]
    full = analyses["full"]
    existence = full["A_minus_b"]["percentile_95"]["lower"] > 0.0
    substantive = (
        existence
        and full["transfer"]["percentile_95"]["lower"]
        >= float(plan["decision_rule"]["substantive_transfer_threshold"])
    )
    if substantive:
        decision = "A_substantive_transfer_exists"
    elif existence:
        decision = "B_transfer_exists_but_not_substantive"
    else:
        decision = "C_no_transfer"
    elapsed = time.perf_counter() - started
    return {
        "decision": {
            "code": decision,
            "existence_gate_passed": existence,
            "substantive_gate_passed": substantive,
            "human_specific_increment_supported": _percentile(paired_boot, 0.025)
            > 0.0,
        },
        "primary_full": analyses["full"],
        "geometry_control": analyses["geometry"],
        "full_minus_geometry_selected_downgrade": {
            "point": float(np.mean(paired)),
            "percentile_95": {
                "lower": _percentile(paired_boot, 0.025),
                "upper": _percentile(paired_boot, 0.975),
            },
            "secondary_only": True,
        },
        "state_rows": state_rows,
        "resource_use": {
            "sanmill_queries": 0,
            "complete_games": 0,
            "model_loads": 0,
            "estimator_refits": 0,
            "training_updates": 0,
            "database_writes": 0,
            "malom_queries": query_count,
            "successor_response_choice_sets": response_sets,
            "successor_response_actions": response_actions,
            "terminal_successors": terminal_successors,
            "active_seconds": elapsed,
            "maximum_malom_queries": query_limit,
            "maximum_active_seconds": time_limit,
        },
    }


__all__ = [
    "AUDIT_SCHEMA",
    "PLAN_SCHEMA",
    "PRIMARY_BUDGET",
    "RESULT_SCHEMA",
    "TransferError",
    "analyze_transfer",
    "audit_coverage",
    "load_sealed",
    "sha256_file",
]
