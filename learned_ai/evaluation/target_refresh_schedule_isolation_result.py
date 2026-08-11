"""Frozen schedule and decision rules for the target-refresh successor."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from learned_ai.evaluation.phase_replay_development_corpus import (
    PHASES,
    validate_phase_replay_development_corpus,
)
from learned_ai.training.run_contract import canonical_sha256


MEASUREMENT_ROW_SCHEMA = "nmm.schedule-isolation-outcome-measurement.v1"
RESULT_SCHEMA = "nmm.target-refresh-schedule-isolation-result.v1"
EXPECTED_CONDITIONS = ("refresh-once", "no-refresh")
OUTCOME_BOUNDARIES = (4096, 8192)
CANDIDATE_COLORS = ("W", "B")
PRIMARY_TEMPERATURE = 0.2
MAX_POST_START_LOGICAL_PLIES = 120
MINIMUM_AGGREGATE_SCORE_EFFECT = 1.0 / 12.0
MINIMUM_PER_SEED_SCORE_EFFECT = 1.0 / 24.0
MAXIMUM_OPPOSITE_PHASE_EFFECT = 0.25
MAXIMUM_TRUNCATION_RATE_INCREASE = 0.10
MAXIMUM_OPPOSITE_MALOM_MASS_EFFECT = 0.05
MINIMUM_SUPPORTING_SEEDS = 2


class ScheduleIsolationResultError(RuntimeError):
    """Raised when successor measurement or decision evidence is incomplete."""


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScheduleIsolationResultError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ScheduleIsolationResultError(f"{field} must be finite")
    return result


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if (
        len(seeds) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or any(seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ScheduleIsolationResultError("measurement requires three unique seeds")
    return tuple(seeds)


def build_outcome_measurement_schedule(
    *,
    seeds: Sequence[int],
    corpus: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the exact paired no-update game schedule before models are read."""
    normalized_seeds = _validated_seeds(seeds)
    validate_phase_replay_development_corpus(corpus)
    rows: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        for boundary in OUTCOME_BOUNDARIES:
            for record in corpus["records"]:
                for candidate_color in CANDIDATE_COLORS:
                    pair_body = {
                        "schema": "nmm.schedule-isolation-outcome-pair.v1",
                        "seed": seed,
                        "boundary": boundary,
                        "record_identity": record["record_identity"],
                        "candidate_color": candidate_color,
                    }
                    pair_identity = canonical_sha256(pair_body)
                    torch_seed = int(pair_identity[:16], 16) & ((1 << 63) - 1)
                    for condition in EXPECTED_CONDITIONS:
                        game_identity = canonical_sha256(
                            {
                                **pair_body,
                                "condition": condition,
                            }
                        )
                        rows.append(
                            {
                                "schema_version": MEASUREMENT_ROW_SCHEMA,
                                "measurement_index": len(rows),
                                "game_id": f"game:{game_identity}",
                                "paired_game_identity": pair_identity,
                                "seed": seed,
                                "post_fork_consumed_transitions": boundary,
                                "condition": condition,
                                "record_index": record["record_index"],
                                "record_identity": record["record_identity"],
                                "phase": record["phase"],
                                "candidate_color": candidate_color,
                                "torch_seed": torch_seed,
                                "sampling_temperature": PRIMARY_TEMPERATURE,
                                "max_post_start_logical_plies": (
                                    MAX_POST_START_LOGICAL_PLIES
                                ),
                                "no_update": True,
                            }
                        )
    return rows


def _wdl(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["outcome_class"]) for row in rows)
    return {
        "games": len(rows),
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "score_rate": sum(float(row["score"]) for row in rows) / len(rows),
        "max_ply_truncations": sum(
            row["termination_reason"] == "max-ply-truncation" for row in rows
        ),
        "max_ply_truncation_rate": sum(
            row["termination_reason"] == "max-ply-truncation" for row in rows
        )
        / len(rows),
    }


def _grouped(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: _wdl(groups[name]) for name in sorted(groups)}


def validate_and_summarize_outcome_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete paired grid and summarize every required stratum."""
    expected = build_outcome_measurement_schedule(seeds=seeds, corpus=corpus)
    if len(rows) != len(expected):
        raise ScheduleIsolationResultError("outcome measurement row count differs")
    result_fields = {
        "candidate_checkpoint_id",
        "anchor_checkpoint_id",
        "start_history_sha256",
        "end_history_sha256",
        "start_logical_ply_count",
        "end_logical_ply_count",
        "training_reward_outcome",
        "outcome_class",
        "score",
        "post_start_logical_plies",
        "termination_reason",
    }
    pair_rows: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    game_ids: set[str] = set()
    for index, (observed, scheduled) in enumerate(zip(rows, expected, strict=True)):
        missing = (set(scheduled) | result_fields) - set(observed)
        if missing:
            raise ScheduleIsolationResultError(
                f"measurement row {index} lacks fields: {sorted(missing)}"
            )
        projection = {key: observed[key] for key in scheduled}
        if projection != scheduled:
            raise ScheduleIsolationResultError(
                f"measurement row {index} differs from frozen schedule"
            )
        game_id = str(observed["game_id"])
        if game_id in game_ids:
            raise ScheduleIsolationResultError("measurement game id is duplicated")
        game_ids.add(game_id)
        outcome_class = str(observed["outcome_class"])
        score = _finite(observed["score"], field=f"row {index} score")
        expected_score = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(
            outcome_class
        )
        if expected_score is None or score != expected_score:
            raise ScheduleIsolationResultError("outcome class and score differ")
        _finite(
            observed["training_reward_outcome"],
            field=f"row {index} training outcome",
        )
        start_ply = observed["start_logical_ply_count"]
        end_ply = observed["end_logical_ply_count"]
        post_start = observed["post_start_logical_plies"]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (
                start_ply,
                end_ply,
                post_start,
            ))
            or start_ply < 0
            or post_start < 0
            or post_start > MAX_POST_START_LOGICAL_PLIES
            or end_ply - start_ply != post_start
        ):
            raise ScheduleIsolationResultError("measurement ply accounting differs")
        for field in ("start_history_sha256", "end_history_sha256"):
            identity = observed[field]
            if not isinstance(identity, str) or len(identity) != 64:
                raise ScheduleIsolationResultError(
                    f"measurement {field} identity differs"
                )
        pair = pair_rows[str(observed["paired_game_identity"])]
        condition = str(observed["condition"])
        if condition in pair:
            raise ScheduleIsolationResultError("paired condition is duplicated")
        pair[condition] = observed

    paired_differences: list[dict[str, Any]] = []
    for pair_identity, pair in pair_rows.items():
        if set(pair) != set(EXPECTED_CONDITIONS):
            raise ScheduleIsolationResultError("paired outcome cell is incomplete")
        refresh = pair["refresh-once"]
        no_refresh = pair["no-refresh"]
        invariant_fields = (
            "seed",
            "post_fork_consumed_transitions",
            "record_index",
            "record_identity",
            "phase",
            "candidate_color",
            "torch_seed",
            "sampling_temperature",
            "max_post_start_logical_plies",
            "start_history_sha256",
            "start_logical_ply_count",
            "anchor_checkpoint_id",
        )
        if any(refresh[field] != no_refresh[field] for field in invariant_fields):
            raise ScheduleIsolationResultError("paired common-random cell differs")
        paired_differences.append(
            {
                "paired_game_identity": pair_identity,
                "seed": refresh["seed"],
                "post_fork_consumed_transitions": refresh[
                    "post_fork_consumed_transitions"
                ],
                "record_index": refresh["record_index"],
                "phase": refresh["phase"],
                "candidate_color": refresh["candidate_color"],
                "no_refresh_minus_refresh_score": (
                    float(no_refresh["score"]) - float(refresh["score"])
                ),
            }
        )

    cells: dict[str, Any] = {}
    for seed in _validated_seeds(seeds):
        seed_rows = [row for row in rows if row["seed"] == seed]
        seed_differences = [row for row in paired_differences if row["seed"] == seed]
        by_boundary: dict[str, Any] = {}
        for boundary in OUTCOME_BOUNDARIES:
            boundary_rows = [
                row
                for row in seed_rows
                if row["post_fork_consumed_transitions"] == boundary
            ]
            differences = [
                row["no_refresh_minus_refresh_score"]
                for row in seed_differences
                if row["post_fork_consumed_transitions"] == boundary
            ]
            by_condition = {
                condition: _wdl(
                    [row for row in boundary_rows if row["condition"] == condition]
                )
                for condition in EXPECTED_CONDITIONS
            }
            by_boundary[str(boundary)] = {
                "by_condition": by_condition,
                "by_phase_and_condition": {
                    phase: {
                        condition: _wdl(
                            [
                                row
                                for row in boundary_rows
                                if row["phase"] == phase
                                and row["condition"] == condition
                            ]
                        )
                        for condition in EXPECTED_CONDITIONS
                    }
                    for phase in PHASES
                },
                "by_candidate_color_and_condition": {
                    color: {
                        condition: _wdl(
                            [
                                row
                                for row in boundary_rows
                                if row["candidate_color"] == color
                                and row["condition"] == condition
                            ]
                        )
                        for condition in EXPECTED_CONDITIONS
                    }
                    for color in CANDIDATE_COLORS
                },
                "by_termination_reason": _grouped(
                    boundary_rows,
                    lambda row: f"{row['condition']}:{row['termination_reason']}",
                ),
                "paired_no_refresh_minus_refresh_mean_score": (
                    sum(differences) / len(differences)
                ),
            }
        cells[str(seed)] = by_boundary
    return {
        "games": len(rows),
        "paired_games": len(paired_differences),
        "by_seed_boundary": cells,
        "paired_differences": paired_differences,
    }


def _effect_direction(value: float, *, minimum: float) -> str:
    if value >= minimum:
        return "no-refresh"
    if value <= -minimum:
        return "refresh-once"
    return "none"


def decide_schedule_isolation_result(
    *,
    policy_decision: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Combine preregistered policy and paired-outcome gates without promotion."""
    normalized_seeds = _validated_seeds(seeds)
    by_seed = outcome_summary.get("by_seed_boundary")
    if not isinstance(by_seed, Mapping) or set(by_seed) != {
        str(seed) for seed in normalized_seeds
    }:
        raise ScheduleIsolationResultError("outcome summary seed grid differs")

    seed_effects: dict[str, dict[str, float | str]] = {}
    for seed in normalized_seeds:
        boundaries = by_seed[str(seed)]
        if set(boundaries) != {str(value) for value in OUTCOME_BOUNDARIES}:
            raise ScheduleIsolationResultError("outcome summary boundaries differ")
        early = _finite(
            boundaries[str(OUTCOME_BOUNDARIES[0])][
                "paired_no_refresh_minus_refresh_mean_score"
            ],
            field=f"seed {seed} early outcome effect",
        )
        final = _finite(
            boundaries[str(OUTCOME_BOUNDARIES[1])][
                "paired_no_refresh_minus_refresh_mean_score"
            ],
            field=f"seed {seed} final outcome effect",
        )
        seed_effects[str(seed)] = {
            "4096": early,
            "8192": final,
            "early_direction": _effect_direction(
                early, minimum=MINIMUM_PER_SEED_SCORE_EFFECT
            ),
            "final_direction": _effect_direction(
                final, minimum=MINIMUM_PER_SEED_SCORE_EFFECT
            ),
        }

    aggregate = {
        str(boundary): sum(
            float(seed_effects[str(seed)][str(boundary)])
            for seed in normalized_seeds
        )
        / len(normalized_seeds)
        for boundary in OUTCOME_BOUNDARIES
    }
    final_direction = _effect_direction(
        aggregate[str(OUTCOME_BOUNDARIES[1])],
        minimum=MINIMUM_AGGREGATE_SCORE_EFFECT,
    )
    supporting_seeds = [
        str(seed)
        for seed in normalized_seeds
        if seed_effects[str(seed)]["early_direction"] == final_direction
        and seed_effects[str(seed)]["final_direction"] == final_direction
    ]
    persistence_pass = (
        final_direction != "none"
        and len(supporting_seeds) >= MINIMUM_SUPPORTING_SEEDS
    )

    final_cells = [
        by_seed[str(seed)][str(OUTCOME_BOUNDARIES[1])]
        for seed in normalized_seeds
    ]
    phase_effects: dict[str, float] = {}
    for phase in PHASES:
        effects = []
        for cell in final_cells:
            phase_cells = cell["by_phase_and_condition"][phase]
            effects.append(
                float(phase_cells["no-refresh"]["score_rate"])
                - float(phase_cells["refresh-once"]["score_rate"])
            )
        phase_effects[phase] = sum(effects) / len(effects)
    if final_direction == "no-refresh":
        phase_safety_pass = all(
            effect >= -MAXIMUM_OPPOSITE_PHASE_EFFECT
            for effect in phase_effects.values()
        )
    elif final_direction == "refresh-once":
        phase_safety_pass = all(
            effect <= MAXIMUM_OPPOSITE_PHASE_EFFECT
            for effect in phase_effects.values()
        )
    else:
        phase_safety_pass = False

    truncation_rates = {
        condition: sum(
            float(cell["by_condition"][condition]["max_ply_truncation_rate"])
            for cell in final_cells
        )
        / len(final_cells)
        for condition in EXPECTED_CONDITIONS
    }
    truncation_delta = (
        truncation_rates["no-refresh"] - truncation_rates["refresh-once"]
    )
    if final_direction == "no-refresh":
        truncation_safety_pass = (
            truncation_delta <= MAXIMUM_TRUNCATION_RATE_INCREASE
        )
    elif final_direction == "refresh-once":
        truncation_safety_pass = (
            truncation_delta >= -MAXIMUM_TRUNCATION_RATE_INCREASE
        )
    else:
        truncation_safety_pass = False

    policy_by_seed = policy_decision.get("by_seed", {})
    signed_malom: dict[str, float] = {}
    for phase in PHASES:
        values = [
            _finite(
                policy_by_seed[str(seed)]["by_transition_boundary"][
                    str(OUTCOME_BOUNDARIES[1])
                ]["observed"]["phase_signed_malom_preserving_mass_delta"][phase],
                field=f"seed {seed} {phase} signed Malom effect",
            )
            for seed in normalized_seeds
        ]
        signed_malom[phase] = sum(values) / len(values)
    if final_direction == "no-refresh":
        malom_safety_pass = all(
            value >= -MAXIMUM_OPPOSITE_MALOM_MASS_EFFECT
            for value in signed_malom.values()
        )
    elif final_direction == "refresh-once":
        malom_safety_pass = all(
            value <= MAXIMUM_OPPOSITE_MALOM_MASS_EFFECT
            for value in signed_malom.values()
        )
    else:
        malom_safety_pass = False

    policy_pass = policy_decision.get("classification") == "materially_diverged"
    supported = all(
        (
            policy_pass,
            persistence_pass,
            phase_safety_pass,
            truncation_safety_pass,
            malom_safety_pass,
        )
    )
    if supported:
        classification = "schedule_isolated_target_refresh_effect"
        selected_condition = final_direction
    elif final_direction == "none":
        classification = "no_material_paired_outcome_effect"
        selected_condition = None
    elif not policy_pass:
        classification = "outcome_effect_without_persistent_policy_gate"
        selected_condition = None
    else:
        classification = "inconclusive_safety_or_persistence_disagreement"
        selected_condition = None
    return {
        "classification": classification,
        "contrast_definition": "no-refresh minus refresh-once",
        "seed_effects": seed_effects,
        "aggregate_effects": aggregate,
        "final_direction": final_direction,
        "supporting_persistent_seeds": supporting_seeds,
        "minimum_supporting_seeds": MINIMUM_SUPPORTING_SEEDS,
        "minimum_aggregate_score_effect": MINIMUM_AGGREGATE_SCORE_EFFECT,
        "minimum_per_seed_score_effect": MINIMUM_PER_SEED_SCORE_EFFECT,
        "phase_effects": phase_effects,
        "maximum_opposite_phase_effect": MAXIMUM_OPPOSITE_PHASE_EFFECT,
        "truncation_rates": truncation_rates,
        "no_refresh_minus_refresh_truncation_rate": truncation_delta,
        "maximum_truncation_rate_increase": MAXIMUM_TRUNCATION_RATE_INCREASE,
        "signed_malom_mass_effects": signed_malom,
        "maximum_opposite_malom_mass_effect": (
            MAXIMUM_OPPOSITE_MALOM_MASS_EFFECT
        ),
        "gates": {
            "persistent_policy_divergence": policy_pass,
            "persistent_outcome_direction": persistence_pass,
            "phase_safety": phase_safety_pass,
            "truncation_safety": truncation_safety_pass,
            "malom_safety": malom_safety_pass,
        },
        "supported": supported,
        "selected_long_run_condition": selected_condition,
        "claim_boundary": (
            "development mechanism evidence only; no held-out strength, "
            "promotion, publication, or long-training launch authority"
        ),
    }


__all__ = [
    "CANDIDATE_COLORS",
    "EXPECTED_CONDITIONS",
    "MAX_POST_START_LOGICAL_PLIES",
    "MEASUREMENT_ROW_SCHEMA",
    "OUTCOME_BOUNDARIES",
    "PRIMARY_TEMPERATURE",
    "RESULT_SCHEMA",
    "ScheduleIsolationResultError",
    "build_outcome_measurement_schedule",
    "decide_schedule_isolation_result",
    "validate_and_summarize_outcome_rows",
]
