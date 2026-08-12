"""Pure decision rules for the mature target-refresh diagnostic."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from learned_ai.evaluation.common_anchor_policy_distribution import (
    DEFAULT_DIVERGENCE_THRESHOLDS,
)
from learned_ai.evaluation.target_refresh_equal_transition_result import (
    _material_triggers,
    _near_identical,
    _persistent_triggers,
    _summary_observation,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


LEDGER_SCHEMA = "nmm.target-refresh-mature-direct-crossplay-game.v1"
RESULT_SCHEMA = "nmm.target-refresh-mature-fork-result.v1"
EXPECTED_SEEDS = (67, 68, 69)
EXPECTED_CONDITIONS = ("refresh-mature", "stale-control")
EXPECTED_RECORD_INDICES = tuple(range(1, 13))
EXPECTED_REPLICATES = 4
EXPECTED_GAMES = 288
POLICY_BOUNDARIES = (4096, 8192)


class MatureTargetRefreshResultError(RuntimeError):
    """Raised when mature target-refresh result evidence is incomplete."""


def _contract_seeds(contract: Mapping[str, Any]) -> tuple[int, ...]:
    sources = contract.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise MatureTargetRefreshResultError("direct cross-play seed cells differ")
    seeds: list[int] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise MatureTargetRefreshResultError("direct cross-play seed cells differ")
        seed = source.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise MatureTargetRefreshResultError("direct cross-play seed cells differ")
        seeds.append(seed)
    if len(set(seeds)) != 3:
        raise MatureTargetRefreshResultError("direct cross-play seed cells differ")
    return tuple(seeds)


def _policy_seeds(
    by_seed_boundary: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[int, ...]:
    if len(by_seed_boundary) != 3:
        raise MatureTargetRefreshResultError("policy seed cells differ")
    try:
        seeds = tuple(sorted(int(seed) for seed in by_seed_boundary))
    except (TypeError, ValueError) as exc:
        raise MatureTargetRefreshResultError("policy seed cells differ") from exc
    if any(str(seed) not in by_seed_boundary or seed < 0 for seed in seeds):
        raise MatureTargetRefreshResultError("policy seed cells differ")
    return seeds


def _derived_seed(plan_identity: str, payload: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes({"plan_identity": plan_identity, **payload})
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_direct_crossplay_schedule(
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the frozen paired colour-swap schedule without touching models."""
    plan_identity = str(contract.get("plan_identity", ""))
    measurement = contract.get("measurement_contract", {}).get("direct_crossplay", {})
    if (
        len(plan_identity) != 64
        or measurement.get("record_indices") != list(EXPECTED_RECORD_INDICES)
        or measurement.get("replicates_per_start") != EXPECTED_REPLICATES
        or measurement.get("colour_swap") is not True
        or measurement.get("common_random_streams_by_colour") is not True
        or measurement.get("expected_games") != EXPECTED_GAMES
        or measurement.get("expected_pairs") != EXPECTED_GAMES // 2
    ):
        raise MatureTargetRefreshResultError(
            "direct cross-play measurement contract differs"
        )
    rows: list[dict[str, Any]] = []
    pair_index = 0
    ordinal = 0
    for seed in _contract_seeds(contract):
        for record_index in EXPECTED_RECORD_INDICES:
            for replicate in range(EXPECTED_REPLICATES):
                pair_core = {
                    "seed": seed,
                    "record_index": record_index,
                    "replicate": replicate,
                }
                pair_identity = canonical_sha256(
                    {"plan_identity": plan_identity, **pair_core}
                )
                colour_seeds = {
                    colour: _derived_seed(
                        plan_identity,
                        {**pair_core, "stream": f"policy-{colour}"},
                    )
                    for colour in ("W", "B")
                }
                for game_in_pair in range(2):
                    refresh_colour = "W" if game_in_pair == 0 else "B"
                    row = {
                        "ordinal": ordinal,
                        "pair_index": pair_index,
                        "pair_identity": pair_identity,
                        "game_in_pair": game_in_pair,
                        **pair_core,
                        "refresh_mature_colour": refresh_colour,
                        "stale_control_colour": ("B" if refresh_colour == "W" else "W"),
                        "policy_seed_white": colour_seeds["W"],
                        "policy_seed_black": colour_seeds["B"],
                        "referee_seed": _derived_seed(
                            plan_identity,
                            {**pair_core, "game_in_pair": game_in_pair},
                        ),
                    }
                    row["game_identity"] = canonical_sha256(
                        {"plan_identity": plan_identity, **row}
                    )
                    rows.append(row)
                    ordinal += 1
                pair_index += 1
    if len(rows) != EXPECTED_GAMES:
        raise MatureTargetRefreshResultError("direct cross-play schedule size differs")
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["outcome_class"]) for row in rows)
    games = len(rows)
    return {
        "games": games,
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "score_rate": (
            (counts["win"] + 0.5 * counts["draw"]) / games if games else None
        ),
        "max_ply_truncations": sum(
            row["termination_reason"] == "max-ply-truncation" for row in rows
        ),
    }


def summarize_direct_crossplay(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the complete ledger and apply the frozen direct-effect gate."""
    expected = build_direct_crossplay_schedule(contract)
    if len(rows) != len(expected):
        raise MatureTargetRefreshResultError("direct cross-play ledger size differs")
    plan_identity = str(contract["plan_identity"])
    measurement = contract["measurement_contract"]["direct_crossplay"]
    required = {
        "schema_version",
        "plan_identity",
        "ordinal",
        "pair_index",
        "pair_identity",
        "game_in_pair",
        "game_identity",
        "seed",
        "record_index",
        "replicate",
        "phase",
        "refresh_mature_colour",
        "stale_control_colour",
        "policy_seed_white",
        "policy_seed_black",
        "referee_seed",
        "refresh_mature_score",
        "outcome_class",
        "winner",
        "termination_reason",
        "post_start_logical_plies",
        "start_history_sha256",
        "end_history_sha256",
        "moves",
    }
    for observed, scheduled in zip(rows, expected, strict=True):
        if not isinstance(observed, Mapping) or set(observed) != required:
            raise MatureTargetRefreshResultError(
                "direct cross-play ledger fields differ"
            )
        if (
            observed["schema_version"] != LEDGER_SCHEMA
            or observed["plan_identity"] != plan_identity
        ):
            raise MatureTargetRefreshResultError(
                "direct cross-play ledger identity differs"
            )
        for field, expected_value in scheduled.items():
            if observed[field] != expected_value:
                raise MatureTargetRefreshResultError(
                    f"direct cross-play schedule differs at {scheduled['ordinal']}"
                )
        score = observed["refresh_mature_score"]
        if score not in {0.0, 0.5, 1.0}:
            raise MatureTargetRefreshResultError("direct cross-play score differs")
        expected_class = {0.0: "loss", 0.5: "draw", 1.0: "win"}[score]
        if observed["outcome_class"] != expected_class:
            raise MatureTargetRefreshResultError(
                "direct cross-play outcome class differs"
            )
        if observed["phase"] not in {"placement", "movement", "flying"}:
            raise MatureTargetRefreshResultError("direct cross-play phase differs")
        plies = observed["post_start_logical_plies"]
        if (
            isinstance(plies, bool)
            or not isinstance(plies, int)
            or not 1 <= plies <= measurement["max_post_start_logical_plies"]
            or not isinstance(observed["moves"], list)
            or len(observed["moves"]) != plies
        ):
            raise MatureTargetRefreshResultError("direct cross-play move count differs")
        winner = observed["winner"]
        reason = observed["termination_reason"]
        if winner not in {None, "white", "black"} or not isinstance(reason, str):
            raise MatureTargetRefreshResultError(
                "direct cross-play terminal state differs"
            )
        if reason == "max-ply-truncation":
            if (
                winner is not None
                or plies != measurement["max_post_start_logical_plies"]
                or score != 0.5
            ):
                raise MatureTargetRefreshResultError(
                    "max-ply truncation semantics differ"
                )
        elif winner is None:
            if score != 0.5:
                raise MatureTargetRefreshResultError("rules-draw score differs")
        else:
            refresh_name = (
                "white" if observed["refresh_mature_colour"] == "W" else "black"
            )
            if score != (1.0 if winner == refresh_name else 0.0):
                raise MatureTargetRefreshResultError("winner and refresh score differ")

    pairs: list[dict[str, Any]] = []
    for index in range(0, len(rows), 2):
        first, second = rows[index : index + 2]
        if (
            first["pair_identity"] != second["pair_identity"]
            or first["game_in_pair"] != 0
            or second["game_in_pair"] != 1
        ):
            raise MatureTargetRefreshResultError(
                "direct cross-play pair ordering differs"
            )
        pairs.append(
            {
                "pair_identity": first["pair_identity"],
                "seed": first["seed"],
                "record_index": first["record_index"],
                "replicate": first["replicate"],
                "phase": first["phase"],
                "refresh_mature_minus_stale_pair_score": (
                    float(first["refresh_mature_score"])
                    + float(second["refresh_mature_score"])
                    - 1.0
                ),
            }
        )
    effect = sum(item["refresh_mature_minus_stale_pair_score"] for item in pairs) / len(
        pairs
    )
    seeds = _contract_seeds(contract)
    seed_effects = {
        str(seed): sum(
            item["refresh_mature_minus_stale_pair_score"]
            for item in pairs
            if item["seed"] == seed
        )
        / sum(item["seed"] == seed for item in pairs)
        for seed in seeds
    }
    phase_effects = {
        phase: sum(
            item["refresh_mature_minus_stale_pair_score"]
            for item in pairs
            if item["phase"] == phase
        )
        / sum(item["phase"] == phase for item in pairs)
        for phase in ("placement", "movement", "flying")
    }
    truncations = sum(row["termination_reason"] == "max-ply-truncation" for row in rows)
    truncation_rate = truncations / len(rows)
    thresholds = contract["measurement_contract"]["direct_effect_thresholds"]
    aggregate_gate = float(thresholds["minimum_aggregate_pair_score_effect"])
    seed_gate = float(thresholds["minimum_per_seed_pair_score_effect"])
    opposite_gate = float(thresholds["maximum_opposite_seed_effect"])
    supporting_mature = [
        seed for seed, value in seed_effects.items() if value >= seed_gate
    ]
    supporting_stale = [
        seed for seed, value in seed_effects.items() if value <= -seed_gate
    ]
    truncation_safe = truncation_rate <= float(thresholds["maximum_truncation_rate"])
    mature_supported = (
        effect >= aggregate_gate
        and len(supporting_mature) >= thresholds["minimum_supporting_seeds"]
        and min(seed_effects.values()) >= -opposite_gate
        and truncation_safe
    )
    stale_supported = (
        effect <= -aggregate_gate
        and len(supporting_stale) >= thresholds["minimum_supporting_seeds"]
        and max(seed_effects.values()) <= opposite_gate
        and truncation_safe
    )
    if mature_supported and stale_supported:
        raise MatureTargetRefreshResultError(
            "direct cross-play classifier is contradictory"
        )
    if not truncation_safe:
        classification = "inconclusive_policy_or_truncation"
    elif mature_supported:
        classification = "material_mature_refresh_direct_effect"
    elif stale_supported:
        classification = "material_stale_target_direct_effect"
    else:
        classification = "no_material_direct_effect"
    report = {
        "games": len(rows),
        "pairs": len(pairs),
        "overall_refresh_mature": _aggregate(rows),
        "by_seed": {
            str(seed): _aggregate([row for row in rows if row["seed"] == seed])
            for seed in seeds
        },
        "by_phase": {
            phase: _aggregate([row for row in rows if row["phase"] == phase])
            for phase in ("placement", "movement", "flying")
        },
        "by_refresh_colour": {
            colour: _aggregate(
                [row for row in rows if row["refresh_mature_colour"] == colour]
            )
            for colour in ("W", "B")
        },
        "paired": {
            "contrast": "refresh-mature minus stale-control",
            "mean_score_effect": effect,
            "seed_effects": seed_effects,
            "phase_effects": phase_effects,
            "differences": dict(
                sorted(
                    Counter(
                        str(item["refresh_mature_minus_stale_pair_score"])
                        for item in pairs
                    ).items()
                )
            ),
        },
        "decision": {
            "classification": classification,
            "supported": mature_supported or stale_supported,
            "development_preference": (
                "refresh-mature"
                if mature_supported
                else "stale-control"
                if stale_supported
                else None
            ),
            "supporting_mature_seeds": supporting_mature,
            "supporting_stale_seeds": supporting_stale,
            "truncation_rate": truncation_rate,
            "truncation_safe": truncation_safe,
            "thresholds": thresholds,
        },
    }
    return {**report, "summary_identity": canonical_sha256(report)}


def classify_mature_policy_divergence(
    by_seed_boundary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    thresholds: Mapping[str, float] = DEFAULT_DIVERGENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Apply the frozen 4,096-to-8,192 persistence gate to three seeds."""
    seeds = _policy_seeds(by_seed_boundary)
    if set(thresholds) != set(DEFAULT_DIVERGENCE_THRESHOLDS):
        raise MatureTargetRefreshResultError("policy thresholds differ")
    seed_audits: dict[str, Any] = {}
    for seed in seeds:
        boundaries = by_seed_boundary[str(seed)]
        if set(boundaries) != {str(value) for value in POLICY_BOUNDARIES}:
            raise MatureTargetRefreshResultError(
                f"policy boundaries differ for seed {seed}"
            )
        observations = {
            boundary: _summary_observation(boundaries[str(boundary)])
            for boundary in POLICY_BOUNDARIES
        }
        persistent = _persistent_triggers(
            observations[4096], observations[8192], thresholds
        )
        seed_audits[str(seed)] = {
            "by_transition_boundary": {
                str(boundary): {
                    "observed": observations[boundary],
                    "near_identical": _near_identical(
                        observations[boundary], thresholds
                    ),
                    "material_triggers": sorted(
                        _material_triggers(observations[boundary], thresholds)
                    ),
                }
                for boundary in POLICY_BOUNDARIES
            },
            "persistent_material_triggers_4096_to_8192": persistent,
            "materially_diverged_with_persistence": bool(persistent),
        }
    supporting = [
        seed
        for seed, audit in seed_audits.items()
        if audit["materially_diverged_with_persistence"]
    ]
    report = {
        "comparison_orientation": (
            "stale-control minus refresh-mature for signed Malom mass; "
            "distance metrics are symmetric"
        ),
        "boundaries": list(POLICY_BOUNDARIES),
        "by_seed": seed_audits,
        "supporting_seeds": supporting,
        "materially_diverged_with_persistence": len(supporting) >= 2,
        "thresholds": dict(thresholds),
    }
    return {**report, "decision_identity": canonical_sha256(report)}


def decide_mature_result(
    *,
    policy_decision: Mapping[str, Any],
    direct_crossplay: Mapping[str, Any],
) -> dict[str, Any]:
    """Join policy and outcome evidence without promoting either condition."""
    classification = str(direct_crossplay["decision"]["classification"])
    persistent = bool(policy_decision.get("materially_diverged_with_persistence"))
    preference = direct_crossplay["decision"]["development_preference"]
    return {
        "classification": classification,
        "development_preference": preference,
        "direct_effect_supported": direct_crossplay["decision"]["supported"],
        "policy_diverged_with_persistence": persistent,
        "mechanism_evidence_supported": bool(preference and persistent),
        "automatic_long_run_selection": False,
        "held_out_strength_claim": False,
        "interpretation": (
            "The paired no-update outcome is the direct-effect decision. "
            "Persistent policy divergence is supporting mechanism evidence "
            "only and cannot promote or launch a model."
        ),
    }


def classify_mature_replication(
    *,
    prior_direct_crossplay: Mapping[str, Any],
    replication_direct_crossplay: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Join two disjoint three-seed cohorts under a frozen replication gate."""
    required_thresholds = {
        "minimum_replication_aggregate_pair_score_effect",
        "minimum_pooled_pair_score_effect",
        "minimum_per_seed_pair_score_effect",
        "minimum_replication_supporting_seeds",
        "minimum_pooled_supporting_seeds",
        "maximum_pooled_opposite_seeds",
        "maximum_pooled_truncation_rate",
    }
    if set(thresholds) != required_thresholds:
        raise MatureTargetRefreshResultError("replication thresholds differ")

    def cohort(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        paired = value.get("paired")
        decision = value.get("decision")
        if not isinstance(paired, Mapping) or not isinstance(decision, Mapping):
            raise MatureTargetRefreshResultError(f"{label} direct summary differs")
        raw_effects = paired.get("seed_effects")
        if not isinstance(raw_effects, Mapping) or len(raw_effects) != 3:
            raise MatureTargetRefreshResultError(f"{label} direct seed effects differ")
        effects: dict[str, float] = {}
        for seed, effect in raw_effects.items():
            try:
                numeric_seed = int(seed)
                numeric_effect = float(effect)
            except (TypeError, ValueError) as exc:
                raise MatureTargetRefreshResultError(
                    f"{label} direct seed effects differ"
                ) from exc
            if str(numeric_seed) != str(seed) or numeric_seed < 0:
                raise MatureTargetRefreshResultError(
                    f"{label} direct seed effects differ"
                )
            if not math.isfinite(numeric_effect) or not -1.0 <= numeric_effect <= 1.0:
                raise MatureTargetRefreshResultError(
                    f"{label} direct seed effects differ"
                )
            effects[str(numeric_seed)] = numeric_effect
        games = value.get("games")
        pairs = value.get("pairs")
        effect = paired.get("mean_score_effect")
        truncation_rate = decision.get("truncation_rate")
        if (
            games != 288
            or pairs != 144
            or isinstance(effect, bool)
            or not isinstance(effect, (int, float))
            or not math.isfinite(float(effect))
            or not -1.0 <= float(effect) <= 1.0
            or isinstance(truncation_rate, bool)
            or not isinstance(truncation_rate, (int, float))
            or not math.isfinite(float(truncation_rate))
            or not 0.0 <= float(truncation_rate) <= 1.0
        ):
            raise MatureTargetRefreshResultError(f"{label} direct summary differs")
        return {
            "classification": str(decision.get("classification")),
            "effect": float(effect),
            "seed_effects": effects,
            "games": games,
            "pairs": pairs,
            "truncation_rate": float(truncation_rate),
        }

    prior = cohort(prior_direct_crossplay, label="prior cohort")
    replication = cohort(
        replication_direct_crossplay,
        label="replication cohort",
    )
    if set(prior["seed_effects"]) & set(replication["seed_effects"]):
        raise MatureTargetRefreshResultError("replication seed cohorts overlap")

    material_classes = {
        "material_mature_refresh_direct_effect": (1, "refresh-mature"),
        "material_stale_target_direct_effect": (-1, "stale-control"),
    }
    direction_record = material_classes.get(replication["classification"])
    all_effects = {**prior["seed_effects"], **replication["seed_effects"]}
    pooled_pairs = prior["pairs"] + replication["pairs"]
    pooled_games = prior["games"] + replication["games"]
    pooled_effect = (
        prior["effect"] * prior["pairs"] + replication["effect"] * replication["pairs"]
    ) / pooled_pairs
    pooled_truncation_rate = (
        prior["truncation_rate"] * prior["games"]
        + replication["truncation_rate"] * replication["games"]
    ) / pooled_games
    seed_gate = float(thresholds["minimum_per_seed_pair_score_effect"])
    supporting: list[str] = []
    opposite: list[str] = []
    selected: str | None = None
    if direction_record is not None:
        direction, candidate = direction_record
        supporting = sorted(
            [
                seed
                for seed, effect in all_effects.items()
                if direction * effect >= seed_gate
            ],
            key=int,
        )
        opposite = sorted(
            [
                seed
                for seed, effect in all_effects.items()
                if direction * effect <= -seed_gate
            ],
            key=int,
        )
        replication_support = sum(
            direction * effect >= seed_gate
            for effect in replication["seed_effects"].values()
        )
        replication_passed = direction * replication["effect"] >= float(
            thresholds["minimum_replication_aggregate_pair_score_effect"]
        ) and replication_support >= int(
            thresholds["minimum_replication_supporting_seeds"]
        )
        pooled_passed = (
            direction * pooled_effect
            >= float(thresholds["minimum_pooled_pair_score_effect"])
            and len(supporting) >= int(thresholds["minimum_pooled_supporting_seeds"])
            and len(opposite) <= int(thresholds["maximum_pooled_opposite_seeds"])
            and pooled_truncation_rate
            <= float(thresholds["maximum_pooled_truncation_rate"])
        )
        if replication_passed and pooled_passed:
            selected = candidate

    if pooled_truncation_rate > float(thresholds["maximum_pooled_truncation_rate"]):
        classification = "inconclusive_replication_truncation"
    elif selected == "refresh-mature":
        classification = "replicated_material_mature_refresh_effect"
    elif selected == "stale-control":
        classification = "replicated_material_stale_target_effect"
    else:
        classification = "no_replicated_material_effect"
    report = {
        "classification": classification,
        "selected_successor_condition": selected,
        "cohort_seed_sets": {
            "prior": sorted(prior["seed_effects"], key=int),
            "replication": sorted(replication["seed_effects"], key=int),
        },
        "cohort_mean_score_effects": {
            "prior": prior["effect"],
            "replication": replication["effect"],
        },
        "pooled_mean_score_effect": pooled_effect,
        "pooled_truncation_rate": pooled_truncation_rate,
        "pooled_seed_effects": all_effects,
        "supporting_seeds": supporting,
        "opposite_seeds": opposite,
        "thresholds": dict(thresholds),
        "automatic_long_run_selection": False,
        "selection_scope": (
            "target-cadence input for a separately frozen retained plan only"
        ),
    }
    return {**report, "decision_identity": canonical_sha256(report)}


__all__ = [
    "EXPECTED_CONDITIONS",
    "EXPECTED_GAMES",
    "EXPECTED_RECORD_INDICES",
    "EXPECTED_REPLICATES",
    "EXPECTED_SEEDS",
    "LEDGER_SCHEMA",
    "MatureTargetRefreshResultError",
    "RESULT_SCHEMA",
    "build_direct_crossplay_schedule",
    "classify_mature_replication",
    "classify_mature_policy_divergence",
    "decide_mature_result",
    "summarize_direct_crossplay",
]
