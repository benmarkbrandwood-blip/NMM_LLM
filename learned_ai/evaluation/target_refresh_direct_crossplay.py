"""Frozen, no-update direct cross-play for target-refresh candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


PLAN_SCHEMA = "nmm.target-refresh-direct-crossplay-plan.v1"
READINESS_SCHEMA = "nmm.target-refresh-direct-crossplay-readiness.v1"
AUTHORIZATION_SCHEMA = "nmm.target-refresh-direct-crossplay-authorization.v1"
LEDGER_SCHEMA = "nmm.target-refresh-direct-crossplay-game.v1"
RESULT_SCHEMA = "nmm.target-refresh-direct-crossplay-result.v1"

_PLAN_FIELDS = {
    "schema_version",
    "objective",
    "plan_identity",
    "source",
    "implementation",
    "data_contract",
    "checkpoint_contract",
    "measurement_contract",
    "decision_contract",
    "resource_envelope",
    "output_contract",
    "claim_boundary",
    "stop_conditions",
    "prohibited_operations",
}
_SOURCE_FIELDS = {
    "schedule_contract_path",
    "schedule_contract_sha256",
    "schedule_plan_identity",
    "source_result_path",
    "source_result_sha256",
    "source_result_identity",
    "training_source_commit",
    "analysis_source_commit",
}
_IMPLEMENTATION_FIELDS = {
    "commit",
    "module_path",
    "module_sha256",
    "prepare_path",
    "prepare_sha256",
    "runner_path",
    "runner_sha256",
}
_DATA_FIELDS = {
    "human_db_identity",
    "human_db_malom_policy",
    "malom_manifest_identity",
    "policy_corpus_path",
    "policy_corpus_sha256",
    "replay_corpus_path",
    "replay_corpus_sha256",
    "replay_corpus_identity",
    "replay_audit_path",
    "replay_audit_sha256",
    "replay_audit_identity",
}
_CHECKPOINT_FIELDS = {
    "post_fork_consumed_transitions",
    "anchors",
    "candidates",
}
_MEASUREMENT_FIELDS = {
    "device",
    "policy_selection",
    "temperature",
    "shared_game50_anchor_features",
    "sanmill_role",
    "record_indices",
    "replicates_per_start",
    "colour_swap",
    "common_random_streams_by_colour",
    "max_post_start_logical_plies",
    "max_ply_disposition",
    "conditions",
    "seeds",
    "expected_pairs",
    "expected_games",
}
_DECISION_FIELDS = {
    "contrast",
    "minimum_aggregate_pair_score_effect",
    "minimum_per_seed_pair_score_effect",
    "minimum_supporting_seeds",
    "maximum_opposite_seed_effect",
    "maximum_truncation_rate",
    "result_classes",
    "automatic_long_run_selection",
}
_RESOURCE_FIELDS = {
    "training_games",
    "optimizer_updates",
    "database_writes",
    "checkpoint_writes",
    "no_update_games",
    "maximum_active_wall_hours",
}
_OUTPUT_FIELDS = {
    "readiness",
    "authorization",
    "launch",
    "ledger",
    "result",
    "completion",
    "failure",
}
_ANCHOR_FIELDS = {
    "seed",
    "path",
    "file_sha256",
    "checkpoint_id",
    "payload_sha256",
    "model_state_sha256",
    "game_count",
    "update_count",
    "optimizer_consumed_transition_count",
    "pending_transition_count",
}
_CANDIDATE_FIELDS = {
    "seed",
    "condition",
    "path",
    "file_sha256",
    "checkpoint_id",
    "model_state_sha256",
    "game_count",
    "update_count",
    "optimizer_consumed_transition_count",
    "post_fork_consumed_transition_count",
    "pending_transition_count",
    "fork_checkpoint_id",
    "immutable_asset_identities",
}


class DirectCrossplayError(RuntimeError):
    """Raised when direct cross-play evidence is not contract-valid."""


def _require_closed(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DirectCrossplayError(f"{label} fields are unknown or incomplete")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DirectCrossplayError(f"{label} is not a SHA-256 value")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DirectCrossplayError(f"{label} is not a SHA-256 value") from exc
    return value


def _identity_body(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: plan[key] for key in sorted(plan) if key != "plan_identity"}


def validate_direct_crossplay_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed plan and return a detached ordinary dictionary."""
    plan = _require_closed(plan, _PLAN_FIELDS, "plan")
    if plan["schema_version"] != PLAN_SCHEMA:
        raise DirectCrossplayError("unsupported direct-crossplay plan schema")
    if not isinstance(plan["objective"], str) or not plan["objective"]:
        raise DirectCrossplayError("plan objective is absent")

    source = _require_closed(plan["source"], _SOURCE_FIELDS, "source")
    implementation = _require_closed(
        plan["implementation"], _IMPLEMENTATION_FIELDS, "implementation"
    )
    data = _require_closed(plan["data_contract"], _DATA_FIELDS, "data contract")
    checkpoints = _require_closed(
        plan["checkpoint_contract"], _CHECKPOINT_FIELDS, "checkpoint contract"
    )
    measurement = _require_closed(
        plan["measurement_contract"], _MEASUREMENT_FIELDS, "measurement contract"
    )
    decision = _require_closed(
        plan["decision_contract"], _DECISION_FIELDS, "decision contract"
    )
    resources = _require_closed(
        plan["resource_envelope"], _RESOURCE_FIELDS, "resource envelope"
    )
    outputs = _require_closed(
        plan["output_contract"], _OUTPUT_FIELDS, "output contract"
    )

    for field in (
        "schedule_contract_sha256",
        "schedule_plan_identity",
        "source_result_sha256",
        "source_result_identity",
    ):
        _require_sha256(source[field], f"source.{field}")
    for field in ("training_source_commit", "analysis_source_commit"):
        value = source[field]
        if not isinstance(value, str) or len(value) != 40:
            raise DirectCrossplayError(f"source.{field} is not a Git commit")
    for field in ("module_sha256", "prepare_sha256", "runner_sha256"):
        _require_sha256(implementation[field], f"implementation.{field}")
    if not isinstance(implementation["commit"], str) or len(
        implementation["commit"]
    ) != 40:
        raise DirectCrossplayError("implementation.commit is not a Git commit")
    for field in (
        "human_db_identity",
        "malom_manifest_identity",
        "policy_corpus_sha256",
        "replay_corpus_sha256",
        "replay_corpus_identity",
        "replay_audit_sha256",
        "replay_audit_identity",
    ):
        _require_sha256(data[field], f"data_contract.{field}")
    if data["human_db_malom_policy"] != "masked_historical_labels":
        raise DirectCrossplayError("historical HumanDB Malom labels must be masked")

    seeds = measurement["seeds"]
    record_indices = measurement["record_indices"]
    conditions = measurement["conditions"]
    if seeds != [67, 68, 69]:
        raise DirectCrossplayError("direct cross-play seeds differ")
    if record_indices != list(range(1, 13)):
        raise DirectCrossplayError("direct cross-play replay records differ")
    if conditions != ["refresh-once", "no-refresh"]:
        raise DirectCrossplayError("direct cross-play conditions differ")
    expected_pairs = len(seeds) * len(record_indices) * int(
        measurement["replicates_per_start"]
    )
    if (
        measurement["device"] != "cpu"
        or measurement["policy_selection"] != "training-policy-sampling"
        or float(measurement["temperature"]) != 0.2
        or measurement["shared_game50_anchor_features"] is not True
        or measurement["sanmill_role"] != "strict-portable-referee-only"
        or int(measurement["replicates_per_start"]) != 4
        or measurement["colour_swap"] is not True
        or measurement["common_random_streams_by_colour"] is not True
        or int(measurement["max_post_start_logical_plies"]) != 120
        or measurement["max_ply_disposition"] != "development-draw-with-flag"
        or int(measurement["expected_pairs"]) != expected_pairs
        or int(measurement["expected_games"]) != expected_pairs * 2
    ):
        raise DirectCrossplayError("direct cross-play measurement contract differs")

    if int(checkpoints["post_fork_consumed_transitions"]) != 8192:
        raise DirectCrossplayError("candidate transition boundary differs")
    anchors = checkpoints["anchors"]
    candidates = checkpoints["candidates"]
    if not isinstance(anchors, list) or not isinstance(candidates, list):
        raise DirectCrossplayError("checkpoint records must be lists")
    if [item.get("seed") for item in anchors] != seeds:
        raise DirectCrossplayError("anchor checkpoint seeds differ")
    expected_cells = [(seed, condition) for seed in seeds for condition in conditions]
    observed_cells = [(item.get("seed"), item.get("condition")) for item in candidates]
    if observed_cells != expected_cells:
        raise DirectCrossplayError("candidate checkpoint cells differ")
    for label, records, fields in (
        ("anchor", anchors, _ANCHOR_FIELDS),
        ("candidate", candidates, _CANDIDATE_FIELDS),
    ):
        for record in records:
            record = _require_closed(record, fields, f"{label} checkpoint")
            _require_sha256(record.get("file_sha256"), f"{label} checkpoint file")
            _require_sha256(
                record.get("model_state_sha256"),
                f"{label} checkpoint model state",
            )
            if label == "anchor":
                _require_sha256(
                    record.get("payload_sha256"),
                    "anchor checkpoint payload",
                )
            if not isinstance(record.get("checkpoint_id"), str) or not record[
                "checkpoint_id"
            ]:
                raise DirectCrossplayError(f"{label} checkpoint ID is absent")
            path = record.get("path")
            if not isinstance(path, str) or Path(path).is_absolute():
                raise DirectCrossplayError(f"{label} checkpoint path is not relative")
    if any(
        int(record["post_fork_consumed_transition_count"]) != 8192
        for record in candidates
    ):
        raise DirectCrossplayError("candidate checkpoint boundary differs")

    output_paths: list[str] = []
    for name, value in outputs.items():
        if not isinstance(value, str) or not value:
            raise DirectCrossplayError(f"output_contract.{name} is absent")
        path = Path(value)
        if path.is_absolute() or not path.as_posix().startswith("out/"):
            raise DirectCrossplayError(
                f"output_contract.{name} is not a repository-local output"
            )
        output_paths.append(path.as_posix())
    if len(set(output_paths)) != len(output_paths):
        raise DirectCrossplayError("direct cross-play outputs are not unique")

    if (
        resources["training_games"] != 0
        or resources["optimizer_updates"] != 0
        or resources["database_writes"] != 0
        or resources["checkpoint_writes"] != 0
        or resources["no_update_games"] != expected_pairs * 2
        or not 0 < float(resources["maximum_active_wall_hours"]) <= 2.0
    ):
        raise DirectCrossplayError("direct cross-play resource envelope differs")
    if decision["contrast"] != "no-refresh minus refresh-once":
        raise DirectCrossplayError("direct cross-play contrast differs")
    if decision["automatic_long_run_selection"] is not False:
        raise DirectCrossplayError("direct cross-play cannot select a long run")
    for field in (
        "minimum_aggregate_pair_score_effect",
        "minimum_per_seed_pair_score_effect",
        "maximum_opposite_seed_effect",
        "maximum_truncation_rate",
    ):
        value = decision[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DirectCrossplayError(f"decision.{field} is not numeric")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise DirectCrossplayError(f"decision.{field} is invalid")
    if int(decision["minimum_supporting_seeds"]) != 2:
        raise DirectCrossplayError("minimum supporting seed count differs")
    if not isinstance(plan["claim_boundary"], str) or not plan["claim_boundary"]:
        raise DirectCrossplayError("claim boundary is absent")
    if not isinstance(plan["stop_conditions"], list) or not plan["stop_conditions"]:
        raise DirectCrossplayError("stop conditions are absent")
    if not isinstance(plan["prohibited_operations"], list) or not plan[
        "prohibited_operations"
    ]:
        raise DirectCrossplayError("prohibited operations are absent")

    detached = json.loads(json.dumps(plan))
    expected_identity = canonical_sha256(_identity_body(detached))
    if detached["plan_identity"] != expected_identity:
        raise DirectCrossplayError("direct cross-play plan identity differs")
    return detached


def load_direct_crossplay_plan(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DirectCrossplayError("cannot load direct cross-play plan") from exc
    return validate_direct_crossplay_plan(value)


def _derived_seed(plan_identity: str, payload: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes({"plan_identity": plan_identity, **payload})
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_direct_crossplay_schedule(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = validate_direct_crossplay_plan(plan)
    measurement = plan["measurement_contract"]
    rows: list[dict[str, Any]] = []
    pair_index = 0
    ordinal = 0
    for seed in measurement["seeds"]:
        for record_index in measurement["record_indices"]:
            for replicate in range(measurement["replicates_per_start"]):
                pair_core = {
                    "seed": seed,
                    "record_index": record_index,
                    "replicate": replicate,
                }
                pair_identity = canonical_sha256(
                    {"plan_identity": plan["plan_identity"], **pair_core}
                )
                colour_seeds = {
                    colour: _derived_seed(
                        plan["plan_identity"],
                        {**pair_core, "stream": f"policy-{colour}"},
                    )
                    for colour in ("W", "B")
                }
                for game_in_pair in range(2):
                    no_refresh_colour = "W" if game_in_pair == 0 else "B"
                    row = {
                        "ordinal": ordinal,
                        "pair_index": pair_index,
                        "pair_identity": pair_identity,
                        "game_in_pair": game_in_pair,
                        **pair_core,
                        "no_refresh_colour": no_refresh_colour,
                        "refresh_once_colour": (
                            "B" if no_refresh_colour == "W" else "W"
                        ),
                        "policy_seed_white": colour_seeds["W"],
                        "policy_seed_black": colour_seeds["B"],
                        "referee_seed": _derived_seed(
                            plan["plan_identity"],
                            {**pair_core, "game_in_pair": game_in_pair},
                        ),
                    }
                    row["game_identity"] = canonical_sha256(
                        {"plan_identity": plan["plan_identity"], **row}
                    )
                    rows.append(row)
                    ordinal += 1
                pair_index += 1
    if len(rows) != measurement["expected_games"]:
        raise DirectCrossplayError("direct cross-play schedule size differs")
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
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a complete ordered ledger and apply the frozen classifier."""
    plan = validate_direct_crossplay_plan(plan)
    measurement = plan["measurement_contract"]
    expected = build_direct_crossplay_schedule(plan)
    if len(rows) != len(expected):
        raise DirectCrossplayError("direct cross-play ledger size differs")
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
        "no_refresh_colour",
        "refresh_once_colour",
        "policy_seed_white",
        "policy_seed_black",
        "referee_seed",
        "no_refresh_score",
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
            raise DirectCrossplayError("direct cross-play ledger fields differ")
        if observed["schema_version"] != LEDGER_SCHEMA:
            raise DirectCrossplayError("direct cross-play ledger schema differs")
        if observed["plan_identity"] != plan["plan_identity"]:
            raise DirectCrossplayError("direct cross-play ledger plan differs")
        for field, value in scheduled.items():
            if observed[field] != value:
                raise DirectCrossplayError(
                    f"direct cross-play schedule differs at ordinal {scheduled['ordinal']}"
                )
        if observed["phase"] not in {"placement", "movement", "flying"}:
            raise DirectCrossplayError("direct cross-play phase differs")
        if observed["no_refresh_score"] not in {0.0, 0.5, 1.0}:
            raise DirectCrossplayError("direct cross-play score differs")
        expected_class = {
            0.0: "loss",
            0.5: "draw",
            1.0: "win",
        }[float(observed["no_refresh_score"])]
        if observed["outcome_class"] != expected_class:
            raise DirectCrossplayError("direct cross-play outcome class differs")
        _require_sha256(
            observed["start_history_sha256"], "ledger start history"
        )
        _require_sha256(observed["end_history_sha256"], "ledger end history")
        post_start = observed["post_start_logical_plies"]
        if (
            isinstance(post_start, bool)
            or not isinstance(post_start, int)
            or not 1
            <= post_start
            <= measurement["max_post_start_logical_plies"]
        ):
            raise DirectCrossplayError("direct cross-play ply count differs")
        if not isinstance(observed["moves"], list) or len(
            observed["moves"]
        ) != post_start:
            raise DirectCrossplayError("direct cross-play move history differs")
        for move in observed["moves"]:
            if not isinstance(move, Mapping) or set(move) != {
                "from",
                "to",
                "capture",
            }:
                raise DirectCrossplayError("direct cross-play move differs")
            if not isinstance(move["to"], str) or not move["to"]:
                raise DirectCrossplayError("direct cross-play move target differs")
            for field in ("from", "capture"):
                if move[field] is not None and not isinstance(move[field], str):
                    raise DirectCrossplayError(
                        f"direct cross-play move {field} differs"
                    )
        winner = observed["winner"]
        if winner not in {None, "white", "black"}:
            raise DirectCrossplayError("direct cross-play winner differs")
        reason = observed["termination_reason"]
        if not isinstance(reason, str) or not reason:
            raise DirectCrossplayError("direct cross-play termination differs")
        if reason == "max-ply-truncation":
            if (
                winner is not None
                or post_start != measurement["max_post_start_logical_plies"]
                or observed["no_refresh_score"] != 0.5
            ):
                raise DirectCrossplayError("max-ply truncation semantics differ")
        elif winner is None:
            if observed["no_refresh_score"] != 0.5:
                raise DirectCrossplayError("rules-draw score differs")
        else:
            no_refresh_name = (
                "white"
                if observed["no_refresh_colour"] == "W"
                else "black"
            )
            expected_score = 1.0 if winner == no_refresh_name else 0.0
            if observed["no_refresh_score"] != expected_score:
                raise DirectCrossplayError("winner and score differ")

    pairs: list[dict[str, Any]] = []
    for index in range(0, len(rows), 2):
        first, second = rows[index : index + 2]
        if (
            first["pair_identity"] != second["pair_identity"]
            or first["game_in_pair"] != 0
            or second["game_in_pair"] != 1
        ):
            raise DirectCrossplayError("direct cross-play pair ordering differs")
        difference = (
            float(first["no_refresh_score"])
            + float(second["no_refresh_score"])
            - 1.0
        )
        pairs.append(
            {
                "pair_identity": first["pair_identity"],
                "seed": first["seed"],
                "record_index": first["record_index"],
                "replicate": first["replicate"],
                "phase": first["phase"],
                "no_refresh_minus_refresh_pair_score": difference,
            }
        )

    pair_mean = sum(
        item["no_refresh_minus_refresh_pair_score"] for item in pairs
    ) / len(pairs)
    seed_effects = {
        str(seed): sum(
            item["no_refresh_minus_refresh_pair_score"]
            for item in pairs
            if item["seed"] == seed
        )
        / sum(item["seed"] == seed for item in pairs)
        for seed in plan["measurement_contract"]["seeds"]
    }
    phase_effects = {
        phase: sum(
            item["no_refresh_minus_refresh_pair_score"]
            for item in pairs
            if item["phase"] == phase
        )
        / sum(item["phase"] == phase for item in pairs)
        for phase in ("placement", "movement", "flying")
    }
    truncations = sum(
        row["termination_reason"] == "max-ply-truncation" for row in rows
    )
    truncation_rate = truncations / len(rows)
    decision = plan["decision_contract"]
    aggregate_gate = float(decision["minimum_aggregate_pair_score_effect"])
    seed_gate = float(decision["minimum_per_seed_pair_score_effect"])
    opposite_gate = float(decision["maximum_opposite_seed_effect"])
    supporting_no_refresh = [
        seed for seed, effect in seed_effects.items() if effect >= seed_gate
    ]
    supporting_refresh = [
        seed for seed, effect in seed_effects.items() if effect <= -seed_gate
    ]
    truncation_safe = truncation_rate <= float(
        decision["maximum_truncation_rate"]
    )
    no_refresh_supported = (
        pair_mean >= aggregate_gate
        and len(supporting_no_refresh) >= decision["minimum_supporting_seeds"]
        and min(seed_effects.values()) >= -opposite_gate
        and truncation_safe
    )
    refresh_supported = (
        pair_mean <= -aggregate_gate
        and len(supporting_refresh) >= decision["minimum_supporting_seeds"]
        and max(seed_effects.values()) <= opposite_gate
        and truncation_safe
    )
    if no_refresh_supported and refresh_supported:
        raise DirectCrossplayError("direct cross-play classifier is contradictory")
    if not truncation_safe:
        classification = "inconclusive_truncation"
    elif no_refresh_supported:
        classification = "material_no_refresh_direct_effect"
    elif refresh_supported:
        classification = "material_refresh_once_direct_effect"
    else:
        classification = "no_material_direct_effect"

    by_seed = {
        str(seed): _aggregate([row for row in rows if row["seed"] == seed])
        for seed in plan["measurement_contract"]["seeds"]
    }
    by_phase = {
        phase: _aggregate([row for row in rows if row["phase"] == phase])
        for phase in ("placement", "movement", "flying")
    }
    by_no_refresh_colour = {
        colour: _aggregate(
            [row for row in rows if row["no_refresh_colour"] == colour]
        )
        for colour in ("W", "B")
    }
    report = {
        "schema_version": RESULT_SCHEMA,
        "plan_identity": plan["plan_identity"],
        "scope": {
            "training_games": 0,
            "optimizer_updates": 0,
            "database_writes": 0,
            "checkpoint_writes": 0,
            "no_update_games": len(rows),
            "held_out_strength_claim": False,
            "automatic_long_run_selection": False,
        },
        "games": len(rows),
        "pairs": len(pairs),
        "overall_no_refresh": _aggregate(rows),
        "by_seed": by_seed,
        "by_phase": by_phase,
        "by_no_refresh_colour": by_no_refresh_colour,
        "paired": {
            "contrast": decision["contrast"],
            "mean_score_effect": pair_mean,
            "seed_effects": seed_effects,
            "phase_effects": phase_effects,
            "differences": dict(
                sorted(
                    Counter(
                        str(item["no_refresh_minus_refresh_pair_score"])
                        for item in pairs
                    ).items()
                )
            ),
        },
        "decision": {
            "classification": classification,
            "supported": no_refresh_supported or refresh_supported,
            "development_preference": (
                "no-refresh"
                if no_refresh_supported
                else "refresh-once"
                if refresh_supported
                else None
            ),
            "supporting_no_refresh_seeds": supporting_no_refresh,
            "supporting_refresh_once_seeds": supporting_refresh,
            "truncation_rate": truncation_rate,
            "truncation_safe": truncation_safe,
            "thresholds": decision,
            "claim_boundary": plan["claim_boundary"],
        },
    }
    report["result_identity"] = canonical_sha256(report)
    return report
