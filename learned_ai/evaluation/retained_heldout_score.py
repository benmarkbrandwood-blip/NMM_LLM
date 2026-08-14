"""Fixed-width held-out score comparison for the retained v3/v4 routes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from learned_ai.evaluation.retained_late_import_heldout_pool import (
    validate_retained_late_import_pool,
)
from learned_ai.evaluation.retained_passivity_diagnostic import _interval
from learned_ai.evaluation.retained_phase_process_generalization import (
    HORIZON_POST_START_LOGICAL_PLIES,
    MAX_POST_START_LOGICAL_PLIES,
    SANMILL_NODE_CEILING,
    RetainedPhaseProcessError,
    _candidate_summary,
    append_game_record,
    load_game_ledger as _load_phase_game_ledger,
    play_phase_process_game,
    replay_frozen_start,
)
from learned_ai.training.run_contract import canonical_sha256


PLAN_SCHEMA = "nmm.retained-heldout-score-plan.v1"
SPEC_SCHEMA = "nmm.retained-heldout-score-spec.v1"
GAME_SCHEMA = "nmm.retained-heldout-score-game.v1"
REPORT_SCHEMA = "nmm.retained-heldout-score-result.v1"
EXPECTED_CANDIDATES = ("retained-v3-refresh50", "retained-v4-no-refresh")
EXPECTED_STARTS = 253
EXPECTED_MATCHED_COLOUR_UNITS = EXPECTED_STARTS * 2
EXPECTED_GAMES = EXPECTED_STARTS * 4
MAX_PRIMARY_HALF_WIDTH = 0.015
EXPECTED_POOL_IDENTITY = (
    "2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7"
)
EXPECTED_POOL_RECORDS_IDENTITY = (
    "4e5f9ecf7508a995b74af6a36bcf966c89d9141940770ebb21c3629446830a31"
)
EXPECTED_PREFIX_RECORDS_IDENTITY = (
    "99951a691c106a86aa5e4affc16ced2b63866e2dd589379527d068b022003c7b"
)
EXPECTED_PHASE_COUNTS = {"flying": 56, "movement": 98, "placement": 99}


class RetainedHeldoutScoreError(RetainedPhaseProcessError):
    """Raised when the held-out score contract or evidence differs."""


def load_corpus_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the frozen pool and return its preregistered 253-start prefix."""
    records = validate_retained_late_import_pool(payload)
    if payload.get("pool_identity") != EXPECTED_POOL_IDENTITY:
        raise RetainedHeldoutScoreError("held-out source-pool identity differs")
    if payload.get("records_identity") != EXPECTED_POOL_RECORDS_IDENTITY:
        raise RetainedHeldoutScoreError("held-out source-pool records differ")
    profiles = payload.get("nested_precision_prefixes")
    if not isinstance(profiles, list):
        raise RetainedHeldoutScoreError("held-out precision profiles are absent")
    selected = next(
        (
            item
            for item in profiles
            if isinstance(item, Mapping)
            and item.get("target_starts") == EXPECTED_STARTS
        ),
        None,
    )
    if not isinstance(selected, Mapping) or (
        selected.get("available") is not True
        or selected.get("target_games") != EXPECTED_GAMES
        or selected.get("records_identity") != EXPECTED_PREFIX_RECORDS_IDENTITY
        or selected.get("phase_counts") != EXPECTED_PHASE_COUNTS
    ):
        raise RetainedHeldoutScoreError("held-out 253-start prefix differs")
    prefix = records[:EXPECTED_STARTS]
    if len(prefix) != EXPECTED_STARTS:
        raise RetainedHeldoutScoreError("held-out 253-start prefix is incomplete")
    if (
        canonical_sha256([str(record["record_identity"]) for record in prefix])
        != EXPECTED_PREFIX_RECORDS_IDENTITY
    ):
        raise RetainedHeldoutScoreError("held-out prefix record order differs")
    phases = Counter(str(record.get("phase") or "") for record in prefix)
    if dict(sorted(phases.items())) != EXPECTED_PHASE_COUNTS:
        raise RetainedHeldoutScoreError("held-out prefix phase counts differ")
    return prefix


def build_schedule(
    records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str] = EXPECTED_CANDIDATES,
) -> list[dict[str, Any]]:
    """Build adjacent v3/v4 games for both colours at every frozen start."""
    if len(records) != EXPECTED_STARTS:
        raise RetainedHeldoutScoreError("held-out corpus must have 253 starts")
    if tuple(candidate_ids) != EXPECTED_CANDIDATES:
        raise RetainedHeldoutScoreError("held-out candidate order differs")
    start_ids = [str(record.get("start_id") or "") for record in records]
    if any(not item for item in start_ids) or len(set(start_ids)) != len(start_ids):
        raise RetainedHeldoutScoreError("held-out start IDs are not unique")

    schedule: list[dict[str, Any]] = []
    for start_index, record in enumerate(records):
        strict_start = record.get("strict_start")
        if not isinstance(strict_start, Mapping):
            raise RetainedHeldoutScoreError("held-out strict start is absent")
        record_identity = record.get("record_identity")
        history_sha256 = strict_start.get("history_sha256")
        start_ply = strict_start.get("logical_ply_count")
        start_turn = record.get("turn")
        if (
            not isinstance(record_identity, str)
            or not isinstance(history_sha256, str)
            or not isinstance(start_ply, int)
            or start_turn not in {"W", "B"}
        ):
            raise RetainedHeldoutScoreError("held-out start identity differs")
        for colour_index, candidate_color in enumerate(("W", "B")):
            unit_index = start_index * 2 + colour_index
            match_key = f"{record['start_id']}:{candidate_color}"
            for candidate_index, candidate_id in enumerate(candidate_ids):
                ordinal = unit_index * 2 + candidate_index
                identity_body = {
                    "ordinal": ordinal,
                    "match_key": match_key,
                    "candidate_id": candidate_id,
                    "candidate_color": candidate_color,
                    "start_id": record["start_id"],
                    "start_record_identity": record_identity,
                }
                schedule.append(
                    {
                        "ordinal": ordinal,
                        "unit_index": unit_index,
                        "candidate_index": candidate_index,
                        "candidate_id": candidate_id,
                        "candidate_color": candidate_color,
                        "match_key": match_key,
                        "start_id": str(record["start_id"]),
                        "phase": str(record["phase"]),
                        "start_turn": str(start_turn),
                        "start_logical_ply": start_ply,
                        "start_record_identity": str(record_identity),
                        "expected_start_history_sha256": str(history_sha256),
                        "game_id": "heldout-score-game:"
                        + canonical_sha256(identity_body),
                    }
                )
    if len(schedule) != EXPECTED_GAMES:
        raise RetainedHeldoutScoreError("held-out schedule size differs")
    return schedule


def play_heldout_score_game(**kwargs: Any) -> dict[str, Any]:
    """Play one strict held-out game using the shared complete-history route."""
    return play_phase_process_game(**kwargs, game_schema=GAME_SCHEMA)


def load_game_ledger(
    spec: Mapping[str, Any],
    path: str | Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Load and validate one exact ordered held-out ledger prefix."""
    return _load_phase_game_ledger(
        spec,
        path,
        expected_games=EXPECTED_GAMES,
        game_schema=GAME_SCHEMA,
    )


def _complete_colour_units(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Mapping[str, Any]]]:
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_unit.setdefault(str(record["match_key"]), {})[
            str(record["candidate_id"])
        ] = record
    return [unit for unit in by_unit.values() if set(unit) == set(EXPECTED_CANDIDATES)]


def _cluster_both_colours(
    colour_values: Mapping[tuple[str, str], float],
) -> list[float]:
    starts = sorted({key[0] for key in colour_values})
    return [
        (colour_values[(start, "W")] + colour_values[(start, "B")]) / 2.0
        for start in starts
        if (start, "W") in colour_values and (start, "B") in colour_values
    ]


def _paired_comparison(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete_units = _complete_colour_units(records)
    score_by_colour: dict[tuple[str, str], float] = {}
    survival_by_colour: dict[tuple[str, str], float] = {}
    length_by_colour: dict[tuple[str, str], float] = {}
    for unit in complete_units:
        v3 = unit[EXPECTED_CANDIDATES[0]]
        v4 = unit[EXPECTED_CANDIDATES[1]]
        key = (str(v3["start_id"]), str(v3["candidate_color"]))
        if (
            v3.get("candidate_score") is not None
            and v4.get("candidate_score") is not None
        ):
            score_by_colour[key] = float(v4["candidate_score"]) - float(
                v3["candidate_score"]
            )
        survival_by_colour[key] = float(
            v4["ongoing_after_post_start_logical_ply_108"]
        ) - float(v3["ongoing_after_post_start_logical_ply_108"])
        length_by_colour[key] = (
            int(v4["post_start_logical_plies"]) - int(v3["post_start_logical_plies"])
        ) / MAX_POST_START_LOGICAL_PLIES

    start_scores = _cluster_both_colours(score_by_colour)
    primary = _interval(start_scores)
    all_games_complete = len(records) == EXPECTED_GAMES
    all_starts_complete = len(start_scores) == EXPECTED_STARTS
    if not all_games_complete:
        decision = "pending"
    elif not all_starts_complete:
        decision = "inconclusive_incomplete_safety_cap"
    elif primary["half_width"] is None or primary["half_width"] > (
        MAX_PRIMARY_HALF_WIDTH
    ):
        decision = "inconclusive_precision"
    elif primary["interval"][0] > 0:
        decision = "v4_higher_fixed_heldout_score"
    elif primary["interval"][1] < 0:
        decision = "v3_higher_fixed_heldout_score"
    else:
        decision = "inconclusive"
    distribution = Counter(start_scores)
    return {
        "matched_colour_units_complete": len(complete_units),
        "matched_colour_units_expected": EXPECTED_MATCHED_COLOUR_UNITS,
        "start_score_units_complete": len(start_scores),
        "start_score_units_expected": EXPECTED_STARTS,
        "primary_start_clustered_score_v4_minus_v3": {
            **primary,
            "decision": decision,
            "maximum_half_width": MAX_PRIMARY_HALF_WIDTH,
            "precision_adequate": (
                primary["half_width"] is not None
                and primary["half_width"] <= MAX_PRIMARY_HALF_WIDTH
            ),
            "all_rules_terminal": all_starts_complete,
            "distribution": {
                str(value): distribution[value] for value in sorted(distribution)
            },
            "interpretation": (
                "named-route fixed held-out corpus engineering interval; "
                "not equivalence, Elo, population strength, or refresh causality"
            ),
        },
        "colour_unit_score_v4_minus_v3": _interval(list(score_by_colour.values())),
        "start_clustered_108_ply_survival_v4_minus_v3": _interval(
            _cluster_both_colours(survival_by_colour)
        ),
        "start_clustered_restricted_length_v4_minus_v3": _interval(
            _cluster_both_colours(length_by_colour)
        ),
    }


def summarize_records(
    spec: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    tail: str | None,
) -> dict[str, Any]:
    """Build the canonical partial or complete held-out score report."""
    grouped = {
        candidate_id: [
            record for record in records if record["candidate_id"] == candidate_id
        ]
        for candidate_id in EXPECTED_CANDIDATES
    }
    phases = sorted({str(record["phase"]) for record in records})
    body = {
        "schema_version": REPORT_SCHEMA,
        "diagnostic_id": spec["diagnostic_id"],
        "spec_identity": spec["spec_identity"],
        "status": "completed" if len(records) == EXPECTED_GAMES else "partial",
        "completed_games": len(records),
        "expected_games": EXPECTED_GAMES,
        "ledger_tail_record_sha256": tail,
        "by_candidate": {
            candidate_id: _candidate_summary(grouped[candidate_id])
            for candidate_id in EXPECTED_CANDIDATES
        },
        "paired": _paired_comparison(records),
        "by_candidate_color": {
            color: {
                candidate_id: _candidate_summary(
                    [
                        record
                        for record in grouped[candidate_id]
                        if record["candidate_color"] == color
                    ]
                )
                for candidate_id in EXPECTED_CANDIDATES
            }
            for color in ("W", "B")
        },
        "by_phase": {
            phase: {
                candidate_id: _candidate_summary(
                    [
                        record
                        for record in grouped[candidate_id]
                        if record["phase"] == phase
                    ]
                )
                for candidate_id in EXPECTED_CANDIDATES
            }
            for phase in phases
        },
        "claim_boundary": {
            "candidate_blind_source_selection": True,
            "held_out": True,
            "named_route_fixed_corpus_score_relation": True,
            "equivalence_claim": False,
            "elo_or_population_strength_claim": False,
            "refresh_causal_claim": False,
            "automatic_promotion_or_publication": False,
            "survival_is_strength": False,
            "malom_is_history_aware": False,
            "safety_cap_is_draw": False,
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def recompute_report(
    spec: Mapping[str, Any],
    ledger_path: str | Path,
) -> dict[str, Any]:
    records, tail = load_game_ledger(spec, ledger_path)
    return summarize_records(spec, records, tail)


__all__ = [
    "EXPECTED_CANDIDATES",
    "EXPECTED_GAMES",
    "EXPECTED_STARTS",
    "GAME_SCHEMA",
    "HORIZON_POST_START_LOGICAL_PLIES",
    "MAX_POST_START_LOGICAL_PLIES",
    "MAX_PRIMARY_HALF_WIDTH",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "SANMILL_NODE_CEILING",
    "SPEC_SCHEMA",
    "RetainedHeldoutScoreError",
    "append_game_record",
    "build_schedule",
    "load_corpus_records",
    "load_game_ledger",
    "play_heldout_score_game",
    "recompute_report",
    "replay_frozen_start",
    "summarize_records",
]
