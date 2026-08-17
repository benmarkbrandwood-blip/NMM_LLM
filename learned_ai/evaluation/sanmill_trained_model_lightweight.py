"""Lightweight, fail-closed trained-policy measurement helpers.

This module deliberately reuses only the strict gameplay and frozen policy
surfaces that predate this measurement.  It does not depend on the abandoned
baseline-v1 authorization, boundary registry, rehearsal, or coverage gates.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from learned_ai.evaluation.human_f0h0_feasibility import canonical_sha256
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ARMS as GUIDANCE_ARMS,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    build_schedule as build_guidance_schedule,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    select_schedule_excluding_starts,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    ARMS as CANDIDATE_ARMS,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    TrainedModelBaselineError,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import (
    build_schedule as build_candidate_schedule_core,
)
from learned_ai.evaluation.sanmill_trained_model_baseline import formal_states


PLAN_SCHEMA = "nmm.sanmill-trained-model-lightweight-plan.v1"
AUTHORIZATION_SCHEMA = (
    "nmm.sanmill-trained-model-lightweight-authorization.v1"
)
RESULT_SCHEMA = "nmm.sanmill-trained-model-lightweight-result.v1"
REPRODUCTION_ARM = "random-safe"
EXPECTED_STARTS = 254
EXPECTED_REPRODUCTION_GAMES = EXPECTED_STARTS * 2
EXPECTED_CANDIDATE_GAMES = EXPECTED_STARTS * 2 * len(CANDIDATE_ARMS)
EXPECTED_TOTAL_GAMES = EXPECTED_REPRODUCTION_GAMES + EXPECTED_CANDIDATE_GAMES
MAXIMUM_HALF_WIDTH = 0.015


class LightweightMeasurementError(TrainedModelBaselineError):
    """Raised when a frozen input or a known-answer check differs."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(
    path: str | Path,
    *,
    schema: str,
    identity_field: str,
) -> tuple[dict[str, Any], str]:
    target = Path(path)
    try:
        raw = target.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LightweightMeasurementError(
            f"cannot load sealed JSON: {target}"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise LightweightMeasurementError(f"sealed schema differs: {target}")
    identity = value.get(identity_field)
    body = dict(value)
    body.pop(identity_field, None)
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise LightweightMeasurementError(f"sealed identity differs: {target}")
    return value, hashlib.sha256(raw).hexdigest()


def load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, digest = load_sealed(
        path,
        schema=PLAN_SCHEMA,
        identity_field="plan_identity",
    )
    experiment = plan.get("experiment", {})
    reproduction = plan.get("known_answer_reproduction", {})
    if (
        plan.get("status") != "frozen_before_any_candidate_outcome"
        or tuple(experiment.get("candidate_arms", ())) != CANDIDATE_ARMS
        or int(experiment.get("starts", -1)) != EXPECTED_STARTS
        or int(experiment.get("reproduction_games", -1))
        != EXPECTED_REPRODUCTION_GAMES
        or int(experiment.get("candidate_games", -1))
        != EXPECTED_CANDIDATE_GAMES
        or int(experiment.get("planned_total_games", -1))
        != EXPECTED_TOTAL_GAMES
        or float(plan.get("primary_decision", {}).get("maximum_95_half_width", -1))
        != MAXIMUM_HALF_WIDTH
        or reproduction.get("arm") != REPRODUCTION_ARM
        or reproduction.get("required_exact_match") is not True
    ):
        raise LightweightMeasurementError("lightweight frozen plan differs")
    return plan, digest


def load_authorization(path: str | Path) -> tuple[dict[str, Any], str]:
    authorization, digest = load_sealed(
        path,
        schema=AUTHORIZATION_SCHEMA,
        identity_field="authorization_identity",
    )
    if (
        authorization.get("operator") != "product-owner-direct"
        or authorization.get("status") != "authorized_once_unconsumed"
        or authorization.get("grant_count") != 1
    ):
        raise LightweightMeasurementError("lightweight authorization differs")
    return authorization, digest


def reproduction_schedule(
    pool: Mapping[str, Any],
    *,
    excluded_start_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Preserve attempt-002 game IDs and random seeds for random-safe only."""
    full = build_guidance_schedule(pool["states"])
    selected = select_schedule_excluding_starts(
        full,
        excluded_start_ids=excluded_start_ids,
    )
    rows = [dict(row) for row in selected if row["arm"] == REPRODUCTION_ARM]
    if (
        len(rows) != EXPECTED_REPRODUCTION_GAMES
        or {str(row["arm"]) for row in rows} != {REPRODUCTION_ARM}
        or len({(row["start_id"], row["candidate_color"]) for row in rows})
        != EXPECTED_REPRODUCTION_GAMES
    ):
        raise LightweightMeasurementError("reproduction schedule differs")
    return rows


def candidate_schedule(
    pool: Mapping[str, Any],
    *,
    excluded_start_ids: Sequence[str],
    namespace: str,
) -> list[dict[str, Any]]:
    states = formal_states(pool, excluded_start_ids=excluded_start_ids)
    rows = build_candidate_schedule_core(states, namespace=namespace)
    if len(rows) != EXPECTED_CANDIDATE_GAMES:
        raise LightweightMeasurementError("candidate schedule differs")
    return rows


def _turn_actions_identity(record: Mapping[str, Any]) -> str:
    turns = record.get("turns")
    if not isinstance(turns, list):
        raise LightweightMeasurementError("game turn collection is absent")
    actions = []
    for turn in turns:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("actions"), list):
            raise LightweightMeasurementError("game turn actions are absent")
        actions.append(list(turn["actions"]))
    return canonical_sha256(actions)


def terminal_fingerprint(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen per-game known-answer comparison surface."""
    final_state = record.get("final_state")
    if not isinstance(final_state, Mapping):
        raise LightweightMeasurementError("final state is absent")
    return {
        "game_id": str(record.get("game_id")),
        "start_id": str(record.get("start_id")),
        "candidate_color": str(record.get("candidate_color")),
        "candidate_score": record.get("candidate_score"),
        "winner": record.get("winner"),
        "outcome_reason": record.get("outcome_reason"),
        "post_start_logical_plies": record.get("post_start_logical_plies"),
        "final_history_sha256": final_state.get("history_sha256"),
        "final_no_capture_count": final_state.get("no_capture_count"),
        "final_repetition_current_count": final_state.get(
            "repetition_current_count"
        ),
        "turn_actions_identity": _turn_actions_identity(record),
    }


def _strict_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [record.get("candidate_score") for record in records]
    if any(score not in {0.0, 0.5, 1.0} for score in scores):
        raise LightweightMeasurementError("strict score is absent")
    return {
        "games": len(records),
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "score_rate": statistics.fmean(float(score) for score in scores),
        "termination_reasons": dict(
            sorted(Counter(str(row.get("outcome_reason")) for row in records).items())
        ),
    }


def exact_reproduction_gate(
    observed_records: Sequence[Mapping[str, Any]],
    reference_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact per-game terminal and full action-sequence reproduction."""
    expected_records = [
        row
        for row in reference_manifest.get("games", [])
        if row.get("arm") == REPRODUCTION_ARM
    ]
    if (
        len(observed_records) != EXPECTED_REPRODUCTION_GAMES
        or len(expected_records) != EXPECTED_REPRODUCTION_GAMES
    ):
        raise LightweightMeasurementError("known-answer game count differs")
    expected = {
        (str(row["start_id"]), str(row["candidate_color"])):
        terminal_fingerprint(row)
        for row in expected_records
    }
    observed = {
        (str(row["start_id"]), str(row["candidate_color"])):
        terminal_fingerprint(row)
        for row in observed_records
    }
    if len(expected) != EXPECTED_REPRODUCTION_GAMES or set(observed) != set(expected):
        raise LightweightMeasurementError("known-answer membership differs")
    mismatches = [
        {
            "start_id": key[0],
            "candidate_color": key[1],
            "expected": expected[key],
            "observed": observed[key],
        }
        for key in sorted(expected)
        if observed[key] != expected[key]
    ]
    summary = _strict_summary(observed_records)
    expected_summary = {
        "games": 508,
        "wins": 21,
        "draws": 414,
        "losses": 73,
        "score_rate": 0.44881889763779526,
        "termination_reasons": {
            "drawFiftyMove": 305,
            "drawThreefoldRepetition": 109,
            "loseFewerThanThree": 47,
            "loseNoLegalMoves": 47,
        },
    }
    passed = not mismatches and summary == expected_summary
    return {
        "passed": passed,
        "candidate_measurement_allowed": passed,
        "comparison_surface": [
            "game_id",
            "start_id",
            "candidate_color",
            "candidate_score",
            "winner",
            "outcome_reason",
            "post_start_logical_plies",
            "final_history_sha256",
            "final_no_capture_count",
            "final_repetition_current_count",
            "turn_actions_identity",
        ],
        "summary": summary,
        "expected_summary": expected_summary,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "observed_fingerprint_identity": canonical_sha256(
            [observed[key] for key in sorted(observed)]
        ),
        "expected_fingerprint_identity": canonical_sha256(
            [expected[key] for key in sorted(expected)]
        ),
    }


def compact_machine_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Retain auditable outcomes while leaving bulky turns in the raw ledger."""
    events = record.get("self_downgrade_events", [])
    if not isinstance(events, list):
        events = []
    return {
        "schema_version": str(record.get("schema_version")),
        "ordinal": int(record["ordinal"]),
        "game_id": str(record["game_id"]),
        "start_id": str(record["start_id"]),
        "phase": str(record["phase"]),
        "arm": str(record["arm"]),
        "candidate_color": str(record["candidate_color"]),
        "termination_class": str(record["termination_class"]),
        "outcome_reason": str(record["outcome_reason"]),
        "winner": record.get("winner"),
        "candidate_score": record.get("candidate_score"),
        "post_start_logical_plies": int(record["post_start_logical_plies"]),
        "final_history_sha256": record["final_state"]["history_sha256"],
        "turn_actions_identity": _turn_actions_identity(record),
        "self_downgrade_events": len(events),
        "self_downgrade_transitions": dict(
            sorted(Counter(str(event["transition"]) for event in events).items())
        ),
    }


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "CANDIDATE_ARMS",
    "EXPECTED_CANDIDATE_GAMES",
    "EXPECTED_REPRODUCTION_GAMES",
    "EXPECTED_STARTS",
    "EXPECTED_TOTAL_GAMES",
    "GUIDANCE_ARMS",
    "LightweightMeasurementError",
    "MAXIMUM_HALF_WIDTH",
    "PLAN_SCHEMA",
    "REPRODUCTION_ARM",
    "RESULT_SCHEMA",
    "candidate_schedule",
    "compact_machine_record",
    "exact_reproduction_gate",
    "load_authorization",
    "load_plan",
    "load_sealed",
    "reproduction_schedule",
    "sha256_file",
    "terminal_fingerprint",
]
