"""Fail-closed client for Sanmill's versioned Mill data-query protocol."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_OPENING_BOOK_ORACLE_ENTRIES,
    EXPECTED_OPENING_BOOK_RECOMMENDATIONS,
    EXPECTED_OPENING_BOOK_SHA256,
    SanmillBridgeError,
    SanmillInstallation,
    validate_uci_action_token,
)
from learned_ai.training.run_contract import canonical_sha256


DATA_QUERY_PROTOCOL_VERSION = 1
DATA_QUERY_COMMAND = ("mill", "data-query", "--jsonl")
_OPERATIONS = frozenset(
    {
        "query_book",
        "query_perfect_db",
        "query_human_db",
        "history_summary",
        "source_identity",
    }
)
_QUERY_STATUSES = {
    "query_book": frozenset({"available", "book_miss", "terminal", "error"}),
    "query_perfect_db": frozenset(
        {"available", "db_miss", "terminal", "error"}
    ),
    "query_human_db": frozenset(
        {"available", "human_db_miss", "terminal", "error"}
    ),
    "history_summary": frozenset({"available", "terminal", "error"}),
    "source_identity": frozenset({"available", "error"}),
}
_SOURCE_KIND_BY_OPERATION = {
    "query_book": "book",
    "query_perfect_db": "perfect_db",
    "query_human_db": "human_db",
}
_LOGICAL_MOVE_ID = re.compile(r"^(book|perfect|human):[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SIDES = frozenset({"white", "black"})
_OUTCOME_KINDS = frozenset({"ongoing", "win", "draw", "abandoned", "win_team"})


class SanmillDataQueryError(SanmillBridgeError):
    """Raised for a local process, schema, identity, or response failure."""


class SanmillDataQueryProtocolError(SanmillDataQueryError):
    """A versioned fail-closed error returned by Sanmill data-query."""

    def __init__(
        self,
        *,
        operation: str,
        code: str,
        message: str,
        request_id: str | None,
        action_index: int | None,
    ) -> None:
        self.operation = operation
        self.code = code
        self.message = message
        self.request_id = request_id
        self.action_index = action_index
        detail = f"Sanmill {operation} data-query error {code}: {message}"
        if action_index is not None:
            detail += f" (action_index={action_index})"
        super().__init__(detail)


@dataclass(frozen=True)
class DataQueryOutcome:
    kind: str
    winner: int | None
    reason: str

    @property
    def terminal(self) -> bool:
        return self.kind != "ongoing"


@dataclass(frozen=True)
class DataQueryState:
    current_fen: str
    side_to_move: str | None
    phase: str
    pending_removal: bool
    pending_removals: tuple[int, int]
    no_capture_plies: int
    action_token_count: int
    logical_ply_count: int
    logical_plies_by_side: tuple[int, int]
    snapshot_history_len: int
    repetition_history_len: int
    history_sha256: str
    outcome: DataQueryOutcome


@dataclass(frozen=True)
class PerfectCandidateData:
    category: str
    wdl: int
    steps: int
    mode: str


@dataclass(frozen=True)
class HumanCandidateData:
    wins: int
    losses: int
    draws: int
    total: int
    frequency_numerator: int
    frequency_denominator: int
    relative_frequency: float
    empirical_win_rate: float
    empirical_draw_rate: float
    empirical_loss_rate: float
    legacy_experience_score: float
    moves_to_end_sum: float
    average_moves_to_end: float | None
    malom_wdl_after: str | None
    malom_dtw_after: int | None


@dataclass(frozen=True)
class DataQueryCandidate:
    logical_move_id: str
    source_group_id: str | None
    stable_index: int
    source_rank: int | None
    raw_notation: str | None
    mapped_notation: str
    full_turn_actions: tuple[str, ...]
    remaining_actions: tuple[str, ...]
    contains_removal: bool
    removal_action: str | None
    logical_ply_delta: int
    turn_prefix_complete: bool
    perfect: PerfectCandidateData | None
    human: HumanCandidateData | None


@dataclass(frozen=True)
class DataQueryResponse:
    protocol_version: int
    request_id: str | None
    operation: str
    status: str
    state: DataQueryState | None
    source: Mapping[str, Any] | None
    candidates: tuple[DataQueryCandidate, ...] | None
    result: Mapping[str, Any] | None
    raw_line: str

    @property
    def available(self) -> bool:
        return self.status == "available"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SanmillDataQueryError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise SanmillDataQueryError(f"{context} has a non-string field name")
    return value


def _fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise SanmillDataQueryError(f"{context} fields: {'; '.join(details)}")


def _string(value: Any, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise SanmillDataQueryError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context=context)


def _integer(
    value: Any,
    *,
    context: str,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SanmillDataQueryError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise SanmillDataQueryError(f"{context} must be at least {minimum}")
    return value


def _optional_integer(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context=context)


def _optional_positive_integer(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context=context, minimum=1)


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise SanmillDataQueryError(f"{context} must be a boolean")
    return value


def _finite_number(value: Any, *, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SanmillDataQueryError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SanmillDataQueryError(f"{context} must be finite")
    return result


def _sha256(value: Any, *, context: str) -> str:
    digest = _string(value, context=context)
    if not _SHA256.fullmatch(digest):
        raise SanmillDataQueryError(f"{context} must be a lowercase SHA-256")
    return digest


def _two_counts(value: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise SanmillDataQueryError(f"{context} must contain two counts")
    return (
        _integer(value[0], context=f"{context}[0]", minimum=0),
        _integer(value[1], context=f"{context}[1]", minimum=0),
    )


def _parse_outcome(value: Any) -> DataQueryOutcome:
    payload = _object(value, context="data-query state outcome")
    _fields(
        payload,
        required={"kind", "reason"},
        optional={"winner"},
        context="data-query state outcome",
    )
    kind = _string(payload["kind"], context="data-query outcome kind")
    if kind not in _OUTCOME_KINDS:
        raise SanmillDataQueryError(f"unknown data-query outcome kind {kind!r}")
    winner = _optional_integer(
        payload.get("winner"),
        context="data-query outcome winner",
    )
    if winner not in {None, 0, 1}:
        raise SanmillDataQueryError("data-query outcome winner is unsupported")
    reason = _string(payload["reason"], context="data-query outcome reason")
    if kind == "ongoing" and (winner is not None or reason != "ongoing"):
        raise SanmillDataQueryError("ongoing data-query outcome is inconsistent")
    if kind == "win" and winner not in {0, 1}:
        raise SanmillDataQueryError("winning data-query outcome lacks a side")
    if kind == "draw" and winner is not None:
        raise SanmillDataQueryError("drawn data-query outcome names a winner")
    return DataQueryOutcome(kind=kind, winner=winner, reason=reason)


def _parse_state(value: Any) -> DataQueryState:
    payload = _object(value, context="data-query state")
    _fields(
        payload,
        required={
            "current_fen",
            "side_to_move",
            "phase",
            "pending_removal",
            "pending_removals",
            "no_capture_plies",
            "action_token_count",
            "logical_ply_count",
            "logical_plies_by_side",
            "snapshot_history_len",
            "repetition_history_len",
            "history_sha256",
            "outcome",
        },
        context="data-query state",
    )
    side = _optional_string(
        payload["side_to_move"],
        context="data-query state side_to_move",
    )
    if side not in _SIDES | {None}:
        raise SanmillDataQueryError("data-query state has an unknown side")
    pending_removals = _two_counts(
        payload["pending_removals"],
        context="data-query state pending_removals",
    )
    logical_counts = _two_counts(
        payload["logical_plies_by_side"],
        context="data-query state logical_plies_by_side",
    )
    logical_count = _integer(
        payload["logical_ply_count"],
        context="data-query state logical_ply_count",
        minimum=0,
    )
    if sum(logical_counts) != logical_count:
        raise SanmillDataQueryError(
            "data-query logical-ply total differs from per-side counts"
        )
    action_count = _integer(
        payload["action_token_count"],
        context="data-query state action_token_count",
        minimum=0,
    )
    snapshot_count = _integer(
        payload["snapshot_history_len"],
        context="data-query state snapshot_history_len",
        minimum=0,
    )
    if snapshot_count != action_count:
        raise SanmillDataQueryError(
            "data-query snapshot history differs from action-token count"
        )
    pending = _boolean(
        payload["pending_removal"],
        context="data-query state pending_removal",
    )
    if pending != (sum(pending_removals) > 0):
        raise SanmillDataQueryError("data-query pending-removal fields disagree")
    outcome = _parse_outcome(payload["outcome"])
    if outcome.terminal and side is not None:
        raise SanmillDataQueryError("terminal data-query state retains side_to_move")
    if not outcome.terminal and side is None:
        raise SanmillDataQueryError("ongoing data-query state lacks side_to_move")
    return DataQueryState(
        current_fen=_string(
            payload["current_fen"],
            context="data-query state current_fen",
        ),
        side_to_move=side,
        phase=_string(payload["phase"], context="data-query state phase"),
        pending_removal=pending,
        pending_removals=pending_removals,
        no_capture_plies=_integer(
            payload["no_capture_plies"],
            context="data-query state no_capture_plies",
            minimum=0,
        ),
        action_token_count=action_count,
        logical_ply_count=logical_count,
        logical_plies_by_side=logical_counts,
        snapshot_history_len=snapshot_count,
        repetition_history_len=_integer(
            payload["repetition_history_len"],
            context="data-query state repetition_history_len",
            minimum=0,
        ),
        history_sha256=_sha256(
            payload["history_sha256"],
            context="data-query state history_sha256",
        ),
        outcome=outcome,
    )


def _parse_perfect(value: Any) -> PerfectCandidateData:
    payload = _object(value, context="Perfect DB candidate data")
    _fields(
        payload,
        required={"category", "wdl", "steps", "mode"},
        context="Perfect DB candidate data",
    )
    mode = _string(payload["mode"], context="Perfect DB candidate mode")
    if mode != "strict_steps":
        raise SanmillDataQueryError("Perfect DB candidate is not StrictSteps")
    category = _string(
        payload["category"],
        context="Perfect DB candidate category",
    )
    if category not in {"win", "draw", "loss"}:
        raise SanmillDataQueryError("Perfect DB candidate category is unknown")
    return PerfectCandidateData(
        category=category,
        wdl=_integer(payload["wdl"], context="Perfect DB candidate wdl"),
        steps=_integer(payload["steps"], context="Perfect DB candidate steps"),
        mode=mode,
    )


def _parse_human(value: Any) -> HumanCandidateData:
    payload = _object(value, context="HumanDB candidate data")
    _fields(
        payload,
        required={
            "wins",
            "losses",
            "draws",
            "total",
            "frequency_numerator",
            "frequency_denominator",
            "relative_frequency",
            "empirical_win_rate",
            "empirical_draw_rate",
            "empirical_loss_rate",
            "legacy_experience_score",
            "moves_to_end_sum",
        },
        optional={"average_moves_to_end", "malom_wdl_after", "malom_dtw_after"},
        context="HumanDB candidate data",
    )
    wins = _integer(payload["wins"], context="HumanDB wins", minimum=0)
    losses = _integer(payload["losses"], context="HumanDB losses", minimum=0)
    draws = _integer(payload["draws"], context="HumanDB draws", minimum=0)
    total = _integer(payload["total"], context="HumanDB total", minimum=0)
    if wins + losses + draws != total:
        raise SanmillDataQueryError("HumanDB outcome counts do not sum to total")
    frequency_numerator = _integer(
        payload["frequency_numerator"],
        context="HumanDB frequency numerator",
        minimum=0,
    )
    frequency_denominator = _integer(
        payload["frequency_denominator"],
        context="HumanDB frequency denominator",
        minimum=1,
    )
    if frequency_numerator != total or frequency_numerator > frequency_denominator:
        raise SanmillDataQueryError("HumanDB frequency counts are inconsistent")
    rates = [
        _finite_number(payload[name], context=f"HumanDB {name}")
        for name in (
            "relative_frequency",
            "empirical_win_rate",
            "empirical_draw_rate",
            "empirical_loss_rate",
        )
    ]
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise SanmillDataQueryError("HumanDB probability lies outside [0, 1]")
    average = payload.get("average_moves_to_end")
    average_moves = (
        None
        if average is None
        else _finite_number(average, context="HumanDB average_moves_to_end")
    )
    return HumanCandidateData(
        wins=wins,
        losses=losses,
        draws=draws,
        total=total,
        frequency_numerator=frequency_numerator,
        frequency_denominator=frequency_denominator,
        relative_frequency=rates[0],
        empirical_win_rate=rates[1],
        empirical_draw_rate=rates[2],
        empirical_loss_rate=rates[3],
        legacy_experience_score=_finite_number(
            payload["legacy_experience_score"],
            context="HumanDB legacy_experience_score",
        ),
        moves_to_end_sum=_finite_number(
            payload["moves_to_end_sum"],
            context="HumanDB moves_to_end_sum",
        ),
        average_moves_to_end=average_moves,
        malom_wdl_after=_optional_string(
            payload.get("malom_wdl_after"),
            context="HumanDB malom_wdl_after",
        ),
        malom_dtw_after=_optional_integer(
            payload.get("malom_dtw_after"),
            context="HumanDB malom_dtw_after",
        ),
    )


def _token_array(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SanmillDataQueryError(f"{context} must be a non-empty array")
    try:
        return tuple(validate_uci_action_token(item) for item in value)
    except SanmillBridgeError as exc:
        raise SanmillDataQueryError(f"{context} contains an invalid token") from exc


def _parse_candidate(value: Any, *, operation: str) -> DataQueryCandidate:
    payload = _object(value, context=f"{operation} candidate")
    _fields(
        payload,
        required={
            "logical_move_id",
            "stable_index",
            "mapped_notation",
            "full_turn_actions",
            "remaining_actions",
            "contains_removal",
            "logical_ply_delta",
            "turn_prefix_complete",
        },
        optional={
            "source_group_id",
            "source_rank",
            "raw_notation",
            "removal_action",
            "perfect",
            "human",
        },
        context=f"{operation} candidate",
    )
    logical_move_id = _string(
        payload["logical_move_id"],
        context=f"{operation} candidate logical_move_id",
    )
    if not _LOGICAL_MOVE_ID.fullmatch(logical_move_id):
        raise SanmillDataQueryError("candidate logical_move_id is malformed")
    expected_prefix = {
        "query_book": "book:",
        "query_perfect_db": "perfect:",
        "query_human_db": "human:",
    }[operation]
    if not logical_move_id.startswith(expected_prefix):
        raise SanmillDataQueryError("candidate identity has the wrong source")
    full_actions = _token_array(
        payload["full_turn_actions"],
        context=f"{operation} candidate full_turn_actions",
    )
    remaining_actions = _token_array(
        payload["remaining_actions"],
        context=f"{operation} candidate remaining_actions",
    )
    if len(remaining_actions) > len(full_actions) or (
        full_actions[-len(remaining_actions) :] != remaining_actions
    ):
        raise SanmillDataQueryError(
            "candidate remaining actions are not a suffix of the full turn"
        )
    contains_removal = _boolean(
        payload["contains_removal"],
        context=f"{operation} candidate contains_removal",
    )
    removal_action = _optional_string(
        payload.get("removal_action"),
        context=f"{operation} candidate removal_action",
    )
    removal_tokens = tuple(token for token in full_actions if token.startswith("x"))
    if (
        len(full_actions) not in {1, 2}
        or full_actions[0].startswith("x")
        or (len(full_actions) == 2 and not full_actions[1].startswith("x"))
    ):
        raise SanmillDataQueryError(
            "candidate actions do not form one NMM primary-plus-removal turn"
        )
    if contains_removal != bool(removal_tokens):
        raise SanmillDataQueryError("candidate removal flag disagrees with its actions")
    if contains_removal:
        if len(removal_tokens) != 1 or removal_action != removal_tokens[0]:
            raise SanmillDataQueryError("candidate removal action is inconsistent")
    elif removal_action is not None:
        raise SanmillDataQueryError("non-removal candidate has a removal action")
    logical_delta = _integer(
        payload["logical_ply_delta"],
        context=f"{operation} candidate logical_ply_delta",
        minimum=0,
    )
    complete = _boolean(
        payload["turn_prefix_complete"],
        context=f"{operation} candidate turn_prefix_complete",
    )
    if logical_delta != 1 or not complete:
        raise SanmillDataQueryError("candidate is not one complete logical ply")
    perfect = (
        _parse_perfect(payload["perfect"]) if "perfect" in payload else None
    )
    human = _parse_human(payload["human"]) if "human" in payload else None
    if operation == "query_perfect_db" and (perfect is None or human is not None):
        raise SanmillDataQueryError("Perfect DB candidate metadata is incomplete")
    if operation == "query_human_db" and (human is None or perfect is not None):
        raise SanmillDataQueryError("HumanDB candidate metadata is incomplete")
    if operation == "query_book" and (perfect is not None or human is not None):
        raise SanmillDataQueryError("book candidate has database metadata")
    source_group_id = _optional_string(
        payload.get("source_group_id"),
        context=f"{operation} candidate source_group_id",
    )
    source_rank = _optional_positive_integer(
        payload.get("source_rank"),
        context=f"{operation} candidate source_rank",
    )
    raw_notation = _optional_string(
        payload.get("raw_notation"),
        context=f"{operation} candidate raw_notation",
    )
    if operation == "query_book" and (
        source_group_id is None or source_rank is None or raw_notation is None
    ):
        raise SanmillDataQueryError("book candidate source metadata is incomplete")
    if operation == "query_perfect_db" and (
        source_group_id is not None
        or source_rank is not None
        or raw_notation is not None
    ):
        raise SanmillDataQueryError("Perfect DB candidate has ranked-source metadata")
    if operation == "query_human_db" and (
        source_group_id is not None
        or source_rank is not None
        or raw_notation is None
    ):
        raise SanmillDataQueryError("HumanDB candidate source metadata is inconsistent")
    return DataQueryCandidate(
        logical_move_id=logical_move_id,
        source_group_id=source_group_id,
        stable_index=_integer(
            payload["stable_index"],
            context=f"{operation} candidate stable_index",
            minimum=0,
        ),
        source_rank=source_rank,
        raw_notation=raw_notation,
        mapped_notation=_string(
            payload["mapped_notation"],
            context=f"{operation} candidate mapped_notation",
        ),
        full_turn_actions=full_actions,
        remaining_actions=remaining_actions,
        contains_removal=contains_removal,
        removal_action=removal_action,
        logical_ply_delta=logical_delta,
        turn_prefix_complete=complete,
        perfect=perfect,
        human=human,
    )


def _validate_book_source(source: Mapping[str, Any]) -> None:
    _fields(
        source,
        required={
            "identity",
            "canonical_fen",
            "transform_to_canonical",
            "candidate_order",
            "selection_weight",
        },
        context="opening-book source",
    )
    if source["candidate_order"] != "source_array":
        raise SanmillDataQueryError("opening-book candidate order changed")
    _string(source["canonical_fen"], context="opening-book canonical_fen")
    _integer(
        source["transform_to_canonical"],
        context="opening-book transform",
        minimum=0,
    )
    weight = _object(source["selection_weight"], context="opening-book weight")
    _fields(
        weight,
        required={"kind", "ratio", "formula"},
        context="opening-book weight",
    )
    if (
        weight["kind"] != "geometric_rank"
        or _finite_number(weight["ratio"], context="opening-book weight ratio")
        != 0.6
        or weight["formula"] != "ratio^(rank-1)"
    ):
        raise SanmillDataQueryError("opening-book selection metadata changed")
    identity = _object(source["identity"], context="opening-book identity")
    _fields(
        identity,
        required={
            "kind",
            "schema_version",
            "variant",
            "symmetry",
            "sha256",
            "byte_length",
            "oracle_positions",
            "oracle_records",
            "source",
        },
        context="opening-book identity",
    )
    if (
        identity["kind"] != "opening_book"
        or identity["schema_version"] != 1
        or identity["variant"] != "nmm"
        or identity["symmetry"] != "ring16"
        or _sha256(identity["sha256"], context="opening-book SHA-256")
        != EXPECTED_OPENING_BOOK_SHA256
        or identity["oracle_positions"] != EXPECTED_OPENING_BOOK_ORACLE_ENTRIES
        or identity["oracle_records"] != EXPECTED_OPENING_BOOK_RECOMMENDATIONS
        or identity["source"] != "bundled"
    ):
        raise SanmillDataQueryError("opening-book identity differs from the pin")
    _integer(identity["byte_length"], context="opening-book byte_length", minimum=1)


def _validate_perfect_source(source: Mapping[str, Any]) -> None:
    _fields(
        source,
        required={
            "identity",
            "query_mode",
            "candidate_order",
            "fallback",
            "coverage",
        },
        context="Perfect DB source",
    )
    if (
        source["query_mode"] != "strict_steps"
        or source["candidate_order"] != "full_turn_uci_lexicographic"
        or source["fallback"] != "none"
    ):
        raise SanmillDataQueryError("Perfect DB source is not strict and stable")
    coverage = _object(source["coverage"], context="Perfect DB coverage")
    _fields(
        coverage,
        required={"placing", "moving", "flying", "pending_removal"},
        context="Perfect DB coverage",
    )
    if (
        coverage["placing"] is not True
        or coverage["moving"] is not True
        or coverage["flying"] is not True
        or coverage["pending_removal"] != "resolved_by_legal_continuation"
    ):
        raise SanmillDataQueryError("Perfect DB coverage contract changed")
    identity = _object(source["identity"], context="Perfect DB identity")
    _fields(
        identity,
        required={
            "kind",
            "database_format",
            "sector_format_version",
            "variant",
            "root",
            "secval_sha256",
            "fast_manifest_sha256",
            "manifest_algorithm",
            "declared_sector_count",
            "available_sector_count",
            "placement_sector_count",
            "settled_sector_count",
            "flying_related_sector_count",
            "fully_available",
        },
        optional={"full_content_algorithm", "full_content_sha256"},
        context="Perfect DB identity",
    )
    if (
        identity["kind"] != "perfect_database"
        or identity["database_format"] != "malom-sector"
        or identity["sector_format_version"] != 2
        or identity["variant"] != "std"
    ):
        raise SanmillDataQueryError("Perfect DB format identity is unsupported")
    _string(identity["root"], context="Perfect DB root")
    _sha256(identity["secval_sha256"], context="Perfect DB secval SHA-256")
    _sha256(
        identity["fast_manifest_sha256"],
        context="Perfect DB fast manifest SHA-256",
    )
    _string(
        identity["manifest_algorithm"],
        context="Perfect DB manifest algorithm",
    )
    for field in (
        "declared_sector_count",
        "available_sector_count",
        "placement_sector_count",
        "settled_sector_count",
        "flying_related_sector_count",
    ):
        _integer(identity[field], context=f"Perfect DB {field}", minimum=0)
    _boolean(identity["fully_available"], context="Perfect DB fully_available")
    if ("full_content_algorithm" in identity) != (
        "full_content_sha256" in identity
    ):
        raise SanmillDataQueryError("Perfect DB full identity fields are incomplete")
    if "full_content_algorithm" in identity:
        _string(
            identity["full_content_algorithm"],
            context="Perfect DB full-content algorithm",
        )
        _sha256(
            identity["full_content_sha256"],
            context="Perfect DB full-content SHA-256",
        )


def _validate_human_source(source: Mapping[str, Any]) -> None:
    _fields(
        source,
        required={
            "identity",
            "state_key",
            "symmetry_index",
            "candidate_order",
            "frequency_denominator_scope",
            "total_matching_candidates",
            "eligible_candidate_count",
            "returned_candidate_count",
            "candidate_limit",
            "min_total",
            "position",
            "fallback",
        },
        context="HumanDB source",
    )
    if (
        source["candidate_order"]
        != "total_desc_then_canonical_notation_then_mapped_notation"
        or source["fallback"] != "none"
    ):
        raise SanmillDataQueryError("HumanDB source order or fallback changed")
    _string(source["state_key"], context="HumanDB state_key")
    _integer(source["symmetry_index"], context="HumanDB symmetry_index", minimum=0)
    scope = _string(
        source["frequency_denominator_scope"],
        context="HumanDB frequency denominator scope",
    )
    if scope not in {
        "all_state_candidates",
        "all_candidates_matching_pending_turn_prefix",
    }:
        raise SanmillDataQueryError("HumanDB frequency scope is unsupported")
    for field in (
        "total_matching_candidates",
        "eligible_candidate_count",
        "returned_candidate_count",
        "min_total",
    ):
        _integer(source[field], context=f"HumanDB {field}", minimum=0)
    if source["candidate_limit"] is not None:
        _integer(
            source["candidate_limit"],
            context="HumanDB candidate_limit",
            minimum=1,
        )
    position = _object(source["position"], context="HumanDB position")
    _fields(
        position,
        required={"total_games", "wins", "losses", "draws"},
        optional={"malom_wdl", "malom_dtw", "canonical_winning_move"},
        context="HumanDB position",
    )
    position_counts = [
        _integer(position[name], context=f"HumanDB position {name}", minimum=0)
        for name in ("wins", "losses", "draws")
    ]
    if sum(position_counts) != _integer(
        position["total_games"],
        context="HumanDB position total_games",
        minimum=0,
    ):
        raise SanmillDataQueryError("HumanDB position counts do not sum")
    identity = _object(source["identity"], context="HumanDB identity")
    _fields(
        identity,
        required={
            "kind",
            "database_format",
            "path",
            "sha256",
            "file_size",
            "schema_version",
            "schema_sha256",
            "build_date",
            "total_games",
            "position_count",
            "move_count",
            "read_only",
            "immutable",
            "sidecars_absent",
            "malom_label_version",
            "malom_trusted",
            "malom_trust_reason",
            "meta",
        },
        context="HumanDB identity",
    )
    if (
        identity["kind"] != "human_database"
        or identity["database_format"] != "nmm-llm-human-db"
        or identity["schema_version"] != "2"
        or identity["read_only"] is not True
        or identity["immutable"] is not True
        or identity["sidecars_absent"] is not True
    ):
        raise SanmillDataQueryError("HumanDB storage identity is unsupported")
    _string(identity["path"], context="HumanDB path")
    _sha256(identity["sha256"], context="HumanDB SHA-256")
    _sha256(identity["schema_sha256"], context="HumanDB schema SHA-256")
    _string(identity["build_date"], context="HumanDB build_date")
    for field in ("file_size", "total_games", "position_count", "move_count"):
        _integer(identity[field], context=f"HumanDB {field}", minimum=0)
    _boolean(identity["malom_trusted"], context="HumanDB malom_trusted")
    _optional_string(
        identity["malom_label_version"],
        context="HumanDB malom_label_version",
    )
    _string(
        identity["malom_trust_reason"],
        context="HumanDB malom_trust_reason",
    )
    if not isinstance(identity["meta"], list):
        raise SanmillDataQueryError("HumanDB metadata must be an array")
    for index, item in enumerate(identity["meta"]):
        entry = _object(item, context=f"HumanDB metadata item {index}")
        _fields(
            entry,
            required={"key", "value"},
            context=f"HumanDB metadata item {index}",
        )
        _string(entry["key"], context=f"HumanDB metadata key {index}")
        _string(
            entry["value"],
            context=f"HumanDB metadata value {index}",
            allow_empty=True,
        )


def _validate_source(operation: str, value: Any) -> dict[str, Any]:
    source = _object(value, context=f"{operation} source")
    if operation == "query_book":
        _validate_book_source(source)
    elif operation == "query_perfect_db":
        _validate_perfect_source(source)
    elif operation == "query_human_db":
        _validate_human_source(source)
    else:
        raise SanmillDataQueryError(f"{operation} has no query-source contract")
    return source


def _parse_candidates(value: Any, *, operation: str) -> tuple[DataQueryCandidate, ...]:
    if not isinstance(value, list):
        raise SanmillDataQueryError(f"{operation} candidates must be an array")
    candidates = tuple(_parse_candidate(item, operation=operation) for item in value)
    stable_indices = [candidate.stable_index for candidate in candidates]
    if stable_indices != list(range(len(candidates))):
        raise SanmillDataQueryError("candidate stable indices are not contiguous")
    identities = [candidate.logical_move_id for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise SanmillDataQueryError("candidate identities are duplicated")
    action_sequences = [candidate.full_turn_actions for candidate in candidates]
    if len(action_sequences) != len(set(action_sequences)):
        raise SanmillDataQueryError("candidate logical turns are duplicated")
    return candidates


def _parse_error(
    payload: Mapping[str, Any],
    *,
    operation: str,
    request_id: str | None,
) -> SanmillDataQueryProtocolError:
    error = _object(payload["error"], context="data-query error")
    _fields(
        error,
        required={"code", "message"},
        optional={"action_index"},
        context="data-query error",
    )
    return SanmillDataQueryProtocolError(
        operation=operation,
        code=_string(error["code"], context="data-query error code"),
        message=_string(error["message"], context="data-query error message"),
        request_id=request_id,
        action_index=_optional_integer(
            error.get("action_index"),
            context="data-query error action_index",
        ),
    )


def parse_data_query_response(
    line: str,
    *,
    expected_operation: str | None = None,
    expected_request_id: str | None = None,
) -> DataQueryResponse:
    """Parse and validate exactly one compact protocol-v1 response line."""
    if not isinstance(line, str) or not line or "\n" in line or "\r" in line:
        raise SanmillDataQueryError("data-query response must be one non-empty line")
    try:
        value = json.loads(line, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SanmillDataQueryError("data-query response is not strict JSON") from exc
    payload = _object(value, context="data-query response")
    base = {"protocol_version", "operation", "status"}
    if "request_id" in payload:
        base.add("request_id")
    operation = _string(
        payload.get("operation"),
        context="data-query response operation",
    )
    if operation not in _OPERATIONS:
        raise SanmillDataQueryError(f"unknown data-query operation {operation!r}")
    if expected_operation is not None and operation != expected_operation:
        raise SanmillDataQueryError(
            f"data-query operation mismatch: {operation!r} != {expected_operation!r}"
        )
    version = _integer(
        payload.get("protocol_version"),
        context="data-query protocol_version",
        minimum=0,
    )
    if version != DATA_QUERY_PROTOCOL_VERSION:
        raise SanmillDataQueryError("unsupported data-query protocol version")
    request_id = _optional_string(
        payload.get("request_id"),
        context="data-query request_id",
    )
    if expected_request_id is not None and request_id != expected_request_id:
        raise SanmillDataQueryError(
            f"data-query request ID mismatch: {request_id!r} != "
            f"{expected_request_id!r}"
        )
    status = _string(payload.get("status"), context="data-query response status")
    if status not in _QUERY_STATUSES[operation]:
        raise SanmillDataQueryError(
            f"status {status!r} is invalid for {operation}"
        )
    if status == "error":
        _fields(
            payload,
            required=base | {"error"},
            context="data-query error response",
        )
        raise _parse_error(
            payload,
            operation=operation,
            request_id=request_id,
        )

    state: DataQueryState | None = None
    source: Mapping[str, Any] | None = None
    candidates: tuple[DataQueryCandidate, ...] | None = None
    result: Mapping[str, Any] | None = None
    if operation in _SOURCE_KIND_BY_OPERATION:
        if status == "terminal":
            _fields(
                payload,
                required=base | {"state", "candidates"},
                context="terminal data-query response",
            )
        else:
            _fields(
                payload,
                required=base | {"state", "source", "candidates"},
                context="source data-query response",
            )
            source = _validate_source(operation, payload["source"])
        state = _parse_state(payload["state"])
        candidates = _parse_candidates(payload["candidates"], operation=operation)
        if status == "available" and not candidates:
            raise SanmillDataQueryError("available data-query response is empty")
        if status != "available" and candidates:
            raise SanmillDataQueryError("non-available response contains candidates")
        if (status == "terminal") != state.outcome.terminal:
            raise SanmillDataQueryError("response status and terminal state disagree")
    elif operation == "history_summary":
        _fields(
            payload,
            required=base | {"state", "result"},
            context="history-summary response",
        )
        state = _parse_state(payload["state"])
        result_payload = _object(
            payload["result"],
            context="history-summary result",
        )
        _fields(
            result_payload,
            required={"count_mode", "selected_count"},
            context="history-summary result",
        )
        if result_payload["count_mode"] not in {"logical", "actions"}:
            raise SanmillDataQueryError("history-summary count mode is unsupported")
        selected_count = _integer(
            result_payload["selected_count"],
            context="history-summary selected_count",
            minimum=0,
        )
        expected_count = (
            state.logical_ply_count
            if result_payload["count_mode"] == "logical"
            else state.action_token_count
        )
        if selected_count != expected_count:
            raise SanmillDataQueryError("history-summary selected count drifted")
        if (status == "terminal") != state.outcome.terminal:
            raise SanmillDataQueryError("history-summary terminal status drifted")
        result = result_payload
    else:
        _fields(
            payload,
            required=base | {"source"},
            context="source-identity response",
        )
        source = _object(payload["source"], context="source identity")

    return DataQueryResponse(
        protocol_version=version,
        request_id=request_id,
        operation=operation,
        status=status,
        state=state,
        source=source,
        candidates=candidates,
        result=result,
        raw_line=line,
    )


def portable_source_identity(
    response: DataQueryResponse,
    *,
    path_lookup_key: str | None = None,
) -> dict[str, Any]:
    """Return a host-path-free source identity suitable for durable evidence."""
    if response.source is None:
        raise SanmillDataQueryError("response has no source identity")
    source_kind = _SOURCE_KIND_BY_OPERATION.get(response.operation)
    if source_kind is None:
        raise SanmillDataQueryError("response is not a source query")
    identity = _object(response.source.get("identity"), context="source identity")
    portable = json.loads(json.dumps(identity, sort_keys=True))
    if source_kind == "perfect_db":
        if not path_lookup_key:
            raise SanmillDataQueryError("Perfect DB identity needs a path lookup key")
        portable.pop("root", None)
        portable["path_lookup_key"] = path_lookup_key
    elif source_kind == "human_db":
        if not path_lookup_key:
            raise SanmillDataQueryError("HumanDB identity needs a path lookup key")
        portable.pop("path", None)
        portable["path_lookup_key"] = path_lookup_key
    elif path_lookup_key is not None:
        raise SanmillDataQueryError("opening-book identity has no path lookup key")
    return {
        "kind": source_kind,
        "identity": portable,
        "identity_sha256": canonical_sha256(portable),
    }


def _position_request(
    actions: Sequence[str],
    *,
    expected_current_fen: str | None,
) -> dict[str, Any]:
    tokens = [validate_uci_action_token(token) for token in actions]
    position: dict[str, Any] = {
        "rule": "nmm",
        "initial": "startpos",
        "history_origin": "game_start",
        "actions": tokens,
    }
    if expected_current_fen is not None:
        position["expected_current_fen"] = _string(
            expected_current_fen,
            context="expected_current_fen",
        )
    return position


class SanmillDataQuerySession:
    """Persistent strict JSONL process with one response per request."""

    def __init__(
        self,
        installation: SanmillInstallation,
        *,
        timeout: float = 120.0,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        if timeout <= 0:
            raise SanmillDataQueryError("data-query timeout must be positive")
        self.installation = installation
        self.timeout = timeout
        self.transcript: list[dict[str, str]] = []
        self._stdout: queue.Queue[str | None] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False
        child_env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("TGF_")
        }
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        try:
            self._process = popen_factory(
                [str(installation.binary), *DATA_QUERY_COMMAND],
                cwd=installation.checkout,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                env=child_env,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise SanmillDataQueryError(
                "cannot start the Sanmill data-query process"
            ) from exc
        if self._process.stdout is None or self._process.stderr is None:
            self.close()
            raise SanmillDataQueryError("data-query pipes were not created")
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def __enter__(self) -> "SanmillDataQuerySession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def stderr_text(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    def _pump_stdout(self, stream: Any) -> None:
        try:
            for line in stream:
                self._stdout.put(line.rstrip("\r\n"))
        finally:
            self._stdout.put(None)

    def _pump_stderr(self, stream: Any) -> None:
        for line in stream:
            with self._stderr_lock:
                self._stderr_lines.append(line.rstrip("\r\n"))

    def _send(
        self,
        request: Mapping[str, Any],
        *,
        expected_operation: str,
        expected_request_id: str,
    ) -> DataQueryResponse:
        if self._closed or self._process.poll() is not None:
            raise SanmillDataQueryError(
                f"Sanmill data-query is not running; stderr={self.stderr_text!r}"
            )
        if request.get("operation") != expected_operation:
            raise SanmillDataQueryError("internal data-query operation mismatch")
        line = json.dumps(
            request,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if "\n" in line or "\r" in line:
            raise SanmillDataQueryError("data-query request contains a newline")
        if self._process.stdin is None:
            raise SanmillDataQueryError("data-query stdin is unavailable")
        self.transcript.append({"direction": "to_engine", "line": line})
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SanmillDataQueryError("data-query stdin failed") from exc
        try:
            response_line = self._stdout.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise SanmillDataQueryError(
                f"data-query timed out; stderr={self.stderr_text!r}"
            ) from exc
        if response_line is None:
            raise SanmillDataQueryError(
                f"data-query exited before a response; stderr={self.stderr_text!r}"
            )
        self.transcript.append(
            {"direction": "from_engine", "line": response_line}
        )
        return parse_data_query_response(
            response_line,
            expected_operation=expected_operation,
            expected_request_id=expected_request_id,
        )

    @staticmethod
    def _base_request(operation: str, request_id: str) -> dict[str, Any]:
        if operation not in _OPERATIONS:
            raise SanmillDataQueryError(f"unsupported operation {operation!r}")
        return {
            "operation": operation,
            "protocol_version": DATA_QUERY_PROTOCOL_VERSION,
            "request_id": _string(request_id, context="request_id"),
        }

    def query_book(
        self,
        actions: Sequence[str],
        *,
        request_id: str,
        expected_current_fen: str | None = None,
    ) -> DataQueryResponse:
        request = self._base_request("query_book", request_id)
        request["position"] = _position_request(
            actions,
            expected_current_fen=expected_current_fen,
        )
        return self._send(
            request,
            expected_operation="query_book",
            expected_request_id=request_id,
        )

    def query_perfect_db(
        self,
        actions: Sequence[str],
        *,
        database_path: str | Path,
        request_id: str,
        expected_current_fen: str | None = None,
        cache_sectors: int | None = None,
    ) -> DataQueryResponse:
        request = self._base_request("query_perfect_db", request_id)
        request["position"] = _position_request(
            actions,
            expected_current_fen=expected_current_fen,
        )
        request["database_path"] = str(Path(database_path).resolve())
        if cache_sectors is not None:
            request["cache_sectors"] = _integer(
                cache_sectors,
                context="cache_sectors",
                minimum=1,
            )
        return self._send(
            request,
            expected_operation="query_perfect_db",
            expected_request_id=request_id,
        )

    def query_human_db(
        self,
        actions: Sequence[str],
        *,
        database_path: str | Path,
        request_id: str,
        expected_current_fen: str | None = None,
        candidate_limit: int | None = None,
        min_total: int = 0,
    ) -> DataQueryResponse:
        request = self._base_request("query_human_db", request_id)
        request["position"] = _position_request(
            actions,
            expected_current_fen=expected_current_fen,
        )
        request["database_path"] = str(Path(database_path).resolve())
        if candidate_limit is not None:
            request["candidate_limit"] = _integer(
                candidate_limit,
                context="candidate_limit",
                minimum=1,
            )
        request["min_total"] = _integer(
            min_total,
            context="min_total",
            minimum=0,
        )
        return self._send(
            request,
            expected_operation="query_human_db",
            expected_request_id=request_id,
        )

    def history_summary(
        self,
        actions: Sequence[str],
        *,
        request_id: str,
        expected_current_fen: str | None = None,
        count_mode: str = "logical",
    ) -> DataQueryResponse:
        if count_mode not in {"logical", "actions"}:
            raise SanmillDataQueryError("unsupported history count mode")
        request = self._base_request("history_summary", request_id)
        request["position"] = _position_request(
            actions,
            expected_current_fen=expected_current_fen,
        )
        request["count_mode"] = count_mode
        return self._send(
            request,
            expected_operation="history_summary",
            expected_request_id=request_id,
        )

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = self._process
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
