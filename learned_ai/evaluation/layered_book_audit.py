"""Fail-closed twelve-ply audits for both Sanmill Book representations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.evaluation.layered_opening_prefix import (
    LAYERED_PREFIX_SCHEMA,
    PREFIX_LOGICAL_PLIES_BY_SIDE_V2,
    PREFIX_LOGICAL_PLIES_V2,
    LayeredOpeningPrefixV2,
    LayeredPrefixError,
    build_layered_prefix_v2,
)
from learned_ai.evaluation.oracle_corpus import ring16_canonical_fen
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    DataQueryState,
    SanmillDataQuerySession,
    portable_source_identity,
)
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_OPENING_BOOK_ORACLE_ENTRIES,
    EXPECTED_OPENING_BOOK_RECOMMENDATIONS,
    EXPECTED_OPENING_BOOK_SHA256,
    SANMILL_OPENING_BOOK_RELATIVE,
    SanmillInstallation,
    nmm_move_base,
    project_stable_sanmill_fen,
)
from learned_ai.training.run_contract import canonical_sha256


LAYERED_BOOK_AUDIT_SCHEMA = "nmm.layered-book-source-audit.v1"
EXPECTED_NAMED_BOOK_VARIATIONS = 107
_SHA40 = set("0123456789abcdef")


class LayeredBookAuditError(LayeredPrefixError):
    """Raised when a Book source or its frozen audit evidence drifts."""


@dataclass(frozen=True)
class _OracleNode:
    actions: tuple[str, ...]
    state: DataQueryState | None


def _expected_side_counts(logical_ply_count: int) -> tuple[int, int]:
    return ((logical_ply_count + 1) // 2, logical_ply_count // 2)


def _validate_state(
    state: DataQueryState,
    *,
    actions: Sequence[str],
    logical_ply_count: int,
    previous: DataQueryState | None,
) -> None:
    if state.pending_removal or sum(state.pending_removals) != 0:
        raise LayeredBookAuditError("Book audit boundary has a pending removal")
    if state.action_token_count != len(actions):
        raise LayeredBookAuditError("Book audit action count drifted")
    if state.logical_ply_count != logical_ply_count:
        raise LayeredBookAuditError("Book audit logical count drifted")
    if state.logical_plies_by_side != _expected_side_counts(logical_ply_count):
        raise LayeredBookAuditError("Book audit side counts drifted")
    if previous is not None and state != previous:
        raise LayeredBookAuditError(
            "Book query state differs from the preceding replay state"
        )
    if state.outcome.terminal:
        if state.side_to_move is not None:
            raise LayeredBookAuditError(
                "terminal Book state retains a side to move"
            )
    else:
        expected_side = "white" if logical_ply_count % 2 == 0 else "black"
        if state.side_to_move != expected_side:
            raise LayeredBookAuditError("Book audit side to move drifted")


def _request_id(
    *,
    generator_commit: str,
    depth: int,
    actions: Sequence[str],
    purpose: str,
) -> str:
    digest = canonical_sha256(
        {
            "schema": LAYERED_BOOK_AUDIT_SCHEMA,
            "generator_commit": generator_commit,
            "target_logical_plies": PREFIX_LOGICAL_PLIES_V2,
            "depth": depth,
            "actions": list(actions),
            "purpose": purpose,
        }
    )
    return f"book-v2-{depth:02d}-{purpose}-{digest[:20]}"


def _candidate_evidence(candidate: DataQueryCandidate) -> dict[str, Any]:
    if (
        candidate.source_group_id is None
        or candidate.source_rank is None
        or candidate.remaining_actions != candidate.full_turn_actions
        or candidate.logical_ply_delta != 1
        or not candidate.turn_prefix_complete
    ):
        raise LayeredBookAuditError(
            "Book candidate is not a ranked complete logical turn"
        )
    return {
        "logical_move_id": candidate.logical_move_id,
        "stable_index": candidate.stable_index,
        "source_group_id": candidate.source_group_id,
        "source_rank": candidate.source_rank,
        "action_tokens": list(candidate.full_turn_actions),
    }


def _audit_oracle_query_graph(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    generator_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frontier = [_OracleNode(actions=(), state=None)]
    depth_audit: list[dict[str, Any]] = []
    leaf_records: list[dict[str, Any]] = []
    source_identity: dict[str, Any] | None = None

    for depth in range(PREFIX_LOGICAL_PLIES_V2):
        next_frontier: list[_OracleNode] = []
        available_count = 0
        miss_count = 0
        terminal_count = 0
        edge_count = 0
        compound_count = 0
        seen_children: set[tuple[str, ...]] = set()

        for node in sorted(frontier, key=lambda item: item.actions):
            response = session.query_book(
                node.actions,
                request_id=_request_id(
                    generator_commit=generator_commit,
                    depth=depth,
                    actions=node.actions,
                    purpose="query",
                ),
                expected_current_fen=(
                    node.state.current_fen if node.state is not None else None
                ),
            )
            if response.state is None:
                raise LayeredBookAuditError("Book response lacks state")
            _validate_state(
                response.state,
                actions=node.actions,
                logical_ply_count=depth,
                previous=node.state,
            )
            if response.source is None:
                raise LayeredBookAuditError("Book response lacks source identity")
            portable = portable_source_identity(response)
            if source_identity is None:
                source_identity = portable
            elif portable != source_identity:
                raise LayeredBookAuditError(
                    "Book source identity changed during enumeration"
                )

            if response.status in {"book_miss", "terminal"}:
                if response.status == "book_miss":
                    miss_count += 1
                else:
                    terminal_count += 1
                leaf_records.append(
                    {
                        "kind": response.status,
                        "logical_ply_count": depth,
                        "action_tokens": list(node.actions),
                        "fen": response.state.current_fen,
                        "history_sha256": response.state.history_sha256,
                    }
                )
                continue
            if response.status != "available" or response.candidates is None:
                raise LayeredBookAuditError(
                    f"unsupported Book status {response.status!r}"
                )

            available_count += 1
            for candidate in response.candidates:
                evidence = _candidate_evidence(candidate)
                child_actions = node.actions + candidate.full_turn_actions
                if child_actions in seen_children:
                    raise LayeredBookAuditError(
                        "Book graph returned a duplicate child history"
                    )
                seen_children.add(child_actions)
                edge_count += 1
                compound_count += int(candidate.contains_removal)
                summary = session.history_summary(
                    child_actions,
                    request_id=_request_id(
                        generator_commit=generator_commit,
                        depth=depth + 1,
                        actions=child_actions,
                        purpose="summary",
                    ),
                    count_mode="logical",
                )
                if summary.state is None or summary.status not in {
                    "available",
                    "terminal",
                }:
                    raise LayeredBookAuditError(
                        "Book child did not replay to an authoritative state"
                    )
                _validate_state(
                    summary.state,
                    actions=child_actions,
                    logical_ply_count=depth + 1,
                    previous=None,
                )
                if evidence["action_tokens"] != list(
                    candidate.full_turn_actions
                ):
                    raise AssertionError("Book candidate evidence changed")
                next_frontier.append(
                    _OracleNode(
                        actions=child_actions,
                        state=summary.state,
                    )
                )

        depth_audit.append(
            {
                "input_logical_ply": depth,
                "input_prefix_count": len(frontier),
                "available_input_count": available_count,
                "book_miss_input_count": miss_count,
                "terminal_input_count": terminal_count,
                "candidate_edge_count": edge_count,
                "compound_edge_count": compound_count,
                "unique_child_history_count": len(seen_children),
            }
        )
        frontier = sorted(next_frontier, key=lambda item: item.actions)

    if source_identity is None:
        raise LayeredBookAuditError("Book graph never bound a source identity")

    complete_paths = []
    for node in frontier:
        if node.state is None:
            raise LayeredBookAuditError("complete Book path lacks final state")
        board = project_stable_sanmill_fen(node.state.current_fen)
        nmm_fen = board.to_fen_string()
        complete_paths.append(
            {
                "action_tokens": list(node.actions),
                "history_sha256": node.state.history_sha256,
                "sanmill_fen": node.state.current_fen,
                "nmm_fen": nmm_fen,
                "ring16_canonical_fen": ring16_canonical_fen(nmm_fen),
                "path_identity": canonical_sha256(
                    {
                        "schema": LAYERED_BOOK_AUDIT_SCHEMA,
                        "subtype": "oracle_query_book",
                        "target_logical_plies": PREFIX_LOGICAL_PLIES_V2,
                        "action_tokens": list(node.actions),
                        "history_sha256": node.state.history_sha256,
                    }
                ),
            }
        )

    leaves = sorted(
        leaf_records,
        key=lambda item: (
            item["logical_ply_count"],
            item["kind"],
            item["action_tokens"],
        ),
    )
    leaf_depths = Counter(item["logical_ply_count"] for item in leaves)
    complete_paths.sort(key=lambda item: item["action_tokens"])
    result = {
        "source_subtype": "oracle_query_book",
        "fallback": "none",
        "depth_audit": depth_audit,
        "complete_history_count": len(complete_paths),
        "complete_histories": complete_paths,
        "incomplete_leaf_count": len(leaves),
        "book_miss_leaf_count": sum(
            item["kind"] == "book_miss" for item in leaves
        ),
        "terminal_leaf_count": sum(
            item["kind"] == "terminal" for item in leaves
        ),
        "incomplete_leaf_count_by_depth": [
            {"logical_ply_count": depth, "count": leaf_depths[depth]}
            for depth in sorted(leaf_depths)
        ],
        "incomplete_leaf_set_sha256": canonical_sha256(leaves),
        "unique_exact_final_fen_count": len(
            {item["nmm_fen"] for item in complete_paths}
        ),
        "unique_ring16_final_orbit_count": len(
            {item["ring16_canonical_fen"] for item in complete_paths}
        ),
    }
    return result, source_identity


def _line_tokens(opening: Mapping[str, Any]) -> tuple[str, ...]:
    raw = opening.get("lineMoves")
    if isinstance(raw, str):
        tokens = tuple(raw.split())
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, str)):
        tokens = tuple(str(token) for token in raw)
    else:
        raise LayeredBookAuditError("named Book variation has invalid lineMoves")
    if not tokens:
        raise LayeredBookAuditError("named Book variation has no lineMoves")
    return tokens


def _token_matches(move: Mapping[str, Any], token: str) -> bool:
    base, separator, capture = token.partition("x")
    return nmm_move_base(move) == base and (
        not separator or move.get("capture") == capture
    )


def _turn_actions(move: Mapping[str, Any]) -> tuple[str, ...]:
    primary = nmm_move_base(move)
    capture = move.get("capture")
    return (primary,) if capture is None else (primary, f"x{capture}")


def _expand_named_variation(
    opening: Mapping[str, Any],
) -> dict[str, Any]:
    opening_id = str(opening.get("id", ""))
    if not opening_id:
        raise LayeredBookAuditError("named Book variation lacks an id")
    tokens = _line_tokens(opening)
    base = {
        "variation_id": opening_id,
        "name": str(opening.get("name", "")),
        "family": str(opening.get("family", "")),
        "author_line_token_count": len(tokens),
        "author_prefix_tokens": list(tokens[:PREFIX_LOGICAL_PLIES_V2]),
    }
    if len(tokens) < PREFIX_LOGICAL_PLIES_V2:
        return {
            **base,
            "status": "shorter_than_12",
            "failed_at_logical_ply": None,
            "expanded_histories": (),
        }

    states: dict[
        tuple[tuple[str, ...], ...],
        BoardState,
    ] = {(): BoardState.new_game()}
    for logical_ply, token in enumerate(
        tokens[:PREFIX_LOGICAL_PLIES_V2],
        1,
    ):
        next_states: dict[tuple[tuple[str, ...], ...], BoardState] = {}
        for history, board in states.items():
            if is_terminal(board)[0]:
                continue
            for move in get_all_legal_moves(board):
                if not _token_matches(move, token):
                    continue
                turn = _turn_actions(move)
                next_states[history + (turn,)] = board.apply_move(move)
        if not next_states:
            return {
                **base,
                "status": "unreplayable",
                "failed_at_logical_ply": logical_ply,
                "expanded_histories": (),
            }
        states = next_states

    histories = tuple(
        (history, states[history])
        for history in sorted(states)
    )
    return {
        **base,
        "status": "complete",
        "failed_at_logical_ply": None,
        "expanded_histories": histories,
    }


def _step_evidence(
    variation_id: str,
    author_tokens: Sequence[str],
    history: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    evidence = []
    for index, (author_token, actions) in enumerate(
        zip(author_tokens, history, strict=True)
    ):
        evidence.append(
            {
                "variation_id": variation_id,
                "author_token_index": index,
                "author_token": author_token,
                "resolved_action_tokens": list(actions),
                "omitted_capture_resolved": (
                    "x" not in author_token and len(actions) == 2
                ),
            }
        )
    return evidence


def _multiplicity(values: Sequence[str]) -> list[dict[str, int]]:
    groups = Counter(Counter(values).values())
    return [
        {"multiplicity": multiplicity, "item_count": groups[multiplicity]}
        for multiplicity in sorted(groups)
    ]


def _audit_named_variations(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    openings: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    record_refs: list[dict[str, str]] = []

    for raw_opening in sorted(openings, key=lambda item: str(item.get("id"))):
        expansion = _expand_named_variation(raw_opening)
        histories = expansion.pop("expanded_histories")
        entry = dict(expansion)
        prefix_records: list[dict[str, Any]] = []
        if entry["status"] == "complete":
            author_tokens = tuple(entry["author_prefix_tokens"])
            for history, _board in histories:
                flattened = [
                    token for turn in history for token in turn
                ]
                source_history_id = canonical_sha256(
                    {
                        "schema": LAYERED_BOOK_AUDIT_SCHEMA,
                        "source_subtype": "named_book_variation",
                        "variation_id": entry["variation_id"],
                        "target_logical_plies": PREFIX_LOGICAL_PLIES_V2,
                        "author_prefix_tokens": list(author_tokens),
                        "action_tokens": flattened,
                        "source_identity_sha256": source_identity[
                            "identity_sha256"
                        ],
                    }
                )
                prefix = build_layered_prefix_v2(
                    session,
                    installation,
                    stratum="book",
                    source_subtype="named_book_variation",
                    source_history_id=source_history_id,
                    source_identity=source_identity,
                    source_evidence={
                        "variation_id": entry["variation_id"],
                        "name": entry["name"],
                        "family": entry["family"],
                        "author_line_token_count": entry[
                            "author_line_token_count"
                        ],
                        "author_prefix_tokens": list(author_tokens),
                        "selection_status": "audit_candidate_not_frozen",
                    },
                    logical_turns=history,
                    step_evidence=_step_evidence(
                        entry["variation_id"],
                        author_tokens,
                        history,
                    ),
                )
                record = prefix.to_dict()
                exact_history_sha256 = canonical_sha256(flattened)
                record["exact_history_sha256"] = exact_history_sha256
                prefix_records.append(record)
                record_refs.append(
                    {
                        "variation_id": entry["variation_id"],
                        "source_history_id": source_history_id,
                        "exact_history_sha256": exact_history_sha256,
                        "final_nmm_fen": prefix.final_nmm_fen,
                        "ring16_canonical_fen": prefix.final_ring16_fen,
                    }
                )
            prefix_records.sort(key=lambda item: item["action_tokens"])
            representative = prefix_records[0]
            entry.update(
                {
                    "expanded_history_count": len(prefix_records),
                    "unique_exact_history_count": len(
                        {
                            item["exact_history_sha256"]
                            for item in prefix_records
                        }
                    ),
                    "unique_exact_final_fen_count": len(
                        {
                            item["final"]["nmm_fen"]
                            for item in prefix_records
                        }
                    ),
                    "unique_ring16_final_orbit_count": len(
                        {
                            item["final"]["ring16_canonical_fen"]
                            for item in prefix_records
                        }
                    ),
                    "representative_rule": (
                        "lexicographically smallest complete action-token "
                        "history; audit candidate only"
                    ),
                    "representative_source_history_id": representative[
                        "source_history_id"
                    ],
                    "prefix_records": prefix_records,
                }
            )
        else:
            entry.update(
                {
                    "expanded_history_count": 0,
                    "unique_exact_history_count": 0,
                    "unique_exact_final_fen_count": 0,
                    "unique_ring16_final_orbit_count": 0,
                    "representative_rule": None,
                    "representative_source_history_id": None,
                    "prefix_records": [],
                }
            )
        entries.append(entry)

    statuses = Counter(entry["status"] for entry in entries)
    line_lengths = Counter(entry["author_line_token_count"] for entry in entries)
    exact_values = [item["exact_history_sha256"] for item in record_refs]
    fen_values = [item["final_nmm_fen"] for item in record_refs]
    orbit_values = [item["ring16_canonical_fen"] for item in record_refs]
    duplicate_histories = []
    by_history: dict[str, list[dict[str, str]]] = {}
    for item in record_refs:
        by_history.setdefault(item["exact_history_sha256"], []).append(item)
    for digest, group in sorted(by_history.items()):
        if len(group) > 1:
            duplicate_histories.append(
                {
                    "exact_history_sha256": digest,
                    "occurrences": [
                        {
                            "variation_id": item["variation_id"],
                            "source_history_id": item["source_history_id"],
                        }
                        for item in group
                    ],
                }
            )

    return {
        "source_subtype": "named_book_variation",
        "fallback": "none",
        "variation_count": len(entries),
        "variation_status_counts": [
            {"status": status, "count": statuses[status]}
            for status in sorted(statuses)
        ],
        "author_line_length_distribution": [
            {"logical_ply_count": length, "variation_count": line_lengths[length]}
            for length in sorted(line_lengths)
        ],
        "complete_variation_count": statuses["complete"],
        "expanded_history_record_count": len(record_refs),
        "unique_exact_history_count": len(set(exact_values)),
        "unique_exact_final_fen_count": len(set(fen_values)),
        "unique_ring16_final_orbit_count": len(set(orbit_values)),
        "exact_history_multiplicity": _multiplicity(exact_values),
        "exact_final_fen_multiplicity": _multiplicity(fen_values),
        "ring16_final_orbit_multiplicity": _multiplicity(orbit_values),
        "duplicate_exact_histories_across_variations": duplicate_histories,
        "variation_orbit_relation": (
            "many-to-many; author variations and ring16 structural orbits "
            "are reported separately"
        ),
        "entries": entries,
    }


def _load_opening_asset(
    installation: SanmillInstallation,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = installation.checkout / SANMILL_OPENING_BOOK_RELATIVE
    try:
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayeredBookAuditError(
            "cannot read the pinned Sanmill opening asset"
        ) from exc
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if digest != EXPECTED_OPENING_BOOK_SHA256:
        raise LayeredBookAuditError("Sanmill opening asset SHA-256 drifted")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("variant") != "nmm"
        or payload.get("symmetry") != "ring16"
    ):
        raise LayeredBookAuditError("Sanmill opening asset contract drifted")
    oracle = payload.get("oracle")
    openings = payload.get("openings")
    if not isinstance(oracle, Mapping) or not isinstance(openings, list):
        raise LayeredBookAuditError("Sanmill opening asset lacks source data")
    recommendations = sum(
        len(value) for value in oracle.values() if isinstance(value, list)
    )
    if (
        len(oracle) != EXPECTED_OPENING_BOOK_ORACLE_ENTRIES
        or recommendations != EXPECTED_OPENING_BOOK_RECOMMENDATIONS
        or len(openings) != EXPECTED_NAMED_BOOK_VARIATIONS
    ):
        raise LayeredBookAuditError("Sanmill opening asset counts drifted")
    metadata = {
        "path_lookup_key": "sanmill_checkout",
        "relative_path": SANMILL_OPENING_BOOK_RELATIVE.as_posix(),
        "sha256": digest,
        "byte_length": len(payload_bytes),
        "schema_version": payload["schemaVersion"],
        "variant": payload["variant"],
        "symmetry": payload["symmetry"],
        "oracle_position_count": len(oracle),
        "oracle_recommendation_count": recommendations,
        "named_variation_count": len(openings),
    }
    return payload, metadata


def build_layered_book_audit(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    generator_commit: str,
    fresh_processes: int = 2,
) -> dict[str, Any]:
    """Build one deterministic source-only Book audit payload."""
    if (
        len(generator_commit) != 40
        or any(char not in _SHA40 for char in generator_commit)
    ):
        raise LayeredBookAuditError("generator commit must be a full Git SHA")
    if fresh_processes < 2:
        raise LayeredBookAuditError(
            "Book audit requires two fresh-process verification runs"
        )
    asset, asset_metadata = _load_opening_asset(installation)
    oracle, source_identity = _audit_oracle_query_graph(
        session,
        installation,
        generator_commit=generator_commit,
    )
    if (
        source_identity["identity"]["sha256"]
        != asset_metadata["sha256"]
    ):
        raise LayeredBookAuditError(
            "Book query identity differs from the named-line asset"
        )
    named = _audit_named_variations(
        session,
        installation,
        openings=asset["openings"],
        source_identity=source_identity,
    )
    body = {
        "schema_version": LAYERED_BOOK_AUDIT_SCHEMA,
        "status": "source-only-needs-decision",
        "candidate_loaded": False,
        "games_played": 0,
        "fallback": "none",
        "target": {
            "prefix_schema": LAYERED_PREFIX_SCHEMA,
            "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
            "logical_plies_by_side": list(
                PREFIX_LOGICAL_PLIES_BY_SIDE_V2
            ),
        },
        "generator": {
            "algorithm": "layered-book-source-audit-v1",
            "nmm_llm_commit": generator_commit,
            "fresh_processes": fresh_processes,
            "byte_identical_runs_required": True,
        },
        "sanmill": installation.portable_record(),
        "asset": asset_metadata,
        "source_identity": source_identity,
        "oracle_query_book": oracle,
        "named_book_variations": named,
        "decision": {
            "final_corpus_frozen": False,
            "book_representatives_frozen": False,
            "book_seeded_perfect_db_continuation_authorized": False,
            "next_gate": (
                "compare pure Book candidates with genuine HumanDB and "
                "StrictSteps Perfect DB audits"
            ),
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def verify_layered_book_audit(payload: Mapping[str, Any]) -> dict[str, int]:
    """Strictly verify identities and the essential Book audit invariants."""
    expected_fields = {
        "schema_version",
        "status",
        "candidate_loaded",
        "games_played",
        "fallback",
        "target",
        "generator",
        "sanmill",
        "asset",
        "source_identity",
        "oracle_query_book",
        "named_book_variations",
        "decision",
        "audit_identity",
    }
    if set(payload) != expected_fields:
        raise LayeredBookAuditError("Book audit top-level fields drifted")
    if payload["schema_version"] != LAYERED_BOOK_AUDIT_SCHEMA:
        raise LayeredBookAuditError("Book audit schema drifted")
    if (
        payload["status"] != "source-only-needs-decision"
        or payload["candidate_loaded"] is not False
        or payload["games_played"] != 0
        or payload["fallback"] != "none"
    ):
        raise LayeredBookAuditError("Book audit scope boundary drifted")
    target = payload["target"]
    if target != {
        "prefix_schema": LAYERED_PREFIX_SCHEMA,
        "logical_ply_count": 12,
        "logical_plies_by_side": [6, 6],
    }:
        raise LayeredBookAuditError("Book audit target drifted")
    body = dict(payload)
    identity = body.pop("audit_identity")
    if canonical_sha256(body) != identity:
        raise LayeredBookAuditError("Book audit identity mismatch")

    oracle = payload["oracle_query_book"]
    named = payload["named_book_variations"]
    if oracle["fallback"] != "none" or named["fallback"] != "none":
        raise LayeredBookAuditError("Book audit fallback policy drifted")
    if len(oracle["depth_audit"]) != PREFIX_LOGICAL_PLIES_V2:
        raise LayeredBookAuditError("Book graph depth evidence is incomplete")
    record_count = 0
    for entry in named["entries"]:
        for raw_record in entry["prefix_records"]:
            record = dict(raw_record)
            exact_history_sha256 = record.pop("exact_history_sha256")
            parsed = LayeredOpeningPrefixV2.from_dict(record)
            if (
                parsed.stratum != "book"
                or parsed.source_subtype != "named_book_variation"
                or canonical_sha256(list(parsed.action_tokens))
                != exact_history_sha256
            ):
                raise LayeredBookAuditError(
                    "named Book v2 prefix record is inconsistent"
                )
            record_count += 1
    if record_count != named["expanded_history_record_count"]:
        raise LayeredBookAuditError("named Book record count drifted")
    return {
        "oracle_complete_histories": oracle["complete_history_count"],
        "oracle_incomplete_leaves": oracle["incomplete_leaf_count"],
        "named_complete_variations": named["complete_variation_count"],
        "named_prefix_records": record_count,
        "named_unique_histories": named["unique_exact_history_count"],
        "named_unique_ring16_orbits": named[
            "unique_ring16_final_orbit_count"
        ],
    }
