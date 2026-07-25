"""Deterministic StrictSteps source audit for twelve-ply Perfect DB routes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from learned_ai.evaluation.layered_book_audit import (
    verify_layered_book_audit,
)
from learned_ai.evaluation.layered_human_audit import (
    HUMAN_HISTORY_LEDGER_SCHEMA,
    HUMAN_HISTORY_SCHEMA,
    verify_layered_human_audit,
)
from learned_ai.evaluation.layered_opening_prefix import (
    LAYERED_PREFIX_SCHEMA,
    PREFIX_LOGICAL_PLIES_BY_SIDE_V2,
    PREFIX_LOGICAL_PLIES_V2,
    LayeredOpeningPrefixV2,
    LayeredPrefixError,
    build_layered_prefix_v2,
)
from learned_ai.evaluation.sanmill_data_query import (
    DataQueryCandidate,
    DataQueryResponse,
    DataQueryState,
    SanmillDataQuerySession,
    portable_source_identity,
)
from learned_ai.evaluation.sanmill_uci import SanmillInstallation
from learned_ai.training.run_contract import (
    canonical_json_bytes,
    canonical_sha256,
)


LAYERED_PERFECT_AUDIT_SCHEMA = "nmm.layered-perfect-source-audit.v1"
LAYERED_PERFECT_SELECTION_SCHEMA = "nmm.layered-perfect-selection.v1"
PERFECT_AUDIT_ID = "twelve-ply-strictsteps-source-audit-v1"
PERFECT_AUDIT_ROUTE_COUNT = 128
PERFECT_AUDIT_BASE_SEED = 42
_SOURCE_SUBTYPE = "strict_steps_deterministic_route"
_PATH_LOOKUP_KEY = "malom_db_path"
_SHA40 = frozenset("0123456789abcdef")


class LayeredPerfectAuditError(LayeredPrefixError):
    """Raised when Perfect DB source evidence is incomplete or unstable."""


@dataclass(frozen=True)
class SourceOverlapIndex:
    """Portable identities plus exact/FEN/orbit sets from prior source audits."""

    evidence: Mapping[str, Any]
    book_exact: frozenset[str]
    book_fen: frozenset[str]
    book_orbit: frozenset[str]
    human_exact: frozenset[str]
    human_fen: frozenset[str]
    human_orbit: frozenset[str]


class _CachedHistorySession:
    """Expose already-returned authoritative states to the v2 record builder."""

    def __init__(self, states: Mapping[tuple[str, ...], DataQueryResponse]) -> None:
        self._states = states

    def history_summary(
        self,
        actions: Sequence[str],
        *,
        request_id: str,
        expected_current_fen: str | None = None,
        count_mode: str = "logical",
    ) -> DataQueryResponse:
        del request_id
        if count_mode != "logical":
            raise LayeredPerfectAuditError(
                "Perfect audit replay requested a non-logical count"
            )
        response = self._states.get(tuple(actions))
        if response is None or response.state is None:
            raise LayeredPerfectAuditError(
                "Perfect audit replay state is absent from the query cache"
            )
        if (
            expected_current_fen is not None
            and response.state.current_fen != expected_current_fen
        ):
            raise LayeredPerfectAuditError(
                "Perfect audit replay FEN differs from the expected state"
            )
        return response


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayeredPerfectAuditError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise LayeredPerfectAuditError(f"{label} is not a JSON object")
    return value


def _book_sets(payload: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    exact: set[str] = set()
    fens: set[str] = set()
    orbits: set[str] = set()
    for entry in payload["named_book_variations"]["entries"]:
        for record in entry["prefix_records"]:
            exact.add(str(record["exact_history_sha256"]))
            fens.add(str(record["final"]["nmm_fen"]))
            orbits.add(str(record["final"]["ring16_canonical_fen"]))
    for record in payload["oracle_query_book"]["complete_histories"]:
        exact.add(canonical_sha256(record["action_tokens"]))
        fens.add(str(record["nmm_fen"]))
        orbits.add(str(record["ring16_canonical_fen"]))
    return exact, fens, orbits


def load_source_overlap_index(
    *,
    book_audit_path: str | Path,
    human_audit_path: str | Path,
    human_ledger_path: str | Path,
) -> SourceOverlapIndex:
    """Load and verify all prior-source identities before overlap comparison."""
    book_path = Path(book_audit_path).resolve()
    human_path = Path(human_audit_path).resolve()
    ledger_path = Path(human_ledger_path).resolve()
    book = _load_json(book_path, label="Book audit evidence")
    human = _load_json(human_path, label="HumanDB audit evidence")
    verify_layered_book_audit(book)
    verify_layered_human_audit(human)

    ledger_expected = human["raw_game_source"]["history_ledger"]
    if ledger_expected["schema_version"] != HUMAN_HISTORY_LEDGER_SCHEMA:
        raise LayeredPerfectAuditError("HumanDB history-ledger schema drifted")
    ledger_sha256 = _file_sha256(ledger_path)
    if (
        ledger_sha256 != ledger_expected["sha256"]
        or ledger_path.stat().st_size != ledger_expected["byte_length"]
    ):
        raise LayeredPerfectAuditError(
            "HumanDB history ledger differs from its tracked audit"
        )

    human_exact: set[str] = set()
    human_fen: set[str] = set()
    human_orbit: set[str] = set()
    try:
        with ledger_path.open("rb") as handle:
            first = handle.readline()
            header = json.loads(first)
            if header != {
                "schema_version": HUMAN_HISTORY_LEDGER_SCHEMA,
                "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
                "logical_plies_by_side": list(
                    PREFIX_LOGICAL_PLIES_BY_SIDE_V2
                ),
                "record_count": ledger_expected["history_count"],
                "ordering": (
                    "distinct_game_count_desc_then_occurrence_count_desc_then_"
                    "history_identity"
                ),
            }:
                raise LayeredPerfectAuditError(
                    "HumanDB history-ledger header drifted"
                )
            record_count = 0
            for raw_line in handle:
                record = json.loads(raw_line)
                if (
                    not isinstance(record, Mapping)
                    or record.get("schema_version") != HUMAN_HISTORY_SCHEMA
                    or record.get("logical_ply_count")
                    != PREFIX_LOGICAL_PLIES_V2
                    or record.get("logical_plies_by_side")
                    != list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2)
                ):
                    raise LayeredPerfectAuditError(
                        "HumanDB history-ledger record drifted"
                    )
                identity = str(record.get("history_identity", ""))
                if canonical_sha256(record.get("action_tokens")) != identity:
                    raise LayeredPerfectAuditError(
                        "HumanDB history identity does not match its actions"
                    )
                final = record.get("final")
                if not isinstance(final, Mapping):
                    raise LayeredPerfectAuditError(
                        "HumanDB history lacks a final state"
                    )
                if identity in human_exact:
                    raise LayeredPerfectAuditError(
                        "HumanDB history ledger contains a duplicate identity"
                    )
                human_exact.add(identity)
                human_fen.add(str(final.get("nmm_fen", "")))
                human_orbit.add(str(final.get("ring16_canonical_fen", "")))
                record_count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayeredPerfectAuditError(
            "cannot parse the HumanDB history ledger"
        ) from exc
    if record_count != ledger_expected["history_count"]:
        raise LayeredPerfectAuditError(
            "HumanDB history-ledger record count drifted"
        )

    book_exact, book_fen, book_orbit = _book_sets(book)
    evidence = {
        "book_audit": {
            "relative_path": book_path.relative_to(Path.cwd()).as_posix(),
            "byte_length": book_path.stat().st_size,
            "sha256": _file_sha256(book_path),
            "audit_identity": book["audit_identity"],
        },
        "human_audit": {
            "relative_path": human_path.relative_to(Path.cwd()).as_posix(),
            "byte_length": human_path.stat().st_size,
            "sha256": _file_sha256(human_path),
            "audit_identity": human["audit_identity"],
        },
        "human_history_ledger": {
            "path_lookup_key": ledger_expected["path_lookup_key"],
            "byte_length": ledger_expected["byte_length"],
            "sha256": ledger_sha256,
            "history_count": record_count,
        },
    }
    return SourceOverlapIndex(
        evidence=evidence,
        book_exact=frozenset(book_exact),
        book_fen=frozenset(book_fen),
        book_orbit=frozenset(book_orbit),
        human_exact=frozenset(human_exact),
        human_fen=frozenset(human_fen),
        human_orbit=frozenset(human_orbit),
    )


def _validate_boundary(
    state: DataQueryState,
    *,
    actions: Sequence[str],
    logical_ply_count: int,
) -> None:
    expected_counts = (
        (logical_ply_count + 1) // 2,
        logical_ply_count // 2,
    )
    if (
        state.outcome.terminal
        or state.pending_removal
        or sum(state.pending_removals) != 0
        or state.action_token_count != len(actions)
        or state.snapshot_history_len != len(actions)
        or state.logical_ply_count != logical_ply_count
        or state.logical_plies_by_side != expected_counts
        or state.side_to_move
        != ("white" if logical_ply_count % 2 == 0 else "black")
    ):
        raise LayeredPerfectAuditError(
            "Perfect DB query boundary is not one complete ongoing history"
        )


def _candidate_record(candidate: DataQueryCandidate) -> dict[str, Any]:
    if (
        candidate.perfect is None
        or candidate.remaining_actions != candidate.full_turn_actions
        or candidate.logical_ply_delta != 1
        or not candidate.turn_prefix_complete
    ):
        raise LayeredPerfectAuditError(
            "Perfect DB candidate is not one complete StrictSteps turn"
        )
    return {
        "logical_move_id": candidate.logical_move_id,
        "stable_index": candidate.stable_index,
        "action_tokens": list(candidate.full_turn_actions),
        "contains_removal": candidate.contains_removal,
        "perfect": {
            "category": candidate.perfect.category,
            "wdl": candidate.perfect.wdl,
            "steps": candidate.perfect.steps,
            "mode": candidate.perfect.mode,
        },
    }


def _validate_candidate_pool(
    candidates: Sequence[DataQueryCandidate],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise LayeredPerfectAuditError(
            "Perfect DB returned no tied-best candidates"
        )
    records = [_candidate_record(candidate) for candidate in candidates]
    if [record["stable_index"] for record in records] != list(
        range(len(records))
    ):
        raise LayeredPerfectAuditError(
            "Perfect DB stable indexes are not contiguous"
        )
    if [record["action_tokens"] for record in records] != sorted(
        record["action_tokens"] for record in records
    ):
        raise LayeredPerfectAuditError(
            "Perfect DB candidates are not in declared lexicographic order"
        )
    first = records[0]["perfect"]
    if first["mode"] != "strict_steps":
        raise LayeredPerfectAuditError(
            "Perfect DB candidate does not use StrictSteps"
        )
    for record in records[1:]:
        current = record["perfect"]
        if (
            current["mode"] != "strict_steps"
            or current["category"] != first["category"]
            or current["wdl"] != first["wdl"]
            or (
                first["category"] != "draw"
                and current["steps"] != first["steps"]
            )
        ):
            raise LayeredPerfectAuditError(
                "Perfect DB candidate pool contains non-tied outcomes"
            )
    tie = {
        "category": first["category"],
        "wdl": first["wdl"],
        "step_policy": (
            "draw_steps_not_ranked"
            if first["category"] == "draw"
            else (
                "minimum_steps"
                if first["category"] == "win"
                else "maximum_steps"
            )
        ),
        "step_values": sorted(
            {int(record["perfect"]["steps"]) for record in records}
        ),
        "candidate_count": len(records),
        "multiple_tied_best": len(records) > 1,
    }
    return records, tie


def _uniform_index(
    pool_size: int,
    *,
    route_id: str,
    seed: int,
    logical_ply: int,
    candidate_pool_identity: str,
) -> tuple[int, dict[str, Any]]:
    if pool_size <= 0:
        raise LayeredPerfectAuditError("candidate pool size must be positive")
    ceiling = 1 << 256
    unbiased_limit = ceiling - (ceiling % pool_size)
    attempt = 0
    while True:
        payload = {
            "schema": LAYERED_PERFECT_SELECTION_SCHEMA,
            "audit_id": PERFECT_AUDIT_ID,
            "route_id": route_id,
            "seed": seed,
            "logical_ply": logical_ply,
            "candidate_pool_identity": candidate_pool_identity,
            "attempt": attempt,
        }
        digest = hashlib.sha256(
            (
                canonical_sha256(payload)
                + ":nmm.layered-perfect-uniform-draw.v1"
            ).encode("ascii")
        ).hexdigest()
        draw = int(digest, 16)
        if draw < unbiased_limit:
            index = draw % pool_size
            return index, {
                "schema_version": LAYERED_PERFECT_SELECTION_SCHEMA,
                "method": "sha256-rejection-sampled-uniform-index",
                "attempt": attempt,
                "draw_sha256": digest,
                "candidate_pool_identity": candidate_pool_identity,
                "selected_stable_index": index,
            }
        attempt += 1


def _request_id(
    *,
    route_id: str,
    logical_ply: int,
    actions: Sequence[str],
    purpose: str,
) -> str:
    digest = canonical_sha256(
        {
            "schema": LAYERED_PERFECT_AUDIT_SCHEMA,
            "audit_id": PERFECT_AUDIT_ID,
            "route_id": route_id,
            "logical_ply": logical_ply,
            "actions": list(actions),
            "purpose": purpose,
        }
    )
    return f"perfect-v2-{logical_ply:02d}-{purpose}-{digest[:20]}"


def _source_contract(response: DataQueryResponse) -> dict[str, Any]:
    source = response.source
    if not isinstance(source, Mapping):
        raise LayeredPerfectAuditError(
            "Perfect DB response lacks source metadata"
        )
    contract = {
        "query_mode": source.get("query_mode"),
        "candidate_order": source.get("candidate_order"),
        "fallback": source.get("fallback"),
        "coverage": source.get("coverage"),
    }
    if contract != {
        "query_mode": "strict_steps",
        "candidate_order": "full_turn_uci_lexicographic",
        "fallback": "none",
        "coverage": {
            "placing": True,
            "moving": True,
            "flying": True,
            "pending_removal": "resolved_by_legal_continuation",
        },
    }:
        raise LayeredPerfectAuditError(
            "Perfect DB source contract is not strict and fallback-free"
        )
    return contract


def _multiplicity(values: Sequence[str]) -> list[dict[str, int]]:
    counts = Counter(Counter(values).values())
    return [
        {"occurrences": occurrences, "value_count": counts[occurrences]}
        for occurrences in sorted(counts)
    ]


def _overlap_summary(
    routes: Sequence[Mapping[str, Any]],
    overlap: SourceOverlapIndex,
) -> dict[str, Any]:
    exact = [str(route["exact_history_sha256"]) for route in routes]
    fens = [str(route["prefix_record"]["final"]["nmm_fen"]) for route in routes]
    orbits = [
        str(route["prefix_record"]["final"]["ring16_canonical_fen"])
        for route in routes
    ]
    return {
        "comparison_inputs": dict(overlap.evidence),
        "with_book": {
            "exact_history_count": sum(
                value in overlap.book_exact for value in exact
            ),
            "final_fen_count": sum(value in overlap.book_fen for value in fens),
            "ring16_orbit_count": sum(
                value in overlap.book_orbit for value in orbits
            ),
        },
        "with_human_db": {
            "exact_history_count": sum(
                value in overlap.human_exact for value in exact
            ),
            "final_fen_count": sum(
                value in overlap.human_fen for value in fens
            ),
            "ring16_orbit_count": sum(
                value in overlap.human_orbit for value in orbits
            ),
        },
    }


def build_layered_perfect_audit(
    session: SanmillDataQuerySession,
    installation: SanmillInstallation,
    *,
    database_path: str | Path,
    generator_commit: str,
    overlap: SourceOverlapIndex,
    route_count: int = PERFECT_AUDIT_ROUTE_COUNT,
    base_seed: int = PERFECT_AUDIT_BASE_SEED,
    fresh_processes: int = 2,
) -> dict[str, Any]:
    """Build a fixed pre-result pool of deterministic twelve-ply DB routes."""
    if (
        len(generator_commit) != 40
        or any(char not in _SHA40 for char in generator_commit)
    ):
        raise LayeredPerfectAuditError("generator commit must be a full Git SHA")
    if route_count <= 0 or base_seed < 0:
        raise LayeredPerfectAuditError("Perfect audit route settings are invalid")
    if fresh_processes < 2:
        raise LayeredPerfectAuditError(
            "Perfect audit requires at least two fresh processes"
        )
    database = Path(database_path).resolve()
    if not database.is_dir():
        raise LayeredPerfectAuditError(
            "Perfect Database directory is unavailable"
        )

    query_cache: dict[tuple[str, ...], DataQueryResponse] = {}
    state_cache: dict[tuple[str, ...], DataQueryResponse] = {}
    bound_source_identity: dict[str, Any] | None = None
    bound_source_contract: dict[str, Any] | None = None
    routes: list[dict[str, Any]] = []

    for route_number in range(route_count):
        route_id = f"perfect-audit-route-{route_number:03d}"
        route_seed = base_seed + route_number
        actions: tuple[str, ...] = ()
        logical_turns: list[tuple[str, ...]] = []
        step_evidence: list[dict[str, Any]] = []

        for logical_ply in range(PREFIX_LOGICAL_PLIES_V2):
            response = query_cache.get(actions)
            if response is None:
                response = session.query_perfect_db(
                    actions,
                    database_path=database,
                    request_id=_request_id(
                        route_id=route_id,
                        logical_ply=logical_ply,
                        actions=actions,
                        purpose="query",
                    ),
                    cache_sectors=8,
                )
                query_cache[actions] = response
            if (
                response.status != "available"
                or response.state is None
                or response.candidates is None
            ):
                raise LayeredPerfectAuditError(
                    f"Perfect DB is {response.status!r} at logical ply "
                    f"{logical_ply}; no fallback is permitted"
                )
            _validate_boundary(
                response.state,
                actions=actions,
                logical_ply_count=logical_ply,
            )
            state_cache[actions] = response
            source_identity = portable_source_identity(
                response,
                path_lookup_key=_PATH_LOOKUP_KEY,
            )
            source_contract = _source_contract(response)
            if bound_source_identity is None:
                bound_source_identity = source_identity
                bound_source_contract = source_contract
            elif (
                source_identity != bound_source_identity
                or source_contract != bound_source_contract
            ):
                raise LayeredPerfectAuditError(
                    "Perfect DB identity or query contract changed during audit"
                )

            candidates, tie = _validate_candidate_pool(response.candidates)
            pool_identity = canonical_sha256(candidates)
            selected_index, selection = _uniform_index(
                len(candidates),
                route_id=route_id,
                seed=route_seed,
                logical_ply=logical_ply,
                candidate_pool_identity=pool_identity,
            )
            selected = response.candidates[selected_index]
            selected_record = candidates[selected_index]
            logical_turns.append(selected.full_turn_actions)
            step_evidence.append(
                {
                    "query_contract": source_contract,
                    "candidate_pool_identity": pool_identity,
                    "candidate_pool": candidates,
                    "theoretical_tie": tie,
                    "selection": selection,
                    "selected_candidate": selected_record,
                }
            )
            actions += selected.full_turn_actions

        final_response = state_cache.get(actions)
        if final_response is None:
            final_response = session.history_summary(
                actions,
                request_id=_request_id(
                    route_id=route_id,
                    logical_ply=PREFIX_LOGICAL_PLIES_V2,
                    actions=actions,
                    purpose="final",
                ),
                count_mode="logical",
            )
            state_cache[actions] = final_response
        if final_response.status != "available" or final_response.state is None:
            raise LayeredPerfectAuditError(
                "Perfect DB route does not end at an ongoing twelve-ply state"
            )
        _validate_boundary(
            final_response.state,
            actions=actions,
            logical_ply_count=PREFIX_LOGICAL_PLIES_V2,
        )
        if bound_source_identity is None:
            raise LayeredPerfectAuditError("Perfect DB source was not bound")
        exact_history = canonical_sha256(list(actions))
        prefix = build_layered_prefix_v2(
            _CachedHistorySession(state_cache),  # type: ignore[arg-type]
            installation,
            stratum="perfect_db",
            source_subtype=_SOURCE_SUBTYPE,
            source_history_id=exact_history,
            source_identity=bound_source_identity,
            source_evidence={
                "audit_id": PERFECT_AUDIT_ID,
                "selection_schema": LAYERED_PERFECT_SELECTION_SCHEMA,
                "route_id": route_id,
                "route_seed": route_seed,
                "fallback": "none",
                "candidate_loaded": False,
            },
            logical_turns=logical_turns,
            step_evidence=step_evidence,
        )
        routes.append(
            {
                "route_id": route_id,
                "route_seed": route_seed,
                "exact_history_sha256": exact_history,
                "prefix_record": prefix.to_dict(),
            }
        )

    if bound_source_identity is None or bound_source_contract is None:
        raise LayeredPerfectAuditError("Perfect DB audit produced no routes")
    exact_values = [str(route["exact_history_sha256"]) for route in routes]
    fen_values = [
        str(route["prefix_record"]["final"]["nmm_fen"]) for route in routes
    ]
    orbit_values = [
        str(route["prefix_record"]["final"]["ring16_canonical_fen"])
        for route in routes
    ]
    selected_steps = [
        step["source_evidence"]["selected_candidate"]["perfect"]
        for route in routes
        for step in route["prefix_record"]["steps"]
    ]
    tied_counts = [
        int(step["source_evidence"]["theoretical_tie"]["candidate_count"])
        for route in routes
        for step in route["prefix_record"]["steps"]
    ]
    body = {
        "schema_version": LAYERED_PERFECT_AUDIT_SCHEMA,
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
            "algorithm": "deterministic-strictsteps-route-audit-v1",
            "audit_id": PERFECT_AUDIT_ID,
            "selection_schema": LAYERED_PERFECT_SELECTION_SCHEMA,
            "nmm_llm_commit": generator_commit,
            "route_count": route_count,
            "base_seed": base_seed,
            "route_pool_is_final_corpus": False,
            "fresh_processes": fresh_processes,
            "byte_identical_runs_required": True,
        },
        "sanmill": installation.portable_record(),
        "source_identity": bound_source_identity,
        "query_contract": bound_source_contract,
        "summary": {
            "route_count": len(routes),
            "unique_exact_history_count": len(set(exact_values)),
            "unique_exact_final_fen_count": len(set(fen_values)),
            "unique_ring16_final_orbit_count": len(set(orbit_values)),
            "exact_history_multiplicity": _multiplicity(exact_values),
            "exact_final_fen_multiplicity": _multiplicity(fen_values),
            "ring16_final_orbit_multiplicity": _multiplicity(orbit_values),
            "selected_outcome_categories": [
                {"category": category, "count": count}
                for category, count in sorted(
                    Counter(
                        str(record["category"]) for record in selected_steps
                    ).items()
                )
            ],
            "selected_wdl_values": [
                {"wdl": wdl, "count": count}
                for wdl, count in sorted(
                    Counter(
                        int(record["wdl"]) for record in selected_steps
                    ).items()
                )
            ],
            "tied_best_step_count": sum(count > 1 for count in tied_counts),
            "single_best_step_count": sum(count == 1 for count in tied_counts),
            "minimum_candidate_pool_size": min(tied_counts),
            "maximum_candidate_pool_size": max(tied_counts),
        },
        "overlap": _overlap_summary(routes, overlap),
        "routes": routes,
        "decision": {
            "final_corpus_frozen": False,
            "perfect_db_quota_frozen": False,
            "route_pool_is_final_corpus": False,
            "selection_status": (
                "fixed source-audit pool only; no corpus membership frozen"
            ),
            "next_gate": (
                "combine Book, HumanDB, and Perfect DB structure and overlap "
                "evidence in a corpus decision brief"
            ),
        },
    }
    return {**body, "audit_identity": canonical_sha256(body)}


def verify_layered_perfect_audit(
    payload: Mapping[str, Any],
) -> dict[str, int]:
    """Verify the scope, identities, routes, and summary of a Perfect audit."""
    expected = {
        "schema_version",
        "status",
        "candidate_loaded",
        "games_played",
        "fallback",
        "target",
        "generator",
        "sanmill",
        "source_identity",
        "query_contract",
        "summary",
        "overlap",
        "routes",
        "decision",
        "audit_identity",
    }
    if set(payload) != expected:
        raise LayeredPerfectAuditError(
            "Perfect DB audit top-level fields drifted"
        )
    if (
        payload["schema_version"] != LAYERED_PERFECT_AUDIT_SCHEMA
        or payload["status"] != "source-only-needs-decision"
        or payload["candidate_loaded"] is not False
        or payload["games_played"] != 0
        or payload["fallback"] != "none"
    ):
        raise LayeredPerfectAuditError("Perfect DB audit scope drifted")
    if payload["target"] != {
        "prefix_schema": LAYERED_PREFIX_SCHEMA,
        "logical_ply_count": PREFIX_LOGICAL_PLIES_V2,
        "logical_plies_by_side": list(PREFIX_LOGICAL_PLIES_BY_SIDE_V2),
    }:
        raise LayeredPerfectAuditError("Perfect DB audit target drifted")
    body = dict(payload)
    identity = body.pop("audit_identity")
    if canonical_sha256(body) != identity:
        raise LayeredPerfectAuditError("Perfect DB audit identity mismatch")
    if payload["query_contract"].get("fallback") != "none":
        raise LayeredPerfectAuditError("Perfect DB fallback policy drifted")
    if payload["source_identity"].get("kind") != "perfect_db":
        raise LayeredPerfectAuditError("Perfect DB source identity drifted")
    routes = payload["routes"]
    if (
        not isinstance(routes, list)
        or len(routes) != payload["generator"]["route_count"]
        or len(routes) != payload["summary"]["route_count"]
    ):
        raise LayeredPerfectAuditError("Perfect DB route count drifted")

    exact_values: list[str] = []
    fen_values: list[str] = []
    orbit_values: list[str] = []
    for index, route in enumerate(routes):
        if route["route_id"] != f"perfect-audit-route-{index:03d}":
            raise LayeredPerfectAuditError("Perfect DB route ordering drifted")
        prefix = LayeredOpeningPrefixV2.from_dict(route["prefix_record"])
        exact = canonical_sha256(list(prefix.action_tokens))
        if (
            prefix.stratum != "perfect_db"
            or prefix.source_subtype != _SOURCE_SUBTYPE
            or prefix.source_history_id != exact
            or route["exact_history_sha256"] != exact
            or prefix.source_identity != payload["source_identity"]
        ):
            raise LayeredPerfectAuditError(
                "Perfect DB route record is inconsistent"
            )
        for step in prefix.steps:
            evidence = step.source_evidence
            candidates = evidence["candidate_pool"]
            selected = evidence["selected_candidate"]
            if (
                evidence["query_contract"] != payload["query_contract"]
                or evidence["candidate_pool_identity"]
                != canonical_sha256(candidates)
                or evidence["theoretical_tie"]["candidate_count"]
                != len(candidates)
                or selected not in candidates
                or selected["stable_index"]
                != evidence["selection"]["selected_stable_index"]
            ):
                raise LayeredPerfectAuditError(
                    "Perfect DB step proof is inconsistent"
                )
        exact_values.append(exact)
        fen_values.append(prefix.final_nmm_fen)
        orbit_values.append(prefix.final_ring16_fen)

    summary = payload["summary"]
    if (
        summary["unique_exact_history_count"] != len(set(exact_values))
        or summary["unique_exact_final_fen_count"] != len(set(fen_values))
        or summary["unique_ring16_final_orbit_count"]
        != len(set(orbit_values))
        or payload["decision"] != {
            "final_corpus_frozen": False,
            "perfect_db_quota_frozen": False,
            "route_pool_is_final_corpus": False,
            "selection_status": (
                "fixed source-audit pool only; no corpus membership frozen"
            ),
            "next_gate": (
                "combine Book, HumanDB, and Perfect DB structure and overlap "
                "evidence in a corpus decision brief"
            ),
        }
    ):
        raise LayeredPerfectAuditError(
            "Perfect DB summary or decision boundary drifted"
        )
    return {
        "route_count": len(routes),
        "unique_histories": len(set(exact_values)),
        "unique_final_fens": len(set(fen_values)),
        "unique_ring16_orbits": len(set(orbit_values)),
        "book_exact_overlap": payload["overlap"]["with_book"][
            "exact_history_count"
        ],
        "human_exact_overlap": payload["overlap"]["with_human_db"][
            "exact_history_count"
        ],
    }
