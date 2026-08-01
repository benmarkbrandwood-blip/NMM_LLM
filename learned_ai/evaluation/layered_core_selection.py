"""Deterministic source-only selection for the twelve-ply balanced core."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from learned_ai.training.run_contract import canonical_sha256


BOOK_CORE_COUNT = 22
EXPERT_BOOK_CORE_COUNT = 15
SANMILL_BOOK_CORE_COUNT = 7
HUMAN_CORE_COUNT = 21
PERFECT_CORE_COUNT = 21


class LayeredCoreSelectionError(ValueError):
    """Raised when a frozen source cannot satisfy the core selection rule."""


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayeredCoreSelectionError(f"{context} must be an object")
    return value


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LayeredCoreSelectionError(f"{context} must be an array")
    return value


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LayeredCoreSelectionError(f"{context} must be a string")
    return value


def _prefix_key(prefix: Mapping[str, Any]) -> tuple[str, str, str]:
    final = _mapping(prefix.get("final"), context="prefix final")
    return (
        _string(
            prefix.get("source_history_id"),
            context="prefix source_history_id",
        ),
        _string(final.get("nmm_fen"), context="prefix final nmm_fen"),
        _string(
            final.get("ring16_canonical_fen"),
            context="prefix final ring16_canonical_fen",
        ),
    )


def _record(
    *,
    core_index: int,
    source_audit_identity: str,
    source_member_id: str,
    source_name: str,
    family: str,
    selection_basis: str,
    prefix: Mapping[str, Any],
) -> dict[str, Any]:
    history, final_fen, ring16 = _prefix_key(prefix)
    logical_count = prefix.get("logical_ply_count")
    side_counts = list(
        _sequence(
            prefix.get("logical_plies_by_side"),
            context="logical_plies_by_side",
        )
    )
    if logical_count != 12 or side_counts != [6, 6]:
        raise LayeredCoreSelectionError("Book prefix target drifted")
    action_tokens = list(
        _sequence(prefix.get("action_tokens"), context="action_tokens")
    )
    if sum(not str(token).startswith("x") for token in action_tokens) != 12:
        raise LayeredCoreSelectionError("Book action history is not 12 ply")

    final = _mapping(prefix.get("final"), context="prefix final")
    return {
        "core_id": f"book-core-{core_index:03d}",
        "stratum": "book",
        "source_subtype": _string(
            prefix.get("source_subtype"),
            context="source_subtype",
        ),
        "source_audit_identity": source_audit_identity,
        "source_member_id": source_member_id,
        "source_name": source_name,
        "family": family,
        "selection_basis": selection_basis,
        "source_history_id": history,
        "prefix_identity": _string(
            prefix.get("prefix_identity"),
            context="prefix_identity",
        ),
        "logical_ply_count": 12,
        "logical_plies_by_side": [6, 6],
        "action_tokens": action_tokens,
        "final": {
            "nmm_fen": final_fen,
            "ring16_canonical_fen": ring16,
            "history_sha256": _string(
                final.get("history_sha256"),
                context="final history_sha256",
            ),
        },
    }


def derive_book_core(
    *,
    sanmill_book_audit: Mapping[str, Any],
    expert_book_audit: Mapping[str, Any],
    expert_coverage: Mapping[str, Any],
    expert_shortlist: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive 15 expert and seven Sanmill Book members without a model."""

    sanmill_identity = _string(
        sanmill_book_audit.get("audit_identity"),
        context="Sanmill Book audit identity",
    )
    expert_identity = _string(
        expert_book_audit.get("audit_identity"),
        context="expert Book audit identity",
    )
    if expert_coverage.get("source_audit_identity") != expert_identity:
        raise LayeredCoreSelectionError("expert coverage audit identity drifted")
    if expert_shortlist.get("source_audit_identity") != expert_identity:
        raise LayeredCoreSelectionError("expert shortlist audit identity drifted")

    expert_records = {
        _string(item.get("variation_id"), context="expert variation_id"): item
        for item in (
            _mapping(raw, context="expert record")
            for raw in _sequence(
                expert_book_audit.get("records"),
                context="expert records",
            )
        )
    }
    coverage_ids = {
        _string(item.get("variation_id"), context="coverage variation_id")
        for item in (
            _mapping(raw, context="coverage record")
            for raw in _sequence(
                expert_coverage.get("coverage_catalog"),
                context="coverage catalogue",
            )
        )
    }

    selected: list[dict[str, Any]] = []
    seen_histories: set[str] = set()
    seen_fens: set[str] = set()
    seen_orbits: set[str] = set()

    def add(
        *,
        source_audit_identity: str,
        source_member_id: str,
        source_name: str,
        family: str,
        selection_basis: str,
        prefix: Mapping[str, Any],
    ) -> bool:
        history, final_fen, ring16 = _prefix_key(prefix)
        if (
            history in seen_histories
            or final_fen in seen_fens
            or ring16 in seen_orbits
        ):
            return False
        selected.append(
            _record(
                core_index=len(selected) + 1,
                source_audit_identity=source_audit_identity,
                source_member_id=source_member_id,
                source_name=source_name,
                family=family,
                selection_basis=selection_basis,
                prefix=prefix,
            )
        )
        seen_histories.add(history)
        seen_fens.add(final_fen)
        seen_orbits.add(ring16)
        return True

    parent_primaries = _sequence(
        expert_shortlist.get("breadth_first_parent_primaries"),
        context="expert parent primaries",
    )
    if len(parent_primaries) != 14:
        raise LayeredCoreSelectionError("expert parent breadth count drifted")
    for raw in parent_primaries:
        primary = _mapping(raw, context="expert parent primary")
        variation_id = _string(
            primary.get("variation_id"),
            context="expert primary variation_id",
        )
        if variation_id not in coverage_ids:
            raise LayeredCoreSelectionError(
                f"expert primary {variation_id} left the coverage catalogue"
            )
        source = _mapping(
            expert_records.get(variation_id),
            context=f"expert record {variation_id}",
        )
        if not add(
            source_audit_identity=expert_identity,
            source_member_id=variation_id,
            source_name=_string(source.get("label"), context="expert label"),
            family=_string(
                primary.get("review_id"),
                context="expert review_id",
            ),
            selection_basis="expert_parent_primary",
            prefix=_mapping(
                source.get("prefix_record"),
                context="expert prefix record",
            ),
        ):
            raise LayeredCoreSelectionError(
                f"expert primary {variation_id} duplicates prior breadth"
            )

    extra_selected = False
    for raw_id in _sequence(
        expert_shortlist.get("recommended_extra_order"),
        context="expert extra order",
    ):
        variation_id = _string(raw_id, context="expert extra variation_id")
        if variation_id not in coverage_ids:
            continue
        source = _mapping(
            expert_records.get(variation_id),
            context=f"expert extra {variation_id}",
        )
        if add(
            source_audit_identity=expert_identity,
            source_member_id=variation_id,
            source_name=_string(source.get("label"), context="expert label"),
            family="P03",
            selection_basis="expert_extended_family_priority",
            prefix=_mapping(
                source.get("prefix_record"),
                context="expert prefix record",
            ),
        ):
            extra_selected = True
            break
    if not extra_selected or len(selected) != EXPERT_BOOK_CORE_COUNT:
        raise LayeredCoreSelectionError("expert Book allocation is incomplete")

    named = _mapping(
        sanmill_book_audit.get("named_book_variations"),
        context="named Book variations",
    )
    complete_entries = [
        _mapping(raw, context="named Book entry")
        for raw in _sequence(named.get("entries"), context="named entries")
        if _mapping(raw, context="named Book entry").get("status")
        == "complete"
    ]
    family_order: list[str] = []
    for entry in complete_entries:
        family = _string(entry.get("family"), context="Book family")
        if family not in family_order:
            family_order.append(family)
    if len(family_order) != SANMILL_BOOK_CORE_COUNT:
        raise LayeredCoreSelectionError("Sanmill declared-family count drifted")

    for family in family_order:
        family_selected = False
        for entry in complete_entries:
            if entry.get("family") != family:
                continue
            representative_id = _string(
                entry.get("representative_source_history_id"),
                context="named representative history",
            )
            records = [
                _mapping(raw, context="named prefix record")
                for raw in _sequence(
                    entry.get("prefix_records"),
                    context="named prefix records",
                )
            ]
            matches = [
                prefix
                for prefix in records
                if prefix.get("source_history_id") == representative_id
            ]
            if len(matches) != 1:
                raise LayeredCoreSelectionError(
                    "named variation representative did not resolve once"
                )
            if add(
                source_audit_identity=sanmill_identity,
                source_member_id=_string(
                    entry.get("variation_id"),
                    context="named variation_id",
                ),
                source_name=_string(entry.get("name"), context="Book name"),
                family=family,
                selection_basis=(
                    "sanmill_family_first_nonduplicate_in_asset_order"
                ),
                prefix=matches[0],
            ):
                family_selected = True
                break
        if not family_selected:
            raise LayeredCoreSelectionError(
                f"no structurally unique representative for {family}"
            )

    if len(selected) != BOOK_CORE_COUNT:
        raise LayeredCoreSelectionError("Book core count drifted")
    if len(seen_histories) != BOOK_CORE_COUNT:
        raise LayeredCoreSelectionError("Book exact-history diversity drifted")
    if len(seen_fens) != BOOK_CORE_COUNT:
        raise LayeredCoreSelectionError("Book final-FEN diversity drifted")
    if len(seen_orbits) != BOOK_CORE_COUNT:
        raise LayeredCoreSelectionError("Book ring16 diversity drifted")

    return {
        "selection_schema": "nmm.layered-opening-prefix-book-core.v1",
        "allocation": {
            "book_total": BOOK_CORE_COUNT,
            "maintainer_expert_curated_play": EXPERT_BOOK_CORE_COUNT,
            "named_book_variation": SANMILL_BOOK_CORE_COUNT,
        },
        "selection_order": [item["core_id"] for item in selected],
        "members": selected,
        "membership_identity": canonical_sha256(selected),
    }


def _human_source_record(
    raw: Mapping[str, Any],
    *,
    source_audit_identity: str,
) -> dict[str, Any]:
    history_identity = _string(
        raw.get("history_identity"),
        context="HumanDB history_identity",
    )
    logical_count = raw.get("logical_ply_count")
    side_counts = list(
        _sequence(
            raw.get("logical_plies_by_side"),
            context="HumanDB logical_plies_by_side",
        )
    )
    turns = [
        list(_sequence(turn, context="HumanDB logical turn"))
        for turn in _sequence(
            raw.get("logical_turns"),
            context="HumanDB logical_turns",
        )
    ]
    actions = list(
        _sequence(raw.get("action_tokens"), context="HumanDB action_tokens")
    )
    if logical_count != 12 or side_counts != [6, 6] or len(turns) != 12:
        raise LayeredCoreSelectionError("HumanDB prefix target drifted")
    if [token for turn in turns for token in turn] != actions:
        raise LayeredCoreSelectionError("HumanDB logical turns do not flatten")
    if sum(not str(token).startswith("x") for token in actions) != 12:
        raise LayeredCoreSelectionError("HumanDB action history is not 12 ply")

    distinct_games = raw.get("distinct_game_count")
    occurrences = raw.get("occurrence_count")
    if (
        not isinstance(distinct_games, int)
        or distinct_games <= 0
        or not isinstance(occurrences, int)
        or occurrences < distinct_games
    ):
        raise LayeredCoreSelectionError("HumanDB frequency evidence drifted")
    results = dict(_mapping(raw.get("results"), context="HumanDB results"))
    if sum(results.values()) != distinct_games:
        raise LayeredCoreSelectionError("HumanDB result counts drifted")

    final = _mapping(raw.get("final"), context="HumanDB final")
    return {
        "stratum": "human_db",
        "source_subtype": "observed_playok_history",
        "source_audit_identity": source_audit_identity,
        "source_history_id": history_identity,
        "logical_ply_count": 12,
        "logical_plies_by_side": [6, 6],
        "action_tokens": actions,
        "logical_turns": turns,
        "distinct_game_count": distinct_games,
        "occurrence_count": occurrences,
        "results": results,
        "side_roles": dict(
            _mapping(raw.get("side_roles"), context="HumanDB side_roles")
        ),
        "final": {
            "nmm_fen": _string(
                final.get("nmm_fen"),
                context="HumanDB final nmm_fen",
            ),
            "ring16_canonical_fen": _string(
                final.get("ring16_canonical_fen"),
                context="HumanDB final ring16_canonical_fen",
            ),
        },
        "execution_record_status": "full_sanmill_replay_pending",
    }


def derive_human_core(
    *,
    human_audit: Mapping[str, Any],
    book_selection: Mapping[str, Any],
    ledger_header: Mapping[str, Any],
    ledger_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select 21 frequent, structurally distinct genuine PlayOK histories."""

    human_identity = _string(
        human_audit.get("audit_identity"),
        context="HumanDB audit identity",
    )
    raw_source = _mapping(
        human_audit.get("raw_game_source"),
        context="HumanDB raw source",
    )
    audit_ledger = _mapping(
        raw_source.get("history_ledger"),
        context="HumanDB audit ledger",
    )
    expected_header = {
        "schema_version": audit_ledger.get("schema_version"),
        "logical_ply_count": 12,
        "logical_plies_by_side": [6, 6],
        "ordering": (
            "distinct_game_count_desc_then_occurrence_count_desc_then_"
            "history_identity"
        ),
        "record_count": audit_ledger.get("history_count"),
    }
    if dict(ledger_header) != expected_header:
        raise LayeredCoreSelectionError("HumanDB ledger header drifted")

    book_members = _sequence(
        book_selection.get("members"),
        context="Book members",
    )
    seen_histories = {
        tuple(
            _sequence(
                _mapping(raw, context="Book member").get("action_tokens"),
                context="Book action tokens",
            )
        )
        for raw in book_members
    }
    seen_fens = {
        _string(
            _mapping(
                _mapping(raw, context="Book member").get("final"),
                context="Book final",
            ).get("nmm_fen"),
            context="Book final nmm_fen",
        )
        for raw in book_members
    }
    seen_orbits = {
        _string(
            _mapping(
                _mapping(raw, context="Book member").get("final"),
                context="Book final",
            ).get("ring16_canonical_fen"),
            context="Book final ring16",
        )
        for raw in book_members
    }

    members: list[dict[str, Any]] = []
    candidate_window: list[dict[str, Any]] = []
    previous_order: tuple[int, int, str] | None = None
    for ledger_rank, raw_value in enumerate(ledger_records, start=1):
        raw = _mapping(raw_value, context="HumanDB ledger record")
        source = _human_source_record(
            raw,
            source_audit_identity=human_identity,
        )
        order = (
            -source["distinct_game_count"],
            -source["occurrence_count"],
            source["source_history_id"],
        )
        if previous_order is not None and order < previous_order:
            raise LayeredCoreSelectionError("HumanDB ledger order drifted")
        previous_order = order

        history_key = tuple(source["action_tokens"])
        final_fen = source["final"]["nmm_fen"]
        ring16 = source["final"]["ring16_canonical_fen"]
        collisions: list[str] = []
        if history_key in seen_histories:
            collisions.append("exact_history")
        if final_fen in seen_fens:
            collisions.append("final_fen")
        if ring16 in seen_orbits:
            collisions.append("ring16")

        trace = {
            "ledger_rank": ledger_rank,
            "source_record": source,
            "disposition": "skipped_duplicate" if collisions else "selected",
            "collision_dimensions": collisions,
        }
        if not collisions:
            member = {
                "core_id": f"human-core-{len(members) + 1:03d}",
                "ledger_rank": ledger_rank,
                "selection_basis": (
                    "frozen_frequency_order_then_structural_deduplication"
                ),
                **source,
            }
            members.append(member)
            trace["core_id"] = member["core_id"]
            seen_histories.add(history_key)
            seen_fens.add(final_fen)
            seen_orbits.add(ring16)
        candidate_window.append(trace)
        if len(members) == HUMAN_CORE_COUNT:
            break

    if len(members) != HUMAN_CORE_COUNT:
        raise LayeredCoreSelectionError("HumanDB core quota is incomplete")
    if len({tuple(item["action_tokens"]) for item in members}) != 21:
        raise LayeredCoreSelectionError("HumanDB exact-history diversity drifted")
    if len({item["final"]["nmm_fen"] for item in members}) != 21:
        raise LayeredCoreSelectionError("HumanDB final-FEN diversity drifted")
    if len(
        {item["final"]["ring16_canonical_fen"] for item in members}
    ) != 21:
        raise LayeredCoreSelectionError("HumanDB ring16 diversity drifted")

    return {
        "selection_schema": "nmm.layered-opening-prefix-human-core.v1",
        "ledger": {
            "schema_version": audit_ledger["schema_version"],
            "sha256": audit_ledger["sha256"],
            "byte_length": audit_ledger["byte_length"],
            "history_count": audit_ledger["history_count"],
            "path_lookup_key": audit_ledger["path_lookup_key"],
        },
        "selection_rule": expected_header["ordering"],
        "cross_source_precedence": ["book", "human_db", "perfect_db"],
        "candidate_window": candidate_window,
        "candidate_window_identity": canonical_sha256(candidate_window),
        "members": members,
        "membership_identity": canonical_sha256(members),
        "summary": {
            "member_count": len(members),
            "last_selected_ledger_rank": members[-1]["ledger_rank"],
            "minimum_distinct_game_count": min(
                item["distinct_game_count"] for item in members
            ),
            "skipped_before_quota": sum(
                item["disposition"] == "skipped_duplicate"
                for item in candidate_window
            ),
        },
    }


def _selected_structure_sets(
    selections: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[Any, ...]], set[str], set[str]]:
    histories: set[tuple[Any, ...]] = set()
    fens: set[str] = set()
    orbits: set[str] = set()
    for selection in selections:
        for raw in _sequence(selection.get("members"), context="core members"):
            member = _mapping(raw, context="core member")
            histories.add(
                tuple(
                    _sequence(
                        member.get("action_tokens"),
                        context="core action tokens",
                    )
                )
            )
            final = _mapping(member.get("final"), context="core final")
            fens.add(
                _string(final.get("nmm_fen"), context="core final nmm_fen")
            )
            orbits.add(
                _string(
                    final.get("ring16_canonical_fen"),
                    context="core final ring16",
                )
            )
    return histories, fens, orbits


def derive_perfect_core(
    *,
    perfect_audit: Mapping[str, Any],
    book_selection: Mapping[str, Any],
    human_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Take 21 fixed StrictSteps routes after source-order deduplication."""

    perfect_identity = _string(
        perfect_audit.get("audit_identity"),
        context="Perfect DB audit identity",
    )
    generator = _mapping(
        perfect_audit.get("generator"),
        context="Perfect DB generator",
    )
    if generator.get("route_pool_is_final_corpus") is not False:
        raise LayeredCoreSelectionError("Perfect audit pool boundary drifted")
    if generator.get("base_seed") != 42 or generator.get("route_count") != 128:
        raise LayeredCoreSelectionError("Perfect audit route contract drifted")
    overlap = _mapping(
        perfect_audit.get("overlap"),
        context="Perfect DB overlap",
    )
    for source in ("with_book", "with_human_db"):
        counts = _mapping(overlap.get(source), context=f"Perfect {source}")
        if any(counts.get(field) != 0 for field in counts):
            raise LayeredCoreSelectionError(
                f"Perfect audit {source} overlap is no longer zero"
            )

    seen_histories, seen_fens, seen_orbits = _selected_structure_sets(
        [book_selection, human_selection]
    )
    members: list[dict[str, Any]] = []
    candidate_window: list[dict[str, Any]] = []
    routes = _sequence(perfect_audit.get("routes"), context="Perfect routes")
    for route_index, raw_value in enumerate(routes):
        route = _mapping(raw_value, context="Perfect route")
        route_id = _string(route.get("route_id"), context="Perfect route_id")
        expected_route_id = f"perfect-audit-route-{route_index:03d}"
        if route_id != expected_route_id:
            raise LayeredCoreSelectionError("Perfect route order drifted")
        if route.get("route_seed") != 42 + route_index:
            raise LayeredCoreSelectionError("Perfect route seed drifted")
        prefix = _mapping(
            route.get("prefix_record"),
            context="Perfect prefix record",
        )
        history_id, final_fen, ring16 = _prefix_key(prefix)
        if route.get("exact_history_sha256") != history_id:
            raise LayeredCoreSelectionError("Perfect history identity drifted")
        if prefix.get("logical_ply_count") != 12 or list(
            _sequence(
                prefix.get("logical_plies_by_side"),
                context="Perfect logical_plies_by_side",
            )
        ) != [6, 6]:
            raise LayeredCoreSelectionError("Perfect prefix target drifted")
        actions = list(
            _sequence(
                prefix.get("action_tokens"),
                context="Perfect action_tokens",
            )
        )
        steps = [
            _mapping(step, context="Perfect step")
            for step in _sequence(prefix.get("steps"), context="Perfect steps")
        ]
        if len(steps) != 12:
            raise LayeredCoreSelectionError("Perfect step count drifted")
        logical_turns = [
            list(
                _sequence(
                    step.get("action_tokens"),
                    context="Perfect step actions",
                )
            )
            for step in steps
        ]
        if [token for turn in logical_turns for token in turn] != actions:
            raise LayeredCoreSelectionError("Perfect steps do not flatten")

        step_evidence = [
            _mapping(
                step.get("source_evidence"),
                context="Perfect step source evidence",
            )
            for step in steps
        ]
        tied_steps = 0
        pool_sizes: list[int] = []
        for evidence in step_evidence:
            selected_candidate = _mapping(
                evidence.get("selected_candidate"),
                context="Perfect selected candidate",
            )
            perfect = _mapping(
                selected_candidate.get("perfect"),
                context="Perfect selected value",
            )
            if perfect.get("mode") != "strict_steps" or perfect.get("wdl") != 0:
                raise LayeredCoreSelectionError("Perfect selected value drifted")
            query = _mapping(
                evidence.get("query_contract"),
                context="Perfect query contract",
            )
            if query.get("query_mode") != "strict_steps" or query.get(
                "fallback"
            ) != "none":
                raise LayeredCoreSelectionError("Perfect query contract drifted")
            tie = _mapping(
                evidence.get("theoretical_tie"),
                context="Perfect theoretical tie",
            )
            pool_size = tie.get("candidate_count")
            if not isinstance(pool_size, int) or pool_size <= 0:
                raise LayeredCoreSelectionError("Perfect candidate pool drifted")
            pool_sizes.append(pool_size)
            tied_steps += int(tie.get("multiple_tied_best") is True)

        history_key = tuple(actions)
        collisions: list[str] = []
        if history_key in seen_histories:
            collisions.append("exact_history")
        if final_fen in seen_fens:
            collisions.append("final_fen")
        if ring16 in seen_orbits:
            collisions.append("ring16")
        trace = {
            "audit_route_index": route_index,
            "route_id": route_id,
            "route_seed": route["route_seed"],
            "disposition": "skipped_duplicate" if collisions else "selected",
            "collision_dimensions": collisions,
        }
        if not collisions:
            final = _mapping(prefix.get("final"), context="Perfect final")
            member = {
                "core_id": f"perfect-core-{len(members) + 1:03d}",
                "stratum": "perfect_db",
                "source_subtype": _string(
                    prefix.get("source_subtype"),
                    context="Perfect source_subtype",
                ),
                "source_audit_identity": perfect_identity,
                "route_id": route_id,
                "route_seed": route["route_seed"],
                "source_history_id": history_id,
                "prefix_identity": _string(
                    prefix.get("prefix_identity"),
                    context="Perfect prefix_identity",
                ),
                "logical_ply_count": 12,
                "logical_plies_by_side": [6, 6],
                "action_tokens": actions,
                "logical_turns": logical_turns,
                "strict_steps_trace_identity": canonical_sha256(
                    step_evidence
                ),
                "theory_summary": {
                    "selected_wdl": 0,
                    "selected_category": "draw",
                    "tied_best_step_count": tied_steps,
                    "single_best_step_count": 12 - tied_steps,
                    "minimum_candidate_pool_size": min(pool_sizes),
                    "maximum_candidate_pool_size": max(pool_sizes),
                },
                "final": {
                    "nmm_fen": final_fen,
                    "ring16_canonical_fen": ring16,
                    "history_sha256": _string(
                        final.get("history_sha256"),
                        context="Perfect final history_sha256",
                    ),
                },
                "selection_basis": (
                    "frozen_audit_route_order_then_structural_deduplication"
                ),
                "execution_record_status": "frozen_source_prefix_available",
            }
            members.append(member)
            trace["core_id"] = member["core_id"]
            seen_histories.add(history_key)
            seen_fens.add(final_fen)
            seen_orbits.add(ring16)
        candidate_window.append(trace)
        if len(members) == PERFECT_CORE_COUNT:
            break

    if len(members) != PERFECT_CORE_COUNT:
        raise LayeredCoreSelectionError("Perfect DB core quota is incomplete")
    if len({tuple(item["action_tokens"]) for item in members}) != 21:
        raise LayeredCoreSelectionError("Perfect exact-history diversity drifted")
    if len({item["final"]["nmm_fen"] for item in members}) != 21:
        raise LayeredCoreSelectionError("Perfect final-FEN diversity drifted")
    if len(
        {item["final"]["ring16_canonical_fen"] for item in members}
    ) != 21:
        raise LayeredCoreSelectionError("Perfect ring16 diversity drifted")

    return {
        "selection_schema": "nmm.layered-opening-prefix-perfect-core.v1",
        "route_pool_audit_identity": perfect_identity,
        "selection_rule": (
            "frozen_route_order_then_exact_history_final_fen_ring16_dedup"
        ),
        "candidate_window": candidate_window,
        "candidate_window_identity": canonical_sha256(candidate_window),
        "members": members,
        "membership_identity": canonical_sha256(members),
        "summary": {
            "member_count": len(members),
            "last_selected_audit_route_index": candidate_window[-1][
                "audit_route_index"
            ],
            "skipped_before_quota": sum(
                item["disposition"] == "skipped_duplicate"
                for item in candidate_window
            ),
            "tied_best_step_count": sum(
                item["theory_summary"]["tied_best_step_count"]
                for item in members
            ),
            "single_best_step_count": sum(
                item["theory_summary"]["single_best_step_count"]
                for item in members
            ),
        },
    }


def build_layered_source_core(
    *,
    composition_decision: Mapping[str, Any],
    book_decision: Mapping[str, Any],
    human_decision: Mapping[str, Any],
    perfect_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine the three frozen source memberships without making them runnable."""

    composition = _mapping(
        composition_decision.get("composition"),
        context="composition decision",
    )
    expected_strata = [
        {"stratum": "book", "count": 22},
        {"stratum": "human_db", "count": 21},
        {"stratum": "perfect_db", "count": 21},
    ]
    if composition.get("total_prefixes") != 64 or composition.get(
        "strata"
    ) != expected_strata:
        raise LayeredCoreSelectionError("accepted composition drifted")
    if composition_decision.get("composition_identity") != canonical_sha256(
        composition
    ):
        raise LayeredCoreSelectionError("composition identity drifted")

    source_decisions = [
        ("book", 22, book_decision),
        ("human_db", 21, human_decision),
        ("perfect_db", 21, perfect_decision),
    ]
    records: list[dict[str, Any]] = []
    for stratum, count, decision in source_decisions:
        selection = _mapping(
            decision.get("selection"),
            context=f"{stratum} selection",
        )
        members = _sequence(
            selection.get("members"),
            context=f"{stratum} members",
        )
        if len(members) != count:
            raise LayeredCoreSelectionError(f"{stratum} quota drifted")
        for raw in members:
            member = dict(_mapping(raw, context=f"{stratum} member"))
            stratum_member_id = _string(
                member.pop("core_id", None),
                context=f"{stratum} core_id",
            )
            if member.get("stratum") != stratum:
                raise LayeredCoreSelectionError(f"{stratum} label drifted")
            if stratum == "book":
                execution_status = "frozen_source_prefix_available"
            else:
                execution_status = _string(
                    member.get("execution_record_status"),
                    context=f"{stratum} execution status",
                )
            records.append(
                {
                    "source_core_id": f"source-core-{len(records) + 1:03d}",
                    "stratum_member_id": stratum_member_id,
                    **member,
                    "execution_record_status": execution_status,
                }
            )

    histories = {tuple(item["action_tokens"]) for item in records}
    fens = {item["final"]["nmm_fen"] for item in records}
    orbits = {item["final"]["ring16_canonical_fen"] for item in records}
    if len(records) != 64 or len(histories) != 64:
        raise LayeredCoreSelectionError("combined exact-history diversity drifted")
    if len(fens) != 64 or len(orbits) != 64:
        raise LayeredCoreSelectionError("combined structural diversity drifted")
    if any(not str(fen).endswith("|W|6|6") for fen in fens):
        raise LayeredCoreSelectionError("combined side/count boundary drifted")

    status_counts: dict[str, int] = {}
    for item in records:
        status = item["execution_record_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    expected_status = {
        "frozen_source_prefix_available": 43,
        "full_sanmill_replay_pending": 21,
    }
    if status_counts != expected_status:
        raise LayeredCoreSelectionError("execution-record status drifted")

    source_inputs = {
        "composition_identity": composition_decision["composition_identity"],
        "book_membership_identity": book_decision["selection"][
            "membership_identity"
        ],
        "human_db_membership_identity": human_decision["selection"][
            "membership_identity"
        ],
        "perfect_db_membership_identity": perfect_decision["selection"][
            "membership_identity"
        ],
    }
    return {
        "source_core_schema": "nmm.layered-opening-prefix-source-core.v1",
        "composition": {
            "total": 64,
            "book": 22,
            "human_db": 21,
            "perfect_db": 21,
        },
        "source_inputs": source_inputs,
        "source_inputs_identity": canonical_sha256(source_inputs),
        "records": records,
        "source_membership_identity": canonical_sha256(records),
        "summary": {
            "record_count": 64,
            "unique_exact_history_count": 64,
            "unique_final_fen_count": 64,
            "unique_ring16_count": 64,
            "side_to_move": "white",
            "logical_ply_count": 12,
            "logical_plies_by_side": [6, 6],
            "execution_record_status_counts": status_counts,
        },
    }
