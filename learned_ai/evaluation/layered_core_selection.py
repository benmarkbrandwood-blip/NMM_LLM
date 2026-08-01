"""Deterministic source-only selection for the twelve-ply balanced core."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from learned_ai.training.run_contract import canonical_sha256


BOOK_CORE_COUNT = 22
EXPERT_BOOK_CORE_COUNT = 15
SANMILL_BOOK_CORE_COUNT = 7


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
