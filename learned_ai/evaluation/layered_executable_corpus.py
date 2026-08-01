"""Assemble the frozen 64-member twelve-ply executable prefix corpus."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from learned_ai.evaluation.layered_book_audit import (
    verify_layered_book_audit,
)
from learned_ai.evaluation.layered_core_selection import (
    build_layered_source_core,
)
from learned_ai.evaluation.layered_expert_book_audit import (
    verify_layered_expert_book_audit,
)
from learned_ai.evaluation.layered_human_audit import (
    verify_layered_human_audit,
)
from learned_ai.evaluation.layered_human_execution import (
    verify_layered_human_execution,
)
from learned_ai.evaluation.layered_opening_prefix import (
    LayeredOpeningPrefixV2,
)
from learned_ai.evaluation.layered_perfect_audit import (
    verify_layered_perfect_audit,
)
from learned_ai.training.run_contract import canonical_sha256


EXECUTABLE_CORPUS_SCHEMA = "nmm.layered-opening-prefix-executable-corpus.v1"
EXECUTABLE_RECORDS_SCHEMA = (
    "nmm.layered-opening-prefix-executable-corpus-records.v1"
)
EXECUTABLE_CORPUS_STATUS = (
    "executable_64_prefix_corpus_frozen_evaluation_not_authorized"
)
EXECUTABLE_CORPUS_COUNT = 64
EXECUTABLE_STRATUM_COUNTS = {
    "book": 22,
    "human_db": 21,
    "perfect_db": 21,
}


class LayeredExecutableCorpusError(ValueError):
    """Raised when a frozen source record cannot enter the executable core."""


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayeredExecutableCorpusError(f"{context} must be an object")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LayeredExecutableCorpusError(f"{context} must be an array")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LayeredExecutableCorpusError(f"{context} is not a SHA-256")
    return value


def _identity_without_field(
    payload: Mapping[str, Any],
    *,
    field: str,
    context: str,
) -> str:
    body = dict(payload)
    identity = _sha256(body.pop(field, None), context=f"{context} identity")
    if identity != canonical_sha256(body):
        raise LayeredExecutableCorpusError(f"{context} identity drifted")
    return identity


def _membership_identity(
    decision: Mapping[str, Any],
    *,
    context: str,
) -> str:
    selection = _mapping(decision.get("selection"), context=f"{context} selection")
    members = list(
        _sequence(selection.get("members"), context=f"{context} members")
    )
    identity = _sha256(
        selection.get("membership_identity"),
        context=f"{context} membership identity",
    )
    if identity != canonical_sha256(members):
        raise LayeredExecutableCorpusError(f"{context} membership drifted")
    return identity


def _validate_inputs(
    *,
    composition_decision: Mapping[str, Any],
    book_core_decision: Mapping[str, Any],
    human_core_decision: Mapping[str, Any],
    perfect_core_decision: Mapping[str, Any],
    source_core_decision: Mapping[str, Any],
    sanmill_book_audit: Mapping[str, Any],
    expert_book_audit: Mapping[str, Any],
    human_audit: Mapping[str, Any],
    perfect_audit: Mapping[str, Any],
    human_execution: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    verify_layered_book_audit(sanmill_book_audit)
    verify_layered_expert_book_audit(expert_book_audit)
    verify_layered_human_audit(human_audit)
    verify_layered_perfect_audit(perfect_audit)
    verify_layered_human_execution(
        human_execution,
        human_core_decision=human_core_decision,
        source_core_decision=source_core_decision,
        human_audit=human_audit,
        runtime_decision=runtime_decision,
    )

    rebuilt_source_core = build_layered_source_core(
        composition_decision=composition_decision,
        book_decision=book_core_decision,
        human_decision=human_core_decision,
        perfect_decision=perfect_core_decision,
    )
    source_core = _mapping(
        source_core_decision.get("source_core"),
        context="source core",
    )
    if rebuilt_source_core != source_core:
        raise LayeredExecutableCorpusError("source core cannot be reproduced")
    source_records = list(
        _sequence(source_core.get("records"), context="source-core records")
    )
    if len(source_records) != EXECUTABLE_CORPUS_COUNT:
        raise LayeredExecutableCorpusError("source core does not contain 64 rows")

    composition = _mapping(
        composition_decision.get("composition"),
        context="composition",
    )
    composition_identity = _sha256(
        composition_decision.get("composition_identity"),
        context="composition identity",
    )
    if composition_identity != canonical_sha256(composition):
        raise LayeredExecutableCorpusError("composition identity drifted")

    input_identities = {
        "composition_identity": composition_identity,
        "book_core_membership_identity": _membership_identity(
            book_core_decision,
            context="Book core",
        ),
        "human_core_membership_identity": _membership_identity(
            human_core_decision,
            context="HumanDB core",
        ),
        "perfect_core_membership_identity": _membership_identity(
            perfect_core_decision,
            context="Perfect DB core",
        ),
        "source_core_membership_identity": _sha256(
            source_core.get("source_membership_identity"),
            context="source-core membership identity",
        ),
        "sanmill_book_audit_identity": _sha256(
            sanmill_book_audit.get("audit_identity"),
            context="Sanmill Book audit identity",
        ),
        "expert_book_audit_identity": _sha256(
            expert_book_audit.get("audit_identity"),
            context="expert Book audit identity",
        ),
        "human_audit_identity": _sha256(
            human_audit.get("audit_identity"),
            context="HumanDB audit identity",
        ),
        "perfect_audit_identity": _sha256(
            perfect_audit.get("audit_identity"),
            context="Perfect DB audit identity",
        ),
        "human_execution_identity": _sha256(
            human_execution.get("human_execution_identity"),
            context="HumanDB execution identity",
        ),
        "human_replay_runtime_identity": _identity_without_field(
            runtime_decision,
            field="runtime_identity",
            context="HumanDB replay runtime",
        ),
    }
    if source_core.get("source_inputs_identity") != canonical_sha256(
        source_core.get("source_inputs")
    ):
        raise LayeredExecutableCorpusError("source-core input identity drifted")
    return source_records, input_identities


def _index_prefixes(
    prefixes: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_prefix in prefixes:
        prefix_payload = dict(raw_prefix)
        prefix_payload.pop("exact_history_sha256", None)
        prefix = LayeredOpeningPrefixV2.from_dict(prefix_payload)
        if prefix.prefix_identity in result:
            raise LayeredExecutableCorpusError(
                f"duplicate prefix identity in {context}"
            )
        result[prefix.prefix_identity] = prefix.to_dict()
    return result


def _source_indexes(
    *,
    sanmill_book_audit: Mapping[str, Any],
    expert_book_audit: Mapping[str, Any],
    perfect_audit: Mapping[str, Any],
    human_execution: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    named_book = _mapping(
        sanmill_book_audit.get("named_book_variations"),
        context="named Book variations",
    )
    named_entries = _sequence(
        named_book.get("entries"),
        context="named Book entries",
    )
    book_prefixes = [
        _mapping(prefix, context="named Book prefix")
        for entry in named_entries
        for prefix in _sequence(
            _mapping(entry, context="named Book entry").get("prefix_records"),
            context="named Book prefix records",
        )
    ]
    expert_records = _sequence(
        expert_book_audit.get("records"),
        context="expert Book records",
    )
    book_prefixes.extend(
        _mapping(record, context="expert Book record").get("prefix_record")
        for record in expert_records
    )
    book_index = _index_prefixes(book_prefixes, context="Book audits")

    perfect_routes = _sequence(
        perfect_audit.get("routes"),
        context="Perfect DB routes",
    )
    perfect_index = _index_prefixes(
        [
            _mapping(route, context="Perfect DB route").get("prefix_record")
            for route in perfect_routes
        ],
        context="Perfect DB audit",
    )

    human_records = _sequence(
        human_execution.get("records"),
        context="HumanDB execution records",
    )
    human_index: dict[str, Mapping[str, Any]] = {}
    for raw_record in human_records:
        record = _mapping(raw_record, context="HumanDB execution record")
        source_core_id = record.get("source_core_id")
        if not isinstance(source_core_id, str) or source_core_id in human_index:
            raise LayeredExecutableCorpusError(
                "HumanDB execution source-core binding drifted"
            )
        human_index[source_core_id] = record
    return book_index, perfect_index, human_index


def _assert_source_binding(
    source_record: Mapping[str, Any],
    prefix: LayeredOpeningPrefixV2,
) -> None:
    final = _mapping(source_record.get("final"), context="source-core final")
    if (
        prefix.stratum != source_record.get("stratum")
        or prefix.source_subtype != source_record.get("source_subtype")
        or prefix.source_history_id != source_record.get("source_history_id")
        or list(prefix.action_tokens) != source_record.get("action_tokens")
        or len(prefix.steps) != source_record.get("logical_ply_count")
        or source_record.get("logical_plies_by_side") != [6, 6]
        or prefix.final_nmm_fen != final.get("nmm_fen")
        or prefix.final_ring16_fen != final.get("ring16_canonical_fen")
    ):
        raise LayeredExecutableCorpusError("source-to-execution binding drifted")
    source_prefix_identity = source_record.get("prefix_identity")
    if (
        source_prefix_identity is not None
        and prefix.prefix_identity != source_prefix_identity
    ):
        raise LayeredExecutableCorpusError("source prefix identity drifted")
    source_history_identity = final.get("history_sha256")
    if (
        source_history_identity is not None
        and prefix.final_history_sha256 != source_history_identity
    ):
        raise LayeredExecutableCorpusError("source final history drifted")


def _build_records(
    *,
    source_records: Sequence[Mapping[str, Any]],
    sanmill_book_audit: Mapping[str, Any],
    expert_book_audit: Mapping[str, Any],
    perfect_audit: Mapping[str, Any],
    human_execution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    book_index, perfect_index, human_index = _source_indexes(
        sanmill_book_audit=sanmill_book_audit,
        expert_book_audit=expert_book_audit,
        perfect_audit=perfect_audit,
        human_execution=human_execution,
    )
    records: list[dict[str, Any]] = []
    for index, raw_source_record in enumerate(source_records, start=1):
        source_record = _mapping(raw_source_record, context="source-core record")
        source_core_id = source_record.get("source_core_id")
        if source_core_id != f"source-core-{index:03d}":
            raise LayeredExecutableCorpusError("source-core order drifted")
        stratum = source_record.get("stratum")
        if stratum == "book":
            raw_prefix = book_index.get(source_record.get("prefix_identity"))
            execution_origin = "frozen_book_source_audit"
        elif stratum == "perfect_db":
            raw_prefix = perfect_index.get(source_record.get("prefix_identity"))
            execution_origin = "frozen_perfect_db_source_audit"
        elif stratum == "human_db":
            human_record = human_index.get(source_core_id)
            raw_prefix = (
                human_record.get("execution_record") if human_record else None
            )
            execution_origin = "frozen_human_db_execution_overlay"
        else:
            raise LayeredExecutableCorpusError("unknown source-core stratum")
        if not isinstance(raw_prefix, Mapping):
            raise LayeredExecutableCorpusError(
                f"missing execution record for {source_core_id}"
            )
        prefix = LayeredOpeningPrefixV2.from_dict(raw_prefix)
        _assert_source_binding(source_record, prefix)
        body = {
            "corpus_id": f"layered-prefix-v2-{index:03d}",
            "source_core_id": source_core_id,
            "stratum_member_id": source_record.get("stratum_member_id"),
            "stratum": stratum,
            "source_subtype": source_record.get("source_subtype"),
            "source_history_id": source_record.get("source_history_id"),
            "execution_origin": execution_origin,
            "execution_record": prefix.to_dict(),
        }
        records.append({**body, "record_identity": canonical_sha256(body)})
    return records


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prefixes = [
        LayeredOpeningPrefixV2.from_dict(record.get("execution_record"))
        for record in records
    ]
    strata = Counter(prefix.stratum for prefix in prefixes)
    subtypes = Counter(prefix.source_subtype for prefix in prefixes)
    origins = Counter(record.get("execution_origin") for record in records)
    runtime_records: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        portable = dict(prefix.sanmill)
        identity = canonical_sha256(portable)
        if identity not in runtime_records:
            runtime_records[identity] = {
                "runtime_record_identity": identity,
                "record_count": 0,
                "commit": portable.get("commit"),
                "checkout_head": portable.get("checkout_head"),
                "tree": portable.get("tree"),
                "binary_sha256": portable.get("binary_sha256"),
                "path_lookup_key": portable.get("path_lookup_key"),
                "strict_failure_protocol_version": portable.get(
                    "strict_failure_protocol_version"
                ),
            }
        runtime_records[identity]["record_count"] += 1

    summary = {
        "record_count": len(records),
        "stratum_counts": dict(sorted(strata.items())),
        "source_subtype_counts": dict(sorted(subtypes.items())),
        "execution_origin_counts": dict(sorted(origins.items())),
        "logical_ply_count_per_record": 12,
        "logical_plies_by_side": [6, 6],
        "total_logical_ply_count": sum(len(prefix.steps) for prefix in prefixes),
        "compound_turn_count": sum(
            len(step.action_tokens) == 2
            for prefix in prefixes
            for step in prefix.steps
        ),
        "unique_prefix_identity_count": len(
            {prefix.prefix_identity for prefix in prefixes}
        ),
        "unique_source_history_count": len(
            {prefix.source_history_id for prefix in prefixes}
        ),
        "unique_action_history_count": len(
            {prefix.action_tokens for prefix in prefixes}
        ),
        "unique_final_fen_count": len(
            {prefix.final_nmm_fen for prefix in prefixes}
        ),
        "unique_ring16_count": len(
            {prefix.final_ring16_fen for prefix in prefixes}
        ),
        "unique_final_history_identity_count": len(
            {prefix.final_history_sha256 for prefix in prefixes}
        ),
        "sanmill_runtime_records": sorted(
            runtime_records.values(),
            key=lambda item: item["runtime_record_identity"],
        ),
    }
    if summary["record_count"] != EXECUTABLE_CORPUS_COUNT:
        raise LayeredExecutableCorpusError("executable record count drifted")
    if summary["stratum_counts"] != EXECUTABLE_STRATUM_COUNTS:
        raise LayeredExecutableCorpusError("executable stratum counts drifted")
    for key in (
        "unique_prefix_identity_count",
        "unique_source_history_count",
        "unique_action_history_count",
        "unique_final_fen_count",
        "unique_ring16_count",
        "unique_final_history_identity_count",
    ):
        if summary[key] != EXECUTABLE_CORPUS_COUNT:
            raise LayeredExecutableCorpusError(f"{key} is not 64")
    if summary["total_logical_ply_count"] != 768:
        raise LayeredExecutableCorpusError("total logical-ply count drifted")
    if sorted(
        record["record_count"] for record in summary["sanmill_runtime_records"]
    ) != [21, 43]:
        raise LayeredExecutableCorpusError("Sanmill runtime grouping drifted")
    return summary


def build_layered_executable_corpus(
    *,
    composition_decision: Mapping[str, Any],
    book_core_decision: Mapping[str, Any],
    human_core_decision: Mapping[str, Any],
    perfect_core_decision: Mapping[str, Any],
    source_core_decision: Mapping[str, Any],
    sanmill_book_audit: Mapping[str, Any],
    expert_book_audit: Mapping[str, Any],
    human_audit: Mapping[str, Any],
    perfect_audit: Mapping[str, Any],
    human_execution: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine the frozen source memberships and execution records."""
    source_records, input_identities = _validate_inputs(
        composition_decision=composition_decision,
        book_core_decision=book_core_decision,
        human_core_decision=human_core_decision,
        perfect_core_decision=perfect_core_decision,
        source_core_decision=source_core_decision,
        sanmill_book_audit=sanmill_book_audit,
        expert_book_audit=expert_book_audit,
        human_audit=human_audit,
        perfect_audit=perfect_audit,
        human_execution=human_execution,
        runtime_decision=runtime_decision,
    )
    records = _build_records(
        source_records=source_records,
        sanmill_book_audit=sanmill_book_audit,
        expert_book_audit=expert_book_audit,
        perfect_audit=perfect_audit,
        human_execution=human_execution,
    )
    corpus = {
        "records_schema": EXECUTABLE_RECORDS_SCHEMA,
        "composition": dict(EXECUTABLE_STRATUM_COUNTS),
        "records": records,
        "records_identity": canonical_sha256(records),
        "summary": _summary(records),
    }
    identity_body = {
        "input_identities": input_identities,
        "input_identities_identity": canonical_sha256(input_identities),
        "corpus": corpus,
    }
    return {
        "schema_version": EXECUTABLE_CORPUS_SCHEMA,
        "status": EXECUTABLE_CORPUS_STATUS,
        "decision_date": "2026-08-01",
        "candidate_loaded": False,
        "games_played": 0,
        "fallback": "none",
        **identity_body,
        "executable_corpus_identity": canonical_sha256(identity_body),
        "decision": {
            "source_membership_unchanged": True,
            "executable_64_prefix_corpus_frozen": True,
            "evaluation_authorized": False,
            "training_authorized": False,
        },
    }


def verify_layered_executable_corpus(
    payload: Mapping[str, Any],
    *,
    composition_decision: Mapping[str, Any],
    book_core_decision: Mapping[str, Any],
    human_core_decision: Mapping[str, Any],
    perfect_core_decision: Mapping[str, Any],
    source_core_decision: Mapping[str, Any],
    sanmill_book_audit: Mapping[str, Any],
    expert_book_audit: Mapping[str, Any],
    human_audit: Mapping[str, Any],
    perfect_audit: Mapping[str, Any],
    human_execution: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the frozen executable corpus from tracked source evidence."""
    expected = build_layered_executable_corpus(
        composition_decision=composition_decision,
        book_core_decision=book_core_decision,
        human_core_decision=human_core_decision,
        perfect_core_decision=perfect_core_decision,
        source_core_decision=source_core_decision,
        sanmill_book_audit=sanmill_book_audit,
        expert_book_audit=expert_book_audit,
        human_audit=human_audit,
        perfect_audit=perfect_audit,
        human_execution=human_execution,
        runtime_decision=runtime_decision,
    )
    if dict(payload) != expected:
        raise LayeredExecutableCorpusError("executable corpus drifted")
    return dict(expected["corpus"]["summary"])
