"""Strict Sanmill execution records for the frozen HumanDB prefix core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from learned_ai.evaluation.layered_human_audit import (
    verify_layered_human_audit,
)
from learned_ai.evaluation.layered_opening_prefix import (
    LayeredOpeningPrefixV2,
    build_layered_prefix_v2,
)
from learned_ai.evaluation.sanmill_data_query import SanmillDataQuerySession
from learned_ai.evaluation.sanmill_uci import (
    EXPECTED_SANMILL_LICENSE_SHA256,
    PINNED_SANMILL_COMMIT,
    PINNED_SANMILL_TREE,
    PREFIX12_REPLAY_INSTALLATION_CONTRACT,
    SanmillInstallation,
)
from learned_ai.training.run_contract import canonical_sha256


HUMAN_EXECUTION_SCHEMA = "nmm.layered-opening-prefix-human-execution.v1"
HUMAN_EXECUTION_STATUS = "human_execution_frozen_executable_corpus_pending"
HUMAN_EXECUTION_PROCESS_COUNT = 2
HUMAN_EXECUTION_RECORD_COUNT = 21


class LayeredHumanExecutionError(RuntimeError):
    """Raised when frozen HumanDB membership cannot be replayed exactly."""


def _portable_replay_installation() -> SanmillInstallation:
    contract = PREFIX12_REPLAY_INSTALLATION_CONTRACT
    return SanmillInstallation(
        checkout=Path("portable-verification-only"),
        commit=PINNED_SANMILL_COMMIT,
        checkout_head=PINNED_SANMILL_COMMIT,
        tree=PINNED_SANMILL_TREE,
        binary=Path("target/release/tgf.exe"),
        binary_sha256=contract.expected_binary_sha256,
        binary_size=contract.expected_binary_size,
        license_sha256=EXPECTED_SANMILL_LICENSE_SHA256,
        path_lookup_key=contract.path_lookup_key,
        require_exact_head=True,
    )


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayeredHumanExecutionError(f"{context} must be an object")
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise LayeredHumanExecutionError(f"{context} must be an array")
    return value


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise LayeredHumanExecutionError(f"{context} must be a string")
    return value


def _sha256(value: Any, *, context: str) -> str:
    text = _string(value, context=context)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise LayeredHumanExecutionError(f"{context} is not a SHA-256")
    return text


def _identity_without_field(
    payload: Mapping[str, Any],
    *,
    field: str,
    context: str,
) -> str:
    body = dict(payload)
    identity = _sha256(body.pop(field, None), context=f"{context} identity")
    if identity != canonical_sha256(body):
        raise LayeredHumanExecutionError(f"{context} identity drifted")
    return identity


def _validate_inputs(
    *,
    human_core_decision: Mapping[str, Any],
    source_core_decision: Mapping[str, Any],
    human_audit: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
    installation: SanmillInstallation,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    Mapping[str, Any],
    str,
]:
    verify_layered_human_audit(human_audit)
    audit_identity = _sha256(
        human_audit.get("audit_identity"),
        context="HumanDB audit identity",
    )
    source_identity = _mapping(
        human_audit.get("source_identity"),
        context="HumanDB source identity",
    )
    if source_identity.get("kind") != "human_db":
        raise LayeredHumanExecutionError("HumanDB source kind drifted")
    _sha256(
        source_identity.get("identity_sha256"),
        context="HumanDB source portable identity",
    )

    selection = _mapping(
        human_core_decision.get("selection"),
        context="HumanDB core selection",
    )
    members = list(
        _sequence(selection.get("members"), context="HumanDB core members")
    )
    membership_identity = _sha256(
        selection.get("membership_identity"),
        context="HumanDB membership identity",
    )
    if (
        len(members) != HUMAN_EXECUTION_RECORD_COUNT
        or membership_identity != canonical_sha256(members)
    ):
        raise LayeredHumanExecutionError("HumanDB frozen membership drifted")
    if any(
        member.get("source_audit_identity") != audit_identity
        or member.get("execution_record_status")
        != "full_sanmill_replay_pending"
        for member in members
    ):
        raise LayeredHumanExecutionError("HumanDB member provenance drifted")

    source_core = _mapping(
        source_core_decision.get("source_core"),
        context="source core",
    )
    source_records = list(
        _sequence(source_core.get("records"), context="source-core records")
    )
    if _sha256(
        source_core.get("source_membership_identity"),
        context="source-core identity",
    ) != canonical_sha256(source_records):
        raise LayeredHumanExecutionError("source-core membership drifted")
    human_source_records = [
        record for record in source_records if record.get("stratum") == "human_db"
    ]
    if len(human_source_records) != HUMAN_EXECUTION_RECORD_COUNT:
        raise LayeredHumanExecutionError("source core does not contain 21 HumanDB rows")
    for member, source_record in zip(members, human_source_records, strict=True):
        if (
            source_record.get("stratum_member_id") != member.get("core_id")
            or source_record.get("source_history_id")
            != member.get("source_history_id")
            or source_record.get("action_tokens") != member.get("action_tokens")
            or source_record.get("final") != member.get("final")
        ):
            raise LayeredHumanExecutionError(
                "HumanDB source-core binding drifted"
            )

    runtime_identity = _identity_without_field(
        runtime_decision,
        field="runtime_identity",
        context="replay runtime",
    )
    runtime_source = _mapping(
        runtime_decision.get("source"),
        context="replay runtime source",
    )
    runtime_binary = _mapping(
        runtime_decision.get("binary"),
        context="replay runtime binary",
    )
    contract = PREFIX12_REPLAY_INSTALLATION_CONTRACT
    if (
        runtime_decision.get("status")
        != "pinned_for_source_only_human_history_replay"
        or runtime_source.get("commit") != PINNED_SANMILL_COMMIT
        or runtime_source.get("tree") != PINNED_SANMILL_TREE
        or runtime_source.get("path_lookup_key") != contract.path_lookup_key
        or runtime_binary.get("sha256") != contract.expected_binary_sha256
        or runtime_binary.get("byte_length") != contract.expected_binary_size
        or installation.commit != PINNED_SANMILL_COMMIT
        or installation.checkout_head != PINNED_SANMILL_COMMIT
        or installation.tree != PINNED_SANMILL_TREE
        or installation.binary_sha256 != contract.expected_binary_sha256
        or installation.binary_size != contract.expected_binary_size
        or installation.path_lookup_key != contract.path_lookup_key
        or not installation.require_exact_head
    ):
        raise LayeredHumanExecutionError("replay runtime does not match the pin")
    return members, human_source_records, source_identity, runtime_identity


def _source_evidence(
    member: Mapping[str, Any],
    *,
    membership_identity: str,
    runtime_identity: str,
) -> dict[str, Any]:
    return {
        "kind": "observed_complete_playok_history",
        "interpretation": "frequent_in_the_frozen_playok_sample_only",
        "human_core_id": member["core_id"],
        "source_audit_identity": member["source_audit_identity"],
        "membership_identity": membership_identity,
        "replay_runtime_identity": runtime_identity,
        "ledger_rank": member["ledger_rank"],
        "distinct_game_count": member["distinct_game_count"],
        "occurrence_count": member["occurrence_count"],
        "results": dict(member["results"]),
        "side_roles": dict(member["side_roles"]),
    }


def _step_evidence(member: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns = _sequence(member.get("logical_turns"), context="logical turns")
    return [
        {
            "kind": "observed_complete_history_step",
            "human_core_id": member["core_id"],
            "source_history_id": member["source_history_id"],
            "logical_ply": logical_ply,
            "action_tokens": list(turn),
        }
        for logical_ply, turn in enumerate(turns)
    ]


def _run_once(
    *,
    installation: SanmillInstallation,
    members: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
    membership_identity: str,
    runtime_identity: str,
    session_factory: Callable[
        [SanmillInstallation], SanmillDataQuerySession
    ],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    with session_factory(installation) as session:
        for member, source_record in zip(members, source_records, strict=True):
            prefix = build_layered_prefix_v2(
                session,
                installation,
                stratum="human_db",
                source_subtype="observed_playok_history",
                source_history_id=member["source_history_id"],
                source_identity=source_identity,
                source_evidence=_source_evidence(
                    member,
                    membership_identity=membership_identity,
                    runtime_identity=runtime_identity,
                ),
                logical_turns=member["logical_turns"],
                step_evidence=_step_evidence(member),
            )
            if (
                list(prefix.action_tokens) != member["action_tokens"]
                or prefix.final_nmm_fen != member["final"]["nmm_fen"]
                or prefix.final_ring16_fen
                != member["final"]["ring16_canonical_fen"]
            ):
                raise LayeredHumanExecutionError(
                    f"strict replay disagrees for {member['core_id']}"
                )
            records.append(
                {
                    "human_core_id": member["core_id"],
                    "source_core_id": source_record["source_core_id"],
                    "source_history_id": member["source_history_id"],
                    "execution_record": prefix.to_dict(),
                }
            )
    return records, [dict(item) for item in session.transcript]


def build_layered_human_execution(
    *,
    human_core_decision: Mapping[str, Any],
    source_core_decision: Mapping[str, Any],
    human_audit: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
    installation: SanmillInstallation,
    session_factory: Callable[
        [SanmillInstallation], SanmillDataQuerySession
    ] = SanmillDataQuerySession,
) -> dict[str, Any]:
    """Replay all frozen HumanDB histories in two fresh strict processes."""
    members, source_records, source_identity, runtime_identity = _validate_inputs(
        human_core_decision=human_core_decision,
        source_core_decision=source_core_decision,
        human_audit=human_audit,
        runtime_decision=runtime_decision,
        installation=installation,
    )
    membership_identity = human_core_decision["selection"][
        "membership_identity"
    ]
    runs = [
        _run_once(
            installation=installation,
            members=members,
            source_records=source_records,
            source_identity=source_identity,
            membership_identity=membership_identity,
            runtime_identity=runtime_identity,
            session_factory=session_factory,
        )
        for _process_index in range(HUMAN_EXECUTION_PROCESS_COUNT)
    ]
    if runs[0] != runs[1]:
        raise LayeredHumanExecutionError(
            "HumanDB execution records differ across fresh processes"
        )
    records, transcript = runs[0]
    request_count = sum(
        item.get("direction") == "to_engine" for item in transcript
    )
    response_count = sum(
        item.get("direction") == "from_engine" for item in transcript
    )
    if (
        request_count != 273
        or response_count != 273
        or len(transcript) != 546
    ):
        raise LayeredHumanExecutionError("HumanDB replay transcript count drifted")
    data_query = {
        "request_count": request_count,
        "response_count": response_count,
        "transcript_line_count": len(transcript),
        "transcript_identity": canonical_sha256(transcript),
        "cross_process_comparison": "exact_ordered_transcript_equality",
    }
    execution_evidence = {"records": records, "data_query": data_query}
    return {
        "schema_version": HUMAN_EXECUTION_SCHEMA,
        "status": HUMAN_EXECUTION_STATUS,
        "decision_date": "2026-08-01",
        "candidate_loaded": False,
        "games_played": 0,
        "fallback": "none",
        "human_core_membership_identity": membership_identity,
        "source_core_membership_identity": source_core_decision["source_core"][
            "source_membership_identity"
        ],
        "replay_runtime_identity": runtime_identity,
        "fresh_process_count": HUMAN_EXECUTION_PROCESS_COUNT,
        "data_query": data_query,
        "records": records,
        "human_execution_identity": canonical_sha256(execution_evidence),
        "decision": {
            "human_execution_records_frozen": True,
            "executable_64_prefix_corpus_frozen": False,
            "evaluation_authorized": False,
            "training_authorized": False,
        },
    }


def verify_layered_human_execution(
    payload: Mapping[str, Any],
    *,
    human_core_decision: Mapping[str, Any],
    source_core_decision: Mapping[str, Any],
    human_audit: Mapping[str, Any],
    runtime_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify frozen records without requiring machine-local source data."""
    if (
        payload.get("schema_version") != HUMAN_EXECUTION_SCHEMA
        or payload.get("status") != HUMAN_EXECUTION_STATUS
        or payload.get("candidate_loaded") is not False
        or payload.get("games_played") != 0
        or payload.get("fallback") != "none"
        or payload.get("fresh_process_count")
        != HUMAN_EXECUTION_PROCESS_COUNT
    ):
        raise LayeredHumanExecutionError("HumanDB execution envelope drifted")
    members, source_records, _source_identity, runtime_identity = _validate_inputs(
        human_core_decision=human_core_decision,
        source_core_decision=source_core_decision,
        human_audit=human_audit,
        runtime_decision=runtime_decision,
        installation=_portable_replay_installation(),
    )
    records = list(_sequence(payload.get("records"), context="execution records"))
    data_query = _mapping(payload.get("data_query"), context="data-query evidence")
    if (
        data_query.get("request_count") != 273
        or data_query.get("response_count") != 273
        or data_query.get("transcript_line_count") != 546
        or data_query.get("cross_process_comparison")
        != "exact_ordered_transcript_equality"
    ):
        raise LayeredHumanExecutionError("data-query evidence drifted")
    _sha256(
        data_query.get("transcript_identity"),
        context="data-query transcript identity",
    )
    if (
        len(records) != HUMAN_EXECUTION_RECORD_COUNT
        or payload.get("human_core_membership_identity")
        != human_core_decision["selection"]["membership_identity"]
        or payload.get("source_core_membership_identity")
        != source_core_decision["source_core"]["source_membership_identity"]
        or payload.get("replay_runtime_identity") != runtime_identity
        or payload.get("human_execution_identity")
        != canonical_sha256({"records": records, "data_query": data_query})
    ):
        raise LayeredHumanExecutionError("HumanDB execution identity drifted")

    prefix_ids: set[str] = set()
    history_ids: set[str] = set()
    expected_sanmill = _portable_replay_installation().portable_record()
    for record, member, source_record in zip(
        records,
        members,
        source_records,
        strict=True,
    ):
        if (
            record.get("human_core_id") != member.get("core_id")
            or record.get("source_core_id") != source_record.get("source_core_id")
            or record.get("source_history_id")
            != member.get("source_history_id")
        ):
            raise LayeredHumanExecutionError("execution-member binding drifted")
        prefix = LayeredOpeningPrefixV2.from_dict(record.get("execution_record"))
        if (
            list(prefix.action_tokens) != member["action_tokens"]
            or prefix.final_nmm_fen != member["final"]["nmm_fen"]
            or prefix.final_ring16_fen
            != member["final"]["ring16_canonical_fen"]
            or dict(prefix.sanmill) != expected_sanmill
            or prefix.source_evidence.get("replay_runtime_identity")
            != runtime_identity
        ):
            raise LayeredHumanExecutionError("execution prefix evidence drifted")
        prefix_ids.add(prefix.prefix_identity)
        history_ids.add(prefix.final_history_sha256)
    if len(prefix_ids) != 21 or len(history_ids) != 21:
        raise LayeredHumanExecutionError("execution identities are not unique")
    if payload.get("decision") != {
        "human_execution_records_frozen": True,
        "executable_64_prefix_corpus_frozen": False,
        "evaluation_authorized": False,
        "training_authorized": False,
    }:
        raise LayeredHumanExecutionError("execution decision boundary drifted")
    return {
        "records": len(records),
        "prefix_identities": len(prefix_ids),
        "history_identities": len(history_ids),
    }
