"""MIF-INTEROP/1 request dispatch for the NMM_LLM implementation."""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    MAX_EVENTS,
    MAX_INTEROP_REQUEST_BYTES,
    MAX_REPETITION_ENTRIES,
    MIF_ADAPTER_PROTOCOL_SHA256,
    MIF_CHINESE_SPEC_SHA256,
    MIF_COMMIT,
    MIF_ENGLISH_SPEC_SHA256,
    MIF_EXECUTABLE_CORPUS_SHA256,
    MIF_INDEX_SHA256,
    MIF_SMOKE_CORPUS_SHA256,
    PROTOCOL,
    MifError,
    enforce_resource_limit,
    fail,
    require_object,
)
from .engine import execute, replay
from .model import canonicalize_mfen, canonicalize_mpk


EXAMPLE_SEMANTIC = "sha256:224f7e368e322a4cc8c1225a025fb548d5b41eb096d34b7ae0543182d1aa9393"
EXAMPLE_DOCUMENT = "sha256:62479b6f40efb8ab478bab3d2b725647213604fcd3cc9cd4c1f69357535ae257"
ORIGIN_SEMANTIC = "sha256:173caf8189defd1ab7d4a3e8b9e26688a07fd77976bf09d56bff5fe0c273e1a1"
ORIGIN_DOCUMENT = "sha256:9e8a7aa8f71fe2d8cc4d0d3bc5571f2c09e21f98b12b336b691f1cdbe5bb2833"


def _limit_array(container: Mapping[str, Any], member: str, name: str, limit: int) -> None:
    value = container.get(member)
    if isinstance(value, list):
        enforce_resource_limit(name, len(value), limit)


def _preflight_resource_limits(
    operation: str,
    payload: Mapping[str, Any],
) -> None:
    """Apply published semantic limits before closed-object validation."""

    if operation == "execute":
        _limit_array(payload, "events", "events", MAX_EVENTS)
        _limit_array(
            payload,
            "repetitionSeed",
            "repetition-entries",
            MAX_REPETITION_ENTRIES,
        )
        return
    if operation in {"replay", "project-logical-turns"}:
        mstate = payload.get("mstate")
        if isinstance(mstate, Mapping):
            _limit_array(mstate, "events", "events", MAX_EVENTS)
            _limit_array(
                mstate,
                "repetitionHistory",
                "repetition-entries",
                MAX_REPETITION_ENTRIES,
            )
        return
    if operation == "transform":
        document = payload.get("document")
        if payload.get("kind") == "mstate" and isinstance(document, Mapping):
            _limit_array(document, "events", "events", MAX_EVENTS)
            _limit_array(
                document,
                "repetitionHistory",
                "repetition-entries",
                MAX_REPETITION_ENTRIES,
            )
        _limit_array(
            payload,
            "repetitionHistory",
            "repetition-entries",
            MAX_REPETITION_ENTRIES,
        )


def capabilities() -> dict[str, Any]:
    """Return an honest pre-suite capability claim for the pinned contract."""

    return {
        "format": "MIFCAP/1.0",
        "implementation": {
            "name": "nmm-llm-independent-mif-adapter",
            "version": f"mif-1.0-{MIF_COMMIT[:12]}",
        },
        "suites": [],
        "classes": [
            {"id": "conversion", "level": "none"},
            {"id": "identity", "level": "implemented"},
            {"id": "key", "level": "implemented"},
            {"id": "position", "level": "implemented"},
            {"id": "replay", "level": "implemented"},
            {"id": "ruleset", "level": "implemented"},
            {"id": "transform", "level": "implemented"},
        ],
        "formats": [
            {"id": "MFEN/1.0", "read": "implemented", "write": "implemented"},
            {"id": "MIFCAP/1.0", "read": "none", "write": "implemented"},
            {"id": "MIFCONV/1.0", "read": "none", "write": "none"},
            {"id": "MIFDIAG/1.0", "read": "none", "write": "implemented"},
            {"id": "MIFINV/1.0", "read": "implemented", "write": "none"},
            {"id": "MIFPOS/1.0", "read": "implemented", "write": "implemented"},
            {"id": "MIFSUITE/1.0", "read": "none", "write": "none"},
            {"id": "MIFTURN/1.0", "read": "none", "write": "implemented"},
            {"id": "MPK/1.0", "read": "implemented", "write": "implemented"},
            {"id": "MRS/1.0", "read": "implemented", "write": "none"},
            {"id": "MSTATE/1.0", "read": "implemented", "write": "implemented"},
        ],
        "profiles": {
            "semantics": ["mif-finite-rules-v3"],
            "semanticProjection": ["mrs-semantic-v1"],
            "state": ["mill24-state-v1"],
            "key": ["structural-d4-v1"],
            "repetitionProjection": ["repetition-observation-v1"],
            "observation": ["stable-moving-v1", "stable-primary-decision-v1"],
            "repetitionSummary": ["reset-count-smt-v1"],
            "resumption": ["resumption-state-v1"],
            "decision": ["decision-state-v1"],
            "claimLifecycle": ["stable-claim-rights-v1"],
            "mpkBinding": ["inline-semantic-digest-v1"],
            "transform": ["mill24-full-state-v1"],
            "logicalTurn": ["logical-turn-v1"],
            "placingLiveness": ["apply-board-full", "draw", "loss"],
        },
        "rulesets": [
            {
                "id": "example-morris",
                "version": 1,
                "semanticDigest": EXAMPLE_SEMANTIC,
                "documentDigest": EXAMPLE_DOCUMENT,
                "level": "implemented",
            },
            {
                "id": "x-origin-stabilization",
                "version": 1,
                "semanticDigest": ORIGIN_SEMANTIC,
                "documentDigest": ORIGIN_DOCUMENT,
                "level": "implemented",
            },
        ],
        "invarianceDeclarations": [],
        "conversions": [],
        "resourceLimits": [
            {"name": "events", "limit": MAX_EVENTS},
            {
                "name": "interop-request-bytes",
                "limit": MAX_INTEROP_REQUEST_BYTES,
            },
            {"name": "repetition-entries", "limit": MAX_REPETITION_ENTRIES},
        ],
        "testedCorpora": [
            {
                "digest": MIF_SMOKE_CORPUS_SHA256,
                "classes": [
                    "identity",
                    "position",
                    "replay",
                    "ruleset",
                    "transform",
                ],
            }
        ],
        "annotations": {
            "contractCommit": MIF_COMMIT,
            "englishSpec": MIF_ENGLISH_SPEC_SHA256,
            "chineseSpec": MIF_CHINESE_SPEC_SHA256,
            "artifactIndex": MIF_INDEX_SHA256,
            "executableCorpus": MIF_EXECUTABLE_CORPUS_SHA256,
            "adapterProtocol": MIF_ADAPTER_PROTOCOL_SHA256,
            "smokeCorpus": MIF_SMOKE_CORPUS_SHA256,
            "scope": "pinned-candidate-corpus-rulesets; no MIFSUITE published",
        },
    }


class MifInteropAdapter:
    """Deterministic request handler; process framing lives in the CLI."""

    def handle(self, request_value: Any) -> dict[str, Any]:
        request_id = "invalid-request"
        operation = "capabilities"
        try:
            request = require_object(
                request_value,
                required={"protocol", "kind", "requestId", "operation", "payload"},
                context="MIF-INTEROP request",
            )
            if request["protocol"] != PROTOCOL or request["kind"] != "request":
                fail("syntax", "invalid-request-envelope", "invalid protocol or kind")
            request_id = request["requestId"]
            operation = request["operation"]
            if not isinstance(request_id, str) or not request_id:
                fail("syntax", "invalid-request-id", "requestId must be text")
            if operation not in {
                "capabilities",
                "canonicalize",
                "execute",
                "replay",
                "transform",
                "project-logical-turns",
                "project-legal-actions",
            }:
                fail("unsupported", "unsupported-operation", "unknown operation")
            if not isinstance(request["payload"], Mapping):
                fail("syntax", "object-required", "payload must be an object")
            _preflight_resource_limits(operation, request["payload"])
            result = self._dispatch(operation, request["payload"])
            return {
                "protocol": PROTOCOL,
                "kind": "response",
                "requestId": request_id,
                "operation": operation,
                "status": "ok",
                "result": result,
            }
        except MifError as exc:
            return {
                "protocol": PROTOCOL,
                "kind": "response",
                "requestId": request_id,
                "operation": operation,
                "status": "error",
                "diagnostics": exc.as_diagnostics(),
            }
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            diagnostic = MifError(
                "syntax",
                "invalid-document",
                f"malformed request document: {exc}",
            )
            return {
                "protocol": PROTOCOL,
                "kind": "response",
                "requestId": request_id,
                "operation": operation,
                "status": "error",
                "diagnostics": diagnostic.as_diagnostics(),
            }

    def _dispatch(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if operation == "capabilities":
            require_object(payload, required=set(), context="capabilities payload")
            return {"capabilities": capabilities()}
        if operation == "canonicalize":
            canonical = require_object(
                payload,
                required={"format", "value"},
                optional={"manifest"},
                context="canonicalize payload",
            )
            if canonical["format"] == "MFEN/1.0":
                value = canonicalize_mfen(canonical["value"], canonical.get("manifest"))
            elif canonical["format"] == "MPK/1.0":
                value = canonicalize_mpk(canonical["value"], canonical.get("manifest"))
            else:
                fail("unsupported", "unsupported-format", "canonicalizer supports MFEN and MPK")
            return {"value": value}
        if operation == "execute":
            execute_payload = require_object(
                payload,
                required={"manifest", "origin", "events", "repetitionSeed", "preOriginClaims"},
                context="execute payload",
            )
            events = execute_payload["events"]
            repetition_seed = execute_payload["repetitionSeed"]
            if isinstance(events, list):
                enforce_resource_limit("events", len(events), MAX_EVENTS)
            if isinstance(repetition_seed, list):
                enforce_resource_limit(
                    "repetition-entries",
                    len(repetition_seed),
                    MAX_REPETITION_ENTRIES,
                )
            execution = execute(
                execute_payload["manifest"],
                execute_payload["origin"],
                execute_payload["events"],
                execute_payload["repetitionSeed"],
                execute_payload["preOriginClaims"],
            )
            return execution.result
        if operation == "replay":
            replay_payload = require_object(
                payload,
                required={"mstate"},
                optional={"manifest"},
                context="replay payload",
            )
            _, result = replay(replay_payload["mstate"], replay_payload.get("manifest"))
            return result
        if operation == "transform":
            from .transform import transform_payload

            return transform_payload(payload)
        if operation == "project-logical-turns":
            from .turns import project_logical_turns

            return project_logical_turns(payload)
        from .legal_actions import project_legal_actions

        return project_legal_actions(payload)
