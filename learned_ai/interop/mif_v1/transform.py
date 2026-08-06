"""MIF 1.0 full-state coordinate conversion and equivalence gate."""

from __future__ import annotations

from typing import Any, Mapping

from .common import (
    MAX_EVENTS,
    MAX_REPETITION_ENTRIES,
    TRANSFORM_IDS,
    deep_copy_json,
    enforce_resource_limit,
    fail,
    require_object,
    sha256_digest,
    transform_board,
    transform_coordinate,
    transform_line_bits,
)
from .engine import replay, repetition_root
from .model import MifState, ResolvedManifest, resolve_manifest, resolve_ruleset_envelope


def _transform_mfen(value: Any, manifest: ResolvedManifest, transform: str) -> str:
    state = MifState.parse(value)
    return state.transformed(manifest, transform).serialize()


def _transform_semantic(
    semantic: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> dict[str, str]:
    if not isinstance(semantic, Mapping):
        fail("syntax", "object-required", "semantic state must be an object")
    result: dict[str, str] = {}
    for key, value in semantic.items():
        if not isinstance(value, str):
            fail("syntax", "invalid-extension", "semantic state value must be text")
        if key == "lm":
            halves = value.split(";")
            if len(halves) != 2:
                fail("syntax", "invalid-extension", "invalid lm semantic state")
            converted = []
            for half in halves:
                points = half.split(",")
                if len(points) != 2:
                    fail("syntax", "invalid-extension", "invalid lm semantic state")
                converted.append(
                    ",".join(
                        point if point == "-" else transform_coordinate(point, transform)
                        for point in points
                    )
                )
            result[key] = ";".join(converted)
        elif key == "pc":
            result[key] = value
        elif key == "ul":
            components = value.split(",")
            if len(components) != 2:
                fail("syntax", "invalid-extension", "invalid ul semantic state")
            width = 4 if manifest.manifest["topology"] == "mill24-orthogonal-v1" else 5
            result[key] = ",".join(
                f"{transform_line_bits(int(item, 16), manifest.manifest['topology'], transform):0{width}x}"
                for item in components
            )
        else:
            fail("unsupported", "unsupported-profile", f"semantic extension {key!r} is unsupported")
    return result


def transform_observation(
    value: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> dict[str, Any]:
    observation = require_object(
        value,
        required={
            "profile",
            "stateProfile",
            "semanticDigest",
            "board",
            "side",
            "phase",
            "action",
            "hands",
            "semantic",
        },
        context="repetition observation",
    )
    if observation["semanticDigest"] != manifest.semantic_digest:
        fail("integrity", "semantic-digest-mismatch")
    result = deep_copy_json(observation)
    result["board"] = transform_board(observation["board"], transform)
    result["semantic"] = _transform_semantic(observation["semantic"], manifest, transform)
    return result


def transform_repetition_history(
    value: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        fail("syntax", "array-required", "repetition history must be an array")
    enforce_resource_limit(
        "repetition-entries",
        len(value),
        MAX_REPETITION_ENTRIES,
    )
    result = []
    for entry_value in value:
        entry = require_object(
            entry_value,
            required={"source", "key"},
            optional={"eventSeq"},
            context="repetition entry",
        )
        transformed = {"source": entry["source"]}
        if "eventSeq" in entry:
            transformed["eventSeq"] = entry["eventSeq"]
        transformed["key"] = transform_observation(entry["key"], manifest, transform)
        result.append(transformed)
    return result


def _transform_event(
    event_value: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> dict[str, Any]:
    if not isinstance(event_value, Mapping):
        fail("syntax", "object-required", "event must be an object")
    event = deep_copy_json(event_value)
    for key in ("at", "from", "to"):
        if key in event:
            event[key] = transform_coordinate(event[key], transform)
    if event.get("type") == "remove":
        target = event.get("target")
        if not isinstance(target, Mapping) or target.get("zone") not in {
            "board",
            "hand",
        }:
            fail(
                "syntax",
                "x-event-shape",
                "invalid structured remove target",
                event_seq=(
                    event.get("seq")
                    if isinstance(event.get("seq"), int)
                    else None
                ),
            )
        if target["zone"] == "board":
            target["at"] = transform_coordinate(target["at"], transform)
    if "interventionLine" in event:
        line_id = event["interventionLine"]
        line_bits = transform_line_bits(
            1 << line_id,
            manifest.manifest["topology"],
            transform,
        )
        event["interventionLine"] = line_bits.bit_length() - 1
    return event


def _validate_invariance(
    value: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> None:
    if value is None:
        fail(
            "ineligible",
            "transform-invariance-undeclared",
            "coordinate conversion has no exact invariance declaration",
        )
    declaration = require_object(
        value,
        required={
            "format",
            "profile",
            "semanticDigest",
            "stateProfile",
            "transformProfile",
            "transforms",
            "extensionTreatments",
            "documentDigest",
        },
        optional={"annotations"},
        context="MIFINV/1.0",
    )
    if (
        declaration["format"] != "MIFINV/1.0"
        or declaration["profile"] != "transform-invariance-v1"
        or declaration["semanticDigest"] != manifest.semantic_digest
        or declaration["stateProfile"] != "mill24-state-v1"
        or declaration["transformProfile"] != "mill24-full-state-v1"
        or transform not in declaration["transforms"]
    ):
        fail("ineligible", "transform-invariance-undeclared")
    digest_input = {key: deep_copy_json(item) for key, item in declaration.items() if key != "documentDigest"}
    if sha256_digest(digest_input) != declaration["documentDigest"]:
        fail("integrity", "document-digest-mismatch")
    if declaration["extensionTreatments"]:
        fail("unsupported", "unsupported-profile", "invariance extension treatments are unsupported")


def _resolve_document_manifest(
    kind: str,
    document: Mapping[str, Any],
    caller_manifest: Any | None,
) -> ResolvedManifest:
    if kind in {"mstate", "mifpos"}:
        if "ruleset" not in document:
            fail("syntax", "closed-object-mismatch", "document lacks ruleset envelope")
        return resolve_ruleset_envelope(document["ruleset"], caller_manifest)
    if caller_manifest is None:
        fail("integrity", "manifest-missing", "decision transform needs MRS context")
    manifest = resolve_manifest(caller_manifest)
    if document.get("semanticDigest") != manifest.semantic_digest:
        fail("integrity", "semantic-digest-mismatch")
    return manifest


def _transform_mstate_document(
    document_value: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> dict[str, Any]:
    document = require_object(
        document_value,
        required={
            "format",
            "positionFormat",
            "stateProfile",
            "ruleset",
            "origin",
            "events",
            "current",
            "repetitionHistory",
            "preOriginClaims",
            "claims",
        },
        optional={"annotations", "extensions"},
        context="MSTATE/1.0",
    )
    if "extensions" in document:
        fail("unsupported", "unsupported-profile", "MSTATE extensions are unsupported")
    if not isinstance(document["events"], list):
        fail("syntax", "array-required", "events must be an array")
    enforce_resource_limit("events", len(document["events"]), MAX_EVENTS)
    if not isinstance(document["repetitionHistory"], list):
        fail("syntax", "array-required", "repetitionHistory must be an array")
    enforce_resource_limit(
        "repetition-entries",
        len(document["repetitionHistory"]),
        MAX_REPETITION_ENTRIES,
    )
    result = deep_copy_json(document)
    result["origin"] = _transform_mfen(document["origin"], manifest, transform)
    result["current"] = _transform_mfen(document["current"], manifest, transform)
    result["events"] = [
        _transform_event(event, manifest, transform) for event in document["events"]
    ]
    result["repetitionHistory"] = transform_repetition_history(
        document["repetitionHistory"], manifest, transform
    )
    return result


def _transform_mifpos_document(
    document_value: Any,
    manifest: ResolvedManifest,
    transform: str,
) -> dict[str, Any]:
    document = require_object(
        document_value,
        required={"format", "positionFormat", "stateProfile", "position", "ruleset"},
        optional={"annotations", "extensions"},
        context="MIFPOS/1.0",
    )
    if "extensions" in document:
        fail("unsupported", "unsupported-profile", "MIFPOS extensions are unsupported")
    result = deep_copy_json(document)
    result["position"] = _transform_mfen(document["position"], manifest, transform)
    return result


def _transform_decision_document(
    document_value: Any,
    manifest: ResolvedManifest,
    transform: str,
    repetition_history_value: Any | None,
) -> dict[str, Any]:
    document = require_object(
        document_value,
        required={
            "profile",
            "stateProfile",
            "semanticDigest",
            "board",
            "side",
            "phase",
            "action",
            "hands",
            "obligations",
            "noProgress",
            "outcome",
            "semantic",
            "repetitionSummary",
            "openOffer",
            "claimRights",
        },
        optional={"extensions"},
        context="decision-state-v1",
    )
    if "extensions" in document:
        fail("unsupported", "unsupported-profile", "decision extensions are unsupported")
    result = deep_copy_json(document)
    result["board"] = transform_board(document["board"], transform)
    state_text = " ".join(
        (
            "MFEN/1.0",
            "mill24-state-v1",
            document["board"],
            document["side"],
            document["phase"],
            document["action"],
            f"{document['hands'][0]},{document['hands'][1]}",
            document["obligations"],
            str(document["noProgress"] or 0),
            "0",
            document["outcome"],
        )
    )
    result["obligations"] = MifState.parse(state_text).transformed(
        manifest, transform
    ).obligations_field
    result["semantic"] = _transform_semantic(document["semantic"], manifest, transform)
    if document["repetitionSummary"] is not None:
        if repetition_history_value is None:
            fail(
                "ineligible",
                "insufficient-transform-history",
                "decision repetition root requires its materialized active history",
            )
        transformed_history = transform_repetition_history(
            repetition_history_value, manifest, transform
        )
        threshold = manifest.manifest["draw"]["repetition"]["count"]
        result["repetitionSummary"] = {
            "profile": "reset-count-smt-v1",
            "root": repetition_root(transformed_history, threshold),
        }
    return result


def transform_payload(payload_value: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_object(
        payload_value,
        required={"kind", "document", "transform", "verifyReplay", "requireEquivalence"},
        optional={"manifest", "repetitionHistory", "invariance"},
        context="transform payload",
    )
    kind = payload["kind"]
    if kind not in {"mstate", "mifpos", "decision-state"}:
        fail("unsupported", "unsupported-profile", "unsupported transform document kind")
    transform = payload["transform"]
    if transform not in TRANSFORM_IDS:
        fail("unsupported", "unsupported-profile", "unsupported D4 transform")
    if not isinstance(payload["verifyReplay"], bool) or not isinstance(payload["requireEquivalence"], bool):
        fail("syntax", "invalid-boolean", "transform gates must be Boolean")
    if not isinstance(payload["document"], Mapping):
        fail("syntax", "object-required", "transform document must be an object")
    manifest = _resolve_document_manifest(kind, payload["document"], payload.get("manifest"))
    if payload["requireEquivalence"]:
        _validate_invariance(payload.get("invariance"), manifest, transform)
    if kind == "mstate":
        document = _transform_mstate_document(payload["document"], manifest, transform)
        _, replay_result = replay(document, payload.get("manifest"))
        return {
            "document": document,
            "decisionState": replay_result["decisionState"],
            "decisionDigest": replay_result["decisionDigest"],
            "resumptionState": replay_result["resumptionState"],
            "resumptionDigest": replay_result["resumptionDigest"],
        }
    if kind == "mifpos":
        return {
            "document": _transform_mifpos_document(
                payload["document"], manifest, transform
            )
        }
    return {
        "document": _transform_decision_document(
            payload["document"],
            manifest,
            transform,
            payload.get("repetitionHistory"),
        )
    }
