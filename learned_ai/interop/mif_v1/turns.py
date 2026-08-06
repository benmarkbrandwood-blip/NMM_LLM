"""MIF 1.0 complete logical-turn projection."""

from __future__ import annotations

from typing import Any, Mapping

from .common import fail, require_object
from .engine import replay
from .model import MifState


def _is_closed(snapshot: Mapping[str, Any]) -> bool:
    decision = snapshot["decisionState"]
    return decision["action"] != "r"


def project_logical_turns(payload_value: Mapping[str, Any]) -> dict[str, Any]:
    payload = require_object(
        payload_value,
        required={"mstate"},
        optional={"manifest"},
        context="project-logical-turns payload",
    )
    if not isinstance(payload["mstate"], Mapping):
        fail("syntax", "object-required", "mstate must be an object")
    execution, replay_result = replay(payload["mstate"], payload.get("manifest"))
    events = execution.events
    snapshots = {snapshot["eventSeq"]: snapshot for snapshot in execution.trace[1:]}
    fragments: list[dict[str, Any]] = []
    event_index = 0

    original_origin = MifState.parse(execution.origin)
    stabilized_origin = execution.trace[0]
    if original_origin.obligations or stabilized_origin["decisionState"]["action"] == "r":
        kind = "origin-obligation" if original_origin.obligations else "origin-stabilization"
        removes: list[int] = []
        complete = _is_closed(stabilized_origin)
        while event_index < len(events) and not complete:
            event = events[event_index]
            if event["type"] != "remove":
                break
            removes.append(event["seq"])
            complete = _is_closed(snapshots[event["seq"]])
            event_index += 1
        fragments.append(
            {
                "kind": kind,
                "removeEventSeqs": removes,
                "status": "complete" if complete else "truncated",
            }
        )

    while event_index < len(events):
        event = events[event_index]
        if event["type"] not in {"place", "move"}:
            event_index += 1
            continue
        primary_seq = event["seq"]
        removes: list[int] = []
        snapshot = snapshots[primary_seq]
        complete = _is_closed(snapshot)
        event_index += 1
        while event_index < len(events) and not complete:
            consequent = events[event_index]
            if consequent["type"] != "remove":
                break
            removes.append(consequent["seq"])
            snapshot = snapshots[consequent["seq"]]
            complete = _is_closed(snapshot)
            event_index += 1
        fragments.append(
            {
                "kind": "logical-turn",
                "primaryEventSeq": primary_seq,
                "removeEventSeqs": removes,
                "status": "complete" if complete else "truncated",
            }
        )

    document = {
        "format": "MIFTURN/1.0",
        "profile": "logical-turn-v1",
        "sourceResumptionDigest": replay_result["resumptionDigest"],
        "fragments": fragments,
    }
    return {"document": document}
