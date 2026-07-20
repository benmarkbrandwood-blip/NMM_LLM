"""Append-only local candidate lifecycle and atomic accepted-bundle promotion."""

from __future__ import annotations

import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from learned_ai.delivery.model_bundle import verify_model_bundle
from learned_ai.evaluation.paired_protocol import load_evaluation_spec, recompute_evaluation
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256


CANDIDATE_SCHEMA = "nmm.candidate.v1"
EVENT_SCHEMA = "nmm.candidate-event.v1"
_TRANSITIONS = {
    None: {"candidate"},
    "candidate": {"validating", "quarantined"},
    "validating": {"evaluating", "quarantined"},
    "evaluating": {"accepted", "rejected", "inconclusive", "quarantined"},
    "inconclusive": {"evaluating", "quarantined"},
    "accepted": set(),
    "rejected": set(),
    "quarantined": set(),
}


class CandidateLifecycleError(RuntimeError):
    """Raised for invalid candidate state or evidence transitions."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    previous: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            event, identity = wrapper["event"], wrapper["event_sha256"]
            if event.get("previous_event_sha256") != previous or canonical_sha256(event) != identity:
                raise CandidateLifecycleError(f"candidate event chain failed at line {line_number}")
            events.append(event)
            previous = identity
    return events


def _append_event(root: Path, *, status: str, details: dict[str, Any]) -> dict[str, Any]:
    events_path = root / "events.jsonl"
    events = _read_events(events_path)
    current = events[-1]["status"] if events else None
    if status not in _TRANSITIONS[current]:
        raise CandidateLifecycleError(f"invalid candidate transition: {current!r} -> {status!r}")
    previous = canonical_sha256(events[-1]) if events else None
    event = {
        "schema_version": EVENT_SCHEMA,
        "candidate_id": root.name,
        "sequence": len(events),
        "status": status,
        "timestamp_utc": _utc_now(),
        "details": details,
        "previous_event_sha256": previous,
    }
    wrapper = {"event": event, "event_sha256": canonical_sha256(event)}
    with events_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(wrapper, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event


def register_candidate(registry: str | Path, candidate_id: str, bundle: str | Path) -> dict[str, Any]:
    """Register a verified immutable bundle as a new candidate."""
    if not candidate_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in candidate_id):
        raise CandidateLifecycleError("candidate ID contains unsupported characters")
    report = verify_model_bundle(bundle)
    root = Path(registry) / candidate_id
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "bundle_path": str(Path(bundle).resolve()),
        "bundle_identity": report["bundle_identity"],
        "model_identity": report["model_identity"],
        "registered_at_utc": _utc_now(),
    }
    (root / "candidate.json").write_bytes(canonical_json_bytes(record))
    _append_event(root, status="candidate", details={"bundle_identity": report["bundle_identity"]})
    return record


def transition_candidate(registry: str | Path, candidate_id: str, status: str, *, reason: str | None = None) -> dict[str, Any]:
    """Move through non-decision states without rewriting earlier evidence."""
    if status not in {"validating", "evaluating", "quarantined"}:
        raise CandidateLifecycleError("manual transition is limited to operational states")
    if status == "quarantined" and not reason:
        raise CandidateLifecycleError("quarantine requires an explicit reason")
    return _append_event(Path(registry) / candidate_id, status=status, details={"reason": reason})


def decide_candidate(
    registry: str | Path,
    candidate_id: str,
    spec_path: str | Path,
    records_path: str | Path,
    accepted_root: str | Path,
) -> dict[str, Any]:
    """Recompute evidence and atomically promote only an accepted candidate."""
    root = Path(registry) / candidate_id
    candidate = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
    spec = load_evaluation_spec(spec_path)
    if spec.candidate_bundle != candidate["bundle_identity"]:
        raise CandidateLifecycleError("evaluation candidate does not match registered bundle")
    result = recompute_evaluation(spec_path, records_path)
    decision = result["decision"]
    details = {
        "bundle_identity": candidate["bundle_identity"],
        "spec_identity": spec.spec_identity,
        "result_identity": result["result_identity"],
        "records_sha256": result["records_sha256"],
        "decision_rule": "frozen paired confidence interval",
    }
    if decision == "accepted":
        target = Path(accepted_root) / candidate["bundle_identity"]
        if target.exists():
            raise FileExistsError(f"accepted bundle already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            shutil.copytree(candidate["bundle_path"], temporary)
            verify_model_bundle(temporary)
            acceptance = {
                "schema_version": "nmm.acceptance-record.v1",
                **details,
                "candidate_id": candidate_id,
                "decision_time_utc": _utc_now(),
                "runtime": {"platform": platform.platform(), "pytorch": torch.__version__},
            }
            (temporary / "acceptance.json").write_bytes(canonical_json_bytes(acceptance))
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        details["accepted_path"] = str(target)
    _append_event(root, status=decision, details=details)
    return {"candidate_id": candidate_id, "status": decision, **details}


def candidate_status(registry: str | Path, candidate_id: str) -> dict[str, Any]:
    root = Path(registry) / candidate_id
    events = _read_events(root / "events.jsonl")
    if not events:
        raise CandidateLifecycleError("candidate has no lifecycle events")
    return {"candidate_id": candidate_id, "status": events[-1]["status"], "events": len(events), "last_event": events[-1]}
