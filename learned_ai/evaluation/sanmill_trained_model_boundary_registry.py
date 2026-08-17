"""Frozen boundary registry and scoped rehearsal coverage for baseline-v1.

The registry is the single source for reflected signatures, public surfaces,
stage requirements, evidence inventories, and dynamic rehearsal coverage.  It
does not wrap or replace any production callable.  Dynamic observation uses
the real Python code objects through a scoped profiling hook and is forbidden
outside a non-evidence rehearsal.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import Any


REGISTRY_SCHEMA = "nmm.sanmill-trained-model-boundary-registry.v1"
COVERAGE_EVENT_SCHEMA = (
    "nmm.sanmill-trained-model-boundary-coverage-event.v1"
)
COVERAGE_CONTRACT_SCHEMA = (
    "nmm.sanmill-trained-model-boundary-coverage-contract.v1"
)

CLASSIFICATIONS = {
    "rehearsal-required": ("rehearsal", True, "profile-event"),
    "preflight-required": ("preflight", True, "explicit-canary"),
    "static-audit-only": ("static-audit", False, "static-audit"),
    "not-required-with-reason": ("not-required", False, "none"),
}


class BoundaryRegistryError(RuntimeError):
    """Raised when a frozen boundary or its evidence differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_for_reference(reference: str) -> tuple[ModuleType, str]:
    if ":" not in reference:
        raise BoundaryRegistryError("boundary callable reference has no module")
    module_name, qualified = reference.split(":", 1)
    if not module_name or not qualified:
        raise BoundaryRegistryError("boundary callable reference is incomplete")
    return importlib.import_module(module_name), qualified


def _resolve_owner(module: ModuleType, qualified: str) -> Any:
    value: Any = module
    for part in qualified.split("."):
        value = getattr(value, part)
    return value


def resolve_registered_callable(row: Mapping[str, Any]) -> Any:
    module, qualified = _module_for_reference(str(row["callable"]))
    kind = row.get("callable_kind", "callable")
    if kind == "callable":
        value = _resolve_owner(module, qualified)
    elif kind == "property-getter":
        owner_name, separator, member = qualified.rpartition(".")
        if not separator:
            raise BoundaryRegistryError("property boundary has no owner")
        owner = _resolve_owner(module, owner_name)
        descriptor = inspect.getattr_static(owner, member)
        if not isinstance(descriptor, property) or descriptor.fget is None:
            raise BoundaryRegistryError("registered property getter differs")
        value = descriptor.fget
    else:
        raise BoundaryRegistryError(f"unknown boundary callable kind: {kind}")
    if not callable(value):
        raise BoundaryRegistryError("registered boundary is not callable")
    return value


def _annotation_text(value: Any) -> str:
    if value is inspect.Signature.empty:
        return "<empty>"
    return inspect.formatannotation(value)


def reflected_signature(value: Any) -> dict[str, Any]:
    signature = inspect.signature(value, eval_str=False)
    parameters = []
    for parameter in signature.parameters.values():
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "annotation": _annotation_text(parameter.annotation),
                "has_default": parameter.default is not inspect.Signature.empty,
                "default": (
                    None
                    if parameter.default is inspect.Signature.empty
                    else repr(parameter.default)
                ),
            }
        )
    return {
        "text": str(signature),
        "parameters": parameters,
        "return_annotation": _annotation_text(signature.return_annotation),
    }


def _python_function(value: Any) -> Any:
    value = inspect.unwrap(value)
    if inspect.ismethod(value):
        value = value.__func__
    return value


def _source_path(value: Any, repository_root: Path) -> str:
    source = inspect.getsourcefile(value)
    if source is None:
        return "<no-python-source>"
    path = Path(source).resolve(strict=False)
    try:
        return path.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def reflected_code_identity(value: Any, repository_root: Path) -> dict[str, Any] | None:
    function = _python_function(value)
    code = getattr(function, "__code__", None)
    if not isinstance(code, CodeType):
        return None
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = ""
    structural = {
        "name": code.co_name,
        "qualname": code.co_qualname,
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    return {
        "source_path": _source_path(function, repository_root),
        "first_line": code.co_firstlineno,
        "structural": structural,
        "identity": canonical_sha256(structural),
    }


def _identity_payload(registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registry)
    payload.pop("registry_identity", None)
    return payload


def _validate_row(row: Mapping[str, Any]) -> None:
    required = {
        "boundary_id",
        "module",
        "callable",
        "callable_kind",
        "role",
        "classification",
        "required_stage",
        "dynamic_required",
        "evidence_mode",
        "reason",
        "result_shape_validator",
        "resource_semantics",
        "signature",
        "code_identity",
    }
    if not required <= row.keys():
        raise BoundaryRegistryError("boundary registry row is incomplete")
    classification = str(row["classification"])
    expected = CLASSIFICATIONS.get(classification)
    if expected is None:
        raise BoundaryRegistryError("boundary classification differs")
    observed = (
        row["required_stage"],
        row["dynamic_required"],
        row["evidence_mode"],
    )
    if observed != expected:
        raise BoundaryRegistryError("boundary stage classification is inconsistent")
    if not isinstance(row["reason"], str) or not row["reason"].strip():
        raise BoundaryRegistryError("boundary classification reason is absent")
    if row["evidence_mode"] == "profile-event" and row["code_identity"] is None:
        raise BoundaryRegistryError("profile boundary has no Python code object")


def load_boundary_registry(path: str | Path) -> tuple[dict[str, Any], str]:
    registry_path = Path(path)
    raw = registry_path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundaryRegistryError("boundary registry is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != REGISTRY_SCHEMA:
        raise BoundaryRegistryError("boundary registry schema differs")
    identity = canonical_sha256(_identity_payload(value))
    if value.get("registry_identity") != identity:
        raise BoundaryRegistryError("boundary registry identity differs")
    rows = value.get("boundaries")
    if not isinstance(rows, list) or not rows:
        raise BoundaryRegistryError("boundary registry has no rows")
    ids = []
    callables = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise BoundaryRegistryError("boundary registry row is not an object")
        _validate_row(row)
        ids.append(str(row["boundary_id"]))
        callables.append((str(row["callable"]), str(row["callable_kind"])))
    if len(ids) != len(set(ids)):
        raise BoundaryRegistryError("boundary registry IDs are not unique")
    if len(callables) != len(set(callables)):
        raise BoundaryRegistryError("boundary registry callables are not unique")
    return value, hashlib.sha256(raw).hexdigest()


def freeze_boundary_registry_draft(
    draft_path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Fill reflection fields and seal a new, not-yet-frozen registry draft."""
    path = Path(draft_path)
    root = Path(repository_root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != REGISTRY_SCHEMA:
        raise BoundaryRegistryError("registry draft schema differs")
    if value.get("registry_identity") not in {None, "DRAFT"}:
        raise BoundaryRegistryError("refusing to rewrite a frozen registry")
    rows = value.get("boundaries")
    if not isinstance(rows, list) or not rows:
        raise BoundaryRegistryError("registry draft rows are absent")
    frozen_rows = []
    for raw_row in rows:
        row = dict(raw_row)
        classification = str(row.get("classification", ""))
        expected = CLASSIFICATIONS.get(classification)
        if expected is None:
            raise BoundaryRegistryError("registry draft classification differs")
        module_name, _qualified = str(row.get("callable", "")).split(":", 1)
        row.setdefault("module", module_name)
        row.setdefault("callable_kind", "callable")
        row.setdefault("required_stage", expected[0])
        row.setdefault("dynamic_required", expected[1])
        row.setdefault("evidence_mode", expected[2])
        row.setdefault("result_shape_validator", "any")
        row.setdefault("resource_semantics", "none")
        callable_value = resolve_registered_callable(row)
        row["signature"] = reflected_signature(callable_value)
        row["code_identity"] = reflected_code_identity(callable_value, root)
        _validate_row(row)
        frozen_rows.append(row)
    value["boundaries"] = frozen_rows
    value["registry_identity"] = canonical_sha256(_identity_payload(value))
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(encoded, encoding="ascii", newline="\n")
    return value


def _public_surface_observation(owner: type[Any]) -> dict[str, list[str]]:
    return {
        "methods": sorted(
            name
            for name in dir(owner)
            if not name.startswith("_") and callable(getattr(owner, name))
        ),
        "properties": sorted(
            name
            for name in dir(owner)
            if not name.startswith("_")
            and isinstance(inspect.getattr_static(owner, name), property)
        ),
    }


def _registered_public_surfaces(
    registry: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    surfaces: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"methods": [], "properties": []}
    )
    for row in registry["boundaries"]:
        owner = row.get("public_owner")
        member = row.get("public_member")
        kind = row.get("public_member_kind")
        if owner is None and member is None and kind is None:
            continue
        if not all(isinstance(value, str) and value for value in (owner, member, kind)):
            raise BoundaryRegistryError("public surface metadata is incomplete")
        if kind not in {"method", "property"}:
            raise BoundaryRegistryError("public surface member kind differs")
        key = "methods" if kind == "method" else "properties"
        surfaces[str(owner)][key].append(str(member))
    return {
        owner: {
            "methods": sorted(values["methods"]),
            "properties": sorted(values["properties"]),
        }
        for owner, values in sorted(surfaces.items())
    }


def _resolve_reference(reference: str) -> Any:
    module, qualified = _module_for_reference(reference)
    return _resolve_owner(module, qualified)


def registry_evidence_inventory(registry: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    by_role: dict[str, list[str]] = defaultdict(list)
    for row in registry["boundaries"]:
        grouped[str(row["classification"])].append(str(row["boundary_id"]))
        by_role[str(row["role"])].append(str(row["boundary_id"]))
    return {
        "registry_identity": registry["registry_identity"],
        "by_classification": {
            key: sorted(value) for key, value in sorted(grouped.items())
        },
        "by_role": {key: sorted(value) for key, value in sorted(by_role.items())},
        "rehearsal_profile_events": expected_dynamic_boundary_ids(
            registry, stage="rehearsal", evidence_mode="profile-event"
        ),
        "preflight_explicit_canaries": expected_dynamic_boundary_ids(
            registry, stage="preflight", evidence_mode="explicit-canary"
        ),
    }


def expected_dynamic_boundary_ids(
    registry: Mapping[str, Any],
    *,
    stage: str,
    evidence_mode: str,
) -> list[str]:
    return sorted(
        str(row["boundary_id"])
        for row in registry["boundaries"]
        if row["required_stage"] == stage
        and row["dynamic_required"] is True
        and row["evidence_mode"] == evidence_mode
    )


def coverage_contract(registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": COVERAGE_CONTRACT_SCHEMA,
        "stage": "rehearsal",
        "formal_result_eligibility": False,
        "observation": "real Python return event by frozen code object",
        "registry_identity": registry["registry_identity"],
        "expected_boundary_ids": expected_dynamic_boundary_ids(
            registry,
            stage="rehearsal",
            evidence_mode="profile-event",
        ),
    }
    return {**payload, "coverage_contract_identity": canonical_sha256(payload)}


def _module_source_paths(registry: Mapping[str, Any]) -> list[Path]:
    values = []
    for row in registry["boundaries"]:
        module, _qualified = _module_for_reference(str(row["callable"]))
        source = getattr(module, "__file__", None)
        if source:
            values.append(Path(source).resolve(strict=False))
    return sorted(set(values), key=lambda path: str(path).casefold())


def _static_source_audit(
    registry: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    transparent_proxies: list[str] = []
    attribute_interceptors: list[str] = []
    malom_rebindings: list[str] = []
    for path in _module_source_paths(registry):
        try:
            relative = path.relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            relative = str(path).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        def enclosing_function(node: ast.AST) -> str | None:
            current = parents.get(node)
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return current.name
                current = parents.get(current)
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if "__getattr__" in methods:
                    transparent_proxies.append(f"{relative}:{node.name}.__getattr__")
                if "__getattribute__" in methods:
                    attribute_interceptors.append(
                        f"{relative}:{node.name}.__getattribute__"
                    )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr in {"malom", "_endgame_db"}
                        and enclosing_function(node) != "__init__"
                    ):
                        malom_rebindings.append(
                            f"{relative}:{target.lineno}:{target.attr}"
                        )
    expected_interceptors = sorted(
        str(row["source_interceptor"])
        for row in registry["boundaries"]
        if row.get("source_interceptor") is not None
    )
    return {
        "source_paths": [
            (
                path.relative_to(repository_root.resolve()).as_posix()
                if path.is_relative_to(repository_root.resolve())
                else str(path).replace("\\", "/")
            )
            for path in _module_source_paths(registry)
        ],
        "transparent_proxy_classes": sorted(transparent_proxies),
        "attribute_interceptors": sorted(attribute_interceptors),
        "expected_attribute_interceptors": expected_interceptors,
        "malom_delegate_rebindings": sorted(malom_rebindings),
    }


def audit_boundary_registry(
    registry: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root)
    signature_checks = []
    code_checks = []
    resolved_code_objects: dict[int, str] = {}
    mismatches: list[str] = []
    for row in registry["boundaries"]:
        boundary_id = str(row["boundary_id"])
        try:
            value = resolve_registered_callable(row)
            signature = reflected_signature(value)
            code_identity = reflected_code_identity(value, root)
        except (AttributeError, BoundaryRegistryError, TypeError, ValueError) as exc:
            signature_checks.append(
                {"boundary_id": boundary_id, "passed": False, "error": str(exc)}
            )
            code_checks.append(
                {"boundary_id": boundary_id, "passed": False, "error": str(exc)}
            )
            continue
        signature_checks.append(
            {
                "boundary_id": boundary_id,
                "expected": row["signature"],
                "observed": signature,
                "passed": signature == row["signature"],
            }
        )
        code_checks.append(
            {
                "boundary_id": boundary_id,
                "expected": row["code_identity"],
                "observed": code_identity,
                "passed": code_identity == row["code_identity"],
            }
        )
        function = _python_function(value)
        code = getattr(function, "__code__", None)
        if row["evidence_mode"] == "profile-event" and isinstance(code, CodeType):
            prior = resolved_code_objects.setdefault(id(code), boundary_id)
            if prior != boundary_id:
                mismatches.append(
                    f"profile code object shared by {prior} and {boundary_id}"
                )

    registered_surfaces = _registered_public_surfaces(registry)
    observed_surfaces = {}
    for owner_reference in registered_surfaces:
        owner = _resolve_reference(owner_reference)
        if not isinstance(owner, type):
            raise BoundaryRegistryError("public surface owner is not a class")
        observed_surfaces[owner_reference] = _public_surface_observation(owner)
    if observed_surfaces != registered_surfaces:
        mismatches.append("public method or property surface differs")
    if not all(row["passed"] for row in signature_checks):
        mismatches.append("one or more reflected signatures differ")
    if not all(row["passed"] for row in code_checks):
        mismatches.append("one or more registered code identities differ")

    source_audit = _static_source_audit(registry, root)
    if source_audit["transparent_proxy_classes"]:
        mismatches.append("transparent proxy reintroduced")
    if source_audit["malom_delegate_rebindings"]:
        mismatches.append("Malom delegate rebound after construction")
    if (
        source_audit["attribute_interceptors"]
        != source_audit["expected_attribute_interceptors"]
    ):
        mismatches.append("attribute interceptor surface differs")
    return {
        "passed": not mismatches,
        "registry_identity": registry["registry_identity"],
        "evidence_inventory": registry_evidence_inventory(registry),
        "registered_public_surfaces": registered_surfaces,
        "observed_public_surfaces": observed_surfaces,
        "signature_checks": signature_checks,
        "code_identity_checks": code_checks,
        **source_audit,
        "mismatches": mismatches,
    }


def _valid_result(validator: str, value: Any) -> bool:
    if validator == "any":
        return True
    if validator == "none":
        return value is None
    if validator == "bool":
        return isinstance(value, bool)
    if validator == "str":
        return isinstance(value, str)
    if validator == "mapping":
        return isinstance(value, Mapping)
    if validator == "mapping-or-none":
        return value is None or isinstance(value, Mapping)
    if validator == "sequence":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if validator == "tuple":
        return isinstance(value, tuple)
    if validator == "context-or-none":
        return value is None or hasattr(value, "__enter__")
    if validator == "object-or-none":
        return True
    if validator == "str-or-none":
        return value is None or isinstance(value, str)
    if validator == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if validator == "number-or-none":
        return value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        )
    if validator == "hex-sha256":
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )
    if validator == "sanmill-applied-turn":
        return all(hasattr(value, name) for name in ("move", "actions", "state", "search"))
    if validator == "uci-logical-turn":
        return all(
            hasattr(value, name)
            for name in ("status", "model_action", "full_turn_actions", "node_budget")
        )
    if validator == "uci-position-state":
        return all(
            hasattr(value, name)
            for name in ("fen", "history_sha256", "no_capture_count", "outcome_reason")
        )
    if validator == "score-triple":
        return isinstance(value, tuple) and len(value) == 3
    if validator == "score-pair":
        return isinstance(value, tuple) and len(value) == 2
    if validator == "move-choice-pair":
        return isinstance(value, tuple) and len(value) == 2
    if validator == "oracle-or-none":
        return value is None or hasattr(value, "outcome")
    if validator == "list":
        return isinstance(value, list)
    raise BoundaryRegistryError(f"unknown result-shape validator: {validator}")


def _code_object(value: Any) -> CodeType | None:
    function = _python_function(value)
    code = getattr(function, "__code__", None)
    return code if isinstance(code, CodeType) else None


class BoundaryCoverageRecorder:
    """Append one successful real-code return event per registered boundary."""

    def __init__(
        self,
        registry: Mapping[str, Any],
        ledger_path: str | Path,
        *,
        formal_result_eligibility: bool,
    ) -> None:
        if formal_result_eligibility is not False:
            raise BoundaryRegistryError(
                "boundary profiling is restricted to non-evidence rehearsal"
            )
        self.registry = registry
        self.path = Path(ledger_path)
        self._rows = {str(row["boundary_id"]): row for row in registry["boundaries"]}
        self._by_code: dict[CodeType, str] = {}
        for boundary_id, row in self._rows.items():
            value = resolve_registered_callable(row)
            code = _code_object(value)
            if code is None:
                continue
            if code in self._by_code:
                raise BoundaryRegistryError("registered code object is ambiguous")
            self._by_code[code] = boundary_id
        self._active_frames: dict[int, str] = {}
        self._observed: set[str] = set()
        self._previous_event_sha256: str | None = None
        self._event_count = 0
        self._previous_profile: Any = None

    def _append_event(self, boundary_id: str, result: Any) -> None:
        row = self._rows[boundary_id]
        passed = _valid_result(str(row["result_shape_validator"]), result)
        record = {
            "schema_version": COVERAGE_EVENT_SCHEMA,
            "event_index": self._event_count,
            "stage": "rehearsal",
            "formal_result_eligibility": False,
            "registry_identity": self.registry["registry_identity"],
            "boundary_id": boundary_id,
            "code_identity": row["code_identity"],
            "result_shape_validator": row["result_shape_validator"],
            "result_shape_passed": passed,
            "previous_event_sha256": self._previous_event_sha256,
        }
        event_sha256 = canonical_sha256(record)
        wrapper = {"event": record, "event_sha256": event_sha256}
        with self.path.open("xb" if self._event_count == 0 else "ab") as handle:
            handle.write(canonical_json_bytes(wrapper) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._previous_event_sha256 = event_sha256
        self._event_count += 1
        self._observed.add(boundary_id)

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call":
            boundary_id = self._by_code.get(frame.f_code)
            if boundary_id is not None and boundary_id not in self._observed:
                self._active_frames[id(frame)] = boundary_id
        elif event == "return":
            boundary_id = self._active_frames.pop(id(frame), None)
            if boundary_id is not None and boundary_id not in self._observed:
                self._append_event(boundary_id, arg)

    def __enter__(self) -> "BoundaryCoverageRecorder":
        if self.path.exists():
            raise BoundaryRegistryError("coverage event ledger already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._previous_profile = sys.getprofile()
        if self._previous_profile is not None:
            raise BoundaryRegistryError("another Python profiler is already active")
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *_exc: object) -> None:
        sys.setprofile(self._previous_profile)


def load_coverage_ledger(
    path: str | Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_path = Path(path)
    raw = ledger_path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise BoundaryRegistryError("coverage ledger is empty or partial")
    rows_by_id = {
        str(row["boundary_id"]): row for row in registry["boundaries"]
    }
    previous = None
    events = []
    seen = set()
    for index, line in enumerate(raw.splitlines()):
        try:
            wrapper = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BoundaryRegistryError("coverage event is invalid JSON") from exc
        event = wrapper.get("event")
        identity = wrapper.get("event_sha256")
        if not isinstance(event, Mapping) or canonical_sha256(event) != identity:
            raise BoundaryRegistryError("coverage event identity differs")
        boundary_id = str(event.get("boundary_id"))
        frozen = rows_by_id.get(boundary_id)
        if (
            event.get("schema_version") != COVERAGE_EVENT_SCHEMA
            or event.get("event_index") != index
            or event.get("stage") != "rehearsal"
            or event.get("formal_result_eligibility") is not False
            or event.get("registry_identity") != registry["registry_identity"]
            or event.get("previous_event_sha256") != previous
            or frozen is None
            or event.get("code_identity") != frozen["code_identity"]
            or event.get("result_shape_validator")
            != frozen["result_shape_validator"]
            or event.get("result_shape_passed") is not True
            or boundary_id in seen
        ):
            raise BoundaryRegistryError("coverage event contract differs")
        events.append(dict(event))
        seen.add(boundary_id)
        previous = str(identity)
    return {
        "events": events,
        "event_count": len(events),
        "observed_boundary_ids": sorted(seen),
        "tail_event_sha256": previous,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "coverage_ledger_identity": canonical_sha256(
            {
                "registry_identity": registry["registry_identity"],
                "observed_boundary_ids": sorted(seen),
                "tail_event_sha256": previous,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
    }


def verify_rehearsal_coverage(
    path: str | Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    recovered = load_coverage_ledger(path, registry)
    expected = set(
        expected_dynamic_boundary_ids(
            registry,
            stage="rehearsal",
            evidence_mode="profile-event",
        )
    )
    observed = set(recovered["observed_boundary_ids"])
    missing = sorted(expected - observed)
    if missing:
        raise BoundaryRegistryError(
            "required rehearsal boundary coverage is absent: " + ", ".join(missing)
        )
    return {
        **recovered,
        "expected_boundary_ids": sorted(expected),
        "missing_boundary_ids": [],
        "passed": True,
    }


def verify_explicit_stage_evidence(
    registry: Mapping[str, Any],
    *,
    stage: str,
    observed_boundary_ids: Sequence[str],
) -> dict[str, Any]:
    expected = set(
        expected_dynamic_boundary_ids(
            registry,
            stage=stage,
            evidence_mode="explicit-canary",
        )
    )
    observed = set(observed_boundary_ids)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise BoundaryRegistryError(
            f"{stage} explicit boundary evidence differs; missing={missing}, extra={extra}"
        )
    return {
        "stage": stage,
        "expected_boundary_ids": sorted(expected),
        "observed_boundary_ids": sorted(observed),
        "passed": True,
    }
