#!/usr/bin/env python3
"""Validate and freeze completed target-refresh direct-crossplay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.target_refresh_direct_crossplay import (  # noqa: E402
    DirectCrossplayError,
    load_direct_crossplay_plan,
    summarize_direct_crossplay,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)


EVIDENCE_SCHEMA = "nmm.target-refresh-direct-crossplay-evidence.v1"
DEFAULT_PLAN = ROOT / (
    "docs/experiments/sanmill-target-refresh-direct-crossplay-v1-attempt-003.json"
)
DEFAULT_RUN_ROOT = ROOT / "out/target-refresh-direct-crossplay-v1-attempt-003"
DEFAULT_OUTPUT = ROOT / (
    "docs/evidence/target-refresh-direct-crossplay-attempt-003-result-2026-08-12.json"
)


class DirectCrossplayEvidenceError(RuntimeError):
    """Raised when the completed evidence chain is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DirectCrossplayEvidenceError(
                    f"duplicate JSON key {key!r} in {path.name}"
                )
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        if b"\r" in raw:
            raise DirectCrossplayEvidenceError(f"{path.name} is not LF-only")
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectCrossplayEvidenceError(f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise DirectCrossplayEvidenceError(f"{path.name} is not an object")
    return value


def _identity(value: Mapping[str, Any], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    observed = value.get(field)
    if not isinstance(observed, str) or canonical_sha256(body) != observed:
        raise DirectCrossplayEvidenceError(f"{field} differs")
    return observed


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise DirectCrossplayEvidenceError("evidence path leaves the repository") from exc


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DirectCrossplayEvidenceError("cannot read direct-crossplay ledger") from exc
    if not raw or b"\r" in raw or not raw.endswith(b"\n"):
        raise DirectCrossplayEvidenceError("direct-crossplay ledger is not LF-framed")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DirectCrossplayEvidenceError(
                f"direct-crossplay ledger line {line_number} is invalid"
            ) from exc
        if not isinstance(row, dict) or line != canonical_json_bytes(row) + b"\n":
            raise DirectCrossplayEvidenceError(
                f"direct-crossplay ledger line {line_number} is not canonical"
            )
        rows.append(row)
    return rows


def _scientific_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "plan_identity",
        "scope",
        "games",
        "pairs",
        "overall_no_refresh",
        "by_seed",
        "by_phase",
        "by_no_refresh_colour",
        "paired",
        "decision",
    }
    if not fields.issubset(result):
        raise DirectCrossplayEvidenceError("direct-crossplay result is incomplete")
    return {field: result[field] for field in fields}


def build_evidence(plan_path: Path, run_root: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    run_root = run_root.resolve(strict=True)
    plan = load_direct_crossplay_plan(plan_path)
    names = (
        "readiness",
        "authorization",
        "launch",
        "game-ledger",
        "result",
        "completion",
    )
    paths = {
        name: run_root / ("game-ledger.jsonl" if name == "game-ledger" else f"{name}.json")
        for name in names
    }
    if (run_root / "failure.json").exists():
        raise DirectCrossplayEvidenceError("attempt-003 contains failure evidence")
    for name, path in paths.items():
        if not path.is_file():
            raise DirectCrossplayEvidenceError(f"attempt-003 {name} is absent")

    readiness = _strict_json(paths["readiness"])
    authorization = _strict_json(paths["authorization"])
    launch = _strict_json(paths["launch"])
    result = _strict_json(paths["result"])
    completion = _strict_json(paths["completion"])
    readiness_identity = _identity(readiness, "readiness_identity")
    authorization_identity = _identity(authorization, "authorization_identity")
    launch_identity = _identity(launch, "launch_identity")
    result_identity = _identity(result, "result_identity")
    completion_identity = _identity(completion, "completion_identity")
    if readiness.get("plan", {}).get("plan_identity") != plan["plan_identity"]:
        raise DirectCrossplayEvidenceError("readiness binds another plan")
    for value, field in (
        (authorization, "authorization"),
        (launch, "launch"),
        (result, "result"),
        (completion, "completion"),
    ):
        if value.get("plan_identity") != plan["plan_identity"]:
            raise DirectCrossplayEvidenceError(f"{field} binds another plan")
    if authorization.get("readiness_identity") != readiness_identity:
        raise DirectCrossplayEvidenceError("authorization readiness differs")
    if launch.get("authorization_identity") != authorization_identity:
        raise DirectCrossplayEvidenceError("launch authorization differs")
    if result.get("launch_identity") != launch_identity:
        raise DirectCrossplayEvidenceError("result launch differs")
    if completion.get("result", {}).get("result_identity") != result_identity:
        raise DirectCrossplayEvidenceError("completion result differs")
    if completion.get("status") != "completed_once":
        raise DirectCrossplayEvidenceError("attempt-003 did not complete once")

    rows = _ledger_rows(paths["game-ledger"])
    ledger_sha256 = _sha256(paths["game-ledger"])
    recomputed = summarize_direct_crossplay(plan, rows)
    if _scientific_projection(result) != _scientific_projection(recomputed):
        raise DirectCrossplayEvidenceError("persisted scientific result differs")
    if result.get("ledger") != {
        "path": _relative(paths["game-ledger"]),
        "rows": len(rows),
        "sha256": ledger_sha256,
    }:
        raise DirectCrossplayEvidenceError("persisted ledger binding differs")
    if completion.get("ledger") != result["ledger"]:
        raise DirectCrossplayEvidenceError("completion ledger binding differs")
    if completion.get("resource_accounting", {}).get("training_games") != 0:
        raise DirectCrossplayEvidenceError("completion reports training games")
    if completion.get("resource_accounting", {}).get("optimizer_updates") != 0:
        raise DirectCrossplayEvidenceError("completion reports optimizer updates")

    observed = {
        "games": result["games"],
        "pairs": result["pairs"],
        "overall_no_refresh": result["overall_no_refresh"],
        "by_seed": result["by_seed"],
        "by_phase": result["by_phase"],
        "by_no_refresh_colour": result["by_no_refresh_colour"],
        "paired": result["paired"],
        "decision": result["decision"],
    }
    body = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "completed_validated_development_evidence",
        "plan": {
            "path": _relative(plan_path),
            "raw_sha256": _sha256(plan_path),
            "plan_identity": plan["plan_identity"],
        },
        "source_commit": result["source_commit"],
        "run": {
            "run_id": result["run_id"],
            "readiness_identity": readiness_identity,
            "authorization_identity": authorization_identity,
            "launch_identity": launch_identity,
            "result_identity": result_identity,
            "completion_identity": completion_identity,
            "recomputed_scientific_result_identity": recomputed["result_identity"],
            "elapsed_active_hours": completion["elapsed_hours"],
        },
        "artifacts": {
            name: {
                "path": _relative(path),
                "raw_sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "resource_accounting": completion["resource_accounting"],
        "observed_facts": observed,
        "interpretation": {
            "selected_development_direction": "no-refresh",
            "permanent_no_refresh_selected": False,
            "reason": (
                "The frozen classifier found a material direct advantage for "
                "the no-refresh condition at the tested boundary, but a "
                "forever-stale target is not an acceptable retained curriculum."
            ),
            "next_discriminating_experiment": (
                "Fork each mature no-refresh boundary checkpoint, refresh the "
                "target once only in the treatment arm, hold post-fork learner "
                "transitions, temperature sequence and 1,000-node Sanmill work "
                "equal, and compare three seeds before selecting a segment-aligned cadence."
            ),
        },
        "claim_boundary": {
            "development_mechanism_evidence_only": True,
            "held_out_strength": False,
            "automatic_long_run_selection": False,
            "promotion": False,
            "publication": False,
        },
    }
    return {**body, "evidence_identity": canonical_sha256(body)}


def validate_evidence(value: Mapping[str, Any]) -> str:
    if value.get("schema_version") != EVIDENCE_SCHEMA:
        raise DirectCrossplayEvidenceError("direct-crossplay evidence schema differs")
    identity = _identity(value, "evidence_identity")
    if value.get("status") != "completed_validated_development_evidence":
        raise DirectCrossplayEvidenceError("direct-crossplay evidence status differs")
    observed = value.get("observed_facts", {})
    if observed.get("games") != 288 or observed.get("pairs") != 144:
        raise DirectCrossplayEvidenceError("direct-crossplay evidence workload differs")
    if observed.get("decision", {}).get("classification") != (
        "material_no_refresh_direct_effect"
    ):
        raise DirectCrossplayEvidenceError("direct-crossplay evidence decision differs")
    if value.get("claim_boundary", {}).get("held_out_strength") is not False:
        raise DirectCrossplayEvidenceError("direct-crossplay evidence overclaims")
    return identity


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DirectCrossplayEvidenceError(f"evidence output already exists: {path}")
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = build_evidence(args.plan, args.run_root)
    validate_evidence(evidence)
    _write_exclusive(args.output.resolve(), evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DirectCrossplayError, DirectCrossplayEvidenceError) as exc:
        print(f"fatal_stop: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
