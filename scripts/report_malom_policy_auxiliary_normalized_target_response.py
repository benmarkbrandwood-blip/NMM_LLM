"""Publish the frozen normalized target-response decision once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.malom_policy_auxiliary_normalized_target_response_result import (  # noqa: E402
    decide_normalized_target_response,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402


DEFAULT_PLAN = ROOT / (
    "docs/experiments/"
    "sanmill-malom-policy-auxiliary-normalized-target-response-audit-v1.json"
)
DEFAULT_AUDIT = ROOT / (
    "out/malom-policy-auxiliary-normalized-target-response-audit-v1/result.json"
)
DEFAULT_OUTPUT = ROOT / (
    "out/malom-policy-auxiliary-normalized-target-response-audit-v1/decision.json"
)
SCHEMA_VERSION = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-target-response-decision.v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _portable(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-source-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status.strip():
        raise RuntimeError("tracked worktree must be clean")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if source_commit != args.expected_source_commit:
        raise RuntimeError("decision source commit differs")

    plan_path = _resolve(args.plan)
    audit_path = _resolve(args.audit)
    output_path = _resolve(args.output)
    if output_path.exists():
        raise RuntimeError("target-response decision already exists")
    plan = _json(plan_path)
    audit = _json(audit_path)
    plan_sha256 = _sha256(plan_path)
    if audit.get("identities", {}).get("plan_sha256") != plan_sha256:
        raise RuntimeError("raw audit does not bind the frozen plan")
    if canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_identity"}
    ) != audit.get("audit_identity"):
        raise RuntimeError("raw audit identity differs")

    decision = decide_normalized_target_response(plan, audit)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identities": {
            "source_commit": source_commit,
            "plan_path": _portable(plan_path),
            "plan_sha256": plan_sha256,
            "raw_audit_path": _portable(audit_path),
            "raw_audit_sha256": _sha256(audit_path),
            "raw_audit_identity": audit["audit_identity"],
        },
        "decision": decision,
    }
    report["decision_identity"] = canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "state": "decision_published",
                "output": _portable(output_path),
                "decision_identity": report["decision_identity"],
                "sha256": _sha256(output_path),
                "verdict": decision["verdict"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
