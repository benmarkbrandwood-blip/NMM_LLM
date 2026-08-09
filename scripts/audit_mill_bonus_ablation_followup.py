#!/usr/bin/env python3
"""Publish a compact, immutable follow-up audit of the six-arm result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.heldout_oracle_alternative_audit import (  # noqa: E402
    write_new_audit,
)
from learned_ai.evaluation.mill_bonus_ablation_followup import (  # noqa: E402
    build_followup_audit,
)


DEFAULT_RESULT = Path("out/mill-bonus-ablation-smoke-v1/result.json")
DEFAULT_OUTPUT = Path(
    "docs/evidence/sanmill-mill-bonus-ablation-followup-audit-2026-08-09.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _inside_root(value: str, *, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"{role} must stay inside the repository") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--write",
        action="store_true",
        help="required acknowledgement before creating immutable output",
    )
    args = parser.parse_args()
    if not args.write:
        parser.error("explicit --write flag is required")
    try:
        result_path = _inside_root(args.result, role="result")
        output = _inside_root(args.output, role="output")
    except ValueError as exc:
        parser.error(str(exc))
    if not result_path.is_file():
        parser.error("published result is missing")
    if output.exists():
        parser.error("audit output already exists")
    if _git_output("branch", "--show-current") != "dev":
        parser.error("audit requires dev")
    head = _git_output("rev-parse", "HEAD")
    if head != _git_output("rev-parse", "origin/dev"):
        parser.error("dev must equal origin/dev")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        parser.error("tracked and untracked worktree must be clean before audit")

    module_path = ROOT / "learned_ai/evaluation/mill_bonus_ablation_followup.py"
    script_path = Path(__file__).resolve()
    audit = build_followup_audit(
        root=ROOT,
        result_path=result_path,
        auditor={
            "implementation_commit": head,
            "implementation_tree": _git_output("rev-parse", "HEAD^{tree}"),
            "module_sha256": _sha256_file(module_path),
            "script_sha256": _sha256_file(script_path),
            "tracked_worktree_clean": True,
        },
    )
    write_new_audit(output, audit)
    print(
        json.dumps(
            {
                "audit_identity": audit["audit_identity"],
                "output": str(output),
                "summary": audit["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
