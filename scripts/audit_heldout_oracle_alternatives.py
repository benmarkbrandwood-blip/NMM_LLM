#!/usr/bin/env python3
"""Audit full Malom alternatives at committed held-out WDL downgrades."""

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

from learned_ai.evaluation.heldout_evaluation import (  # noqa: E402
    load_frozen_heldout_contract,
    resolve_heldout_paths,
)
from learned_ai.evaluation.heldout_oracle_alternative_audit import (  # noqa: E402
    build_heldout_oracle_alternative_audit,
    write_new_audit,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        default="data/training_paths.local.json",
        help="machine-local path registry",
    )
    parser.add_argument(
        "--output",
        default=(
            "learned_ai/checkpoints/evaluation/"
            "sanmill-corrected-retained-v2-heldout-v1/"
            "oracle-alternative-audit.json"
        ),
        help="new ignored canonical output",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="required acknowledgement before creating the audit output",
    )
    args = parser.parse_args()
    if not args.write:
        parser.error("explicit --write flag is required")

    contract = load_frozen_heldout_contract()
    paths = resolve_heldout_paths(contract, args.paths_config)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve(strict=False)
    if output.parent != paths.output_root.resolve():
        parser.error("audit output must remain in the frozen evaluation output root")
    relative_output = output.relative_to(ROOT).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_output],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        parser.error("audit output must be ignored by Git")
    if _git_output("status", "--porcelain"):
        parser.error("tracked and untracked worktree must be clean before audit")

    module_path = (
        ROOT / "learned_ai/evaluation/heldout_oracle_alternative_audit.py"
    )
    script_path = Path(__file__).resolve()
    auditor = {
        "implementation_commit": _git_output("rev-parse", "HEAD"),
        "implementation_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "module_sha256": _sha256_file(module_path),
        "script_sha256": _sha256_file(script_path),
        "tracked_worktree_clean": True,
    }
    audit = build_heldout_oracle_alternative_audit(
        spec_path=paths.output_spec,
        ledger_path=paths.output_ledger,
        report_path=paths.output_report,
        wdl_audit_path=paths.output_root / "wdl-transition-audit.json",
        malom_path=paths.malom_db,
        malom_manifest_path=paths.malom_manifest,
        auditor=auditor,
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
