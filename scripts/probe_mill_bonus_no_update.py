#!/usr/bin/env python3
"""Probe legacy and corrected mill shaping on 19 frozen downgrade turns."""

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
    write_new_audit,
)
from learned_ai.validation.mill_bonus_no_update_probe import (  # noqa: E402
    build_mill_bonus_no_update_probe,
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
            "mill-bonus-no-update-probe.json"
        ),
        help="new ignored canonical output",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="required acknowledgement before creating the probe output",
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
        parser.error("probe output must remain in the frozen evaluation root")
    relative_output = output.relative_to(ROOT).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_output],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        parser.error("probe output must be ignored by Git")
    if _git_output("status", "--porcelain"):
        parser.error("tracked and untracked worktree must be clean before probe")

    module_path = (
        ROOT / "learned_ai/validation/mill_bonus_no_update_probe.py"
    )
    script_path = Path(__file__).resolve()
    auditor = {
        "implementation_commit": _git_output("rev-parse", "HEAD"),
        "implementation_tree": _git_output("rev-parse", "HEAD^{tree}"),
        "module_sha256": _sha256_file(module_path),
        "script_sha256": _sha256_file(script_path),
        "tracked_worktree_clean": True,
    }
    probe = build_mill_bonus_no_update_probe(
        wdl_audit_path=paths.output_root / "wdl-transition-audit.json",
        oracle_audit_path=paths.output_root / "oracle-alternative-audit.json",
        malom_path=paths.malom_db,
        malom_manifest_path=paths.malom_manifest,
        auditor=auditor,
    )
    write_new_audit(output, probe)
    print(
        json.dumps(
            {
                "output": str(output),
                "probe_identity": probe["probe_identity"],
                "summary": probe["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
