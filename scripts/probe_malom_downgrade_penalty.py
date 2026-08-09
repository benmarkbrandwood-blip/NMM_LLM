#!/usr/bin/env python3
"""Probe the new downgrade penalty on the frozen 19-state cohort."""

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
from learned_ai.validation.malom_downgrade_penalty_probe import (  # noqa: E402
    build_malom_downgrade_penalty_probe,
)


DEFAULT_SOURCE = Path(
    "learned_ai/checkpoints/evaluation/"
    "sanmill-corrected-retained-v2-heldout-v1/"
    "mill-bonus-no-update-probe.json"
)
DEFAULT_OUTPUT = Path(
    "learned_ai/checkpoints/evaluation/"
    "sanmill-corrected-retained-v2-heldout-v1/"
    "malom-downgrade-penalty-no-update-probe.json"
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


def _repository_ignored_path(value: str, *, role: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{role} must stay inside the repository") from exc
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError(f"{role} must be ignored by Git")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
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
        source = _repository_ignored_path(args.source, role="source probe")
        output = _repository_ignored_path(args.output, role="probe output")
    except ValueError as exc:
        parser.error(str(exc))
    if not source.is_file():
        parser.error("frozen source probe is missing")
    if output.exists():
        parser.error("probe output already exists")
    if _git_output("branch", "--show-current") != "dev":
        parser.error("probe requires dev")
    head = _git_output("rev-parse", "HEAD")
    if head != _git_output("rev-parse", "origin/dev"):
        parser.error("dev must equal origin/dev")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        parser.error("tracked and untracked worktree must be clean before probe")

    module_path = (
        ROOT / "learned_ai/validation/malom_downgrade_penalty_probe.py"
    )
    script_path = Path(__file__).resolve()
    probe = build_malom_downgrade_penalty_probe(
        source_probe_path=source,
        auditor={
            "implementation_commit": head,
            "implementation_tree": _git_output("rev-parse", "HEAD^{tree}"),
            "module_sha256": _sha256_file(module_path),
            "script_sha256": _sha256_file(script_path),
            "tracked_worktree_clean": True,
        },
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
