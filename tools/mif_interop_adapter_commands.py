#!/usr/bin/env python3
"""Emit a three-adapter MIF-INTEROP/1 harness configuration.

The generated command arrays keep the implementations process-isolated.  The
MIF reference runner is launched only as a comparison process; neither the
NMM_LLM adapter nor the Sanmill adapter imports it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


MIF_COMMIT = "0693353fe0821dcbbf547cc1eb9b679dcf2f90b8"
MIF_PINNED_FILES = {
    "mif-1.0.md": "330e65145ceb26fe582e58b89405d87bd73e8be200b476aef82c0ee27731d995",
    "docs/zh-CN/mif-1.0.md": (
        "9cc06abb57425e2bc2e26432b6da53abe503e9b5415ea0b4f854f19f68722cc1"
    ),
    "artifacts/mif-1.0/index.json": (
        "2bd247cd7e27ff4b0e142d8a0b2d6dececd619c882bb67f0be11bf763a794895"
    ),
    "artifacts/mif-1.0/corpus/executable/reference-cases.json": (
        "a48c50352caebce30deb1de11f8f73dbc4540ee538651c3a139d9bcb166ba983"
    ),
    "interop/adapter-protocol-v1.md": (
        "253c1d201ea1db625e0c534da445ca4ecaa0b07597dfc7dbf59fbd6adf89874f"
    ),
    "interop/cases/smoke-v1.json": (
        "a6d292f4d19381172fbc19f89d3ee42145a6d5533d6d81fd719394e25342bb53"
    ),
    "interop/cases/deterministic-v1.json": (
        "c2d7017b2a8583914aff1eeea38bc02b078814ca11346c484e0a2b38b5e94f0c"
    ),
}
NMM_ROOT = Path(__file__).resolve().parents[1]


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot resolve Git HEAD for {repository}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_worktree_changes(repository: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot inspect Git worktree for {repository}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_mif_sources(repository: Path) -> None:
    for relative, expected in MIF_PINNED_FILES.items():
        path = repository / Path(relative)
        if not path.is_file():
            raise ValueError(f"pinned MIF source is missing: {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                "MIF source hash mismatch for "
                f"{relative}: expected {expected}, got {actual}"
            )


def command_config(
    *,
    mif_root: Path,
    sanmill_root: Path,
    sanmill_binary: Path | None,
) -> dict[str, object]:
    mif_root = mif_root.resolve()
    sanmill_root = sanmill_root.resolve()
    mif_head = _git_head(mif_root)
    if mif_head != MIF_COMMIT:
        raise ValueError(
            f"MIF checkout must be exactly {MIF_COMMIT}; got {mif_head}"
        )
    worktree_changes = _git_worktree_changes(mif_root)
    if worktree_changes:
        raise ValueError(
            "MIF checkout has worktree changes and cannot produce "
            f"locked evidence: {worktree_changes}"
        )
    _verify_mif_sources(mif_root)
    reference_adapter = mif_root / "tools" / "mif_1_0_reference_adapter.py"
    nmm_adapter = NMM_ROOT / "tools" / "nmm_llm_mif_adapter.py"
    if not reference_adapter.is_file() or not nmm_adapter.is_file():
        raise ValueError("one of the Python adapter entry points is missing")
    if sanmill_binary is None:
        cargo_manifest = sanmill_root / "Cargo.toml"
        if not cargo_manifest.is_file():
            raise ValueError(f"Sanmill Cargo.toml not found: {cargo_manifest}")
        sanmill_command = [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(cargo_manifest),
            "-p",
            "tgf-cli",
            "--",
            "mill",
            "mif-interop",
        ]
    else:
        binary = sanmill_binary.resolve()
        if not binary.is_file():
            raise ValueError(f"Sanmill adapter binary not found: {binary}")
        sanmill_command = [str(binary), "mill", "mif-interop"]
    return {
        "protocol": "MIF-INTEROP-CONFIG/1",
        "adapters": [
            {
                "name": "mif-reference",
                "command": [
                    "{python}",
                    "-B",
                    "tools/mif_1_0_reference_adapter.py",
                ],
                "workingDirectory": "{repo}",
            },
            {
                "name": "nmm-llm",
                "command": ["{python}", "-B", str(nmm_adapter)],
                "workingDirectory": "{repo}",
            },
            {
                "name": "sanmill",
                "command": sanmill_command,
                "workingDirectory": "{repo}",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mif-root", type=Path, required=True)
    parser.add_argument("--sanmill-root", type=Path, required=True)
    parser.add_argument("--sanmill-binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        config = command_config(
            mif_root=args.mif_root,
            sanmill_root=args.sanmill_root,
            sanmill_binary=args.sanmill_binary,
        )
    except ValueError as exc:
        print(f"mif-interoperability-config: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
