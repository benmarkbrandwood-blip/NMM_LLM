#!/usr/bin/env python3
"""Prepare the mature target-refresh diagnostic without launching training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.validation.target_refresh_mature_fork_diagnostic import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_READINESS,
    MatureTargetRefreshDiagnosticError,
    prepare_mature_fork_diagnostic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    readiness = prepare_mature_fork_diagnostic(
        root=ROOT,
        contract_path=(ROOT / args.contract if not args.contract.is_absolute() else args.contract),
        paths_config=(
            ROOT / args.paths_config
            if not args.paths_config.is_absolute()
            else args.paths_config
        ),
        readiness_path=(
            ROOT / args.readiness if not args.readiness.is_absolute() else args.readiness
        ),
    )
    print(json.dumps(readiness, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MatureTargetRefreshDiagnosticError as exc:
        print(f"fatal_stop: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
