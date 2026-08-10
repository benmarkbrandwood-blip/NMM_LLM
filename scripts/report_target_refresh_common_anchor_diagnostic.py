#!/usr/bin/env python3
"""Analyze the frozen common-anchor diagnostic; publish only if requested."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.evaluation.target_refresh_common_anchor_result import (  # noqa: E402
    DEFAULT_RESULT,
    TargetRefreshCommonAnchorResultError,
    analyze_target_refresh_common_anchor_result,
    publish_result,
)
from learned_ai.validation.target_refresh_common_anchor_diagnostic import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--readiness", default=str(DEFAULT_REPORT))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_RESULT))
    parser.add_argument(
        "--publish",
        action="store_true",
        help="write the ignored immutable raw result; default is read-only",
    )
    args = parser.parse_args(argv)
    try:
        report = analyze_target_refresh_common_anchor_result(
            root=ROOT,
            contract_path=_resolve(args.contract),
            readiness_path=_resolve(args.readiness),
            paths_config=_resolve(args.paths_config),
        )
        if args.publish:
            publish_result(_resolve(args.output), report)
    except (TargetRefreshCommonAnchorResultError, OSError, ValueError) as exc:
        print(json.dumps({"state": "not_ready", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
