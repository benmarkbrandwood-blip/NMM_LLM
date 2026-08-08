#!/usr/bin/env python3
"""Preflight or explicitly launch the pinned Sanmill node calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.validation.sanmill_node_calibration import (  # noqa: E402
    DEFAULT_PATHS_RELATIVE,
    DEFAULT_PLAN_RELATIVE,
    inspect_published_source,
    load_calibration_plan,
    load_local_installation,
    preflight_calibration,
    publish_calibration_result,
    run_calibration,
    tracked_plan_record,
    validate_calibration_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight",
        action="store_true",
        help="validate identities and replay fixtures without timed search",
    )
    action.add_argument(
        "--launch",
        choices=("calibration",),
        help="run the full bounded calibration; requires separate authorization",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=_ROOT / DEFAULT_PLAN_RELATIVE,
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=_ROOT / DEFAULT_PATHS_RELATIVE,
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        if args.run_id is not None or args.output is not None:
            raise SystemExit("--run-id and --output are launch-only options")
        report = preflight_calibration(
            args.plan,
            args.paths_config,
            require_published=True,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if not args.run_id or args.output is None:
        raise SystemExit("--launch calibration requires --run-id and --output")
    plan = load_calibration_plan(args.plan)
    tracked_plan_record(plan)
    source = inspect_published_source(require_published=True)
    installation = load_local_installation(args.paths_config)
    output = validate_calibration_output(args.output)
    invocation = [
        str(Path(sys.executable).resolve()),
        "scripts/calibrate_sanmill_nodes.py",
        "--launch",
        "calibration",
        "--plan",
        plan.path.resolve().relative_to(_ROOT).as_posix(),
        "--paths-config",
        args.paths_config.resolve().relative_to(_ROOT).as_posix(),
        "--run-id",
        args.run_id,
        "--output",
        output.relative_to(_ROOT).as_posix(),
    ]
    report = run_calibration(
        plan,
        installation,
        source=source,
        run_id=args.run_id,
        invocation=invocation,
    )
    publish_calibration_result(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": report["run_id"],
                "output": str(output),
                "report_identity": report["report_identity"],
                "samples": report["bounded_work"]["samples"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
