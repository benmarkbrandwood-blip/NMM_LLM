#!/usr/bin/env python3
"""Preflight or launch a bounded continuation of the Sanmill route probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.validation.sanmill_route_continuation import (  # noqa: E402
    continuation_probe_plan,
    load_probe_continuation_plan,
    preflight_probe_continuation,
)
from learned_ai.validation.sanmill_route_probe import (  # noqa: E402
    DEFAULT_PATHS_RELATIVE,
    SanmillRouteProbeRunFailure,
    inspect_published_source,
    probe_failure_output,
    publish_probe_failure,
    publish_probe_result,
    resolve_probe_inputs,
    run_probe,
    tracked_plan_record,
    validate_probe_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--launch", choices=("continuation",))
    parser.add_argument("--plan", type=Path, required=True)
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
        report = preflight_probe_continuation(args.plan, args.paths_config)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if not args.run_id or args.output is None:
        raise SystemExit("--launch continuation requires --run-id and --output")
    continuation = load_probe_continuation_plan(args.plan)
    plan = continuation_probe_plan(continuation)
    tracked_plan_record(plan)
    source = inspect_published_source(require_published=True)
    inputs = resolve_probe_inputs(args.paths_config)
    output = validate_probe_output(args.output)
    failure_output = validate_probe_output(probe_failure_output(output))
    invocation = [
        str(Path(sys.executable).resolve()),
        "scripts/probe_sanmill_integrated_route_continuation.py",
        "--launch",
        "continuation",
        "--plan",
        continuation.path.resolve().relative_to(_ROOT).as_posix(),
        "--paths-config",
        args.paths_config.resolve().relative_to(_ROOT).as_posix(),
        "--run-id",
        args.run_id,
        "--output",
        output.relative_to(_ROOT).as_posix(),
    ]
    try:
        report = run_probe(
            plan,
            inputs,
            source=source,
            run_id=args.run_id,
            invocation=invocation,
        )
    except SanmillRouteProbeRunFailure as exc:
        publish_probe_failure(failure_output, exc.report, plan)
        failure = exc.report["failure"]
        print(
            json.dumps(
                {
                    "status": exc.report["status"],
                    "run_id": exc.report["run_id"],
                    "failure_output": failure_output.relative_to(_ROOT).as_posix(),
                    "report_identity": exc.report["report_identity"],
                    "completed_samples": failure["completed_sample_count"],
                    "failed_scheduled_index": failure["failed_schedule"][
                        "scheduled_index"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 1
    publish_probe_result(output, report, plan)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": report["run_id"],
                "output": output.relative_to(_ROOT).as_posix(),
                "report_identity": report["report_identity"],
                "games": len(report["samples"]),
                "start_scheduled_index": continuation.start_scheduled_index,
                "end_scheduled_index_exclusive": (
                    continuation.end_scheduled_index_exclusive
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
