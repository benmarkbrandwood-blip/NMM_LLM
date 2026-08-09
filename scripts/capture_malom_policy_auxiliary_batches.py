#!/usr/bin/env python3
"""Preflight or launch the frozen no-update auxiliary batch capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.validation.malom_policy_auxiliary_batch_capture import (  # noqa: E402
    DEFAULT_PATHS_RELATIVE,
    DEFAULT_PLAN_RELATIVE,
    MalomPolicyAuxiliaryBatchCaptureRunFailure,
    failure_output_path,
    load_batch_capture_plan,
    preflight_batch_capture,
    publish_report,
    run_batch_capture,
    tracked_plan_record,
    validate_output_path,
    validate_readiness,
)
from learned_ai.validation.sanmill_route_probe import (  # noqa: E402
    inspect_published_source,
    resolve_probe_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight",
        action="store_true",
        help="verify published source, inputs, routes, and fresh initializations",
    )
    action.add_argument(
        "--launch",
        choices=("capture",),
        help="run once after separate readiness-bound authorization",
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
    parser.add_argument("--readiness", type=Path)
    parser.add_argument("--expected-readiness-identity")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    return parser


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _portable(path: Path) -> str:
    return path.resolve().relative_to(_ROOT).as_posix()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight:
        if any(
            value is not None
            for value in (
                args.readiness,
                args.expected_readiness_identity,
                args.run_id,
            )
        ):
            raise SystemExit(
                "--readiness, --expected-readiness-identity and --run-id "
                "are launch-only options"
            )
        report = preflight_batch_capture(
            args.plan,
            args.paths_config,
            require_published=True,
        )
        if args.output is not None:
            publish_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if (
        args.readiness is None
        or not args.expected_readiness_identity
        or not args.run_id
        or args.output is None
    ):
        raise SystemExit(
            "--launch capture requires --readiness, "
            "--expected-readiness-identity, --run-id and --output"
        )
    plan = load_batch_capture_plan(args.plan)
    tracked_plan_record(plan)
    source = inspect_published_source(require_published=True)
    readiness = _json(args.readiness)
    validate_readiness(
        readiness,
        plan,
        expected_identity=args.expected_readiness_identity,
        source=source,
    )
    inputs = resolve_probe_inputs(args.paths_config)
    output = validate_output_path(args.output)
    failure = validate_output_path(failure_output_path(output))
    invocation = [
        str(Path(sys.executable).resolve()),
        "scripts/capture_malom_policy_auxiliary_batches.py",
        "--launch",
        "capture",
        "--plan",
        _portable(args.plan),
        "--paths-config",
        _portable(args.paths_config),
        "--readiness",
        _portable(args.readiness),
        "--expected-readiness-identity",
        args.expected_readiness_identity,
        "--run-id",
        args.run_id,
        "--output",
        _portable(output),
    ]
    try:
        report = run_batch_capture(
            plan,
            inputs,
            source=source,
            readiness_identity=args.expected_readiness_identity,
            run_id=args.run_id,
            invocation=invocation,
        )
    except MalomPolicyAuxiliaryBatchCaptureRunFailure as exc:
        publish_report(failure, exc.report)
        print(
            json.dumps(
                {
                    "status": exc.report["status"],
                    "run_id": exc.report["run_id"],
                    "failure_output": _portable(failure),
                    "report_identity": exc.report["report_identity"],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 1
    publish_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": report["run_id"],
                "output": _portable(output),
                "report_identity": report["report_identity"],
                "games": len(report["samples"]),
                "batches": len(report["batches"]),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
