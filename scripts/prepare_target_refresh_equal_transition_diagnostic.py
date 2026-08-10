#!/usr/bin/env python3
"""Audit or prepare only the shared prefixes for the equal-transition test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.validation.target_refresh_equal_transition_diagnostic import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT,
    DEFAULT_SOURCE_REPORT,
    TargetRefreshEqualTransitionError,
    inspect_source_readiness,
    prepare_prefix_plans,
    publish_source_readiness,
)
from learned_ai.validation.target_refresh_equal_transition_arms import (  # noqa: E402
    prepare_seed_arms,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument("--write-source-readiness", action="store_true")
    parser.add_argument(
        "--prepare-prefixes",
        action="store_true",
        help="create prefix plans/preflights only; never starts training",
    )
    parser.add_argument(
        "--prepare-seed-arms",
        type=int,
        choices=(64, 65, 66),
        help=(
            "prepare two same-seed arms only after its prefix completed; "
            "never starts training"
        ),
    )
    parser.add_argument(
        "--arm-report",
        help="exclusive ignored readiness path for --prepare-seed-arms",
    )
    args = parser.parse_args(argv)
    try:
        if args.prepare_seed_arms is not None:
            if args.prepare_prefixes or args.write_source_readiness:
                raise TargetRefreshEqualTransitionError(
                    "arm, prefix, and source-readiness modes are exclusive"
                )
            if not args.arm_report:
                raise TargetRefreshEqualTransitionError(
                    "--prepare-seed-arms requires --arm-report"
                )
            report = prepare_seed_arms(
                root=ROOT,
                contract_path=_resolve(args.contract),
                paths_config=_resolve(args.paths_config),
                seed=args.prepare_seed_arms,
                report_path=_resolve(args.arm_report),
                python_executable=sys.executable,
            )
        elif args.prepare_prefixes:
            if args.write_source_readiness:
                raise TargetRefreshEqualTransitionError(
                    "--prepare-prefixes and --write-source-readiness are exclusive"
                )
            report = prepare_prefix_plans(
                root=ROOT,
                contract_path=_resolve(args.contract),
                paths_config=_resolve(args.paths_config),
                report_path=_resolve(args.report),
                python_executable=sys.executable,
            )
        else:
            report = inspect_source_readiness(
                root=ROOT,
                contract_path=_resolve(args.contract),
                paths_config=_resolve(args.paths_config),
                report_path=_resolve(args.report),
                python_executable=sys.executable,
            )
            if args.write_source_readiness:
                publish_source_readiness(_resolve(args.source_report), report)
    except TargetRefreshEqualTransitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
