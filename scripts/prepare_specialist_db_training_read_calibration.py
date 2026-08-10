#!/usr/bin/env python3
"""Audit or prepare the SpecialistDB training-read calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_ai.validation.specialist_db_training_read_calibration import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT,
    DEFAULT_SOURCE_REPORT,
    SpecialistReadCalibrationError,
    inspect_source_readiness,
    prepare_specialist_read_calibration,
    publish_source_readiness,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--source-report", default=str(DEFAULT_SOURCE_REPORT))
    parser.add_argument(
        "--write-source-readiness",
        action="store_true",
        help="persist one ignored source-only report; never creates arm plans",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="create plans and preflights only; never authorizes training",
    )
    args = parser.parse_args(argv)
    contract_path = _resolve(args.contract)
    paths_config = _resolve(args.paths_config)
    report_path = _resolve(args.report)
    source_report_path = _resolve(args.source_report)
    try:
        if args.prepare:
            if args.write_source_readiness:
                parser.error("--prepare and --write-source-readiness are exclusive")
            report = prepare_specialist_read_calibration(
                root=ROOT,
                contract_path=contract_path,
                paths_config=paths_config,
                report_path=report_path,
                python_executable=sys.executable,
            )
        else:
            report = inspect_source_readiness(
                root=ROOT,
                contract_path=contract_path,
                paths_config=paths_config,
                python_executable=sys.executable,
            )
            if args.write_source_readiness:
                publish_source_readiness(source_report_path, report)
    except (SpecialistReadCalibrationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "state": "not_ready",
                    "launch_authorized": False,
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
