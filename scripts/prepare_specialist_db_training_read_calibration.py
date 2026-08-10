#!/usr/bin/env python3
"""Read-only audit for the SpecialistDB training-read calibration."""

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
    SpecialistReadCalibrationError,
    inspect_source_readiness,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS_CONFIG))
    args = parser.parse_args(argv)
    try:
        report = inspect_source_readiness(
            root=ROOT,
            contract_path=_resolve(args.contract),
            paths_config=_resolve(args.paths_config),
            python_executable=sys.executable,
        )
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
