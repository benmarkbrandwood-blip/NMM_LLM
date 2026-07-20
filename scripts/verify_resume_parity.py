#!/usr/bin/env python3
"""Verify that segmented exact resume matches a continuous reference run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.validation.resume_parity import verify_resume_parity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous-checkpoint", required=True)
    parser.add_argument("--resumed-checkpoint", required=True)
    parser.add_argument("--continuous-log", required=True)
    parser.add_argument("--resumed-log", action="append", required=True)
    parser.add_argument("--continuous-database", required=True)
    parser.add_argument("--resumed-database", required=True)
    args = parser.parse_args()
    report = verify_resume_parity(
        continuous_checkpoint=args.continuous_checkpoint,
        resumed_checkpoint=args.resumed_checkpoint,
        continuous_log=args.continuous_log,
        resumed_logs=args.resumed_log,
        continuous_database=args.continuous_database,
        resumed_database=args.resumed_database,
    )
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
