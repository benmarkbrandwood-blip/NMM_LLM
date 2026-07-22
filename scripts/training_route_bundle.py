#!/usr/bin/env python3
"""Export, describe, or verify an immutable s_gen_v2 route bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.delivery.training_route_bundle import (
    export_training_route_bundle,
    verify_training_route_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("checkpoint")
    export.add_argument("run_manifest")
    export.add_argument("destination")
    export.add_argument("--license", default=str(_ROOT / "LICENSE"))
    describe = commands.add_parser("describe")
    describe.add_argument("bundle")
    verify = commands.add_parser("verify")
    verify.add_argument("bundle")
    verify.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if args.command == "export":
        result = export_training_route_bundle(
            args.checkpoint,
            args.run_manifest,
            args.destination,
            license_path=args.license,
        )
    elif args.command == "describe":
        result = json.loads(
            (Path(args.bundle) / "bundle.json").read_text(encoding="utf-8")
        )
    else:
        result = verify_training_route_bundle(args.bundle, device=args.device)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
