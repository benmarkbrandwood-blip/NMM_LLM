#!/usr/bin/env python3
"""Export, describe, verify, or compare immutable NMM model bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.delivery.model_bundle import compare_model_bundles, export_model_bundle, verify_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("checkpoint")
    export.add_argument("destination")
    export.add_argument("--license", default=str(_ROOT / "LICENSE"))
    for name in ("describe", "verify"):
        command = commands.add_parser(name)
        command.add_argument("bundle")
        if name == "verify":
            command.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    compare = commands.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    args = parser.parse_args()
    if args.command == "export":
        result = export_model_bundle(args.checkpoint, args.destination, license_path=args.license)
    elif args.command == "describe":
        result = json.loads((Path(args.bundle) / "bundle.json").read_text(encoding="utf-8"))
    elif args.command == "verify":
        result = verify_model_bundle(args.bundle, device=args.device)
    else:
        result = compare_model_bundles(args.left, args.right)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
