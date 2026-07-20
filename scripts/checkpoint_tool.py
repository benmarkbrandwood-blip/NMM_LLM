#!/usr/bin/env python3
"""Describe, verify, compare, or explicitly migrate NMM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.training.checkpoint_envelope import (
    CheckpointDescriptor,
    inspect_checkpoint,
    load_checkpoint,
    verify_checkpoint_compatibility,
)
from learned_ai.training.checkpoint_migration import (
    inspect_legacy_checkpoint,
    migrate_legacy_checkpoint,
)


def _asset_pairs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        name, separator, identity = value.partition("=")
        if not separator or not name or not identity or name in result:
            raise ValueError(f"invalid or duplicate asset assignment: {value!r}")
        result[name] = identity
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("describe", "verify"):
        command = commands.add_parser(name)
        command.add_argument("checkpoint")
    compare = commands.add_parser("compare")
    compare.add_argument("checkpoint")
    compare.add_argument("--config-sha256", required=True)
    compare.add_argument("--feature-schema", required=True)
    compare.add_argument("--label-schema", required=True)
    compare.add_argument("--asset", action="append", default=[])
    compare.add_argument("--run-id")
    migrate = commands.add_parser("migrate")
    migrate.add_argument("source")
    migrate.add_argument("destination")
    migrate.add_argument("--descriptor", required=True)
    migrate.add_argument("--write", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "describe":
        try:
            descriptor, payload_hash, payload_size = inspect_checkpoint(
                args.checkpoint, verify_payload=False
            )
            result = {
                "format": "checkpoint-envelope-v2",
                "descriptor": descriptor.to_dict(),
                "payload_sha256": payload_hash,
                "payload_size": payload_size,
            }
        except Exception:
            result = inspect_legacy_checkpoint(args.checkpoint)
    elif args.command == "verify":
        envelope = load_checkpoint(args.checkpoint, map_location="cpu")
        result = {
            "status": "verified",
            "checkpoint_id": envelope.descriptor.checkpoint_id,
            "payload_sha256": envelope.payload_sha256,
        }
    elif args.command == "compare":
        envelope = load_checkpoint(args.checkpoint, map_location="cpu")
        verify_checkpoint_compatibility(
            envelope,
            config_sha256=args.config_sha256,
            feature_schema_version=args.feature_schema,
            label_schema_version=args.label_schema,
            asset_identities=_asset_pairs(args.asset),
            run_id=args.run_id,
        )
        result = {"status": "compatible", "checkpoint_id": envelope.descriptor.checkpoint_id}
    else:
        descriptor_value = json.loads(Path(args.descriptor).read_text(encoding="utf-8"))
        result = migrate_legacy_checkpoint(
            args.source,
            args.destination,
            CheckpointDescriptor.from_dict(descriptor_value),
            write=args.write,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
