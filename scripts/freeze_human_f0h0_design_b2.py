"""Freeze or verify the official F0-H0 Design B2 membership."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learned_ai.evaluation.human_f0h0_b2_freeze import (  # noqa: E402
    B2FreezeError,
    MEMBERSHIP_SCHEMA,
    build_membership,
    load_membership,
    load_plan,
    verify_implementation_artifacts,
    write_sealed_json,
)
from learned_ai.evaluation.human_f0h0_split_retest import (  # noqa: E402
    SplitRetestError,
    canonical_sha256,
    load_boundary,
)


DEFAULT_F0D0 = Path(
    "docs/evidence/"
    "f0-d0-human-raw-reconstructability-manifest-2026-08-14.json"
)
DEFAULT_PLAN = Path(
    "docs/experiments/f0-h0-design-b2-freeze-and-characterization-v1.json"
)
DEFAULT_MEMBERSHIP = Path(
    "docs/experiments/f0-h0-design-b2-frozen-membership-v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze B2 membership from F0-D0 metadata only."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=Path("."))
        command.add_argument("--f0d0-manifest", type=Path, default=DEFAULT_F0D0)
        command.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        command.add_argument(
            "--membership",
            type=Path,
            default=DEFAULT_MEMBERSHIP,
        )
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    plan, plan_sha = load_plan(root / args.plan)
    verify_implementation_artifacts(root, plan)
    boundary = load_boundary(root / args.f0d0_manifest)
    expected = build_membership(boundary, plan)
    if args.command == "freeze":
        membership = write_sealed_json(
            root / args.membership,
            expected,
            identity_field="membership_identity",
        )
        return {
            "schema_version": MEMBERSHIP_SCHEMA,
            "status": membership["status"],
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": plan_sha,
            "membership_identity": membership["membership_identity"],
            "counts": {
                name: membership["partitions"][name]["games"]
                for name in membership["partitions"]
            },
            "raw_game_files_opened": 0,
        }
    membership, membership_sha = load_membership(root / args.membership)
    expected_identity = canonical_sha256(expected)
    if membership["membership_identity"] != expected_identity:
        raise B2FreezeError("stored B2 membership differs from recomputation")
    if membership.get("plan_identity") != plan.get("plan_identity"):
        raise B2FreezeError("stored B2 membership plan lineage differs")
    return {
        "schema_version": MEMBERSHIP_SCHEMA,
        "verified": True,
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_sha,
        "membership_identity": membership["membership_identity"],
        "membership_file_sha256": membership_sha,
        "raw_game_files_opened": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _execute(args)
    except (
        B2FreezeError,
        SplitRetestError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(json.dumps({"status": "fatal_stop", "error": str(exc)}))
        return 1
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
