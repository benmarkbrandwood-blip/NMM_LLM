"""Characterize the frozen B2 nonfinal partitions and benchmark Malom cost."""

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
    RESULT_SCHEMA,
    load_membership,
    load_plan,
    load_result,
    run_characterization,
    verify_implementation_artifacts,
    write_sealed_json,
)
from learned_ai.evaluation.human_f0h0_feasibility import (  # noqa: E402
    F0H0Error,
    load_f0d0_boundary,
)
from learned_ai.evaluation.human_f0h0_split_retest import (  # noqa: E402
    SplitRetestError,
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
DEFAULT_RESULT = Path(
    "docs/evidence/"
    "f0-h0-design-b2-freeze-characterization-manifest-2026-08-15.json"
)
DEFAULT_PATHS = Path("data/training_paths.local.json")
DEFAULT_MALOM_MANIFEST = Path("data/manifests/malom-sector-corrected-v1.json")


def _paths_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B2FreezeError("cannot read the local training path registry") from exc
    if not isinstance(value, dict):
        raise B2FreezeError("local training path registry is not an object")
    return value


def _malom_path(args: argparse.Namespace) -> Path:
    if args.malom_path is not None:
        return args.malom_path
    value = _paths_config(args.paths_config).get("malom_db_path")
    if not isinstance(value, str) or not value:
        raise B2FreezeError("malom_db_path is absent from the local registry")
    return Path(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or verify the frozen B2 characterization."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=Path("."))
        command.add_argument("--f0d0-manifest", type=Path, default=DEFAULT_F0D0)
        command.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        command.add_argument(
            "--membership",
            type=Path,
            default=DEFAULT_MEMBERSHIP,
        )
        command.add_argument("--result", type=Path, default=DEFAULT_RESULT)
        command.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
        command.add_argument("--malom-path", type=Path)
        command.add_argument(
            "--malom-manifest",
            type=Path,
            default=DEFAULT_MALOM_MANIFEST,
        )
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    plan, plan_sha = load_plan(root / args.plan)
    verify_implementation_artifacts(root, plan)
    membership, membership_sha = load_membership(root / args.membership)
    if args.command == "run":
        boundary = load_boundary(root / args.f0d0_manifest)
        raw_boundary = load_f0d0_boundary(root / args.f0d0_manifest)

        def progress(current: int, total: int) -> None:
            if current == total or current % 2_500 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "nonfinal_strict_characterization",
                            "games_complete": current,
                            "games_total": total,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        payload = run_characterization(
            repository_root=root,
            plan=plan,
            plan_file_sha256=plan_sha,
            membership=membership,
            membership_file_sha256=membership_sha,
            boundary=boundary,
            raw_boundary=raw_boundary,
            malom_path=_malom_path(args),
            malom_manifest_path=root / args.malom_manifest,
            progress=progress,
        )
        result = write_sealed_json(
            root / args.result,
            payload,
            identity_field="result_identity",
        )
        return {
            "schema_version": RESULT_SCHEMA,
            "status": result["status"],
            "result_identity": result["result_identity"],
            "membership_identity": membership["membership_identity"],
            "malom_cost_decision": result["malom_cost_benchmark"]["projection"][
                "decision"
            ],
        }
    result, result_sha = load_result(root / args.result)
    lineage = result.get("lineage", {})
    if (
        lineage.get("plan_identity") != plan.get("plan_identity")
        or lineage.get("plan_file_sha256") != plan_sha
        or lineage.get("membership_identity")
        != membership.get("membership_identity")
        or lineage.get("membership_file_sha256") != membership_sha
    ):
        raise B2FreezeError("B2 characterization lineage differs")
    if result.get("final_test") != {
        "games": 847,
        "session_ids_identity": membership["partitions"]["final-test"][
            "session_ids_identity"
        ],
        "sealed": True,
        "content_statistics": None,
    }:
        raise B2FreezeError("B2 final-test seal differs")
    if any(
        result["access_audit"].get(field) != 0
        for field in (
            "final_test_raw_game_files_opened",
            "final_test_decisions_loaded",
            "final_test_derived_features_loaded",
            "human_db_reads",
            "database_writes",
            "source_pool_2eb04f54_reads",
            "source_pool_2eb04f54_records_consumed",
        )
    ):
        raise B2FreezeError("B2 characterization crossed a protected boundary")
    if any(
        result["scope"].get(field) is not False
        for field in (
            "independent_support_computed",
            "modifiable_state_reachability_computed",
            "concentration_computed",
            "product_effect_upper_bound_computed",
        )
    ):
        raise B2FreezeError("B2 characterization contains a screening endpoint")
    return {
        "schema_version": RESULT_SCHEMA,
        "verified": True,
        "result_identity": result["result_identity"],
        "result_file_sha256": result_sha,
        "membership_identity": membership["membership_identity"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _execute(args)
    except (
        B2FreezeError,
        F0H0Error,
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
