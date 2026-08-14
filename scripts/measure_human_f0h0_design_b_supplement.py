"""Run the read-only F0-H0 Design B supplement measurement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learned_ai.evaluation.human_f0h0_design_b_supplement import (  # noqa: E402
    RESULT_SCHEMA,
    SplitRetestError,
    load_measurement_inputs,
    load_plan,
    load_result,
    run_measurement,
    write_sealed_json,
)


DEFAULT_F0D0 = Path(
    "docs/evidence/"
    "f0-d0-human-raw-reconstructability-manifest-2026-08-14.json"
)
DEFAULT_PREVIOUS_RESULT = Path(
    "docs/evidence/"
    "f0-h0-corrected-split-feasibility-manifest-2026-08-15.json"
)
DEFAULT_PLAN = Path(
    "docs/experiments/f0-h0-design-b-supplement-measurement-v1.json"
)
DEFAULT_RESULT = Path(
    "docs/evidence/f0-h0-design-b-supplement-manifest-2026-08-15.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Design B support, second-level splits, and ring16 "
            "comparators without selecting a split."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=Path("."))
        command.add_argument("--f0d0-manifest", type=Path, default=DEFAULT_F0D0)
        command.add_argument(
            "--previous-result",
            type=Path,
            default=DEFAULT_PREVIOUS_RESULT,
        )
        command.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        command.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    inputs = load_measurement_inputs(
        f0d0_path=root / args.f0d0_manifest,
        previous_result_path=root / args.previous_result,
    )
    plan, plan_sha = load_plan(root / args.plan)
    if args.command == "run":

        def progress(current: int, total: int) -> None:
            if current == total or current % 2_500 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "strict_phase_and_ring16_replay",
                            "games_complete": current,
                            "games_total": total,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        payload = run_measurement(
            repository_root=root,
            inputs=inputs,
            plan=plan,
            plan_file_sha256=plan_sha,
            f0d0_manifest_path=args.f0d0_manifest,
            previous_result_path=args.previous_result,
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
            "result": args.result.as_posix(),
            "decision": None,
            "recommendation": None,
        }

    result, result_sha = load_result(root / args.result)
    if result["lineage"]["plan_identity"] != plan["plan_identity"]:
        raise SplitRetestError("result plan lineage differs")
    if result["lineage"]["plan_file_sha256"] != plan_sha:
        raise SplitRetestError("result plan file SHA-256 differs")
    if (
        result["lineage"]["f0d0_manifest_identity"]
        != inputs.boundary.manifest_identity
    ):
        raise SplitRetestError("result F0-D0 lineage differs")
    if result["decision"] is not None or result["recommendation"] is not None:
        raise SplitRetestError("measurement result makes a decision")
    if any(result["prohibited_operations_observed"].values()):
        raise SplitRetestError("measurement observed a prohibited operation")
    if any(
        result["access_audit"][field] != 0
        for field in (
            "humandb_reads",
            "database_writes",
            "malom_queries",
            "source_pool_2eb04f54_artifact_reads",
            "source_pool_records_consumed",
        )
    ):
        raise SplitRetestError("measurement crossed a prohibited boundary")
    return {
        "schema_version": RESULT_SCHEMA,
        "verified": True,
        "result_identity": result["result_identity"],
        "result_file_sha256": result_sha,
        "decision": None,
        "recommendation": None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _execute(args)
    except (SplitRetestError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "fatal_stop", "error": str(exc)}))
        return 1
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
