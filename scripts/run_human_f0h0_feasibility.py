"""Freeze and execute the read-only F0-H0 rejection screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from learned_ai.evaluation.human_f0h0_feasibility import (
    COST_SCHEMA,
    RESULT_SCHEMA,
    SPLIT_SCHEMA,
    F0H0Error,
    build_cost_decision,
    build_split,
    load_cost_decision,
    load_f0d0_boundary,
    load_plan,
    load_split,
    run_screen,
    write_sealed_json,
)


DEFAULT_F0D0 = Path(
    "docs/evidence/"
    "f0-d0-human-raw-reconstructability-manifest-2026-08-14.json"
)
DEFAULT_PLAN = Path("docs/experiments/f0-h0-human-feasibility-screen-v1.json")
DEFAULT_SPLIT = Path(
    "docs/experiments/f0-h0-human-player-split-membership-v1.json"
)
DEFAULT_COST = Path("docs/experiments/f0-h0-human-malom-cost-decision-v1.json")
DEFAULT_RESULT = Path(
    "docs/evidence/f0-h0-human-feasibility-screen-manifest-2026-08-14.json"
)
DEFAULT_MALOM_MANIFEST = Path("data/manifests/malom-sector-corrected-v1.json")
DEFAULT_PATHS = Path("data/training_paths.local.json")
DEFAULT_RULESET = Path("data/rulesets/nmm-training-core@2.json")


def _paths_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise F0H0Error("cannot read the local training path registry") from exc
    if not isinstance(value, dict):
        raise F0H0Error("local training path registry is not an object")
    return value


def _malom_path(args: argparse.Namespace) -> Path:
    if args.malom_path is not None:
        return Path(args.malom_path)
    value = _paths_config(Path(args.paths_config)).get("malom_db_path")
    if not isinstance(value, str) or not value:
        raise F0H0Error("malom_db_path is absent from the local path registry")
    return Path(value)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--f0d0-manifest", type=Path, default=DEFAULT_F0D0)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only F0-H0 rejection screen in frozen stages."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-split")
    _common(freeze)
    freeze.add_argument("--output", type=Path, default=DEFAULT_SPLIT)

    cost = commands.add_parser("estimate-cost")
    _common(cost)
    cost.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    cost.add_argument("--output", type=Path, default=DEFAULT_COST)
    cost.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    cost.add_argument("--malom-path", type=Path)
    cost.add_argument(
        "--malom-manifest",
        type=Path,
        default=DEFAULT_MALOM_MANIFEST,
    )

    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    run.add_argument("--cost", type=Path, default=DEFAULT_COST)
    run.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    run.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    run.add_argument("--malom-path", type=Path)
    run.add_argument(
        "--malom-manifest",
        type=Path,
        default=DEFAULT_MALOM_MANIFEST,
    )
    run.add_argument("--ruleset", type=Path, default=DEFAULT_RULESET)

    verify = commands.add_parser("verify")
    _common(verify)
    verify.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    verify.add_argument("--cost", type=Path, default=DEFAULT_COST)
    verify.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    boundary = load_f0d0_boundary(root / args.f0d0_manifest)
    plan, plan_sha = load_plan(root / args.plan)
    if args.command == "freeze-split":
        payload = build_split(boundary=boundary, plan=plan)
        sealed = write_sealed_json(
            root / args.output,
            payload,
            identity_field="split_identity",
        )
        return {
            "schema_version": SPLIT_SCHEMA,
            "output": args.output.as_posix(),
            "split_identity": sealed["split_identity"],
            "counts": sealed["counts"],
            "component_count": sealed["component_count"],
            "largest_component": sealed["largest_component"],
        }

    split, split_sha = load_split(
        root / args.split,
        boundary=boundary,
        plan=plan,
    )
    if args.command == "estimate-cost":
        payload = build_cost_decision(
            repository_root=root,
            boundary=boundary,
            plan=plan,
            split=split,
            malom_path=_malom_path(args),
            malom_manifest_path=root / args.malom_manifest,
        )
        sealed = write_sealed_json(
            root / args.output,
            payload,
            identity_field="cost_decision_identity",
        )
        return {
            "schema_version": COST_SCHEMA,
            "output": args.output.as_posix(),
            "cost_decision_identity": sealed["cost_decision_identity"],
            "decision": sealed["decision"],
            "analysis_games": sealed["analysis_games"],
            "projection": sealed["projection"],
        }

    cost, cost_sha = load_cost_decision(
        root / args.cost,
        plan=plan,
        split=split,
    )
    if args.command == "run":
        payload = run_screen(
            repository_root=root,
            boundary=boundary,
            plan=plan,
            plan_file_sha256=plan_sha,
            split=split,
            split_file_sha256=split_sha,
            cost=cost,
            cost_file_sha256=cost_sha,
            malom_path=_malom_path(args),
            malom_manifest_path=root / args.malom_manifest,
            ruleset_path=root / args.ruleset,
        )
        sealed = write_sealed_json(
            root / args.output,
            payload,
            identity_field="result_identity",
        )
        return {
            "schema_version": RESULT_SCHEMA,
            "output": args.output.as_posix(),
            "result_identity": sealed["result_identity"],
            "decision": sealed["decision"],
            "gate_results": sealed["gate_results"],
        }

    result_path = root / args.result
    from learned_ai.evaluation.human_f0h0_feasibility import load_result

    result, result_sha = load_result(result_path)
    if (
        result["lineage"]["plan_identity"] != plan["plan_identity"]
        or result["lineage"]["split_identity"] != split["split_identity"]
        or result["lineage"]["cost_decision_identity"]
        != cost["cost_decision_identity"]
    ):
        raise F0H0Error("result lineage differs")
    if result["access_audit"]["final-test_raw_record_reads"] != 0:
        raise F0H0Error("result opened final-test source records")
    return {
        "schema_version": RESULT_SCHEMA,
        "result_identity": result["result_identity"],
        "result_file_sha256": result_sha,
        "decision": result["decision"],
        "verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _execute(args)
    except (F0H0Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "fatal_stop", "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
