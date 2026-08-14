"""Run or verify the frozen B2 train-only F0-H0 rejection screen."""

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
    load_membership,
)
from learned_ai.evaluation.human_f0h0_b2_train_screen import (  # noqa: E402
    EXPECTED_MEMBERSHIP_IDENTITY,
    TrainScreenError,
    load_characterization_identity,
    load_screen_plan,
    load_screen_result,
    run_train_screen,
    verify_implementation_artifacts,
    write_screen_result,
)
from learned_ai.evaluation.human_f0h0_feasibility import (  # noqa: E402
    F0H0Error,
    load_f0d0_boundary,
)


DEFAULT_PLAN = Path("docs/experiments/f0-h0-b2-train-rejection-screen-v1.json")
DEFAULT_MEMBERSHIP = Path("docs/experiments/f0-h0-design-b2-frozen-membership-v1.json")
DEFAULT_CHARACTERIZATION = Path(
    "docs/evidence/f0-h0-design-b2-freeze-characterization-manifest-2026-08-15.json"
)
DEFAULT_F0D0 = Path(
    "docs/evidence/f0-d0-human-raw-reconstructability-manifest-2026-08-14.json"
)
DEFAULT_MALOM_MANIFEST = Path("data/manifests/malom-sector-corrected-v1.json")
DEFAULT_RULESET = Path("data/rulesets/nmm-training-core@2.json")
DEFAULT_OUTPUT = Path(
    "docs/evidence/f0-h0-b2-train-rejection-screen-manifest-2026-08-15.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")


def _load_paths(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainScreenError(f"local path registry is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise TrainScreenError("local path registry root is not an object")
    return value


def _malom_path(args: argparse.Namespace, root: Path) -> Path:
    if args.malom_path is not None:
        return args.malom_path.resolve()
    config = _load_paths(root / args.paths_config)
    value = config.get("malom_db_path")
    if not isinstance(value, str) or not value.strip():
        raise TrainScreenError("malom_db_path is absent from the local path registry")
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--repository-root", type=Path, default=Path.cwd())
    execute.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    execute.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    execute.add_argument(
        "--characterization",
        type=Path,
        default=DEFAULT_CHARACTERIZATION,
    )
    execute.add_argument("--f0d0-manifest", type=Path, default=DEFAULT_F0D0)
    execute.add_argument("--malom-manifest", type=Path, default=DEFAULT_MALOM_MANIFEST)
    execute.add_argument("--ruleset", type=Path, default=DEFAULT_RULESET)
    execute.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    execute.add_argument("--malom-path", type=Path)
    execute.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repository-root", type=Path, default=Path.cwd())
    verify.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    verify.add_argument("--result", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    plan, plan_sha = load_screen_plan(root / args.plan)
    verify_implementation_artifacts(root, plan)
    membership, membership_sha = load_membership(root / args.membership)
    if membership.get("membership_identity") != EXPECTED_MEMBERSHIP_IDENTITY:
        raise TrainScreenError("official B2 membership identity differs")
    _characterization, characterization_sha = load_characterization_identity(
        root / args.characterization
    )
    boundary = load_f0d0_boundary(root / args.f0d0_manifest)

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    result = run_train_screen(
        repository_root=root,
        boundary=boundary,
        plan=plan,
        plan_file_sha256=plan_sha,
        membership=membership,
        membership_file_sha256=membership_sha,
        characterization_file_sha256=characterization_sha,
        malom_path=_malom_path(args, root),
        malom_manifest_path=root / args.malom_manifest,
        ruleset_path=root / args.ruleset,
        progress=progress,
    )
    sealed = write_screen_result(root / args.output, result)
    return {
        "status": sealed["status"],
        "decision": sealed["decision"],
        "result_identity": sealed["result_identity"],
        "plan_identity": plan["plan_identity"],
        "analysis_games": sealed["sample"]["analysis_games"],
        "analysis_decisions": sealed["sample"]["analysis_decisions"],
        "protected_content_reads": sealed["prohibited_operations_observed"][
            "protected_partition_content_reads"
        ],
    }


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve()
    plan, plan_sha = load_screen_plan(root / args.plan)
    verify_implementation_artifacts(root, plan)
    result, result_sha = load_screen_result(root / args.result)
    if (
        result.get("lineage", {}).get("plan_identity") != plan["plan_identity"]
        or result.get("lineage", {}).get("plan_file_sha256") != plan_sha
    ):
        raise TrainScreenError("screen result plan lineage differs")
    access = result.get("access_audit", {})
    prohibited = result.get("prohibited_operations_observed", {})
    if (
        access.get("statistics_partitions") != ["train"]
        or access.get("selection_raw_games_or_decisions_or_features_read") != 0
        or access.get("confirmation_raw_games_or_decisions_or_features_read") != 0
        or access.get("final_test_raw_games_or_decisions_or_features_read") != 0
        or prohibited.get("protected_partition_content_reads") != 0
        or prohibited.get("database_writes_or_rebuilds") != 0
        or prohibited.get("source_pool_records_read_or_consumed") != 0
    ):
        raise TrainScreenError("screen result protected-access boundary differs")
    four_a = (
        result.get("dimensions", {})
        .get("product_effect_upper_bound", {})
        .get("four_a_estimability", {})
    )
    four_b = (
        result.get("dimensions", {})
        .get("product_effect_upper_bound", {})
        .get("four_b_state_conditioned_effect", {})
    )
    if (
        four_a.get("decision") == "state_level_not_empirically_estimable"
        and four_b.get("status") != "skipped_because_four_a_failed"
    ):
        raise TrainScreenError("four-B ran after four-A failed")
    return {
        "status": "verified",
        "decision": result["decision"],
        "result_identity": result["result_identity"],
        "result_file_sha256": result_sha,
        "plan_identity": plan["plan_identity"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _execute(args) if args.command == "execute" else _verify(args)
    except (TrainScreenError, B2FreezeError, F0H0Error, OSError) as exc:
        print(json.dumps({"status": "fatal_stop", "error": str(exc)}))
        return 1
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
