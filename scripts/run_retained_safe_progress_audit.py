"""Run or verify the zero-game retained-v3/v4 safe-progress audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.data.data_contract import (  # noqa: E402
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.evaluation.heldout_evaluation import write_new_canonical  # noqa: E402
from learned_ai.evaluation.retained_passivity_diagnostic import (  # noqa: E402
    load_game_ledger,
    recompute_diagnostic,
    sha256_file,
)
from learned_ai.evaluation.retained_safe_progress_audit import (  # noqa: E402
    ENGINEERING_Z,
    EXPECTED_GAMES,
    MAX_PRIMARY_HALF_WIDTH,
    MIN_SAFE_CAPTURE_OPPORTUNITIES,
    REPORT_SCHEMA,
    RetainedSafeProgressAuditError,
    recompute_safe_progress_audit,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from scripts.run_retained_passivity_diagnostic import (  # noqa: E402
    DEFAULT_PATHS,
    load_plan as load_source_plan,
    resolve_paths as resolve_source_paths,
)


DEFAULT_PLAN = (
    _ROOT
    / "docs"
    / "experiments"
    / "sanmill-retained-v3-v4-safe-progress-audit-v1.json"
)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedSafeProgressAuditError(
            f"cannot read strict JSON: {path.name}"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise RetainedSafeProgressAuditError(f"{path.name} is not canonical JSON")
    return value


def load_audit_plan(path: str | Path) -> dict[str, Any]:
    plan = _strict_json(Path(path).resolve(strict=True))
    identity = plan.get("plan_identity")
    body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if (
        plan.get("schema_version") != "nmm.retained-safe-progress-audit-plan.v1"
        or not isinstance(identity, str)
        or canonical_sha256(body) != identity
    ):
        raise RetainedSafeProgressAuditError("safe-progress audit plan differs")
    if plan.get("workload") != {
        "new_games": 0,
        "model_updates": 0,
        "database_writes": 0,
        "checkpoint_writes": 0,
    }:
        raise RetainedSafeProgressAuditError("safe-progress workload differs")
    analysis = plan.get("analysis")
    if not isinstance(analysis, Mapping) or (
        analysis.get("engineering_interval")
        != {
            "interpretation": (
                "fixed-corpus paired variation summary, not population inference"
            ),
            "maximum_primary_half_width": MAX_PRIMARY_HALF_WIDTH,
            "method": "normal-interval-on-matched-per-game-rate-difference",
            "z": ENGINEERING_Z,
        }
        or analysis.get("minimum_safe_capture_opportunities_per_candidate")
        != MIN_SAFE_CAPTURE_OPPORTUNITIES
    ):
        raise RetainedSafeProgressAuditError("safe-progress analysis differs")
    if plan.get("claim_boundary") != {
        "automatic_training_setting_selection": False,
        "development_corpus_reused": True,
        "held_out_strength_claim": False,
        "malom_history_aware": False,
        "playing_strength_claim": False,
        "promotion_or_publication": False,
        "refresh_causal_claim": False,
        "training_or_update": False,
        "zero_game_reanalysis": True,
    }:
        raise RetainedSafeProgressAuditError("safe-progress claim boundary differs")
    if (
        plan.get("source", {}).get("games") != EXPECTED_GAMES
        or plan.get("implementation", {}).get("branch") != "dev"
        or plan.get("output", {}).get("path")
        != (
            "learned_ai/checkpoints/evaluation/"
            "sanmill-retained-v3-v4-passivity-diagnostic-v1/"
            "safe-progress-report.json"
        )
        or plan.get("malom", {}).get("history_aware") is not False
        or plan.get("malom", {}).get("read_only") is not True
        or plan.get("malom", {}).get("label_version") != "sector-corrected-v1"
    ):
        raise RetainedSafeProgressAuditError("safe-progress resources differ")
    return plan


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _load_source(
    plan: Mapping[str, Any], paths_config: str | Path
) -> tuple[dict[str, Any], list[Mapping[str, Any]], Any, dict[str, Any]]:
    source_plan_path = (_ROOT / plan["source"]["plan_path"]).resolve(strict=True)
    source_plan = load_source_plan(source_plan_path)
    if source_plan["plan_identity"] != plan["source"]["plan_identity"]:
        raise RetainedSafeProgressAuditError("source plan identity differs")
    paths = resolve_source_paths(
        source_plan,
        plan_path=source_plan_path,
        paths_config=paths_config,
    )
    expected_files = plan["source"]["files"]
    observed_paths = {
        "spec": paths.spec,
        "ledger": paths.ledger,
        "report": paths.report,
        "completion": paths.completion,
    }
    for key, path in observed_paths.items():
        expected = expected_files[key]
        expected_path = (_ROOT / expected["path"]).resolve()
        if (
            path != expected_path
            or not path.is_file()
            or sha256_file(path) != expected["sha256"]
        ):
            raise RetainedSafeProgressAuditError(f"source {key} differs")

    spec = _strict_json(paths.spec)
    report = _strict_json(paths.report)
    completion = _strict_json(paths.completion)
    if spec.get("spec_identity") != plan["source"]["spec_identity"]:
        raise RetainedSafeProgressAuditError("source spec identity differs")
    if report.get("result_identity") != plan["source"]["result_identity"]:
        raise RetainedSafeProgressAuditError("source result identity differs")
    if completion.get("completion_identity") != plan["source"]["completion_identity"]:
        raise RetainedSafeProgressAuditError("source completion identity differs")
    records, tail = load_game_ledger(spec, paths.ledger)
    recomputed = recompute_diagnostic(spec, paths.ledger)
    if recomputed != report or tail != completion.get("ledger_tail_record_sha256"):
        raise RetainedSafeProgressAuditError("source report does not recompute")

    manifest = load_dataset_manifest(paths.malom_manifest)
    expected_manifest = (_ROOT / plan["malom"]["manifest"]).resolve()
    if (
        paths.malom_manifest != expected_manifest
        or manifest.manifest_sha256 != plan["malom"]["identity"]
    ):
        raise RetainedSafeProgressAuditError("Malom manifest identity differs")
    snapshot = verify_dataset_snapshot(paths.malom_db, manifest, full_hash=False)
    malom = ExternalSolvedDB(str(paths.malom_db), strict=True)
    if not malom.is_available():
        raise RetainedSafeProgressAuditError("Malom is unavailable")
    return spec, records, malom, {
        "source_ledger_tail_record_sha256": tail,
        "malom_snapshot": snapshot,
        "source_result_identity": report["result_identity"],
    }


def recompute(plan_path: str | Path, paths_config: str | Path) -> dict[str, Any]:
    plan = load_audit_plan(plan_path)
    head = _head()
    implementation = str(plan["implementation"]["commit"])
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation, head],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RetainedSafeProgressAuditError(
            "audit implementation is not an ancestor of HEAD"
        ) from exc
    spec, records, malom, source = _load_source(plan, paths_config)
    try:
        return recompute_safe_progress_audit(
            source_spec=spec,
            source_records=records,
            source_ledger_sha256=plan["source"]["files"]["ledger"]["sha256"],
            source_result_identity=source["source_result_identity"],
            audit_plan_identity=plan["plan_identity"],
            implementation_commit=implementation,
            malom=malom,
        )
    finally:
        inner = getattr(malom, "_malom", None)
        if inner is not None:
            inner.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("run")
    commands.add_parser("verify")
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = load_audit_plan(args.plan)
    output = (_ROOT / plan["output"]["path"]).resolve()
    if args.command == "preflight":
        _, records, malom, source = _load_source(plan, args.paths_config)
        inner = getattr(malom, "_malom", None)
        if inner is not None:
            inner.close()
        print(
            json.dumps(
                {
                    "verdict": (
                        "ready_for_verification"
                        if output.exists()
                        else "ready_for_zero_game_audit"
                    ),
                    "plan_identity": plan["plan_identity"],
                    "new_games": 0,
                    "source_records": len(records),
                    "source_result_identity": source["source_result_identity"],
                    "malom_snapshot": source["malom_snapshot"],
                    "output_exists": output.exists(),
                },
                indent=2,
            )
        )
        return 0

    result = recompute(args.plan, args.paths_config)
    if result.get("schema_version") != REPORT_SCHEMA:
        raise RetainedSafeProgressAuditError("safe-progress report schema differs")
    if args.command == "run":
        write_new_canonical(output, result)
    else:
        observed = _strict_json(output)
        if observed != result:
            raise RetainedSafeProgressAuditError("stored safe-progress report differs")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
