"""Run or verify the zero-game retained-v3/v4 complete-order audit."""

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

from learned_ai.evaluation.heldout_evaluation import write_new_canonical  # noqa: E402
from learned_ai.evaluation.retained_oracle_order_audit import (  # noqa: E402
    ENGINEERING_Z,
    EXPECTED_GAMES,
    MAX_PRIMARY_HALF_WIDTH,
    MIN_ORDERABLE_COVERAGE,
    MIN_ORDERING_OPPORTUNITIES,
    REPORT_SCHEMA,
    RetainedOracleOrderAuditError,
    recompute_oracle_order_audit,
)
from learned_ai.evaluation.retained_passivity_diagnostic import (  # noqa: E402
    sha256_file,
)
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from scripts.run_retained_passivity_diagnostic import DEFAULT_PATHS  # noqa: E402
from scripts.run_retained_safe_progress_audit import (  # noqa: E402
    _load_source,
)


DEFAULT_PLAN = (
    _ROOT
    / "docs"
    / "experiments"
    / "sanmill-retained-v3-v4-oracle-order-audit-v1.json"
)


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetainedOracleOrderAuditError(
            f"cannot read strict JSON: {path.name}"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise RetainedOracleOrderAuditError(f"{path.name} is not canonical JSON")
    return value


def load_audit_plan(path: str | Path) -> dict[str, Any]:
    plan = _strict_json(Path(path).resolve(strict=True))
    identity = plan.get("plan_identity")
    body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if (
        plan.get("schema_version") != "nmm.retained-oracle-order-audit-plan.v1"
        or not isinstance(identity, str)
        or canonical_sha256(body) != identity
    ):
        raise RetainedOracleOrderAuditError("oracle-order audit plan differs")
    if plan.get("workload") != {
        "new_games": 0,
        "model_updates": 0,
        "database_writes": 0,
        "checkpoint_writes": 0,
    }:
        raise RetainedOracleOrderAuditError("oracle-order workload differs")
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
        or analysis.get("minimum_orderable_coverage") != MIN_ORDERABLE_COVERAGE
        or analysis.get("minimum_ordering_opportunities_per_candidate")
        != MIN_ORDERING_OPPORTUNITIES
    ):
        raise RetainedOracleOrderAuditError("oracle-order analysis differs")
    if plan.get("claim_boundary") != {
        "automatic_training_setting_selection": False,
        "complete_malom_order_is_positional": True,
        "development_corpus_reused": True,
        "distance_to_terminal_claim": False,
        "history_aware_liveness": False,
        "passivity_causal_claim": False,
        "playing_strength_claim": False,
        "promotion_or_publication": False,
        "refresh_causal_claim": False,
        "zero_game_reanalysis": True,
    }:
        raise RetainedOracleOrderAuditError("oracle-order claim boundary differs")
    if (
        plan.get("source", {}).get("games") != EXPECTED_GAMES
        or plan.get("implementation", {}).get("branch") != "dev"
        or plan.get("output", {}).get("path")
        != (
            "learned_ai/checkpoints/evaluation/"
            "sanmill-retained-v3-v4-passivity-diagnostic-v1/"
            "oracle-order-report.json"
        )
        or plan.get("malom", {}).get("history_aware") is not False
        or plan.get("malom", {}).get("read_only") is not True
        or plan.get("malom", {}).get("label_version") != "sector-corrected-v1"
    ):
        raise RetainedOracleOrderAuditError("oracle-order resources differ")
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


def _load_safe_progress(plan: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    expected = plan["safe_progress_source"]
    path = (_ROOT / expected["path"]).resolve()
    if not path.is_file() or sha256_file(path) != expected["file_sha256"]:
        raise RetainedOracleOrderAuditError("safe-progress source differs")
    report = _strict_json(path)
    if (
        report.get("schema_version")
        != "nmm.retained-safe-progress-audit-result.v1"
        or report.get("result_identity") != expected["result_identity"]
        or report.get("audit_plan_identity") != expected["audit_plan_identity"]
    ):
        raise RetainedOracleOrderAuditError("safe-progress source identity differs")
    return report, expected["file_sha256"]


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
        raise RetainedOracleOrderAuditError(
            "oracle-order implementation is not an ancestor of HEAD"
        ) from exc
    spec, records, malom, source = _load_source(plan, paths_config)
    safe, safe_file_sha256 = _load_safe_progress(plan)
    try:
        return recompute_oracle_order_audit(
            source_spec=spec,
            source_records=records,
            source_ledger_sha256=plan["source"]["files"]["ledger"]["sha256"],
            source_result_identity=source["source_result_identity"],
            safe_progress_result_identity=safe["result_identity"],
            safe_progress_file_sha256=safe_file_sha256,
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
        safe, safe_file_sha256 = _load_safe_progress(plan)
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
                    "safe_progress_result_identity": safe["result_identity"],
                    "safe_progress_file_sha256": safe_file_sha256,
                    "malom_snapshot": source["malom_snapshot"],
                    "output_exists": output.exists(),
                },
                indent=2,
            )
        )
        return 0

    result = recompute(args.plan, args.paths_config)
    if result.get("schema_version") != REPORT_SCHEMA:
        raise RetainedOracleOrderAuditError("oracle-order report schema differs")
    if args.command == "run":
        write_new_canonical(output, result)
    else:
        observed = _strict_json(output)
        if observed != result:
            raise RetainedOracleOrderAuditError("stored oracle-order report differs")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
