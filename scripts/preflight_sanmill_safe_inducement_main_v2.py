#!/usr/bin/env python3
"""Create the one-time authorization and run the zero-measurement preflight."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.human_f0h0_feasibility import (
    sha256_file,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_AUTHORIZATION_SCHEMA,
    MAIN_POOL_SCHEMA,
    MAIN_PREFLIGHT_SCHEMA,
    PHASES,
    SafeInducementError,
    load_main_authorization,
    load_main_plan,
    load_main_preflight,
    load_state_pool,
    run_determinism_gate,
)
from learned_ai.training.sanmill_referee import (
    inspect_sanmill_training_installation,
    training_installation_record,
)


def _paths(config_path: Path) -> dict[str, object]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("local path registry is not an object")
    return value


def _local_path(value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"local path is absent: {key}")
    path = Path(value)
    return path if path.is_absolute() else (_ROOT / path).resolve()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _run_gate(command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    record = {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_tail": result.stdout[-4_000:],
        "stderr_tail": result.stderr[-4_000:],
    }
    if result.returncode:
        raise SafeInducementError(f"preflight verification failed: {command}")
    return record


def _fixtures(pool: dict) -> list[dict[str, object]]:
    fixtures: list[dict[str, object]] = []
    for phase in PHASES:
        rows = sorted(
            (row for row in pool["states"] if row["phase"] == phase),
            key=lambda row: (row["logical_ply"], row["state_id"]),
        )
        eligible = [row for row in rows if not row["a_pos"][0]["board_terminal"]]
        if len(eligible) < 2:
            raise SafeInducementError("insufficient nonterminal determinism fixtures")
        fixtures.extend(
            {
                "state_id": row["state_id"],
                "a_pos_index": 0,
                "selection": (
                    "two lowest logical-ply states per phase having a "
                    "nonterminal canonical first A_pos action"
                ),
            }
            for row in eligible[:2]
        )
    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/sanmill-safe-inducement-mechanism-v2.json",
    )
    parser.add_argument(
        "--state-pool",
        default=(
            "docs/experiments/sanmill-safe-inducement-main-"
            "state-pool-v2.json"
        ),
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-safe-inducement-main-v2/"
            "authorization.json"
        ),
    )
    parser.add_argument(
        "--preflight-output",
        default=(
            "docs/evidence/sanmill-safe-inducement-main-v2-"
            "preflight-2026-08-16.json"
        ),
    )
    parser.add_argument("--prior-preflight")
    parser.add_argument(
        "--run-output-dir",
        default=(
            "out/evaluation/sanmill-safe-inducement-main-v2-"
            "20260816-001"
        ),
    )
    parser.add_argument("--paths-config", default="data/training_paths.local.json")
    parser.add_argument(
        "--malom-manifest",
        default="data/manifests/malom-sector-corrected-v1.json",
    )
    args = parser.parse_args()

    plan_path = _ROOT / args.plan
    pool_path = _ROOT / args.state_pool
    authorization_path = _ROOT / args.authorization
    preflight_path = _ROOT / args.preflight_output
    run_output = _ROOT / args.run_output_dir
    if preflight_path.exists() or run_output.exists():
        parser.error("new preflight and run output must both be absent")
    if args.prior_preflight:
        if not authorization_path.is_file():
            parser.error("existing authorization is required for a correction")
    elif authorization_path.exists():
        parser.error("authorization must be absent for an initial preflight")

    plan, plan_file_sha = load_main_plan(plan_path)
    pool, pool_file_sha = load_state_pool(pool_path, schema=MAIN_POOL_SCHEMA)
    if (
        pool.get("plan_binding", {}).get("plan_identity") != plan["plan_identity"]
        or int(pool["state_count"]) != 360
    ):
        parser.error("main plan and child state pool differ")
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    authorization_payload = {
        "schema_version": MAIN_AUTHORIZATION_SCHEMA,
        "status": "authorized_once_measurement_unconsumed",
        "operator": "product-owner-direct",
        "authorized_on": "2026-08-16",
        "authorization_basis": (
            "Direct product-owner authorization in the task that freezes "
            "protocol v2 and permits exactly one bounded main execution."
        ),
        "grant_count": 1,
        "plan": {
            "identity": plan["plan_identity"],
            "file_sha256": plan_file_sha,
            "tracked_file": args.plan,
        },
        "state_pool": {
            "identity": pool["pool_identity"],
            "membership_identity": pool["state_membership_identity"],
            "file_sha256": pool_file_sha,
            "tracked_file": args.state_pool,
            "states": pool["state_count"],
        },
        "execution_scope": plan["main_experiment"]["resource_envelope"],
        "permitted": [
            "one zero-measurement deterministic preflight within the envelope",
            "one exhaustive execution under the exact frozen plan and pool",
            "read-only Sanmill single-step searches",
            "read-only sector-corrected-v1 Malom queries",
        ],
        "prohibited": [
            "automatic retry, resume, recovery, or extension",
            "second execution or result-dependent batching",
            "complete games, model loads, training, or database writes",
            "promotion, deployment, publication, or release",
        ],
        "consumption_rule": (
            "Consumed when the first exhaustive measurement search starts; "
            "preflight searches count against resources but are not measurement."
        ),
        "host_interruption": (
            "No recovery is authorized; an exact missing-suffix continuation "
            "requires separate product-owner authorization."
        ),
        "source_commit_at_freeze": source_commit,
        "source_tree_at_freeze": source_tree,
    }
    prior_preflight = None
    if args.prior_preflight:
        authorization, authorization_file_sha = load_main_authorization(
            authorization_path
        )
        prior_preflight, _prior_file_sha = load_main_preflight(
            _ROOT / args.prior_preflight
        )
        prior_marker = (
            _ROOT
            / str(prior_preflight["run_output_namespace"])
            / "measurement-started.json"
        )
        if prior_marker.exists():
            parser.error("prior preflight has a consumed measurement marker")
    else:
        authorization_path.parent.mkdir(parents=True, exist_ok=False)
        authorization = write_sealed_json(
            authorization_path,
            authorization_payload,
            identity_field="authorization_identity",
        )
        loaded_authorization, authorization_file_sha = load_main_authorization(
            authorization_path
        )
        if loaded_authorization["authorization_identity"] != authorization[
            "authorization_identity"
        ]:
            raise SafeInducementError("authorization verification differs")

    run_output.mkdir(parents=True, exist_ok=False)
    verification_started = time.perf_counter()
    python = str(_ROOT / ".venv/Scripts/python.exe")
    focused = _run_gate(
        [
            python,
            "-m",
            "pytest",
            "tests/test_sanmill_safe_inducement.py",
            "-q",
            "--basetemp",
            str(run_output / "pytest-focused"),
        ]
    )
    ruff_targets = [
        "learned_ai/evaluation/sanmill_safe_inducement.py",
        "scripts/freeze_sanmill_safe_inducement_plan_v2.py",
        "scripts/freeze_sanmill_safe_inducement_main_pool_v2.py",
        "scripts/preflight_sanmill_safe_inducement_main_v2.py",
        "scripts/run_sanmill_safe_inducement_main_v2.py",
        "tests/test_sanmill_safe_inducement.py",
    ]
    ruff = _run_gate(["ruff", "check", *ruff_targets])

    paths = _paths(_ROOT / args.paths_config)
    checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
    malom_path = _local_path(paths.get("malom_db_path"), key="malom")
    installation = inspect_sanmill_training_installation(checkout)
    runtime = training_installation_record(
        installation,
        seed=int(plan["sanmill_contract"]["seed"]),
    )
    if runtime["identity"] != plan["sanmill_contract"]["runtime_identity"]:
        raise SafeInducementError("Sanmill runtime identity differs from protocol")
    malom = verify_malom_snapshot(
        malom_path=malom_path,
        manifest_path=_ROOT / args.malom_manifest,
        full_hash=False,
    )
    if (
        malom.get("trust_level") != "sector-corrected-v1"
        or malom.get("content_sha256")
        != plan["input_identities"]["malom_content_sha256"]
    ):
        raise SafeInducementError("Malom snapshot differs from protocol")

    if prior_preflight is None:
        runtime_plan = dict(plan)
        runtime_plan["determinism_gate"] = {
            **plan["determinism_gate"],
            "fixtures": _fixtures(pool),
        }
        engine_queries = 0

        def count_query() -> None:
            nonlocal engine_queries
            engine_queries += 1
            if engine_queries > int(
                plan["main_experiment"]["resource_envelope"][
                    "maximum_engine_single_step_queries"
                ]
            ):
                raise SafeInducementError("preflight engine query ceiling exceeded")

        determinism = run_determinism_gate(
            installation=installation,
            pool=pool,
            plan=runtime_plan,
            query_counter=count_query,
        )
        if not determinism["passed"]:
            raise SafeInducementError("determinism gate failed")
        previous_preflight_seconds = 0.0
    else:
        determinism = prior_preflight["determinism"]
        engine_queries = int(
            prior_preflight["resource_components"]["preflight"][
                "engine_single_step_queries"
            ]
        )
        previous_preflight_seconds = float(
            prior_preflight["resource_components"]["preflight"]["active_seconds"]
        )
    preflight_seconds = time.perf_counter() - verification_started
    pool_resource = {
        "engine_single_step_queries": 0,
        "malom_queries": int(pool["resource_use"]["malom_queries"]),
        "active_seconds": float(
            pool["resource_use"]["construction_active_seconds"]
        ),
    }
    preflight_resource = {
        "engine_single_step_queries": engine_queries,
        "malom_queries": 0,
        "active_seconds": previous_preflight_seconds + preflight_seconds,
    }
    aggregate = {
        key: pool_resource[key] + preflight_resource[key]
        for key in ("engine_single_step_queries", "malom_queries", "active_seconds")
    }
    envelope = plan["main_experiment"]["resource_envelope"]
    if (
        aggregate["engine_single_step_queries"]
        >= int(envelope["maximum_engine_single_step_queries"])
        or aggregate["malom_queries"] >= int(envelope["maximum_malom_queries"])
        or aggregate["active_seconds"] >= float(envelope["maximum_active_seconds"])
    ):
        raise SafeInducementError("resource ceiling reached during preflight")
    preflight_payload = {
        "schema_version": MAIN_PREFLIGHT_SCHEMA,
        "status": "ready_for_one_authorized_execution",
        "measurement_searches": 0,
        "supersedes_preflight_identity": (
            prior_preflight["preflight_identity"] if prior_preflight else None
        ),
        "technical_correction": (
            {
                "reason": (
                    "pre-measurement tasklist process inspection was denied; "
                    "the replacement uses fail-closed PowerShell Get-Process"
                ),
                "prior_measurement_marker_absent": True,
                "additional_determinism_searches": 0,
                "additional_measurement_searches": 0,
            }
            if prior_preflight
            else None
        ),
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": plan_file_sha,
        "state_pool_identity": pool["pool_identity"],
        "state_pool_membership_identity": pool["state_membership_identity"],
        "state_pool_file_sha256": pool_file_sha,
        "authorization_identity": authorization["authorization_identity"],
        "authorization_file_sha256": authorization_file_sha,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "run_output_namespace": args.run_output_dir,
        "run_output_was_absent_before_preflight": True,
        "verification": {"focused_pytest": focused, "task_scope_ruff": ruff},
        "sanmill_runtime": runtime,
        "malom_snapshot": malom,
        "determinism": determinism,
        "resource_components": {
            "pool_construction": pool_resource,
            "preflight": preflight_resource,
        },
        "aggregate_resource_use_before_measurement": aggregate,
        "protected_access": {
            "guard_test_executed_and_failed_closed": True,
            "official_selection_content_reads": 0,
            "official_confirmation_content_reads": 0,
            "official_final_test_content_reads": 0,
            "research_confirmation_content_reads": 0,
            "source_pool_2eb04f54_reads_or_consumption": 0,
        },
        "forbidden_operations": {
            "complete_games": 0,
            "model_loads": 0,
            "training_or_weight_updates": 0,
            "database_writes": 0,
        },
        "implementation_files": {
            path: sha256_file(_ROOT / path)
            for path in ruff_targets
            if not path.startswith("tests/")
        },
    }
    preflight = write_sealed_json(
        preflight_path,
        preflight_payload,
        identity_field="preflight_identity",
    )
    (run_output / "authorization-binding.json").write_text(
        json.dumps(
            {
                "authorization_identity": authorization["authorization_identity"],
                "preflight_identity": preflight["preflight_identity"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(authorization["authorization_identity"])
    print(preflight["preflight_identity"])
    print(aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
