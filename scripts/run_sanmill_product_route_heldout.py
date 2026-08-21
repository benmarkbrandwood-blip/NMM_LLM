#!/usr/bin/env python3
"""Preflight and execute the one-shot held-out product-route comparison."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.human_db import HumanDB
from ai.malom_db import MalomDB
from ai.malom_runtime import resolve_product_malom_runtime
from learned_ai.agents.positional_safety import ProductPositionalSafetyGate
from learned_ai.agents.specialist_router import load_specialist_router
from learned_ai.data.specialist_db import SpecialistDB
from learned_ai.evaluation.human_f0h0_feasibility import (
    canonical_sha256,
    verify_malom_snapshot,
    write_sealed_json,
)
from learned_ai.evaluation.retained_phase_process_generalization import (
    replay_frozen_start,
)
from learned_ai.evaluation.sanmill_classical_positional_safety_strength import (
    ProductDevRuntime,
)
from learned_ai.evaluation.sanmill_classical_search_strength import board_from_state
from learned_ai.evaluation.sanmill_product_route_heldout import (
    AUTHORIZATION_SCHEMA,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    ProductRouteHeldoutError,
    ProductRouteRuntime,
    analyze_records,
    append_game_record,
    build_schedule,
    choose_product_route_move,
    compact_game,
    load_game_records,
    load_sealed,
    membership_only_suffix,
    play_product_route_game,
    validated_suffix_records,
)
from learned_ai.evaluation.sanmill_safe_guidance_gameplay import (
    ResourceLedger,
    sha256_file,
    write_json_atomic,
)
from learned_ai.sentinel.config import load_config as load_sentinel_config
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.infer import load_advisor
from learned_ai.training.sanmill_referee import (
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
    training_installation_record,
)


def _git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _running_tgf_processes() -> int:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "@(Get-Process -Name tgf -ErrorAction SilentlyContinue).Count",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise ProductRouteHeldoutError("cannot inspect Sanmill processes")
    return int(result.stdout.strip())


def _local_paths(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProductRouteHeldoutError("local path registry is not an object")
    return value


def _local_path(value: Any, *, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProductRouteHeldoutError(f"local path is absent: {key}")
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _require_runtime_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    fields = (
        "identity",
        "commit",
        "tree",
        "binary_relative_path",
        "binary_sha256",
        "binary_size",
        "strict_options",
        "strict_referee",
    )
    mismatches = [field for field in fields if observed.get(field) != expected.get(field)]
    if mismatches:
        raise ProductRouteHeldoutError(
            f"Sanmill runtime comparability differs: {mismatches}"
        )


def _require_implementation_binding(plan: Mapping[str, Any]) -> dict[str, str]:
    expected = plan.get("implementation_files")
    if not isinstance(expected, Mapping):
        raise ProductRouteHeldoutError("implementation binding is absent")
    observed = {str(path): sha256_file(ROOT / str(path)) for path in expected}
    if observed != dict(expected):
        raise ProductRouteHeldoutError("measurement implementation binding differs")
    return observed


def _require_product_source_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    product = plan["product_contract"]
    filter_observed = {
        "web_app": sha256_file(ROOT / "web/app.py"),
        "positional_safety": sha256_file(
            ROOT / "learned_ai/agents/positional_safety.py"
        ),
        "malom_runtime": sha256_file(ROOT / "ai/malom_runtime.py"),
    }
    if filter_observed != product["filter_implementation_sha256"]:
        raise ProductRouteHeldoutError("delivered product filter source differs")
    specialist_observed = {
        "specialist_router": sha256_file(
            ROOT / "learned_ai/agents/specialist_router.py"
        ),
        "scaffolded_encoder": sha256_file(
            ROOT / "learned_ai/models/scaffolded_encoder.py"
        ),
        "web_app": sha256_file(ROOT / "web/app.py"),
    }
    if specialist_observed != plan["specialist_contract"][
        "route_source_sha256"
    ]:
        raise ProductRouteHeldoutError("delivered specialist route source differs")
    return {"filter": filter_observed, "specialist": specialist_observed}


def _database_snapshot(paths: Sequence[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        stat = path.stat()
        result[relative] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
            "journal_exists": Path(f"{path}-journal").exists(),
            "wal_exists": Path(f"{path}-wal").exists(),
            "shm_exists": Path(f"{path}-shm").exists(),
        }
    return result


def _require_file_identity(path: Path, expected: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(expected["bytes"])
        or sha256_file(path) != str(expected["sha256"])
    ):
        raise ProductRouteHeldoutError(f"product specialist resource differs: {path}")


def _sanitized_malom_status(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "validation": status["validation"],
        "selected_source": status["selected_source"],
        "manifest_file_sha256": status["manifest_file_sha256"],
        "manifest_sha256": status["manifest_sha256"],
        "content_sha256": status["content_sha256"],
        "label_version": status["label_version"],
        "component_count": status["component_count"],
        "size_bytes": status["size_bytes"],
        "inventory_validation": status["inventory_validation"],
        "candidate_outcomes": [
            {
                "source": row["source"],
                "status": row["status"],
                "reason": row["reason"],
                "path_recorded_only_in_local_preflight": bool(row.get("path")),
            }
            for row in status["candidates"]
        ],
    }


def _load_product_route_runtime(
    *,
    plan: Mapping[str, Any],
    product_malom_adapter: ExternalSolvedDB,
    product_oracle: Any,
) -> ProductRouteRuntime:
    contract = plan["specialist_contract"]
    resources = contract["resource_files"]
    paths = {name: ROOT / value["path"] for name, value in resources.items()}
    for name, path in paths.items():
        _require_file_identity(path, resources[name])
    specialist_path = ROOT / contract["specialist_db"]["path"]
    expected_state = contract["specialist_db"]["expected"]
    if expected_state == "absent":
        if specialist_path.exists():
            raise ProductRouteHeldoutError("product SpecialistDB presence changed")
        specialist_db = None
    elif expected_state == "present-read-only":
        _require_file_identity(specialist_path, contract["specialist_db"])
        specialist_db = SpecialistDB(str(specialist_path), read_only=True)
    else:
        raise ProductRouteHeldoutError("SpecialistDB contract differs")

    human = HumanDB(paths["human_db"], read_only=True, immutable=True)
    if not human.is_available():
        raise ProductRouteHeldoutError("product HumanDB open failed")
    sentinel = load_advisor(
        str(paths["sentinel_checkpoint"]),
        load_sentinel_config(),
        device="cpu",
    )
    classical = ProductDevRuntime(
        resource_root=ROOT, expected=plan["product_contract"]
    )
    if sentinel is None or classical.value_net is None or classical.gap_net is None:
        human.close()
        classical.close()
        raise ProductRouteHeldoutError("product specialist dependency failed to load")
    router = load_specialist_router(
        ckpt_dir=ROOT / contract["checkpoint_root"],
        sentinel_advisor=sentinel,
        db=None,
        human_db=human,
        value_net=classical.value_net,
        gap_net=classical.gap_net,
        specialist_db=specialist_db,
        runtime_quarantine=None,
        ply_depth=int(contract["ply_depth"]),
    )
    if (
        router is None
        or router._spec_open is None
        or router._spec_mid is None
        or router._spec_end is None
        or router._la_open is None
        or router._la_mid is None
        or router._la_end is None
    ):
        human.close()
        classical.close()
        raise ProductRouteHeldoutError("all three product specialists are required")
    router.set_db(product_malom_adapter)
    router.configure_positional_safety(
        product_oracle,
        label_version=plan["malom_contract"]["label_version"],
        manifest_sha256=plan["malom_contract"]["manifest_sha256"],
        content_sha256=plan["malom_contract"]["content_sha256"],
    )
    if canonical_sha256(contract["identity_body"]) != contract["runtime_identity"]:
        human.close()
        classical.close()
        raise ProductRouteHeldoutError("specialist runtime identity differs")
    return ProductRouteRuntime(
        classical=classical,
        specialist=router,
        human_db=human,
        specialist_db=specialist_db,
        runtime_identity=contract["runtime_identity"],
    )


def _preflight_histories(
    records: Sequence[Mapping[str, Any]],
    *,
    installation: Any,
    seed: int,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    for record in records:
        ledger.require_within()
        with SanmillTrainingGame(installation, seed=seed) as game:
            _board, start = replay_frozen_start(game, record)
        observed.append(
            {
                "start_id": start["start_id"],
                "record_identity": start["start_record_identity"],
                "history_sha256": start["observed_history_sha256"],
                "logical_ply_count": start["logical_ply_count"],
            }
        )
    return {
        "records": len(observed),
        "all_nonterminal_histories_replayed": len(observed) == EXPECTED_STARTS,
        "observations_identity": canonical_sha256(observed),
    }


def _product_route_canaries(
    *,
    plan: Mapping[str, Any],
    route_runtime: ProductRouteRuntime,
    gate: ProductPositionalSafetyGate,
    ledger: ResourceLedger,
) -> dict[str, Any]:
    canary = plan["preflight_canary"]
    pool_path = ROOT / canary["pool_path"]
    if sha256_file(pool_path) != canary["pool_file_sha256"]:
        raise ProductRouteHeldoutError("product canary pool differs")
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    state = next(
        row for row in payload["states"] if row["state_id"] == canary["state_id"]
    )
    board = board_from_state(state)
    rows = []
    for route in ("specialist-first", "classical-first"):
        ai = route_runtime.classical.new_ai(
            color=board.turn,
            difficulty=9,
            node_budget=13_887_000,
            search_threads=int(
                plan["product_contract"]["deterministic_search_threads"]
            ),
            max_depth=int(plan["product_contract"]["max_depth"]),
            malom_adapter=plan["product_contract"]["product_malom_adapter"],
        )
        try:
            move, choice = choose_product_route_move(
                board=board,
                ai=ai,
                route_runtime=route_runtime,
                gate=gate,
                route=route,
                difficulty=9,
                ledger=ledger,
            )
        finally:
            del ai
            gc.collect()
        expected_source = "specialist" if route == "specialist-first" else "classical-coordinator"
        if (
            choice["product_source"] != expected_source
            or choice["final_gate"]["status"] != "applied"
            or choice["final_gate"]["selected_move"] != move
        ):
            raise ProductRouteHeldoutError(f"{route} product canary differs")
        rows.append(
            {
                "route": route,
                "product_source": choice["product_source"],
                "move": move,
                "gate_status": choice["final_gate"]["status"],
                "gate_selection_rule": choice["final_gate"]["selection_rule"],
                "specialist_succeeded": choice["specialist"]["succeeded"],
            }
        )
    return {
        "state_id": canary["state_id"],
        "main_sample": False,
        "complete_games": 0,
        "passed": len(rows) == 2,
        "routes": rows,
        "identity": canonical_sha256(rows),
    }


def _result_body(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    execution_commit: str,
    execution_tree: str,
    records: Sequence[Mapping[str, Any]],
    recovery: Mapping[str, Any],
    analysis: Mapping[str, Any],
    resources: Mapping[str, Any],
    preflight: Mapping[str, Any],
    database_snapshot_unchanged: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "status": analysis["status"],
        "plan_identity": plan["plan_identity"],
        "authorization_identity": authorization["authorization_identity"],
        "execution_commit": execution_commit,
        "execution_tree": execution_tree,
        "source_pool": {
            **plan["source_pool"],
            "consumed_in_this_execution": EXPECTED_STARTS,
            "remaining_after_this_execution": 0,
        },
        "schedule_identity": plan["schedule"]["identity"],
        "games": [compact_game(row) for row in records],
        "raw_ledger": {
            "path": plan["outputs"]["games"],
            "records": recovery["record_count"],
            "tail_record_sha256": recovery["tail_record_sha256"],
            "file_sha256": recovery["file_sha256"],
            "tracked": False,
        },
        "analysis": analysis,
        "resources": {
            **resources,
            "complete_games": len(records),
            "envelope": plan["resource_envelope"],
            "within_all_limits": (
                len(records) <= plan["resource_envelope"]["maximum_complete_games"]
                and resources["active_seconds"]
                <= plan["resource_envelope"]["maximum_active_seconds"]
            ),
        },
        "preflight": preflight,
        "access_audit": {
            "unconsumed_suffix_records_opened_after_plan_freeze": EXPECTED_STARTS,
            "consumed_prefix_records_used_as_new_evidence": 0,
            "old_48_development_starts_in_main_sample": 0,
            "official_selection_reads": 0,
            "official_confirmation_reads": 0,
            "official_final_test_reads": 0,
            "research_confirmation_reads": 0,
            "database_writes": 0,
            "database_snapshot_unchanged": database_snapshot_unchanged,
            "training_updates": 0,
            "model_fits": 0,
        },
        "claim_boundary": plan["claim_boundary"],
    }


def run(args: argparse.Namespace) -> int:
    stage = "static-preflight"
    output: Path | None = None
    lock_path = ROOT / "out/evaluation/sanmill-product-route-heldout-v1.lock"
    descriptor: int | None = None
    product_resolution: Any | None = None
    route_runtime: ProductRouteRuntime | None = None
    try:
        if _git("branch", "--show-current") != "dev":
            raise ProductRouteHeldoutError("formal measurement requires dev")
        if _git("status", "--short", "--untracked-files=no"):
            raise ProductRouteHeldoutError("tracked worktree must be clean")
        head = _git("rev-parse", "HEAD")
        origin = _git("rev-parse", "origin/dev")
        if head != origin:
            raise ProductRouteHeldoutError("execution commit is not published origin/dev")
        if _running_tgf_processes() != 0:
            raise ProductRouteHeldoutError("a Sanmill process is already running")

        plan, plan_sha = load_sealed(
            ROOT / args.plan, schema=PLAN_SCHEMA, identity_field="plan_identity"
        )
        authorization, authorization_sha = load_sealed(
            ROOT / args.authorization,
            schema=AUTHORIZATION_SCHEMA,
            identity_field="authorization_identity",
        )
        freeze_audit, freeze_audit_sha = load_sealed(
            ROOT / plan["outputs"]["freeze_audit"],
            schema="nmm.sanmill-product-route-heldout-freeze-audit.v1",
            identity_field="freeze_audit_identity",
        )
        if (
            authorization["plan_identity"] != plan["plan_identity"]
            or authorization["plan_file_sha256"] != plan_sha
            or authorization["freeze_audit_identity"]
            != freeze_audit["freeze_audit_identity"]
            or authorization["freeze_audit_file_sha256"] != freeze_audit_sha
            or freeze_audit["plan_identity"] != plan["plan_identity"]
            or freeze_audit["status"]
            != "post_freeze_contract_self_consistency_passed"
            or freeze_audit["candidate_moves_read"] != 0
            or freeze_audit["candidate_results_read"] != 0
            or authorization["resource_envelope"] != plan["resource_envelope"]
            or authorization["output_namespace"] != plan["outputs"]["namespace"]
            or authorization["operator"] != "product-owner-direct"
            or not _is_ancestor(plan["implementation_commit"], head)
        ):
            raise ProductRouteHeldoutError("authorization binding differs")
        implementation = _require_implementation_binding(plan)
        product_sources = _require_product_source_binding(plan)

        output = ROOT / plan["outputs"]["namespace"]
        result_path = ROOT / plan["outputs"]["result"]
        if output.exists() or result_path.exists() or lock_path.exists():
            raise ProductRouteHeldoutError("fresh once-only output namespace is unavailable")
        output.mkdir(parents=True, exist_ok=False)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, authorization["authorization_identity"].encode("ascii"))
        os.close(descriptor)
        descriptor = None

        ledger = ResourceLedger(
            engine_searches=0,
            malom_queries=0,
            active_seconds_before_run=0.0,
            maximum_engine_searches=int(
                plan["resource_envelope"]["engine_search_anomaly_ceiling"]
            ),
            maximum_malom_queries=int(
                plan["resource_envelope"]["malom_query_anomaly_ceiling"]
            ),
            maximum_active_seconds=float(
                plan["resource_envelope"]["maximum_active_seconds"]
            ),
        )
        paths = _local_paths(ROOT / args.paths_config)
        checkout = _local_path(paths.get("sanmill_training_checkout"), key="sanmill")
        malom_path = _local_path(paths.get("malom_db_path"), key="malom")
        installation = inspect_sanmill_training_installation(checkout)
        sanmill_runtime = training_installation_record(
            installation, seed=int(plan["sanmill_contract"]["seed"])
        )
        _require_runtime_equal(sanmill_runtime, plan["sanmill_contract"])
        if sanmill_runtime["identity"] != (
            "705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea"
        ):
            raise ProductRouteHeldoutError("required Sanmill runtime identity differs")
        malom_snapshot = verify_malom_snapshot(
            malom_path=malom_path,
            manifest_path=ROOT / plan["malom_contract"]["manifest_path"],
            full_hash=False,
        )
        if (
            malom_snapshot["trust_level"] != "sector-corrected-v1"
            or malom_snapshot["content_sha256"]
            != plan["malom_contract"]["content_sha256"]
        ):
            raise ProductRouteHeldoutError("Malom snapshot differs")

        corpus_path = ROOT / plan["source_pool"]["path"]
        if sha256_file(corpus_path) != plan["source_pool"]["file_sha256"]:
            raise ProductRouteHeldoutError("source-pool file differs")
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        membership = membership_only_suffix(corpus)
        if (
            canonical_sha256(membership)
            != plan["source_pool"]["suffix_membership_identity"]
            or [row["start_id"] for row in membership]
            != plan["source_pool"]["suffix_start_ids"]
        ):
            raise ProductRouteHeldoutError("source-pool suffix membership differs")
        records = validated_suffix_records(corpus)
        schedule = build_schedule(
            membership, namespace=plan["schedule"]["namespace"]
        )
        if canonical_sha256(schedule) != plan["schedule"]["identity"]:
            raise ProductRouteHeldoutError("formal schedule identity differs")
        if len(schedule) != EXPECTED_GAMES:
            raise ProductRouteHeldoutError("formal schedule cardinality differs")
        records_by_id = {str(row["start_id"]): row for row in records}

        history_preflight = _preflight_histories(
            records,
            installation=installation,
            seed=int(plan["sanmill_contract"]["seed"]),
            ledger=ledger,
        )

        def adapter_factory(path: str, *, strict: bool) -> ExternalSolvedDB:
            return ExternalSolvedDB(
                path, strict=strict, query_observer=ledger.add_malom
            )

        settings = json.loads((ROOT / "data/settings.json").read_text(encoding="utf-8"))
        product_resolution = resolve_product_malom_runtime(
            repo_root=ROOT,
            settings=settings,
            sentinel_path="",
            local_paths_path=ROOT / args.paths_config,
            manifest_path=ROOT / plan["malom_contract"]["manifest_path"],
            adapter_factory=adapter_factory,
        )
        if (
            product_resolution.database is None
            or product_resolution.oracle is None
            or product_resolution.status["validation"] != "passed"
            or product_resolution.status["label_version"] != "sector-corrected-v1"
            or product_resolution.status["content_sha256"]
            != plan["malom_contract"]["content_sha256"]
        ):
            raise ProductRouteHeldoutError("product Malom resolver did not validate")
        product_malom_status = _sanitized_malom_status(product_resolution.status)
        plan["product_contract"]["product_malom_adapter"] = (
            product_resolution.database
        )
        gate = ProductPositionalSafetyGate(high_difficulty_minimum=9)
        gate.configure(
            product_resolution.oracle,
            label_version=product_resolution.status["label_version"],
            manifest_sha256=product_resolution.status["manifest_sha256"],
            content_sha256=product_resolution.status["content_sha256"],
        )
        route_runtime = _load_product_route_runtime(
            plan=plan,
            product_malom_adapter=product_resolution.database,
            product_oracle=product_resolution.oracle,
        )

        tracked_resources = [
            ROOT / "data/human_db.sqlite",
            ROOT / "data/endgame/fullgame.bin",
            ROOT / "data/value_net_phase_place.npz",
            ROOT / "data/value_net_phase_move.npz",
            ROOT / "data/value_net_phase_fly.npz",
            ROOT / "data/gap_net.npz",
            ROOT / "learned_ai/sentinel/checkpoints/best.pt",
            ROOT / "learned_ai/checkpoints/scaffolded/s_open_v2/best.pt",
            ROOT / "learned_ai/checkpoints/scaffolded/s_mid_v2/best.pt",
            ROOT / "learned_ai/checkpoints/scaffolded/s_end_v2/best.pt",
        ]
        before_snapshot = _database_snapshot(tracked_resources)
        canaries = _product_route_canaries(
            plan=plan, route_runtime=route_runtime, gate=gate, ledger=ledger
        )
        after_canary_snapshot = _database_snapshot(tracked_resources)
        if before_snapshot != after_canary_snapshot:
            raise ProductRouteHeldoutError("read-only product resource changed in preflight")
        if _running_tgf_processes() != 0:
            raise ProductRouteHeldoutError("preflight left a Sanmill process running")

        preflight_body = {
            "schema_version": "nmm.sanmill-product-route-heldout-preflight.v1",
            "status": "ready_for_smoke",
            "meaning": "authorized bounded held-out evaluation; no training",
            "plan_identity": plan["plan_identity"],
            "authorization_identity": authorization["authorization_identity"],
            "authorization_file_sha256": authorization_sha,
            "execution_commit": head,
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "execution_published": head == origin,
            "implementation_files": implementation,
            "product_source_files": product_sources,
            "source_pool_membership": {
                "records": len(membership),
                "identity": canonical_sha256(membership),
                "consumed_prefix_records_used_as_new_evidence": 0,
                "old_48_overlap": 0,
            },
            "history_replay": history_preflight,
            "product_route_canaries": canaries,
            "sanmill_runtime": sanmill_runtime,
            "malom_runtime": product_malom_status,
            "specialist_runtime_identity": route_runtime.runtime_identity,
            "read_only_snapshot_unchanged": True,
            "planned_complete_games": EXPECTED_GAMES,
            "preflight_complete_games": 0,
            "running_sanmill_processes": 0,
            "protected_segments_opened": 0,
            "contract_self_consistency_after_freeze": plan[
                "contract_self_consistency"
            ],
            "resources": ledger.record(),
        }
        write_sealed_json(
            output / "preflight.json",
            preflight_body,
            identity_field="preflight_identity",
        )
        write_json_atomic(
            output / "measurement-started.json",
            {
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "execution_commit": head,
                "started_at_unix": time.time(),
                "execution_count": 1,
                "resume_or_retry_allowed": False,
            },
        )

        stage = "formal-execution"
        game_path = output / "games.jsonl"
        previous: str | None = None
        formal_records: list[dict[str, Any]] = []
        database = MalomDB(malom_path)
        try:
            for index, item in enumerate(schedule, start=1):
                ledger.require_within()
                record = play_product_route_game(
                    schedule_item=item,
                    start_record=records_by_id[str(item["start_id"])],
                    route_runtime=route_runtime,
                    product_contract=plan["product_contract"],
                    database=database,
                    gate=gate,
                    installation=installation,
                    ledger=ledger,
                )
                previous = append_game_record(
                    game_path,
                    record,
                    previous_record_sha256=previous,
                )
                formal_records.append(record)
                write_json_atomic(
                    output / "progress.json",
                    {
                        "stage": stage,
                        "completed_games": index,
                        "planned_games": EXPECTED_GAMES,
                        "outcomes_hidden_until_all_arms_complete": True,
                        "resources": ledger.record(),
                    },
                )
        finally:
            database.close()
        if len(formal_records) != EXPECTED_GAMES:
            raise ProductRouteHeldoutError("formal execution did not complete all arms")

        stage = "post-completion-analysis"
        recovery = load_game_records(game_path, schedule=schedule)
        analysis = analyze_records(
            recovery["records"],
            start_ids=plan["source_pool"]["suffix_start_ids"],
        )
        after_snapshot = _database_snapshot(tracked_resources)
        database_unchanged = before_snapshot == after_snapshot
        if not database_unchanged:
            raise ProductRouteHeldoutError("read-only product resource changed")
        resources = ledger.record()
        if _running_tgf_processes() != 0:
            raise ProductRouteHeldoutError("formal execution left Sanmill running")
        preflight, _preflight_sha = load_sealed(
            output / "preflight.json",
            schema="nmm.sanmill-product-route-heldout-preflight.v1",
            identity_field="preflight_identity",
        )
        body = _result_body(
            plan=plan,
            authorization=authorization,
            execution_commit=head,
            execution_tree=_git("rev-parse", "HEAD^{tree}"),
            records=recovery["records"],
            recovery=recovery,
            analysis=analysis,
            resources=resources,
            preflight=preflight,
            database_snapshot_unchanged=database_unchanged,
        )
        write_sealed_json(result_path, body, identity_field="result_identity")
        write_json_atomic(
            output / "completion.json",
            {
                "status": "completed_once",
                "plan_identity": plan["plan_identity"],
                "authorization_identity": authorization["authorization_identity"],
                "result": plan["outputs"]["result"],
                "result_file_sha256": sha256_file(result_path),
                "games": len(formal_records),
                "resources": resources,
                "completed_at_unix": time.time(),
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed_once",
                    "games": len(formal_records),
                    "result": str(result_path),
                    "result_identity": json.loads(
                        result_path.read_text(encoding="utf-8")
                    )["result_identity"],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        if output is not None and output.exists():
            write_json_atomic(
                output / "failure.json",
                {
                    "status": "failed_closed_no_retry_or_resume",
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at_unix": time.time(),
                },
            )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if route_runtime is not None:
            route_runtime.close()
        if product_resolution is not None:
            close = getattr(product_resolution.database, "close", None)
            if callable(close):
                close()
        lock_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="docs/experiments/sanmill-product-route-heldout-v1.json",
    )
    parser.add_argument(
        "--authorization",
        default=(
            "docs/experiments/sanmill-product-route-heldout-v1/authorization.json"
        ),
    )
    parser.add_argument(
        "--paths-config", default="data/training_paths.local.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    raise SystemExit(run(parse_args()))
