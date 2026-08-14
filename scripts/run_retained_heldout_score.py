#!/usr/bin/env python3
"""Preflight or run the frozen retained-v3/v4 held-out score comparison."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.retained_heldout_score import (  # noqa: E402
    EXPECTED_CANDIDATES,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    HORIZON_POST_START_LOGICAL_PLIES,
    MAX_POST_START_LOGICAL_PLIES,
    MAX_PRIMARY_HALF_WIDTH,
    PLAN_SCHEMA,
    SANMILL_NODE_CEILING,
    SPEC_SCHEMA,
    RetainedHeldoutScoreError,
    append_game_record,
    build_schedule,
    load_corpus_records,
    load_game_ledger,
    play_heldout_score_game,
    recompute_report,
    replay_frozen_start,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from scripts import run_retained_phase_process_generalization as shared  # noqa: E402
from scripts.run_retained_passivity_diagnostic import (  # noqa: E402
    DiagnosticPaths,
    _strict_json,
)
from tools.prepare_retained_heldout_score_inputs import (  # noqa: E402
    TARGET_ROOT as SNAPSHOT_ROOT,
    build_manifest as build_input_manifest,
)


DEFAULT_PLAN = _ROOT / ("docs/experiments/sanmill-retained-v3-v4-heldout-score-v1.json")
SOURCE_READINESS_SCHEMA = "nmm.retained-heldout-score-source-readiness.v1"
READINESS_SCHEMA = "nmm.retained-heldout-score-readiness.v1"
AUTHORIZATION_SCHEMA = "nmm.retained-heldout-score-authorization.v1"
LAUNCH_SCHEMA = "nmm.retained-heldout-score-launch.v1"
PROGRESS_SCHEMA = "nmm.retained-heldout-score-progress.v1"
FAILURE_SCHEMA = "nmm.retained-heldout-score-failure.v1"
COMPLETION_SCHEMA = "nmm.retained-heldout-score-completion.v1"
MAX_ACTIVE_HOURS = 4.0
POST_PLAN_STATUS_DOCUMENTS = {
    "docs/evidence/sanmill-retained-v3-v4-heldout-score-readiness-2026-08-14.md",
    "docs/experiments/sanmill-retained-v3-v4-evaluation-decision-brief.md",
    "docs/experiments/sanmill-retained-v3-v4-heldout-score-v1.md",
    "docs/handoff/windows-training-2026-07-20.md",
    "docs/local-training-layout.md",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate the exact high-precision held-out plan."""
    plan_path = Path(path).resolve(strict=True)
    plan = _strict_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RetainedHeldoutScoreError("held-out score plan schema differs")
    identity = plan.get("plan_identity")
    body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise RetainedHeldoutScoreError("held-out score plan identity differs")
    candidates = plan.get("candidates")
    if (
        not isinstance(candidates, list)
        or tuple(candidate.get("candidate_id") for candidate in candidates)
        != EXPECTED_CANDIDATES
    ):
        raise RetainedHeldoutScoreError("held-out candidate order differs")
    workload = plan.get("workload")
    if not isinstance(workload, Mapping) or (
        workload.get("games") != EXPECTED_GAMES
        or workload.get("unique_starts") != EXPECTED_STARTS
        or workload.get("max_active_hours") != MAX_ACTIVE_HOURS
    ):
        raise RetainedHeldoutScoreError("held-out workload differs")
    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("horizon_post_start_logical_plies")
        != HORIZON_POST_START_LOGICAL_PLIES
        or protocol.get("max_post_start_logical_plies") != MAX_POST_START_LOGICAL_PLIES
        or protocol.get("sanmill_node_ceiling_per_turn") != SANMILL_NODE_CEILING
        or protocol.get("mechanism_reanalysis") != "none"
    ):
        raise RetainedHeldoutScoreError("held-out protocol differs")
    analysis = plan.get("analysis")
    interval = (
        analysis.get("engineering_interval") if isinstance(analysis, Mapping) else None
    )
    if (
        not isinstance(interval, Mapping)
        or interval.get("maximum_primary_half_width") != MAX_PRIMARY_HALF_WIDTH
    ):
        raise RetainedHeldoutScoreError("held-out precision target differs")
    boundary = plan.get("claim_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("held_out") is not True
        or boundary.get("equivalence_claim") is not False
        or boundary.get("refresh_causal_claim") is not False
    ):
        raise RetainedHeldoutScoreError("held-out claim boundary differs")
    if plan.get("status") != "frozen_awaiting_product_authorization":
        raise RetainedHeldoutScoreError("held-out plan status differs")
    return plan


def _corpus_record(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if sha256_file(paths.corpus) != plan["corpus"]["file_sha256"]:
        raise RetainedHeldoutScoreError("held-out source-pool file differs")
    payload = json.loads(paths.corpus.read_text(encoding="utf-8"))
    if (
        payload.get("pool_identity") != plan["corpus"]["pool_identity"]
        or payload.get("records_identity") != plan["corpus"]["pool_records_identity"]
    ):
        raise RetainedHeldoutScoreError("held-out source-pool binding differs")
    records = load_corpus_records(payload)
    prefix_identity = canonical_sha256(
        [str(record["record_identity"]) for record in records]
    )
    if prefix_identity != plan["corpus"]["prefix_records_identity"]:
        raise RetainedHeldoutScoreError("held-out prefix binding differs")
    schedule = build_schedule(records)
    return (
        {
            "records": len(records),
            "held_out": True,
            "candidate_blind_selection": True,
            "pool_identity": payload["pool_identity"],
            "pool_records_identity": payload["records_identity"],
            "prefix_records_identity": prefix_identity,
            "schedule_identity": canonical_sha256(schedule),
            "games": len(schedule),
        },
        records,
    )


def _test_record() -> dict[str, Any]:
    common = ["-q", "-p", "no:cacheprovider"]
    focused = shared._stable_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_retained_late_import_heldout_pool.py",
            "tests/test_retained_heldout_score.py",
            "tests/test_prepare_retained_heldout_score_inputs.py",
            "tests/test_freeze_retained_heldout_score_plan.py",
            "tests/test_run_retained_heldout_score.py",
            "tests/test_retained_phase_process_generalization.py",
            "tests/test_training_aligned_policy.py",
            "tests/test_sanmill_training_referee.py",
            "tests/test_training_route_bundle.py",
            "tests/test_checkpoint_envelope.py",
            *common,
            "--basetemp",
            ".tmp/pytest-retained-heldout-score-preflight",
        ],
        label="retained held-out score focused tests",
    )
    mandatory = shared._stable_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_malom_db.py",
            "tests/test_sentinel_db_teacher.py",
            "tests/test_malom_label_provenance.py",
            *common,
            "--basetemp",
            ".tmp/pytest-retained-heldout-score-provenance",
        ],
        label="mandatory Malom and provenance tests",
    )
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RetainedHeldoutScoreError("Ruff is unavailable")
    lint = shared._stable_check(
        [
            ruff,
            "check",
            "learned_ai/evaluation/retained_phase_process_generalization.py",
            "learned_ai/evaluation/retained_heldout_score.py",
            "scripts/run_retained_phase_process_generalization.py",
            "scripts/run_retained_heldout_score.py",
            "tools/prepare_retained_heldout_score_inputs.py",
            "tools/freeze_retained_heldout_score_plan.py",
            "tools/serve_retained_phase_process_generalization.py",
            "tests/test_retained_heldout_score.py",
            "tests/test_run_retained_heldout_score.py",
        ],
        label="retained held-out score Ruff checks",
    )
    return {"focused": focused, "mandatory": mandatory, "ruff": lint}


def build_authorization(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_commit: str,
    source_readiness_identity: str,
    authority_text_sha256: str,
    operator: str = "product-owner-direct",
) -> dict[str, Any]:
    """Build, but never write, the exact grant after owner approval."""
    for value, label, length in (
        (source_readiness_identity, "source readiness identity", 64),
        (authority_text_sha256, "authority text identity", 64),
        (plan_commit, "plan commit", 40),
    ):
        if len(value) != length or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RetainedHeldoutScoreError(f"authorization {label} is invalid")
    if operator != "product-owner-direct":
        raise RetainedHeldoutScoreError("authorization operator differs")
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "diagnostic_id": plan["diagnostic_id"],
        "operator": operator,
        "authority_text_sha256": authority_text_sha256,
        "source_readiness_identity": source_readiness_identity,
        "grant": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": sha256_file(plan_path),
            "plan_commit": plan_commit,
            "games": EXPECTED_GAMES,
            "max_active_hours": MAX_ACTIVE_HOURS,
            "host_interruption_exact_resume_only": True,
            "same_spec_exact_resume": True,
            "automatic_retry": False,
            "semantic_failure_recovery": False,
            "expansion": False,
            "training": False,
            "updates": False,
            "held_out_evaluation": True,
            "named_route_fixed_corpus_score_relation": True,
            "equivalence_claim": False,
            "elo_or_population_strength_claim": False,
            "refresh_causal_claim": False,
            "promotion": False,
            "publication": False,
            "release": False,
        },
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


@contextlib.contextmanager
def configured_shared_runner() -> Iterator[None]:
    """Temporarily bind the reviewed shared lifecycle to this immutable profile."""
    overrides = {
        "DEFAULT_PLAN": DEFAULT_PLAN,
        "SOURCE_READINESS_SCHEMA": SOURCE_READINESS_SCHEMA,
        "READINESS_SCHEMA": READINESS_SCHEMA,
        "AUTHORIZATION_SCHEMA": AUTHORIZATION_SCHEMA,
        "LAUNCH_SCHEMA": LAUNCH_SCHEMA,
        "PROGRESS_SCHEMA": PROGRESS_SCHEMA,
        "FAILURE_SCHEMA": FAILURE_SCHEMA,
        "COMPLETION_SCHEMA": COMPLETION_SCHEMA,
        "POST_PLAN_STATUS_DOCUMENTS": POST_PLAN_STATUS_DOCUMENTS,
        "PLAN_SCHEMA": PLAN_SCHEMA,
        "SPEC_SCHEMA": SPEC_SCHEMA,
        "EXPECTED_CANDIDATES": EXPECTED_CANDIDATES,
        "EXPECTED_GAMES": EXPECTED_GAMES,
        "EXPECTED_STARTS": EXPECTED_STARTS,
        "HORIZON_POST_START_LOGICAL_PLIES": HORIZON_POST_START_LOGICAL_PLIES,
        "MAX_POST_START_LOGICAL_PLIES": MAX_POST_START_LOGICAL_PLIES,
        "SANMILL_NODE_CEILING": SANMILL_NODE_CEILING,
        "SNAPSHOT_ROOT": SNAPSHOT_ROOT,
        "build_input_manifest": build_input_manifest,
        "load_plan": load_plan,
        "_corpus_record": _corpus_record,
        "_test_record": _test_record,
        "build_authorization": build_authorization,
        "build_schedule": build_schedule,
        "load_corpus_records": load_corpus_records,
        "load_game_ledger": load_game_ledger,
        "play_phase_process_game": play_heldout_score_game,
        "recompute_report": recompute_report,
        "replay_frozen_start": replay_frozen_start,
        "append_game_record": append_game_record,
    }
    originals = {name: getattr(shared, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(shared, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(shared, name, value)


def build_readiness_report(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
    *,
    resume: bool,
    run_tests: bool,
    audit_histories: bool,
) -> dict[str, Any]:
    with configured_shared_runner():
        return shared.build_readiness_report(
            plan,
            paths,
            resume=resume,
            run_tests=run_tests,
            audit_histories=audit_histories,
        )


def resolve_paths(
    plan: Mapping[str, Any],
    *,
    plan_path: str | Path,
    paths_config: str | Path,
) -> DiagnosticPaths:
    with configured_shared_runner():
        return shared.resolve_paths(
            plan,
            plan_path=plan_path,
            paths_config=paths_config,
        )


def main(argv: Sequence[str] | None = None) -> int:
    with configured_shared_runner():
        return shared.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
