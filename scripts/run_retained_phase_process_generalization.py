#!/usr/bin/env python3
"""Preflight or run the frozen retained phase-process confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from learned_ai.evaluation.heldout_evaluation import (  # noqa: E402
    ActiveClock,
    EvaluatorLock,
    replace_canonical,
    utc_now,
    write_new_canonical,
)
from learned_ai.evaluation.retained_phase_process_generalization import (  # noqa: E402
    EXPECTED_CANDIDATES,
    EXPECTED_GAMES,
    EXPECTED_STARTS,
    HORIZON_POST_START_LOGICAL_PLIES,
    MAX_POST_START_LOGICAL_PLIES,
    PLAN_SCHEMA,
    SANMILL_NODE_CEILING,
    SPEC_SCHEMA,
    RetainedPhaseProcessError,
    append_game_record,
    build_schedule,
    load_corpus_records,
    load_game_ledger,
    play_phase_process_game,
    recompute_report,
    replay_frozen_start,
)
from learned_ai.evaluation.retained_phase_process_mechanism_audit import (  # noqa: E402
    recompute_mechanism_audit,
)
from learned_ai.evaluation.training_aligned_policy import (  # noqa: E402
    TrainingAlignedPolicy,
)
from learned_ai.training.run_contract import canonical_sha256  # noqa: E402
from learned_ai.training.sanmill_referee import (  # noqa: E402
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
)
from scripts.run_retained_passivity_diagnostic import (  # noqa: E402
    DiagnosticPaths,
    RetainedPassivityDiagnosticError,
    _candidate_record,
    _git,
    _load_policy,
    _local_path,
    _output_record as _base_output_record,
    _repo_path,
    _repository_record as _base_repository_record,
    _sanmill_record,
    _strict_json,
    _assert_ignored,
)
from tools.prepare_retained_phase_process_inputs import (  # noqa: E402
    TARGET_ROOT as SNAPSHOT_ROOT,
    build_manifest as build_input_manifest,
)


DEFAULT_PLAN = _ROOT / (
    "docs/experiments/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1.json"
)
DEFAULT_PATHS = _ROOT / "data/training_paths.local.json"
SOURCE_READINESS_SCHEMA = (
    "nmm.retained-phase-process-generalization-source-readiness.v1"
)
READINESS_SCHEMA = "nmm.retained-phase-process-generalization-readiness.v1"
AUTHORIZATION_SCHEMA = (
    "nmm.retained-phase-process-generalization-authorization.v1"
)
LAUNCH_SCHEMA = "nmm.retained-phase-process-generalization-launch.v1"
PROGRESS_SCHEMA = "nmm.retained-phase-process-generalization-progress.v1"
FAILURE_SCHEMA = "nmm.retained-phase-process-generalization-failure.v1"
COMPLETION_SCHEMA = "nmm.retained-phase-process-generalization-completion.v1"
POST_PLAN_STATUS_DOCUMENTS = {
    "docs/evidence/"
    "sanmill-retained-v3-v4-phase-process-corpus-readiness-2026-08-13.md",
    "docs/experiments/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1.md",
    "docs/handoff/windows-training-2026-07-20.md",
    "docs/local-training-layout.md",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_record(paths: DiagnosticPaths, *, resume: bool) -> dict[str, Any]:
    result = _base_output_record(paths, resume=resume)
    mechanism = paths.output_root / "mechanism-report.json"
    _assert_ignored(mechanism)
    if mechanism.exists():
        raise RetainedPhaseProcessError(
            "phase-process mechanism report already exists"
        )
    return {**result, "mechanism_report": "absent"}


def _repository_record(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> dict[str, Any]:
    observed = _base_repository_record(plan, paths)
    relative_plan = paths.plan.relative_to(_ROOT).as_posix()
    plan_commit = str(
        _git("log", "-1", "--format=%H", "--", relative_plan)
    )
    if len(plan_commit) != 40:
        raise RetainedPhaseProcessError("tracked plan commit is absent")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", plan_commit, "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RetainedPhaseProcessError(
            "tracked plan commit is not an ancestor"
        ) from exc
    changed = {
        line
        for line in str(
            _git("diff", "--name-only", f"{plan_commit}..HEAD", "--")
        ).splitlines()
        if line
    }
    unexpected = sorted(changed - POST_PLAN_STATUS_DOCUMENTS)
    if unexpected:
        raise RetainedPhaseProcessError(
            "runtime-affecting files changed after the frozen plan: "
            + ", ".join(unexpected)
        )
    return {
        **observed,
        "plan_commit": plan_commit,
        "post_plan_runtime_files_unchanged": True,
        "post_plan_status_documents_only": True,
    }


def _source_gate_view(gates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stable = []
    for gate in gates:
        item = dict(gate)
        if gate.get("gate") == "repository" and gate.get("result") == "pass":
            observed = gate.get("observed")
            if not isinstance(observed, Mapping):
                raise RetainedPhaseProcessError(
                    "repository source-readiness evidence is absent"
                )
            fields = (
                "branch",
                "plan_commit",
                "published",
                "tracked_worktree",
                "implementation_commit_is_ancestor",
                "post_plan_runtime_files_unchanged",
                "post_plan_status_documents_only",
            )
            missing = [field for field in fields if field not in observed]
            if missing:
                raise RetainedPhaseProcessError(
                    "repository source-readiness evidence is incomplete: "
                    + ", ".join(missing)
                )
            item["observed"] = {
                field: observed[field] for field in fields
            }
        stable.append(item)
    return stable


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve(strict=True)
    plan = _strict_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RetainedPhaseProcessError("phase-process plan schema differs")
    identity = plan.get("plan_identity")
    body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise RetainedPhaseProcessError("phase-process plan identity differs")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or tuple(
        candidate.get("candidate_id") for candidate in candidates
    ) != EXPECTED_CANDIDATES:
        raise RetainedPhaseProcessError("phase-process candidate order differs")
    workload = plan.get("workload")
    if not isinstance(workload, Mapping) or (
        workload.get("games") != EXPECTED_GAMES
        or workload.get("unique_starts") != EXPECTED_STARTS
        or workload.get("max_active_hours") != 2.0
    ):
        raise RetainedPhaseProcessError("phase-process workload differs")
    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("horizon_post_start_logical_plies")
        != HORIZON_POST_START_LOGICAL_PLIES
        or protocol.get("max_post_start_logical_plies")
        != MAX_POST_START_LOGICAL_PLIES
        or protocol.get("sanmill_node_ceiling_per_turn")
        != SANMILL_NODE_CEILING
    ):
        raise RetainedPhaseProcessError("phase-process protocol differs")
    if plan.get("status") != "frozen_awaiting_product_authorization":
        raise RetainedPhaseProcessError("phase-process plan status differs")
    return plan


def resolve_paths(
    plan: Mapping[str, Any],
    *,
    plan_path: str | Path,
    paths_config: str | Path,
) -> DiagnosticPaths:
    config_path = Path(paths_config).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RetainedPhaseProcessError("local path registry is invalid")
    output_root = _repo_path(plan["outputs"]["root"], field="outputs.root")
    candidates = list(plan["candidates"])
    bundles = {
        str(candidate["candidate_id"]): _repo_path(
            candidate["bundle"]["path"],
            field="candidate.bundle.path",
        )
        for candidate in candidates
    }
    checkpoints = {
        str(candidate["candidate_id"]): _repo_path(
            candidate["checkpoint"]["path"],
            field="candidate.checkpoint.path",
        )
        for candidate in candidates
    }
    specialist = {
        str(candidate["candidate_id"]): _repo_path(
            candidate["specialist_db"]["path"],
            field="candidate.specialist_db.path",
        )
        for candidate in candidates
    }
    return DiagnosticPaths(
        plan=Path(plan_path).resolve(strict=True),
        paths_config=config_path,
        corpus=_repo_path(plan["corpus"]["path"], field="corpus.path"),
        human_db=_local_path(config.get("human_db_path"), field="human_db_path"),
        malom_db=_local_path(config.get("malom_db_path"), field="malom_db_path"),
        malom_manifest=_repo_path(
            plan["data"]["malom_manifest"],
            field="data.malom_manifest",
        ),
        sanmill_checkout=_local_path(
            config.get("sanmill_training_checkout"),
            field="sanmill_training_checkout",
        ),
        candidate_bundles=bundles,
        candidate_checkpoints=checkpoints,
        candidate_specialist_dbs=specialist,
        output_root=output_root,
        authorization=output_root / "authorization.json",
        runtime_plan=output_root / "plan.json",
        readiness=output_root / "readiness.json",
        spec=output_root / "spec.json",
        launch=output_root / "launch.json",
        ledger=output_root / "games.jsonl",
        progress=output_root / "progress.json",
        report=output_root / "report.json",
        completion=output_root / "completion.json",
        failure=output_root / "failure.json",
    )


def _input_record(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> dict[str, Any]:
    expected_root = _repo_path(plan["inputs"]["root"], field="inputs.root")
    if expected_root != SNAPSHOT_ROOT.resolve():
        raise RetainedPhaseProcessError("phase-process snapshot root differs")
    manifest_path = _repo_path(
        plan["inputs"]["manifest_path"],
        field="inputs.manifest_path",
    )
    if manifest_path != expected_root / "manifest.json":
        raise RetainedPhaseProcessError("phase-process manifest path differs")
    if sha256_file(manifest_path) != plan["inputs"]["manifest_file_sha256"]:
        raise RetainedPhaseProcessError("phase-process manifest file differs")
    manifest = build_input_manifest()
    if manifest["snapshot_identity"] != plan["inputs"]["snapshot_identity"]:
        raise RetainedPhaseProcessError("phase-process snapshot identity differs")
    by_candidate = {
        item["candidate_id"]: item for item in manifest["candidates"]
    }
    for candidate in plan["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        observed = by_candidate.get(candidate_id)
        if not isinstance(observed, Mapping):
            raise RetainedPhaseProcessError("snapshot candidate is absent")
        if (
            observed["route_bundle"]["path"] != candidate["bundle"]["path"]
            or observed["route_bundle"]["identity"]
            != candidate["bundle"]["identity"]
            or observed["specialist_db"]["path"]
            != candidate["specialist_db"]["path"]
            or observed["specialist_db"]["sha256"]
            != candidate["specialist_db"]["file_sha256"]
        ):
            raise RetainedPhaseProcessError(
                f"{candidate_id} snapshot binding differs"
            )
        if not (
            observed["route_bundle"]["read_only_files"]
            and observed["specialist_db"]["read_only_file"]
            and observed["specialist_db"]["sidecars_absent"]
        ):
            raise RetainedPhaseProcessError(
                f"{candidate_id} snapshot is not immutable"
            )
    relative = expected_root.relative_to(_ROOT).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=_ROOT,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise RetainedPhaseProcessError("phase-process inputs are not ignored")
    return {
        "snapshot_identity": manifest["snapshot_identity"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "candidate_ids": list(by_candidate),
        "successor_owned": True,
        "read_only": True,
        "sqlite_sidecars_absent": True,
    }


def _corpus_record(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if sha256_file(paths.corpus) != plan["corpus"]["file_sha256"]:
        raise RetainedPhaseProcessError("phase-process corpus file differs")
    payload = json.loads(paths.corpus.read_text(encoding="utf-8"))
    if (
        payload.get("corpus_identity") != plan["corpus"]["identity"]
        or payload.get("records_identity") != plan["corpus"]["records_identity"]
    ):
        raise RetainedPhaseProcessError("phase-process corpus binding differs")
    records = load_corpus_records(payload)
    schedule = build_schedule(records)
    return (
        {
            "records": len(records),
            "corpus_previously_project_visible": True,
            "corpus_identity": payload["corpus_identity"],
            "records_identity": payload["records_identity"],
            "schedule_identity": canonical_sha256(schedule),
            "games": len(schedule),
        },
        records,
    )


def _strict_history_record(
    records: Sequence[Mapping[str, Any]],
    installation: Any,
) -> dict[str, Any]:
    observations = []
    for record in records:
        with SanmillTrainingGame(installation, seed=42) as game:
            _board, start = replay_frozen_start(game, record)
        observations.append(
            {
                "start_id": record["start_id"],
                "start_record_identity": record["record_identity"],
                "logical_ply_count": start["logical_ply_count"],
                "history_sha256": start["observed_history_sha256"],
                "no_capture_count": start["no_capture_count"],
                "repetition_current_count": start["repetition_current_count"],
                "repetition_history_length": start["repetition_history_length"],
            }
        )
    return {
        "records": len(observations),
        "fresh_processes": len(observations),
        "candidate_loaded": False,
        "games_played": 0,
        "observations_identity": canonical_sha256(observations),
    }


def _competing_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        raise RetainedPhaseProcessError(
            "process audit is implemented only for the Windows training host"
        )
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        raise RetainedPhaseProcessError("PowerShell is unavailable")
    pattern = (
        "train_s_gen_v2\\.py|manage_generalist_run\\.py|"
        "run_heldout_evaluation\\.py|run_retained_passivity_diagnostic\\.py|"
        "run_retained_phase_process_generalization\\.py"
    )
    script = (
        f"$self={os.getpid()}; $parent={os.getppid()}; $scanner=$PID; "
        "@(Get-CimInstance Win32_Process | Where-Object { "
        "$_.ProcessId -ne $self -and $_.ProcessId -ne $parent -and "
        "$_.ProcessId -ne $scanner -and "
        f"$_.CommandLine -match '{pattern}' "
        "} | Select-Object ProcessId,Name) | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RetainedPhaseProcessError("process audit shape differs")
    return [
        {"pid": int(item["ProcessId"]), "name": str(item.get("Name") or "unknown")}
        for item in payload
    ]


def _stable_check(command: Sequence[str], *, label: str) -> dict[str, Any]:
    result = subprocess.run(
        list(command),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RetainedPhaseProcessError(
            f"{label} failed with exit {result.returncode}"
        )
    return {
        "label": label,
        "exit_code": 0,
        "command_identity": canonical_sha256(list(command)),
        "result": "pass",
    }


def _test_record() -> dict[str, Any]:
    common = ["-q", "-p", "no:cacheprovider"]
    focused = _stable_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_retained_phase_process_corpus.py",
            "tests/test_freeze_retained_phase_process_plan.py",
            "tests/test_retained_phase_process_generalization.py",
            "tests/test_retained_phase_process_mechanism_audit.py",
            "tests/test_prepare_retained_phase_process_inputs.py",
            "tests/test_training_aligned_policy.py",
            "tests/test_sanmill_training_referee.py",
            "tests/test_training_route_bundle.py",
            "tests/test_checkpoint_envelope.py",
            *common,
            "--basetemp",
            ".tmp/pytest-retained-phase-process-preflight",
        ],
        label="retained phase-process focused tests",
    )
    mandatory = _stable_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_malom_db.py",
            "tests/test_sentinel_db_teacher.py",
            "tests/test_malom_label_provenance.py",
            *common,
            "--basetemp",
            ".tmp/pytest-retained-phase-process-provenance",
        ],
        label="mandatory Malom and provenance tests",
    )
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RetainedPhaseProcessError("Ruff is unavailable")
    lint = _stable_check(
        [
            ruff,
            "check",
            "learned_ai/evaluation/retained_phase_process_generalization.py",
            "learned_ai/evaluation/retained_phase_process_mechanism_audit.py",
            "scripts/run_retained_phase_process_generalization.py",
            "tools/freeze_retained_phase_process_plan.py",
            "tools/prepare_retained_phase_process_inputs.py",
            "tools/serve_retained_phase_process_generalization.py",
            "tests/test_prepare_retained_phase_process_inputs.py",
            "tests/test_retained_phase_process_generalization.py",
        ],
        label="retained phase-process Ruff checks",
    )
    return {"focused": focused, "mandatory": mandatory, "ruff": lint}


def _gate(
    gates: list[dict[str, Any]],
    gate: str,
    expected: str,
    operation: Callable[[], Any],
) -> Any | None:
    try:
        observed = operation()
    except Exception as exc:
        gates.append(
            {
                "gate": gate,
                "expected": expected,
                "observed": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "result": "fail",
            }
        )
        return None
    gates.append(
        {
            "gate": gate,
            "expected": expected,
            "observed": observed,
            "result": "pass",
        }
    )
    return observed


def _raise_competing(processes: list[dict[str, Any]]) -> None:
    raise RetainedPhaseProcessError(
        "competing trainer/evaluator processes exist: "
        + ", ".join(str(item["pid"]) for item in processes)
    )


def build_authorization(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_commit: str,
    source_readiness_identity: str,
    authority_text_sha256: str,
    operator: str = "product-owner-direct",
) -> dict[str, Any]:
    """Build, but do not write, the exact grant after owner approval."""
    if (
        len(source_readiness_identity) != 64
        or any(character not in "0123456789abcdef" for character in source_readiness_identity)
    ):
        raise RetainedPhaseProcessError("source readiness identity is invalid")
    if len(plan_commit) != 40 or any(
        character not in "0123456789abcdef" for character in plan_commit
    ):
        raise RetainedPhaseProcessError("authorization plan commit is invalid")
    if len(authority_text_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in authority_text_sha256
    ):
        raise RetainedPhaseProcessError("authority text identity is invalid")
    if operator != "product-owner-direct":
        raise RetainedPhaseProcessError("authorization operator differs")
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
            "max_active_hours": 2.0,
            "host_interruption_exact_resume_only": True,
            "same_spec_exact_resume": True,
            "automatic_retry": False,
            "semantic_failure_recovery": False,
            "expansion": False,
            "training": False,
            "updates": False,
            "held_out_strength_claim": False,
            "promotion": False,
            "publication": False,
            "release": False,
        },
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


def _load_authorization(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
    *,
    expected_source_readiness_identity: str | None,
) -> dict[str, Any]:
    if not paths.authorization.is_file():
        raise RetainedPhaseProcessError(
            "exact plan-bound product authorization is absent"
        )
    authorization = _strict_json(paths.authorization)
    identity = authorization.get("authorization_identity")
    body = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_identity"
    }
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or not isinstance(identity, str)
        or canonical_sha256(body) != identity
    ):
        raise RetainedPhaseProcessError("phase-process authorization differs")
    source_identity = authorization.get("source_readiness_identity")
    if not isinstance(source_identity, str) or len(source_identity) != 64:
        raise RetainedPhaseProcessError(
            "authorization source readiness identity is absent"
        )
    if (
        expected_source_readiness_identity is not None
        and source_identity != expected_source_readiness_identity
    ):
        raise RetainedPhaseProcessError(
            "authorization source readiness identity differs"
        )
    expected = build_authorization(
        plan=plan,
        plan_path=paths.plan,
        plan_commit=str(authorization.get("grant", {}).get("plan_commit") or ""),
        source_readiness_identity=source_identity,
        authority_text_sha256=str(authorization.get("authority_text_sha256") or ""),
        operator=str(authorization.get("operator") or ""),
    )
    if expected != authorization:
        raise RetainedPhaseProcessError("authorization grant differs")
    plan_commit = authorization["grant"]["plan_commit"]
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", plan_commit, "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RetainedPhaseProcessError(
            "authorization plan commit is not published in runtime source"
        ) from exc
    plan_blob = _git(
        "show",
        f"{plan_commit}:{paths.plan.relative_to(_ROOT).as_posix()}",
        binary=True,
    )
    if not isinstance(plan_blob, bytes) or hashlib.sha256(plan_blob).hexdigest() != (
        authorization["grant"]["plan_file_sha256"]
    ):
        raise RetainedPhaseProcessError("authorized plan blob differs")
    return authorization


def _resume_record(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> dict[str, Any]:
    if paths.runtime_plan.read_bytes() != paths.plan.read_bytes():
        raise RetainedPhaseProcessError("resume plan copy differs")
    spec = _strict_json(paths.spec)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise RetainedPhaseProcessError("resume spec schema differs")
    identity = spec.get("spec_identity")
    body = {key: value for key, value in spec.items() if key != "spec_identity"}
    if not isinstance(identity, str) or canonical_sha256(body) != identity:
        raise RetainedPhaseProcessError("resume spec identity differs")
    if (
        spec["plan"]["identity"] != plan["plan_identity"]
        or spec["authorization"]["identity"]
        != authorization["authorization_identity"]
        or spec["source_readiness_identity"]
        != authorization["source_readiness_identity"]
        or spec["implementation"]["commit"] != str(_git("rev-parse", "HEAD"))
    ):
        raise RetainedPhaseProcessError("resume immutable spec differs")
    records, tail = load_game_ledger(spec, paths.ledger)
    progress = _strict_json(paths.progress)
    if (
        progress.get("spec_identity") != identity
        or progress.get("completed_games") != len(records)
        or progress.get("ledger_tail_record_sha256") != tail
    ):
        raise RetainedPhaseProcessError("resume progress differs from ledger")
    if records and not paths.launch.is_file():
        raise RetainedPhaseProcessError("resume ledger lacks launch evidence")
    return {
        "spec_identity": identity,
        "completed_games": len(records),
        "missing_suffix_games": EXPECTED_GAMES - len(records),
        "ledger_tail_record_sha256": tail,
        "authorization_consumed": paths.launch.is_file(),
    }


def build_readiness_report(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
    *,
    resume: bool,
    run_tests: bool,
    audit_histories: bool,
) -> dict[str, Any]:
    technical: list[dict[str, Any]] = []
    _gate(
        technical,
        "repository",
        "clean published dev containing the tracked plan and implementation",
        lambda: _repository_record(plan, paths),
    )
    _gate(
        technical,
        "plan",
        "canonical 156-game two-hour fixed-corpus process plan",
        lambda: {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": sha256_file(paths.plan),
            "games": plan["workload"]["games"],
            "max_active_hours": plan["workload"]["max_active_hours"],
            "held_out": plan["claim_boundary"]["held_out"],
        },
    )
    _gate(
        technical,
        "outputs",
        "ignored absent runtime targets or one exact valid partial prefix",
        lambda: _output_record(paths, resume=resume),
    )
    _gate(
        technical,
        "inputs",
        "successor-owned read-only route and sidecar-free DB snapshots",
        lambda: _input_record(plan, paths),
    )
    corpus_result = _gate(
        technical,
        "corpus",
        "exact 39-start corpus and adjacent 156-game schedule",
        lambda: _corpus_record(plan, paths),
    )
    if corpus_result is not None:
        technical[-1]["observed"] = corpus_result[0]
    _gate(
        technical,
        "candidates",
        "both exact final checkpoints and successor-owned read-only routes",
        lambda: _candidate_record(plan, paths),
    )
    sanmill_result = _gate(
        technical,
        "sanmill",
        "pinned strict runtime and deterministic 500,000-node canary",
        lambda: _sanmill_record(plan, paths),
    )
    if sanmill_result is not None:
        technical[-1]["observed"] = sanmill_result[0]
    if audit_histories:
        if corpus_result is None or sanmill_result is None:
            technical.append(
                {
                    "gate": "strict_history_replay",
                    "expected": "all 39 variable histories replay without a candidate",
                    "observed": {"error": "corpus or Sanmill gate failed"},
                    "result": "fail",
                }
            )
        else:
            _gate(
                technical,
                "strict_history_replay",
                "all 39 variable histories replay without a candidate",
                lambda: _strict_history_record(
                    corpus_result[1],
                    sanmill_result[1],
                ),
            )
    _gate(
        technical,
        "process_ownership",
        "no competing trainer or evaluator",
        lambda: (
            {"competing_processes": []}
            if not (processes := _competing_processes())
            else _raise_competing(processes)
        ),
    )
    if run_tests:
        _gate(
            technical,
            "tests",
            "focused, mandatory provenance and Ruff checks pass",
            _test_record,
        )

    technical_ready = all(gate["result"] == "pass" for gate in technical)
    source_body = {
        "schema_version": SOURCE_READINESS_SCHEMA,
        "diagnostic_id": plan["diagnostic_id"],
        "plan_identity": plan["plan_identity"],
        "mode": "resume" if resume else "fresh",
        "gates": _source_gate_view(technical),
        "technically_ready": technical_ready,
        "verdict": "ready_for_authorization" if technical_ready else "fatal_stop",
    }
    source_identity = canonical_sha256(source_body)

    authority_gates: list[dict[str, Any]] = []
    authorization = _gate(
        authority_gates,
        "authorization",
        "separate product grant bound to this plan and source readiness",
        lambda: _load_authorization(
            plan,
            paths,
            expected_source_readiness_identity=(
                None if resume else source_identity
            ),
        ),
    )
    if resume and authorization is not None:
        _gate(
            authority_gates,
            "resume_continuity",
            "same plan, grant, source, spec and completed ledger prefix",
            lambda: _resume_record(plan, authorization, paths),
        )
    gates = [*technical, *authority_gates]
    authority_ready = all(gate["result"] == "pass" for gate in authority_gates)
    ready = technical_ready and authority_ready
    verdict = (
        "ready_for_evaluation"
        if ready
        else "fatal_stop"
        if not technical_ready
        else "needs_decision"
    )
    body = {
        "schema_version": READINESS_SCHEMA,
        "diagnostic_id": plan["diagnostic_id"],
        "plan_identity": plan["plan_identity"],
        "mode": "resume" if resume else "fresh",
        "corpus_candidate_moves_requested": 0,
        "corpus_games_played": 0,
        "source_readiness_identity": source_identity,
        "gates": gates,
        "ready": ready,
        "verdict": verdict,
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def require_launch_ready(
    readiness: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if readiness.get("ready") is not True or readiness.get("verdict") != (
        "ready_for_evaluation"
    ):
        raise RetainedPhaseProcessError("phase-process readiness did not pass")
    gates = readiness.get("gates")
    if not isinstance(gates, list):
        raise RetainedPhaseProcessError("phase-process readiness gates are absent")
    required = {
        "repository",
        "plan",
        "outputs",
        "inputs",
        "corpus",
        "candidates",
        "sanmill",
        "strict_history_replay",
        "process_ownership",
        "tests",
        "authorization",
    }
    if resume:
        required.add("resume_continuity")
    observed = [gate.get("gate") for gate in gates if isinstance(gate, Mapping)]
    if set(observed) != required or len(observed) != len(required):
        raise RetainedPhaseProcessError(
            "phase-process readiness skipped or duplicated a required gate"
        )
    if any(gate.get("result") != "pass" for gate in gates):
        raise RetainedPhaseProcessError("phase-process launch gate did not pass")


def _build_spec(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    readiness: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    plan_path: Path,
) -> dict[str, Any]:
    repository_gate = next(
        gate for gate in readiness["gates"] if gate["gate"] == "repository"
    )
    observed = repository_gate["observed"]
    body = {
        "schema_version": SPEC_SCHEMA,
        "diagnostic_id": plan["diagnostic_id"],
        "plan": {
            "identity": plan["plan_identity"],
            "file_sha256": sha256_file(plan_path),
        },
        "authorization": {"identity": authorization["authorization_identity"]},
        "source_readiness_identity": readiness["source_readiness_identity"],
        "implementation": {
            "branch": observed["branch"],
            "commit": observed["head"],
            "tree": observed["tree"],
            "upstream_commit": observed["upstream_commit"],
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": "cpu",
            "precision": "float32",
            "seed": 42,
        },
        "candidates": plan["candidates"],
        "baseline": plan["baseline"],
        "corpus": plan["corpus"],
        "inputs": plan["inputs"],
        "protocol": plan["protocol"],
        "analysis": plan["analysis"],
        "workload": plan["workload"],
        "claim_boundary": plan["claim_boundary"],
        "readiness_identity": readiness["readiness_identity"],
        "schedule": build_schedule(records),
    }
    return {**body, "spec_identity": canonical_sha256(body)}


def _progress_body(
    spec_identity: str,
    *,
    completed_games: int,
    current_game_ordinal: int | None,
    current_stage: str | None,
    current_stage_ply: int,
    active_seconds: float,
    ledger_tail_record_sha256: str | None,
) -> dict[str, Any]:
    body = {
        "schema_version": PROGRESS_SCHEMA,
        "spec_identity": spec_identity,
        "completed_games": completed_games,
        "current_game_ordinal": current_game_ordinal,
        "current_stage": current_stage,
        "current_stage_ply": current_stage_ply,
        "active_seconds": round(active_seconds, 6),
        "ledger_tail_record_sha256": ledger_tail_record_sha256,
    }
    return {**body, "progress_identity": canonical_sha256(body)}


def _launch_body(spec: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": LAUNCH_SCHEMA,
        "diagnostic_id": spec["diagnostic_id"],
        "spec_identity": spec["spec_identity"],
        "authorization_identity": spec["authorization"]["identity"],
        "authorization_consumed": True,
        "consumption_reason": "first fixed-corpus evaluation game opened",
        "first_game_id": spec["schedule"][0]["game_id"],
        "launched_at_utc": utc_now(),
    }
    return {**body, "launch_identity": canonical_sha256(body)}


def _failure_body(
    spec_identity: str,
    exc: BaseException,
    *,
    completed_games: int,
    ledger_tail: str | None,
) -> dict[str, Any]:
    body = {
        "schema_version": FAILURE_SCHEMA,
        "spec_identity": spec_identity,
        "failed_at_utc": utc_now(),
        "error_type": type(exc).__name__,
        "message": str(exc).replace(str(_ROOT), "<repo>"),
        "completed_games": completed_games,
        "ledger_tail_record_sha256": ledger_tail,
        "automatic_retry_allowed": False,
        "resume_after_semantic_failure_allowed": False,
    }
    return {**body, "failure_identity": canonical_sha256(body)}


def run_once(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
    readiness: Mapping[str, Any],
    *,
    resume: bool,
    game_factory: Callable[..., Any] = SanmillTrainingGame,
) -> dict[str, Any]:
    require_launch_ready(readiness, resume=resume)
    authorization = _load_authorization(
        plan,
        paths,
        expected_source_readiness_identity=(
            None if resume else str(readiness["source_readiness_identity"])
        ),
    )
    _corpus_observation, corpus_records = _corpus_record(plan, paths)
    if resume:
        spec = _strict_json(paths.spec)
        _resume_record(plan, authorization, paths)
    else:
        spec = _build_spec(
            plan,
            authorization,
            readiness,
            corpus_records,
            plan_path=paths.plan,
        )

    with EvaluatorLock(paths.output_root, spec["spec_identity"], resume=resume):
        if not resume:
            paths.output_root.mkdir(parents=True, exist_ok=True)
            write_new_canonical(paths.runtime_plan, plan)
            write_new_canonical(paths.readiness, readiness)
            write_new_canonical(paths.spec, spec)
            write_new_canonical(
                paths.progress,
                _progress_body(
                    spec["spec_identity"],
                    completed_games=0,
                    current_game_ordinal=None,
                    current_stage=None,
                    current_stage_ply=0,
                    active_seconds=0.0,
                    ledger_tail_record_sha256=None,
                ),
            )
        records, previous_hash = load_game_ledger(spec, paths.ledger)
        progress = _strict_json(paths.progress)
        if progress.get("completed_games") != len(records):
            raise RetainedPhaseProcessError("progress differs from ledger")
        base_active = max(
            float(progress["active_seconds"]),
            float(records[-1]["cumulative_active_seconds"]) if records else 0.0,
        )
        clock = ActiveClock(
            base_seconds=base_active,
            max_seconds=float(plan["workload"]["max_active_hours"]) * 3600.0,
        )
        policies: dict[str, TrainingAlignedPolicy] = {}
        try:
            for candidate in plan["candidates"]:
                candidate_id = str(candidate["candidate_id"])
                policy = _load_policy(candidate_id, paths)
                if policy.bundle_identity != candidate["bundle"]["identity"]:
                    raise RetainedPhaseProcessError(
                        f"runtime {candidate_id} bundle differs"
                    )
                policies[candidate_id] = policy
            installation = inspect_sanmill_training_installation(
                paths.sanmill_checkout
            )
            corpus_by_id = {
                str(record["start_id"]): record for record in corpus_records
            }
            for ordinal in range(len(records), EXPECTED_GAMES):
                if not paths.launch.exists():
                    write_new_canonical(paths.launch, _launch_body(spec))
                item = spec["schedule"][ordinal]

                def persist(stage: str, stage_ply: int) -> None:
                    replace_canonical(
                        paths.progress,
                        _progress_body(
                            spec["spec_identity"],
                            completed_games=ordinal,
                            current_game_ordinal=ordinal,
                            current_stage=stage,
                            current_stage_ply=stage_ply,
                            active_seconds=clock.require_within_budget(),
                            ledger_tail_record_sha256=previous_hash,
                        ),
                    )

                persist("start", 0)
                record = play_phase_process_game(
                    spec=spec,
                    schedule_item=item,
                    corpus_record=corpus_by_id[item["start_id"]],
                    policy=policies[item["candidate_id"]],
                    installation=installation,
                    previous_record_sha256=previous_hash,
                    clock=clock,
                    progress_callback=persist,
                    game_factory=game_factory,
                )
                previous_hash = append_game_record(
                    paths.ledger,
                    record,
                    must_create=ordinal == 0,
                )
                records.append(record)
                replace_canonical(
                    paths.progress,
                    _progress_body(
                        spec["spec_identity"],
                        completed_games=ordinal + 1,
                        current_game_ordinal=None,
                        current_stage=None,
                        current_stage_ply=0,
                        active_seconds=clock.require_within_budget(),
                        ledger_tail_record_sha256=previous_hash,
                    ),
                )
            report = recompute_report(spec, paths.ledger)
            ledger_sha256 = sha256_file(paths.ledger)

            def audit_progress(game_index: int, turn_index: int) -> None:
                replace_canonical(
                    paths.progress,
                    _progress_body(
                        spec["spec_identity"],
                        completed_games=EXPECTED_GAMES,
                        current_game_ordinal=game_index,
                        current_stage="mechanism_audit",
                        current_stage_ply=turn_index,
                        active_seconds=clock.require_within_budget(),
                        ledger_tail_record_sha256=previous_hash,
                    ),
                )

            mechanism = recompute_mechanism_audit(
                source_spec=spec,
                source_records=records,
                source_ledger_sha256=ledger_sha256,
                source_result_identity=report["result_identity"],
                implementation_commit=spec["implementation"]["commit"],
                malom=policies[EXPECTED_CANDIDATES[0]].malom,
                progress=audit_progress,
            )
            replace_canonical(
                paths.progress,
                _progress_body(
                    spec["spec_identity"],
                    completed_games=EXPECTED_GAMES,
                    current_game_ordinal=None,
                    current_stage=None,
                    current_stage_ply=0,
                    active_seconds=clock.require_within_budget(),
                    ledger_tail_record_sha256=previous_hash,
                ),
            )
            write_new_canonical(paths.report, report)
            mechanism_path = paths.output_root / "mechanism-report.json"
            write_new_canonical(mechanism_path, mechanism)
            completion_body = {
                "schema_version": COMPLETION_SCHEMA,
                "diagnostic_id": spec["diagnostic_id"],
                "spec_identity": spec["spec_identity"],
                "result_identity": report["result_identity"],
                "completed_games": EXPECTED_GAMES,
                "completed_at_utc": utc_now(),
                "ledger_sha256": ledger_sha256,
                "ledger_tail_record_sha256": previous_hash,
                "mechanism_result_identity": mechanism["result_identity"],
                "mechanism_report_sha256": sha256_file(mechanism_path),
            }
            write_new_canonical(
                paths.completion,
                {
                    **completion_body,
                    "completion_identity": canonical_sha256(completion_body),
                },
            )
            return report
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            if not paths.failure.exists():
                write_new_canonical(
                    paths.failure,
                    _failure_body(
                        spec["spec_identity"],
                        exc,
                        completed_games=len(records),
                        ledger_tail=previous_hash,
                    ),
                )
            raise
        finally:
            for policy in policies.values():
                policy.close()


def _load_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], DiagnosticPaths]:
    plan = load_plan(args.plan)
    paths = resolve_paths(
        plan,
        plan_path=args.plan,
        paths_config=args.paths_config,
    )
    return plan, paths


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-history-audit", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("preflight-resume")
    run = commands.add_parser("run")
    run.add_argument("--launch", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--launch", action="store_true")
    commands.add_parser("status")
    commands.add_parser("recompute")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"run", "resume"} and not args.launch:
            raise RetainedPhaseProcessError(
                "run requires the explicit --launch flag"
            )
        if args.command in {"run", "resume"} and (
            args.skip_tests or args.skip_history_audit
        ):
            raise RetainedPhaseProcessError(
                "launch cannot skip tests or the 39-history replay audit"
            )
        plan, paths = _load_context(args)
        if args.command in {"preflight", "preflight-resume"}:
            report = build_readiness_report(
                plan,
                paths,
                resume=args.command == "preflight-resume",
                run_tests=not args.skip_tests,
                audit_histories=not args.skip_history_audit,
            )
            _print(report)
            return 0 if report["ready"] else 2
        if args.command in {"run", "resume"}:
            resume = args.command == "resume"
            readiness = build_readiness_report(
                plan,
                paths,
                resume=resume,
                run_tests=not args.skip_tests,
                audit_histories=not args.skip_history_audit,
            )
            _print(run_once(plan, paths, readiness, resume=resume))
            return 0
        if args.command == "recompute":
            spec = _strict_json(paths.spec)
            records, _tail = load_game_ledger(spec, paths.ledger)
            result = recompute_report(spec, paths.ledger)
            if paths.report.is_file() and _strict_json(paths.report) != result:
                raise RetainedPhaseProcessError(
                    "persisted report differs from recomputation"
                )
            mechanism_path = paths.output_root / "mechanism-report.json"
            mechanism = None
            if mechanism_path.is_file():
                policy = _load_policy(EXPECTED_CANDIDATES[0], paths)
                try:
                    mechanism = recompute_mechanism_audit(
                        source_spec=spec,
                        source_records=records,
                        source_ledger_sha256=sha256_file(paths.ledger),
                        source_result_identity=result["result_identity"],
                        implementation_commit=spec["implementation"]["commit"],
                        malom=policy.malom,
                    )
                finally:
                    policy.close()
                if _strict_json(mechanism_path) != mechanism:
                    raise RetainedPhaseProcessError(
                        "persisted mechanism report differs from recomputation"
                    )
            _print({"report": result, "mechanism": mechanism})
            return 0
        if not paths.spec.is_file():
            _print(
                {
                    "diagnostic_id": plan["diagnostic_id"],
                    "status": "not_started",
                    "authorization_present": paths.authorization.is_file(),
                    "completed_games": 0,
                    "expected_games": EXPECTED_GAMES,
                }
            )
            return 0
        spec = _strict_json(paths.spec)
        records, tail = load_game_ledger(spec, paths.ledger)
        _print(
            {
                "diagnostic_id": spec["diagnostic_id"],
                "spec_identity": spec["spec_identity"],
                "status": (
                    "completed"
                    if paths.completion.is_file()
                    else "failed"
                    if paths.failure.is_file()
                    else "partial"
                ),
                "authorization_consumed": paths.launch.is_file(),
                "completed_games": len(records),
                "expected_games": EXPECTED_GAMES,
                "ledger_tail_record_sha256": tail,
            }
        )
        return 0
    except (RetainedPhaseProcessError, RetainedPassivityDiagnosticError) as exc:
        print(f"retained phase-process confirmation stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
