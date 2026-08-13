#!/usr/bin/env python3
"""Preflight or run the frozen retained-v3/v4 passivity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from game.board import BoardState  # noqa: E402
from game.rules import get_all_legal_moves  # noqa: E402
from learned_ai.delivery.training_route_bundle import (  # noqa: E402
    verify_training_route_bundle,
)
from learned_ai.evaluation.heldout_evaluation import (  # noqa: E402
    ActiveClock,
    EvaluatorLock,
    replace_canonical,
    replay_frozen_prefix,
    utc_now,
    write_new_canonical,
)
from learned_ai.evaluation.heldout_exposure import (  # noqa: E402
    validate_executable_corpus,
)
from learned_ai.evaluation.retained_passivity_diagnostic import (  # noqa: E402
    EXPECTED_CANDIDATES,
    EXPECTED_GAMES,
    MAX_POST_PREFIX_LOGICAL_PLIES,
    PLAN_SCHEMA,
    SANMILL_NODE_CEILING,
    SPEC_SCHEMA,
    RetainedPassivityDiagnosticError,
    append_game_record,
    build_schedule,
    load_game_ledger,
    play_diagnostic_game,
    recompute_diagnostic,
    sha256_file,
)
from learned_ai.evaluation.training_aligned_policy import (  # noqa: E402
    TrainingAlignedPolicy,
    load_training_aligned_policy,
)
from learned_ai.training.checkpoint_envelope import load_checkpoint  # noqa: E402
from learned_ai.training.run_contract import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.sanmill_referee import (  # noqa: E402
    SanmillTrainingGame,
    inspect_sanmill_training_installation,
    probe_sanmill_training_runtime,
    training_installation_record,
)


DEFAULT_PLAN = (
    _ROOT
    / "docs"
    / "experiments"
    / "sanmill-retained-v3-v4-passivity-diagnostic-v1.json"
)
DEFAULT_PATHS = _ROOT / "data" / "training_paths.local.json"
READINESS_SCHEMA = "nmm.retained-passivity-diagnostic-readiness.v1"
AUTHORIZATION_SCHEMA = "nmm.retained-passivity-diagnostic-authorization.v1"
LAUNCH_SCHEMA = "nmm.retained-passivity-diagnostic-launch.v1"
PROGRESS_SCHEMA = "nmm.retained-passivity-diagnostic-progress.v1"
FAILURE_SCHEMA = "nmm.retained-passivity-diagnostic-failure.v1"
COMPLETION_SCHEMA = "nmm.retained-passivity-diagnostic-completion.v1"


@dataclass(frozen=True)
class DiagnosticPaths:
    plan: Path
    paths_config: Path
    corpus: Path
    human_db: Path
    malom_db: Path
    malom_manifest: Path
    sanmill_checkout: Path
    candidate_bundles: Mapping[str, Path]
    candidate_checkpoints: Mapping[str, Path]
    candidate_specialist_dbs: Mapping[str, Path]
    output_root: Path
    authorization: Path
    runtime_plan: Path
    readiness: Path
    spec: Path
    launch: Path
    ledger: Path
    progress: Path
    report: Path
    completion: Path
    failure: Path


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RetainedPassivityDiagnosticError(
            f"cannot read strict JSON: {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise RetainedPassivityDiagnosticError(f"{path.name} must contain an object")
    if canonical_json_bytes(value) != raw:
        raise RetainedPassivityDiagnosticError(f"{path.name} is not canonical JSON")
    return value


def _repo_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RetainedPassivityDiagnosticError(f"{field} is absent")
    path = (_ROOT / value).resolve()
    try:
        path.relative_to(_ROOT)
    except ValueError as exc:
        raise RetainedPassivityDiagnosticError(f"{field} leaves the repository") from exc
    return path


def _local_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RetainedPassivityDiagnosticError(f"{field} is absent")
    path = Path(value)
    if not path.is_absolute():
        path = _ROOT / path
    return path.resolve()


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve(strict=True)
    plan = _strict_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RetainedPassivityDiagnosticError("diagnostic plan schema differs")
    identity = plan.get("plan_identity")
    if not isinstance(identity, str):
        raise RetainedPassivityDiagnosticError("diagnostic plan identity is absent")
    body = {key: value for key, value in plan.items() if key != "plan_identity"}
    if canonical_sha256(body) != identity:
        raise RetainedPassivityDiagnosticError("diagnostic plan identity differs")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or tuple(
        candidate.get("candidate_id") for candidate in candidates
    ) != EXPECTED_CANDIDATES:
        raise RetainedPassivityDiagnosticError("diagnostic candidate order differs")
    workload = plan.get("workload")
    if not isinstance(workload, Mapping) or (
        workload.get("games") != EXPECTED_GAMES
        or workload.get("max_active_hours") != 2.0
    ):
        raise RetainedPassivityDiagnosticError("diagnostic workload differs")
    protocol = plan.get("protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("max_post_prefix_logical_plies")
        != MAX_POST_PREFIX_LOGICAL_PLIES
        or protocol.get("horizon_total_logical_ply") != 120
        or protocol.get("sanmill_node_ceiling_per_turn") != SANMILL_NODE_CEILING
    ):
        raise RetainedPassivityDiagnosticError("diagnostic protocol differs")
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
        raise RetainedPassivityDiagnosticError("local path registry is invalid")
    output_root = _repo_path(plan["outputs"]["root"], field="outputs.root")
    candidates = list(plan["candidates"])
    bundles = {
        str(candidate["candidate_id"]): _repo_path(
            candidate["bundle"]["path"], field="candidate.bundle.path"
        )
        for candidate in candidates
    }
    checkpoints = {
        str(candidate["candidate_id"]): _repo_path(
            candidate["checkpoint"]["path"], field="candidate.checkpoint.path"
        )
        for candidate in candidates
    }
    specialist = {
        str(candidate["candidate_id"]): _repo_path(
            candidate["specialist_db"]["path"], field="candidate.specialist_db.path"
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
            plan["data"]["malom_manifest"], field="data.malom_manifest"
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


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RetainedPassivityDiagnosticError("cannot inspect repository") from exc
    return result.stdout if binary else result.stdout.strip()


def _repository_record(plan: Mapping[str, Any], paths: DiagnosticPaths) -> dict[str, Any]:
    branch = str(_git("branch", "--show-current"))
    head = str(_git("rev-parse", "HEAD"))
    tree = str(_git("rev-parse", "HEAD^{tree}"))
    upstream = str(_git("rev-parse", "@{upstream}"))
    status = str(_git("status", "--porcelain=v1", "--untracked-files=all"))
    if branch != "dev":
        raise RetainedPassivityDiagnosticError("diagnostic must run from dev")
    if status:
        raise RetainedPassivityDiagnosticError("diagnostic requires a clean tree")
    if head != upstream:
        raise RetainedPassivityDiagnosticError("diagnostic requires dev == origin/dev")
    implementation_commit = str(plan["implementation"]["commit"])
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", implementation_commit, head],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RetainedPassivityDiagnosticError(
            "diagnostic implementation commit is not an ancestor"
        ) from exc
    tracked_blob = _git("show", f"HEAD:{paths.plan.relative_to(_ROOT).as_posix()}", binary=True)
    if not isinstance(tracked_blob, bytes) or tracked_blob != paths.plan.read_bytes():
        raise RetainedPassivityDiagnosticError("tracked diagnostic plan differs")
    subprocess.run(
        ["git", "diff", "--check"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        "branch": branch,
        "head": head,
        "tree": tree,
        "upstream_commit": upstream,
        "published": True,
        "tracked_worktree": "clean",
        "implementation_commit_is_ancestor": True,
    }


def _load_authorization(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> dict[str, Any]:
    if not paths.authorization.is_file():
        raise RetainedPassivityDiagnosticError(
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
        raise RetainedPassivityDiagnosticError("diagnostic authorization differs")
    expected = {
        "plan_identity": plan["plan_identity"],
        "plan_file_sha256": sha256_file(paths.plan),
        "games": EXPECTED_GAMES,
        "max_active_hours": 2.0,
        "same_spec_exact_resume": True,
        "automatic_retry": False,
        "training": False,
        "held_out_strength_claim": False,
        "promotion": False,
        "publication": False,
        "release": False,
    }
    for field, value in expected.items():
        if authorization.get("grant", {}).get(field) != value:
            raise RetainedPassivityDiagnosticError(
                f"diagnostic authorization grant differs for {field}"
            )
    plan_commit = authorization.get("grant", {}).get("plan_commit")
    if not isinstance(plan_commit, str):
        raise RetainedPassivityDiagnosticError("authorization plan commit is absent")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", plan_commit, "HEAD"],
            cwd=_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RetainedPassivityDiagnosticError(
            "authorization plan commit is not published in runtime source"
        ) from exc
    plan_blob = _git(
        "show",
        f"{plan_commit}:{paths.plan.relative_to(_ROOT).as_posix()}",
        binary=True,
    )
    if not isinstance(plan_blob, bytes) or hashlib.sha256(plan_blob).hexdigest() != (
        expected["plan_file_sha256"]
    ):
        raise RetainedPassivityDiagnosticError("authorized plan blob differs")
    return authorization


def build_authorization(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    plan_commit: str,
    authority_text_sha256: str,
    operator: str = "product-owner-direct",
) -> dict[str, Any]:
    """Build, but do not write, the exact ordinary grant after owner approval."""
    body = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "diagnostic_id": plan["diagnostic_id"],
        "operator": operator,
        "authority_text_sha256": authority_text_sha256,
        "grant": {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": sha256_file(plan_path),
            "plan_commit": plan_commit,
            "games": EXPECTED_GAMES,
            "max_active_hours": 2.0,
            "same_spec_exact_resume": True,
            "automatic_retry": False,
            "training": False,
            "held_out_strength_claim": False,
            "promotion": False,
            "publication": False,
            "release": False,
        },
    }
    return {**body, "authorization_identity": canonical_sha256(body)}


def _assert_ignored(path: Path) -> None:
    try:
        relative = path.resolve(strict=False).relative_to(_ROOT).as_posix()
    except ValueError as exc:
        raise RetainedPassivityDiagnosticError("diagnostic output leaves repository") from exc
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RetainedPassivityDiagnosticError("diagnostic output is not ignored")


def _output_record(paths: DiagnosticPaths, *, resume: bool) -> dict[str, Any]:
    for path in (
        paths.runtime_plan,
        paths.readiness,
        paths.spec,
        paths.launch,
        paths.ledger,
        paths.progress,
        paths.report,
        paths.completion,
        paths.failure,
        paths.output_root / "evaluator.lock",
    ):
        _assert_ignored(path)
    if not all(path.is_dir() for path in paths.candidate_bundles.values()):
        raise RetainedPassivityDiagnosticError("candidate route bundle is absent")
    runtime_targets = (
        paths.runtime_plan,
        paths.readiness,
        paths.spec,
        paths.launch,
        paths.ledger,
        paths.progress,
        paths.report,
        paths.completion,
        paths.failure,
    )
    if not resume:
        existing = [path.name for path in runtime_targets if path.exists()]
        if existing:
            raise RetainedPassivityDiagnosticError(
                "fresh diagnostic runtime targets already exist: " + ", ".join(existing)
            )
    else:
        if paths.failure.exists():
            raise RetainedPassivityDiagnosticError(
                "failed diagnostic cannot be retried or resumed"
            )
        if paths.report.exists() or paths.completion.exists():
            raise RetainedPassivityDiagnosticError("completed diagnostic cannot resume")
        for required in (
            paths.runtime_plan,
            paths.readiness,
            paths.spec,
            paths.progress,
        ):
            if not required.is_file():
                raise RetainedPassivityDiagnosticError(
                    "same-spec resume runtime prefix is incomplete"
                )
    return {
        "root_ignored": True,
        "route_bundles_present": sorted(paths.candidate_bundles),
        "mode": "resume" if resume else "fresh",
        "runtime_targets": "validated partial" if resume else "absent",
    }


def _corpus_record(plan: Mapping[str, Any], paths: DiagnosticPaths) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if sha256_file(paths.corpus) != plan["corpus"]["file_sha256"]:
        raise RetainedPassivityDiagnosticError("diagnostic corpus file differs")
    payload = json.loads(paths.corpus.read_text(encoding="utf-8"))
    records = validate_executable_corpus(
        payload,
        expected_corpus_identity=plan["corpus"]["identity"],
        expected_records_identity=plan["corpus"]["records_identity"],
    )
    schedule = build_schedule(records)
    return (
        {
            "records": len(records),
            "development_reuse": True,
            "corpus_identity": plan["corpus"]["identity"],
            "schedule_identity": canonical_sha256(schedule),
            "games": len(schedule),
        },
        records,
    )


def _candidate_record(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for candidate in plan["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        checkpoint_path = paths.candidate_checkpoints[candidate_id]
        if sha256_file(checkpoint_path) != candidate["checkpoint"]["file_sha256"]:
            raise RetainedPassivityDiagnosticError(
                f"{candidate_id} checkpoint file differs"
            )
        envelope = load_checkpoint(checkpoint_path, map_location="cpu")
        if (
            envelope.payload_sha256 != candidate["checkpoint"]["payload_sha256"]
            or envelope.descriptor.checkpoint_id != candidate["checkpoint"]["id"]
        ):
            raise RetainedPassivityDiagnosticError(
                f"{candidate_id} checkpoint payload differs"
            )
        bundle = verify_training_route_bundle(
            paths.candidate_bundles[candidate_id], device="cpu"
        )
        if bundle["bundle_identity"] != candidate["bundle"]["identity"]:
            raise RetainedPassivityDiagnosticError(
                f"{candidate_id} route bundle differs"
            )
        if sha256_file(paths.candidate_specialist_dbs[candidate_id]) != candidate[
            "specialist_db"
        ]["file_sha256"]:
            raise RetainedPassivityDiagnosticError(
                f"{candidate_id} SpecialistDB differs"
            )
        policy = load_training_aligned_policy(
            paths.candidate_bundles[candidate_id],
            human_db_path=paths.human_db,
            specialist_db_path=paths.candidate_specialist_dbs[candidate_id],
            malom_path=paths.malom_db,
            malom_manifest_path=paths.malom_manifest,
            device="cpu",
        )
        try:
            board = BoardState.new_game()
            first = policy.choose_move(board)
            second = policy.choose_move(board)
            if first != second or first not in get_all_legal_moves(board):
                raise RetainedPassivityDiagnosticError(
                    f"{candidate_id} synthetic argmax is not deterministic and legal"
                )
            observed[candidate_id] = {
                "checkpoint_id": envelope.descriptor.checkpoint_id,
                "checkpoint_file_sha256": candidate["checkpoint"]["file_sha256"],
                "checkpoint_payload_sha256": envelope.payload_sha256,
                "bundle_identity": policy.bundle_identity,
                "specialist_db_sha256": candidate["specialist_db"]["file_sha256"],
                "human_db_identity": policy.resource_reports["human_db"]["identity"],
                "malom_identity": policy.resource_reports["malom_tablebase"]["identity"],
                "synthetic_move_identity": canonical_sha256(first),
                "corpus_moves_requested": 0,
            }
        finally:
            policy.close()
    return observed


def _sanmill_record(plan: Mapping[str, Any], paths: DiagnosticPaths) -> tuple[dict[str, Any], Any]:
    installation = inspect_sanmill_training_installation(paths.sanmill_checkout)
    installed = training_installation_record(installation, seed=42)
    baseline = plan["baseline"]
    for field in ("commit", "tree", "binary_sha256"):
        if installed[field] != baseline[field]:
            raise RetainedPassivityDiagnosticError(f"Sanmill {field} differs")
    if installed["strict_referee"]["semanticDigest"] != baseline[
        "strict_referee_semantic_digest"
    ]:
        raise RetainedPassivityDiagnosticError("Sanmill strict referee differs")
    probe = probe_sanmill_training_runtime(
        paths.sanmill_checkout,
        node_budget=SANMILL_NODE_CEILING,
        depth=None,
        seed=42,
    )
    return (
        {
            "commit": installed["commit"],
            "tree": installed["tree"],
            "binary_sha256": installed["binary_sha256"],
            "strict_referee_semantic_digest": installed["strict_referee"][
                "semanticDigest"
            ],
            "runtime_identity": installed["identity"],
            "deterministic_fresh_processes": probe["probe"]["fresh_processes"],
            "node_ceiling": probe["probe"]["node_budget"],
            "probe_observation_sha256": probe["probe"]["observation_sha256"],
        },
        installation,
    )


def _prefix_record(
    records: Sequence[Mapping[str, Any]],
    installation: Any,
) -> dict[str, Any]:
    observed = []
    for record in records:
        with SanmillTrainingGame(installation, seed=42) as game:
            _board, prefix = replay_frozen_prefix(game, record)
        observed.append(
            {
                "source_core_id": record["source_core_id"],
                "prefix_identity": prefix["prefix_identity"],
                "history_sha256": prefix["observed_history_sha256"],
            }
        )
    return {
        "records": len(observed),
        "fresh_processes": len(observed),
        "candidate_loaded": False,
        "games_played": 0,
        "observations_identity": canonical_sha256(observed),
    }


def _competing_processes() -> list[dict[str, Any]]:
    if os.name != "nt":
        raise RetainedPassivityDiagnosticError(
            "process audit is implemented only for the Windows training host"
        )
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        raise RetainedPassivityDiagnosticError("PowerShell is unavailable")
    pattern = (
        "train_s_gen_v2\\.py|manage_generalist_run\\.py|"
        "run_heldout_evaluation\\.py|run_retained_passivity_diagnostic\\.py"
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
        raise RetainedPassivityDiagnosticError("process audit shape differs")
    return [
        {
            "pid": int(item["ProcessId"]),
            "name": str(item.get("Name") or "unknown"),
        }
        for item in payload
    ]


def _run_check(command: Sequence[str], *, label: str) -> dict[str, Any]:
    result = subprocess.run(
        list(command),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RetainedPassivityDiagnosticError(
            f"{label} failed with exit {result.returncode}"
        )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return {
        "label": label,
        "exit_code": result.returncode,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "last_output_line": lines[-1] if lines else "",
    }


def _test_record() -> dict[str, Any]:
    focused = _run_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_retained_passivity_diagnostic.py",
            "tests/test_training_aligned_policy.py",
            "tests/test_sanmill_training_referee.py",
            "tests/test_training_route_bundle.py",
            "tests/test_checkpoint_envelope.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            ".tmp/pytest-retained-passivity-preflight",
        ],
        label="retained passivity focused tests",
    )
    mandatory = _run_check(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_malom_db.py",
            "tests/test_sentinel_db_teacher.py",
            "tests/test_malom_label_provenance.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            ".tmp/pytest-retained-passivity-provenance",
        ],
        label="mandatory Malom and provenance tests",
    )
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RetainedPassivityDiagnosticError("Ruff is unavailable")
    lint = _run_check(
        [
            ruff,
            "check",
            "learned_ai/evaluation/retained_passivity_diagnostic.py",
            "scripts/run_retained_passivity_diagnostic.py",
            "tools/serve_retained_passivity_diagnostic.py",
            "tests/test_retained_passivity_diagnostic.py",
        ],
        label="retained passivity Ruff checks",
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


def _resume_record(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    paths: DiagnosticPaths,
) -> dict[str, Any]:
    if paths.runtime_plan.read_bytes() != paths.plan.read_bytes():
        raise RetainedPassivityDiagnosticError("resume plan copy differs")
    spec = _strict_json(paths.spec)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise RetainedPassivityDiagnosticError("resume spec schema differs")
    identity = spec.get("spec_identity")
    body = {key: value for key, value in spec.items() if key != "spec_identity"}
    if canonical_sha256(body) != identity:
        raise RetainedPassivityDiagnosticError("resume spec identity differs")
    if (
        spec["plan"]["identity"] != plan["plan_identity"]
        or spec["authorization"]["identity"]
        != authorization["authorization_identity"]
        or spec["implementation"]["commit"] != str(_git("rev-parse", "HEAD"))
    ):
        raise RetainedPassivityDiagnosticError("resume immutable spec differs")
    records, tail = load_game_ledger(spec, paths.ledger)
    progress = _strict_json(paths.progress)
    if progress.get("spec_identity") != identity:
        raise RetainedPassivityDiagnosticError("resume progress spec differs")
    if progress.get("completed_games") != len(records):
        raise RetainedPassivityDiagnosticError("resume progress differs from ledger")
    expected_tail = tail if records else None
    if progress.get("ledger_tail_record_sha256") != expected_tail:
        raise RetainedPassivityDiagnosticError("resume ledger tail differs")
    if records and not paths.launch.is_file():
        raise RetainedPassivityDiagnosticError("resume ledger lacks launch evidence")
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
    audit_prefixes: bool,
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    _gate(
        gates,
        "repository",
        "clean published dev containing the tracked plan and implementation",
        lambda: _repository_record(plan, paths),
    )
    _gate(
        gates,
        "plan",
        "canonical plan with exact 256-game two-hour development claim boundary",
        lambda: {
            "plan_identity": plan["plan_identity"],
            "plan_file_sha256": sha256_file(paths.plan),
            "games": plan["workload"]["games"],
            "max_active_hours": plan["workload"]["max_active_hours"],
            "development_corpus_reused": plan["claim_boundary"][
                "development_corpus_reused"
            ],
        },
    )
    authorization = _gate(
        gates,
        "authorization",
        "separate exact plan-bound product grant",
        lambda: _load_authorization(plan, paths),
    )
    _gate(
        gates,
        "outputs",
        "ignored absent runtime targets or one exact valid partial prefix",
        lambda: _output_record(paths, resume=resume),
    )
    corpus_result = _gate(
        gates,
        "corpus",
        "exact reused 64-start corpus and adjacent 256-game schedule",
        lambda: _corpus_record(plan, paths),
    )
    _gate(
        gates,
        "candidates",
        "both exact final checkpoints, route bundles and read-only data routes",
        lambda: _candidate_record(plan, paths),
    )
    sanmill_result = _gate(
        gates,
        "sanmill",
        "pinned strict runtime and deterministic 500,000-node canary",
        lambda: _sanmill_record(plan, paths),
    )
    if sanmill_result is not None:
        gates[-1]["observed"] = sanmill_result[0]
    if audit_prefixes:
        if corpus_result is None or sanmill_result is None:
            gates.append(
                {
                    "gate": "prefix_replay",
                    "expected": "all 64 prefixes replay without loading a candidate",
                    "observed": {"error": "corpus or Sanmill gate failed"},
                    "result": "fail",
                }
            )
        else:
            _gate(
                gates,
                "prefix_replay",
                "all 64 prefixes replay without loading a candidate",
                lambda: _prefix_record(corpus_result[1], sanmill_result[1]),
            )
    _gate(
        gates,
        "process_ownership",
        "no competing trainer or evaluator",
        lambda: (
            {"competing_processes": []}
            if not (processes := _competing_processes())
            else (_raise_competing(processes))
        ),
    )
    if resume and authorization is not None:
        _gate(
            gates,
            "resume_continuity",
            "same plan, grant, source, spec and completed ledger prefix",
            lambda: _resume_record(plan, authorization, paths),
        )
    if run_tests:
        _gate(
            gates,
            "tests",
            "focused, mandatory provenance and Ruff checks pass",
            _test_record,
        )

    non_authority_failures = [
        gate
        for gate in gates
        if gate["result"] != "pass" and gate["gate"] != "authorization"
    ]
    authority_pass = any(
        gate["gate"] == "authorization" and gate["result"] == "pass"
        for gate in gates
    )
    ready = not non_authority_failures and authority_pass
    if non_authority_failures:
        verdict = "fatal_stop"
    elif not authority_pass:
        verdict = "needs_decision"
    else:
        verdict = "ready_for_long_run"
    body = {
        "schema_version": READINESS_SCHEMA,
        "diagnostic_id": plan["diagnostic_id"],
        "plan_identity": plan["plan_identity"],
        "mode": "resume" if resume else "fresh",
        "corpus_candidate_moves_requested": 0,
        "corpus_games_played": 0,
        "gates": gates,
        "ready": ready,
        "verdict": verdict,
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def _raise_competing(processes: list[dict[str, Any]]) -> None:
    raise RetainedPassivityDiagnosticError(
        "competing trainer/evaluator processes exist: "
        + ", ".join(str(item["pid"]) for item in processes)
    )


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
        "authorization": {
            "identity": authorization["authorization_identity"],
        },
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
        "consumption_reason": "first development corpus game opened",
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


def _load_policy(candidate_id: str, paths: DiagnosticPaths) -> TrainingAlignedPolicy:
    return load_training_aligned_policy(
        paths.candidate_bundles[candidate_id],
        human_db_path=paths.human_db,
        specialist_db_path=paths.candidate_specialist_dbs[candidate_id],
        malom_path=paths.malom_db,
        malom_manifest_path=paths.malom_manifest,
        device="cpu",
    )


def run_once(
    plan: Mapping[str, Any],
    paths: DiagnosticPaths,
    readiness: Mapping[str, Any],
    *,
    resume: bool,
    game_factory: Callable[..., Any] = SanmillTrainingGame,
) -> dict[str, Any]:
    if readiness.get("ready") is not True or readiness.get("verdict") != (
        "ready_for_long_run"
    ):
        raise RetainedPassivityDiagnosticError("diagnostic readiness did not pass")
    authorization = _load_authorization(plan, paths)
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
            raise RetainedPassivityDiagnosticError("progress differs from ledger")
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
                    raise RetainedPassivityDiagnosticError(
                        f"runtime {candidate_id} bundle differs"
                    )
                policies[candidate_id] = policy
            installation = inspect_sanmill_training_installation(paths.sanmill_checkout)
            corpus_by_id = {
                str(record["source_core_id"]): record for record in corpus_records
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

                persist("opening", 0)
                record = play_diagnostic_game(
                    spec=spec,
                    schedule_item=item,
                    corpus_record=corpus_by_id[item["source_core_id"]],
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
            report = recompute_diagnostic(spec, paths.ledger)
            write_new_canonical(paths.report, report)
            completion_body = {
                "schema_version": COMPLETION_SCHEMA,
                "diagnostic_id": spec["diagnostic_id"],
                "spec_identity": spec["spec_identity"],
                "result_identity": report["result_identity"],
                "completed_games": EXPECTED_GAMES,
                "completed_at_utc": utc_now(),
                "ledger_sha256": sha256_file(paths.ledger),
                "ledger_tail_record_sha256": previous_hash,
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


def _load_context(args: argparse.Namespace) -> tuple[dict[str, Any], DiagnosticPaths]:
    plan = load_plan(args.plan)
    paths = resolve_paths(plan, plan_path=args.plan, paths_config=args.paths_config)
    return plan, paths


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--paths-config", default=str(DEFAULT_PATHS))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-prefix-audit", action="store_true")
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
            raise RetainedPassivityDiagnosticError(
                "run requires the explicit --launch flag"
            )
        plan, paths = _load_context(args)
        if args.command in {"preflight", "preflight-resume"}:
            report = build_readiness_report(
                plan,
                paths,
                resume=args.command == "preflight-resume",
                run_tests=not args.skip_tests,
                audit_prefixes=not args.skip_prefix_audit,
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
                audit_prefixes=not args.skip_prefix_audit,
            )
            _print(run_once(plan, paths, readiness, resume=resume))
            return 0
        if args.command == "recompute":
            spec = _strict_json(paths.spec)
            result = recompute_diagnostic(spec, paths.ledger)
            if paths.report.is_file() and _strict_json(paths.report) != result:
                raise RetainedPassivityDiagnosticError(
                    "persisted report differs from recomputation"
                )
            _print(result)
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
    except RetainedPassivityDiagnosticError as exc:
        print(f"retained passivity diagnostic stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
