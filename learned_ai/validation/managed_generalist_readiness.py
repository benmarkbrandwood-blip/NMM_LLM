"""Reproducible, fail-closed readiness evidence for one managed run plan."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from learned_ai.training.generalist_run_manifest import utc_now_text
from learned_ai.training.managed_generalist import (
    CONTROLLER_LEDGER_NAME,
    ManagedPlan,
    build_segment_command,
    load_managed_plan,
)
from learned_ai.training.run_contract import (
    canonical_json_bytes,
    canonical_sha256,
    load_run_events,
)


READINESS_SCHEMA = "nmm.managed-generalist-readiness.v1"
COMMAND_SCHEMA = "nmm.managed-generalist-preflight-command.v1"
PRODUCT_AUTHORIZATION_DECISION = (
    "long-run launch requires a frozen managed plan and separate "
    "product authorization"
)


class ManagedReadinessError(RuntimeError):
    """A launch input, raw preflight, or persisted identity differs."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ManagedReadinessError(
                    f"duplicate JSON key {key!r}: {label}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManagedReadinessError(f"invalid JSON object: {label}") from exc
    if not isinstance(value, dict):
        raise ManagedReadinessError(f"JSON root is not an object: {label}")
    return value


def _repository_path(root: Path, value: str | Path, *, field: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ManagedReadinessError(
            f"{field} must stay inside the repository"
        ) from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve()).as_posix()


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManagedReadinessError(
            "Git audit failed: " + " ".join(arguments)
        ) from exc
    return result.stdout.strip()


def inspect_published_source(
    root: Path,
    *,
    plan: ManagedPlan,
    reviewed_main: str,
) -> dict[str, Any]:
    """Require the exact clean dev commit frozen by the managed plan."""
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    origin_dev = _git_output(root, "rev-parse", "origin/dev")
    origin_main = _git_output(root, "rev-parse", "origin/main")
    status = _git_output(
        root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if branch != "dev":
        raise ManagedReadinessError("managed readiness requires branch dev")
    if head != origin_dev or head != plan.git_commit:
        raise ManagedReadinessError(
            "HEAD, origin/dev, and the managed-plan commit must match"
        )
    if origin_main != reviewed_main:
        raise ManagedReadinessError("origin/main moved after source review")
    if status:
        raise ManagedReadinessError("tracked worktree must be clean")
    diff_check = subprocess.run(
        ["git", "diff", "--check"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if diff_check.returncode != 0:
        raise ManagedReadinessError("git diff --check failed")
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "origin_main_reviewed": origin_main,
        "tracked_worktree_clean": True,
        "git_diff_check": "passed",
    }


def build_first_segment_preflight_command(
    plan: ManagedPlan,
    *,
    root: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build the trainer preflight for the plan's real first segment."""
    segment_index = 1
    initial = plan.initial_resume
    completed_games = 0 if initial is None else initial.completed_games
    segment_stop_game = min(
        completed_games + plan.segment_games,
        plan.game_bound,
    )
    command = [
        python_executable,
        str(root / "scripts/train_s_gen_v2.py"),
        "--preflight",
        "long-run",
        "--run-id",
        f"{plan.plan_id}-segment-{segment_index:04d}",
        "--out-dir",
        str(
            Path(plan.control_dir)
            / "segments"
            / f"segment-{segment_index:04d}"
        ),
        "--segment-games",
        str(plan.segment_games),
        "--segment-stop-game",
        str(segment_stop_game),
        *plan.common_trainer_args,
    ]
    if initial is None:
        command.extend(("--start-mode", "fresh"))
    else:
        command.extend(
            (
                "--start-mode",
                "exact-resume",
                "--resume",
                initial.checkpoint_path,
                "--parent-run-id",
                initial.parent_run_id,
            )
        )
    return command


def _specialist_db_path(plan: ManagedPlan) -> Path:
    arguments = list(plan.common_trainer_args)
    try:
        index = arguments.index("--specialist-db")
        value = arguments[index + 1]
    except (ValueError, IndexError) as exc:
        raise ManagedReadinessError(
            "managed plan must explicitly freeze --specialist-db"
        ) from exc
    return Path(value).resolve(strict=False)


def inspect_empty_specialist_db(root: Path, plan: ManagedPlan) -> dict[str, Any]:
    """Audit the fresh writable DB without creating SQLite sidecars."""
    path = _specialist_db_path(plan)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ManagedReadinessError(
            "managed SpecialistDB must stay inside the repository"
        ) from exc
    if not path.is_file():
        raise ManagedReadinessError("managed SpecialistDB is missing")
    sidecars = [
        Path(f"{path}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
        if Path(f"{path}{suffix}").exists()
    ]
    if sidecars:
        raise ManagedReadinessError(
            "managed SpecialistDB has SQLite sidecars: "
            + ", ".join(item.name for item in sidecars)
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        connection.execute("PRAGMA query_only=ON")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        metadata = dict(connection.execute("SELECT key, value FROM meta"))
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
            for table in ("positions", "winning_lines", "preferred_plays")
        }
    except sqlite3.Error as exc:
        raise ManagedReadinessError(
            "managed SpecialistDB immutable audit failed"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
    if quick_check != ("ok",):
        raise ManagedReadinessError("managed SpecialistDB quick_check failed")
    if metadata.get("malom_label_version") != "sector-corrected-v1":
        raise ManagedReadinessError(
            "managed SpecialistDB Malom label version differs"
        )
    if counts != {
        "positions": 0,
        "winning_lines": 0,
        "preferred_plays": 0,
    }:
        raise ManagedReadinessError("managed SpecialistDB is not empty")
    return {
        "path": _relative(root, path),
        "file_sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "quick_check": "ok",
        "malom_label_version": metadata["malom_label_version"],
        "counts": counts,
        "sidecars": [],
        "immutable_read_only_audit": True,
    }


def _inspect_controller(root: Path, plan: ManagedPlan) -> dict[str, Any]:
    control_dir = Path(plan.control_dir).resolve(strict=False)
    authorization = control_dir / "authorization.json"
    segments = control_dir / "segments"
    if authorization.exists():
        raise ManagedReadinessError("managed plan is already authorized")
    if segments.exists():
        raise ManagedReadinessError("managed segment output already exists")
    ledger = control_dir / CONTROLLER_LEDGER_NAME
    events = load_run_events(ledger)
    if (
        len(events) != 1
        or events[0].sequence != 0
        or events[0].event_type != "managed_plan_published"
        or events[0].run_id != plan.plan_id
        or events[0].details.get("plan_sha256") != plan.plan_sha256
    ):
        raise ManagedReadinessError(
            "controller ledger is not one unlaunched published-plan event"
        )
    return {
        "ledger": _relative(root, ledger),
        "ledger_sha256": _sha256_file(ledger),
        "event_count": 1,
        "last_event_identity": events[0].event_sha256,
        "last_event_type": events[0].event_type,
        "completed_games": 0,
        "completed_segments": 0,
        "authorization_present": False,
        "segments_present": False,
    }


def _assert_tracked(root: Path, path: Path) -> None:
    relative = _relative(root, path)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ManagedReadinessError(
            f"experiment document is not tracked: {relative}"
        )


def _assert_ignored(root: Path, path: Path) -> None:
    relative = _relative(root, path)
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ManagedReadinessError(
            f"machine-local evidence must be ignored: {relative}"
        )


def _validate_preflight(
    preflight: Mapping[str, Any],
    *,
    plan: ManagedPlan,
    source_commit: str,
) -> None:
    if preflight.get("schema_version") != "nmm.generalist-preflight.v1":
        raise ManagedReadinessError("preflight schema differs")
    if preflight.get("mode") != "long-run":
        raise ManagedReadinessError("preflight mode differs")
    if preflight.get("verdict") != "needs_decision":
        raise ManagedReadinessError("preflight verdict differs")
    if preflight.get("errors") != []:
        raise ManagedReadinessError("preflight has technical errors")
    if preflight.get("unresolved_decisions") != [
        PRODUCT_AUTHORIZATION_DECISION
    ]:
        raise ManagedReadinessError(
            "preflight has an unresolved decision beyond launch authority"
        )
    if preflight.get("resume_config_sha256") != plan.resume_config_sha256:
        raise ManagedReadinessError("preflight does not bind managed semantics")
    git = preflight.get("git")
    if (
        not isinstance(git, Mapping)
        or git.get("commit") != source_commit
        or git.get("dirty") is not False
    ):
        raise ManagedReadinessError("preflight source identity differs")
    config = preflight.get("resolved_config")
    initial = plan.initial_resume
    completed_games = 0 if initial is None else initial.completed_games
    expected = {
        "experiment_id": plan.experiment_id,
        "run_id": f"{plan.plan_id}-segment-0001",
        "segment_games": plan.segment_games,
        "segment_stop_game": min(
            completed_games + plan.segment_games,
            plan.game_bound,
        ),
        "start_mode": "fresh" if initial is None else "exact-resume",
    }
    if not isinstance(config, Mapping) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise ManagedReadinessError("preflight first-segment contract differs")
    if initial is None and (
        config.get("resume") or config.get("auto_resume_best")
    ):
        raise ManagedReadinessError("fresh preflight selects a resume source")
    expected_output = (
        Path(plan.control_dir) / "segments" / "segment-0001"
    ).resolve(strict=False)
    observed_output = Path(str(config.get("out_dir", ""))).resolve(
        strict=False
    )
    if observed_output != expected_output:
        raise ManagedReadinessError("preflight output directory differs")
    checks = preflight.get("checks")
    output = checks.get("output") if isinstance(checks, Mapping) else None
    if output != {
        "exists": False,
        "isolated": True,
        "kind": "run_directory",
    }:
        raise ManagedReadinessError("preflight output isolation differs")


def _dependency_identities(preflight: Mapping[str, Any]) -> dict[str, Any]:
    checks = preflight.get("checks")
    if not isinstance(checks, Mapping):
        raise ManagedReadinessError("preflight checks are missing")
    required = ("human_db", "malom", "ruleset", "sanmill_training")
    if any(not isinstance(checks.get(key), Mapping) for key in required):
        raise ManagedReadinessError("preflight dependency identity is missing")
    human = checks["human_db"]
    malom = checks["malom"]
    ruleset = checks["ruleset"]
    sanmill = checks["sanmill_training"]
    strict_identity = (
        sanmill.get("probe", {})
        .get("first_turn", {})
        .get("state", {})
        .get("strict_referee_identity", {})
    )
    return {
        "human_db": {
            "identity": human.get("identity"),
            "quick_check": human.get("quick_check"),
            "trust": human.get("trust"),
            "malom_columns_policy": human.get("malom_columns_policy"),
        },
        "malom": {
            "identity": malom.get("identity"),
            "component_count": malom.get("component_count"),
            "size_bytes": malom.get("size_bytes"),
            "manifest_schema": malom.get("manifest_schema"),
        },
        "ruleset": {
            "id": ruleset.get("id"),
            "version": ruleset.get("version"),
            "semanticDigest": ruleset.get("semanticDigest"),
            "documentDigest": ruleset.get("documentDigest"),
        },
        "mif_suite": preflight.get("mifSuite"),
        "sanmill_training": {
            "identity": sanmill.get("identity"),
            "commit": sanmill.get("commit"),
            "checkout_head": sanmill.get("checkout_head"),
            "binary_sha256": sanmill.get("binary_sha256"),
            "binary_size": sanmill.get("binary_size"),
            "strict_referee_identity": strict_identity,
            "probe_observation_sha256": sanmill.get("probe", {}).get(
                "observation_sha256"
            ),
        },
        "path_sources": preflight.get("path_sources"),
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        raise ManagedReadinessError(
            f"readiness artifact already exists: {path}"
        ) from exc


def generate_readiness(
    *,
    root: Path,
    plan_path: Path,
    experiment_document: Path,
    reviewed_main: str,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    """Run and persist a real first-segment preflight without authorizing it."""
    root = root.resolve()
    plan_path = _repository_path(root, plan_path, field="managed plan")
    experiment_document = _repository_path(
        root, experiment_document, field="experiment document"
    )
    if not plan_path.is_file() or not experiment_document.is_file():
        raise ManagedReadinessError("plan or experiment document is missing")
    plan = load_managed_plan(plan_path)
    if plan_path.parent.resolve() != Path(plan.control_dir).resolve():
        raise ManagedReadinessError("plan is outside its control directory")
    _assert_tracked(root, experiment_document)
    source = inspect_published_source(
        root,
        plan=plan,
        reviewed_main=reviewed_main,
    )
    controller = _inspect_controller(root, plan)
    database = inspect_empty_specialist_db(root, plan)
    if _sha256_file(Path(plan.paths_config)) != plan.paths_config_sha256:
        raise ManagedReadinessError("local path registry changed")
    if plan.publication_allowed or plan.promotion_allowed:
        raise ManagedReadinessError("managed plan permits publication or promotion")

    control_dir = Path(plan.control_dir).resolve()
    command_path = control_dir / "first-segment-preflight-command.json"
    preflight_path = control_dir / "first-segment-preflight.json"
    readiness_path = control_dir / "technical-readiness.json"
    for path in (command_path, preflight_path, readiness_path):
        _assert_ignored(root, path)
        if path.exists():
            raise ManagedReadinessError(
                f"readiness artifact already exists: {_relative(root, path)}"
            )

    preflight_command = build_first_segment_preflight_command(
        plan,
        root=root,
        python_executable=python_executable,
    )
    initial = plan.initial_resume
    launch_command = build_segment_command(
        plan,
        plan_path=plan_path,
        authorization_path=control_dir / "authorization.json",
        segment_index=1,
        previous_checkpoint=(
            None if initial is None else Path(initial.checkpoint_path)
        ),
        previous_run_id=None if initial is None else initial.parent_run_id,
        previous_completed_games=0 if initial is None else initial.completed_games,
        python_executable=python_executable,
    )
    try:
        result = runner(
            preflight_command,
            cwd=root,
            check=False,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagedReadinessError("first-segment preflight could not run") from exc
    if result.returncode != 2:
        raise ManagedReadinessError(
            f"first-segment preflight returned {result.returncode}; expected 2"
        )
    stdout = result.stdout
    stderr = result.stderr
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        raise ManagedReadinessError("preflight runner must return raw bytes")
    preflight = _strict_json_bytes(stdout, label="first-segment preflight")
    _validate_preflight(preflight, plan=plan, source_commit=source["head"])

    command_record = {
        "schema_version": COMMAND_SCHEMA,
        "working_directory": str(root),
        "preflight_argv": preflight_command,
        "launch_argv": launch_command,
    }
    command_bytes = canonical_json_bytes(command_record)
    dependencies = _dependency_identities(preflight)
    policy_health = None
    if plan.policy_health is not None:
        policy_health = {
            **plan.policy_health.to_dict(),
            "corpus_path": _relative(root, Path(plan.policy_health.corpus_path)),
            "audit_script_path": _relative(
                root, Path(plan.policy_health.audit_script_path)
            ),
        }
    report_body = {
        "schema_version": READINESS_SCHEMA,
        "generated_at_utc": utc_now_text(),
        "state": "technically_ready_awaiting_product_authorization",
        "verdict": "needs_decision",
        "launch_authorized": False,
        "source": source,
        "experiment_document": {
            "path": _relative(root, experiment_document),
            "file_sha256": _sha256_file(experiment_document),
        },
        "plan": {
            "path": _relative(root, plan_path),
            "file_sha256": _sha256_file(plan_path),
            "plan_id": plan.plan_id,
            "plan_identity": plan.plan_sha256,
            "experiment_id": plan.experiment_id,
            "objective": plan.objective,
            "git_commit": plan.git_commit,
            "paths_config_sha256": plan.paths_config_sha256,
            "resume_config_sha256": plan.resume_config_sha256,
            "max_games": plan.max_games,
            "game_bound": plan.game_bound,
            "segment_games": plan.segment_games,
            "max_active_wall_hours": plan.max_wall_hours,
            "fresh_start": plan.initial_resume is None,
            "publication_allowed": plan.publication_allowed,
            "promotion_allowed": plan.promotion_allowed,
        },
        "controller": controller,
        "database": database,
        "policy_health": policy_health,
        "first_segment": {
            "preflight_command": preflight_command,
            "launch_command": launch_command,
            "command_artifact": {
                "path": _relative(root, command_path),
                "sha256": _sha256_bytes(command_bytes),
            },
            "raw_preflight_artifact": {
                "path": _relative(root, preflight_path),
                "sha256": _sha256_bytes(stdout),
                "byte_length": len(stdout),
            },
            "raw_stderr_sha256": _sha256_bytes(stderr),
            "return_code": result.returncode,
            "reported_verdict": preflight["verdict"],
            "errors": preflight["errors"],
            "unresolved_decisions": preflight["unresolved_decisions"],
            "config_sha256": preflight.get("config_sha256"),
            "resume_config_sha256": preflight.get("resume_config_sha256"),
            "experiment_digest": preflight.get("experimentDigest"),
        },
        "dependencies": dependencies,
        "authority": {
            "authorization_present": False,
            "long_training_started": False,
            "next_action": (
                "request one product decision bound to this exact readiness "
                "identity and managed plan"
            ),
        },
        "claim_boundary": (
            "Technical readiness for one exact managed plan only. This is not "
            "launch authority, held-out evaluation, causal evidence, playing-"
            "strength evidence, promotion, publication, or release authority."
        ),
    }
    report = {
        **report_body,
        "readiness_identity": canonical_sha256(report_body),
    }
    _write_exclusive(command_path, command_bytes)
    _write_exclusive(preflight_path, stdout)
    _write_exclusive(readiness_path, canonical_json_bytes(report))
    return report


def verify_persisted_readiness(
    *, root: Path, readiness_path: Path
) -> dict[str, Any]:
    """Recompute every persisted bundle hash and canonical identity."""
    root = root.resolve()
    readiness_path = _repository_path(
        root, readiness_path, field="readiness report"
    )
    raw = readiness_path.read_bytes()
    report = _strict_json_bytes(raw, label=str(readiness_path))
    if raw != canonical_json_bytes(report):
        raise ManagedReadinessError("readiness report is not canonical JSON")
    identity = report.get("readiness_identity")
    body = {
        key: value for key, value in report.items() if key != "readiness_identity"
    }
    if identity != canonical_sha256(body):
        raise ManagedReadinessError("readiness identity differs")
    first = report.get("first_segment")
    if not isinstance(first, Mapping):
        raise ManagedReadinessError("readiness first-segment record is missing")
    for field in ("command_artifact", "raw_preflight_artifact"):
        record = first.get(field)
        if not isinstance(record, Mapping):
            raise ManagedReadinessError(f"{field} is missing")
        path = _repository_path(root, str(record.get("path", "")), field=field)
        if _sha256_file(path) != record.get("sha256"):
            raise ManagedReadinessError(f"{field} SHA-256 differs")
    command_record = _strict_json_bytes(
        _repository_path(
            root,
            str(first["command_artifact"]["path"]),
            field="command artifact",
        ).read_bytes(),
        label="command artifact",
    )
    if (
        command_record.get("preflight_argv")
        != first.get("preflight_command")
        or command_record.get("launch_argv") != first.get("launch_command")
    ):
        raise ManagedReadinessError("persisted command arrays differ")
    return report
