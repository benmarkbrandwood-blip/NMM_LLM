"""Read-only configuration and readiness checks for Generalist v2 training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from learned_ai.data.malom_label_provenance import (
    CURRENT_MALOM_LABEL_VERSION,
    read_malom_label_version,
)
from learned_ai.data.data_contract import (
    DATASET_MANIFEST_SCHEMA,
    load_dataset_manifest,
    verify_dataset_snapshot,
)
from learned_ai.training.run_contract import (
    ContractValidationError,
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.checkpoint_envelope import (
    CheckpointError,
    is_checkpoint_envelope,
    load_checkpoint,
)


PREFLIGHT_SCHEMA = "nmm.generalist-preflight.v1"

TRAINING_PATH_KEYS = frozenset(
    {
        "generalist_output_dir",
        "sentinel_checkpoint",
        "malom_db_path",
        "malom_manifest_path",
        "value_net_path",
        "gap_net_path",
        "human_db_path",
        "specialist_db_path",
        "sanmill_checkout",
    }
)

PATH_SPECS = {
    "out_dir": (
        "NMM_GENERALIST_OUT_DIR",
        "generalist_output_dir",
        "learned_ai/checkpoints/scaffolded/s_gen_v2",
    ),
    "sentinel": (
        "NMM_SENTINEL_CHECKPOINT",
        "sentinel_checkpoint",
        "learned_ai/sentinel/checkpoints/best.pt",
    ),
    "malom": ("NMM_MALOM_DB", "malom_db_path", ""),
    "malom_manifest": (
        "NMM_MALOM_MANIFEST",
        "malom_manifest_path",
        "data/manifests/malom-sector-corrected-v1.json",
    ),
    "value_net": ("NMM_VALUE_NET", "value_net_path", "data/value_net.npz"),
    "gap_net": ("NMM_GAP_NET", "gap_net_path", "data/gap_net.npz"),
    "human_db": ("NMM_HUMAN_DB", "human_db_path", "data/human_db.sqlite"),
    "specialist_db": (
        "NMM_SPECIALIST_DB",
        "specialist_db_path",
        "data/specialist_db.sqlite",
    ),
}


class PreflightConfigurationError(ValueError):
    """Raised when Generalist configuration is invalid before resource probes."""


@dataclass(frozen=True)
class LoadedTrainingSettings:
    """Merged shared/local settings with provenance for every resulting key."""

    values: Mapping[str, Any]
    sources: Mapping[str, str]
    local_config_path: Path | None


@dataclass(frozen=True)
class GitState:
    """Repository identity captured without changing Git state."""

    commit: str
    dirty: bool
    diff_sha256: str | None


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreflightConfigurationError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise PreflightConfigurationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightConfigurationError(
            f"training settings must contain a JSON object: {path}"
        )
    return value


def load_training_settings(
    root: Path, paths_config: str | None
) -> LoadedTrainingSettings:
    """Load shared settings plus a strict machine-local path overlay."""
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    shared_path = root / "data" / "settings.json"
    if shared_path.exists():
        shared = _strict_json_object(shared_path)
        values.update(shared)
        sources.update({key: "shared_config" for key in shared})

    local_path = (
        Path(paths_config).expanduser()
        if paths_config
        else root / "data" / "training_paths.local.json"
    )
    if not local_path.is_absolute():
        local_path = root / local_path
    if local_path.exists():
        local = _strict_json_object(local_path)
        unknown = sorted(set(local) - TRAINING_PATH_KEYS)
        if unknown:
            raise PreflightConfigurationError(
                "unknown training path keys: " + ", ".join(unknown)
            )
        values.update(local)
        sources.update({key: "local_path_config" for key in local})
        return LoadedTrainingSettings(values, sources, local_path)
    if paths_config:
        raise FileNotFoundError(f"training paths config not found: {local_path}")
    return LoadedTrainingSettings(values, sources, None)


def _resolve_path(root: Path, value: Any) -> str:
    if value is None or not str(value).strip():
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(value).strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = root / path
    return str(path.resolve(strict=False))


def configure_generalist_paths(
    args: Any,
    *,
    root: Path,
    settings: LoadedTrainingSettings,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply the documented path precedence and return source provenance."""
    environment = os.environ if environ is None else environ
    sources: dict[str, str] = {}
    for attr, (environment_name, setting_name, default) in PATH_SPECS.items():
        cli_value = getattr(args, attr, None)
        if cli_value is not None:
            value = cli_value
            source = "cli"
        elif environment_name in environment:
            value = environment[environment_name]
            source = f"environment:{environment_name}"
        elif setting_name in settings.values:
            value = settings.values[setting_name]
            source = f"{settings.sources[setting_name]}:{setting_name}"
        else:
            value = default
            source = "code_default"
        setattr(args, attr, _resolve_path(root, value))
        sources[attr] = source

    for attr, flag in (
        ("sentinel", "no_sentinel"),
        ("value_net", "no_value_net"),
        ("gap_net", "no_gap_net"),
    ):
        if getattr(args, flag, False):
            setattr(args, attr, "")
            sources[attr] = f"cli:{flag}"
    return sources


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightConfigurationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PreflightConfigurationError(f"{field} must be finite")
    return number


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreflightConfigurationError(f"{field} must be a positive integer")
    return value


def validate_generalist_configuration(args: Any) -> None:
    """Reject invalid and internally conflicting Generalist options."""
    for field in (
        "max_games",
        "update_every",
        "rolling_win",
        "diff_max",
        "log_every",
        "max_ply",
        "max_ply_branch",
        "update_target_every",
        "branch_every",
        "bucket_window",
        "max_per_bucket",
        "batch_games",
        "sim_ply_depth",
    ):
        _positive_integer(getattr(args, field), field=field)

    for field in ("lr", "temp_start", "s1b_refresher_lr"):
        if _finite_number(getattr(args, field), field=field) <= 0:
            raise PreflightConfigurationError(f"{field} must be greater than zero")
    gamma_td = _finite_number(args.gamma_td, field="gamma_td")
    if not 0 <= gamma_td <= 1:
        raise PreflightConfigurationError("gamma_td must be between zero and one")
    entropy_coef = _finite_number(args.entropy_coef, field="entropy_coef")
    if entropy_coef < 0:
        raise PreflightConfigurationError("entropy_coef must not be negative")
    self_play_ratio = _finite_number(args.self_play_ratio, field="self_play_ratio")
    if not 0 <= self_play_ratio <= 1:
        raise PreflightConfigurationError(
            "self_play_ratio must be between zero and one"
        )
    time_budget = _finite_number(args.time_budget, field="time_budget")
    if time_budget != -1 and time_budget <= 0:
        raise PreflightConfigurationError(
            "time_budget must be -1 or a positive number"
        )
    heuristic_node_budget = getattr(args, "heuristic_node_budget", None)
    if heuristic_node_budget is not None:
        _positive_integer(
            heuristic_node_budget,
            field="heuristic_node_budget",
        )
        if time_budget != -1:
            raise PreflightConfigurationError(
                "heuristic node and time budgets are mutually exclusive"
            )
    if args.max_branches_per_game < 0:
        raise PreflightConfigurationError(
            "max_branches_per_game must not be negative"
        )
    if args.s1b_refresher_epochs < 0:
        raise PreflightConfigurationError(
            "s1b_refresher_epochs must not be negative"
        )
    if args.diff_start is not None and not 1 <= args.diff_start <= args.diff_max:
        raise PreflightConfigurationError(
            "diff_start must be between one and diff_max"
        )
    if args.batch_games > args.max_games:
        raise PreflightConfigurationError(
            "batch_games must not exceed max_games"
        )
    if args.batch_games != 1:
        raise PreflightConfigurationError(
            "batch_games must remain 1 until shared rollout state is removed"
        )
    segment_games = getattr(args, "segment_games", None)
    if segment_games is not None:
        _positive_integer(segment_games, field="segment_games")
        if segment_games > args.max_games:
            raise PreflightConfigurationError(
                "segment_games must not exceed max_games"
            )
    if not args.policy_hidden or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        for width in args.policy_hidden
    ):
        raise PreflightConfigurationError(
            "policy_hidden must contain positive integer widths"
        )
    if args.resume and args.auto_resume_best:
        raise PreflightConfigurationError(
            "resume and auto_resume_best are mutually exclusive"
        )
    if args.auto_resume_best:
        raise PreflightConfigurationError(
            "auto_resume_best is not permitted by contract-backed launches"
        )
    if args.start_mode == "fresh" and args.resume:
        raise PreflightConfigurationError("fresh start must not provide resume")
    if args.start_mode in {"weights-only", "exact-resume"} and not args.resume:
        raise PreflightConfigurationError(
            f"{args.start_mode} requires an explicit resume checkpoint"
        )


def _read_git_state(root: Path) -> GitState:
    def run_git(*arguments: str) -> bytes:
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=root, stderr=subprocess.STDOUT
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PreflightConfigurationError(
                f"cannot inspect Git state with {' '.join(arguments)}"
            ) from exc

    commit = run_git("rev-parse", "HEAD").decode("ascii").strip()
    status = run_git("status", "--porcelain=v1")
    dirty = bool(status.strip())
    return GitState(
        commit=commit,
        dirty=dirty,
        diff_sha256=canonical_sha256(
            {
                "status": status.decode("utf-8", errors="replace"),
                "diff": run_git("diff", "--binary").decode(
                    "utf-8", errors="replace"
                ),
                "staged_diff": run_git("diff", "--cached", "--binary").decode(
                    "utf-8", errors="replace"
                ),
            }
        )
        if dirty
        else None,
    )


def _sqlite_read_only(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _probe_sqlite(path: Path) -> tuple[sqlite3.Connection | None, dict[str, Any]]:
    report: dict[str, Any] = {"exists": path.exists(), "kind": "sqlite"}
    if not path.is_file():
        report["error"] = "path is not an existing file"
        return None, report
    try:
        connection = _sqlite_read_only(path)
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        report["quick_check"] = quick_check[0] if quick_check else None
        if report["quick_check"] != "ok":
            report["error"] = "SQLite quick_check did not return ok"
            connection.close()
            return None, report
        stat = path.stat()
        report["identity"] = canonical_sha256(
            {
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "page_count": connection.execute("PRAGMA page_count").fetchone()[0],
                "page_size": connection.execute("PRAGMA page_size").fetchone()[0],
                "schema_version": connection.execute(
                    "PRAGMA schema_version"
                ).fetchone()[0],
                "user_version": connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
            }
        )
        return connection, report
    except sqlite3.Error as exc:
        report["error"] = f"SQLite read-only probe failed: {exc}"
        return None, report


def _probe_specialist_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        parent = path.parent
        return {
            "exists": False,
            "kind": "new_sqlite",
            "parent_exists": parent.is_dir(),
            "label_version": CURRENT_MALOM_LABEL_VERSION,
            "trust": "new_database_adopts_current_version",
            "identity": canonical_sha256(
                {
                    "state": "new_database",
                    "label_version": CURRENT_MALOM_LABEL_VERSION,
                }
            ),
        }
    connection, report = _probe_sqlite(path)
    if connection is None:
        return report
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"positions", "winning_lines", "preferred_plays", "meta"}
        missing = sorted(required - tables)
        if missing:
            report["error"] = "missing tables: " + ", ".join(missing)
            return report
        report["label_version"] = read_malom_label_version(connection)
        report["counts"] = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("positions", "winning_lines", "preferred_plays")
        }
        report["malom_label_count"] = int(
            connection.execute(
                "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
            ).fetchone()[0]
        )
        lineage = connection.execute(
            "SELECT value FROM meta WHERE key='training_lineage_root_run_id'"
        ).fetchone()
        report["training_lineage_root_run_id"] = (
            lineage[0] if lineage and lineage[0] else None
        )
        if report["label_version"] != CURRENT_MALOM_LABEL_VERSION:
            report["error"] = (
                "SpecialistDB does not declare the trusted Malom label version"
            )
        else:
            report["trust"] = "trusted"
        report["content_sha256"] = _file_sha256(path)
        return report
    except sqlite3.Error as exc:
        report["error"] = f"SpecialistDB schema probe failed: {exc}"
        return report
    finally:
        connection.close()


def _probe_human_db(path: Path) -> dict[str, Any]:
    connection, report = _probe_sqlite(path)
    if connection is None:
        return report
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = sorted({"positions", "moves"} - tables)
        if missing:
            report["error"] = "missing tables: " + ", ".join(missing)
            return report
        report["label_version"] = read_malom_label_version(connection)
        report["malom_columns_policy"] = (
            "trusted"
            if report["label_version"] == CURRENT_MALOM_LABEL_VERSION
            else "masked_historical_labels"
        )
        report["trust"] = "empirical_frequencies_and_outcomes"
        return report
    except sqlite3.Error as exc:
        report["error"] = f"HumanDB schema probe failed: {exc}"
        return report
    finally:
        connection.close()


def _probe_malom(path: Path, manifest_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "exists": path.exists(),
        "kind": "malom_directory",
    }
    if not path.is_dir():
        report["error"] = "Malom path is not an existing directory"
        return report
    try:
        manifest = load_dataset_manifest(manifest_path)
        if manifest.logical_name != "malom_tablebase":
            raise ContractValidationError("manifest is not for the Malom tablebase")
        if manifest.trust_level != CURRENT_MALOM_LABEL_VERSION:
            raise ContractValidationError("Malom manifest trust level is incompatible")
        structural = verify_dataset_snapshot(path, manifest)
        anchor = next(
            (
                component
                for component in manifest.components
                if component.relative_path == "std.secval"
            ),
            None,
        )
        if anchor is None or _file_sha256(path / "std.secval") != anchor.sha256:
            raise ContractValidationError("Malom std.secval anchor hash has changed")
        from ai.malom_db import MalomDB

        database = MalomDB(path)
        report["secval_exists"] = (path / "std.secval").is_file()
        report["available"] = database.is_available()
        if not database.is_available():
            report["error"] = "MalomDB did not find a usable secval/sector set"
        else:
            report["manifest_schema"] = DATASET_MANIFEST_SCHEMA
            report["manifest_path"] = str(manifest_path)
            report["component_count"] = structural["component_count"]
            report["size_bytes"] = structural["size_bytes"]
            report["identity"] = manifest.manifest_sha256
    except Exception as exc:
        report["error"] = f"Malom read-only probe failed: {exc}"
    return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_RESUME_CONFIG_EXCLUDED_FIELDS = {
    "auto_resume_best",
    "experiment_id",
    "launch",
    "out_dir",
    "parent_run_id",
    "paths_config",
    "preflight",
    "resume",
    "run_id",
    "segment_games",
    "start_mode",
}


def resolved_resume_config(args: Any) -> dict[str, Any]:
    """Return only settings that affect the continuation trajectory."""
    raw = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key not in _RESUME_CONFIG_EXCLUDED_FIELDS
    }
    return json.loads(canonical_json_bytes(raw))


def resume_config_sha256(args: Any) -> str:
    """Hash training semantics independently of run-segment invocation fields."""
    return canonical_sha256(resolved_resume_config(args))


def _probe_source_checkpoint(
    path: Path,
    args: Any,
    *,
    feature_schema_version: str,
    expected_move_feature_dim: int,
    expected_value_input_dim: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"exists": path.exists(), "kind": "checkpoint"}
    if not path.is_file():
        report["error"] = "resume checkpoint is not an existing file"
        return report
    try:
        if is_checkpoint_envelope(path):
            envelope = load_checkpoint(path)
            descriptor = envelope.descriptor
            config = envelope.payload.trainer_state["model_config"]
            report.update(
                {
                    "format": "checkpoint-envelope-v2",
                    "identity": canonical_sha256(
                        {
                            "checkpoint_id": descriptor.checkpoint_id,
                            "payload_sha256": envelope.payload_sha256,
                        }
                    ),
                    "checkpoint_id": descriptor.checkpoint_id,
                    "source_run_id": descriptor.run_id,
                    "resume_config_sha256": descriptor.config_sha256,
                    "feature_schema_version": descriptor.feature_schema_version,
                    "payload_size": envelope.payload_size,
                    "model_config": dict(config),
                    "mutable_assets": dict(
                        envelope.payload.data_state["mutable_assets"]
                    ),
                }
            )
            if descriptor.feature_schema_version != feature_schema_version:
                report["error"] = "checkpoint feature schema is incompatible"
            expected = {
                "policy_hidden": tuple(args.policy_hidden),
                "move_feat_dim": expected_move_feature_dim,
                "value_input_dim": expected_value_input_dim,
            }
            observed = {
                "policy_hidden": tuple(config.get("policy_hidden", ())),
                "move_feat_dim": config.get("move_feat_dim"),
                "value_input_dim": config.get("value_input_dim"),
            }
            if observed != expected:
                report["error"] = "checkpoint model_config is incompatible"
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, Mapping):
                raise ValueError("legacy checkpoint is not a mapping")
            config = checkpoint.get("model_config")
            if not isinstance(config, Mapping):
                report["error"] = "legacy checkpoint has no explicit model_config"
                return report
            expected = {
                "policy_hidden": tuple(args.policy_hidden),
                "move_feat_dim": expected_move_feature_dim,
                "value_input_dim": expected_value_input_dim,
            }
            observed = {
                "policy_hidden": tuple(config.get("policy_hidden", ())),
                "move_feat_dim": config.get("move_feat_dim"),
                "value_input_dim": config.get("value_input_dim"),
            }
            identity = _file_sha256(path)
            report.update(
                {
                    "format": "legacy-pytorch-weights",
                    "identity": identity,
                    "checkpoint_id": f"legacy-sha256:{identity}",
                    "source_run_id": None,
                    "model_config": dict(config),
                }
            )
            if observed != expected:
                report["error"] = (
                    "legacy checkpoint model_config is incompatible with the request"
                )
    except (OSError, ValueError, RuntimeError, CheckpointError) as exc:
        report["error"] = f"checkpoint probe failed: {exc}"
    return report


def _resolved_config(args: Any, mode: str) -> dict[str, Any]:
    raw = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key != "preflight"
    }
    raw["preflight_mode"] = mode
    return json.loads(canonical_json_bytes(raw))


def run_generalist_preflight(
    args: Any,
    *,
    mode: str,
    root: Path,
    path_sources: Mapping[str, str],
    feature_schema_version: str = "unspecified",
    expected_move_feature_dim: int = -1,
    expected_value_input_dim: int = -1,
    git_state: GitState | None = None,
) -> dict[str, Any]:
    """Return a complete read-only readiness report for the corrected baseline."""
    if mode not in {"smoke", "long-run"}:
        raise PreflightConfigurationError(f"unsupported preflight mode: {mode}")
    validate_generalist_configuration(args)
    state = _read_git_state(root) if git_state is None else git_state
    errors: list[str] = []
    decisions: list[str] = []

    if state.dirty:
        errors.append("Git worktree must be clean")
    if (
        args.start_mode != "fresh"
        and args.experiment_id == "dev-v4-malom-corrected-fresh-v1"
    ):
        errors.append(
            "non-fresh imports require an explicit non-fresh experiment ID"
        )
    for flag in (
        "no_sentinel",
        "no_value_net",
        "no_gap_net",
        "no_s1a_warmstart",
        "no_imitation_mix",
    ):
        if not getattr(args, flag):
            errors.append(f"corrected fresh baseline requires explicit --{flag.replace('_', '-')}")
    if args.ppo:
        errors.append("corrected fresh baseline must not enable PPO")
    if mode == "smoke":
        bounded_games = getattr(args, "segment_games", None) or args.max_games
        if bounded_games not in {1, 2}:
            errors.append("smoke preflight requires a one- or two-game segment")
        if args.batch_games != 1:
            errors.append("smoke preflight requires batch_games=1")
    else:
        decisions.append(
            "long-run update, opponent, budget, cadence, and stop choices are not "
            "yet frozen in the experiment contract"
        )
    output = Path(args.out_dir)
    output_report = {
        "exists": output.exists(),
        "kind": "run_directory",
        "isolated": not output.exists(),
    }
    if output.exists():
        output_report["error"] = "fresh output path already exists"
        errors.append("fresh output path must not already exist")

    malom_report = _probe_malom(
        Path(args.malom), Path(args.malom_manifest)
    ) if args.malom else {
        "exists": False,
        "error": "Malom path is not configured",
    }
    specialist_report = _probe_specialist_db(Path(args.specialist_db))
    human_report = _probe_human_db(Path(args.human_db))
    checkpoint_report: dict[str, Any] | None = None
    expected_resume_config_sha256 = resume_config_sha256(args)
    if args.start_mode in {"weights-only", "exact-resume"}:
        checkpoint_report = _probe_source_checkpoint(
            Path(args.resume),
            args,
            feature_schema_version=feature_schema_version,
            expected_move_feature_dim=expected_move_feature_dim,
            expected_value_input_dim=expected_value_input_dim,
        )
    for name, report in (
        ("malom", malom_report),
        ("specialist_db", specialist_report),
        ("human_db", human_report),
    ):
        if report.get("error"):
            errors.append(f"{name}: {report['error']}")
    if checkpoint_report is not None and checkpoint_report.get("error"):
        errors.append(f"checkpoint: {checkpoint_report['error']}")
    if args.start_mode == "exact-resume" and checkpoint_report is not None:
        if checkpoint_report.get("format") != "checkpoint-envelope-v2":
            errors.append("exact resume requires a CheckpointEnvelope v2 source")
        elif checkpoint_report.get("resume_config_sha256") != (
            expected_resume_config_sha256
        ):
            errors.append("checkpoint: resume configuration is incompatible")
        checkpoint_specialist = checkpoint_report.get("mutable_assets", {}).get(
            "specialist_db", {}
        )
        if checkpoint_specialist.get("sha256") != specialist_report.get(
            "content_sha256"
        ):
            errors.append("checkpoint: SpecialistDB content identity has changed")
    if args.start_mode in {"fresh", "weights-only"} and specialist_report.get(
        "exists"
    ):
        counts = specialist_report.get("counts", {})
        if any(
            counts.get(name, 0)
            for name in ("positions", "winning_lines", "preferred_plays")
        ):
            errors.append(
                "specialist_db: non-resume training requires an empty isolated database"
            )
        if specialist_report.get("training_lineage_root_run_id") is not None:
            errors.append(
                "specialist_db: database is already bound to another training lineage"
            )
    if not specialist_report.get("exists") and not specialist_report.get(
        "parent_exists"
    ):
        errors.append("specialist_db: parent directory does not exist")

    config = _resolved_config(args, mode)
    if errors:
        verdict = "fatal_stop"
    elif decisions:
        verdict = "needs_decision"
    else:
        verdict = "ready_for_smoke"
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "mode": mode,
        "verdict": verdict,
        "git": {
            "commit": state.commit,
            "dirty": state.dirty,
            "diff_sha256": state.diff_sha256,
        },
        "resolved_config": config,
        "config_sha256": canonical_sha256(config),
        "resume_config_sha256": expected_resume_config_sha256,
        "path_sources": dict(path_sources),
        "checks": {
            "output": output_report,
            "malom": malom_report,
            "specialist_db": specialist_report,
            "human_db": human_report,
            "checkpoint": checkpoint_report,
            "components": {
                "sentinel": not args.no_sentinel,
                "value_net": not args.no_value_net,
                "gap_net": not args.no_gap_net,
                "ppo": bool(args.ppo),
                "imitation_warmstart": not args.no_s1a_warmstart,
                "imitation_mix": not args.no_imitation_mix,
            },
        },
        "errors": errors,
        "unresolved_decisions": decisions,
    }
