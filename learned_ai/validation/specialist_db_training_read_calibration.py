"""Fail-closed contract checks for the SpecialistDB read calibration."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from learned_ai.training.run_contract import canonical_sha256
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


CONTRACT_SCHEMA = "nmm.specialist-db-training-read-calibration-plan.v1"
DEFAULT_CONTRACT = Path(
    "docs/experiments/specialist-db-training-read-calibration-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
EXPECTED_SEEDS = (61, 62, 63)
EXPECTED_CONDITIONS = ("full", "theoretical-only")


class SpecialistReadCalibrationError(RuntimeError):
    """Raised when the frozen calibration contract does not reconcile."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecialistReadCalibrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SpecialistReadCalibrationError(f"non-finite JSON number: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpecialistReadCalibrationError(
            f"could not load calibration contract: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecialistReadCalibrationError("calibration contract must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SpecialistReadCalibrationError("Git audit failed: " + " ".join(arguments))
    return result.stdout.strip()


def _ordered_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["arms"], key=lambda arm: int(arm["launch_order"]))


def _schedule_counts(seed: int, games: int, frozen_ratio: float) -> dict[str, int]:
    counts = {
        "frozen_black": 0,
        "frozen_white": 0,
        "sanmill_black": 0,
        "sanmill_white": 0,
    }
    for scheduled_index in range(games):
        _, torch_seed = trainer._derive_game_identity(
            seed,
            scheduled_index,
            "primary",
        )
        rng = random.Random(torch_seed)
        colour = "white" if rng.random() < 0.5 else "black"
        opponent = "frozen" if rng.random() < frozen_ratio else "sanmill"
        counts[f"{opponent}_{colour}"] += 1
    return counts


def load_specialist_read_calibration_contract(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the immutable six-arm design."""
    contract = _strict_json(Path(path))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise SpecialistReadCalibrationError("unsupported calibration schema")
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise SpecialistReadCalibrationError("calibration plan identity differs")
    if contract.get("status") != "designed_unlaunched_needs_publication":
        raise SpecialistReadCalibrationError("calibration status differs")
    if contract.get("authorization") != {
        "authorized_segments_per_arm": 0,
        "launch_authorized": False,
        "promotion_allowed": False,
        "publication_allowed": False,
    }:
        raise SpecialistReadCalibrationError("calibration must remain unauthorized")

    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 6:
        raise SpecialistReadCalibrationError("calibration must contain six arms")
    ordered = _ordered_arms(contract)
    expected = [
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    observed = [(arm.get("seed"), arm.get("specialist_read_mode")) for arm in ordered]
    if observed != expected:
        raise SpecialistReadCalibrationError("seed or read-mode order differs")
    if [arm.get("launch_order") for arm in ordered] != list(range(1, 7)):
        raise SpecialistReadCalibrationError("launch order differs")
    for arm in ordered:
        if arm.get("condition") != arm.get("specialist_read_mode"):
            raise SpecialistReadCalibrationError("condition and read mode differ")
    for field in (
        "arm_id",
        "control_dir",
        "experiment_id",
        "launch_order",
        "plan_id",
        "specialist_db",
    ):
        values = [arm.get(field) for arm in ordered]
        if len(values) != len(set(values)):
            raise SpecialistReadCalibrationError(f"arm field is not unique: {field}")

    pairing = contract.get("pairing")
    if not isinstance(pairing, Mapping):
        raise SpecialistReadCalibrationError("pairing contract is missing")
    if (
        pairing.get("seeds") != list(EXPECTED_SEEDS)
        or pairing.get("conditions") != list(EXPECTED_CONDITIONS)
        or pairing.get("single_changed_training_factor") != "specialist_read_mode"
        or pairing.get("same_fresh_initialization_and_schedule_within_seed") is not True
        or pairing.get("single_process_at_a_time") is not True
        or pairing.get("pair_order_is_frozen") is not True
    ):
        raise SpecialistReadCalibrationError("pairing semantics differ")
    allowlist = set(pairing.get("arm_difference_allowlist", []))
    for seed in EXPECTED_SEEDS:
        pair = [arm for arm in ordered if arm["seed"] == seed]
        left = {key: value for key, value in pair[0].items() if key not in allowlist}
        right = {key: value for key, value in pair[1].items() if key not in allowlist}
        if left != right:
            raise SpecialistReadCalibrationError(
                f"seed {seed} arms differ outside the allowlist"
            )

    common = contract.get("common_training_contract")
    resources = contract.get("resources")
    if not isinstance(common, Mapping) or not isinstance(resources, Mapping):
        raise SpecialistReadCalibrationError("training resource contract is missing")
    games = resources.get("completed_games_per_arm")
    if (
        games != 250
        or common.get("one_segment_games") != games
        or resources.get("maximum_completed_games_total") != 1500
        or resources.get("active_wall_hours_per_arm") != 0.5
        or resources.get("maximum_active_wall_hours_total") != 3.0
        or common.get("batch_games") != 1
        or common.get("start_mode") != "fresh"
        or common.get("malom_policy_aux_coefficient") != 0.0
        or common.get("mill_bonus_mode") != "malom-preserving-only"
    ):
        raise SpecialistReadCalibrationError("training resource values differ")
    if games >= common["fixed_resource_stage_games"][0]:
        raise SpecialistReadCalibrationError("calibration must stay at node level one")
    schedules = resources.get("schedule_counts_by_seed")
    if not isinstance(schedules, Mapping):
        raise SpecialistReadCalibrationError("schedule counts are missing")
    sanmill_games_per_condition = 0
    for seed in EXPECTED_SEEDS:
        counts = _schedule_counts(seed, games, common["frozen_target_ratio"])
        if schedules.get(str(seed)) != counts:
            raise SpecialistReadCalibrationError(f"seed {seed} schedule differs")
        sanmill_games_per_condition += counts["sanmill_black"] + counts["sanmill_white"]
    requested_nodes = (
        sanmill_games_per_condition
        * len(EXPECTED_CONDITIONS)
        * common["max_logical_plies"]
        * common["sanmill_node_ladder"][0]
    )
    if resources.get("maximum_requested_sanmill_nodes_total") != requested_nodes:
        raise SpecialistReadCalibrationError("Sanmill node ceiling differs")
    return contract


def build_prepare_command(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    paths_config: Path,
    python_executable: str,
) -> list[str]:
    """Build one manager prepare command without authorizing or running it."""
    common = contract["common_training_contract"]
    resources = contract["resources"]
    return [
        python_executable,
        str(root / "scripts/manage_generalist_run.py"),
        "prepare",
        "--control-dir",
        str((root / arm["control_dir"]).resolve()),
        "--max-wall-hours",
        str(resources["active_wall_hours_per_arm"]),
        "--plan-id",
        arm["plan_id"],
        "--objective",
        f"{contract['objective']}; arm={arm['arm_id']}",
        "--paths-config",
        str(paths_config),
        "--experiment-id",
        arm["experiment_id"],
        "--seed",
        str(arm["seed"]),
        "--max-games",
        str(common["max_games_schedule"]),
        "--completion-game-bound",
        str(resources["completed_games_per_arm"]),
        "--segment-games",
        str(common["one_segment_games"]),
        "--engine-profile",
        "sanmill-fixed-resource",
        "--self-play-ratio",
        str(common["frozen_target_ratio"]),
        "--sanmill-node-ladder",
        ",".join(str(value) for value in common["sanmill_node_ladder"]),
        "--sanmill-stage-games",
        ",".join(str(value) for value in common["fixed_resource_stage_games"]),
        "--max-ply",
        str(common["max_logical_plies"]),
        "--mill-bonus-mode",
        common["mill_bonus_mode"],
        "--malom-policy-aux-coef",
        str(common["malom_policy_aux_coefficient"]),
        "--malom-policy-aux-mode",
        common["malom_policy_aux_mode"],
        "--specialist-read-mode",
        arm["specialist_read_mode"],
        "--specialist-db",
        str((root / arm["specialist_db"]).resolve()),
        "--policy-health-gate",
        "--policy-health-device",
        "auto",
    ]


def build_prepare_commands(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    """Return all six frozen preparation commands in launch order."""
    return [
        build_prepare_command(
            root=root,
            contract=contract,
            arm=arm,
            paths_config=paths_config,
            python_executable=python_executable,
        )
        for arm in _ordered_arms(contract)
    ]


def _command_training_semantics(command: list[str], *, ignore_seed: bool) -> str:
    manager_args = manager._build_parser().parse_args(command[2:])
    common_args = manager._common_trainer_args(
        manager_args,
        Path(manager_args.paths_config),
    )
    trainer_args = trainer._build_argument_parser().parse_args(
        ["--preflight", "long-run", *common_args]
    )
    ignored = {"experiment_id", "specialist_db", "specialist_read_mode"}
    if ignore_seed:
        ignored.add("seed")
    semantics = {
        key: value
        for key, value in vars(trainer_args).items()
        if not key.startswith("_") and key not in ignored
    }
    return canonical_sha256(semantics)


def validate_prepare_commands(
    contract: Mapping[str, Any], commands: list[list[str]]
) -> None:
    """Prove that generated commands differ only by seed and read mode."""
    if len(commands) != 6:
        raise SpecialistReadCalibrationError("prepare command count differs")
    pair_semantics: dict[int, set[str]] = {}
    global_semantics: set[str] = set()
    for arm, command in zip(_ordered_arms(contract), commands, strict=True):
        if "authorize" in command or "run-next" in command:
            raise SpecialistReadCalibrationError("preparation command can train")
        args = manager._build_parser().parse_args(command[2:])
        if args.specialist_read_mode != arm["specialist_read_mode"]:
            raise SpecialistReadCalibrationError("command read mode differs")
        if args.seed != arm["seed"]:
            raise SpecialistReadCalibrationError("command seed differs")
        pair_semantics.setdefault(arm["seed"], set()).add(
            _command_training_semantics(command, ignore_seed=False)
        )
        global_semantics.add(_command_training_semantics(command, ignore_seed=True))
    if any(len(values) != 1 for values in pair_semantics.values()):
        raise SpecialistReadCalibrationError(
            "seed pair differs outside SpecialistDB read mode"
        )
    if len(global_semantics) != 1:
        raise SpecialistReadCalibrationError(
            "arms differ outside seed and SpecialistDB read mode"
        )


def inspect_source_readiness(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Perform a read-only source audit; never create plans or databases."""
    root = root.resolve()
    contract_path = contract_path.resolve()
    paths_config = paths_config.resolve()
    contract = load_specialist_read_calibration_contract(contract_path)
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    origin_dev = _git_output(root, "rev-parse", "origin/dev")
    origin_main = _git_output(root, "rev-parse", "origin/main")
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if branch != "dev":
        raise SpecialistReadCalibrationError("source audit requires dev")
    if origin_main != contract["lineage"]["main_review"]["reviewed_tip"]:
        raise SpecialistReadCalibrationError("origin/main moved after review")
    for commit in contract["lineage"]["required_implementation_commits"]:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise SpecialistReadCalibrationError(
                f"required implementation commit is absent: {commit}"
            )

    evidence: dict[str, Any] = {}
    for name, record in contract["source_evidence"].items():
        path = (root / record["path"]).resolve()
        observed = _sha256_file(path)
        if observed != record["sha256"]:
            raise SpecialistReadCalibrationError(f"source evidence differs: {name}")
        evidence[name] = {"path": record["path"], "sha256": observed}
    template = contract["data_contract"]["specialist_db_initial_template"]
    template_path = (root / template["path"]).resolve()
    if (
        template_path.stat().st_size != template["byte_length"]
        or _sha256_file(template_path) != template["sha256"]
    ):
        raise SpecialistReadCalibrationError("SpecialistDB template differs")
    sidecars = [
        str(Path(str(template_path) + suffix))
        for suffix in ("-wal", "-shm", "-journal")
        if Path(str(template_path) + suffix).exists()
    ]
    if sidecars:
        raise SpecialistReadCalibrationError("SpecialistDB template has sidecars")
    if not paths_config.is_file():
        raise SpecialistReadCalibrationError("local paths config is missing")

    commands = build_prepare_commands(
        root=root,
        contract=contract,
        paths_config=paths_config,
        python_executable=python_executable,
    )
    validate_prepare_commands(contract, commands)
    existing_targets = []
    for arm in _ordered_arms(contract):
        for field in ("control_dir", "specialist_db"):
            path = (root / arm[field]).resolve()
            if path.exists():
                existing_targets.append(
                    {"arm_id": arm["arm_id"], "field": field, "path": str(path)}
                )

    published = head == origin_dev
    clean = not status
    if not clean:
        state = "implementation_in_progress"
    elif not published:
        state = "implementation_complete_needs_publication"
    elif existing_targets:
        state = "published_source_needs_target_quarantine"
    else:
        state = "source_ready_for_local_preparation"
    body = {
        "schema_version": "nmm.specialist-db-training-read-source-readiness.v1",
        "state": state,
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "source": {
            "branch": branch,
            "head": head,
            "origin_dev": origin_dev,
            "origin_main": origin_main,
            "published": published,
            "tracked_and_untracked_clean": clean,
        },
        "source_evidence": evidence,
        "template": {
            "path": str(template_path),
            "sha256": template["sha256"],
            "sidecars": [],
        },
        "preparation_targets": {
            "absent": not existing_targets,
            "existing": existing_targets,
        },
        "commands": commands,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": [
            "publish source before creating plans or databases",
            "implement and freeze the result analyzer before launch authorization",
            "no training launch is authorized",
        ],
    }
    return {**body, "readiness_identity": canonical_sha256(body)}
