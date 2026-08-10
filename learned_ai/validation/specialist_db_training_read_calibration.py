"""Fail-closed contract checks for the SpecialistDB read calibration."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from learned_ai.training.managed_generalist import ManagedPlan, load_managed_plan
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.mill_bonus_ablation_readiness import (
    PRODUCT_AUTHORIZATION_DECISION,
    _build_fresh_preflight_command,
    _inspect_specialist_database,
    _repository_path,
    _run_checked,
    _validate_unlaunched_preflight,
    assert_preparation_outputs_ignored,
    assert_preparation_targets_absent,
    inspect_runtime_identities,
    inspect_template,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


CONTRACT_SCHEMA = "nmm.specialist-db-training-read-calibration-plan.v1"
SOURCE_READINESS_SCHEMA = "nmm.specialist-db-training-read-source-readiness.v1"
READINESS_SCHEMA = "nmm.specialist-db-training-read-calibration-readiness.v1"
RESULT_SCHEMA = "nmm.specialist-db-training-read-calibration-result.v1"
DEFAULT_CONTRACT = Path(
    "docs/experiments/specialist-db-training-read-calibration-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_SOURCE_REPORT = Path(
    "out/specialist-db-training-read-calibration-v1/source-readiness.json"
)
DEFAULT_REPORT = Path("out/specialist-db-training-read-calibration-v1/readiness.json")
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

    analysis = contract.get("analysis")
    implementation = (
        analysis.get("result_implementation") if isinstance(analysis, Mapping) else None
    )
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != {"module", "publisher", "result_schema"}
        or implementation.get("result_schema") != RESULT_SCHEMA
    ):
        raise SpecialistReadCalibrationError("result implementation contract differs")
    for name in ("module", "publisher"):
        record = implementation.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise SpecialistReadCalibrationError(
                f"result implementation identity differs: {name}"
            )
    gate = analysis.get("policy_health_gate")
    required_gate_fields = {
        "audit_script",
        "audit_script_sha256",
        "corpus",
        "corpus_sha256",
        "critical_states",
        "device",
        "minimum_candidate_preserving_minus_downgrading_logit_margin",
        "minimum_candidate_preserving_rate",
        "required_direct_signal_preserving_rate",
        "role",
    }
    if not isinstance(gate, Mapping) or set(gate) != required_gate_fields:
        raise SpecialistReadCalibrationError("policy-health contract differs")

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


def _tracked_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _inspect_source(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
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
    if status:
        raise SpecialistReadCalibrationError(
            "source audit requires a clean tracked and untracked worktree"
        )
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
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "origin_main": origin_main,
        "published": head == origin_dev,
        "tracked_and_untracked_clean": True,
    }


def _inspect_source_evidence(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name, record in contract["source_evidence"].items():
        path = _repository_path(root, record["path"], field="source evidence")
        observed = _sha256_file(path)
        if observed != record["sha256"]:
            raise SpecialistReadCalibrationError(f"source evidence differs: {name}")
        evidence[name] = {"path": record["path"], "sha256": observed}
        if "evidence_identity" in record:
            evidence[name]["evidence_identity"] = record["evidence_identity"]
    return evidence


def inspect_result_implementation(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the pre-result analyzer and publisher bytes are frozen."""
    implementation = contract["analysis"]["result_implementation"]
    result = {"result_schema": implementation["result_schema"]}
    for name in ("module", "publisher"):
        expected = implementation[name]
        path = _repository_path(root, expected["path"], field=f"result {name}")
        if not path.is_file() or not _tracked_file(root, path):
            raise SpecialistReadCalibrationError(f"result {name} is not tracked")
        observed = _sha256_file(path)
        if observed != expected["sha256"]:
            raise SpecialistReadCalibrationError(f"result {name} SHA-256 differs")
        result[name] = {"path": expected["path"], "sha256": observed}
    return result


def inspect_preparation_targets(
    root: Path,
    contract: Mapping[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    targets: list[tuple[str, Path]] = [("readiness_report", report_path)]
    for arm in _ordered_arms(contract):
        targets.extend(
            (
                (
                    f"{arm['arm_id']}:control_dir",
                    _repository_path(root, arm["control_dir"], field="control_dir"),
                ),
                (
                    f"{arm['arm_id']}:specialist_db",
                    _repository_path(root, arm["specialist_db"], field="specialist_db"),
                ),
            )
        )
    existing = [
        {
            "label": label,
            "path": path.relative_to(root.resolve()).as_posix(),
            "kind": "directory" if path.is_dir() else "file",
        }
        for label, path in targets
        if path.exists()
    ]
    return {"absent": not existing, "existing": existing}


def _objective(contract: Mapping[str, Any], arm: Mapping[str, Any]) -> str:
    return f"{contract['objective']}; arm={arm['arm_id']}"


def _trainer_args(plan: ManagedPlan) -> Any:
    return trainer._build_argument_parser().parse_args(
        ["--preflight", "long-run", *plan.common_trainer_args]
    )


def _plan_training_semantics(args: Any, *, ignore_seed: bool) -> str:
    ignored = {"experiment_id", "specialist_db", "specialist_read_mode"}
    if ignore_seed:
        ignored.add("seed")
    return canonical_sha256(
        {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_") and key not in ignored
        }
    )


def _assert_plan_semantics(
    plan: ManagedPlan,
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> Any:
    common = contract["common_training_contract"]
    resources = contract["resources"]
    expected = {
        "plan_id": arm["plan_id"],
        "objective": _objective(contract, arm),
        "experiment_id": arm["experiment_id"],
        "git_commit": source_commit,
        "control_dir": str(
            _repository_path(root, arm["control_dir"], field="control_dir")
        ),
        "paths_config": str(paths_config),
        "max_games": common["max_games_schedule"],
        "completion_game_bound": resources["completed_games_per_arm"],
        "segment_games": common["one_segment_games"],
        "max_wall_hours": resources["active_wall_hours_per_arm"],
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            raise SpecialistReadCalibrationError(
                f"managed plan differs for {arm['arm_id']}: {field}"
            )
    if plan.game_bound != resources["completed_games_per_arm"]:
        raise SpecialistReadCalibrationError("managed completion bound differs")
    if plan.paths_config_sha256 != _sha256_file(paths_config):
        raise SpecialistReadCalibrationError("managed path registry hash differs")
    args = _trainer_args(plan)
    expected_args = {
        "experiment_id": arm["experiment_id"],
        "seed": arm["seed"],
        "specialist_read_mode": arm["specialist_read_mode"],
        "mill_bonus_mode": common["mill_bonus_mode"],
        "malom_policy_aux_coef": common["malom_policy_aux_coefficient"],
        "malom_policy_aux_mode": common["malom_policy_aux_mode"],
        "max_games": common["max_games_schedule"],
        "max_ply": common["max_logical_plies"],
        "max_ply_branch": common["max_logical_plies"],
        "max_branches_per_game": 0,
        "batch_games": common["batch_games"],
        "log_every": common["log_and_checkpoint_every_games"],
        "update_target_every": common["target_refresh_every_games"],
        "sim_ply_depth": common["sim_ply_depth"],
        "self_play_ratio": common["frozen_target_ratio"],
        "temp_start": common["temperature_start"],
        "lr": common["learning_rate"],
        "gamma_td": common["gamma_td"],
        "entropy_coef": common["entropy_coefficient"],
        "update_every": common["update_every_steps"],
        "curriculum_advance_policy": common["curriculum_advance_policy"],
        "referee_engine": "sanmill",
        "opponent_engine": "sanmill",
        "minimal_rollouts": True,
        "no_recovery": True,
        "no_sentinel": True,
        "no_value_net": True,
        "no_gap_net": True,
        "no_s1a_warmstart": True,
        "no_imitation_mix": True,
        "no_s1b_refresher": True,
        "no_opening_forcing": True,
        "ppo": False,
    }
    for field, value in expected_args.items():
        if getattr(args, field) != value:
            raise SpecialistReadCalibrationError(
                f"trainer argument differs for {arm['arm_id']}: {field}"
            )
    if list(args.sanmill_node_ladder) != common["sanmill_node_ladder"]:
        raise SpecialistReadCalibrationError("Sanmill node ladder differs")
    if list(args.sanmill_stage_games) != common["fixed_resource_stage_games"]:
        raise SpecialistReadCalibrationError("Sanmill stage durations differ")
    specialist_db = _repository_path(root, arm["specialist_db"], field="specialist_db")
    if Path(args.specialist_db).resolve() != specialist_db:
        raise SpecialistReadCalibrationError("arm SpecialistDB path differs")
    diagnostic = contract["analysis"]["policy_health_gate"]
    gate = plan.policy_health
    if gate is None or gate.device != diagnostic["device"]:
        raise SpecialistReadCalibrationError("policy-health device differs")
    if (
        Path(gate.corpus_path).resolve()
        != _repository_path(root, diagnostic["corpus"], field="policy corpus")
        or gate.corpus_sha256 != diagnostic["corpus_sha256"]
        or Path(gate.audit_script_path).resolve()
        != _repository_path(root, diagnostic["audit_script"], field="policy audit")
        or gate.audit_script_sha256 != diagnostic["audit_script_sha256"]
        or gate.exact_critical_states != diagnostic["critical_states"]
        or gate.required_direct_preserving_rate
        != diagnostic["required_direct_signal_preserving_rate"]
        or gate.min_candidate_preserving_rate
        != diagnostic["minimum_candidate_preserving_rate"]
        or gate.min_candidate_logit_margin
        != diagnostic["minimum_candidate_preserving_minus_downgrading_logit_margin"]
    ):
        raise SpecialistReadCalibrationError("policy-health contract differs")
    return args


def audit_prepared_plans(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
    preflight_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    template = contract["data_contract"]["specialist_db_initial_template"]
    audited: list[dict[str, Any]] = []
    pair_semantics: dict[int, set[str]] = {}
    global_semantics: set[str] = set()
    for arm in _ordered_arms(contract):
        control_dir = _repository_path(root, arm["control_dir"], field="control_dir")
        plan_path = control_dir / "plan.json"
        plan = load_managed_plan(plan_path)
        args = _assert_plan_semantics(
            plan,
            root=root,
            contract=contract,
            arm=arm,
            paths_config=paths_config,
            source_commit=source_commit,
        )
        specialist_db = _repository_path(
            root, arm["specialist_db"], field="specialist_db"
        )
        database = _inspect_specialist_database(specialist_db, template)
        if (control_dir / "authorization.json").exists():
            raise SpecialistReadCalibrationError(
                f"arm is already authorized: {arm['arm_id']}"
            )
        if (control_dir / "segments").exists():
            raise SpecialistReadCalibrationError(
                f"arm already has segment output: {arm['arm_id']}"
            )
        pair_semantics.setdefault(int(arm["seed"]), set()).add(
            _plan_training_semantics(args, ignore_seed=False)
        )
        global_semantics.add(_plan_training_semantics(args, ignore_seed=True))
        record = {
            "arm_id": arm["arm_id"],
            "condition": arm["condition"],
            "launch_order": arm["launch_order"],
            "seed": arm["seed"],
            "specialist_read_mode": arm["specialist_read_mode"],
            "plan_path": str(plan_path),
            "plan_sha256": plan.plan_sha256,
            "resume_config_sha256": plan.resume_config_sha256,
            "specialist_db": database,
            "authorization_present": False,
            "segment_output_present": False,
        }
        if preflight_records is not None:
            record["preflight"] = dict(preflight_records[arm["arm_id"]])
        audited.append(record)
    if any(len(values) != 1 for values in pair_semantics.values()):
        raise SpecialistReadCalibrationError(
            "a seed pair differs outside SpecialistDB read mode"
        )
    if len(global_semantics) != 1:
        raise SpecialistReadCalibrationError(
            "arms differ outside seed and SpecialistDB read mode"
        )
    return audited


def inspect_source_readiness(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Perform a read-only source audit; never create plans or databases."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_specialist_read_calibration_contract(contract_path)
    source = _inspect_source(root, contract)
    evidence = _inspect_source_evidence(root, contract)
    template = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    result_analysis = inspect_result_implementation(root, contract)
    targets = inspect_preparation_targets(
        root, contract, report_path=root / DEFAULT_REPORT
    )
    commands = build_prepare_commands(
        root=root,
        contract=contract,
        paths_config=paths_config,
        python_executable=python_executable,
    )
    validate_prepare_commands(contract, commands)
    if not source["published"]:
        state = "implementation_complete_needs_publication"
    elif not targets["absent"]:
        state = "published_source_needs_target_quarantine"
    else:
        state = "source_ready_for_local_preparation"
    unresolved: list[str] = []
    if not source["published"]:
        unresolved.append("publish the frozen analyzer source to origin/dev")
    if not targets["absent"]:
        unresolved.append("quarantine pre-existing preparation targets")
    if not unresolved:
        unresolved.append(
            "generate six immutable plans and preflights; training remains unauthorized"
        )
    body = {
        "schema_version": SOURCE_READINESS_SCHEMA,
        "state": state,
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "source": source,
        "source_evidence": evidence,
        "template": template,
        "runtime": runtime,
        "result_analysis": result_analysis,
        "preparation_targets": targets,
        "commands": commands,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": unresolved,
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def prepare_specialist_read_calibration(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create six plans and preflights without authorization or training."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    report_path = report_path.resolve(strict=False)
    contract = load_specialist_read_calibration_contract(contract_path)
    source = _inspect_source(root, contract)
    if not source["published"]:
        raise SpecialistReadCalibrationError("dev must equal origin/dev")
    evidence = _inspect_source_evidence(root, contract)
    template_record = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    result_analysis = inspect_result_implementation(root, contract)
    assert_preparation_outputs_ignored(root, contract, report_path=report_path)
    assert_preparation_targets_absent(root, contract, report_path=report_path)
    commands = build_prepare_commands(
        root=root,
        contract=contract,
        paths_config=paths_config,
        python_executable=python_executable,
    )
    validate_prepare_commands(contract, commands)
    template_path = Path(template_record["path"])
    preflights: dict[str, dict[str, Any]] = {}
    for arm, command in zip(_ordered_arms(contract), commands, strict=True):
        specialist_db = _repository_path(
            root, arm["specialist_db"], field="specialist_db"
        )
        specialist_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template_path, specialist_db)
        _inspect_specialist_database(
            specialist_db,
            contract["data_contract"]["specialist_db_initial_template"],
        )
        manager_result = _run_checked(command, root=root, runner=runner)
        try:
            manager_output = json.loads(manager_result.stdout)
        except json.JSONDecodeError as exc:
            raise SpecialistReadCalibrationError(
                f"manager output is not JSON: {arm['arm_id']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise SpecialistReadCalibrationError(
                f"manager state differs: {arm['arm_id']}"
            )
        plan_path = (
            _repository_path(root, arm["control_dir"], field="control_dir")
            / "plan.json"
        )
        plan = load_managed_plan(plan_path)
        preflight_command = _build_fresh_preflight_command(
            plan, root=root, python_executable=python_executable
        )
        preflight_result = _run_checked(
            preflight_command,
            root=root,
            runner=runner,
            accepted_return_codes=(2,),
        )
        try:
            preflight = json.loads(preflight_result.stdout)
        except json.JSONDecodeError as exc:
            raise SpecialistReadCalibrationError(
                f"preflight output is not JSON: {arm['arm_id']}"
            ) from exc
        _validate_unlaunched_preflight(
            preflight,
            plan=plan,
            source_commit=source["head"],
            arm_id=arm["arm_id"],
        )
        preflight_path = plan_path.parent / "preflight.json"
        try:
            with preflight_path.open("xb") as handle:
                handle.write(canonical_json_bytes(preflight))
        except FileExistsError as exc:
            raise SpecialistReadCalibrationError(
                f"preflight evidence already exists: {arm['arm_id']}"
            ) from exc
        preflights[arm["arm_id"]] = {
            "path": str(preflight_path),
            "sha256": _sha256_file(preflight_path),
            "verdict": preflight["verdict"],
            "unresolved_decisions": preflight["unresolved_decisions"],
            "resume_config_sha256": preflight["resume_config_sha256"],
        }
    audited = audit_prepared_plans(
        root=root,
        contract=contract,
        paths_config=paths_config,
        source_commit=source["head"],
        preflight_records=preflights,
    )
    body = {
        "schema_version": READINESS_SCHEMA,
        "state": "ready_for_product_authorization",
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "source": source,
        "source_evidence": evidence,
        "template": template_record,
        "runtime": runtime,
        "result_analysis": result_analysis,
        "commands": commands,
        "arms": audited,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": [PRODUCT_AUTHORIZATION_DECISION],
    }
    report = {**body, "readiness_identity": canonical_sha256(body)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise SpecialistReadCalibrationError(
            f"readiness report already exists: {report_path}"
        ) from exc
    return report


def publish_source_readiness(path: Path, report: Mapping[str, Any]) -> None:
    """Persist one ignored source-only readiness report without overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise SpecialistReadCalibrationError(
            f"source readiness report already exists: {path}"
        ) from exc
