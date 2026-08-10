"""Prepare the fixed-anchor, optimizer-matched target-refresh diagnostic."""

from __future__ import annotations

import hashlib
import json
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


CONTRACT_SCHEMA = "nmm.target-refresh-common-anchor-diagnostic-plan.v1"
SOURCE_READINESS_SCHEMA = (
    "nmm.target-refresh-common-anchor-diagnostic-source-readiness.v1"
)
READINESS_SCHEMA = "nmm.target-refresh-common-anchor-diagnostic-readiness.v1"
RESULT_SCHEMA = "nmm.target-refresh-common-anchor-diagnostic-result.v1"
DEFAULT_CONTRACT = Path(
    "docs/experiments/sanmill-target-refresh-common-anchor-diagnostic-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_SOURCE_REPORT = Path(
    "out/target-refresh-common-anchor-diagnostic-v1/source-readiness.json"
)
DEFAULT_REPORT = Path(
    "out/target-refresh-common-anchor-diagnostic-v1/readiness.json"
)
EXPECTED_SEEDS = (64, 65)
EXPECTED_CONDITIONS = ("refresh", "no-refresh")
TARGET_REFRESH_BY_CONDITION = {"refresh": 50, "no-refresh": 5001}


class TargetRefreshCommonAnchorError(RuntimeError):
    """Raised when the frozen common-anchor diagnostic does not reconcile."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TargetRefreshCommonAnchorError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                TargetRefreshCommonAnchorError(
                    f"non-finite JSON number: {value}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetRefreshCommonAnchorError(
            f"could not load common-anchor contract: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise TargetRefreshCommonAnchorError(
            "common-anchor contract must be an object"
        )
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
        raise TargetRefreshCommonAnchorError(
            "Git audit failed: " + " ".join(arguments)
        )
    return result.stdout.strip()


def _tracked_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _ordered_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["arms"], key=lambda arm: int(arm["launch_order"]))


def load_target_refresh_common_anchor_contract(
    path: str | Path,
) -> dict[str, Any]:
    """Load and fully validate the immutable four-arm design."""
    contract = _strict_json(Path(path))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise TargetRefreshCommonAnchorError("unsupported common-anchor schema")
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise TargetRefreshCommonAnchorError("common-anchor plan identity differs")
    if contract.get("status") != "designed_unlaunched_needs_publication":
        raise TargetRefreshCommonAnchorError("common-anchor status differs")
    if contract.get("authorization") != {
        "authorized_segments_per_arm": 0,
        "launch_authorized": False,
        "promotion_allowed": False,
        "publication_allowed": False,
    }:
        raise TargetRefreshCommonAnchorError(
            "common-anchor diagnostic must remain unauthorized"
        )

    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise TargetRefreshCommonAnchorError(
            "common-anchor diagnostic must contain four arms"
        )
    ordered = _ordered_arms(contract)
    expected = [
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    observed = [(arm.get("seed"), arm.get("condition")) for arm in ordered]
    if observed != expected:
        raise TargetRefreshCommonAnchorError("seed or condition order differs")
    if [arm.get("launch_order") for arm in ordered] != list(range(1, 5)):
        raise TargetRefreshCommonAnchorError("launch order differs")
    measurement = contract.get("measurement_contract")
    if not isinstance(measurement, Mapping):
        raise TargetRefreshCommonAnchorError("measurement contract is missing")
    anchor_updates = measurement.get("anchor_update_count_by_seed")
    final_updates = measurement.get("optimizer_update_bound_by_seed")
    expected_anchor = {"64": 18, "65": 16}
    expected_final = {"64": 34, "65": 32}
    if anchor_updates != expected_anchor or final_updates != expected_final:
        raise TargetRefreshCommonAnchorError("optimizer exposure contract differs")
    if (
        measurement.get("anchor_game") != 50
        or measurement.get("post_anchor_optimizer_updates") != 16
        or measurement.get("measurement_every_updates") != 4
        or measurement.get("games_per_opponent_per_checkpoint") != 8
        or measurement.get("measurement_temperature") != 0.2
        or measurement.get("sanmill_node_budget") != 1000
        or measurement.get("opponents")
        != ["fixed_model_anchor", "sanmill_fixed_node"]
        or measurement.get("specialist_read_mode") != "disabled"
        or measurement.get("writes_training_data") is not False
    ):
        raise TargetRefreshCommonAnchorError("measurement semantics differ")
    for arm in ordered:
        seed_key = str(arm["seed"])
        if (
            arm.get("target_refresh_every_games")
            != TARGET_REFRESH_BY_CONDITION[arm["condition"]]
            or arm.get("anchor_expected_update_count")
            != anchor_updates[seed_key]
            or arm.get("optimizer_update_bound") != final_updates[seed_key]
            or arm.get("lr_adaptation_mode") != "fixed"
            or arm.get("specialist_read_mode") != "theoretical-only"
        ):
            raise TargetRefreshCommonAnchorError(
                f"arm factor mapping differs: {arm['arm_id']}"
            )
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
            raise TargetRefreshCommonAnchorError(
                f"arm field is not unique: {field}"
            )

    pairing = contract.get("pairing")
    if not isinstance(pairing, Mapping):
        raise TargetRefreshCommonAnchorError("pairing contract is missing")
    if (
        pairing.get("seeds") != list(EXPECTED_SEEDS)
        or pairing.get("conditions") != list(EXPECTED_CONDITIONS)
        or pairing.get("changed_training_factors")
        != ["target_refresh_every_games"]
        or pairing.get("same_first_50_games_within_seed") is not True
        or pairing.get("equal_post_anchor_optimizer_updates") != 16
        or pairing.get("single_process_at_a_time") is not True
        or pairing.get("launch_order_is_frozen") is not True
    ):
        raise TargetRefreshCommonAnchorError("pairing semantics differ")
    allowlist = set(pairing.get("arm_difference_allowlist", []))
    for seed in EXPECTED_SEEDS:
        seed_arms = [arm for arm in ordered if arm["seed"] == seed]
        bases = [
            {key: value for key, value in arm.items() if key not in allowlist}
            for arm in seed_arms
        ]
        if bases[0] != bases[1]:
            raise TargetRefreshCommonAnchorError(
                f"seed {seed} arms differ outside the factor allowlist"
            )

    common = contract.get("common_training_contract")
    resources = contract.get("resources")
    if not isinstance(common, Mapping) or not isinstance(resources, Mapping):
        raise TargetRefreshCommonAnchorError("training resource contract is missing")
    if (
        common.get("algorithm") != "A2C"
        or common.get("start_mode") != "fresh"
        or common.get("one_segment_game_safety_ceiling") != 150
        or common.get("max_games_schedule") != 5000
        or common.get("batch_games") != 1
        or common.get("minimal_rollouts") is not True
        or common.get("branches") is not False
        or common.get("learning_rate_mode") != "fixed"
        or common.get("specialist_read_mode") != "theoretical-only"
        or resources.get("maximum_training_games_per_arm") != 150
        or resources.get("maximum_training_games_total") != 600
        or resources.get("maximum_measurement_games_per_arm") != 64
        or resources.get("maximum_measurement_games_total") != 256
        or resources.get("active_wall_hours_per_arm") != 0.5
        or resources.get("maximum_active_wall_hours_total") != 2.0
    ):
        raise TargetRefreshCommonAnchorError("training resource values differ")
    implementation = contract.get("analysis", {}).get("result_implementation")
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != {"module", "publisher", "result_schema"}
        or implementation.get("result_schema") != RESULT_SCHEMA
    ):
        raise TargetRefreshCommonAnchorError(
            "result implementation contract differs"
        )
    for name in ("module", "publisher"):
        record = implementation.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise TargetRefreshCommonAnchorError(
                f"result implementation identity differs: {name}"
            )
    return contract


def build_prepare_command(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    paths_config: Path,
    python_executable: str,
) -> list[str]:
    """Build one manager command that can only prepare an unauthorized plan."""
    common = contract["common_training_contract"]
    measurement = contract["measurement_contract"]
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
        str(common["one_segment_game_safety_ceiling"]),
        "--segment-games",
        str(common["one_segment_game_safety_ceiling"]),
        "--optimizer-update-bound",
        str(arm["optimizer_update_bound"]),
        "--measurement-anchor-game",
        str(measurement["anchor_game"]),
        "--measurement-anchor-expected-update-count",
        str(arm["anchor_expected_update_count"]),
        "--measurement-every-updates",
        str(measurement["measurement_every_updates"]),
        "--measurement-games-per-opponent",
        str(measurement["games_per_opponent_per_checkpoint"]),
        "--measurement-sanmill-node-budget",
        str(measurement["sanmill_node_budget"]),
        "--measurement-temperature",
        str(measurement["measurement_temperature"]),
        "--no-exact-resume",
        "--engine-profile",
        "sanmill-fixed-resource",
        "--self-play-ratio",
        str(common["frozen_target_ratio"]),
        "--target-refresh-every",
        str(arm["target_refresh_every_games"]),
        "--lr-adaptation-mode",
        "fixed",
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
        common["specialist_read_mode"],
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


def _command_semantics(command: list[str], *, ignore_arm: bool) -> str:
    manager_args = manager._build_parser().parse_args(command[2:])
    common_args = manager._common_trainer_args(
        manager_args,
        Path(manager_args.paths_config),
    )
    args = trainer._build_argument_parser().parse_args(
        ["--preflight", "long-run", *common_args]
    )
    ignored = {"experiment_id", "specialist_db"}
    if ignore_arm:
        ignored.update(
            {
                "seed",
                "update_target_every",
                "optimizer_update_bound",
                "measurement_anchor_expected_update_count",
            }
        )
    semantics = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key not in ignored
    }
    return canonical_sha256(semantics)


def validate_prepare_commands(
    contract: Mapping[str, Any], commands: list[list[str]]
) -> None:
    if len(commands) != 4:
        raise TargetRefreshCommonAnchorError("prepare command count differs")
    base_semantics: set[str] = set()
    for arm, command in zip(_ordered_arms(contract), commands, strict=True):
        if "authorize" in command or "run-next" in command:
            raise TargetRefreshCommonAnchorError("preparation command can train")
        args = manager._build_parser().parse_args(command[2:])
        if (
            args.seed != arm["seed"]
            or args.target_refresh_every
            != arm["target_refresh_every_games"]
            or args.optimizer_update_bound != arm["optimizer_update_bound"]
            or args.measurement_anchor_expected_update_count
            != arm["anchor_expected_update_count"]
            or args.lr_adaptation_mode != "fixed"
            or not args.no_exact_resume
        ):
            raise TargetRefreshCommonAnchorError(
                f"command factor differs: {arm['arm_id']}"
            )
        base_semantics.add(_command_semantics(command, ignore_arm=True))
    if len(base_semantics) != 1:
        raise TargetRefreshCommonAnchorError(
            "commands differ outside seed and frozen target-refresh treatment"
        )


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
        raise TargetRefreshCommonAnchorError("source audit requires dev")
    if status:
        raise TargetRefreshCommonAnchorError(
            "source audit requires a clean tracked and untracked worktree"
        )
    if origin_main != contract["lineage"]["main_review"]["reviewed_tip"]:
        raise TargetRefreshCommonAnchorError("origin/main moved after review")
    for commit in contract["lineage"]["required_implementation_commits"]:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise TargetRefreshCommonAnchorError(
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


def _inspect_source_evidence(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name, record in contract["source_evidence"].items():
        path = _repository_path(root, record["path"], field="source evidence")
        observed = _sha256_file(path)
        if observed != record["sha256"]:
            raise TargetRefreshCommonAnchorError(
                f"source evidence differs: {name}"
            )
        evidence[name] = {"path": record["path"], "sha256": observed}
    return evidence


def inspect_result_implementation(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    implementation = contract["analysis"]["result_implementation"]
    result = {"result_schema": implementation["result_schema"]}
    for name in ("module", "publisher"):
        expected = implementation[name]
        path = _repository_path(root, expected["path"], field=f"result {name}")
        if not path.is_file() or not _tracked_file(root, path):
            raise TargetRefreshCommonAnchorError(f"result {name} is not tracked")
        observed = _sha256_file(path)
        if observed != expected["sha256"]:
            raise TargetRefreshCommonAnchorError(
                f"result {name} SHA-256 differs"
            )
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
                    _repository_path(
                        root,
                        arm["specialist_db"],
                        field="specialist_db",
                    ),
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


def _trainer_args(plan: ManagedPlan) -> Any:
    return trainer._build_argument_parser().parse_args(
        ["--preflight", "long-run", *plan.common_trainer_args]
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
    measurement = contract["measurement_contract"]
    resources = contract["resources"]
    expected = {
        "plan_id": arm["plan_id"],
        "objective": f"{contract['objective']}; arm={arm['arm_id']}",
        "experiment_id": arm["experiment_id"],
        "git_commit": source_commit,
        "control_dir": str(
            _repository_path(root, arm["control_dir"], field="control_dir")
        ),
        "paths_config": str(paths_config),
        "max_games": common["max_games_schedule"],
        "completion_game_bound": common["one_segment_game_safety_ceiling"],
        "segment_games": common["one_segment_game_safety_ceiling"],
        "max_wall_hours": resources["active_wall_hours_per_arm"],
        "allow_safe_exact_resume": False,
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            raise TargetRefreshCommonAnchorError(
                f"managed plan differs for {arm['arm_id']}: {field}"
            )
    if plan.paths_config_sha256 != _sha256_file(paths_config):
        raise TargetRefreshCommonAnchorError("managed path registry hash differs")
    args = _trainer_args(plan)
    expected_args = {
        "experiment_id": arm["experiment_id"],
        "seed": arm["seed"],
        "update_target_every": arm["target_refresh_every_games"],
        "optimizer_update_bound": arm["optimizer_update_bound"],
        "measurement_anchor_game": measurement["anchor_game"],
        "measurement_anchor_expected_update_count": arm[
            "anchor_expected_update_count"
        ],
        "measurement_every_updates": measurement["measurement_every_updates"],
        "measurement_games_per_opponent": measurement[
            "games_per_opponent_per_checkpoint"
        ],
        "measurement_sanmill_node_budget": measurement[
            "sanmill_node_budget"
        ],
        "measurement_temperature": measurement["measurement_temperature"],
        "specialist_read_mode": common["specialist_read_mode"],
        "lr_adaptation_mode": "fixed",
        "mill_bonus_mode": common["mill_bonus_mode"],
        "malom_policy_aux_coef": common["malom_policy_aux_coefficient"],
        "malom_policy_aux_mode": common["malom_policy_aux_mode"],
        "max_games": common["max_games_schedule"],
        "max_ply": common["max_logical_plies"],
        "max_ply_branch": common["max_logical_plies"],
        "max_branches_per_game": 0,
        "batch_games": 1,
        "sim_ply_depth": common["sim_ply_depth"],
        "self_play_ratio": common["frozen_target_ratio"],
        "curriculum_advance_policy": "fixed-resource",
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
            raise TargetRefreshCommonAnchorError(
                f"trainer argument differs for {arm['arm_id']}: {field}"
            )
    if list(args.sanmill_node_ladder) != common["sanmill_node_ladder"]:
        raise TargetRefreshCommonAnchorError("Sanmill node ladder differs")
    if list(args.sanmill_stage_games) != common["fixed_resource_stage_games"]:
        raise TargetRefreshCommonAnchorError("Sanmill stage durations differ")
    specialist_db = _repository_path(
        root, arm["specialist_db"], field="specialist_db"
    )
    if Path(args.specialist_db).resolve() != specialist_db:
        raise TargetRefreshCommonAnchorError("arm SpecialistDB path differs")
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
    base_semantics: set[str] = set()
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
        database = _inspect_specialist_database(
            _repository_path(root, arm["specialist_db"], field="specialist_db"),
            template,
        )
        if (control_dir / "authorization.json").exists():
            raise TargetRefreshCommonAnchorError(
                f"arm is already authorized: {arm['arm_id']}"
            )
        if (control_dir / "segments").exists():
            raise TargetRefreshCommonAnchorError(
                f"arm already has segment output: {arm['arm_id']}"
            )
        semantics = {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_")
            and key
            not in {
                "experiment_id",
                "seed",
                "specialist_db",
                "update_target_every",
                "optimizer_update_bound",
                "measurement_anchor_expected_update_count",
            }
        }
        base_semantics.add(canonical_sha256(semantics))
        record: dict[str, Any] = {
            "arm_id": arm["arm_id"],
            "condition": arm["condition"],
            "launch_order": arm["launch_order"],
            "seed": arm["seed"],
            "target_refresh_every_games": arm["target_refresh_every_games"],
            "anchor_expected_update_count": arm[
                "anchor_expected_update_count"
            ],
            "optimizer_update_bound": arm["optimizer_update_bound"],
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
    if len(base_semantics) != 1:
        raise TargetRefreshCommonAnchorError(
            "prepared arms differ outside the frozen treatment and seed bounds"
        )
    return audited


def inspect_source_readiness(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Perform a read-only audit without creating plans or databases."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    if report_path is None:
        report_path = root / DEFAULT_REPORT
    report_path = report_path.resolve(strict=False)
    contract = load_target_refresh_common_anchor_contract(contract_path)
    source = _inspect_source(root, contract)
    evidence = _inspect_source_evidence(root, contract)
    template = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    result_analysis = inspect_result_implementation(root, contract)
    targets = inspect_preparation_targets(
        root,
        contract,
        report_path=report_path,
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
        unresolved.append("publish the exact frozen source to origin/dev")
    if not targets["absent"]:
        unresolved.append("quarantine pre-existing preparation targets")
    if not unresolved:
        unresolved.append(
            "generate four immutable plans and preflights; training remains unauthorized"
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


def prepare_target_refresh_common_anchor_diagnostic(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create four plans and preflights without authorization or training."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    report_path = report_path.resolve(strict=False)
    contract = load_target_refresh_common_anchor_contract(contract_path)
    source = _inspect_source(root, contract)
    if not source["published"]:
        raise TargetRefreshCommonAnchorError("dev must equal origin/dev")
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
            raise TargetRefreshCommonAnchorError(
                f"manager output is not JSON: {arm['arm_id']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise TargetRefreshCommonAnchorError(
                f"manager state differs: {arm['arm_id']}"
            )
        plan_path = (
            _repository_path(root, arm["control_dir"], field="control_dir")
            / "plan.json"
        )
        plan = load_managed_plan(plan_path)
        preflight_command = _build_fresh_preflight_command(
            plan,
            root=root,
            python_executable=python_executable,
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
            raise TargetRefreshCommonAnchorError(
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
            raise TargetRefreshCommonAnchorError(
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
        raise TargetRefreshCommonAnchorError(
            f"readiness report already exists: {report_path}"
        ) from exc
    return report


def publish_source_readiness(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise TargetRefreshCommonAnchorError(
            f"source readiness report already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_REPORT",
    "DEFAULT_SOURCE_REPORT",
    "EXPECTED_CONDITIONS",
    "EXPECTED_SEEDS",
    "READINESS_SCHEMA",
    "RESULT_SCHEMA",
    "TargetRefreshCommonAnchorError",
    "build_prepare_commands",
    "inspect_source_readiness",
    "load_target_refresh_common_anchor_contract",
    "prepare_target_refresh_common_anchor_diagnostic",
    "publish_source_readiness",
    "validate_prepare_commands",
]
