"""Fail-closed preparation for the paired normalized auxiliary calibration."""

from __future__ import annotations

import json
import math
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
    MillBonusAblationReadinessError,
    _build_fresh_preflight_command,
    _inspect_specialist_database,
    _repository_path,
    _run_checked,
    _sha256_file,
    _strict_json,
    _validate_unlaunched_preflight,
    assert_preparation_outputs_ignored,
    assert_preparation_targets_absent,
    inspect_published_source,
    inspect_runtime_identities,
    inspect_template,
)
from scripts import train_s_gen_v2 as trainer


CALIBRATION_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-calibration-plan.v1"
)
SOURCE_READINESS_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-source-readiness.v1"
)
READINESS_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-calibration-readiness.v1"
)
RESULT_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-calibration-result.v1"
)
DEFAULT_CONTRACT = Path(
    "docs/experiments/"
    "sanmill-malom-policy-auxiliary-normalized-calibration-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_SOURCE_REPORT = Path(
    "out/malom-policy-auxiliary-normalized-calibration-v1/"
    "source-readiness.json"
)
DEFAULT_REPORT = Path(
    "out/malom-policy-auxiliary-normalized-calibration-v1/readiness.json"
)


MalomPolicyAuxiliaryNormalizedCalibrationReadinessError = (
    MillBonusAblationReadinessError
)


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "Git audit failed: " + " ".join(arguments)
        )
    return result.stdout.strip()


def _ordered_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["arms"], key=lambda arm: int(arm["launch_order"]))


def _tracked_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _ignored_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _assert_ancestor(root: Path, ancestor: str, descendant: str, *, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            f"required {label} commit is absent from the source lineage"
        )


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
        config_rng = random.Random(torch_seed)
        colour = "white" if config_rng.random() < 0.5 else "black"
        opponent = (
            "frozen" if config_rng.random() < frozen_ratio else "sanmill"
        )
        counts[f"{opponent}_{colour}"] += 1
    return counts


def load_normalized_calibration_contract(path: str | Path) -> dict[str, Any]:
    """Load and fully validate the immutable six-arm contract."""
    contract = _strict_json(Path(path))
    if contract.get("schema_version") != CALIBRATION_SCHEMA:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "unsupported normalized calibration schema"
        )
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration plan identity differs"
        )
    if contract.get("status") != "designed_unlaunched_needs_publication":
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration status is not unlaunched"
        )
    authorization = contract.get("authorization")
    if not isinstance(authorization, Mapping) or authorization != {
        "authorized_segments_per_arm": 0,
        "launch_authorized": False,
        "promotion_allowed": False,
        "publication_allowed": False,
    }:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration must not authorize launch or publication"
        )

    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 6:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration must contain six arms"
        )
    ordered = _ordered_arms(contract)
    if [arm.get("launch_order") for arm in ordered] != list(range(1, 7)):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration launch order differs"
        )
    expected_seeds = [55, 55, 56, 56, 57, 57]
    expected_conditions = ["control", "normalized-0.25"] * 3
    expected_modes = ["fixed", "policy-head-normalized"] * 3
    if (
        [arm.get("seed") for arm in ordered] != expected_seeds
        or [arm.get("condition") for arm in ordered] != expected_conditions
        or [arm.get("malom_policy_aux_mode") for arm in ordered]
        != expected_modes
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration seed pairing differs"
        )
    for arm in ordered:
        expected_scaling = {
            "malom_policy_aux_coef": 0.0,
            "malom_policy_aux_target_ratio": 0.25,
            "malom_policy_aux_coef_cap": 0.25,
            "malom_policy_aux_denominator_floor": 1e-12,
            "mill_bonus_mode": "malom-preserving-only",
        }
        if any(arm.get(key) != value for key, value in expected_scaling.items()):
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"normalized scaling fields differ: {arm.get('arm_id')}"
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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"normalized calibration arm field is not unique: {field}"
            )

    pairing = contract.get("pairing")
    if (
        not isinstance(pairing, Mapping)
        or pairing.get("seeds") != [55, 56, 57]
        or pairing.get("conditions") != ["control", "normalized-0.25"]
        or pairing.get("single_changed_training_factor")
        != "malom_policy_aux_mode"
        or pairing.get("same_fresh_initialization_and_schedule_within_seed")
        is not True
        or pairing.get("single_process_at_a_time") is not True
        or pairing.get("pair_order_is_frozen") is not True
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration pairing contract differs"
        )

    implementation = contract.get("analysis", {}).get("result_implementation")
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != {"module", "publisher", "result_schema"}
        or implementation.get("result_schema") != RESULT_SCHEMA
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized result implementation contract differs"
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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"normalized result implementation identity differs: {name}"
            )

    common = contract.get("common_training_contract")
    resources = contract.get("resources")
    if not isinstance(common, Mapping) or not isinstance(resources, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration resource contract is missing"
        )
    games = int(resources.get("completed_games_per_arm", -1))
    if (
        games != 100
        or resources.get("maximum_completed_games_total") != 600
        or resources.get("maximum_active_wall_hours_total") != 2.0
        or not math.isclose(
            float(resources.get("active_wall_hours_per_arm", -1.0)) * 6,
            2.0,
        )
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration resource envelope differs"
        )
    requested_nodes = 0
    schedules = resources.get("schedule_counts_by_seed")
    if not isinstance(schedules, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration schedules are missing"
        )
    for seed in pairing["seeds"]:
        observed = _schedule_counts(seed, games, common["frozen_target_ratio"])
        if schedules.get(str(seed)) != observed:
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"normalized calibration schedule differs for seed {seed}"
            )
        sanmill_games = observed["sanmill_black"] + observed["sanmill_white"]
        requested_nodes += (
            sanmill_games
            * 2
            * (int(common["max_logical_plies"]) // 2)
            * int(common["sanmill_node_ladder"][0])
        )
    if resources.get("maximum_requested_sanmill_nodes_total") != requested_nodes:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration node ceiling differs"
        )
    return contract


def inspect_local_source(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Report local source state without treating an unpublished tip as ready."""
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    upstream = _git_output(root, "rev-parse", "origin/dev")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != "dev":
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration preparation requires dev"
        )
    if status:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "normalized calibration source audit requires a clean worktree"
        )
    for name, commit in contract["lineage"]["implementation_commits"].items():
        _assert_ancestor(root, commit, head, label=name)
    actual_main = _git_output(root, "rev-parse", "origin/main")
    if actual_main != contract["lineage"]["main_review"]["reviewed_tip"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "origin/main moved after the recorded review"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": upstream,
        "origin_main": actual_main,
        "published": head == upstream,
        "worktree_clean": True,
    }


def inspect_batch_capture_evidence(
    root: Path,
    contract: Mapping[str, Any],
    *,
    source_head: str,
) -> dict[str, Any]:
    """Bind the tracked interpretation to the exact ignored raw result."""
    preparation = contract.get("preparation_evidence")
    if not isinstance(preparation, Mapping) or set(preparation) != {
        "no_update_batch_capture"
    }:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture preparation evidence contract differs"
        )
    spec = preparation["no_update_batch_capture"]
    if not isinstance(spec, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture evidence identity is missing"
        )

    interpretation_path = _repository_path(
        root,
        str(spec["interpretation_path"]),
        field="batch-capture interpretation",
    )
    if not interpretation_path.is_file() or not _tracked_file(
        root, interpretation_path
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture interpretation is not tracked"
        )
    if (
        interpretation_path.stat().st_size != spec["interpretation_size_bytes"]
        or _sha256_file(interpretation_path) != spec["interpretation_sha256"]
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture interpretation bytes differ"
        )
    interpretation_commit = _git_output(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        interpretation_path.relative_to(root).as_posix(),
    )
    if interpretation_commit != spec["interpretation_commit"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture interpretation commit differs"
        )
    _assert_ancestor(
        root,
        str(spec["interpretation_commit"]),
        source_head,
        label="batch-capture interpretation",
    )

    result_path = _repository_path(
        root,
        str(spec["result_path"]),
        field="batch-capture raw result",
    )
    if not result_path.is_file() or not _ignored_file(root, result_path):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture raw result must exist and remain ignored"
        )
    if _sha256_file(result_path) != spec["result_sha256"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture raw result SHA-256 differs"
        )
    result = _strict_json(result_path)
    identity = result.get("report_identity")
    body = {key: value for key, value in result.items() if key != "report_identity"}
    if identity != spec["result_identity"] or identity != canonical_sha256(body):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture raw result identity differs"
        )
    if (
        result.get("status") != spec["result_status"]
        or result.get("readiness_identity") != spec["readiness_identity"]
        or result.get("source", {}).get("commit")
        != spec["result_source_commit"]
        or result.get("source", {}).get("tree") != spec["result_source_tree"]
        or result.get("plan", {}).get("identity") != spec["plan_identity"]
        or result.get("plan", {}).get("raw_sha256") != spec["plan_raw_sha256"]
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture source or plan identity differs"
        )
    claim = result.get("claim_boundary")
    if not isinstance(claim, Mapping) or any(
        claim.get(field) not in (False, 0)
        for field in (
            "candidate_checkpoint_loaded",
            "coefficient_or_normalization_selected",
            "optimizer_constructed",
            "optimizer_steps",
            "strength_or_promotion_claim",
            "training_updates",
        )
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture no-update boundary differs"
        )
    if (
        result.get("data_before") != result.get("data_after")
        or result.get("source_evidence_before")
        != result.get("source_evidence_after")
        or any(
            not model.get("learner_unchanged")
            or not model.get("frozen_unchanged")
            or model.get("learner_gradients_populated")
            for model in result.get("models", [])
        )
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture mutation checks differ"
        )
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture result summary is missing"
        )
    expected = spec["result_summary"]
    target = next(
        (
            item
            for item in summary.get("candidate_scale_distributions", [])
            if item.get("target_policy_head_ratio") == 0.25
        ),
        None,
    )
    if not isinstance(target, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture target-0.25 distribution is missing"
        )
    coefficient = target.get("effective_coefficient")
    if not isinstance(coefficient, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture target-0.25 coefficient distribution is missing"
        )
    observed_summary = {
        "batches": summary.get("batches"),
        "batches_with_informative_steps": summary.get(
            "batches_with_informative_steps"
        ),
        "fresh_seeds": [item.get("seed") for item in summary.get("per_seed", [])],
        "games": summary.get("games"),
        "informative_steps": sum(
            summary.get("informative_steps_by_phase", {}).values()
        ),
        "labelled_steps": sum(summary.get("labelled_steps_by_phase", {}).values()),
        "target_025_effective_coefficient_max": coefficient.get("max"),
        "target_025_effective_coefficient_median": coefficient.get("median"),
        "target_025_effective_coefficient_min": coefficient.get("min"),
    }
    if observed_summary != expected or summary.get("selection_made") is not False:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "batch-capture result summary does not reconcile"
        )
    return {
        "interpretation_path": str(interpretation_path),
        "interpretation_sha256": spec["interpretation_sha256"],
        "result_path": str(result_path),
        "result_sha256": spec["result_sha256"],
        "result_identity": identity,
        "summary": observed_summary,
    }


def inspect_result_implementation(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the pre-result analyzer and publisher bytes are frozen."""
    implementation = contract["analysis"]["result_implementation"]
    result = {"result_schema": implementation["result_schema"]}
    for name in ("module", "publisher"):
        expected = implementation[name]
        path = _repository_path(
            root,
            expected["path"],
            field=f"normalized result {name}",
        )
        if not path.is_file() or not _tracked_file(root, path):
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"normalized result {name} is not tracked"
            )
        observed = _sha256_file(path)
        if observed != expected["sha256"]:
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"normalized result {name} SHA-256 differs"
            )
        result[name] = {
            "path": expected["path"],
            "sha256": observed,
        }
    return result


def _objective(contract: Mapping[str, Any], arm: Mapping[str, Any]) -> str:
    return f'{contract["objective"]}; arm={arm["arm_id"]}'


def build_prepare_command(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    paths_config: Path,
    python_executable: str,
) -> list[str]:
    """Build the exact manager command for one unlaunched arm."""
    common = contract["common_training_contract"]
    resources = contract["resources"]
    diagnostic = contract["analysis"]["fixed_development_diagnostic"]
    return [
        python_executable,
        str(root / "scripts/manage_generalist_run.py"),
        "prepare",
        "--control-dir",
        str(_repository_path(root, arm["control_dir"], field="control_dir")),
        "--max-wall-hours",
        str(resources["active_wall_hours_per_arm"]),
        "--plan-id",
        arm["plan_id"],
        "--objective",
        _objective(contract, arm),
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
        arm["mill_bonus_mode"],
        "--malom-policy-aux-coef",
        str(arm["malom_policy_aux_coef"]),
        "--malom-policy-aux-mode",
        arm["malom_policy_aux_mode"],
        "--malom-policy-aux-target-ratio",
        str(arm["malom_policy_aux_target_ratio"]),
        "--malom-policy-aux-coef-cap",
        str(arm["malom_policy_aux_coef_cap"]),
        "--malom-policy-aux-denominator-floor",
        str(arm["malom_policy_aux_denominator_floor"]),
        "--specialist-db",
        str(_repository_path(root, arm["specialist_db"], field="specialist_db")),
        "--policy-health-gate",
        "--policy-health-device",
        diagnostic["device"],
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


def _trainer_args(plan: ManagedPlan) -> Any:
    parser = trainer._build_argument_parser()
    return parser.parse_args(["--preflight", "long-run", *plan.common_trainer_args])


def _normalised_training_semantics(args: Any, *, ignore_seed: bool) -> str:
    ignored = {
        "experiment_id",
        "malom_policy_aux_mode",
        "specialist_db",
    }
    if ignore_seed:
        ignored.add("seed")
    value = {
        key: item
        for key, item in vars(args).items()
        if not key.startswith("_") and key not in ignored
    }
    return canonical_sha256(value)


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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"managed plan differs for {arm['arm_id']}: {field}"
            )
    if plan.game_bound != resources["completed_games_per_arm"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "managed completion bound differs"
        )
    if plan.paths_config_sha256 != _sha256_file(paths_config):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "managed path registry hash differs"
        )
    args = _trainer_args(plan)
    expected_args = {
        "experiment_id": arm["experiment_id"],
        "seed": arm["seed"],
        "mill_bonus_mode": arm["mill_bonus_mode"],
        "malom_policy_aux_coef": arm["malom_policy_aux_coef"],
        "malom_policy_aux_mode": arm["malom_policy_aux_mode"],
        "malom_policy_aux_target_ratio": arm["malom_policy_aux_target_ratio"],
        "malom_policy_aux_coef_cap": arm["malom_policy_aux_coef_cap"],
        "malom_policy_aux_denominator_floor": arm[
            "malom_policy_aux_denominator_floor"
        ],
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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"trainer argument differs for {arm['arm_id']}: {field}"
            )
    if list(args.sanmill_node_ladder) != common["sanmill_node_ladder"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "Sanmill node ladder differs"
        )
    if list(args.sanmill_stage_games) != common["fixed_resource_stage_games"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "Sanmill stage durations differ"
        )
    specialist_db = _repository_path(
        root,
        arm["specialist_db"],
        field="specialist_db",
    )
    if Path(args.specialist_db).resolve() != specialist_db:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "arm SpecialistDB path differs"
        )
    diagnostic = contract["analysis"]["fixed_development_diagnostic"]
    gate = plan.policy_health
    if gate is None or gate.device != diagnostic["device"]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "policy-health device differs"
        )
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
        != diagnostic[
            "minimum_candidate_preserving_minus_downgrading_logit_margin"
        ]
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "policy-health contract differs"
        )
    return args


def audit_prepared_plans(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
    preflight_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Verify managed plans and paired one-factor equivalence."""
    template = contract["data_contract"]["specialist_db_initial_template"]
    audited: list[dict[str, Any]] = []
    pair_semantics: dict[int, set[str]] = {}
    global_semantics: set[str] = set()
    for arm in _ordered_arms(contract):
        control_dir = _repository_path(
            root,
            arm["control_dir"],
            field="control_dir",
        )
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
            root,
            arm["specialist_db"],
            field="specialist_db",
        )
        database = _inspect_specialist_database(specialist_db, template)
        if (control_dir / "authorization.json").exists():
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"arm is already authorized: {arm['arm_id']}"
            )
        if (control_dir / "segments").exists():
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"arm already has segment output: {arm['arm_id']}"
            )
        pair_semantics.setdefault(int(arm["seed"]), set()).add(
            _normalised_training_semantics(args, ignore_seed=False)
        )
        global_semantics.add(
            _normalised_training_semantics(args, ignore_seed=True)
        )
        record = {
            "arm_id": arm["arm_id"],
            "condition": arm["condition"],
            "launch_order": arm["launch_order"],
            "seed": arm["seed"],
            "malom_policy_aux_mode": arm["malom_policy_aux_mode"],
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
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "a seed pair differs outside the normalized scaling mode"
        )
    if len(global_semantics) != 1:
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            "seed pairs differ outside seed and normalized scaling mode"
        )
    return audited


def inspect_source_readiness(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Audit implementation inputs without creating plans or databases."""
    contract = load_normalized_calibration_contract(contract_path)
    source = inspect_local_source(root, contract)
    evidence = inspect_batch_capture_evidence(
        root,
        contract,
        source_head=source["head"],
    )
    template = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    result_analysis = inspect_result_implementation(root, contract)
    body = {
        "schema_version": SOURCE_READINESS_SCHEMA,
        "state": (
            "source_ready_for_local_preparation"
            if source["published"]
            else "implementation_complete_needs_publication"
        ),
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "source": source,
        "batch_capture_evidence": evidence,
        "template": template,
        "runtime": runtime,
        "result_analysis": result_analysis,
        "commands": build_prepare_commands(
            root=root,
            contract=contract,
            paths_config=paths_config,
            python_executable=python_executable,
        ),
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": (
            ["publish the implementation and preparation commits to origin/dev"]
            if not source["published"]
            else ["run fail-closed local preparation; training remains unauthorized"]
        ),
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def prepare_normalized_calibration(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create six plans and preflights without authorization or training."""
    contract = load_normalized_calibration_contract(contract_path)
    source = inspect_published_source(root, contract)
    evidence = inspect_batch_capture_evidence(
        root,
        contract,
        source_head=source["head"],
    )
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
    preflights: dict[str, dict[str, Any]] = {}
    template_path = Path(template_record["path"])
    for arm, command in zip(_ordered_arms(contract), commands, strict=True):
        specialist_db = _repository_path(
            root,
            arm["specialist_db"],
            field="specialist_db",
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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"manager output is not JSON: {arm['arm_id']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
                f"manager state differs: {arm['arm_id']}"
            )
        plan_path = _repository_path(
            root,
            arm["control_dir"],
            field="control_dir",
        ) / "plan.json"
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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
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
            raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
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
        "batch_capture_evidence": evidence,
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
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
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
        raise MalomPolicyAuxiliaryNormalizedCalibrationReadinessError(
            f"source readiness report already exists: {path}"
        ) from exc
