"""Fail-closed preparation for the Malom policy-auxiliary calibration."""

from __future__ import annotations

import json
import math
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
    "nmm.sanmill-malom-policy-auxiliary-calibration-plan.v1"
)
READINESS_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-calibration-readiness.v1"
)
DEFAULT_CONTRACT = Path(
    "docs/experiments/"
    "sanmill-malom-policy-auxiliary-calibration-smoke-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_REPORT = Path(
    "out/malom-policy-auxiliary-calibration-smoke-v1/readiness.json"
)


MalomPolicyAuxiliaryCalibrationReadinessError = (
    MillBonusAblationReadinessError
)


def _ordered_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["arms"], key=lambda arm: int(arm["launch_order"]))


def load_calibration_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable four-arm calibration contract."""
    contract = _strict_json(Path(path))
    if contract.get("schema_version") != CALIBRATION_SCHEMA:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "unsupported policy-auxiliary calibration schema"
        )
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration plan identity differs"
        )
    if contract.get("status") != (
        "designed_unlaunched_needs_product_authorization"
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration contract status is not unlaunched"
        )
    authorization = contract.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(field) is not False
        for field in (
            "launch_authorized",
            "publication_allowed",
            "promotion_allowed",
        )
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration contract must not authorize launch or publication"
        )
    if authorization.get("authorized_segments_per_arm") != 0:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration contract authorizes a segment"
        )

    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 4:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration contract must contain four arms"
        )
    ordered = _ordered_arms(contract)
    if [arm.get("launch_order") for arm in ordered] != [1, 2, 3, 4]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration launch order is incomplete"
        )
    if [arm.get("malom_policy_aux_coef") for arm in ordered] != [
        0.0,
        0.03,
        0.1,
        0.3,
    ]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration coefficients differ"
        )
    if {arm.get("seed") for arm in ordered} != {51}:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration arms do not share seed 51"
        )
    if {arm.get("mill_bonus_mode") for arm in ordered} != {
        "malom-preserving-only"
    }:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration reward mode differs"
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
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"calibration arm field is not unique: {field}"
            )

    pairing = contract.get("pairing")
    expected_allowlist = [
        "experiment_id",
        "plan_id",
        "control_dir",
        "specialist_db",
        "malom_policy_aux_coef",
    ]
    if (
        not isinstance(pairing, Mapping)
        or pairing.get("arm_difference_allowlist") != expected_allowlist
        or pairing.get("single_changed_factor") != "malom_policy_aux_coef"
        or pairing.get("single_process_at_a_time") is not True
        or pairing.get("same_fresh_initialization_and_schedule") is not True
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration pairing contract differs"
        )

    common = contract.get("common_training_contract")
    resources = contract.get("resources")
    if not isinstance(common, Mapping) or not isinstance(resources, Mapping):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration resource contract is missing"
        )
    schedule = resources.get("schedule_counts_per_arm")
    if not isinstance(schedule, Mapping) or set(schedule) != {
        "frozen_black",
        "frozen_white",
        "sanmill_black",
        "sanmill_white",
    }:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration schedule counts are invalid"
        )
    if sum(int(value) for value in schedule.values()) != 100:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration schedule count does not equal one arm"
        )
    sanmill_games = (
        int(schedule["sanmill_black"]) + int(schedule["sanmill_white"])
    ) * len(arms)
    expected_nodes = (
        sanmill_games
        * (int(common["max_logical_plies"]) // 2)
        * int(common["sanmill_node_ladder"][0])
    )
    if (
        resources.get("completed_games_per_arm") != 100
        or resources.get("maximum_completed_games_total") != 400
        or resources.get("active_wall_hours_per_arm") != 0.5
        or resources.get("maximum_active_wall_hours_total") != 2.0
        or resources.get("maximum_requested_sanmill_nodes_total")
        != expected_nodes
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "calibration resource envelope differs"
        )
    return contract


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


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def inspect_gradient_evidence(
    root: Path,
    contract: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the tracked summary to its ignored exact raw probe."""
    preparation = contract.get("preparation_evidence")
    if not isinstance(preparation, Mapping) or set(preparation) != {
        "tracked_manifest"
    }:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient preparation evidence contract differs"
        )
    spec = preparation["tracked_manifest"]
    expected_keys = {
        "path",
        "probe_identity",
        "probe_path",
        "probe_schema_version",
        "probe_sha256",
        "probe_size_bytes",
        "probe_source_commit",
        "schema_version",
        "sha256",
    }
    if not isinstance(spec, Mapping) or set(spec) != expected_keys:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence identity fields differ"
        )

    manifest_path = _repository_path(
        root, str(spec["path"]), field="gradient evidence manifest"
    )
    if not manifest_path.is_file() or not _tracked_file(root, manifest_path):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence manifest is not tracked"
        )
    if _sha256_file(manifest_path) != spec["sha256"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence manifest SHA-256 differs"
        )
    manifest = _strict_json(manifest_path)
    if manifest.get("schema_version") != spec["schema_version"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence manifest schema differs"
        )
    decision = manifest.get("decision")
    if not isinstance(decision, Mapping) or decision != {
        "next_stage": "bounded_optimizer_integration_smoke",
        "reward_only_escalation": "closed",
        "training_launch_authorized": False,
        "verdict": "ready_for_optimizer_integration_smoke_preparation",
    }:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence decision differs"
        )
    expected_probe_ref = {
        "identity": spec["probe_identity"],
        "path": spec["probe_path"],
        "schema_version": spec["probe_schema_version"],
        "sha256": spec["probe_sha256"],
        "size_bytes": spec["probe_size_bytes"],
        "source_commit": spec["probe_source_commit"],
    }
    if manifest.get("probe") != expected_probe_ref:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence probe reference differs"
        )

    probe_path = _repository_path(
        root, str(spec["probe_path"]), field="raw gradient probe"
    )
    if not probe_path.is_file() or not _ignored_file(root, probe_path):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "raw gradient probe must exist and remain ignored"
        )
    if (
        probe_path.stat().st_size != spec["probe_size_bytes"]
        or _sha256_file(probe_path) != spec["probe_sha256"]
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "raw gradient probe bytes differ"
        )
    probe = _strict_json(probe_path)
    if probe.get("schema_version") != spec["probe_schema_version"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "raw gradient probe schema differs"
        )
    probe_identity = probe.get("probe_identity")
    probe_body = {
        key: value for key, value in probe.items() if key != "probe_identity"
    }
    if (
        probe_identity != spec["probe_identity"]
        or probe_identity != canonical_sha256(probe_body)
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "raw gradient probe identity differs"
        )
    identities = probe.get("identities")
    if (
        not isinstance(identities, Mapping)
        or identities.get("source_commit") != spec["probe_source_commit"]
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "raw gradient probe source differs"
        )
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            str(spec["probe_source_commit"]),
            str(source["head"]),
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient probe source is not in the published lineage"
        )
    if probe.get("mutation_checks") != {
        "human_db_unchanged": True,
        "malom_anchor_unchanged": True,
        "specialist_db_unchanged": True,
        "tracked_worktree_clean_after": True,
    }:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient probe mutation checks differ"
        )
    scope = probe.get("scope")
    if not isinstance(scope, Mapping) or scope != {
        "development_corpus": True,
        "in_memory_sgd_direction_is_not_adam_trajectory": True,
        "no_checkpoint_write": True,
        "no_database_write": True,
        "no_persistent_model_update": True,
        "no_strength_or_promotion_claim": True,
        "no_training_run": True,
    }:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient probe scope differs"
        )

    seed_reports = probe.get("seed_reports")
    if (
        not isinstance(seed_reports, list)
        or [report.get("seed") for report in seed_reports] != [48, 49, 50]
    ):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient probe seed cohort differs"
        )
    gradients: list[float] = []
    derivatives: list[float] = []
    cosines: list[float] = []
    probabilities: list[float] = []
    losses: list[float] = []
    warm_label_seconds: list[float] = []
    for index, report in enumerate(seed_reports):
        coverage = report.get("coverage")
        gradient_probe = report.get("gradient_probe")
        if not isinstance(coverage, Mapping) or not isinstance(
            gradient_probe, Mapping
        ):
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                "gradient probe seed record is invalid"
            )
        required_coverage = {
            "states": 64,
            "actions": 1583,
            "preserving_actions": 1168,
            "downgrading_actions": 415,
            "phase_states": {"flying": 21, "movement": 21, "placement": 22},
            "phase_informative_states": {
                "flying": 6,
                "movement": 8,
                "placement": 15,
            },
            "root_wdl_counts": {"draw": 24, "loss": 22, "win": 18},
        }
        if any(coverage.get(key) != value for key, value in required_coverage.items()):
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                "gradient probe exact-label coverage differs"
            )
        gradient = gradient_probe.get("gradient")
        alignment = gradient_probe.get("gradient_alignment")
        baseline = gradient_probe.get("baseline")
        trials = gradient_probe.get("coefficient_trials")
        if (
            not isinstance(gradient, Mapping)
            or gradient.get("finite") is not True
            or not _finite_positive(gradient.get("l2_norm"))
            or not isinstance(alignment, Mapping)
            or not _finite_positive(alignment.get("directional_derivative"))
            or not _finite_positive(alignment.get("descent_cosine"))
            or not isinstance(baseline, Mapping)
            or not isinstance(baseline.get("all"), Mapping)
            or not isinstance(trials, list)
            or gradient_probe.get("original_model_unchanged") is not True
        ):
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                "gradient probe direction record differs"
            )
        if [trial.get("coefficient") for trial in trials] != [0.03, 0.1, 0.3]:
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                "gradient probe coefficient trials differ"
            )
        for trial in trials:
            if (
                not _finite_positive(trial.get("scaled_gradient_l2_norm"))
                or not _finite_positive(
                    trial.get(
                        "predicted_informative_preserving_probability_delta"
                    )
                )
                or trial.get("realized_informative_preserving_probability_delta")
                != 0.0
                or trial.get("all_safe_max_probability_delta") != 0.0
            ):
                raise MalomPolicyAuxiliaryCalibrationReadinessError(
                    "gradient probe coefficient evidence differs"
                )
        all_baseline = baseline["all"]
        if (
            all_baseline.get("states") != 64
            or all_baseline.get("informative_states") != 29
            or all_baseline.get("all_safe_states") != 35
        ):
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                "gradient probe baseline support differs"
            )
        gradients.append(float(gradient["l2_norm"]))
        derivatives.append(float(alignment["directional_derivative"]))
        cosines.append(float(alignment["descent_cosine"]))
        probabilities.append(
            float(all_baseline["mean_informative_preserving_probability"])
        )
        losses.append(float(gradient_probe["auxiliary_loss"]))
        if index > 0:
            warm_label_seconds.append(float(coverage["label_seconds"]))

    result = manifest.get("result")
    expected_result = {
        "all_safe_states": 35,
        "corpus_states_per_seed": 64,
        "downgrading_actions_per_seed": 415,
        "fresh_seeds": [48, 49, 50],
        "gradient_descent_cosine_range": [min(cosines), max(cosines)],
        "gradient_directional_derivative_range": [
            min(derivatives),
            max(derivatives),
        ],
        "gradient_l2_norm_range": [min(gradients), max(gradients)],
        "informative_states": 29,
        "labelled_actions_per_seed": 1583,
        "mean_informative_preserving_probability_range": [
            min(probabilities),
            max(probabilities),
        ],
        "mean_preserving_set_loss_range": [min(losses), max(losses)],
        "mutation_checks_passed": True,
        "preserving_actions_per_seed": 1168,
        "warm_label_seconds_range": [
            min(warm_label_seconds),
            max(warm_label_seconds),
        ],
    }
    if result != expected_result:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "gradient evidence summary does not reconcile"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": spec["sha256"],
        "probe_path": str(probe_path),
        "probe_sha256": spec["probe_sha256"],
        "probe_identity": probe_identity,
        "probe_source_commit": spec["probe_source_commit"],
        "decision": dict(decision),
        "result": dict(result),
    }


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
    diagnostic = contract["analysis"]["fixed_development_diagnostic"]
    expected_scalars = {
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
    for field, expected in expected_scalars.items():
        if getattr(plan, field) != expected:
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"managed plan field differs for {arm['arm_id']}: {field}"
            )
    if plan.game_bound != resources["completed_games_per_arm"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            f"managed completion bound differs for {arm['arm_id']}"
        )
    if plan.paths_config_sha256 != _sha256_file(paths_config):
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "managed path registry hash differs"
        )
    args = _trainer_args(plan)
    expected_args = {
        "experiment_id": arm["experiment_id"],
        "seed": arm["seed"],
        "mill_bonus_mode": arm["mill_bonus_mode"],
        "malom_policy_aux_coef": arm["malom_policy_aux_coef"],
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
    for field, expected in expected_args.items():
        if getattr(args, field) != expected:
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"trainer argument differs for {arm['arm_id']}: {field}"
            )
    if list(args.sanmill_node_ladder) != common["sanmill_node_ladder"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "Sanmill node ladder differs"
        )
    if list(args.sanmill_stage_games) != common["fixed_resource_stage_games"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "Sanmill stage durations differ"
        )
    specialist_db = _repository_path(
        root, arm["specialist_db"], field="specialist_db"
    )
    if Path(args.specialist_db).resolve() != specialist_db:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "arm SpecialistDB path differs"
        )
    gate = plan.policy_health
    if gate is None or gate.device != diagnostic["device"]:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "policy-health device differs"
        )
    expected_corpus = _repository_path(
        root, diagnostic["corpus"], field="policy-health corpus"
    )
    expected_audit = _repository_path(
        root, diagnostic["audit_script"], field="policy-health audit"
    )
    if (
        Path(gate.corpus_path).resolve() != expected_corpus
        or gate.corpus_sha256 != diagnostic["corpus_sha256"]
        or Path(gate.audit_script_path).resolve() != expected_audit
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
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "policy-health contract differs"
        )
    return args


def _normalised_training_semantics(args: Any) -> str:
    ignored = {
        "experiment_id",
        "malom_policy_aux_coef",
        "specialist_db",
    }
    value = {
        key: item
        for key, item in vars(args).items()
        if not key.startswith("_") and key not in ignored
    }
    return canonical_sha256(value)


def audit_prepared_plans(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
    preflight_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Verify exact managed plans and one-factor equivalence."""
    template = contract["data_contract"]["specialist_db_initial_template"]
    audited: list[dict[str, Any]] = []
    normalised: set[str] = set()
    for arm in _ordered_arms(contract):
        control_dir = _repository_path(
            root, arm["control_dir"], field="control_dir"
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
            root, arm["specialist_db"], field="specialist_db"
        )
        database = _inspect_specialist_database(specialist_db, template)
        if (control_dir / "authorization.json").exists():
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"arm is already authorized: {arm['arm_id']}"
            )
        if (control_dir / "segments").exists():
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"arm already has segment output: {arm['arm_id']}"
            )
        normalised.add(_normalised_training_semantics(args))
        record = {
            "arm_id": arm["arm_id"],
            "launch_order": arm["launch_order"],
            "malom_policy_aux_coef": arm["malom_policy_aux_coef"],
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
    if len(normalised) != 1:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            "arms differ outside the permitted auxiliary coefficient"
        )
    return audited


def prepare_calibration(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create four plans and read-only preflights without authorization."""
    contract = load_calibration_contract(contract_path)
    source = inspect_published_source(root, contract)
    gradient_evidence = inspect_gradient_evidence(
        root, contract, source=source
    )
    template_record = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    assert_preparation_outputs_ignored(
        root, contract, report_path=report_path
    )
    assert_preparation_targets_absent(
        root, contract, report_path=report_path
    )
    template_path = Path(template_record["path"])
    commands = build_prepare_commands(
        root=root,
        contract=contract,
        paths_config=paths_config,
        python_executable=python_executable,
    )
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
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"manager output is not JSON: {arm['arm_id']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
                f"manager state differs: {arm['arm_id']}"
            )
        plan_path = _repository_path(
            root, arm["control_dir"], field="control_dir"
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
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
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
            raise MalomPolicyAuxiliaryCalibrationReadinessError(
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
    report_body = {
        "schema_version": READINESS_SCHEMA,
        "state": "ready_for_product_authorization",
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "source": source,
        "gradient_evidence": gradient_evidence,
        "template": template_record,
        "runtime": runtime,
        "commands": commands,
        "arms": audited,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": [PRODUCT_AUTHORIZATION_DECISION],
    }
    report = {
        **report_body,
        "readiness_identity": canonical_sha256(report_body),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise MalomPolicyAuxiliaryCalibrationReadinessError(
            f"readiness report already exists: {report_path}"
        ) from exc
    return report
