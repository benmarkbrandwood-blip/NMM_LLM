"""Prepare equal-transition treatment arms from one completed shared prefix."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_ai.training.checkpoint_envelope import (
    CheckpointEnvelope,
    load_checkpoint,
)
from learned_ai.training.generalist_preflight import (
    resolved_resume_config,
    resume_config_sha256,
)
from learned_ai.training.managed_generalist import (
    ManagedPlan,
    load_managed_plan,
    managed_status,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.training.target_refresh_branch import (
    publish_target_refresh_branch_checkpoint,
)
from learned_ai.training.training_identity import (
    experiment_digest,
    load_trainer_ruleset,
)
from learned_ai.validation.mill_bonus_ablation_readiness import (
    PRODUCT_AUTHORIZATION_DECISION,
    _run_checked,
)
from learned_ai.validation.target_refresh_equal_transition_diagnostic import (
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    TargetRefreshEqualTransitionError,
    _assert_prefix_plan,
    _inspect_source,
    _repository_path,
    _sha256_file,
    _trainer_args,
    load_equal_transition_contract,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


ARM_READINESS_SCHEMA = "nmm.target-refresh-equal-transition-arm-readiness.v1"
BRANCH_CHECKPOINT_NAME = "initial-target-refresh-fork.pt"


@dataclass(frozen=True)
class CompletedPrefix:
    """Audited immutable inputs shared by the two same-seed arms."""

    prefix: Mapping[str, Any]
    plan: ManagedPlan
    checkpoint_path: Path
    checkpoint: CheckpointEnvelope
    specialist_db_path: Path
    specialist_db: Mapping[str, Any]
    status: Mapping[str, Any]


def _seed_prefix_and_arms(
    contract: Mapping[str, Any], seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefixes = [item for item in contract["prefixes"] if item["seed"] == seed]
    arms = sorted(
        [item for item in contract["arms"] if item["seed"] == seed],
        key=lambda item: int(item["launch_order"]),
    )
    if len(prefixes) != 1 or len(arms) != 2:
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} does not identify one prefix and two arms"
        )
    return dict(prefixes[0]), [dict(item) for item in arms]


def _inspect_closed_specialist_database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TargetRefreshEqualTransitionError(
            f"closed prefix SpecialistDB is absent: {path}"
        )
    sidecars = [
        Path(f"{path}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
        if Path(f"{path}{suffix}").exists()
    ]
    if sidecars:
        raise TargetRefreshEqualTransitionError(
            "closed prefix SpecialistDB has sidecars: "
            + ", ".join(item.name for item in sidecars)
        )
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        metadata = dict(connection.execute("SELECT key, value FROM meta"))
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("positions", "winning_lines", "preferred_plays")
        }
    except sqlite3.Error as exc:
        raise TargetRefreshEqualTransitionError(
            f"closed prefix SpecialistDB audit failed: {path}"
        ) from exc
    finally:
        if "connection" in locals():
            connection.close()
    if quick_check != ("ok",):
        raise TargetRefreshEqualTransitionError(
            "closed prefix SpecialistDB quick_check differs"
        )
    if metadata.get("malom_label_version") != "sector-corrected-v1":
        raise TargetRefreshEqualTransitionError(
            "closed prefix SpecialistDB label version differs"
        )
    return {
        "path": str(path.resolve()),
        "byte_length": path.stat().st_size,
        "sha256": _sha256_file(path),
        "quick_check": "ok",
        "label_version": "sector-corrected-v1",
        "counts": counts,
        "sidecars": [],
    }


def inspect_completed_prefix(
    *,
    root: Path,
    contract: Mapping[str, Any],
    seed: int,
    paths_config: Path,
    source_commit: str,
) -> CompletedPrefix:
    """Fail closed unless the shared prefix reached its frozen fork."""
    prefix, _arms = _seed_prefix_and_arms(contract, seed)
    fork_game = int(contract["common_training_contract"]["target_refresh_fork_game"])
    control_dir = _repository_path(
        root, prefix["control_dir"], field="prefix control"
    )
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    if not plan_path.is_file() or not authorization_path.is_file():
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} prefix plan or authorization is absent"
        )
    plan = load_managed_plan(plan_path)
    _assert_prefix_plan(
        plan,
        root=root,
        contract=contract,
        prefix=prefix,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    status = managed_status(plan_path, authorization_path)
    progress = status.get("progress", {})
    if (
        status.get("state") != "completed"
        or progress.get("completed_games") != fork_game
        or progress.get("completed_segments") != 1
        or status.get("technical", {}).get("authorization_error") is not None
    ):
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} prefix is not one clean completed segment"
        )
    if (control_dir / "controller.lock").exists():
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} prefix still has a controller lock"
        )
    segment_root = control_dir / "segments"
    segment_dirs = sorted(
        item.name for item in segment_root.glob("segment-*") if item.is_dir()
    )
    if segment_dirs != ["segment-0001"]:
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} prefix segment set differs"
        )
    checkpoint_path = _repository_path(
        root,
        next(
            arm["resume_checkpoint"]
            for arm in contract["arms"]
            if arm["seed"] == seed
        ),
        field="prefix fork checkpoint",
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    descriptor = checkpoint.descriptor
    fork_state = checkpoint.payload.trainer_state["recovery_state"].get(
        "target_refresh_fork_state"
    )
    expected_fork = {
        "schema_version": "nmm.target-refresh-fork-state.v1",
        "fork_game": fork_game,
        "captured": True,
        "treatment": None,
        "post_fork_transition_origin": None,
    }
    if (
        descriptor.role != "target_refresh_fork"
        or descriptor.experiment_id != prefix["experiment_id"]
        or descriptor.config_sha256 != plan.resume_config_sha256
        or checkpoint.payload.trainer_state["game_count"] != fork_game
        or fork_state != expected_fork
    ):
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} prefix fork checkpoint differs"
        )
    specialist_db_path = _repository_path(
        root, prefix["specialist_db"], field="prefix SpecialistDB"
    )
    specialist_db = _inspect_closed_specialist_database(specialist_db_path)
    payload_specialist = checkpoint.payload.data_state["mutable_assets"].get(
        "specialist_db", {}
    )
    if (
        descriptor.asset_identities.get("specialist_db")
        != specialist_db["sha256"]
        or payload_specialist.get("sha256") != specialist_db["sha256"]
    ):
        raise TargetRefreshEqualTransitionError(
            f"seed {seed} fork and closed SpecialistDB identities differ"
        )
    return CompletedPrefix(
        prefix=prefix,
        plan=plan,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        specialist_db_path=specialist_db_path,
        specialist_db=specialist_db,
        status=status,
    )


def _branch_checkpoint_path(root: Path, arm: Mapping[str, Any]) -> Path:
    control = _repository_path(root, arm["control_dir"], field="arm control")
    return control / BRANCH_CHECKPOINT_NAME


def build_arm_prepare_command(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    paths_config: Path,
    branch_checkpoint: Path,
    python_executable: str,
) -> list[str]:
    """Build one no-launch manager command for a completed-prefix arm."""
    common = contract["common_training_contract"]
    resources = contract["resources"]
    command = [
        python_executable,
        str(root / "scripts/manage_generalist_run.py"),
        "prepare",
        "--control-dir",
        str(_repository_path(root, arm["control_dir"], field="arm control")),
        "--max-wall-hours",
        str(resources["arm_active_wall_hours"]),
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
        str(resources["arm_absolute_game_count_ceiling"]),
        "--segment-games",
        str(resources["arm_post_fork_game_execution_ceiling"]),
        "--initial-resume-checkpoint",
        str(branch_checkpoint),
        "--initial-resume-completed-games",
        str(common["target_refresh_fork_game"]),
        "--no-exact-resume",
        "--engine-profile",
        "sanmill-fixed-resource",
        "--self-play-ratio",
        str(common["frozen_target_ratio"]),
        "--target-refresh-every",
        str(common["target_refresh_fork_game"]),
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
        str(_repository_path(root, arm["specialist_db"], field="arm database")),
        "--exact-transition-batches",
        "--target-refresh-fork-game",
        str(common["target_refresh_fork_game"]),
        "--target-refresh-fork-treatment",
        arm["condition"],
        "--post-fork-transition-bound",
        str(resources["scientific_post_fork_transitions_per_arm"]),
    ]
    if common.get("temperature_schedule_axis", "global-games") != "global-games":
        command.extend(
            (
                "--temperature-schedule-axis",
                str(common["temperature_schedule_axis"]),
                "--post-fork-temperature-anneal-transitions",
                str(common["post_fork_temperature_anneal_transitions"]),
            )
        )
    return command


def build_seed_arm_prepare_commands(
    *,
    root: Path,
    contract: Mapping[str, Any],
    seed: int,
    paths_config: Path,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    _prefix, arms = _seed_prefix_and_arms(contract, seed)
    return [
        build_arm_prepare_command(
            root=root,
            contract=contract,
            arm=arm,
            paths_config=paths_config,
            branch_checkpoint=_branch_checkpoint_path(root, arm),
            python_executable=python_executable,
        )
        for arm in arms
    ]


def prospective_arm_trainer_args(
    command: Sequence[str], *, paths_config: Path
) -> Any:
    """Resolve arm semantics without requiring its branch file to exist."""
    parsed = manager._build_parser().parse_args(list(command)[2:])
    common_args = manager._common_trainer_args(parsed, paths_config)
    args = trainer._build_argument_parser().parse_args(
        [
            "--preflight",
            "long-run",
            *common_args,
            "--start-mode",
            "exact-resume",
            "--resume",
            str(Path(parsed.initial_resume_checkpoint).resolve(strict=False)),
            "--parent-run-id",
            "prospective-prefix-run",
        ]
    )
    trainer._configure_paths(args)
    trainer.validate_generalist_configuration(args)
    return args


def _immutable_experiment_assets(
    checkpoint: CheckpointEnvelope,
) -> dict[str, str]:
    available = checkpoint.descriptor.asset_identities
    required = ("malom_tablebase", "human_db", "sanmill_training_runtime")
    if any(not available.get(name) for name in required):
        raise TargetRefreshEqualTransitionError(
            "prefix checkpoint lacks an immutable experiment asset"
        )
    return {
        name: str(available[name])
        for name in (
            "malom_tablebase",
            "human_db",
            "opening_forcing_sources",
            "sanmill_training_runtime",
        )
        if name in available
    }


def _target_experiment_digest(
    *,
    root: Path,
    arm: Mapping[str, Any],
    source_commit: str,
    target_resume_config_sha256: str,
    checkpoint: CheckpointEnvelope,
) -> str:
    ruleset = load_trainer_ruleset(
        root / "data/rulesets/nmm-training-core@2.json"
    )
    return experiment_digest(
        experiment_id=str(arm["experiment_id"]),
        git_commit=source_commit,
        resume_config_sha256=target_resume_config_sha256,
        immutable_assets=_immutable_experiment_assets(checkpoint),
        ruleset=ruleset,
    )


def _normalised_pair_semantics(args: Any) -> str:
    semantics = resolved_resume_config(args)
    semantics["specialist_db"] = "<same-seed-byte-identical-clone>"
    return canonical_sha256(semantics)


def _assert_arm_plan(
    plan: ManagedPlan,
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
    branch_checkpoint: Path,
    source_checkpoint: CheckpointEnvelope,
) -> Any:
    resources = contract["resources"]
    common = contract["common_training_contract"]
    expected = {
        "plan_id": arm["plan_id"],
        "objective": f"{contract['objective']}; arm={arm['arm_id']}",
        "experiment_id": arm["experiment_id"],
        "git_commit": source_commit,
        "control_dir": str(
            _repository_path(root, arm["control_dir"], field="arm control")
        ),
        "paths_config": str(paths_config),
        "max_games": common["max_games_schedule"],
        "completion_game_bound": resources["arm_absolute_game_count_ceiling"],
        "segment_games": resources["arm_post_fork_game_execution_ceiling"],
        "max_wall_hours": resources["arm_active_wall_hours"],
        "allow_safe_exact_resume": False,
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            raise TargetRefreshEqualTransitionError(
                f"arm plan field differs: {arm['arm_id']}:{field}"
            )
    initial = plan.initial_resume
    branch = load_checkpoint(branch_checkpoint, map_location="cpu")
    if (
        initial is None
        or Path(initial.checkpoint_path).resolve() != branch_checkpoint.resolve()
        or initial.checkpoint_sha256 != _sha256_file(branch_checkpoint)
        or initial.checkpoint_id != branch.descriptor.checkpoint_id
        or initial.checkpoint_role != "target_refresh_fork"
        or initial.parent_run_id != source_checkpoint.descriptor.run_id
        or initial.completed_games != common["target_refresh_fork_game"]
        or branch.payload_sha256 != source_checkpoint.payload_sha256
    ):
        raise TargetRefreshEqualTransitionError(
            f"arm initial resume differs: {arm['arm_id']}"
        )
    args = _trainer_args(plan)
    expected_args = {
        "seed": arm["seed"],
        "target_refresh_fork_game": common["target_refresh_fork_game"],
        "target_refresh_fork_treatment": arm["condition"],
        "post_fork_transition_bound": resources[
            "scientific_post_fork_transitions_per_arm"
        ],
        "exact_transition_batches": True,
        "lr_adaptation_mode": "fixed",
        "update_target_every": common["target_refresh_fork_game"],
        "specialist_read_mode": "theoretical-only",
        "mill_bonus_mode": "malom-preserving-only",
        "malom_policy_aux_coef": 0.0,
        "referee_engine": "sanmill",
        "opponent_engine": "sanmill",
        "no_recovery": True,
        "no_sentinel": True,
        "no_value_net": True,
        "no_gap_net": True,
        "no_s1a_warmstart": True,
        "no_imitation_mix": True,
        "no_s1b_refresher": True,
        "no_opening_forcing": True,
        "ppo": False,
        "temperature_schedule_axis": common.get(
            "temperature_schedule_axis", "global-games"
        ),
        "post_fork_temperature_anneal_transitions": common.get(
            "post_fork_temperature_anneal_transitions"
        ),
    }
    for field, value in expected_args.items():
        if getattr(args, field) != value:
            raise TargetRefreshEqualTransitionError(
                f"arm trainer argument differs: {arm['arm_id']}:{field}"
            )
    expected_db = _repository_path(
        root, arm["specialist_db"], field="arm database"
    )
    if Path(args.specialist_db).resolve() != expected_db:
        raise TargetRefreshEqualTransitionError(
            f"arm database path differs: {arm['arm_id']}"
        )
    return args


def _build_resume_preflight_command(
    *,
    root: Path,
    plan: ManagedPlan,
    branch_checkpoint: Path,
    python_executable: str,
) -> list[str]:
    initial = plan.initial_resume
    if initial is None:
        raise TargetRefreshEqualTransitionError("arm initial resume is absent")
    output = Path(plan.control_dir) / "segments" / "segment-0001"
    return [
        python_executable,
        str(root / "scripts/train_s_gen_v2.py"),
        "--preflight",
        "long-run",
        "--run-id",
        f"{plan.plan_id}-segment-0001",
        "--out-dir",
        str(output),
        "--segment-games",
        str(plan.segment_games),
        "--segment-stop-game",
        str(plan.game_bound),
        *plan.common_trainer_args,
        "--start-mode",
        "exact-resume",
        "--resume",
        str(branch_checkpoint),
        "--parent-run-id",
        initial.parent_run_id,
    ]


def _validate_resume_preflight(
    preflight: Mapping[str, Any],
    *,
    plan: ManagedPlan,
    arm: Mapping[str, Any],
    source_commit: str,
    branch_checkpoint: Path,
) -> None:
    if (
        preflight.get("schema_version") != "nmm.generalist-preflight.v1"
        or preflight.get("mode") != "long-run"
        or preflight.get("verdict") != "needs_decision"
        or preflight.get("errors") != []
        or preflight.get("unresolved_decisions")
        != [PRODUCT_AUTHORIZATION_DECISION]
        or preflight.get("resume_config_sha256") != plan.resume_config_sha256
    ):
        raise TargetRefreshEqualTransitionError(
            f"arm preflight verdict differs: {arm['arm_id']}"
        )
    git = preflight.get("git", {})
    config = preflight.get("resolved_config", {})
    initial = plan.initial_resume
    expected = {
        "experiment_id": plan.experiment_id,
        "run_id": f"{plan.plan_id}-segment-0001",
        "segment_games": plan.segment_games,
        "segment_stop_game": plan.game_bound,
        "start_mode": "exact-resume",
        "resume": str(branch_checkpoint),
        "parent_run_id": None if initial is None else initial.parent_run_id,
    }
    if (
        git.get("commit") != source_commit
        or git.get("dirty") is not False
        or not isinstance(config, Mapping)
        or any(config.get(key) != value for key, value in expected.items())
    ):
        raise TargetRefreshEqualTransitionError(
            f"arm preflight identity differs: {arm['arm_id']}"
        )
    expected_output = (
        Path(plan.control_dir) / "segments" / "segment-0001"
    ).resolve(strict=False)
    if Path(str(config.get("out_dir", ""))).resolve(strict=False) != expected_output:
        raise TargetRefreshEqualTransitionError(
            f"arm preflight output differs: {arm['arm_id']}"
        )
    output = preflight.get("checks", {}).get("output")
    if output != {"exists": False, "isolated": True, "kind": "run_directory"}:
        raise TargetRefreshEqualTransitionError(
            f"arm preflight output isolation differs: {arm['arm_id']}"
        )
    branch = load_checkpoint(branch_checkpoint, map_location="cpu")
    if branch.descriptor.implementation.get("experiment_digest") != preflight.get(
        "experimentDigest"
    ):
        raise TargetRefreshEqualTransitionError(
            f"arm checkpoint experiment digest differs: {arm['arm_id']}"
        )


def _assert_arm_targets_absent(
    root: Path,
    arms: Sequence[Mapping[str, Any]],
    *,
    report_path: Path,
) -> None:
    targets = [report_path]
    for arm in arms:
        targets.extend(
            (
                _repository_path(root, arm["control_dir"], field="arm control"),
                _repository_path(root, arm["specialist_db"], field="arm database"),
            )
        )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise TargetRefreshEqualTransitionError(
            "arm preparation targets already exist: " + ", ".join(existing)
        )


def _assert_arm_outputs_ignored(
    root: Path,
    arms: Sequence[Mapping[str, Any]],
    *,
    report_path: Path,
) -> None:
    for path in [report_path] + [
        item
        for arm in arms
        for item in (
            _repository_path(root, arm["control_dir"], field="arm control"),
            _repository_path(root, arm["specialist_db"], field="arm database"),
        )
    ]:
        relative = path.relative_to(root.resolve()).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise TargetRefreshEqualTransitionError(
                f"arm preparation output is not ignored: {relative}"
            )


def prepare_seed_arms(
    *,
    root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    paths_config: Path = DEFAULT_PATHS_CONFIG,
    seed: int,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Prepare two immutable same-seed plans without authorizing training."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    report_path = report_path.resolve(strict=False)
    contract = load_equal_transition_contract(contract_path)
    source = _inspect_source(root, contract)
    if not source["published"]:
        raise TargetRefreshEqualTransitionError("dev must equal origin/dev")
    prefix, arms = _seed_prefix_and_arms(contract, seed)
    _assert_arm_outputs_ignored(root, arms, report_path=report_path)
    _assert_arm_targets_absent(root, arms, report_path=report_path)
    completed = inspect_completed_prefix(
        root=root,
        contract=contract,
        seed=seed,
        paths_config=paths_config,
        source_commit=source["head"],
    )
    commands = build_seed_arm_prepare_commands(
        root=root,
        contract=contract,
        seed=seed,
        paths_config=paths_config,
        python_executable=python_executable,
    )
    arm_records: list[dict[str, Any]] = []
    normalised_semantics: set[str] = set()
    for arm, command in zip(arms, commands, strict=True):
        specialist_db = _repository_path(
            root, arm["specialist_db"], field="arm database"
        )
        specialist_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(completed.specialist_db_path, specialist_db)
        clone = _inspect_closed_specialist_database(specialist_db)
        if clone != {
            **dict(completed.specialist_db),
            "path": str(specialist_db.resolve()),
        }:
            raise TargetRefreshEqualTransitionError(
                f"arm SpecialistDB clone differs: {arm['arm_id']}"
            )

        prospective_args = prospective_arm_trainer_args(
            command, paths_config=paths_config
        )
        target_config = resume_config_sha256(prospective_args)
        target_digest = _target_experiment_digest(
            root=root,
            arm=arm,
            source_commit=source["head"],
            target_resume_config_sha256=target_config,
            checkpoint=completed.checkpoint,
        )
        branch_checkpoint = _branch_checkpoint_path(root, arm)
        branch_record = publish_target_refresh_branch_checkpoint(
            completed.checkpoint_path,
            branch_checkpoint,
            treatment=arm["condition"],
            expected_source_config_sha256=completed.plan.resume_config_sha256,
            target_resume_config_sha256=target_config,
            expected_experiment_id=arm["experiment_id"],
            expected_game_count=contract["common_training_contract"][
                "target_refresh_fork_game"
            ],
            expected_specialist_db_sha256=clone["sha256"],
            target_experiment_digest=target_digest,
        )
        manager_result = _run_checked(command, root=root, runner=runner)
        try:
            manager_output = json.loads(manager_result.stdout)
        except json.JSONDecodeError as exc:
            raise TargetRefreshEqualTransitionError(
                f"arm manager output is not JSON: {arm['arm_id']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise TargetRefreshEqualTransitionError(
                f"arm manager state differs: {arm['arm_id']}"
            )
        control_dir = _repository_path(
            root, arm["control_dir"], field="arm control"
        )
        plan_path = control_dir / "plan.json"
        plan = load_managed_plan(plan_path)
        plan_args = _assert_arm_plan(
            plan,
            root=root,
            contract=contract,
            arm=arm,
            paths_config=paths_config,
            source_commit=source["head"],
            branch_checkpoint=branch_checkpoint,
            source_checkpoint=completed.checkpoint,
        )
        if plan.resume_config_sha256 != target_config:
            raise TargetRefreshEqualTransitionError(
                f"arm resume identity differs: {arm['arm_id']}"
            )
        normalised_semantics.add(_normalised_pair_semantics(plan_args))
        preflight_command = _build_resume_preflight_command(
            root=root,
            plan=plan,
            branch_checkpoint=branch_checkpoint,
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
            raise TargetRefreshEqualTransitionError(
                f"arm preflight is not JSON: {arm['arm_id']}"
            ) from exc
        _validate_resume_preflight(
            preflight,
            plan=plan,
            arm=arm,
            source_commit=source["head"],
            branch_checkpoint=branch_checkpoint,
        )
        preflight_path = control_dir / "preflight.json"
        try:
            with preflight_path.open("xb") as handle:
                handle.write(canonical_json_bytes(preflight))
        except FileExistsError as exc:
            raise TargetRefreshEqualTransitionError(
                f"arm preflight already exists: {arm['arm_id']}"
            ) from exc
        if (control_dir / "authorization.json").exists() or (
            control_dir / "segments"
        ).exists():
            raise TargetRefreshEqualTransitionError(
                f"arm preparation created runnable state: {arm['arm_id']}"
            )
        arm_records.append(
            {
                "arm_id": arm["arm_id"],
                "launch_order": arm["launch_order"],
                "condition": arm["condition"],
                "plan_path": str(plan_path),
                "plan_sha256": plan.plan_sha256,
                "resume_config_sha256": plan.resume_config_sha256,
                "branch_checkpoint": branch_record,
                "specialist_db": clone,
                "preflight": {
                    "path": str(preflight_path),
                    "sha256": _sha256_file(preflight_path),
                    "verdict": preflight["verdict"],
                },
                "authorization_present": False,
                "segment_output_present": False,
            }
        )
    if len(normalised_semantics) != 1:
        raise TargetRefreshEqualTransitionError(
            "same-seed arm semantics differ outside isolated identities"
        )
    payloads = {
        item["branch_checkpoint"]["branch_payload_sha256"]
        for item in arm_records
    }
    clone_hashes = {item["specialist_db"]["sha256"] for item in arm_records}
    if payloads != {completed.checkpoint.payload_sha256} or clone_hashes != {
        completed.specialist_db["sha256"]
    }:
        raise TargetRefreshEqualTransitionError(
            "same-seed arm branch inputs are not identical"
        )
    body = {
        "schema_version": ARM_READINESS_SCHEMA,
        "state": "seed_arm_plans_ready_for_product_authorization",
        "verdict": "needs_decision",
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256_file(contract_path),
            "plan_identity": contract["plan_identity"],
        },
        "source": source,
        "seed": seed,
        "prefix": {
            "launch_order": prefix["launch_order"],
            "plan_sha256": completed.plan.plan_sha256,
            "checkpoint_path": str(completed.checkpoint_path),
            "checkpoint_sha256": _sha256_file(completed.checkpoint_path),
            "checkpoint_id": completed.checkpoint.descriptor.checkpoint_id,
            "payload_sha256": completed.checkpoint.payload_sha256,
            "specialist_db": dict(completed.specialist_db),
            "managed_status": dict(completed.status),
        },
        "arms": arm_records,
        "pairing": {
            "byte_identical_source_payload": True,
            "byte_identical_specialist_db_clones": True,
            "normalised_resume_semantics_identity": next(
                iter(normalised_semantics)
            ),
        },
        "resource_envelope": {
            "maximum_games_per_arm_after_prefix": contract["resources"][
                "arm_post_fork_game_execution_ceiling"
            ],
            "absolute_game_ceiling": contract["resources"][
                "arm_absolute_game_count_ceiling"
            ],
            "post_fork_transition_bound": contract["resources"][
                "scientific_post_fork_transitions_per_arm"
            ],
            "maximum_active_wall_hours_per_arm": contract["resources"][
                "arm_active_wall_hours"
            ],
        },
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": [
            PRODUCT_AUTHORIZATION_DECISION,
            (
                f"authorize only launch order {arms[0]['launch_order']} "
                f"({arms[0]['arm_id']}) next"
            ),
        ],
    }
    report = {**body, "readiness_identity": canonical_sha256(body)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise TargetRefreshEqualTransitionError(
            "arm readiness report already exists"
        ) from exc
    return report


__all__ = [
    "ARM_READINESS_SCHEMA",
    "BRANCH_CHECKPOINT_NAME",
    "CompletedPrefix",
    "build_seed_arm_prepare_commands",
    "inspect_completed_prefix",
    "prepare_seed_arms",
    "prospective_arm_trainer_args",
]
