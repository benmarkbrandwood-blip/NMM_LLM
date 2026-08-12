"""Prepare a paired mature-boundary target-refresh diagnostic."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from learned_ai.training.checkpoint_envelope import load_checkpoint
from learned_ai.training.generalist_preflight import (
    resolved_resume_config,
    resume_config_sha256,
)
from learned_ai.training.managed_generalist import load_managed_plan
from learned_ai.training.mature_target_refresh_fork import (
    publish_mature_target_refresh_fork,
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
from learned_ai.validation.target_refresh_equal_transition_arms import (
    _inspect_closed_specialist_database,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


PLAN_SCHEMA = "nmm.target-refresh-mature-fork-diagnostic-plan.v1"
READINESS_SCHEMA = "nmm.target-refresh-mature-fork-diagnostic-readiness.v1"
EXPECTED_SEEDS = (67, 68, 69)
EXPECTED_CONDITIONS = ("refresh-mature", "stale-control")
TRAINER_TREATMENT = {
    "refresh-mature": "refresh-once",
    "stale-control": "no-refresh",
}
DEFAULT_CONTRACT = Path(
    "docs/experiments/sanmill-target-refresh-mature-fork-diagnostic-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_READINESS = Path(
    "out/target-refresh-mature-fork-diagnostic-v1/readiness.json"
)


class MatureTargetRefreshDiagnosticError(RuntimeError):
    """Raised when the mature target-refresh diagnostic is not reproducible."""


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MatureTargetRefreshDiagnosticError(
                    f"duplicate key {key!r} in {path.name}"
                )
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        if b"\r" in raw or not raw.endswith(b"\n"):
            raise MatureTargetRefreshDiagnosticError(
                f"{path.name} is not canonical LF JSON"
            )
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatureTargetRefreshDiagnosticError(
            f"cannot read {path.name}"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise MatureTargetRefreshDiagnosticError(
            f"{path.name} is not canonical JSON"
        )
    return value


def _identity(value: Mapping[str, Any], field: str) -> str:
    observed = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    if not isinstance(observed, str) or observed != canonical_sha256(body):
        raise MatureTargetRefreshDiagnosticError(f"{field} differs")
    return observed


def _repository_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MatureTargetRefreshDiagnosticError(f"{field} path is required")
    path = Path(value)
    if path.is_absolute():
        raise MatureTargetRefreshDiagnosticError(f"{field} path must be relative")
    resolved = (root / path).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MatureTargetRefreshDiagnosticError(
            f"{field} path leaves the repository"
        ) from exc
    return resolved


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    if contract.get("schema_version") != PLAN_SCHEMA:
        raise MatureTargetRefreshDiagnosticError("plan schema differs")
    _identity(contract, "plan_identity")
    if contract.get("status") != "designed_unlaunched_needs_publication":
        raise MatureTargetRefreshDiagnosticError("plan status differs")
    sources = contract.get("sources")
    arms = contract.get("arms")
    if not isinstance(sources, list) or not isinstance(arms, list):
        raise MatureTargetRefreshDiagnosticError("plan sources or arms are absent")
    if [item.get("seed") for item in sources] != list(EXPECTED_SEEDS):
        raise MatureTargetRefreshDiagnosticError("plan source seeds differ")
    expected_arms = [
        (seed, condition)
        for seed in EXPECTED_SEEDS
        for condition in EXPECTED_CONDITIONS
    ]
    if [(item.get("seed"), item.get("condition")) for item in arms] != expected_arms:
        raise MatureTargetRefreshDiagnosticError("plan arm order differs")
    common = contract.get("common_training_contract", {})
    required_common = {
        "algorithm": "A2C",
        "exact_transition_batch_size": 64,
        "post_mature_fork_transitions_per_arm": 8_192,
        "temperature_schedule_axis": "post-fork-transitions",
        "post_fork_temperature_anneal_transitions": 98_112,
        "sanmill_node_budget": 1_000,
        "max_games_schedule": 5_000,
        "max_logical_plies": 120,
        "specialist_read_mode": "theoretical-only",
        "target_refresh_after_fork": "none",
    }
    if any(common.get(key) != expected for key, expected in required_common.items()):
        raise MatureTargetRefreshDiagnosticError("common training contract differs")
    temperature_origin = common.get("temperature_origin")
    if not isinstance(temperature_origin, (int, float)) or not (
        0.2 <= float(temperature_origin) <= 0.9
    ):
        raise MatureTargetRefreshDiagnosticError("temperature origin differs")
    resources = contract.get("resources", {})
    if (
        resources.get("maximum_training_games_total") != 3_600
        or resources.get("maximum_active_wall_hours_total") != 4.0
        or resources.get("maximum_training_games_per_arm") != 600
        or resources.get("maximum_no_update_games_total") != 288
    ):
        raise MatureTargetRefreshDiagnosticError("resource envelope differs")
    if contract.get("authorization", {}).get("launch_authorized") is not False:
        raise MatureTargetRefreshDiagnosticError("plan unexpectedly authorizes launch")
    return contract


def load_contract(path: Path) -> dict[str, Any]:
    return validate_contract(_strict_json(path.resolve(strict=True)))


def _source_state(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = _repository_path(
        root, source.get("checkpoint_path"), field="source checkpoint"
    ).resolve(strict=True)
    db_path = _repository_path(
        root, source.get("specialist_db_path"), field="source SpecialistDB"
    ).resolve(strict=True)
    if _sha256_file(checkpoint_path) != source.get("checkpoint_file_sha256"):
        raise MatureTargetRefreshDiagnosticError("source checkpoint file drifted")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    descriptor = checkpoint.descriptor
    trainer_state = checkpoint.payload.trainer_state
    recovery = trainer_state.get("recovery_state", {})
    prior = recovery.get("target_refresh_fork_state", {})
    expected = {
        "checkpoint_id": descriptor.checkpoint_id,
        "checkpoint_payload_sha256": checkpoint.payload_sha256,
        "checkpoint_config_sha256": descriptor.config_sha256,
        "checkpoint_experiment_id": descriptor.experiment_id,
        "checkpoint_run_id": descriptor.run_id,
        "game_count": trainer_state.get("game_count"),
        "update_count": trainer_state.get("update_count"),
        "optimizer_consumed_transitions": recovery.get(
            "optimizer_consumed_transition_count"
        ),
        "prior_post_fork_origin": prior.get("post_fork_transition_origin"),
        "pending_transition_count": len(recovery.get("pending_steps", [])),
    }
    if any(source.get(key) != observed for key, observed in expected.items()):
        raise MatureTargetRefreshDiagnosticError("source checkpoint state drifted")
    if (
        descriptor.role != "transition_diagnostic_candidate"
        or prior.get("treatment") != "no-refresh"
        or expected["optimizer_consumed_transitions"]
        - expected["prior_post_fork_origin"]
        != 8_192
    ):
        raise MatureTargetRefreshDiagnosticError("source boundary is incompatible")
    database = _inspect_closed_specialist_database(db_path)
    if (
        database["sha256"] != source.get("specialist_db_sha256")
        or database["byte_length"] != source.get("specialist_db_bytes")
        or descriptor.asset_identities.get("specialist_db") != database["sha256"]
    ):
        raise MatureTargetRefreshDiagnosticError("source SpecialistDB drifted")
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "specialist_db_path": db_path,
        "specialist_db": database,
    }


def _arm_for(
    contract: Mapping[str, Any], seed: int, condition: str
) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in contract["arms"]
        if item["seed"] == seed and item["condition"] == condition
    ]
    if len(matches) != 1:
        raise MatureTargetRefreshDiagnosticError("arm identity is ambiguous")
    return matches[0]


def build_arm_prepare_command(
    *,
    root: Path,
    contract: Mapping[str, Any],
    source: Mapping[str, Any],
    arm: Mapping[str, Any],
    branch_checkpoint: Path,
    paths_config: Path,
    python_executable: str,
) -> list[str]:
    common = contract["common_training_contract"]
    resources = contract["resources"]
    game_count = int(source["game_count"])
    return [
        python_executable,
        str(root / "scripts/manage_generalist_run.py"),
        "prepare",
        "--control-dir",
        str(_repository_path(root, arm["control_dir"], field="arm control")),
        "--max-wall-hours",
        str(resources["maximum_active_wall_hours_per_arm"]),
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
        str(game_count + resources["maximum_training_games_per_arm"]),
        "--segment-games",
        str(resources["maximum_training_games_per_arm"]),
        "--initial-resume-checkpoint",
        str(branch_checkpoint),
        "--initial-resume-completed-games",
        str(game_count),
        "--no-exact-resume",
        "--engine-profile",
        "sanmill-fixed-resource",
        "--self-play-ratio",
        str(common["frozen_target_ratio"]),
        "--target-refresh-every",
        str(game_count),
        "--lr-adaptation-mode",
        "fixed",
        "--sanmill-node-ladder",
        str(common["sanmill_node_budget"]),
        "--sanmill-stage-games",
        str(common["max_games_schedule"]),
        "--max-ply",
        str(common["max_logical_plies"]),
        "--mill-bonus-mode",
        "malom-preserving-only",
        "--malom-policy-aux-coef",
        "0",
        "--malom-policy-aux-mode",
        "fixed",
        "--specialist-read-mode",
        common["specialist_read_mode"],
        "--specialist-db",
        str(_repository_path(root, arm["specialist_db"], field="arm database")),
        "--policy-health-gate",
        "--policy-health-device",
        "auto",
        "--exact-transition-batches",
        "--target-refresh-fork-game",
        str(game_count),
        "--target-refresh-fork-treatment",
        TRAINER_TREATMENT[arm["condition"]],
        "--post-fork-transition-bound",
        str(common["post_mature_fork_transitions_per_arm"]),
        "--temperature-schedule-axis",
        common["temperature_schedule_axis"],
        "--post-fork-temperature-anneal-transitions",
        str(common["post_fork_temperature_anneal_transitions"]),
        "--post-fork-temperature-origin",
        repr(float(common["temperature_origin"])),
    ]


def _prospective_args(command: Sequence[str], paths_config: Path) -> Any:
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
            "prospective-mature-fork",
        ]
    )
    trainer._configure_paths(args)
    trainer.validate_generalist_configuration(args)
    return args


def _immutable_assets(checkpoint: Any) -> dict[str, str]:
    available = checkpoint.descriptor.asset_identities
    required = ("malom_tablebase", "human_db", "sanmill_training_runtime")
    if any(not available.get(name) for name in required):
        raise MatureTargetRefreshDiagnosticError(
            "source checkpoint lacks an immutable experiment asset"
        )
    return {name: str(available[name]) for name in required}


def _validated_policy_health_gate(plan: Any) -> dict[str, Any]:
    gate = plan.policy_health
    expected_corpus = manager.DEFAULT_POLICY_HEALTH_CORPUS.resolve(strict=True)
    expected_audit = manager.DEFAULT_POLICY_HEALTH_AUDIT.resolve(strict=True)
    if gate is None:
        raise MatureTargetRefreshDiagnosticError(
            "managed plan omitted the required policy-health gate"
        )
    observed = gate.to_dict()
    expected = {
        "schema_version": "nmm.managed-policy-health-gate.v1",
        "corpus_path": str(expected_corpus),
        "corpus_sha256": manager.DEFAULT_POLICY_HEALTH_CORPUS_SHA256,
        "audit_script_path": str(expected_audit),
        "audit_script_sha256": _sha256_file(expected_audit),
        "exact_critical_states": 29,
        "required_direct_preserving_rate": 1.0,
        "min_candidate_preserving_rate": 0.5,
        "min_candidate_logit_margin": -0.1,
        "device": "auto",
    }
    if observed != expected:
        raise MatureTargetRefreshDiagnosticError(
            "managed plan policy-health gate differs"
        )
    return observed


def _target_experiment_digest(
    *,
    root: Path,
    experiment_id: str,
    source_commit: str,
    config_sha256: str,
    checkpoint: Any,
) -> str:
    return experiment_digest(
        experiment_id=experiment_id,
        git_commit=source_commit,
        resume_config_sha256=config_sha256,
        immutable_assets=_immutable_assets(checkpoint),
        ruleset=load_trainer_ruleset(
            root / "data/rulesets/nmm-training-core@2.json"
        ),
    )


def _preflight_experiment_digest_matches(
    preflight: Mapping[str, Any], expected: str
) -> bool:
    """Compare the already-prefixed experiment identity without rewriting it."""
    return preflight.get("experimentDigest") == expected


def _git_source(root: Path, required_commit: str) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    head = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    dirty = bool(run("status", "--porcelain"))
    origin = run("rev-parse", "origin/dev")
    required_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_commit, head],
        cwd=root,
        check=False,
    ).returncode == 0
    if branch != "dev" or dirty or head != origin or not required_is_ancestor:
        raise MatureTargetRefreshDiagnosticError(
            "preparation requires clean published dev with implementation ancestry"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin,
        "dirty": False,
        "required_implementation_commit": required_commit,
        "required_implementation_is_ancestor": True,
    }


def _assert_ignored(root: Path, paths: Sequence[Path]) -> None:
    for path in paths:
        relative = path.resolve(strict=False).relative_to(root.resolve()).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative],
            cwd=root,
            check=False,
        )
        if result.returncode != 0:
            raise MatureTargetRefreshDiagnosticError(
                f"generated path is not ignored: {relative}"
            )


def _build_preflight_command(root: Path, plan: Any, branch: Path) -> list[str]:
    initial = plan.initial_resume
    return [
        sys.executable,
        str(root / "scripts/train_s_gen_v2.py"),
        "--preflight",
        "long-run",
        "--run-id",
        f"{plan.plan_id}-segment-0001",
        "--out-dir",
        str(Path(plan.control_dir) / "segments" / "segment-0001"),
        "--segment-games",
        str(plan.segment_games),
        "--segment-stop-game",
        str(plan.game_bound),
        *plan.common_trainer_args,
        "--start-mode",
        "exact-resume",
        "--resume",
        str(branch),
        "--parent-run-id",
        initial.parent_run_id,
    ]


def prepare_mature_fork_diagnostic(
    *,
    root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    paths_config: Path = DEFAULT_PATHS_CONFIG,
    readiness_path: Path = DEFAULT_READINESS,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Generate all six authorization-free plans and one parent readiness."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=False)
    contract = load_contract(contract_path)
    source_git = _git_source(
        root, contract["lineage"]["required_implementation_commit"]
    )
    generated_paths = [readiness_path]
    for source in contract["sources"]:
        generated_paths.append(
            _repository_path(root, source["common_fork_path"], field="common fork")
        )
    for arm in contract["arms"]:
        generated_paths.extend(
            (
                _repository_path(root, arm["control_dir"], field="arm control"),
                _repository_path(root, arm["specialist_db"], field="arm database"),
            )
        )
    _assert_ignored(root, generated_paths)
    existing = [str(path) for path in generated_paths if path.exists()]
    if existing:
        raise MatureTargetRefreshDiagnosticError(
            "preparation target already exists: " + ", ".join(existing)
        )

    seed_records: list[dict[str, Any]] = []
    for source in contract["sources"]:
        seed = int(source["seed"])
        inspected = _source_state(root, source)
        arms = [_arm_for(contract, seed, condition) for condition in EXPECTED_CONDITIONS]
        branch_paths = {
            arm["condition"]: _repository_path(
                root,
                f"{arm['control_dir']}/initial-mature-target-refresh-fork.pt",
                field="branch checkpoint",
            )
            for arm in arms
        }
        commands = [
            build_arm_prepare_command(
                root=root,
                contract=contract,
                source=source,
                arm=arm,
                branch_checkpoint=branch_paths[arm["condition"]],
                paths_config=paths_config,
                python_executable=python_executable,
            )
            for arm in arms
        ]
        args_by_condition = {
            arm["condition"]: _prospective_args(command, paths_config)
            for arm, command in zip(arms, commands, strict=True)
        }
        normalised = set()
        for args in args_by_condition.values():
            semantics = resolved_resume_config(args)
            semantics["specialist_db"] = "<same-seed-byte-identical-clone>"
            normalised.add(canonical_sha256(semantics))
        if len(normalised) != 1:
            raise MatureTargetRefreshDiagnosticError(
                f"seed {seed} pair differs outside allowlisted identities"
            )
        common_config = canonical_sha256(
            {
                "schema_version": "nmm.mature-target-refresh-common-fork-config.v1",
                "seed": seed,
                "normalised_pair_semantics": next(iter(normalised)),
            }
        )
        common_digest = _target_experiment_digest(
            root=root,
            experiment_id=arms[0]["experiment_id"],
            source_commit=source_git["head"],
            config_sha256=common_config,
            checkpoint=inspected["checkpoint"],
        )
        common_fork_path = _repository_path(
            root, source["common_fork_path"], field="common fork"
        )
        common_fork = publish_mature_target_refresh_fork(
            inspected["checkpoint_path"],
            common_fork_path,
            expected_source_file_sha256=source["checkpoint_file_sha256"],
            expected_source_payload_sha256=source["checkpoint_payload_sha256"],
            expected_source_config_sha256=source["checkpoint_config_sha256"],
            expected_source_experiment_id=source["checkpoint_experiment_id"],
            expected_source_game_count=source["game_count"],
            expected_source_update_count=source["update_count"],
            expected_source_post_fork_transitions=8_192,
            expected_specialist_db_sha256=source["specialist_db_sha256"],
            target_resume_config_sha256=common_config,
            target_experiment_id=arms[0]["experiment_id"],
            target_experiment_digest=common_digest,
            target_run_id=f"target-refresh-mature-fork-v1-s{seed}-common-fork",
            temperature_origin=contract["common_training_contract"][
                "temperature_origin"
            ],
        )
        arm_records: list[dict[str, Any]] = []
        for arm, command in zip(arms, commands, strict=True):
            db_path = _repository_path(
                root, arm["specialist_db"], field="arm database"
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inspected["specialist_db_path"], db_path)
            clone = _inspect_closed_specialist_database(db_path)
            if clone["sha256"] != source["specialist_db_sha256"]:
                raise MatureTargetRefreshDiagnosticError("arm database clone differs")
            args = args_by_condition[arm["condition"]]
            target_config = resume_config_sha256(args)
            target_digest = _target_experiment_digest(
                root=root,
                experiment_id=arm["experiment_id"],
                source_commit=source_git["head"],
                config_sha256=target_config,
                checkpoint=inspected["checkpoint"],
            )
            branch = publish_target_refresh_branch_checkpoint(
                common_fork_path,
                branch_paths[arm["condition"]],
                treatment=TRAINER_TREATMENT[arm["condition"]],
                expected_source_config_sha256=common_config,
                target_resume_config_sha256=target_config,
                expected_experiment_id=arm["experiment_id"],
                expected_game_count=source["game_count"],
                expected_specialist_db_sha256=clone["sha256"],
                target_experiment_digest=target_digest,
            )
            manager_result = _run_checked(command, root=root, runner=runner)
            manager_output = json.loads(manager_result.stdout)
            if manager_output.get("state") != "awaiting_product_authorization":
                raise MatureTargetRefreshDiagnosticError("manager state differs")
            control_dir = _repository_path(
                root, arm["control_dir"], field="arm control"
            )
            plan_path = control_dir / "plan.json"
            plan = load_managed_plan(plan_path)
            policy_health = _validated_policy_health_gate(plan)
            if (
                plan.resume_config_sha256 != target_config
                or plan.git_commit != source_git["head"]
                or plan.initial_resume is None
                or Path(plan.initial_resume.checkpoint_path).resolve()
                != branch_paths[arm["condition"]].resolve()
                or plan.initial_resume.completed_games != source["game_count"]
            ):
                raise MatureTargetRefreshDiagnosticError("managed plan differs")
            preflight_result = _run_checked(
                _build_preflight_command(
                    root, plan, branch_paths[arm["condition"]]
                ),
                root=root,
                runner=runner,
                accepted_return_codes=(2,),
            )
            preflight = json.loads(preflight_result.stdout)
            if (
                preflight.get("verdict") != "needs_decision"
                or preflight.get("errors") != []
                or preflight.get("unresolved_decisions")
                != [PRODUCT_AUTHORIZATION_DECISION]
                or not _preflight_experiment_digest_matches(
                    preflight, target_digest
                )
            ):
                raise MatureTargetRefreshDiagnosticError("arm preflight differs")
            preflight_path = control_dir / "preflight.json"
            preflight_path.write_bytes(canonical_json_bytes(preflight) + b"\n")
            arm_records.append(
                {
                    "arm_id": arm["arm_id"],
                    "condition": arm["condition"],
                    "launch_order": arm["launch_order"],
                    "plan_path": str(plan_path),
                    "plan_sha256": plan.plan_sha256,
                    "resume_config_sha256": target_config,
                    "branch_checkpoint": branch,
                    "specialist_db": clone,
                    "preflight": {
                        "path": str(preflight_path),
                        "sha256": _sha256_file(preflight_path),
                        "verdict": preflight["verdict"],
                    },
                    "policy_health": policy_health,
                    "authorization_present": False,
                    "segment_output_present": False,
                }
            )
        payloads = {
            item["branch_checkpoint"]["branch_payload_sha256"]
            for item in arm_records
        }
        if len(payloads) != 1:
            raise MatureTargetRefreshDiagnosticError("pair payloads differ")
        seed_records.append(
            {
                "seed": seed,
                "source": dict(source),
                "common_fork": common_fork,
                "normalised_pair_semantics_identity": next(iter(normalised)),
                "arms": arm_records,
            }
        )

    body = {
        "schema_version": READINESS_SCHEMA,
        "state": "six_arm_plans_ready_for_one_parent_product_authorization",
        "verdict": "needs_decision",
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256_file(contract_path),
            "plan_identity": contract["plan_identity"],
        },
        "source": source_git,
        "seeds": seed_records,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": [
            PRODUCT_AUTHORIZATION_DECISION,
            "authorize this six-arm sequence once in frozen launch order",
        ],
    }
    readiness = {**body, "readiness_identity": canonical_sha256(body)}
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_bytes(canonical_json_bytes(readiness) + b"\n")
    return readiness


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_READINESS",
    "EXPECTED_CONDITIONS",
    "EXPECTED_SEEDS",
    "MatureTargetRefreshDiagnosticError",
    "PLAN_SCHEMA",
    "READINESS_SCHEMA",
    "TRAINER_TREATMENT",
    "_preflight_experiment_digest_matches",
    "build_arm_prepare_command",
    "load_contract",
    "prepare_mature_fork_diagnostic",
    "validate_contract",
]
