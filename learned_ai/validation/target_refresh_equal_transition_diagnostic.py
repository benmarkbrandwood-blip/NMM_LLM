"""Prepare the staged equal-transition target-refresh diagnostic safely."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from learned_ai.training.managed_generalist import ManagedPlan, load_managed_plan
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.evaluation.target_refresh_schedule_isolation_result import (
    CANDIDATE_COLORS,
    MAX_POST_START_LOGICAL_PLIES,
    MAXIMUM_OPPOSITE_MALOM_MASS_EFFECT,
    MAXIMUM_OPPOSITE_PHASE_EFFECT,
    MAXIMUM_TRUNCATION_RATE_INCREASE,
    MINIMUM_AGGREGATE_SCORE_EFFECT,
    MINIMUM_PER_SEED_SCORE_EFFECT,
    MINIMUM_SUPPORTING_SEEDS,
    OUTCOME_BOUNDARIES,
    PRIMARY_TEMPERATURE,
)
from learned_ai.validation.mill_bonus_ablation_readiness import (
    PRODUCT_AUTHORIZATION_DECISION,
    _build_fresh_preflight_command,
    _inspect_specialist_database,
    _repository_path,
    _run_checked,
    _validate_unlaunched_preflight,
    inspect_runtime_identities,
    inspect_template,
)
from scripts import manage_generalist_run as manager
from scripts import train_s_gen_v2 as trainer


CONTRACT_SCHEMA = "nmm.target-refresh-equal-transition-diagnostic-plan.v1"
SCHEDULE_ISOLATION_CONTRACT_SCHEMA = (
    "nmm.target-refresh-equal-transition-diagnostic-plan.v2"
)
SOURCE_READINESS_SCHEMA = (
    "nmm.target-refresh-equal-transition-source-readiness.v1"
)
READINESS_SCHEMA = "nmm.target-refresh-equal-transition-readiness.v1"
DEFAULT_CONTRACT = Path(
    "docs/experiments/sanmill-target-refresh-equal-transition-diagnostic-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_SOURCE_REPORT = Path(
    "out/target-refresh-equal-transition-diagnostic-v1/source-readiness.json"
)
DEFAULT_REPORT = Path(
    "out/target-refresh-equal-transition-diagnostic-v1/readiness.json"
)
EXPECTED_SEEDS = (64, 65, 66)
EXPECTED_CONDITIONS = ("refresh-once", "no-refresh")
EXPECTED_BOUNDARIES = (1024, 2048, 4096, 8192)
EXPECTED_REPLAY_CORPUS_PATH = (
    "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
EXPECTED_REPLAY_CORPUS_SHA256 = (
    "9637efaae21074eefb4fab9e22550f5729999b30d03ed469dc88cf75aae07c2f"
)
EXPECTED_REPLAY_CORPUS_IDENTITY = (
    "ca4b410dd2913933d3ecbd8672fe274ea4a2f8ad42db3f039dabfa52af196aa4"
)
EXPECTED_REPLAY_AUDIT_PATH = (
    "docs/evidence/"
    "phase-replay-development-corpus-sanmill-audit-2026-08-11.json"
)
EXPECTED_REPLAY_AUDIT_SHA256 = (
    "4634ba61a4e43c0b6d80a80c882aea5ca985b9bc8923e7895b39bf8ad557e42e"
)
EXPECTED_REPLAY_AUDIT_IDENTITY = (
    "9d4c54270c6e66dd9e16b4dae5af9291b1fea6d1385856650e71119dc4c0dbbf"
)


class TargetRefreshEqualTransitionError(RuntimeError):
    """Raised when a staged preparation invariant cannot be established."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TargetRefreshEqualTransitionError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TargetRefreshEqualTransitionError(
                    f"non-finite JSON value: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TargetRefreshEqualTransitionError(
            f"cannot load equal-transition contract: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise TargetRefreshEqualTransitionError("contract must be an object")
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
        raise TargetRefreshEqualTransitionError(
            "Git audit failed: " + " ".join(arguments)
        )
    return result.stdout.strip()


def _is_tracked(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _ordered_prefixes(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["prefixes"], key=lambda item: int(item["launch_order"]))


def _ordered_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["arms"], key=lambda item: int(item["launch_order"]))


def _contract_seeds(contract: Mapping[str, Any]) -> tuple[int, ...]:
    raw = contract.get("pairing", {}).get("seeds")
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw)
        or any(seed < 0 for seed in raw)
        or len(set(raw)) != len(raw)
    ):
        raise TargetRefreshEqualTransitionError("contract seed set differs")
    return tuple(raw)


def load_equal_transition_contract(path: str | Path) -> dict[str, Any]:
    """Load and fully validate the staged three-prefix, six-arm contract."""
    contract = _strict_json(Path(path))
    schema = contract.get("schema_version")
    if schema not in {CONTRACT_SCHEMA, SCHEDULE_ISOLATION_CONTRACT_SCHEMA}:
        raise TargetRefreshEqualTransitionError("contract schema differs")
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise TargetRefreshEqualTransitionError("contract plan identity differs")
    if contract.get("status") != "designed_unlaunched_needs_publication":
        raise TargetRefreshEqualTransitionError("contract status differs")
    authorization = contract.get("authorization")
    if authorization != {
        "arm_segments_authorized": 0,
        "launch_authorized": False,
        "prefix_segments_authorized": 0,
        "promotion_allowed": False,
        "publication_allowed": False,
    }:
        raise TargetRefreshEqualTransitionError("contract grants authority")

    prefixes = contract.get("prefixes")
    arms = contract.get("arms")
    seeds = _contract_seeds(contract)
    if not isinstance(prefixes, list) or len(prefixes) != len(seeds):
        raise TargetRefreshEqualTransitionError("prefix count differs")
    if not isinstance(arms, list) or len(arms) != 2 * len(seeds):
        raise TargetRefreshEqualTransitionError("arm count differs")
    if [(item.get("seed"), item.get("launch_order")) for item in prefixes] != [
        (seed, 1 + 3 * index) for index, seed in enumerate(seeds)
    ]:
        raise TargetRefreshEqualTransitionError("prefix order differs")
    observed_cells = [
        (item.get("seed"), item.get("condition"), item.get("launch_order"))
        for item in arms
    ]
    expected_cells = [
        (seed, condition, 2 + 3 * index + condition_index)
        for index, seed in enumerate(seeds)
        for condition_index, condition in enumerate(EXPECTED_CONDITIONS)
    ]
    if observed_cells != expected_cells:
        raise TargetRefreshEqualTransitionError("arm order differs")
    for field in ("control_dir", "plan_id", "specialist_db"):
        values = [item[field] for item in [*prefixes, *arms]]
        if len(values) != len(set(values)):
            raise TargetRefreshEqualTransitionError(
                f"sequence field is not unique: {field}"
            )
    for seed in seeds:
        prefix = next(item for item in prefixes if item["seed"] == seed)
        seed_arms = [item for item in arms if item["seed"] == seed]
        if any(
            arm["experiment_id"] != prefix["experiment_id"]
            or arm["prefix_specialist_db"] != prefix["specialist_db"]
            or arm["resume_checkpoint"]
            != (
                f"{prefix['control_dir']}/segments/segment-0001/"
                "target-refresh-fork.pt"
            )
            for arm in seed_arms
        ):
            raise TargetRefreshEqualTransitionError(
                f"seed {seed} arm lineage differs"
            )
        allowlist = set(contract["pairing"]["arm_difference_allowlist"])
        bases = [
            {key: value for key, value in arm.items() if key not in allowlist}
            for arm in seed_arms
        ]
        if bases[0] != bases[1]:
            raise TargetRefreshEqualTransitionError(
                f"seed {seed} arms differ outside the treatment allowlist"
            )

    common = contract.get("common_training_contract", {})
    resources = contract.get("resources", {})
    measurement = contract.get("measurement_contract", {})
    base_values_differ = (
        common.get("algorithm") != "A2C"
        or common.get("exact_transition_batch_size") != 64
        or common.get("target_refresh_fork_game") != 50
        or common.get("learning_rate_mode") != "fixed"
        or common.get("specialist_read_mode") != "theoretical-only"
        or common.get("no_final_undersized_flush") is not True
        or measurement.get("transition_boundaries")
        != list(EXPECTED_BOUNDARIES)
        or measurement.get("writes_training_data") is not False
        or resources.get("scientific_post_fork_transitions_total") != 49_152
        or resources.get("maximum_contract_training_games_total") != 3_600
        or resources.get("maximum_active_wall_hours_total") != 6.0
    )
    if base_values_differ:
        raise TargetRefreshEqualTransitionError("frozen scientific values differ")
    if schema == CONTRACT_SCHEMA:
        if (
            common.get("sanmill_node_ladder")
            != [1_000, 5_000, 25_000, 100_000, 500_000]
            or common.get("fixed_resource_stage_games")
            != [500, 500, 500, 1_000, 2_500]
            or common.get("temperature_schedule_axis", "global-games")
            != "global-games"
            or common.get("post_fork_temperature_anneal_transitions")
            is not None
        ):
            raise TargetRefreshEqualTransitionError(
                "legacy equal-transition schedule differs"
            )
    else:
        if (
            common.get("sanmill_node_ladder") != [1_000]
            or common.get("fixed_resource_stage_games") != [5_000]
            or common.get("temperature_schedule_axis")
            != "post-fork-transitions"
            or common.get("post_fork_temperature_anneal_transitions")
            != 106_304
        ):
            raise TargetRefreshEqualTransitionError(
                "schedule-isolation controls differ"
            )
        outcome = measurement.get("outcome_measurement", {})
        expected_outcome = {
            "candidate_colors": list(CANDIDATE_COLORS),
            "common_random_numbers_within_pairs": True,
            "fixed_replay_corpus": EXPECTED_REPLAY_CORPUS_PATH,
            "fixed_replay_corpus_identity": EXPECTED_REPLAY_CORPUS_IDENTITY,
            "fixed_replay_corpus_sha256": EXPECTED_REPLAY_CORPUS_SHA256,
            "games_per_checkpoint_condition_seed": 24,
            "held_out": False,
            "max_post_start_logical_plies": MAX_POST_START_LOGICAL_PLIES,
            "opponent": "common-game-50-anchor",
            "optimizer_updates": 0,
            "sampling_temperature": PRIMARY_TEMPERATURE,
            "strict_replay_audit": EXPECTED_REPLAY_AUDIT_PATH,
            "strict_replay_audit_identity": EXPECTED_REPLAY_AUDIT_IDENTITY,
            "strict_replay_audit_sha256": EXPECTED_REPLAY_AUDIT_SHA256,
            "total_games": 288,
            "training_games": 0,
            "transition_boundaries": list(OUTCOME_BOUNDARIES),
            "writes_training_data": False,
        }
        if outcome != expected_outcome:
            raise TargetRefreshEqualTransitionError(
                "schedule-isolation outcome measurement differs"
            )
        expected_outcome_thresholds = {
            "maximum_opposite_malom_mass_effect": (
                MAXIMUM_OPPOSITE_MALOM_MASS_EFFECT
            ),
            "maximum_opposite_phase_effect": MAXIMUM_OPPOSITE_PHASE_EFFECT,
            "maximum_truncation_rate_increase": (
                MAXIMUM_TRUNCATION_RATE_INCREASE
            ),
            "minimum_aggregate_score_effect": MINIMUM_AGGREGATE_SCORE_EFFECT,
            "minimum_per_seed_score_effect": MINIMUM_PER_SEED_SCORE_EFFECT,
            "minimum_supporting_seeds": MINIMUM_SUPPORTING_SEEDS,
        }
        if contract.get("analysis", {}).get(
            "outcome_classification"
        ) != expected_outcome_thresholds:
            raise TargetRefreshEqualTransitionError(
                "schedule-isolation outcome thresholds differ"
            )
    if contract.get("preparation_stages", {}).get("current_stage") != (
        "source_and_prefix_plan_preparation_only"
    ):
        raise TargetRefreshEqualTransitionError("preparation stage differs")
    return contract


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
        raise TargetRefreshEqualTransitionError("source audit requires dev")
    if status:
        raise TargetRefreshEqualTransitionError("source audit requires a clean worktree")
    if origin_main != contract["lineage"]["main_review"]["reviewed_tip"]:
        raise TargetRefreshEqualTransitionError("origin/main moved after review")
    for commit in contract["lineage"]["required_implementation_commits"]:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise TargetRefreshEqualTransitionError(
                f"required implementation commit is absent: {commit}"
            )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "origin_main": origin_main,
        "published": head == origin_dev,
        "tracked_clean": True,
    }


def _inspect_tracked_inputs(
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    inputs = {
        **contract["source_evidence"],
        "result_module": contract["analysis"]["result_implementation"]["module"],
        "result_publisher": contract["analysis"]["result_implementation"][
            "publisher"
        ],
        "fixed_phase_corpus": {
            "path": contract["measurement_contract"]["fixed_phase_corpus"],
            "sha256": contract["measurement_contract"][
                "fixed_phase_corpus_sha256"
            ],
        },
    }
    if contract.get("schema_version") == SCHEDULE_ISOLATION_CONTRACT_SCHEMA:
        outcome = contract["measurement_contract"]["outcome_measurement"]
        inputs.update(
            {
                "fixed_replay_corpus": {
                    "path": outcome["fixed_replay_corpus"],
                    "sha256": outcome["fixed_replay_corpus_sha256"],
                },
                "strict_replay_audit": {
                    "path": outcome["strict_replay_audit"],
                    "sha256": outcome["strict_replay_audit_sha256"],
                },
            }
        )
    for name, expected in inputs.items():
        path = _repository_path(root, expected["path"], field=name)
        if not path.is_file() or not _is_tracked(root, path):
            raise TargetRefreshEqualTransitionError(f"tracked input is absent: {name}")
        observed = _sha256_file(path)
        if observed != expected["sha256"]:
            raise TargetRefreshEqualTransitionError(f"tracked input differs: {name}")
        records[name] = {"path": expected["path"], "sha256": observed}
    return records


def _all_generated_targets(
    root: Path,
    contract: Mapping[str, Any],
    *,
    report_path: Path,
) -> list[tuple[str, Path]]:
    targets = [("readiness", report_path)]
    for prefix in contract["prefixes"]:
        targets.extend(
            (
                (
                    f"{prefix['seed']}:prefix-control",
                    _repository_path(
                        root, prefix["control_dir"], field="prefix control"
                    ),
                ),
                (
                    f"{prefix['seed']}:prefix-db",
                    _repository_path(
                        root, prefix["specialist_db"], field="prefix database"
                    ),
                ),
            )
        )
    for arm in contract["arms"]:
        targets.extend(
            (
                (
                    f"{arm['arm_id']}:control",
                    _repository_path(root, arm["control_dir"], field="arm control"),
                ),
                (
                    f"{arm['arm_id']}:db",
                    _repository_path(root, arm["specialist_db"], field="arm database"),
                ),
            )
        )
    return targets


def inspect_preparation_targets(
    root: Path,
    contract: Mapping[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    existing = [
        {
            "label": label,
            "path": path.relative_to(root.resolve()).as_posix(),
            "kind": "directory" if path.is_dir() else "file",
        }
        for label, path in _all_generated_targets(
            root, contract, report_path=report_path
        )
        if path.exists()
    ]
    return {"absent": not existing, "existing": existing}


def _assert_outputs_ignored(
    root: Path,
    contract: Mapping[str, Any],
    *,
    report_path: Path,
) -> None:
    for _label, path in _all_generated_targets(
        root, contract, report_path=report_path
    ):
        relative = path.relative_to(root.resolve()).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise TargetRefreshEqualTransitionError(
                f"generated output is not ignored by Git: {relative}"
            )


def build_prefix_prepare_command(
    *,
    root: Path,
    contract: Mapping[str, Any],
    prefix: Mapping[str, Any],
    paths_config: Path,
    python_executable: str,
) -> list[str]:
    common = contract["common_training_contract"]
    resources = contract["resources"]
    return [
        python_executable,
        str(root / "scripts/manage_generalist_run.py"),
        "prepare",
        "--control-dir",
        str(_repository_path(root, prefix["control_dir"], field="control_dir")),
        "--max-wall-hours",
        str(resources["prefix_active_wall_hours"]),
        "--plan-id",
        prefix["plan_id"],
        "--objective",
        f"{contract['objective']}; shared-prefix seed={prefix['seed']}",
        "--paths-config",
        str(paths_config),
        "--experiment-id",
        prefix["experiment_id"],
        "--seed",
        str(prefix["seed"]),
        "--max-games",
        str(common["max_games_schedule"]),
        "--completion-game-bound",
        str(resources["prefix_game_count"]),
        "--segment-games",
        str(resources["prefix_game_count"]),
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
        str(
            _repository_path(
                root, prefix["specialist_db"], field="prefix database"
            )
        ),
        "--exact-transition-batches",
        "--target-refresh-fork-game",
        str(common["target_refresh_fork_game"]),
        "--target-refresh-fork-treatment",
        "capture",
    ]


def build_prefix_prepare_commands(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    python_executable: str = sys.executable,
) -> list[list[str]]:
    return [
        build_prefix_prepare_command(
            root=root,
            contract=contract,
            prefix=prefix,
            paths_config=paths_config,
            python_executable=python_executable,
        )
        for prefix in _ordered_prefixes(contract)
    ]


def _trainer_args(plan: ManagedPlan) -> Any:
    return trainer._build_argument_parser().parse_args(
        ["--preflight", "long-run", *plan.common_trainer_args]
    )


def _command_semantics(command: Sequence[str]) -> str:
    manager_args = manager._build_parser().parse_args(list(command)[2:])
    common_args = manager._common_trainer_args(
        manager_args, Path(manager_args.paths_config)
    )
    args = trainer._build_argument_parser().parse_args(
        ["--preflight", "long-run", *common_args]
    )
    semantics = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
        and key not in {"experiment_id", "seed", "specialist_db"}
    }
    return canonical_sha256(semantics)


def validate_prefix_prepare_commands(
    contract: Mapping[str, Any], commands: Sequence[Sequence[str]]
) -> None:
    if len(commands) != len(_contract_seeds(contract)):
        raise TargetRefreshEqualTransitionError("prefix command count differs")
    semantics: set[str] = set()
    for prefix, command in zip(
        _ordered_prefixes(contract), commands, strict=True
    ):
        if "authorize" in command or "run-next" in command or "run-all" in command:
            raise TargetRefreshEqualTransitionError("preparation command can train")
        args = manager._build_parser().parse_args(list(command)[2:])
        if (
            args.seed != prefix["seed"]
            or not args.exact_transition_batches
            or args.target_refresh_fork_game
            != contract["common_training_contract"]["target_refresh_fork_game"]
            or args.target_refresh_fork_treatment != "capture"
            or args.post_fork_transition_bound is not None
            or not args.no_exact_resume
        ):
            raise TargetRefreshEqualTransitionError("prefix command semantics differ")
        semantics.add(_command_semantics(command))
    if len(semantics) != 1:
        raise TargetRefreshEqualTransitionError(
            "prefix commands differ outside seed-bound identities"
        )


def _assert_prefix_plan(
    plan: ManagedPlan,
    *,
    root: Path,
    contract: Mapping[str, Any],
    prefix: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> Any:
    common = contract["common_training_contract"]
    resources = contract["resources"]
    expected = {
        "plan_id": prefix["plan_id"],
        "objective": f"{contract['objective']}; shared-prefix seed={prefix['seed']}",
        "experiment_id": prefix["experiment_id"],
        "git_commit": source_commit,
        "control_dir": str(
            _repository_path(root, prefix["control_dir"], field="control_dir")
        ),
        "paths_config": str(paths_config),
        "max_games": common["max_games_schedule"],
        "completion_game_bound": resources["prefix_game_count"],
        "segment_games": resources["prefix_game_count"],
        "max_wall_hours": resources["prefix_active_wall_hours"],
        "allow_safe_exact_resume": False,
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    for field, value in expected.items():
        if getattr(plan, field) != value:
            raise TargetRefreshEqualTransitionError(
                f"prefix plan field differs: seed {prefix['seed']}:{field}"
            )
    if plan.paths_config_sha256 != _sha256_file(paths_config):
        raise TargetRefreshEqualTransitionError("path registry identity differs")
    args = _trainer_args(plan)
    expected_args = {
        "seed": prefix["seed"],
        "experiment_id": prefix["experiment_id"],
        "max_games": common["max_games_schedule"],
        "max_ply": common["max_logical_plies"],
        "max_ply_branch": common["max_logical_plies"],
        "self_play_ratio": common["frozen_target_ratio"],
        "update_target_every": common["target_refresh_fork_game"],
        "lr_adaptation_mode": "fixed",
        "specialist_read_mode": "theoretical-only",
        "mill_bonus_mode": "malom-preserving-only",
        "exact_transition_batches": True,
        "target_refresh_fork_game": common["target_refresh_fork_game"],
        "target_refresh_fork_treatment": "capture",
        "post_fork_transition_bound": None,
        "temperature_schedule_axis": "global-games",
        "post_fork_temperature_anneal_transitions": None,
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
            raise TargetRefreshEqualTransitionError(
                f"prefix trainer argument differs: seed {prefix['seed']}:{field}"
            )
    expected_db = _repository_path(
        root, prefix["specialist_db"], field="prefix database"
    )
    if Path(args.specialist_db).resolve() != expected_db:
        raise TargetRefreshEqualTransitionError("prefix database path differs")
    return args


def inspect_source_readiness(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path | None = None,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Audit source and prospective prefix commands without creating outputs."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    if report_path is None:
        report_path = root / DEFAULT_REPORT
    report_path = report_path.resolve(strict=False)
    contract = load_equal_transition_contract(contract_path)
    source = _inspect_source(root, contract)
    tracked_inputs = _inspect_tracked_inputs(root, contract)
    template = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    targets = inspect_preparation_targets(
        root, contract, report_path=report_path
    )
    commands = build_prefix_prepare_commands(
        root=root,
        contract=contract,
        paths_config=paths_config,
        python_executable=python_executable,
    )
    validate_prefix_prepare_commands(contract, commands)
    unresolved: list[str] = []
    if not source["published"]:
        unresolved.append("publish the exact frozen source to origin/dev")
    if not targets["absent"]:
        unresolved.append("quarantine pre-existing sequence targets")
    if not unresolved:
        unresolved.append(
            "generate three immutable prefix plans and preflights; training remains unauthorized"
        )
    state = (
        "implementation_complete_needs_publication"
        if not source["published"]
        else (
            "published_source_needs_target_quarantine"
            if not targets["absent"]
            else "source_ready_for_prefix_preparation"
        )
    )
    body = {
        "schema_version": SOURCE_READINESS_SCHEMA,
        "state": state,
        "verdict": "needs_decision",
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256_file(contract_path),
            "plan_identity": contract["plan_identity"],
        },
        "source": source,
        "tracked_inputs": tracked_inputs,
        "template": template,
        "runtime": runtime,
        "preparation_targets": targets,
        "prefix_prepare_commands": commands,
        "deferred_arm_plans": {
            "count": len(contract["arms"]),
            "reason": (
                "each arm must bind the real game-50 fork checkpoint and a "
                "closed byte-identical clone of its prefix SpecialistDB"
            ),
            "placeholder_sources_forbidden": True,
        },
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": unresolved,
    }
    return {**body, "readiness_identity": canonical_sha256(body)}


def prepare_prefix_plans(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create three prefix plans and preflights without launching training."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    report_path = report_path.resolve(strict=False)
    contract = load_equal_transition_contract(contract_path)
    source = _inspect_source(root, contract)
    if not source["published"]:
        raise TargetRefreshEqualTransitionError("dev must equal origin/dev")
    tracked_inputs = _inspect_tracked_inputs(root, contract)
    template = inspect_template(root, contract)
    runtime = inspect_runtime_identities(root, paths_config, contract)
    targets = inspect_preparation_targets(
        root, contract, report_path=report_path
    )
    if not targets["absent"]:
        raise TargetRefreshEqualTransitionError("preparation targets already exist")
    _assert_outputs_ignored(root, contract, report_path=report_path)
    commands = build_prefix_prepare_commands(
        root=root,
        contract=contract,
        paths_config=paths_config,
        python_executable=python_executable,
    )
    validate_prefix_prepare_commands(contract, commands)

    prefix_records: list[dict[str, Any]] = []
    template_path = Path(template["path"])
    for prefix, command in zip(
        _ordered_prefixes(contract), commands, strict=True
    ):
        specialist_db = _repository_path(
            root, prefix["specialist_db"], field="prefix database"
        )
        specialist_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template_path, specialist_db)
        database = _inspect_specialist_database(
            specialist_db,
            contract["data_contract"]["specialist_db_initial_template"],
        )
        manager_result = _run_checked(command, root=root, runner=runner)
        try:
            manager_output = json.loads(manager_result.stdout)
        except json.JSONDecodeError as exc:
            raise TargetRefreshEqualTransitionError(
                f"manager output is not JSON: seed {prefix['seed']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise TargetRefreshEqualTransitionError("manager state differs")
        control_dir = _repository_path(
            root, prefix["control_dir"], field="prefix control"
        )
        plan_path = control_dir / "plan.json"
        plan = load_managed_plan(plan_path)
        _assert_prefix_plan(
            plan,
            root=root,
            contract=contract,
            prefix=prefix,
            paths_config=paths_config,
            source_commit=source["head"],
        )
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
            raise TargetRefreshEqualTransitionError(
                f"preflight output is not JSON: seed {prefix['seed']}"
            ) from exc
        _validate_unlaunched_preflight(
            preflight,
            plan=plan,
            source_commit=source["head"],
            arm_id=f"seed{prefix['seed']}-prefix",
        )
        preflight_path = control_dir / "preflight.json"
        try:
            with preflight_path.open("xb") as handle:
                handle.write(canonical_json_bytes(preflight))
        except FileExistsError as exc:
            raise TargetRefreshEqualTransitionError(
                f"prefix preflight already exists: seed {prefix['seed']}"
            ) from exc
        prefix_records.append(
            {
                "seed": prefix["seed"],
                "launch_order": prefix["launch_order"],
                "plan_path": str(plan_path),
                "plan_sha256": plan.plan_sha256,
                "resume_config_sha256": plan.resume_config_sha256,
                "specialist_db": database,
                "preflight": {
                    "path": str(preflight_path),
                    "sha256": _sha256_file(preflight_path),
                    "verdict": preflight["verdict"],
                    "unresolved_decisions": preflight["unresolved_decisions"],
                },
                "authorization_present": False,
                "segment_output_present": False,
            }
        )
    if len({record["resume_config_sha256"] for record in prefix_records}) != len(
        prefix_records
    ):
        raise TargetRefreshEqualTransitionError(
            "prefix resume identities collide across seed-owned lineages"
        )
    body = {
        "schema_version": READINESS_SCHEMA,
        "state": "prefix_plans_ready_for_product_authorization",
        "verdict": "needs_decision",
        "launch_authorized": False,
        "contract": {
            "path": str(contract_path),
            "sha256": _sha256_file(contract_path),
            "plan_identity": contract["plan_identity"],
        },
        "source": source,
        "tracked_inputs": tracked_inputs,
        "template": template,
        "runtime": runtime,
        "prefixes": prefix_records,
        "arms": {
            "prepared": False,
            "count": len(contract["arms"]),
            "gate": (
                "complete and audit each prefix before cloning its closed "
                "SpecialistDB and binding the real fork checkpoint"
            ),
        },
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
        "unresolved_decisions": [
            PRODUCT_AUTHORIZATION_DECISION,
            "arm plans and preflights require completed prefix artefacts",
        ],
    }
    report = {**body, "readiness_identity": canonical_sha256(body)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with report_path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise TargetRefreshEqualTransitionError(
            "prefix readiness report already exists"
        ) from exc
    return report


def publish_source_readiness(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise TargetRefreshEqualTransitionError(
            f"source readiness already exists: {path}"
        ) from exc


__all__ = [
    "CONTRACT_SCHEMA",
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_REPORT",
    "DEFAULT_SOURCE_REPORT",
    "EXPECTED_BOUNDARIES",
    "EXPECTED_CONDITIONS",
    "EXPECTED_SEEDS",
    "READINESS_SCHEMA",
    "SOURCE_READINESS_SCHEMA",
    "TargetRefreshEqualTransitionError",
    "build_prefix_prepare_commands",
    "inspect_source_readiness",
    "load_equal_transition_contract",
    "prepare_prefix_plans",
    "publish_source_readiness",
    "validate_prefix_prepare_commands",
]
