"""Fail-closed preparation for the six-arm mill-bonus ablation smoke."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from learned_ai.training.managed_generalist import (
    ManagedPlan,
    load_managed_plan,
)
from learned_ai.training.run_contract import (
    canonical_json_bytes,
    canonical_sha256,
)
from learned_ai.training.sanmill_referee import (
    TRAINING_REFEREE_SEMANTIC_DIGEST,
    inspect_sanmill_training_installation,
)
from learned_ai.training.training_identity import (
    MIF_RELEASE_COMMIT,
    MIF_SUITE_JCS_SHA256,
    MIF_SUITE_TAG,
    TRAINER_RULESET_SEMANTIC_DIGEST,
    load_trainer_ruleset,
)
from scripts import train_s_gen_v2 as trainer


ABLATION_SCHEMA = "nmm.sanmill-mill-bonus-ablation-smoke-plan.v1"
READINESS_SCHEMA = "nmm.sanmill-mill-bonus-ablation-readiness.v1"
DEFAULT_CONTRACT = Path(
    "docs/experiments/sanmill-mill-bonus-ablation-smoke-v1.json"
)
DEFAULT_PATHS_CONFIG = Path("data/training_paths.local.json")
DEFAULT_REPORT = Path("out/mill-bonus-ablation-smoke-v1/readiness.json")
PRODUCT_AUTHORIZATION_DECISION = (
    "long-run launch requires a frozen managed plan and separate "
    "product authorization"
)


class MillBonusAblationReadinessError(RuntimeError):
    """A frozen input, preparation target, or managed plan differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise MillBonusAblationReadinessError(
                    f"duplicate JSON key {key!r}: {path}"
                )
            value[key] = item
        return value

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise MillBonusAblationReadinessError(
            f"cannot read JSON object: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise MillBonusAblationReadinessError(f"JSON root is not an object: {path}")
    return value


def _repository_path(root: Path, value: str, *, field: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MillBonusAblationReadinessError(
            f"{field} must stay inside the repository"
        ) from exc
    return resolved


def load_ablation_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable source-only ablation contract."""
    contract_path = Path(path)
    contract = _strict_json(contract_path)
    if contract.get("schema_version") != ABLATION_SCHEMA:
        raise MillBonusAblationReadinessError("unsupported ablation contract schema")
    identity = contract.get("plan_identity")
    body = {key: value for key, value in contract.items() if key != "plan_identity"}
    if identity != canonical_sha256(body):
        raise MillBonusAblationReadinessError("ablation plan identity differs")
    if contract.get("status") != "designed_unlaunched_needs_product_authorization":
        raise MillBonusAblationReadinessError("ablation contract status is not unlaunched")
    authorization = contract.get("authorization")
    if not isinstance(authorization, Mapping) or any(
        authorization.get(field) is not False
        for field in (
            "launch_authorized",
            "publication_allowed",
            "promotion_allowed",
        )
    ):
        raise MillBonusAblationReadinessError(
            "source contract must not authorize launch or publication"
        )
    if authorization.get("authorized_segments_per_arm") != 0:
        raise MillBonusAblationReadinessError("source contract authorizes a segment")
    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != 6:
        raise MillBonusAblationReadinessError("ablation contract must contain six arms")
    if sorted(arm.get("launch_order") for arm in arms) != list(range(1, 7)):
        raise MillBonusAblationReadinessError("ablation launch order is incomplete")
    return contract


def _inspect_specialist_database(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise MillBonusAblationReadinessError(
            f"SpecialistDB input is missing: {path}"
        )
    if path.stat().st_size != expected.get("byte_length"):
        raise MillBonusAblationReadinessError("SpecialistDB byte length differs")
    sha256 = _sha256_file(path)
    if sha256 != expected.get("sha256"):
        raise MillBonusAblationReadinessError("SpecialistDB SHA-256 differs")
    sidecars = [
        Path(f"{path}{suffix}")
        for suffix in ("-wal", "-shm", "-journal")
        if Path(f"{path}{suffix}").exists()
    ]
    if sidecars:
        raise MillBonusAblationReadinessError(
            "SpecialistDB has SQLite sidecars: "
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
        raise MillBonusAblationReadinessError(
            f"SpecialistDB immutable audit failed: {path}"
        ) from exc
    finally:
        if "connection" in locals():
            connection.close()
    if quick_check != (expected.get("quick_check"),):
        raise MillBonusAblationReadinessError("SpecialistDB quick_check differs")
    if metadata.get("malom_label_version") != expected.get("label_version"):
        raise MillBonusAblationReadinessError("SpecialistDB label version differs")
    expected_counts = {
        table: int(expected[table])
        for table in ("positions", "winning_lines", "preferred_plays")
    }
    if counts != expected_counts:
        raise MillBonusAblationReadinessError("SpecialistDB table counts differ")
    return {
        "path": str(path),
        "byte_length": path.stat().st_size,
        "sha256": sha256,
        "quick_check": quick_check[0],
        "label_version": metadata["malom_label_version"],
        "counts": counts,
        "sidecars": [],
    }


def inspect_template(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the ignored closed template without creating SQLite sidecars."""
    expected = contract["data_contract"]["specialist_db_initial_template"]
    path = _repository_path(root, expected["path"], field="template path")
    return _inspect_specialist_database(path, expected)


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
        raise MillBonusAblationReadinessError(
            "Git audit failed: " + " ".join(arguments)
        ) from exc
    return result.stdout.strip()


def inspect_published_source(
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one clean, published dev source tip and a reviewed main tip."""
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    upstream = _git_output(root, "rev-parse", "origin/dev")
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if branch != "dev":
        raise MillBonusAblationReadinessError("ablation preparation requires dev")
    if head != upstream:
        raise MillBonusAblationReadinessError("dev must equal origin/dev")
    if status:
        raise MillBonusAblationReadinessError("worktree must be clean")
    reviewed_main = contract["lineage"]["main_review"]["reviewed_tip"]
    actual_main = _git_output(root, "rev-parse", "origin/main")
    if actual_main != reviewed_main:
        raise MillBonusAblationReadinessError(
            "origin/main moved after the recorded review"
        )
    required_commits = contract["lineage"]["implementation_commits"]
    for name, commit in required_commits.items():
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise MillBonusAblationReadinessError(
                f"required implementation commit is absent: {name}"
            )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": upstream,
        "origin_main": actual_main,
        "worktree_clean": True,
    }


def inspect_preparation_evidence(
    root: Path,
    contract: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate an optional, experiment-specific no-update probe.

    The original Mill-bonus experiment predates this gate and has no such
    entry.  Successor contracts can require one canonical probe without
    creating a circular dependency between the contract identity and the
    probe identity: the contract freezes the source cohort, implementation
    paths and expected summary, while the probe binds the clean published
    commit that actually produced it.
    """
    evidence = contract.get("preparation_evidence")
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "downgrade_penalty_no_update_probe"
    }:
        raise MillBonusAblationReadinessError(
            "preparation evidence contract is invalid"
        )
    spec = evidence["downgrade_penalty_no_update_probe"]
    required_spec = {
        "path",
        "schema_version",
        "source_probe_identity",
        "source_probe_sha256",
        "module_path",
        "script_path",
        "expected_summary",
    }
    if not isinstance(spec, Mapping) or set(spec) != required_spec:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe contract is invalid"
        )
    probe_path = _repository_path(
        root, str(spec["path"]), field="downgrade-penalty probe"
    )
    if not probe_path.is_file():
        raise MillBonusAblationReadinessError(
            "required downgrade-penalty probe is missing"
        )
    relative = probe_path.relative_to(root.resolve()).as_posix()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe must remain ignored by Git"
        )
    try:
        raw = probe_path.read_bytes()
    except OSError as exc:
        raise MillBonusAblationReadinessError(
            "cannot read downgrade-penalty probe"
        ) from exc
    probe = _strict_json(probe_path)
    if raw != canonical_json_bytes(probe):
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe is not canonical JSON"
        )
    identity = probe.get("probe_identity")
    body = {key: value for key, value in probe.items() if key != "probe_identity"}
    if not isinstance(identity, str) or identity != canonical_sha256(body):
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe identity differs"
        )
    if probe.get("schema_version") != spec["schema_version"]:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe schema differs"
        )
    expected_source = {
        "probe_identity": spec["source_probe_identity"],
        "sha256": spec["source_probe_sha256"],
    }
    if probe.get("source_probe") != expected_source:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty source probe differs"
        )
    if probe.get("summary") != spec["expected_summary"]:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe summary differs"
        )
    rows = probe.get("per_state")
    if not isinstance(rows, list) or not rows:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe has no state records"
        )
    ordinals: set[int] = set()
    affected = 0
    mill_forming = 0
    quality_ranks: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    control_total = 0.0
    treatment_total = 0.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise MillBonusAblationReadinessError(
                "downgrade-penalty state record is invalid"
            )
        ordinal = row.get("ordinal")
        mills_formed = row.get("mills_formed")
        quality = row.get("malom_quality")
        phase = row.get("phase")
        stratum = row.get("stratum")
        rewards = row.get("rewards")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal in ordinals
            or isinstance(mills_formed, bool)
            or not isinstance(mills_formed, int)
            or mills_formed < 0
            or isinstance(quality, bool)
            or not isinstance(quality, (int, float))
            or float(quality) not in (-1.0, -2.0)
            or not isinstance(phase, str)
            or not phase
            or not isinstance(stratum, str)
            or not stratum
            or not isinstance(rewards, Mapping)
        ):
            raise MillBonusAblationReadinessError(
                "downgrade-penalty state record is invalid"
            )
        ordinals.add(ordinal)
        totals: dict[str, float] = {}
        for mode in ("control", "treatment"):
            components = rewards.get(mode)
            total = (
                components.get("total")
                if isinstance(components, Mapping)
                else None
            )
            if (
                isinstance(total, bool)
                or not isinstance(total, (int, float))
                or not math.isfinite(float(total))
            ):
                raise MillBonusAblationReadinessError(
                    "downgrade-penalty state reward is invalid"
                )
            totals[mode] = float(total)
        affected += int(totals["control"] != totals["treatment"])
        mill_forming += int(mills_formed > 0)
        quality_ranks[str(int(-float(quality)))] += 1
        phases[phase] += 1
        strata[stratum] += 1
        control_total += totals["control"]
        treatment_total += totals["treatment"]
    reconstructed_summary = {
        "states": len(rows),
        "affected_states": affected,
        "mill_forming_states": mill_forming,
        "non_mill_states": len(rows) - mill_forming,
        "quality_rank_counts": dict(sorted(quality_ranks.items())),
        "phase_counts": dict(sorted(phases.items())),
        "stratum_counts": dict(sorted(strata.items())),
        "control_reward_total": control_total,
        "treatment_reward_total": treatment_total,
        "treatment_minus_control": treatment_total - control_total,
    }
    if reconstructed_summary != probe["summary"]:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe records do not reconcile"
        )
    claim = probe.get("claim_boundary")
    required_claim = {
        "candidate_policy_loaded": False,
        "new_games": False,
        "optimizer_created": False,
        "weights_updated": False,
        "actions_changed_between_modes": False,
        "states_changed_between_modes": False,
        "reward_component_only": True,
        "causal_training_effect_proven": False,
    }
    if claim != required_claim:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe claim boundary differs"
        )
    auditor = probe.get("auditor")
    required_auditor = {
        "implementation_commit",
        "implementation_tree",
        "module_sha256",
        "script_sha256",
        "tracked_worktree_clean",
    }
    if not isinstance(auditor, Mapping) or set(auditor) != required_auditor:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe auditor differs"
        )
    commit = auditor.get("implementation_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe commit is invalid"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, str(source["head"])],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe commit is not in the published lineage"
        )
    if (
        auditor.get("implementation_tree")
        != _git_output(root, "rev-parse", f"{commit}^{{tree}}")
        or auditor.get("tracked_worktree_clean") is not True
    ):
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe source identity differs"
        )
    module_path = _repository_path(
        root, str(spec["module_path"]), field="probe module"
    )
    script_path = _repository_path(
        root, str(spec["script_path"]), field="probe script"
    )
    if (
        not module_path.is_file()
        or not script_path.is_file()
        or auditor.get("module_sha256") != _sha256_file(module_path)
        or auditor.get("script_sha256") != _sha256_file(script_path)
    ):
        raise MillBonusAblationReadinessError(
            "downgrade-penalty probe implementation bytes differ"
        )
    return {
        "path": str(probe_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "probe_identity": identity,
        "auditor": dict(auditor),
        "summary": probe["summary"],
        "claim_boundary": probe["claim_boundary"],
    }


def _load_paths_config(root: Path, path: Path) -> dict[str, Any]:
    config = _strict_json(path)
    required = {"sanmill_training_checkout"}
    if not required <= set(config):
        raise MillBonusAblationReadinessError(
            "training path registry lacks the Sanmill runtime"
        )
    return config


def _resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def inspect_runtime_identities(
    root: Path,
    paths_config: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify runtime identities used by every arm before local preparation."""
    config = _load_paths_config(root, paths_config)
    rules = contract["rules_and_runtime"]
    observed_mif = {
        "mif_tag": MIF_SUITE_TAG,
        "mif_release_commit": MIF_RELEASE_COMMIT,
        "mif_suite_jcs_sha256": MIF_SUITE_JCS_SHA256.removeprefix("sha256:"),
        "rules_semantic_digest": TRAINER_RULESET_SEMANTIC_DIGEST,
        "sanmill_strict_referee_semantic_digest": (
            TRAINING_REFEREE_SEMANTIC_DIGEST
        ),
    }
    for field, value in observed_mif.items():
        if rules.get(field) != value:
            raise MillBonusAblationReadinessError(
                f"frozen runtime identity differs: {field}"
            )
    ruleset = load_trainer_ruleset(
        root / "data/rulesets/nmm-training-core@2.json"
    )
    if ruleset.semantic_digest != rules["rules_semantic_digest"]:
        raise MillBonusAblationReadinessError("ruleset manifest identity differs")
    checkout = _resolve_config_path(root, config["sanmill_training_checkout"])
    installation = inspect_sanmill_training_installation(checkout)
    expected_sanmill = {
        "commit": rules["sanmill_commit"],
        "tree": rules["sanmill_tree"],
        "binary_sha256": rules["sanmill_binary_sha256"],
    }
    observed_sanmill = {
        "commit": installation.commit,
        "tree": installation.tree,
        "binary_sha256": installation.binary_sha256,
    }
    if observed_sanmill != expected_sanmill:
        raise MillBonusAblationReadinessError("Sanmill runtime identity differs")
    return {
        "mif": {
            "tag": MIF_SUITE_TAG,
            "release_commit": MIF_RELEASE_COMMIT,
            "suite_jcs_sha256": MIF_SUITE_JCS_SHA256,
        },
        "ruleset": ruleset.to_dict(),
        "sanmill": {**observed_sanmill, "checkout": str(checkout)},
    }


def _ordered_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(contract["arms"], key=lambda arm: int(arm["launch_order"]))


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


def assert_preparation_targets_absent(
    root: Path,
    contract: Mapping[str, Any],
    *,
    report_path: Path,
) -> None:
    """Refuse all overwrite or continuation behavior during preparation."""
    targets = [report_path]
    for arm in _ordered_arms(contract):
        targets.extend(
            (
                _repository_path(root, arm["control_dir"], field="control_dir"),
                _repository_path(root, arm["specialist_db"], field="specialist_db"),
            )
        )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise MillBonusAblationReadinessError(
            "preparation targets already exist: " + ", ".join(existing)
        )


def assert_preparation_outputs_ignored(
    root: Path,
    contract: Mapping[str, Any],
    *,
    report_path: Path,
) -> None:
    """Require every generated plan, database, and report to stay untracked."""
    targets = [
        _repository_path(root, str(report_path), field="readiness report")
    ]
    for arm in _ordered_arms(contract):
        targets.extend(
            (
                _repository_path(root, arm["control_dir"], field="control_dir"),
                _repository_path(root, arm["specialist_db"], field="specialist_db"),
            )
        )
    for target in targets:
        relative = target.relative_to(root.resolve()).as_posix()
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if ignored.returncode != 0:
            raise MillBonusAblationReadinessError(
                f"preparation output is not ignored by Git: {relative}"
            )


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
        "completion_game_bound": contract["resources"][
            "completed_games_per_arm"
        ],
        "segment_games": common["one_segment_games"],
        "max_wall_hours": contract["resources"]["active_wall_hours_per_arm"],
        "publication_allowed": False,
        "promotion_allowed": False,
    }
    for field, expected in expected_scalars.items():
        if getattr(plan, field) != expected:
            raise MillBonusAblationReadinessError(
                f"managed plan field differs for {arm['arm_id']}: {field}"
            )
    if plan.game_bound != contract["resources"]["completed_games_per_arm"]:
        raise MillBonusAblationReadinessError(
            f"managed plan completion bound differs for {arm['arm_id']}"
        )
    if plan.paths_config_sha256 != _sha256_file(paths_config):
        raise MillBonusAblationReadinessError("managed path registry hash differs")
    args = _trainer_args(plan)
    expected_args = {
        "experiment_id": arm["experiment_id"],
        "seed": arm["seed"],
        "mill_bonus_mode": arm["mill_bonus_mode"],
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
            raise MillBonusAblationReadinessError(
                f"trainer argument differs for {arm['arm_id']}: {field}"
            )
    if list(args.sanmill_node_ladder) != common["sanmill_node_ladder"]:
        raise MillBonusAblationReadinessError("Sanmill node ladder differs")
    if list(args.sanmill_stage_games) != common["fixed_resource_stage_games"]:
        raise MillBonusAblationReadinessError("Sanmill stage durations differ")
    specialist_db = _repository_path(
        root, arm["specialist_db"], field="specialist_db"
    )
    if Path(args.specialist_db).resolve() != specialist_db:
        raise MillBonusAblationReadinessError("arm SpecialistDB path differs")
    gate = plan.policy_health
    if gate is None or gate.device != diagnostic["device"]:
        raise MillBonusAblationReadinessError("policy-health device differs")
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
    ):
        raise MillBonusAblationReadinessError(
            "policy-health input identities differ"
        )
    if (
        gate.exact_critical_states != diagnostic["critical_states"]
        or gate.required_direct_preserving_rate
        != diagnostic["required_direct_signal_preserving_rate"]
        or gate.min_candidate_preserving_rate
        != diagnostic["minimum_candidate_preserving_rate"]
        or gate.min_candidate_logit_margin
        != diagnostic["minimum_candidate_preserving_minus_downgrading_logit_margin"]
    ):
        raise MillBonusAblationReadinessError("policy-health thresholds differ")
    return args


def _normalised_pair_semantics(args: Any) -> str:
    ignored = {
        "experiment_id",
        "mill_bonus_mode",
        "specialist_db",
    }
    value = {
        key: item
        for key, item in vars(args).items()
        if not key.startswith("_") and key not in ignored
    }
    return canonical_sha256(json.loads(canonical_json_bytes(value)))


def audit_prepared_plans(
    *,
    root: Path,
    contract: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
    preflight_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Verify exact managed plans and same-seed one-factor equivalence."""
    template = contract["data_contract"]["specialist_db_initial_template"]
    audited: list[dict[str, Any]] = []
    normalised_by_seed: dict[int, set[str]] = {}
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
            raise MillBonusAblationReadinessError(
                f"arm is already authorized: {arm['arm_id']}"
            )
        if (control_dir / "segments").exists():
            raise MillBonusAblationReadinessError(
                f"arm already has segment output: {arm['arm_id']}"
            )
        normalised_by_seed.setdefault(int(arm["seed"]), set()).add(
            _normalised_pair_semantics(args)
        )
        record = {
            "arm_id": arm["arm_id"],
            "launch_order": arm["launch_order"],
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
    if any(len(identities) != 1 for identities in normalised_by_seed.values()):
        raise MillBonusAblationReadinessError(
            "same-seed arms differ outside the permitted reward factor"
        )
    return audited


def _run_checked(
    command: Sequence[str],
    *,
    root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    accepted_return_codes: Sequence[int] = (0,),
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(
            list(command),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MillBonusAblationReadinessError(
            f"command could not run: {command[1]}"
        ) from exc
    if result.returncode not in accepted_return_codes:
        details = []
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            details.append(f"stdout={stdout}")
        if stderr:
            details.append(f"stderr={stderr}")
        raise MillBonusAblationReadinessError(
            f"command failed with {result.returncode}: "
            + ("; ".join(details) if details else "<no output>")
        )
    return result


def _validate_unlaunched_preflight(
    preflight: Mapping[str, Any],
    *,
    plan: ManagedPlan,
    source_commit: str,
    arm_id: str,
) -> None:
    """Require a clean technical preflight with only the product gate open."""
    if preflight.get("schema_version") != "nmm.generalist-preflight.v1":
        raise MillBonusAblationReadinessError(
            f"preflight schema differs: {arm_id}"
        )
    if preflight.get("mode") != "long-run":
        raise MillBonusAblationReadinessError(
            f"preflight mode differs: {arm_id}"
        )
    if preflight.get("verdict") != "needs_decision":
        raise MillBonusAblationReadinessError(
            f"unlaunched preflight verdict differs: {arm_id}"
        )
    if preflight.get("errors") != []:
        raise MillBonusAblationReadinessError(
            f"unlaunched preflight has errors: {arm_id}"
        )
    if preflight.get("unresolved_decisions") != [
        PRODUCT_AUTHORIZATION_DECISION
    ]:
        raise MillBonusAblationReadinessError(
            f"unlaunched preflight decisions differ: {arm_id}"
        )
    if preflight.get("resume_config_sha256") != plan.resume_config_sha256:
        raise MillBonusAblationReadinessError(
            f"preflight did not bind the managed plan: {arm_id}"
        )
    git = preflight.get("git")
    if (
        not isinstance(git, Mapping)
        or git.get("commit") != source_commit
        or git.get("dirty") is not False
    ):
        raise MillBonusAblationReadinessError(
            f"preflight source identity differs: {arm_id}"
        )
    config = preflight.get("resolved_config")
    expected_config = {
        "experiment_id": plan.experiment_id,
        "run_id": f"{plan.plan_id}-segment-0001",
        "segment_games": plan.segment_games,
        "segment_stop_game": min(plan.segment_games, plan.game_bound),
        "start_mode": "fresh",
    }
    if not isinstance(config, Mapping) or any(
        config.get(key) != value for key, value in expected_config.items()
    ):
        raise MillBonusAblationReadinessError(
            f"preflight segment contract differs: {arm_id}"
        )
    expected_output = (
        Path(plan.control_dir) / "segments" / "segment-0001"
    ).resolve(strict=False)
    observed_output = Path(str(config.get("out_dir", ""))).resolve(strict=False)
    if observed_output != expected_output:
        raise MillBonusAblationReadinessError(
            f"preflight output path differs: {arm_id}"
        )
    checks = preflight.get("checks")
    output = checks.get("output") if isinstance(checks, Mapping) else None
    if not isinstance(output, Mapping) or output != {
        "exists": False,
        "isolated": True,
        "kind": "run_directory",
    }:
        raise MillBonusAblationReadinessError(
            f"preflight output isolation differs: {arm_id}"
        )


def _build_fresh_preflight_command(
    plan: ManagedPlan,
    *,
    root: Path,
    python_executable: str,
) -> list[str]:
    """Build a read-only preflight for the actual first managed segment."""
    segment_index = 1
    segment_output = (
        Path(plan.control_dir) / "segments" / f"segment-{segment_index:04d}"
    )
    segment_stop_game = min(plan.segment_games, plan.game_bound)
    return [
        python_executable,
        str(root / "scripts/train_s_gen_v2.py"),
        "--preflight",
        "long-run",
        "--run-id",
        f"{plan.plan_id}-segment-{segment_index:04d}",
        "--out-dir",
        str(segment_output),
        "--segment-games",
        str(plan.segment_games),
        "--segment-stop-game",
        str(segment_stop_game),
        *plan.common_trainer_args,
        "--start-mode",
        "fresh",
    ]


def prepare_ablation(
    *,
    root: Path,
    contract_path: Path,
    paths_config: Path,
    report_path: Path,
    python_executable: str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create six plans and read-only preflights without authorizing a run."""
    contract = load_ablation_contract(contract_path)
    source = inspect_published_source(root, contract)
    preparation_evidence = inspect_preparation_evidence(
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
            raise MillBonusAblationReadinessError(
                f"manager output is not JSON: {arm['arm_id']}"
            ) from exc
        if manager_output.get("state") != "awaiting_product_authorization":
            raise MillBonusAblationReadinessError(
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
            raise MillBonusAblationReadinessError(
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
            raise MillBonusAblationReadinessError(
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
        "preparation_evidence": preparation_evidence,
        "template": template_record,
        "runtime": runtime,
        "commands": commands,
        "arms": audited,
        "resource_envelope": contract["resources"],
        "claim_boundary": contract["claim_boundary"],
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
        raise MillBonusAblationReadinessError(
            f"readiness report already exists: {report_path}"
        ) from exc
    return report
