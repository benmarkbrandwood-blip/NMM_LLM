"""Fail-closed analysis for the target-refresh/LR factorial diagnostic."""

from __future__ import annotations

import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from learned_ai.evaluation.mill_bonus_ablation_result import (
    MillBonusAblationResultError,
    _artifact_record,
    _outcome_label,
    _require_finite,
    _require_int,
    _sha256_file,
    _strict_json,
    _strict_jsonl,
    _validate_authorization,
    _validate_controller_completion,
    _validate_finite_tree,
    _validate_policy_health,
    summarize_update_rows,
)
from learned_ai.training import managed_generalist as managed
from learned_ai.training.managed_generalist import (
    load_managed_authorization,
    load_managed_plan,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.target_refresh_lr_factorial_diagnostic import (
    CONDITION_FACTORS,
    EXPECTED_CONDITIONS,
    EXPECTED_SEEDS,
    PRODUCT_AUTHORIZATION_DECISION,
    READINESS_SCHEMA,
    RESULT_SCHEMA,
    _assert_plan_semantics,
    _ordered_arms,
    _repository_path,
    load_target_refresh_lr_contract,
)
from scripts import train_s_gen_v2 as trainer


DEFAULT_RESULT = Path("out/target-refresh-lr-factorial-diagnostic-v1/result.json")
BOUNDARY_GAME = 50
FINAL_GAME = 100
MINIMUM_MATERIAL_SCORE_EFFECT = 0.10

TargetRefreshLrResultError = MillBonusAblationResultError


_REQUIRED_GAME_FIELDS = {
    "game_id",
    "game",
    "difficulty",
    "learner_color",
    "temperature",
    "outcome",
    "ply",
    "update_policy_loss",
    "update_value_loss",
    "update_entropy",
    "reward_total_mean",
    "chosen_prob_mean",
    "entropy_mean",
    "policy_top1_rate",
    "heuristic_top1_rate",
    "malom_preserving_move_rate",
    "malom_downgrade_move_rate",
    "game_type",
    "phase_bucket",
    "is_branch",
    "target_age",
    "termination_reason",
    "opponent_node_budget",
    "lr",
    "specialist_read_mode",
    "specialist_read_empirical_suppressed",
}


def _wdl(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(_outcome_label(row["outcome"]) for row in rows)
    total = len(rows)
    score = None
    if total:
        score = (counts["win"] + 0.5 * counts["draw"]) / total
    return {
        "games": total,
        "wins": counts["win"],
        "draws": counts["draw"],
        "losses": counts["loss"],
        "score": score,
    }


def _group_wdl(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: _wdl(groups[name]) for name in sorted(groups)}


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(_require_finite(row[field], field=field) for row in rows) / len(rows)


def _curve_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "temperature",
        "ply",
        "reward_total_mean",
        "chosen_prob_mean",
        "entropy_mean",
        "policy_top1_rate",
        "heuristic_top1_rate",
        "malom_preserving_move_rate",
        "malom_downgrade_move_rate",
        "lr",
        "target_age",
    )
    return {
        field: _mean(rows, field)
        for field in fields
    }


def _validate_game_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_schedule: Mapping[str, Any],
) -> None:
    if len(rows) != FINAL_GAME:
        raise TargetRefreshLrResultError(
            f"expected {FINAL_GAME} game rows, observed {len(rows)}"
        )
    game_ids: set[str] = set()
    observed_schedule: Counter[str] = Counter()
    for expected_game, row in enumerate(rows, start=1):
        missing = _REQUIRED_GAME_FIELDS - set(row)
        if missing:
            raise TargetRefreshLrResultError(
                f"game {expected_game} lacks fields: {sorted(missing)}"
            )
        _validate_finite_tree(row, field=f"game[{expected_game}]")
        if _require_int(row["game"], field="game", minimum=1) != expected_game:
            raise TargetRefreshLrResultError("training games are not exactly 1..100")
        game_id = row["game_id"]
        if not isinstance(game_id, str) or not game_id or game_id in game_ids:
            raise TargetRefreshLrResultError(
                "game identities are invalid or repeated"
            )
        game_ids.add(game_id)
        if row["learner_color"] not in {"W", "B"}:
            raise TargetRefreshLrResultError("learner colour is invalid")
        if row["phase_bucket"] != "main" or row["is_branch"] != 0:
            raise TargetRefreshLrResultError(
                "factorial log contains a branch rollout"
            )
        if row["difficulty"] != 1:
            raise TargetRefreshLrResultError(
                "factorial diagnostic left Sanmill node level one"
            )
        if row["game_type"] not in {"vs_frozen", "vs_sanmill"}:
            raise TargetRefreshLrResultError("opponent source is invalid")
        if row["game_type"] == "vs_sanmill":
            if row["opponent_node_budget"] != 1000:
                raise TargetRefreshLrResultError(
                    "Sanmill node budget is not 1,000"
                )
            source = "sanmill"
        else:
            if row["opponent_node_budget"] is not None:
                raise TargetRefreshLrResultError(
                    "frozen opponent has a node budget"
                )
            source = "frozen"
        colour = "white" if row["learner_color"] == "W" else "black"
        observed_schedule[f"{source}_{colour}"] += 1
        if row["specialist_read_mode"] != "theoretical-only":
            raise TargetRefreshLrResultError(
                "SpecialistDB read projection changed"
            )
        if _require_int(
            row["specialist_read_empirical_suppressed"],
            field="specialist_read_empirical_suppressed",
        ) < 0:
            raise AssertionError("unreachable")
        _outcome_label(row["outcome"])
    wanted = Counter({key: int(value) for key, value in expected_schedule.items()})
    if observed_schedule != wanted:
        raise TargetRefreshLrResultError(
            f"scheduled opponent/colour counts differ: {dict(observed_schedule)}"
        )


def summarize_game_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one 100-game arm and return pre/post stratified metrics."""
    _validate_game_rows(rows, expected_schedule=expected_schedule)
    pre = list(rows[:BOUNDARY_GAME])
    post = list(rows[BOUNDARY_GAME:])

    def source(row: Mapping[str, Any]) -> Any:
        return row["game_type"]

    def colour(row: Mapping[str, Any]) -> Any:
        return row["learner_color"]

    def source_colour(row: Mapping[str, Any]) -> Any:
        return f"{row['game_type']}:{row['learner_color']}"

    def termination(row: Mapping[str, Any]) -> Any:
        return row["termination_reason"]

    return {
        "games": len(rows),
        "windows": {
            "pre_boundary_1_50": {
                "wdl": _wdl(pre),
                "by_opponent_source": _group_wdl(pre, source),
                "by_learner_colour": _group_wdl(pre, colour),
                "by_opponent_and_colour": _group_wdl(pre, source_colour),
                "by_termination_reason": _group_wdl(pre, termination),
                "curve_means": _curve_summary(pre),
            },
            "post_boundary_51_100": {
                "wdl": _wdl(post),
                "by_opponent_source": _group_wdl(post, source),
                "by_learner_colour": _group_wdl(post, colour),
                "by_opponent_and_colour": _group_wdl(post, source_colour),
                "by_termination_reason": _group_wdl(post, termination),
                "curve_means": _curve_summary(post),
            },
        },
        "curves": {
            "interpretation": (
                "observed training diagnostics only; no forecast or held-out "
                "strength curve"
            ),
            "raw": [
                {
                    "game": row["game"],
                    "game_type": row["game_type"],
                    "learner_color": row["learner_color"],
                    "outcome": row["outcome"],
                    "termination_reason": row["termination_reason"],
                    "temperature": row["temperature"],
                    "ply": row["ply"],
                    "reward_total_mean": row["reward_total_mean"],
                    "chosen_prob_mean": row["chosen_prob_mean"],
                    "entropy_mean": row["entropy_mean"],
                    "policy_top1_rate": row["policy_top1_rate"],
                    "heuristic_top1_rate": row["heuristic_top1_rate"],
                    "malom_preserving_move_rate": row[
                        "malom_preserving_move_rate"
                    ],
                    "malom_downgrade_move_rate": row[
                        "malom_downgrade_move_rate"
                    ],
                    "lr": row["lr"],
                    "target_age": row["target_age"],
                }
                for row in rows
            ],
            "validation": {
                "available": False,
                "reason": "ordinary RL run has no supervised validation curve",
            },
        },
    }


def _adaptive_rate_after_boundary(
    rows: Sequence[Mapping[str, Any]], *, base_rate: float
) -> float:
    sanmill = [row for row in rows[:BOUNDARY_GAME] if row["game_type"] == "vs_sanmill"]
    if not sanmill:
        raise TargetRefreshLrResultError(
            "pre-boundary schedule has no Sanmill reference games"
        )
    wins = sum(
        math.isclose(float(row["outcome"]), trainer.WIN_REWARD)
        for row in sanmill
    )
    win_rate = wins / len(sanmill)
    scale = max(
        trainer.LR_SCALE_MIN,
        min(trainer.LR_SCALE_MAX, win_rate / trainer.LR_SCALE_WIN),
    )
    return base_rate * scale


def validate_paired_boundary(
    arms: Sequence[Mapping[str, Any]],
    *,
    base_rate: float,
) -> list[dict[str, Any]]:
    """Prove exact pre-boundary pairing and both post-boundary interventions."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for arm in arms:
        by_seed[int(arm["seed"])][str(arm["condition"])] = arm
    records: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        conditions = by_seed[seed]
        if set(conditions) != set(EXPECTED_CONDITIONS):
            raise TargetRefreshLrResultError(f"seed {seed} factorial is incomplete")
        reference_rows = conditions[EXPECTED_CONDITIONS[0]]["game_rows"]
        reference_pre = canonical_json_bytes(reference_rows[:BOUNDARY_GAME])
        for condition in EXPECTED_CONDITIONS[1:]:
            compared = canonical_json_bytes(
                conditions[condition]["game_rows"][:BOUNDARY_GAME]
            )
            if compared != reference_pre:
                raise TargetRefreshLrResultError(
                    f"seed {seed} arms differ before the intervention boundary"
                )
        adaptive_expected = _adaptive_rate_after_boundary(
            reference_rows,
            base_rate=base_rate,
        )
        observed: dict[str, Any] = {}
        for condition in EXPECTED_CONDITIONS:
            arm = conditions[condition]
            rows = arm["game_rows"]
            game50 = rows[49]
            game51 = rows[50]
            refresh_every, lr_mode = CONDITION_FACTORS[condition]
            if game50["target_age"] != 50 or not math.isclose(
                float(game50["lr"]), base_rate, abs_tol=1e-15
            ):
                raise TargetRefreshLrResultError(
                    f"seed {seed} pre-boundary state differs: {condition}"
                )
            expected_age = 1 if refresh_every == 50 else 51
            expected_lr = base_rate if lr_mode == "fixed" else adaptive_expected
            if game51["target_age"] != expected_age or not math.isclose(
                float(game51["lr"]), expected_lr, abs_tol=1e-15
            ):
                raise TargetRefreshLrResultError(
                    f"seed {seed} intervention did not engage: {condition}"
                )
            observed[condition] = {
                "game_50_target_age": game50["target_age"],
                "game_50_lr": game50["lr"],
                "game_51_target_age": game51["target_age"],
                "game_51_lr": game51["lr"],
            }
        records.append(
            {
                "seed": seed,
                "first_50_games_byte_identical": True,
                "adaptive_expected_game_51_lr": adaptive_expected,
                "conditions": observed,
            }
        )
    return records


def _post_frozen_score(arm: Mapping[str, Any]) -> float:
    summary = arm["metrics"]["games"]["windows"]["post_boundary_51_100"]
    record = summary["by_opponent_source"].get("vs_frozen")
    if not isinstance(record, Mapping) or not isinstance(record.get("score"), float):
        raise TargetRefreshLrResultError(
            f"post-boundary frozen score is absent: {arm['arm_id']}"
        )
    return float(record["score"])


def _support(values: Sequence[float], *, threshold: float) -> dict[str, Any]:
    if len(values) != len(EXPECTED_SEEDS):
        raise TargetRefreshLrResultError("factor contrast count differs")
    same_positive = all(value > 0 for value in values)
    same_negative = all(value < 0 for value in values)
    median_value = median(values)
    supported = (same_positive or same_negative) and abs(median_value) >= threshold
    return {
        "values_by_seed": list(values),
        "median": median_value,
        "same_direction": same_positive or same_negative,
        "direction": (
            "positive" if same_positive else "negative" if same_negative else "mixed"
        ),
        "material_threshold": threshold,
        "supported": supported,
    }


def decide_factorial_result(
    arms: Sequence[Mapping[str, Any]],
    *,
    threshold: float = MINIMUM_MATERIAL_SCORE_EFFECT,
) -> dict[str, Any]:
    """Apply the preregistered two-seed factorial contrasts."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for arm in arms:
        by_seed[int(arm["seed"])][str(arm["condition"])] = arm
    seed_records: list[dict[str, Any]] = []
    refresh_main_values: list[float] = []
    fixed_lr_main_values: list[float] = []
    interaction_values: list[float] = []
    for seed in EXPECTED_SEEDS:
        conditions = by_seed[seed]
        if set(conditions) != set(EXPECTED_CONDITIONS):
            raise TargetRefreshLrResultError(f"seed {seed} factorial is incomplete")
        scores = {name: _post_frozen_score(conditions[name]) for name in conditions}
        refresh_at_adaptive = (
            scores["no-refresh-adaptive"] - scores["refresh-adaptive"]
        )
        refresh_at_fixed = scores["no-refresh-fixed"] - scores["refresh-fixed"]
        fixed_at_refresh = scores["refresh-fixed"] - scores["refresh-adaptive"]
        fixed_at_no_refresh = (
            scores["no-refresh-fixed"] - scores["no-refresh-adaptive"]
        )
        refresh_main = 0.5 * (refresh_at_adaptive + refresh_at_fixed)
        fixed_main = 0.5 * (fixed_at_refresh + fixed_at_no_refresh)
        interaction = fixed_at_no_refresh - fixed_at_refresh
        refresh_main_values.append(refresh_main)
        fixed_lr_main_values.append(fixed_main)
        interaction_values.append(interaction)
        seed_records.append(
            {
                "seed": seed,
                "post_boundary_frozen_scores": scores,
                "no_refresh_minus_refresh_at_adaptive_lr": refresh_at_adaptive,
                "no_refresh_minus_refresh_at_fixed_lr": refresh_at_fixed,
                "no_refresh_main_effect": refresh_main,
                "fixed_minus_adaptive_at_refresh": fixed_at_refresh,
                "fixed_minus_adaptive_at_no_refresh": fixed_at_no_refresh,
                "fixed_lr_main_effect": fixed_main,
                "interaction_difference_in_differences": interaction,
            }
        )
    refresh = _support(refresh_main_values, threshold=threshold)
    fixed_lr = _support(fixed_lr_main_values, threshold=threshold)
    interaction = _support(interaction_values, threshold=threshold)
    supported = [
        name
        for name, record in (
            ("target_refresh", refresh),
            ("learning_rate", fixed_lr),
            ("interaction", interaction),
        )
        if record["supported"]
    ]
    return {
        "classification": (
            "factor_signal_detected" if supported else "inconclusive_short_diagnostic"
        ),
        "supported_terms": supported,
        "seed_results": seed_records,
        "target_refresh": refresh,
        "learning_rate": fixed_lr,
        "interaction": interaction,
        "primary_metric": (
            "games 51-100 learner score against the frozen-model opponent"
        ),
        "claim_boundary": (
            "a supported term permits only a separately frozen successor-design "
            "probe; it does not authorize held-out games or long training"
        ),
    }


def _validate_readiness(
    path: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    readiness = _strict_json(path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise TargetRefreshLrResultError("readiness schema differs")
    identity = readiness.get("readiness_identity")
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise TargetRefreshLrResultError("readiness identity is invalid")
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
    ):
        raise TargetRefreshLrResultError("readiness state differs")
    record = readiness.get("contract")
    if not isinstance(record, Mapping):
        raise TargetRefreshLrResultError("readiness contract record is absent")
    if (
        record.get("plan_identity") != contract["plan_identity"]
        or record.get("file_sha256") != _sha256_file(contract_path)
        or len(readiness.get("arms", ())) != 8
    ):
        raise TargetRefreshLrResultError("readiness binds another design")
    return readiness


def _readiness_arm(
    readiness: Mapping[str, Any], arm_id: str
) -> Mapping[str, Any]:
    matches = [item for item in readiness["arms"] if item.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise TargetRefreshLrResultError(f"readiness arm differs: {arm_id}")
    return matches[0]


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    plan,
    arm: Mapping[str, Any],
    contract: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    if (
        manifest.get("schema_version") != "nmm.run-manifest.v1"
        or manifest.get("git_commit") != plan.git_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("experiment_id") != plan.experiment_id
    ):
        raise TargetRefreshLrResultError("run manifest source identity differs")
    config = manifest.get("resolved_config")
    if not isinstance(config, Mapping):
        raise TargetRefreshLrResultError("run manifest config is absent")
    expected = {
        "seed": arm["seed"],
        "update_target_every": arm["target_refresh_every_games"],
        "lr_adaptation_mode": arm["lr_adaptation_mode"],
        "specialist_read_mode": "theoretical-only",
        "max_games": contract["common_training_contract"]["max_games_schedule"],
        "segment_games": contract["common_training_contract"]["one_segment_games"],
        "segment_stop_game": contract["resources"]["completed_games_per_arm"],
        "start_mode": "fresh",
        "referee_engine": "sanmill",
        "opponent_engine": "sanmill",
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise TargetRefreshLrResultError(
                f"run manifest config differs: {field}"
            )
    if (
        preflight.get("schema_version") != "nmm.generalist-preflight.v1"
        or preflight.get("verdict") != "needs_decision"
        or preflight.get("errors") != []
        or preflight.get("unresolved_decisions")
        != [PRODUCT_AUTHORIZATION_DECISION]
        or preflight.get("resume_config_sha256") != plan.resume_config_sha256
    ):
        raise TargetRefreshLrResultError("readiness preflight content differs")
    policy = manifest.get("checkpoint_policy")
    if not isinstance(policy, Mapping):
        raise TargetRefreshLrResultError("run checkpoint policy is absent")
    mif = policy.get("mifSuite")
    ruleset = policy.get("ruleset")
    runtime = contract["rules_and_runtime"]
    if (
        not isinstance(mif, Mapping)
        or not isinstance(ruleset, Mapping)
        or mif.get("tag") != runtime["mif_tag"]
        or mif.get("releaseCommit") != runtime["mif_release_commit"]
        or mif.get("suiteJcsSha256")
        != "sha256:" + runtime["mif_suite_jcs_sha256"]
        or ruleset.get("semanticDigest") != runtime["rules_semantic_digest"]
    ):
        raise TargetRefreshLrResultError("run protocol identity differs")


def _analyze_arm(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> dict[str, Any]:
    control_dir = _repository_path(root, arm["control_dir"], field="control_dir")
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    plan = load_managed_plan(plan_path)
    _assert_plan_semantics(
        plan,
        root=root,
        contract=contract,
        arm=arm,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    ready_arm = _readiness_arm(readiness, str(arm["arm_id"]))
    if ready_arm.get("plan_sha256") != plan.plan_sha256:
        raise TargetRefreshLrResultError("readiness plan hash differs")
    preflight_record = ready_arm.get("preflight")
    if not isinstance(preflight_record, Mapping):
        raise TargetRefreshLrResultError("arm preflight evidence is absent")
    preflight_path = Path(str(preflight_record.get("path", "")))
    if (
        not preflight_path.is_file()
        or _sha256_file(preflight_path) != preflight_record.get("sha256")
    ):
        raise TargetRefreshLrResultError("arm preflight evidence changed")
    preflight = _strict_json(preflight_path)
    authorization = load_managed_authorization(authorization_path)
    _validate_authorization(authorization, plan)
    completed_details, checkpoint = _validate_controller_completion(plan)
    health = _validate_policy_health(
        plan,
        details=completed_details,
        checkpoint=checkpoint,
    )
    segment = control_dir / "segments" / "segment-0001"
    manifest_path = segment / "run-manifest.json"
    train_log_path = segment / "train_log.jsonl"
    update_log_path = segment / "update_log.jsonl"
    run_events_path = segment / managed.RUN_EVENT_LEDGER_NAME
    manifest = _strict_json(manifest_path)
    _validate_manifest(
        manifest,
        plan=plan,
        arm=arm,
        contract=contract,
        preflight=preflight,
    )
    run_events = managed.load_run_events(run_events_path)
    if not run_events or run_events[-1].status != "completed":
        raise TargetRefreshLrResultError("trainer lifecycle is incomplete")
    game_rows = _strict_jsonl(train_log_path)
    metrics = {
        "games": summarize_game_rows(
            game_rows,
            expected_schedule=contract["resources"]["schedule_counts_by_seed"][
                str(arm["seed"])
            ]["all"],
        ),
        "updates": summarize_update_rows(
            _strict_jsonl(update_log_path),
            expected_games=FINAL_GAME,
        ),
    }
    _validate_finite_tree(metrics, field=f"arm[{arm['arm_id']}].metrics")
    specialist_db = _repository_path(
        root, arm["specialist_db"], field="specialist_db"
    )
    return {
        "arm_id": arm["arm_id"],
        "seed": arm["seed"],
        "condition": arm["condition"],
        "target_refresh_every_games": arm["target_refresh_every_games"],
        "lr_adaptation_mode": arm["lr_adaptation_mode"],
        "source_commit": source_commit,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": _sha256_file(authorization_path),
        "optimizer_updates": metrics["updates"]["updates"],
        "policy_health": health,
        "metrics": metrics,
        "game_rows": game_rows,
        "artifacts": {
            "plan": _artifact_record(root, plan_path),
            "authorization": _artifact_record(root, authorization_path),
            "preflight": _artifact_record(root, preflight_path),
            "run_manifest": _artifact_record(root, manifest_path),
            "run_events": _artifact_record(root, run_events_path),
            "train_log": _artifact_record(root, train_log_path),
            "update_log": _artifact_record(root, update_log_path),
            "checkpoint": _artifact_record(root, checkpoint),
            "specialist_db": _artifact_record(root, specialist_db),
        },
    }


def _inspect_analysis_source(root: Path, expected_commit: str) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise TargetRefreshLrResultError("Git evidence audit failed")
        return result.stdout.strip()

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    origin_dev = git("rev-parse", "origin/dev")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if branch != "dev" or head != origin_dev or head != expected_commit or status:
        raise TargetRefreshLrResultError(
            "result analysis requires the clean published training source"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": origin_dev,
        "training_source_commit": expected_commit,
        "worktree_clean": True,
    }


def analyze_target_refresh_lr_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Analyze the one authorized factorial sequence without changing inputs."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_target_refresh_lr_contract(contract_path)
    readiness = _validate_readiness(
        readiness_path,
        contract=contract,
        contract_path=contract_path,
    )
    source_commit = str(readiness["source"]["head"])
    source = _inspect_analysis_source(root, source_commit)
    arm_records = [
        _analyze_arm(
            root=root,
            contract=contract,
            arm=arm,
            readiness=readiness,
            paths_config=paths_config,
            source_commit=source_commit,
        )
        for arm in _ordered_arms(contract)
    ]
    boundary = validate_paired_boundary(
        arm_records,
        base_rate=float(contract["common_training_contract"]["learning_rate"]),
    )
    decision_arms = [
        {key: value for key, value in arm.items() if key != "game_rows"}
        for arm in arm_records
    ]
    decision = decide_factorial_result(
        decision_arms,
        threshold=float(
            contract["analysis"]["decision_rule"][
                "minimum_material_score_effect"
            ]
        ),
    )
    body = {
        "schema_version": RESULT_SCHEMA,
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "plan_identity": contract["plan_identity"],
            "sha256": _sha256_file(contract_path),
        },
        "readiness": {
            "path": readiness_path.relative_to(root).as_posix(),
            "readiness_identity": readiness["readiness_identity"],
            "sha256": _sha256_file(readiness_path),
        },
        "analysis_source": source,
        "hyperparameters": contract["common_training_contract"],
        "data_and_runtime_versions": {
            "data_contract": contract["data_contract"],
            "rules_and_runtime": contract["rules_and_runtime"],
        },
        "arms": decision_arms,
        "paired_boundary_validation": boundary,
        "decision": decision,
        "claim_boundary": contract["claim_boundary"],
        "interpretation": {
            "observed_fact": (
                "the report contains completed training logs, update curves, "
                "stratified outcomes, policy-health results, and exact boundary "
                "checks"
            ),
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": (
                "same-seed arms are byte-identical through game 50 and differ "
                "only through the preregistered post-boundary factors"
            ),
            "counterevidence": (
                "short endogenous training W/D/L cannot establish held-out "
                "strength or long-run benefit"
            ),
            "next_validation": decision["claim_boundary"],
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def publish_result(path: Path, report: Mapping[str, Any]) -> None:
    """Persist the immutable raw result once without overwriting evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise TargetRefreshLrResultError(
            f"result already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_RESULT",
    "TargetRefreshLrResultError",
    "analyze_target_refresh_lr_result",
    "decide_factorial_result",
    "publish_result",
    "summarize_game_rows",
    "validate_paired_boundary",
]
