"""Fail-closed result analysis for the Malom policy-auxiliary calibration."""

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
    _group_wdl,
    _readiness_arm,
    _require_finite,
    _require_int,
    _sha256_file,
    _strict_json,
    _strict_jsonl,
    _validate_authorization,
    _validate_controller_completion,
    _validate_finite_tree,
    _validate_game_row,
    _validate_manifest,
    _validate_policy_health,
    _validate_schedule_counts,
    _wdl,
)
from learned_ai.training import managed_generalist as managed
from learned_ai.training.managed_generalist import (
    load_managed_authorization,
    load_managed_plan,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.malom_policy_auxiliary_calibration_readiness import (
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT as DEFAULT_READINESS_REPORT,
    READINESS_SCHEMA,
    _assert_plan_semantics,
    _ordered_arms,
    load_calibration_contract,
)
from learned_ai.validation.mill_bonus_ablation_readiness import _repository_path


RESULT_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-calibration-result.v1"
)
DEFAULT_RESULT = Path(
    "out/malom-policy-auxiliary-calibration-smoke-v1/result.json"
)
ROLLING_WINDOW = 50
PHASES = ("place", "move", "fly")
REPETITION_REASON = "draw_threefold_repetition"

_POST_TRAINING_ANALYSIS_PATHS = frozenset(
    {
        "learned_ai/evaluation/"
        "malom_policy_auxiliary_calibration_result.py",
        "scripts/report_malom_policy_auxiliary_calibration.py",
        "tests/test_malom_policy_auxiliary_calibration_result.py",
    }
)

_AUXILIARY_GAME_FIELDS = {
    "malom_action_labelled_move_rate",
    "malom_preserving_action_count_mean",
    "malom_downgrading_action_count_mean",
    "malom_informative_action_set_rate",
    "malom_preserving_probability_mean",
    "malom_known_move_rate",
    "malom_known_place",
    "malom_known_move",
    "malom_known_fly",
    "malom_downgrade_place",
    "malom_downgrade_move",
    "malom_downgrade_fly",
    "malom_downgrade_count",
}

_AUXILIARY_UPDATE_FIELDS = {
    "malom_policy_aux_loss",
    "malom_policy_aux_informative_steps",
    "malom_policy_aux_labelled_steps",
    "malom_policy_aux_mean_preserving_mass",
}

_CURVE_FIELDS = (
    "temperature",
    "ply",
    "outcome",
    "reward_total_mean",
    "chosen_prob_mean",
    "entropy_mean",
    "policy_top1_rate",
    "heuristic_top1_rate",
    "malom_preserving_move_rate",
    "malom_downgrade_move_rate",
    "malom_action_labelled_move_rate",
    "malom_preserving_action_count_mean",
    "malom_downgrading_action_count_mean",
    "malom_informative_action_set_rate",
    "malom_preserving_probability_mean",
)


MalomPolicyAuxiliaryCalibrationResultError = MillBonusAblationResultError


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validate_auxiliary_game_row(
    row: Mapping[str, Any],
    *,
    expected_game: int,
    coefficient: float,
) -> None:
    _validate_game_row(row, expected_game=expected_game)
    missing = _AUXILIARY_GAME_FIELDS - set(row)
    if missing:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            f"game {expected_game} lacks auxiliary fields: {sorted(missing)}"
        )
    known_by_phase: dict[str, int] = {}
    downgrade_by_phase: dict[str, int] = {}
    for phase in PHASES:
        known_by_phase[phase] = _require_int(
            row[f"malom_known_{phase}"], field=f"malom_known_{phase}"
        )
        downgrade_by_phase[phase] = _require_int(
            row[f"malom_downgrade_{phase}"],
            field=f"malom_downgrade_{phase}",
        )
        if downgrade_by_phase[phase] > known_by_phase[phase]:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                f"{phase} selected-action downgrade exceeds support"
            )
    known = sum(known_by_phase.values())
    downgrade = sum(downgrade_by_phase.values())
    steps = _require_int(row["steps"], field="steps", minimum=1)
    known_rate = _require_finite(
        row["malom_known_move_rate"], field="malom_known_move_rate"
    )
    if known != steps or known_rate != 1.0:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "selected-action exact Malom coverage is incomplete"
        )
    if downgrade != _require_int(
        row["malom_downgrade_count"], field="malom_downgrade_count"
    ):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "selected-action downgrade counts do not reconcile"
        )
    logged_rate = _require_finite(
        row["malom_downgrade_move_rate"],
        field="malom_downgrade_move_rate",
    )
    expected_rate = _rate(downgrade, known) or 0.0
    if not math.isclose(logged_rate, expected_rate, abs_tol=1e-12):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "selected-action downgrade rate differs"
        )

    labelled_rate = _require_finite(
        row["malom_action_labelled_move_rate"],
        field="malom_action_labelled_move_rate",
    )
    preserving_count = _require_finite(
        row["malom_preserving_action_count_mean"],
        field="malom_preserving_action_count_mean",
    )
    downgrading_count = _require_finite(
        row["malom_downgrading_action_count_mean"],
        field="malom_downgrading_action_count_mean",
    )
    informative_rate = _require_finite(
        row["malom_informative_action_set_rate"],
        field="malom_informative_action_set_rate",
    )
    preserving_mass = _require_finite(
        row["malom_preserving_probability_mean"],
        field="malom_preserving_probability_mean",
    )
    if not all(
        0.0 <= value <= 1.0
        for value in (labelled_rate, informative_rate, preserving_mass)
    ) or preserving_count < 0.0 or downgrading_count < 0.0:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "game auxiliary diagnostics are outside their domains"
        )
    if coefficient == 0.0:
        if any(
            value != 0.0
            for value in (
                labelled_rate,
                preserving_count,
                downgrading_count,
                informative_rate,
                preserving_mass,
            )
        ):
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "zero-coefficient arm logged auxiliary labels"
            )
    elif labelled_rate != 1.0 or preserving_count < 1.0:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "active auxiliary arm has incomplete game label support"
        )


def _selected_action_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    known_by_phase = {
        phase: sum(int(row[f"malom_known_{phase}"]) for row in rows)
        for phase in PHASES
    }
    downgrade_by_phase = {
        phase: sum(int(row[f"malom_downgrade_{phase}"]) for row in rows)
        for phase in PHASES
    }
    known = sum(known_by_phase.values())
    downgrade = sum(downgrade_by_phase.values())
    return {
        "known_actions": known,
        "downgrading_actions": downgrade,
        "downgrade_rate": _rate(downgrade, known),
        "by_phase": {
            phase: {
                "known_actions": known_by_phase[phase],
                "downgrading_actions": downgrade_by_phase[phase],
                "downgrade_rate": _rate(
                    downgrade_by_phase[phase], known_by_phase[phase]
                ),
            }
            for phase in PHASES
        },
    }


def _group_selected_actions(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {
        name: _selected_action_counts(group)
        for name, group in sorted(groups.items())
    }


def _raw_curve(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "game": int(row["game"]),
            **{field: row[field] for field in _CURVE_FIELDS},
        }
        for row in rows
    ]


def _rolling_curve(
    rows: Sequence[Mapping[str, Any]], window: int = ROLLING_WINDOW
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for end in range(window, len(rows) + 1):
        sample = rows[end - window : end]
        values.append(
            {
                "game": int(sample[-1]["game"]),
                "window_games": window,
                **{
                    field: sum(float(row[field]) for row in sample) / window
                    for field in _CURVE_FIELDS
                },
            }
        )
    return values


def summarize_game_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    coefficient: float,
    expected_games: int,
    expected_schedule_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one 100-game arm and preserve stratified diagnostics."""
    if len(rows) != expected_games:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            f"expected {expected_games} game rows, observed {len(rows)}"
        )
    game_ids: set[str] = set()
    for expected_game, row in enumerate(rows, start=1):
        _validate_auxiliary_game_row(
            row,
            expected_game=expected_game,
            coefficient=coefficient,
        )
        game_id = row["game_id"]
        if not isinstance(game_id, str) or not game_id or game_id in game_ids:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "game identities are invalid or repeated"
            )
        game_ids.add(game_id)
    _validate_schedule_counts(rows, expected_schedule_counts)
    last_50 = list(rows[-50:])

    def opponent(row: Mapping[str, Any]) -> Any:
        return row["game_type"]

    def colour(row: Mapping[str, Any]) -> Any:
        return row["learner_color"]

    def termination(row: Mapping[str, Any]) -> Any:
        return row["termination_reason"]

    termination_counts = Counter(str(row["termination_reason"]) for row in rows)
    return {
        "games": len(rows),
        "selected_action_quality": {
            "whole_run": _selected_action_counts(rows),
            "last_50": _selected_action_counts(last_50),
            "by_opponent_source": _group_selected_actions(rows, opponent),
            "by_learner_colour": _group_selected_actions(rows, colour),
            "by_termination_reason": _group_selected_actions(rows, termination),
        },
        "wdl": {
            "all": _wdl(rows),
            "by_opponent_source": _group_wdl(rows, opponent),
            "by_learner_colour": _group_wdl(rows, colour),
            "by_termination_reason": _group_wdl(rows, termination),
        },
        "termination": {
            "counts": dict(sorted(termination_counts.items())),
            "repetition_draw_rate": (
                termination_counts[REPETITION_REASON] / len(rows)
            ),
        },
        "curves": {
            "interpretation": (
                "observed training diagnostics only; no forecast or held-out "
                "strength curve"
            ),
            "raw": _raw_curve(rows),
            "rolling_50_complete_windows_only": _rolling_curve(rows),
            "validation": {
                "available": False,
                "reason": "ordinary RL run has no supervised validation curve",
            },
        },
    }


def summarize_update_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    coefficient: float,
    expected_games: int,
) -> dict[str, Any]:
    """Validate optimizer updates and quantify auxiliary scale."""
    if not rows:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "optimizer update log is empty"
        )
    previous_game = 0
    raw: list[dict[str, Any]] = []
    absolute_policy_losses: list[float] = []
    auxiliary_losses: list[float] = []
    scaled_auxiliary_losses: list[float] = []
    total_batch_steps = 0
    total_labelled_steps = 0
    total_informative_steps = 0
    for index, row in enumerate(rows, start=1):
        missing = _AUXILIARY_UPDATE_FIELDS - set(row)
        if missing:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                f"update {index} lacks auxiliary fields: {sorted(missing)}"
            )
        _validate_finite_tree(row, field=f"update[{index}]")
        game = _require_int(row.get("game"), field="update.game", minimum=1)
        if game < previous_game or game > expected_games:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "update games are not monotonic"
            )
        previous_game = game
        policy_loss = _require_finite(
            row.get("policy_loss"), field="update.policy_loss"
        )
        value_loss = _require_finite(
            row.get("value_loss"), field="update.value_loss"
        )
        entropy = _require_finite(row.get("entropy"), field="update.entropy")
        learning_rate = _require_finite(row.get("lr"), field="update.lr")
        batch_steps = _require_int(
            row.get("batch_steps"), field="update.batch_steps", minimum=1
        )
        if row.get("reason") not in {"periodic", "final_flush"}:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "update reason is invalid"
            )
        auxiliary_loss = _require_finite(
            row["malom_policy_aux_loss"], field="malom_policy_aux_loss"
        )
        informative_steps = _require_int(
            row["malom_policy_aux_informative_steps"],
            field="malom_policy_aux_informative_steps",
        )
        labelled_steps = _require_int(
            row["malom_policy_aux_labelled_steps"],
            field="malom_policy_aux_labelled_steps",
        )
        preserving_mass = _require_finite(
            row["malom_policy_aux_mean_preserving_mass"],
            field="malom_policy_aux_mean_preserving_mass",
        )
        if (
            auxiliary_loss < 0.0
            or informative_steps > labelled_steps
            or labelled_steps > batch_steps
            or not 0.0 <= preserving_mass <= 1.0
        ):
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "optimizer auxiliary diagnostics are inconsistent"
            )
        if coefficient == 0.0:
            if any(
                value != 0
                for value in (
                    auxiliary_loss,
                    informative_steps,
                    labelled_steps,
                    preserving_mass,
                )
            ):
                raise MalomPolicyAuxiliaryCalibrationResultError(
                    "zero-coefficient update contains auxiliary evidence"
                )
        elif labelled_steps != batch_steps:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "active auxiliary update has incomplete exact labels"
            )
        scaled_auxiliary_loss = coefficient * auxiliary_loss
        raw.append(
            {
                "game": game,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "lr": learning_rate,
                "batch_steps": batch_steps,
                "reason": row["reason"],
                "malom_policy_aux_loss": auxiliary_loss,
                "malom_policy_aux_informative_steps": informative_steps,
                "malom_policy_aux_labelled_steps": labelled_steps,
                "malom_policy_aux_mean_preserving_mass": preserving_mass,
                "scaled_malom_policy_aux_loss": scaled_auxiliary_loss,
            }
        )
        absolute_policy_losses.append(abs(policy_loss))
        auxiliary_losses.append(auxiliary_loss)
        scaled_auxiliary_losses.append(scaled_auxiliary_loss)
        total_batch_steps += batch_steps
        total_labelled_steps += labelled_steps
        total_informative_steps += informative_steps
    median_absolute_policy_loss = median(absolute_policy_losses)
    median_scaled_auxiliary_loss = median(scaled_auxiliary_losses)
    scale_ratio = (
        median_scaled_auxiliary_loss / median_absolute_policy_loss
        if median_absolute_policy_loss > 0.0
        else None
    )
    return {
        "updates": len(raw),
        "raw": raw,
        "summary": {
            "total_batch_steps": total_batch_steps,
            "total_labelled_steps": total_labelled_steps,
            "total_informative_steps": total_informative_steps,
            "label_coverage": total_labelled_steps / total_batch_steps,
            "informative_rate": total_informative_steps / total_batch_steps,
            "median_absolute_policy_loss": median_absolute_policy_loss,
            "median_auxiliary_loss": median(auxiliary_losses),
            "median_scaled_auxiliary_loss": median_scaled_auxiliary_loss,
            "scaled_auxiliary_to_absolute_policy_loss_ratio": scale_ratio,
        },
        "validation": {
            "available": False,
            "reason": "ordinary RL run has no supervised validation updates",
        },
    }


def _policy_health_calibration_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    fixed = report.get("fixed_state_diagnostic")
    if not isinstance(fixed, Mapping):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "full policy-health diagnostic is absent"
        )
    candidate = fixed.get("candidate")
    scratch = fixed.get("scratch")
    if not isinstance(candidate, Mapping) or not isinstance(scratch, Mapping):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "policy-health candidate or scratch metrics are absent"
        )
    candidate_metrics = candidate.get("metrics")
    scratch_metrics = scratch.get("metrics")
    if not isinstance(candidate_metrics, Mapping) or not isinstance(
        scratch_metrics, Mapping
    ):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "policy-health phase metrics are absent"
        )
    required_groups = {"all", "placement", "movement", "flying"}
    if set(candidate_metrics) != required_groups or set(scratch_metrics) != required_groups:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "policy-health phase coverage differs"
        )
    for source in (candidate_metrics, scratch_metrics):
        for group in required_groups:
            metrics = source[group]
            if not isinstance(metrics, Mapping):
                raise MalomPolicyAuxiliaryCalibrationResultError(
                    "policy-health group is invalid"
                )
            _require_finite(
                metrics.get(
                    "critical_value_preserving_probability_mass_scheduled"
                ),
                field=f"policy_health.{group}.preserving_mass",
            )
            _require_finite(
                metrics.get("mean_entropy_scheduled"),
                field=f"policy_health.{group}.entropy",
            )
    return {
        "candidate": {
            group: {
                "critical_value_preserving_probability_mass_scheduled": (
                    candidate_metrics[group][
                        "critical_value_preserving_probability_mass_scheduled"
                    ]
                ),
                "critical_argmax_value_preserving_rate": candidate_metrics[group][
                    "critical_argmax_value_preserving_rate"
                ],
                "critical_mean_preserving_minus_downgrading_logit": (
                    candidate_metrics[group][
                        "critical_mean_preserving_minus_downgrading_logit"
                    ]
                ),
                "mean_entropy_scheduled": candidate_metrics[group][
                    "mean_entropy_scheduled"
                ],
            }
            for group in sorted(required_groups)
        },
        "scratch": {
            group: {
                "critical_value_preserving_probability_mass_scheduled": (
                    scratch_metrics[group][
                        "critical_value_preserving_probability_mass_scheduled"
                    ]
                ),
                "mean_entropy_scheduled": scratch_metrics[group][
                    "mean_entropy_scheduled"
                ],
            }
            for group in sorted(required_groups)
        },
    }


def decide_calibration_result(
    arm_summaries: Sequence[Mapping[str, Any]],
    *,
    decision_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered engineering-scale coefficient gate."""
    ordered = sorted(
        arm_summaries, key=lambda arm: float(arm["malom_policy_aux_coef"])
    )
    if len(ordered) != 4 or [
        float(arm["malom_policy_aux_coef"]) for arm in ordered
    ] != [0.0, 0.03, 0.1, 0.3]:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "calibration arm cohort differs"
        )
    control = ordered[0]
    control_fixed = control["fixed_state_metrics"]["candidate"]["all"]
    control_scratch = control["fixed_state_metrics"]["scratch"]["all"]
    control_mass = float(
        control_fixed[
            "critical_value_preserving_probability_mass_scheduled"
        ]
    )
    control_scratch_mass = float(
        control_scratch[
            "critical_value_preserving_probability_mass_scheduled"
        ]
    )
    control_mass_training_change = control_mass - control_scratch_mass
    control_entropy = float(control_fixed["mean_entropy_scheduled"])
    control_scratch_entropy = float(control_scratch["mean_entropy_scheduled"])
    control_entropy_training_change = control_entropy - control_scratch_entropy
    control_repetition = float(
        control["metrics"]["termination"]["repetition_draw_rate"]
    )
    minimum_mass_gain = float(
        decision_rule["minimum_fixed_state_preserving_mass_gain_over_control"]
    )
    maximum_scale_ratio = float(
        decision_rule[
            "maximum_scaled_auxiliary_to_absolute_policy_loss_ratio"
        ]
    )
    maximum_repetition_increase = float(
        decision_rule["maximum_repetition_draw_rate_increase_over_control"]
    )
    maximum_entropy_drop = float(
        decision_rule["maximum_fixed_state_entropy_drop_over_control"]
    )
    comparisons: list[dict[str, Any]] = []
    for arm in ordered[1:]:
        fixed = arm["fixed_state_metrics"]["candidate"]["all"]
        scratch = arm["fixed_state_metrics"]["scratch"]["all"]
        candidate_mass = float(
            fixed[
                "critical_value_preserving_probability_mass_scheduled"
            ]
        )
        scratch_mass = float(
            scratch[
                "critical_value_preserving_probability_mass_scheduled"
            ]
        )
        mass_training_change = candidate_mass - scratch_mass
        mass_gain = mass_training_change - control_mass_training_change
        candidate_entropy = float(fixed["mean_entropy_scheduled"])
        scratch_entropy = float(scratch["mean_entropy_scheduled"])
        entropy_training_change = candidate_entropy - scratch_entropy
        entropy_drop = (
            control_entropy_training_change - entropy_training_change
        )
        repetition_increase = float(
            arm["metrics"]["termination"]["repetition_draw_rate"]
        ) - control_repetition
        update = arm["optimizer_updates"]["summary"]
        scale_ratio = update[
            "scaled_auxiliary_to_absolute_policy_loss_ratio"
        ]
        checks = {
            "fixed_state_mass_detectable": mass_gain >= minimum_mass_gain,
            "scaled_auxiliary_not_dominant": (
                scale_ratio is not None
                and float(scale_ratio) <= maximum_scale_ratio
            ),
            "complete_update_label_coverage": update["label_coverage"] == 1.0,
            "informative_update_support": update["total_informative_steps"] > 0,
            "fixed_state_entropy_safe": entropy_drop <= maximum_entropy_drop,
            "repetition_rate_safe": (
                repetition_increase <= maximum_repetition_increase
            ),
            "policy_health_passed": arm["policy_health"]["passed"] is True,
        }
        comparisons.append(
            {
                "arm_id": arm["arm_id"],
                "coefficient": arm["malom_policy_aux_coef"],
                "fixed_state_preserving_mass_training_change": (
                    mass_training_change
                ),
                "fixed_state_preserving_mass_gain_over_control_change": (
                    mass_gain
                ),
                "fixed_state_entropy_training_change": entropy_training_change,
                "fixed_state_entropy_additional_drop_over_control_change": (
                    entropy_drop
                ),
                "repetition_draw_rate_increase_over_control": (
                    repetition_increase
                ),
                "scaled_auxiliary_to_absolute_policy_loss_ratio": scale_ratio,
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
    selected = next((item for item in comparisons if item["eligible"]), None)
    return {
        "verdict": (
            "coefficient_selected_for_multiseed_effectiveness_preparation"
            if selected is not None
            else "inconclusive_recalibration_required"
        ),
        "selected_arm_id": selected["arm_id"] if selected else None,
        "selected_coefficient": selected["coefficient"] if selected else None,
        "control": {
            "arm_id": control["arm_id"],
            "scratch_fixed_state_preserving_mass": control_scratch_mass,
            "fixed_state_preserving_mass": control_mass,
            "fixed_state_preserving_mass_training_change": (
                control_mass_training_change
            ),
            "scratch_fixed_state_entropy": control_scratch_entropy,
            "fixed_state_entropy": control_entropy,
            "fixed_state_entropy_training_change": control_entropy_training_change,
            "repetition_draw_rate": control_repetition,
        },
        "thresholds": dict(decision_rule),
        "comparisons": comparisons,
        "claim_boundary": (
            "optimizer-integration coefficient calibration only; a selected "
            "coefficient still requires a separately frozen multi-seed "
            "effectiveness experiment"
        ),
    }


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
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "Git evidence audit failed"
        ) from exc
    return result.stdout.strip()


def _inspect_analysis_source(root: Path, expected_commit: str) -> dict[str, Any]:
    branch = _git_output(root, "branch", "--show-current")
    head = _git_output(root, "rev-parse", "HEAD")
    upstream = _git_output(root, "rev-parse", "origin/dev")
    dirty = _git_output(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if branch != "dev" or head != upstream or dirty:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "result analysis requires a clean published dev"
        )
    changed_paths: list[str] = []
    if head != expected_commit:
        merge_base = _git_output(root, "merge-base", expected_commit, head)
        if merge_base != expected_commit:
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "analysis source does not descend from the training source"
            )
        changed_paths = sorted(
            path
            for path in _git_output(
                root,
                "diff",
                "--name-only",
                f"{expected_commit}..{head}",
                "--",
            ).splitlines()
            if path
        )
        if not changed_paths or not set(changed_paths).issubset(
            _POST_TRAINING_ANALYSIS_PATHS
        ):
            raise MalomPolicyAuxiliaryCalibrationResultError(
                "post-training source changes are not analysis-only"
            )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": upstream,
        "training_source_commit": expected_commit,
        "post_training_analysis_paths": changed_paths,
        "worktree_clean": True,
    }


def _validate_readiness(
    path: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    readiness = _strict_json(path)
    if readiness.get("schema_version") != READINESS_SCHEMA:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "readiness schema differs"
        )
    identity = readiness.get("readiness_identity")
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "readiness identity is invalid"
        )
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
    ):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "readiness state differs"
        )
    contract_record = readiness.get("contract")
    if not isinstance(contract_record, Mapping) or (
        contract_record.get("plan_identity") != contract["plan_identity"]
        or contract_record.get("file_sha256") != _sha256_file(contract_path)
    ):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "readiness binds another contract"
        )
    if len(readiness.get("arms", ())) != 4:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "readiness does not bind four arms"
        )
    return readiness


def _analyze_arm(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> dict[str, Any]:
    control_dir = _repository_path(
        root, arm["control_dir"], field="control_dir"
    )
    plan_path = control_dir / "plan.json"
    authorization_path = control_dir / "authorization.json"
    plan = load_managed_plan(plan_path)
    args = _assert_plan_semantics(
        plan,
        root=root,
        contract=contract,
        arm=arm,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    ready_arm = _readiness_arm(readiness, str(arm["arm_id"]))
    if (
        ready_arm.get("plan_sha256") != plan.plan_sha256
        or ready_arm.get("malom_policy_aux_coef")
        != arm["malom_policy_aux_coef"]
    ):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "readiness arm identity differs"
        )
    preflight = ready_arm.get("preflight")
    if not isinstance(preflight, Mapping):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "arm preflight evidence is absent"
        )
    preflight_path = Path(str(preflight.get("path", "")))
    if (
        not preflight_path.is_file()
        or _sha256_file(preflight_path) != preflight.get("sha256")
    ):
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "arm preflight evidence changed"
        )
    preflight_report = _strict_json(preflight_path)
    authorization = load_managed_authorization(authorization_path)
    _validate_authorization(authorization, plan)
    completed_details, checkpoint = _validate_controller_completion(plan)
    health = _validate_policy_health(
        plan,
        details=completed_details,
        checkpoint=checkpoint,
    )
    full_health_path = Path(str(health["report"]))
    full_health = _strict_json(full_health_path)
    fixed_state_metrics = _policy_health_calibration_metrics(full_health)

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
        preflight=preflight_report,
    )
    config = manifest.get("resolved_config")
    if not isinstance(config, Mapping) or config.get(
        "malom_policy_aux_coef"
    ) != arm["malom_policy_aux_coef"]:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "run manifest auxiliary coefficient differs"
        )
    run_events = managed.load_run_events(run_events_path)
    if not run_events or run_events[-1].event_type != "training_completed":
        raise MalomPolicyAuxiliaryCalibrationResultError(
            "trainer lifecycle is incomplete"
        )
    coefficient = float(arm["malom_policy_aux_coef"])
    metrics = summarize_game_rows(
        _strict_jsonl(train_log_path),
        coefficient=coefficient,
        expected_games=contract["resources"]["completed_games_per_arm"],
        expected_schedule_counts=contract["resources"][
            "schedule_counts_per_arm"
        ],
    )
    update_metrics = summarize_update_rows(
        _strict_jsonl(update_log_path),
        coefficient=coefficient,
        expected_games=contract["resources"]["completed_games_per_arm"],
    )
    specialist_db = Path(args.specialist_db).resolve(strict=True)
    artifacts = {
        "plan": _artifact_record(root, plan_path),
        "authorization": _artifact_record(root, authorization_path),
        "controller_events": _artifact_record(
            root, control_dir / managed.CONTROLLER_LEDGER_NAME
        ),
        "preflight": _artifact_record(root, preflight_path),
        "run_manifest": _artifact_record(root, manifest_path),
        "run_events": _artifact_record(root, run_events_path),
        "train_log": _artifact_record(root, train_log_path),
        "update_log": _artifact_record(root, update_log_path),
        "checkpoint": _artifact_record(root, checkpoint),
        "specialist_db": _artifact_record(root, specialist_db),
        "policy_health": _artifact_record(root, full_health_path),
    }
    return {
        "arm_id": arm["arm_id"],
        "seed": arm["seed"],
        "malom_policy_aux_coef": coefficient,
        "mill_bonus_mode": arm["mill_bonus_mode"],
        "plan_sha256": plan.plan_sha256,
        "authorization_file_sha256": _sha256_file(authorization_path),
        "experiment_id": plan.experiment_id,
        "source_commit": plan.git_commit,
        "schedule_max_games": plan.max_games,
        "completion_game_bound": plan.game_bound,
        "policy_health": health,
        "fixed_state_metrics": fixed_state_metrics,
        "metrics": metrics,
        "optimizer_updates": update_metrics,
        "runtime_identities": {
            "mif": manifest["checkpoint_policy"]["mifSuite"],
            "ruleset": manifest["checkpoint_policy"]["ruleset"],
            "assets": manifest["assets"],
            "experiment_digest": manifest["checkpoint_policy"][
                "experimentDigest"
            ],
            "resume_config_sha256": plan.resume_config_sha256,
        },
        "artifacts": artifacts,
    }


def analyze_calibration_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Validate four completed arms and produce one deterministic result."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_calibration_contract(contract_path)
    readiness = _validate_readiness(
        readiness_path,
        contract=contract,
        contract_path=contract_path,
    )
    source_commit = str(readiness["source"]["head"])
    source = _inspect_analysis_source(root, source_commit)
    arms = [
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
    decision = decide_calibration_result(
        arms,
        decision_rule=contract["analysis"]["decision_rule"],
    )
    report_body = {
        "schema_version": RESULT_SCHEMA,
        "claim_boundary": contract["claim_boundary"],
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "plan_identity": contract["plan_identity"],
            "file_sha256": _sha256_file(contract_path),
        },
        "readiness": {
            "path": readiness_path.relative_to(root).as_posix(),
            "readiness_identity": readiness["readiness_identity"],
            "file_sha256": _sha256_file(readiness_path),
        },
        "analysis_source": source,
        "data_and_runtime_versions": {
            "data_contract": contract["data_contract"],
            "rules_and_runtime": contract["rules_and_runtime"],
        },
        "hyperparameters": contract["common_training_contract"],
        "baseline": {
            "type": "matched fresh zero-coefficient arm",
            "seed": contract["pairing"]["seed"],
            "pairing": contract["pairing"],
        },
        "arms": arms,
        "decision": decision,
        "interpretation": {
            "observation_facts": [
                "All four same-seed 100-game arms completed with finite logs, "
                "exact identities, complete active-arm labels, and passing "
                "fixed-state safety gates.",
                "Loss, entropy, fixed-state, termination, W/D/L, phase, "
                "opponent-source, and colour values are observations rather "
                "than forecasts.",
            ],
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": decision["comparisons"],
            "counter_evidence_and_limits": [
                "Only one fresh seed is used, so this cannot establish a "
                "learning-effect distribution.",
                "The fixed 64-state policy-health corpus is inspected "
                "development data, not held-out validation.",
                "Ordinary RL provides no supervised validation curve.",
                "Training W/D/L is diagnostic and is not a selection metric.",
                "A selected coefficient is not a strength, promotion, "
                "retention, or long-run decision.",
            ],
            "next_verification_experiment": (
                "A selected coefficient may enter a separately frozen "
                "multi-seed effectiveness experiment. If none is selected, "
                "redesign auxiliary normalization before any new training."
            ),
        },
    }
    return {
        **report_body,
        "result_identity": canonical_sha256(report_body),
    }


def publish_result(path: Path, report: Mapping[str, Any]) -> None:
    """Publish one immutable canonical result after all validation passes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise MalomPolicyAuxiliaryCalibrationResultError(
            f"result output already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_READINESS_REPORT",
    "DEFAULT_RESULT",
    "MalomPolicyAuxiliaryCalibrationResultError",
    "RESULT_SCHEMA",
    "analyze_calibration_result",
    "decide_calibration_result",
    "publish_result",
    "summarize_game_rows",
    "summarize_update_rows",
]
