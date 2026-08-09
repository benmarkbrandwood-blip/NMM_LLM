"""Fail-closed result analysis for normalized Malom policy supervision."""

from __future__ import annotations

import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from learned_ai.evaluation.malom_policy_auxiliary_calibration_result import (
    _policy_health_calibration_metrics,
    summarize_game_rows,
)
from learned_ai.evaluation.mill_bonus_ablation_result import (
    MillBonusAblationResultError,
    _artifact_record,
    _readiness_arm,
    _require_finite,
    _require_int,
    _sha256_file,
    _strict_json,
    _strict_jsonl,
    _validate_authorization,
    _validate_controller_completion,
    _validate_finite_tree,
    _validate_manifest,
    _validate_policy_health,
)
from learned_ai.training import managed_generalist as managed
from learned_ai.training.managed_generalist import (
    load_managed_authorization,
    load_managed_plan,
)
from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from learned_ai.validation.malom_policy_auxiliary_normalized_calibration_readiness import (
    DEFAULT_CONTRACT,
    DEFAULT_PATHS_CONFIG,
    DEFAULT_REPORT as DEFAULT_READINESS_REPORT,
    READINESS_SCHEMA,
    _assert_plan_semantics,
    _ordered_arms,
    load_normalized_calibration_contract,
)
from learned_ai.validation.mill_bonus_ablation_readiness import _repository_path


RESULT_SCHEMA = (
    "nmm.sanmill-malom-policy-auxiliary-normalized-calibration-result.v1"
)
DEFAULT_RESULT = Path(
    "out/malom-policy-auxiliary-normalized-calibration-v1/result.json"
)
PHASES = ("placement", "movement", "flying")
UPDATE_ROLLING_WINDOW = 5

_BASIC_UPDATE_FIELDS = {
    "game",
    "policy_loss",
    "value_loss",
    "entropy",
    "lr",
    "batch_steps",
    "reason",
}
_BASIC_AUXILIARY_FIELDS = {
    "malom_policy_aux_loss",
    "malom_policy_aux_informative_steps",
    "malom_policy_aux_labelled_steps",
    "malom_policy_aux_mean_preserving_mass",
    "malom_policy_aux_labelled_by_phase",
    "malom_policy_aux_informative_by_phase",
}
_NORMALIZATION_FIELDS = {
    "malom_policy_aux_mode",
    "malom_policy_aux_scale_status",
    "malom_policy_aux_target_policy_head_ratio",
    "malom_policy_aux_coef_cap",
    "malom_policy_aux_denominator_floor",
    "malom_policy_aux_effective_coef",
    "malom_policy_aux_coefficient_capped",
    "malom_policy_aux_ordinary_policy_head_gradient_l2",
    "malom_policy_aux_raw_auxiliary_gradient_l2",
    "malom_policy_aux_applied_auxiliary_gradient_l2",
    "malom_policy_aux_applied_to_ordinary_policy_head_ratio",
    "malom_policy_auxiliary_to_ordinary_policy_head_cosine",
}
_ALL_AUXILIARY_FIELDS = _BASIC_AUXILIARY_FIELDS | _NORMALIZATION_FIELDS
_SCALE_STATUSES = {
    "normalized",
    "capped",
    "no_informative_steps",
    "ordinary_policy_gradient_below_floor",
}


MalomPolicyAuxiliaryNormalizedCalibrationResultError = (
    MillBonusAblationResultError
)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }


def _phase_counts(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not set(value).issubset(PHASES):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            f"{field} contains an invalid phase"
        )
    return {
        phase: _require_int(value.get(phase, 0), field=f"{field}.{phase}")
        for phase in PHASES
    }


def _basic_update(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    missing = _BASIC_UPDATE_FIELDS - set(row)
    if missing:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            f"update {index} lacks fields: {sorted(missing)}"
        )
    _validate_finite_tree(row, field=f"update[{index}]")
    reason = row["reason"]
    if reason not in {"periodic", "final_flush"}:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "update reason is invalid"
        )
    return {
        "game": _require_int(row["game"], field="update.game", minimum=1),
        "policy_loss": _require_finite(
            row["policy_loss"], field="update.policy_loss"
        ),
        "value_loss": _require_finite(
            row["value_loss"], field="update.value_loss"
        ),
        "entropy": _require_finite(row["entropy"], field="update.entropy"),
        "lr": _require_finite(row["lr"], field="update.lr"),
        "batch_steps": _require_int(
            row["batch_steps"], field="update.batch_steps", minimum=1
        ),
        "reason": reason,
    }


def _rolling_update_curve(
    rows: Sequence[Mapping[str, Any]],
    *,
    window: int = UPDATE_ROLLING_WINDOW,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for end in range(window, len(rows) + 1):
        sample = rows[end - window : end]
        cosines = [
            float(row["malom_policy_auxiliary_to_ordinary_policy_head_cosine"])
            for row in sample
            if row["malom_policy_auxiliary_to_ordinary_policy_head_cosine"]
            is not None
        ]
        values.append(
            {
                "game": sample[-1]["game"],
                "window_updates": window,
                "policy_loss": sum(float(row["policy_loss"]) for row in sample)
                / window,
                "value_loss": sum(float(row["value_loss"]) for row in sample)
                / window,
                "entropy": sum(float(row["entropy"]) for row in sample) / window,
                "malom_policy_aux_loss": sum(
                    float(row["malom_policy_aux_loss"]) for row in sample
                )
                / window,
                "malom_policy_aux_effective_coef": sum(
                    float(row["malom_policy_aux_effective_coef"])
                    for row in sample
                )
                / window,
                "malom_policy_aux_applied_to_ordinary_policy_head_ratio": sum(
                    float(
                        row[
                            "malom_policy_aux_applied_to_ordinary_policy_head_ratio"
                        ]
                    )
                    for row in sample
                )
                / window,
                "malom_policy_auxiliary_to_ordinary_policy_head_cosine": (
                    sum(cosines) / len(cosines) if cosines else None
                ),
                "label_coverage": sum(
                    float(row["malom_policy_aux_labelled_steps"])
                    / float(row["batch_steps"])
                    for row in sample
                )
                / window,
                "informative_rate": sum(
                    float(row["malom_policy_aux_informative_steps"])
                    / float(row["batch_steps"])
                    for row in sample
                )
                / window,
            }
        )
    return values


def _validate_normalized_update(
    row: Mapping[str, Any],
    *,
    index: int,
    basic: Mapping[str, Any],
    target_ratio: float,
    coefficient_cap: float,
    denominator_floor: float,
) -> dict[str, Any]:
    missing = _ALL_AUXILIARY_FIELDS - set(row)
    if missing:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            f"normalized update {index} lacks fields: {sorted(missing)}"
        )
    if row["malom_policy_aux_mode"] != "policy-head-normalized":
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized update mode differs"
        )
    status = row["malom_policy_aux_scale_status"]
    if status not in _SCALE_STATUSES:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized update scale status differs"
        )

    loss = _require_finite(
        row["malom_policy_aux_loss"], field="malom_policy_aux_loss"
    )
    informative = _require_int(
        row["malom_policy_aux_informative_steps"],
        field="malom_policy_aux_informative_steps",
    )
    labelled = _require_int(
        row["malom_policy_aux_labelled_steps"],
        field="malom_policy_aux_labelled_steps",
    )
    preserving_mass = _require_finite(
        row["malom_policy_aux_mean_preserving_mass"],
        field="malom_policy_aux_mean_preserving_mass",
    )
    if (
        loss < 0.0
        or informative > labelled
        or labelled != basic["batch_steps"]
        or not 0.0 <= preserving_mass <= 1.0
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized update label diagnostics are inconsistent"
        )
    labelled_phase = _phase_counts(
        row["malom_policy_aux_labelled_by_phase"],
        field="malom_policy_aux_labelled_by_phase",
    )
    informative_phase = _phase_counts(
        row["malom_policy_aux_informative_by_phase"],
        field="malom_policy_aux_informative_by_phase",
    )
    if sum(labelled_phase.values()) != labelled or sum(
        informative_phase.values()
    ) != informative:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized update phase support does not reconcile"
        )
    if any(
        informative_phase[phase] > labelled_phase[phase] for phase in PHASES
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized informative phase support exceeds labels"
        )

    observed_target = _require_finite(
        row["malom_policy_aux_target_policy_head_ratio"],
        field="malom_policy_aux_target_policy_head_ratio",
    )
    observed_cap = _require_finite(
        row["malom_policy_aux_coef_cap"], field="malom_policy_aux_coef_cap"
    )
    observed_floor = _require_finite(
        row["malom_policy_aux_denominator_floor"],
        field="malom_policy_aux_denominator_floor",
    )
    if (
        observed_target != target_ratio
        or observed_cap != coefficient_cap
        or observed_floor != denominator_floor
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized update scale contract differs"
        )

    coefficient = _require_finite(
        row["malom_policy_aux_effective_coef"],
        field="malom_policy_aux_effective_coef",
    )
    ordinary_norm = _require_finite(
        row["malom_policy_aux_ordinary_policy_head_gradient_l2"],
        field="malom_policy_aux_ordinary_policy_head_gradient_l2",
    )
    raw_norm = _require_finite(
        row["malom_policy_aux_raw_auxiliary_gradient_l2"],
        field="malom_policy_aux_raw_auxiliary_gradient_l2",
    )
    applied_norm = _require_finite(
        row["malom_policy_aux_applied_auxiliary_gradient_l2"],
        field="malom_policy_aux_applied_auxiliary_gradient_l2",
    )
    applied_ratio = _require_finite(
        row["malom_policy_aux_applied_to_ordinary_policy_head_ratio"],
        field="malom_policy_aux_applied_to_ordinary_policy_head_ratio",
    )
    capped = row["malom_policy_aux_coefficient_capped"]
    if not isinstance(capped, bool):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized coefficient-capped flag is invalid"
        )
    if (
        coefficient < 0.0
        or coefficient > coefficient_cap
        or ordinary_norm < 0.0
        or raw_norm < 0.0
        or applied_norm < 0.0
        or applied_ratio < 0.0
        or not math.isclose(
            applied_norm,
            coefficient * raw_norm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized gradient scale diagnostics are inconsistent"
        )
    expected_ratio = (
        applied_norm / ordinary_norm
        if ordinary_norm > denominator_floor
        else 0.0
    )
    if not math.isclose(
        applied_ratio, expected_ratio, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized applied ratio does not reconcile"
        )

    raw_cosine = row["malom_policy_auxiliary_to_ordinary_policy_head_cosine"]
    cosine = None
    if raw_cosine is not None:
        cosine = _require_finite(raw_cosine, field="malom_policy_aux_cosine")
        if not -1.000001 <= cosine <= 1.000001:
            raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
                "normalized gradient cosine is outside its domain"
            )

    if status == "no_informative_steps":
        valid_status = (
            informative == 0
            and loss == 0.0
            and coefficient == 0.0
            and raw_norm == 0.0
            and applied_norm == 0.0
            and applied_ratio == 0.0
            and cosine is None
            and not capped
        )
    elif status == "ordinary_policy_gradient_below_floor":
        valid_status = (
            informative > 0
            and ordinary_norm <= denominator_floor
            and raw_norm > denominator_floor
            and coefficient == 0.0
            and applied_norm == 0.0
            and applied_ratio == 0.0
            and not capped
        )
    elif status == "normalized":
        valid_status = (
            informative > 0
            and ordinary_norm > denominator_floor
            and raw_norm > denominator_floor
            and not capped
            and cosine is not None
            and math.isclose(
                applied_ratio, target_ratio, rel_tol=1e-9, abs_tol=1e-12
            )
        )
    else:
        valid_status = (
            informative > 0
            and ordinary_norm > denominator_floor
            and raw_norm > denominator_floor
            and capped
            and coefficient == coefficient_cap
            and cosine is not None
            and applied_ratio <= target_ratio
        )
    if not valid_status:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            f"normalized update status is inconsistent: {status}"
        )

    return {
        **basic,
        "malom_policy_aux_loss": loss,
        "malom_policy_aux_informative_steps": informative,
        "malom_policy_aux_labelled_steps": labelled,
        "malom_policy_aux_mean_preserving_mass": preserving_mass,
        "malom_policy_aux_labelled_by_phase": labelled_phase,
        "malom_policy_aux_informative_by_phase": informative_phase,
        "malom_policy_aux_mode": "policy-head-normalized",
        "malom_policy_aux_scale_status": status,
        "malom_policy_aux_target_policy_head_ratio": observed_target,
        "malom_policy_aux_coef_cap": observed_cap,
        "malom_policy_aux_denominator_floor": observed_floor,
        "malom_policy_aux_effective_coef": coefficient,
        "malom_policy_aux_coefficient_capped": capped,
        "malom_policy_aux_ordinary_policy_head_gradient_l2": ordinary_norm,
        "malom_policy_aux_raw_auxiliary_gradient_l2": raw_norm,
        "malom_policy_aux_applied_auxiliary_gradient_l2": applied_norm,
        "malom_policy_aux_applied_to_ordinary_policy_head_ratio": applied_ratio,
        "malom_policy_auxiliary_to_ordinary_policy_head_cosine": cosine,
    }


def summarize_normalized_update_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    normalized: bool,
    expected_games: int,
    target_ratio: float,
    coefficient_cap: float,
    denominator_floor: float,
) -> dict[str, Any]:
    """Validate update logs and retain raw normalized-scale diagnostics."""
    if not rows:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "optimizer update log is empty"
        )
    raw: list[dict[str, Any]] = []
    previous_game = 0
    total_steps = 0
    total_labelled = 0
    total_informative = 0
    effective_coefficients: list[float] = []
    applied_ratios: list[float] = []
    cosines: list[float] = []
    ordinary_norms: list[float] = []
    raw_auxiliary_norms: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    phase_labelled = Counter({phase: 0 for phase in PHASES})
    phase_informative = Counter({phase: 0 for phase in PHASES})
    statuses: Counter[str] = Counter()
    capped_updates = 0

    for index, row in enumerate(rows, start=1):
        basic = _basic_update(row, index=index)
        if basic["game"] < previous_game or basic["game"] > expected_games:
            raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
                "update games are not monotonic"
            )
        previous_game = int(basic["game"])
        policy_losses.append(float(basic["policy_loss"]))
        value_losses.append(float(basic["value_loss"]))
        entropies.append(float(basic["entropy"]))
        total_steps += int(basic["batch_steps"])

        if normalized:
            record = _validate_normalized_update(
                row,
                index=index,
                basic=basic,
                target_ratio=target_ratio,
                coefficient_cap=coefficient_cap,
                denominator_floor=denominator_floor,
            )
            total_labelled += int(record["malom_policy_aux_labelled_steps"])
            total_informative += int(
                record["malom_policy_aux_informative_steps"]
            )
            phase_labelled.update(record["malom_policy_aux_labelled_by_phase"])
            phase_informative.update(
                record["malom_policy_aux_informative_by_phase"]
            )
            effective_coefficients.append(
                float(record["malom_policy_aux_effective_coef"])
            )
            applied_ratios.append(
                float(
                    record[
                        "malom_policy_aux_applied_to_ordinary_policy_head_ratio"
                    ]
                )
            )
            ordinary_norms.append(
                float(
                    record[
                        "malom_policy_aux_ordinary_policy_head_gradient_l2"
                    ]
                )
            )
            raw_auxiliary_norms.append(
                float(record["malom_policy_aux_raw_auxiliary_gradient_l2"])
            )
            cosine = record[
                "malom_policy_auxiliary_to_ordinary_policy_head_cosine"
            ]
            if cosine is not None:
                cosines.append(float(cosine))
            statuses[str(record["malom_policy_aux_scale_status"])] += 1
            capped_updates += int(
                bool(record["malom_policy_aux_coefficient_capped"])
            )
        else:
            present = _ALL_AUXILIARY_FIELDS & set(row)
            if present:
                raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
                    "control update contains auxiliary diagnostics: "
                    f"{sorted(present)}"
                )
            record = {
                **basic,
                "malom_policy_aux_mode": "fixed",
                "malom_policy_aux_scale_status": "disabled",
                "malom_policy_aux_loss": 0.0,
                "malom_policy_aux_informative_steps": 0,
                "malom_policy_aux_labelled_steps": 0,
                "malom_policy_aux_mean_preserving_mass": 0.0,
                "malom_policy_aux_labelled_by_phase": {
                    phase: 0 for phase in PHASES
                },
                "malom_policy_aux_informative_by_phase": {
                    phase: 0 for phase in PHASES
                },
                "malom_policy_aux_effective_coef": 0.0,
                "malom_policy_aux_applied_to_ordinary_policy_head_ratio": 0.0,
                "malom_policy_auxiliary_to_ordinary_policy_head_cosine": None,
            }
            statuses["disabled"] += 1
        raw.append(record)

    return {
        "updates": len(raw),
        "raw": raw,
        "curves": {
            "interpretation": (
                "observed optimizer diagnostics only; incomplete leading "
                "windows are omitted"
            ),
            "raw": raw,
            "rolling_5_complete_windows_only": _rolling_update_curve(raw),
        },
        "summary": {
            "total_batch_steps": total_steps,
            "total_labelled_steps": total_labelled,
            "total_informative_steps": total_informative,
            "label_coverage": total_labelled / total_steps,
            "informative_rate": total_informative / total_steps,
            "status_counts": dict(sorted(statuses.items())),
            "capped_updates": capped_updates,
            "effective_coefficient": _distribution(effective_coefficients),
            "applied_to_ordinary_policy_head_ratio": _distribution(
                applied_ratios
            ),
            "ordinary_policy_head_gradient_l2": _distribution(ordinary_norms),
            "raw_auxiliary_gradient_l2": _distribution(raw_auxiliary_norms),
            "auxiliary_to_ordinary_policy_head_cosine": _distribution(cosines),
            "labelled_steps_by_phase": dict(phase_labelled),
            "informative_steps_by_phase": dict(phase_informative),
            "policy_loss": _distribution(policy_losses),
            "value_loss": _distribution(value_losses),
            "entropy": _distribution(entropies),
        },
        "validation": {
            "available": False,
            "reason": "ordinary RL run has no supervised validation updates",
        },
    }


def _fixed_delta(arm: Mapping[str, Any]) -> dict[str, float]:
    candidate = arm["fixed_state_metrics"]["candidate"]["all"]
    scratch = arm["fixed_state_metrics"]["scratch"]["all"]
    mass_field = "critical_value_preserving_probability_mass_scheduled"
    candidate_mass = float(candidate[mass_field])
    scratch_mass = float(scratch[mass_field])
    candidate_entropy = float(candidate["mean_entropy_scheduled"])
    scratch_entropy = float(scratch["mean_entropy_scheduled"])
    return {
        "candidate_preserving_mass": candidate_mass,
        "scratch_preserving_mass": scratch_mass,
        "preserving_mass_training_change": candidate_mass - scratch_mass,
        "candidate_entropy": candidate_entropy,
        "scratch_entropy": scratch_entropy,
        "entropy_training_change": candidate_entropy - scratch_entropy,
    }


def decide_normalized_calibration_result(
    arm_summaries: Sequence[Mapping[str, Any]],
    *,
    decision_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen paired three-seed mechanism decision."""
    if len(arm_summaries) != 6:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized calibration arm cohort differs"
        )
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for arm in arm_summaries:
        seed = _require_int(arm.get("seed"), field="arm.seed")
        condition = arm.get("condition")
        if seed not in {55, 56, 57} or condition not in {
            "control",
            "normalized-0.25",
        }:
            raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
                "normalized calibration seed or condition differs"
            )
        key = (seed, str(condition))
        if key in indexed:
            raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
                "normalized calibration arm is duplicated"
            )
        indexed[key] = arm
    expected = {
        (seed, condition)
        for seed in (55, 56, 57)
        for condition in ("control", "normalized-0.25")
    }
    if set(indexed) != expected:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "normalized calibration pairing differs"
        )

    maximum_ratio = float(decision_rule["normalized_applied_ratio_upper_bound"])
    maximum_entropy_drop = float(
        decision_rule["maximum_fixed_state_entropy_drop_over_control"]
    )
    maximum_repetition_increase = float(
        decision_rule["maximum_repetition_draw_rate_increase_over_control"]
    )
    minimum_gain = float(
        decision_rule["minimum_fixed_state_preserving_mass_median_gain"]
    )
    minimum_positive = _require_int(
        decision_rule["minimum_positive_seed_pairs"],
        field="minimum_positive_seed_pairs",
        minimum=1,
    )
    if decision_rule.get("training_wdl_is_not_a_selection_metric") is not True:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "training W/D/L decision boundary differs"
        )

    pairs: list[dict[str, Any]] = []
    gains: list[float] = []
    for seed in (55, 56, 57):
        control = indexed[(seed, "control")]
        treatment = indexed[(seed, "normalized-0.25")]
        control_fixed = _fixed_delta(control)
        treatment_fixed = _fixed_delta(treatment)
        mass_gain = (
            treatment_fixed["preserving_mass_training_change"]
            - control_fixed["preserving_mass_training_change"]
        )
        entropy_drop = (
            control_fixed["entropy_training_change"]
            - treatment_fixed["entropy_training_change"]
        )
        repetition_increase = float(
            treatment["metrics"]["termination"]["repetition_draw_rate"]
        ) - float(control["metrics"]["termination"]["repetition_draw_rate"])
        update = treatment["optimizer_updates"]["summary"]
        observed_max_ratio = update[
            "applied_to_ordinary_policy_head_ratio"
        ]["max"]
        checks = {
            "complete_update_label_coverage": update["label_coverage"] == 1.0,
            "informative_update_support": update["total_informative_steps"] > 0,
            "normalized_ratio_bounded": (
                observed_max_ratio is not None
                and float(observed_max_ratio) <= maximum_ratio
            ),
            "fixed_state_entropy_safe": entropy_drop <= maximum_entropy_drop,
            "repetition_rate_safe": (
                repetition_increase <= maximum_repetition_increase
            ),
            "control_policy_health_passed": (
                control["policy_health"]["passed"] is True
            ),
            "treatment_policy_health_passed": (
                treatment["policy_health"]["passed"] is True
            ),
        }
        gains.append(mass_gain)
        pairs.append(
            {
                "seed": seed,
                "control_arm_id": control["arm_id"],
                "treatment_arm_id": treatment["arm_id"],
                "control_fixed_state": control_fixed,
                "treatment_fixed_state": treatment_fixed,
                "paired_preserving_mass_gain": mass_gain,
                "positive_gain": mass_gain > 0.0,
                "paired_additional_entropy_drop": entropy_drop,
                "repetition_draw_rate_increase": repetition_increase,
                "maximum_applied_gradient_ratio": observed_max_ratio,
                "checks": checks,
                "safety_eligible": all(checks.values()),
            }
        )

    positive_pairs = sum(item["positive_gain"] for item in pairs)
    median_gain = median(gains)
    safety_passed = all(item["safety_eligible"] for item in pairs)
    eligible = (
        safety_passed
        and positive_pairs >= minimum_positive
        and median_gain >= minimum_gain
    )
    return {
        "verdict": (
            "normalized_mechanism_eligible_for_effectiveness_experiment"
            if eligible
            else "inconclusive_stop_and_redesign"
        ),
        "eligible": eligible,
        "positive_seed_pairs": positive_pairs,
        "median_paired_preserving_mass_gain": median_gain,
        "safety_gates_passed": safety_passed,
        "thresholds": dict(decision_rule),
        "pairs": pairs,
        "training_wdl_used_for_selection": False,
        "claim_boundary": (
            "optimizer-integration mechanism calibration only; eligibility "
            "permits designing, but not launching, a later effectiveness "
            "experiment"
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
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
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
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "result analysis requires a clean published dev"
        )
    if head != expected_commit:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "result analysis must use the exact training source commit"
        )
    return {
        "branch": branch,
        "head": head,
        "origin_dev": upstream,
        "training_source_commit": expected_commit,
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
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness schema differs"
        )
    identity = readiness.get("readiness_identity")
    body = dict(readiness)
    body.pop("readiness_identity", None)
    if identity != canonical_sha256(body):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness identity is invalid"
        )
    if (
        readiness.get("state") != "ready_for_product_authorization"
        or readiness.get("launch_authorized") is not False
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness state differs"
        )
    record = readiness.get("contract")
    if not isinstance(record, Mapping) or (
        record.get("plan_identity") != contract["plan_identity"]
        or record.get("file_sha256") != _sha256_file(contract_path)
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness binds another contract"
        )
    if readiness.get("result_analysis") != contract["analysis"][
        "result_implementation"
    ]:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness result-analyzer identity differs"
        )
    if len(readiness.get("arms", ())) != 6:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness does not bind six arms"
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
        or ready_arm.get("malom_policy_aux_mode")
        != arm["malom_policy_aux_mode"]
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "readiness arm identity differs"
        )
    preflight = ready_arm.get("preflight")
    if not isinstance(preflight, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "arm preflight evidence is absent"
        )
    preflight_path = Path(str(preflight.get("path", "")))
    if (
        not preflight_path.is_file()
        or _sha256_file(preflight_path) != preflight.get("sha256")
    ):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
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
    fixed_state_metrics = _policy_health_calibration_metrics(
        _strict_json(full_health_path)
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
        preflight=preflight_report,
    )
    config = manifest.get("resolved_config")
    if not isinstance(config, Mapping):
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "run manifest config is absent"
        )
    expected_config = {
        "malom_policy_aux_coef": arm["malom_policy_aux_coef"],
        "malom_policy_aux_mode": arm["malom_policy_aux_mode"],
        "malom_policy_aux_target_ratio": arm[
            "malom_policy_aux_target_ratio"
        ],
        "malom_policy_aux_coef_cap": arm["malom_policy_aux_coef_cap"],
        "malom_policy_aux_denominator_floor": arm[
            "malom_policy_aux_denominator_floor"
        ],
    }
    for field, value in expected_config.items():
        if config.get(field) != value:
            raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
                f"run manifest normalized config differs: {field}"
            )
    run_events = managed.load_run_events(run_events_path)
    if not run_events or run_events[-1].event_type != "training_completed":
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            "trainer lifecycle is incomplete"
        )

    normalized = arm["malom_policy_aux_mode"] == "policy-head-normalized"
    schedule = contract["resources"]["schedule_counts_by_seed"][
        str(arm["seed"])
    ]
    metrics = summarize_game_rows(
        _strict_jsonl(train_log_path),
        coefficient=1.0 if normalized else 0.0,
        expected_games=contract["resources"]["completed_games_per_arm"],
        expected_schedule_counts=schedule,
    )
    updates = summarize_normalized_update_rows(
        _strict_jsonl(update_log_path),
        normalized=normalized,
        expected_games=contract["resources"]["completed_games_per_arm"],
        target_ratio=float(arm["malom_policy_aux_target_ratio"]),
        coefficient_cap=float(arm["malom_policy_aux_coef_cap"]),
        denominator_floor=float(arm["malom_policy_aux_denominator_floor"]),
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
        "condition": arm["condition"],
        "seed": arm["seed"],
        "malom_policy_aux_mode": arm["malom_policy_aux_mode"],
        "plan_sha256": plan.plan_sha256,
        "authorization_file_sha256": _sha256_file(authorization_path),
        "experiment_id": plan.experiment_id,
        "source_commit": plan.git_commit,
        "schedule_max_games": plan.max_games,
        "completion_game_bound": plan.game_bound,
        "policy_health": health,
        "fixed_state_metrics": fixed_state_metrics,
        "metrics": metrics,
        "optimizer_updates": updates,
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


def analyze_normalized_calibration_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Validate six completed arms and produce one deterministic result."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_normalized_calibration_contract(contract_path)
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
    decision = decide_normalized_calibration_result(
        arms,
        decision_rule=contract["analysis"]["decision_rule"],
    )
    body = {
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
            "type": "matched fresh auxiliary-off control for each seed",
            "seeds": contract["pairing"]["seeds"],
            "pairing": contract["pairing"],
        },
        "arms": arms,
        "decision": decision,
        "interpretation": {
            "observation_facts": [
                "All six paired 100-game arms completed with finite logs, "
                "exact identities, and passing safety gates.",
                "Training curves, normalization diagnostics, fixed-state "
                "changes, termination classes and W/D/L are observations, "
                "not forecasts.",
            ],
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": decision["pairs"],
            "counter_evidence_and_limits": [
                "Three short seeds estimate mechanism repeatability but do "
                "not establish long-run learning or playing strength.",
                "The fixed 29-state corpus is inspected development data, "
                "not held-out validation.",
                "Ordinary RL provides no supervised validation-loss curve.",
                "Training W/D/L is stratified diagnostic evidence and is not "
                "a selection metric.",
                "Eligibility is not promotion, publication, retention or "
                "authority to launch another experiment.",
            ],
            "next_verification_experiment": (
                "If eligible, design and separately authorize a bounded "
                "multi-seed effectiveness experiment. Otherwise stop and "
                "redesign normalization."
            ),
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


def publish_result(path: Path, report: Mapping[str, Any]) -> None:
    """Publish one immutable canonical result after all validation passes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    except FileExistsError as exc:
        raise MalomPolicyAuxiliaryNormalizedCalibrationResultError(
            f"result output already exists: {path}"
        ) from exc


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_PATHS_CONFIG",
    "DEFAULT_READINESS_REPORT",
    "DEFAULT_RESULT",
    "MalomPolicyAuxiliaryNormalizedCalibrationResultError",
    "RESULT_SCHEMA",
    "analyze_normalized_calibration_result",
    "decide_normalized_calibration_result",
    "publish_result",
    "summarize_normalized_update_rows",
]
