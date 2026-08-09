"""Fail-closed result analysis for the Malom downgrade-penalty ablation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from learned_ai.evaluation import mill_bonus_ablation_result as base
from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.malom_downgrade_penalty_probe import (
    CONTROL_MODE,
    TREATMENT_MODE,
)
from learned_ai.validation.mill_bonus_ablation_readiness import (
    _ordered_arms,
    load_ablation_contract,
)
from scripts import train_s_gen_v2 as trainer


RESULT_SCHEMA = "nmm.sanmill-malom-downgrade-penalty-ablation-result.v1"
DEFAULT_CONTRACT = Path(
    "docs/experiments/sanmill-malom-downgrade-penalty-ablation-smoke-v1.json"
)
DEFAULT_READINESS_REPORT = Path(
    "out/malom-downgrade-penalty-ablation-smoke-v1/readiness.json"
)
DEFAULT_RESULT = Path(
    "out/malom-downgrade-penalty-ablation-smoke-v1/result.json"
)
PHASES = ("place", "move", "fly")
TAIL_FIRST_GAME = 301
TAIL_LAST_GAME = 500


MalomDowngradePenaltyResultError = base.MillBonusAblationResultError


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MalomDowngradePenaltyResultError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _require_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalomDowngradePenaltyResultError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MalomDowngradePenaltyResultError(f"{field} must be finite")
    return result


def _reward_counts(row: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    required = {
        "reward_malom_mean",
        "malom_downgrade_count",
        "malom_downgrade_rank_total",
        "malom_reward_total",
        *(f"malom_known_{phase}" for phase in PHASES),
        *(f"malom_downgrade_{phase}" for phase in PHASES),
    }
    missing = required - set(row)
    if missing:
        raise MalomDowngradePenaltyResultError(
            f"penalty log lacks fields: {sorted(missing)}"
        )
    steps = _require_int(row.get("steps"), field="steps")
    known_by_phase = {
        phase: _require_int(
            row[f"malom_known_{phase}"], field=f"malom_known_{phase}"
        )
        for phase in PHASES
    }
    downgrade_by_phase = {
        phase: _require_int(
            row[f"malom_downgrade_{phase}"],
            field=f"malom_downgrade_{phase}",
        )
        for phase in PHASES
    }
    known = sum(known_by_phase.values())
    downgrade = _require_int(
        row["malom_downgrade_count"], field="malom_downgrade_count"
    )
    rank_total = _require_int(
        row["malom_downgrade_rank_total"],
        field="malom_downgrade_rank_total",
    )
    if known != steps:
        raise MalomDowngradePenaltyResultError(
            "exact Malom support does not cover every learner action"
        )
    if sum(downgrade_by_phase.values()) != downgrade or any(
        downgrade_by_phase[phase] > known_by_phase[phase] for phase in PHASES
    ):
        raise MalomDowngradePenaltyResultError(
            "phase downgrade counts do not reconcile"
        )
    if not downgrade <= rank_total <= 2 * downgrade:
        raise MalomDowngradePenaltyResultError(
            "Malom downgrade rank total is outside the exact WDL range"
        )
    expected_reward = (
        -trainer.MALOM_DOWNGRADE_PENALTY * rank_total
        if mode == TREATMENT_MODE
        else 0.0
    )
    reward_total = _require_finite(
        row["malom_reward_total"], field="malom_reward_total"
    )
    reward_mean = _require_finite(
        row["reward_malom_mean"], field="reward_malom_mean"
    )
    if not math.isclose(reward_total, expected_reward, abs_tol=1e-12):
        raise MalomDowngradePenaltyResultError(
            "logged Malom reward total differs from the selected mode"
        )
    expected_mean = reward_total / steps if steps else 0.0
    if not math.isclose(reward_mean, expected_mean, abs_tol=1e-12):
        raise MalomDowngradePenaltyResultError(
            "logged Malom reward mean does not reconcile"
        )
    logged_rate = _require_finite(
        row.get("malom_downgrade_move_rate"),
        field="malom_downgrade_move_rate",
    )
    expected_rate = downgrade / known if known else 0.0
    if not math.isclose(logged_rate, expected_rate, abs_tol=1e-12):
        raise MalomDowngradePenaltyResultError(
            "logged all-action downgrade rate does not reconcile"
        )
    return {
        "known_actions": known,
        "downgrade_actions": downgrade,
        "downgrade_rank_total": rank_total,
        "malom_reward_total": reward_total,
        "known_by_phase": known_by_phase,
        "downgrade_by_phase": downgrade_by_phase,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    known = 0
    downgrade = 0
    rank_total = 0
    reward_total = 0.0
    known_by_phase = {phase: 0 for phase in PHASES}
    downgrade_by_phase = {phase: 0 for phase in PHASES}
    for row in rows:
        counts = _reward_counts(row, mode=mode)
        known += counts["known_actions"]
        downgrade += counts["downgrade_actions"]
        rank_total += counts["downgrade_rank_total"]
        reward_total += counts["malom_reward_total"]
        for phase in PHASES:
            known_by_phase[phase] += counts["known_by_phase"][phase]
            downgrade_by_phase[phase] += counts["downgrade_by_phase"][phase]
    return {
        "known_actions": known,
        "downgrade_actions": downgrade,
        "downgrade_rank_total": rank_total,
        "rate": downgrade / known if known else None,
        "malom_reward_total": reward_total,
        "by_phase": {
            phase: {
                "known_actions": known_by_phase[phase],
                "downgrade_actions": downgrade_by_phase[phase],
                "rate": (
                    downgrade_by_phase[phase] / known_by_phase[phase]
                    if known_by_phase[phase]
                    else None
                ),
            }
            for phase in PHASES
        },
    }


def _group_aggregate(
    rows: Sequence[Mapping[str, Any]], *, mode: str, key
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {
        name: _aggregate(groups[name], mode=mode) for name in sorted(groups)
    }


def summarize_penalty_rows(
    rows: Sequence[Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    """Validate new observability and summarize the frozen primary window."""
    if mode not in (CONTROL_MODE, TREATMENT_MODE):
        raise MalomDowngradePenaltyResultError("unsupported penalty arm mode")
    tail = [
        row
        for row in rows
        if TAIL_FIRST_GAME <= int(row["game"]) <= TAIL_LAST_GAME
    ]
    if len(rows) != 500 or len(tail) != 200:
        raise MalomDowngradePenaltyResultError(
            "penalty ablation requires 500 games and a complete 200-game tail"
        )
    return {
        "whole_run": _aggregate(rows, mode=mode),
        "tail_301_500": _aggregate(tail, mode=mode),
        "tail_by_opponent_source": _group_aggregate(
            tail, mode=mode, key=lambda row: row["game_type"]
        ),
        "tail_by_learner_colour": _group_aggregate(
            tail, mode=mode, key=lambda row: row["learner_color"]
        ),
        "tail_by_termination_reason": _group_aggregate(
            tail, mode=mode, key=lambda row: row["termination_reason"]
        ),
    }


def decide_penalty_result(
    arm_summaries: Sequence[Mapping[str, Any]],
    *,
    material_reduction: float,
    minimum_tail_support: int,
    maximum_seed_harm: float,
) -> dict[str, Any]:
    """Apply the frozen paired-seed all-action downgrade rule."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for arm in arm_summaries:
        by_seed[int(arm["seed"])][str(arm["mill_bonus_mode"])] = arm
    paired: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        modes = by_seed[seed]
        if set(modes) != {CONTROL_MODE, TREATMENT_MODE}:
            raise MalomDowngradePenaltyResultError(
                f"seed {seed} penalty pair is incomplete"
            )
        control = modes[CONTROL_MODE]["penalty_metrics"]["tail_301_500"]
        treatment = modes[TREATMENT_MODE]["penalty_metrics"]["tail_301_500"]
        if control["rate"] is None or treatment["rate"] is None:
            reduction = None
        else:
            reduction = control["rate"] - treatment["rate"]
        paired.append(
            {
                "seed": seed,
                "control": control,
                "treatment": treatment,
                "control_minus_treatment_rate": reduction,
            }
        )
    reductions = [
        pair["control_minus_treatment_rate"]
        for pair in paired
        if pair["control_minus_treatment_rate"] is not None
    ]
    treatment_arms = [
        arm for arm in arm_summaries if arm["mill_bonus_mode"] == TREATMENT_MODE
    ]
    treatment_safe = all(
        bool(arm["policy_health"]["passed"]) for arm in treatment_arms
    )
    support_pass = all(
        arm["penalty_metrics"]["tail_301_500"]["known_actions"]
        >= minimum_tail_support
        for arm in arm_summaries
    )
    directions = sum(value > 0.0 for value in reductions)
    no_material_seed_harm = all(value >= -maximum_seed_harm for value in reductions)
    median_reduction = (
        median(reductions) if len(reductions) == len(paired) else None
    )
    supports = (
        treatment_safe
        and support_pass
        and len(reductions) == len(paired) == 3
        and directions >= 2
        and no_material_seed_harm
        and median_reduction is not None
        and median_reduction >= material_reduction
    )
    return {
        "verdict": (
            "supports_downgrade_penalty" if supports else "inconclusive"
        ),
        "paired_seed_results": paired,
        "treatment_arms_pass_safety": treatment_safe,
        "tail_support_pass": support_pass,
        "pairs_favouring_treatment": directions,
        "no_seed_exceeds_harm_limit": no_material_seed_harm,
        "median_control_minus_treatment_rate": median_reduction,
        "required_material_reduction": material_reduction,
        "minimum_tail_known_actions_per_arm": minimum_tail_support,
        "maximum_allowed_seed_harm": maximum_seed_harm,
    }


def _analyze_penalty_arm(
    *,
    root: Path,
    contract: Mapping[str, Any],
    arm: Mapping[str, Any],
    readiness: Mapping[str, Any],
    paths_config: Path,
    source_commit: str,
) -> dict[str, Any]:
    summary = base._analyze_arm(
        root=root,
        contract=contract,
        arm=arm,
        readiness=readiness,
        paths_config=paths_config,
        source_commit=source_commit,
    )
    train_log = root / summary["artifacts"]["train_log"]["path"]
    rows = base._strict_jsonl(train_log)
    return {
        **summary,
        "penalty_metrics": summarize_penalty_rows(
            rows, mode=str(arm["mill_bonus_mode"])
        ),
    }


def analyze_penalty_ablation_result(
    *,
    root: Path,
    contract_path: Path,
    readiness_path: Path,
    paths_config: Path,
) -> dict[str, Any]:
    """Validate all six completed arms and produce one immutable result."""
    root = root.resolve()
    contract_path = contract_path.resolve(strict=True)
    readiness_path = readiness_path.resolve(strict=True)
    paths_config = paths_config.resolve(strict=True)
    contract = load_ablation_contract(contract_path)
    readiness = base._validate_readiness(
        readiness_path,
        contract=contract,
        contract_path=contract_path,
    )
    source_commit = str(readiness["source"]["head"])
    source = base._inspect_analysis_source(root, source_commit)
    arms = [
        _analyze_penalty_arm(
            root=root,
            contract=contract,
            arm=arm,
            readiness=readiness,
            paths_config=paths_config,
            source_commit=source_commit,
        )
        for arm in _ordered_arms(contract)
    ]
    rule = contract["analysis"]["decision_rule"]
    decision = decide_penalty_result(
        arms,
        material_reduction=float(rule["material_absolute_reduction"]),
        minimum_tail_support=int(rule["minimum_tail_known_actions_per_arm"]),
        maximum_seed_harm=float(rule["maximum_allowed_seed_harm"]),
    )
    body = {
        "schema_version": RESULT_SCHEMA,
        "claim_boundary": contract["claim_boundary"],
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "plan_identity": contract["plan_identity"],
            "file_sha256": base._sha256_file(contract_path),
        },
        "readiness": {
            "path": readiness_path.relative_to(root).as_posix(),
            "readiness_identity": readiness["readiness_identity"],
            "file_sha256": base._sha256_file(readiness_path),
        },
        "analysis_source": source,
        "data_and_runtime_versions": {
            "data_contract": contract["data_contract"],
            "rules_and_runtime": contract["rules_and_runtime"],
        },
        "hyperparameters": contract["common_training_contract"],
        "baseline": {
            "type": f"matched fresh {CONTROL_MODE} reward arm",
            "pairing": contract["pairing"],
        },
        "arms": arms,
        "decision": decision,
        "interpretation": {
            "observed_facts": [
                "Every primary numerator and denominator is reconstructed from exact per-game counts.",
                "All raw and rolling curves are observations, not forecasts.",
                "The earlier six-arm Mill-only result remains inconclusive and is not pooled into this decision.",
            ],
            "hypothesis": contract["hypothesis"],
            "supporting_evidence": decision["paired_seed_results"],
            "counterevidence": [
                "The 29-state policy-health corpus is inspected development data, not held-out validation.",
                "Training W/D/L is secondary and is not a playing-strength endpoint.",
                "Three seed pairs can detect a consistent engineering effect but cannot establish broad population generality.",
            ],
            "next_validation_experiment": (
                "A supporting result may justify one separately frozen longer "
                "successor and then a newly independent held-out evaluation. "
                "An inconclusive result ends reward-only escalation."
            ),
        },
    }
    return {**body, "result_identity": canonical_sha256(body)}


publish_result = base.publish_result


__all__ = [
    "DEFAULT_CONTRACT",
    "DEFAULT_READINESS_REPORT",
    "DEFAULT_RESULT",
    "MalomDowngradePenaltyResultError",
    "RESULT_SCHEMA",
    "analyze_penalty_ablation_result",
    "decide_penalty_result",
    "publish_result",
    "summarize_penalty_rows",
]
