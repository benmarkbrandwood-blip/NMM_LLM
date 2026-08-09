"""Focused tests for downgrade-penalty result metrics and decisions."""

from __future__ import annotations

import copy

import pytest

from learned_ai.evaluation.malom_downgrade_penalty_result import (
    MalomDowngradePenaltyResultError,
    decide_penalty_result,
    summarize_penalty_rows,
)
from learned_ai.validation.malom_downgrade_penalty_probe import (
    CONTROL_MODE,
    TREATMENT_MODE,
)
from scripts import train_s_gen_v2 as trainer


def _row(game: int, *, mode: str, downgrade: int = 1) -> dict:
    rank_total = downgrade
    reward_total = (
        -trainer.MALOM_DOWNGRADE_PENALTY * rank_total
        if mode == TREATMENT_MODE
        else 0.0
    )
    return {
        "game": game,
        "steps": 10,
        "game_type": "vs_frozen" if game % 2 else "vs_sanmill",
        "learner_color": "W" if game % 2 else "B",
        "termination_reason": "material",
        "malom_downgrade_move_rate": downgrade / 10,
        "reward_malom_mean": reward_total / 10,
        "malom_downgrade_count": downgrade,
        "malom_downgrade_rank_total": rank_total,
        "malom_reward_total": reward_total,
        "malom_known_place": 4,
        "malom_known_move": 6,
        "malom_known_fly": 0,
        "malom_downgrade_place": downgrade,
        "malom_downgrade_move": 0,
        "malom_downgrade_fly": 0,
    }


def test_penalty_metrics_reconcile_reward_and_phase_counts() -> None:
    rows = [_row(game, mode=TREATMENT_MODE) for game in range(1, 501)]

    metrics = summarize_penalty_rows(rows, mode=TREATMENT_MODE)

    assert metrics["whole_run"]["known_actions"] == 5000
    assert metrics["whole_run"]["downgrade_actions"] == 500
    assert metrics["whole_run"]["rate"] == pytest.approx(0.1)
    assert metrics["whole_run"]["malom_reward_total"] == pytest.approx(-125.0)
    assert metrics["tail_301_500"]["known_actions"] == 2000
    assert metrics["tail_by_opponent_source"]["vs_sanmill"][
        "downgrade_actions"
    ] == 100


def test_penalty_metrics_reject_reward_drift() -> None:
    rows = [_row(game, mode=TREATMENT_MODE) for game in range(1, 501)]
    rows[0] = {**rows[0], "malom_reward_total": 0.0}

    with pytest.raises(
        MalomDowngradePenaltyResultError,
        match="reward total differs",
    ):
        summarize_penalty_rows(rows, mode=TREATMENT_MODE)


def _arm(seed: int, mode: str, rate: float, *, safe: bool = True) -> dict:
    known = 3000
    downgrade = round(rate * known)
    return {
        "seed": seed,
        "mill_bonus_mode": mode,
        "policy_health": {"passed": safe},
        "penalty_metrics": {
            "tail_301_500": {
                "known_actions": known,
                "downgrade_actions": downgrade,
                "downgrade_rank_total": downgrade,
                "rate": downgrade / known,
                "malom_reward_total": 0.0,
                "by_phase": {},
            }
        },
    }


def test_decision_requires_material_consistent_safe_reduction() -> None:
    arms = []
    for seed, treatment_rate in ((45, 0.04), (46, 0.05), (47, 0.055)):
        arms.extend(
            (
                _arm(seed, CONTROL_MODE, 0.08),
                _arm(seed, TREATMENT_MODE, treatment_rate),
            )
        )

    decision = decide_penalty_result(
        arms,
        material_reduction=0.02,
        minimum_tail_support=2000,
        maximum_seed_harm=0.02,
    )

    assert decision["verdict"] == "supports_downgrade_penalty"
    assert decision["pairs_favouring_treatment"] == 3
    assert decision["treatment_arms_pass_safety"] is True
    assert decision["tail_support_pass"] is True


@pytest.mark.parametrize("failure", ["unsafe", "support", "harm", "median"])
def test_decision_fails_closed_on_each_gate(failure: str) -> None:
    arms = []
    for seed in (45, 46, 47):
        arms.extend(
            (
                _arm(seed, CONTROL_MODE, 0.08),
                _arm(seed, TREATMENT_MODE, 0.05),
            )
        )
    if failure == "unsafe":
        arms[1]["policy_health"]["passed"] = False
    elif failure == "support":
        arms[1]["penalty_metrics"]["tail_301_500"]["known_actions"] = 1999
    elif failure == "harm":
        arms[1] = _arm(45, TREATMENT_MODE, 0.11)
    else:
        for arm in arms:
            if arm["mill_bonus_mode"] == TREATMENT_MODE:
                replacement = _arm(int(arm["seed"]), TREATMENT_MODE, 0.065)
                arm.clear()
                arm.update(copy.deepcopy(replacement))

    decision = decide_penalty_result(
        arms,
        material_reduction=0.02,
        minimum_tail_support=2000,
        maximum_seed_harm=0.02,
    )

    assert decision["verdict"] == "inconclusive"
