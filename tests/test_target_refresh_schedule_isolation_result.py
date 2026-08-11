"""Tests for the schedule-isolated target-refresh outcome contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.target_refresh_schedule_isolation_result import (
    ScheduleIsolationResultError,
    build_outcome_measurement_schedule,
    decide_schedule_isolation_result,
    validate_and_summarize_outcome_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = (
    ROOT / "docs/experiments/dev-v4-phase-replay-development-corpus-v1.json"
)
SEEDS = (67, 68, 69)


def _corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _completed_rows(*, no_refresh_wins: bool = True) -> list[dict]:
    rows = build_outcome_measurement_schedule(seeds=SEEDS, corpus=_corpus())
    completed = []
    for row in rows:
        favored = (
            row["condition"] == "no-refresh"
            if no_refresh_wins
            else row["condition"] == "refresh-once"
        )
        outcome_class = "win" if favored else "draw"
        completed.append(
            {
                **row,
                "candidate_checkpoint_id": f"candidate:{row['condition']}",
                "anchor_checkpoint_id": f"anchor:{row['seed']}",
                "start_history_sha256": "a" * 64,
                "end_history_sha256": "b" * 64,
                "start_logical_ply_count": 12,
                "end_logical_ply_count": 32,
                "training_reward_outcome": 1.5 if favored else -0.15,
                "outcome_class": outcome_class,
                "score": 1.0 if favored else 0.5,
                "post_start_logical_plies": 20,
                "termination_reason": "loseFewerThanThree",
            }
        )
    return completed


def _policy_decision(*, classification: str = "materially_diverged") -> dict:
    return {
        "classification": classification,
        "by_seed": {
            str(seed): {
                "by_transition_boundary": {
                    "8192": {
                        "observed": {
                            "phase_signed_malom_preserving_mass_delta": {
                                phase: 0.06
                                for phase in ("placement", "movement", "flying")
                            }
                        }
                    }
                }
            }
            for seed in SEEDS
        },
    }


def test_schedule_is_exactly_paired_with_common_random_numbers() -> None:
    rows = build_outcome_measurement_schedule(seeds=SEEDS, corpus=_corpus())

    assert len(rows) == 288
    first_pair = rows[:2]
    assert [row["condition"] for row in first_pair] == [
        "refresh-once",
        "no-refresh",
    ]
    assert first_pair[0]["paired_game_identity"] == first_pair[1][
        "paired_game_identity"
    ]
    assert first_pair[0]["torch_seed"] == first_pair[1]["torch_seed"]
    assert first_pair[0]["game_id"] != first_pair[1]["game_id"]


def test_complete_grid_summarizes_phase_color_and_pairs() -> None:
    summary = validate_and_summarize_outcome_rows(
        _completed_rows(), seeds=SEEDS, corpus=_corpus()
    )

    assert summary["games"] == 288
    assert summary["paired_games"] == 144
    final = summary["by_seed_boundary"]["67"]["8192"]
    assert final["by_condition"]["no-refresh"]["score_rate"] == 1.0
    assert final["by_condition"]["refresh-once"]["score_rate"] == 0.5
    assert final["paired_no_refresh_minus_refresh_mean_score"] == 0.5
    assert set(final["by_phase_and_condition"]) == {
        "placement",
        "movement",
        "flying",
    }


def test_schedule_tampering_fails_closed() -> None:
    rows = _completed_rows()
    rows[1]["torch_seed"] += 1

    with pytest.raises(
        ScheduleIsolationResultError,
        match="frozen schedule",
    ):
        validate_and_summarize_outcome_rows(rows, seeds=SEEDS, corpus=_corpus())


def test_material_persistent_safe_result_selects_condition() -> None:
    summary = validate_and_summarize_outcome_rows(
        _completed_rows(), seeds=SEEDS, corpus=_corpus()
    )

    decision = decide_schedule_isolation_result(
        policy_decision=_policy_decision(),
        outcome_summary=summary,
        seeds=SEEDS,
    )

    assert decision["supported"] is True
    assert decision["selected_long_run_condition"] == "no-refresh"
    assert decision["gates"] == {
        "persistent_policy_divergence": True,
        "persistent_outcome_direction": True,
        "phase_safety": True,
        "truncation_safety": True,
        "malom_safety": True,
    }


def test_outcome_signal_without_policy_gate_does_not_select() -> None:
    summary = validate_and_summarize_outcome_rows(
        _completed_rows(), seeds=SEEDS, corpus=_corpus()
    )

    decision = decide_schedule_isolation_result(
        policy_decision=_policy_decision(classification="near_identical"),
        outcome_summary=summary,
        seeds=SEEDS,
    )

    assert decision["supported"] is False
    assert decision["classification"] == (
        "outcome_effect_without_persistent_policy_gate"
    )
    assert decision["selected_long_run_condition"] is None


def test_phase_harm_blocks_selection() -> None:
    rows = _completed_rows()
    for row in rows:
        if (
            row["phase"] == "flying"
            and row["condition"] == "no-refresh"
            and row["post_fork_consumed_transitions"] == 8192
        ):
            row["outcome_class"] = "loss"
            row["score"] = 0.0
            row["training_reward_outcome"] = -1.0
    summary = validate_and_summarize_outcome_rows(
        rows, seeds=SEEDS, corpus=_corpus()
    )

    decision = decide_schedule_isolation_result(
        policy_decision=_policy_decision(),
        outcome_summary=summary,
        seeds=SEEDS,
    )

    assert decision["supported"] is False
    assert decision["gates"]["phase_safety"] is False


def test_opposite_malom_direction_blocks_selection() -> None:
    summary = validate_and_summarize_outcome_rows(
        _completed_rows(), seeds=SEEDS, corpus=_corpus()
    )
    policy = copy.deepcopy(_policy_decision())
    for seed in SEEDS:
        policy["by_seed"][str(seed)]["by_transition_boundary"]["8192"][
            "observed"
        ]["phase_signed_malom_preserving_mass_delta"]["movement"] = -0.06

    decision = decide_schedule_isolation_result(
        policy_decision=policy,
        outcome_summary=summary,
        seeds=SEEDS,
    )

    assert decision["supported"] is False
    assert decision["gates"]["malom_safety"] is False
