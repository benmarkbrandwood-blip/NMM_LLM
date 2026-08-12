from __future__ import annotations

import copy

import pytest

from learned_ai.evaluation.target_refresh_mature_fork_result import (
    LEDGER_SCHEMA,
    MatureTargetRefreshResultError,
    build_direct_crossplay_schedule,
    classify_mature_policy_divergence,
    classify_mature_replication,
    decide_mature_result,
    summarize_direct_crossplay,
)


def _contract() -> dict:
    return {
        "plan_identity": "a" * 64,
        "sources": [{"seed": seed} for seed in (67, 68, 69)],
        "measurement_contract": {
            "direct_crossplay": {
                "record_indices": list(range(1, 13)),
                "replicates_per_start": 4,
                "colour_swap": True,
                "common_random_streams_by_colour": True,
                "expected_pairs": 144,
                "expected_games": 288,
                "max_post_start_logical_plies": 120,
            },
            "direct_effect_thresholds": {
                "minimum_aggregate_pair_score_effect": 1 / 12,
                "minimum_per_seed_pair_score_effect": 1 / 12,
                "minimum_supporting_seeds": 2,
                "maximum_opposite_seed_effect": 1 / 24,
                "maximum_truncation_rate": 0.25,
            },
        },
    }


def _rows(contract: dict, *, winning_seeds: set[int]) -> list[dict]:
    rows = []
    for scheduled in build_direct_crossplay_schedule(contract):
        refresh_wins = scheduled["seed"] in winning_seeds
        refresh_name = "white" if scheduled["refresh_mature_colour"] == "W" else "black"
        score = 1.0 if refresh_wins else 0.5
        rows.append(
            {
                "schema_version": LEDGER_SCHEMA,
                "plan_identity": contract["plan_identity"],
                **scheduled,
                "phase": ("placement", "movement", "flying")[
                    (scheduled["record_index"] - 1) % 3
                ],
                "refresh_mature_score": score,
                "outcome_class": "win" if refresh_wins else "draw",
                "winner": refresh_name if refresh_wins else None,
                "termination_reason": (
                    "win_fewer_than_three" if refresh_wins else "draw_threefold"
                ),
                "post_start_logical_plies": 12,
                "start_history_sha256": "b" * 64,
                "end_history_sha256": "c" * 64,
                "moves": [
                    {"from": None, "to": "a1", "capture": None} for _ in range(12)
                ],
            }
        )
    return rows


def test_schedule_is_deterministic_and_colour_swapped() -> None:
    contract = _contract()
    first = build_direct_crossplay_schedule(contract)
    second = build_direct_crossplay_schedule(copy.deepcopy(contract))

    assert first == second
    assert len(first) == 288
    assert first[0]["pair_identity"] == first[1]["pair_identity"]
    assert first[0]["refresh_mature_colour"] == "W"
    assert first[1]["refresh_mature_colour"] == "B"
    assert first[0]["policy_seed_white"] == first[1]["policy_seed_white"]
    assert first[0]["policy_seed_black"] == first[1]["policy_seed_black"]


def test_direct_effect_requires_aggregate_and_two_seed_support() -> None:
    contract = _contract()
    summary = summarize_direct_crossplay(
        contract,
        _rows(contract, winning_seeds={67, 68}),
    )

    assert summary["paired"]["mean_score_effect"] == 2 / 3
    assert summary["paired"]["seed_effects"] == {
        "67": 1.0,
        "68": 1.0,
        "69": 0.0,
    }
    assert summary["decision"]["classification"] == (
        "material_mature_refresh_direct_effect"
    )
    assert summary["decision"]["development_preference"] == "refresh-mature"


def test_combined_decision_keeps_policy_evidence_secondary() -> None:
    contract = _contract()
    outcome = summarize_direct_crossplay(
        contract,
        _rows(contract, winning_seeds={67, 68}),
    )

    without_persistence = decide_mature_result(
        policy_decision={"materially_diverged_with_persistence": False},
        direct_crossplay=outcome,
    )
    with_persistence = decide_mature_result(
        policy_decision={"materially_diverged_with_persistence": True},
        direct_crossplay=outcome,
    )

    assert without_persistence["direct_effect_supported"] is True
    assert without_persistence["mechanism_evidence_supported"] is False
    assert with_persistence["mechanism_evidence_supported"] is True
    assert with_persistence["automatic_long_run_selection"] is False


def _policy_summary(js: float) -> dict:
    metrics = {
        "mean_jensen_shannon_nats": js,
        "mean_total_variation": 0.03,
        "mean_abs_malom_preserving_probability_mass_delta": 0.01,
        "mean_no_refresh_minus_refresh_malom_preserving_mass": 0.01,
    }
    return {
        phase: {
            "top1_agreement_rate": 0.75,
            "distributions": {"temperature_0.2": copy.deepcopy(metrics)},
        }
        for phase in ("all", "placement", "movement", "flying")
    }


def test_policy_divergence_requires_two_persistent_mature_seeds() -> None:
    matrix = {
        str(seed): {
            "4096": _policy_summary(0.001),
            "8192": _policy_summary(0.001),
        }
        for seed in (67, 68, 69)
    }
    for seed in (67, 68):
        matrix[str(seed)]["4096"] = _policy_summary(0.006)
        matrix[str(seed)]["8192"] = _policy_summary(0.007)

    decision = classify_mature_policy_divergence(matrix)

    assert decision["supporting_seeds"] == ["67", "68"]
    assert decision["materially_diverged_with_persistence"] is True


def test_direct_schedule_uses_the_contract_seed_cohort() -> None:
    contract = _contract()
    contract["sources"] = [{"seed": seed} for seed in (64, 65, 66)]

    schedule = build_direct_crossplay_schedule(contract)

    assert sorted({row["seed"] for row in schedule}) == [64, 65, 66]
    assert len(schedule) == 288


def _direct_summary(
    *,
    effects: dict[str, float],
    mean: float,
    classification: str,
    truncation_rate: float = 0.05,
) -> dict:
    return {
        "games": 288,
        "pairs": 144,
        "paired": {"mean_score_effect": mean, "seed_effects": effects},
        "decision": {
            "classification": classification,
            "truncation_rate": truncation_rate,
        },
    }


def _replication_thresholds() -> dict:
    return {
        "minimum_replication_aggregate_pair_score_effect": 1 / 12,
        "minimum_pooled_pair_score_effect": 1 / 12,
        "minimum_per_seed_pair_score_effect": 1 / 12,
        "minimum_replication_supporting_seeds": 2,
        "minimum_pooled_supporting_seeds": 3,
        "maximum_pooled_opposite_seeds": 1,
        "maximum_pooled_truncation_rate": 0.25,
    }


def test_replication_gate_selects_only_a_pooled_cross_cohort_effect() -> None:
    decision = classify_mature_replication(
        prior_direct_crossplay=_direct_summary(
            effects={
                "67": -0.2291666667,
                "68": 0.0104166667,
                "69": -0.0104166667,
            },
            mean=-0.0763888889,
            classification="no_material_direct_effect",
        ),
        replication_direct_crossplay=_direct_summary(
            effects={"64": -0.15, "65": -0.14, "66": -0.13},
            mean=-0.14,
            classification="material_stale_target_direct_effect",
        ),
        thresholds=_replication_thresholds(),
    )

    assert decision["classification"] == ("replicated_material_stale_target_effect")
    assert decision["selected_successor_condition"] == "stale-control"
    assert decision["automatic_long_run_selection"] is False
    assert decision["supporting_seeds"] == ["64", "65", "66", "67"]


def test_replication_gate_preserves_no_selection_without_replication() -> None:
    decision = classify_mature_replication(
        prior_direct_crossplay=_direct_summary(
            effects={"67": -0.23, "68": 0.01, "69": -0.01},
            mean=-0.076,
            classification="no_material_direct_effect",
        ),
        replication_direct_crossplay=_direct_summary(
            effects={"64": -0.02, "65": 0.01, "66": 0.0},
            mean=-0.003,
            classification="no_material_direct_effect",
        ),
        thresholds=_replication_thresholds(),
    )

    assert decision["classification"] == "no_replicated_material_effect"
    assert decision["selected_successor_condition"] is None


def test_replication_gate_rejects_overlapping_seed_cohorts() -> None:
    with pytest.raises(
        MatureTargetRefreshResultError,
        match="replication seed cohorts overlap",
    ):
        classify_mature_replication(
            prior_direct_crossplay=_direct_summary(
                effects={"64": -0.2, "68": 0.0, "69": 0.0},
                mean=-0.0666666667,
                classification="no_material_direct_effect",
            ),
            replication_direct_crossplay=_direct_summary(
                effects={"64": -0.15, "65": -0.14, "66": -0.13},
                mean=-0.14,
                classification="material_stale_target_direct_effect",
            ),
            thresholds=_replication_thresholds(),
        )
