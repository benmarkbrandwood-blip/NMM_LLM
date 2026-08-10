"""Focused tests for common-anchor full-action policy comparisons."""

from __future__ import annotations

import pytest

from learned_ai.evaluation.common_anchor_policy_distribution import (
    CommonAnchorPolicyDistributionError,
    classify_final_policy_divergence,
    compare_action_logits,
    summarize_state_comparisons,
)


def _record(phase: str, comparison: dict) -> dict:
    return {"phase": phase, "comparison": comparison}


def _comparison(
    refresh: list[float],
    no_refresh: list[float],
) -> dict:
    return compare_action_logits(
        refresh_logits=refresh,
        no_refresh_logits=no_refresh,
        action_keys=[f"move-{index}" for index in range(len(refresh))],
        malom_qualities=[0.0, -1.0, -2.0][: len(refresh)],
    )


def test_identical_logits_preserve_complete_distribution_and_ranks() -> None:
    comparison = _comparison([2.0, 1.0, 0.0], [2.0, 1.0, 0.0])

    assert comparison["top1_agreement"] is True
    assert comparison["ranking"]["discordant_pair_rate"] == 0.0
    assert comparison["ranking"]["mean_normalized_rank_displacement"] == 0.0
    scheduled = comparison["distributions"]["temperature_0.2"]
    assert scheduled["jensen_shannon_nats"] == pytest.approx(0.0)
    assert scheduled["total_variation"] == pytest.approx(0.0)
    assert scheduled["malom"][
        "no_refresh_minus_refresh_preserving_mass"
    ] == pytest.approx(0.0)
    assert len(comparison["actions"]) == 3


def test_shifted_logits_expose_directional_kl_malom_and_rank_changes() -> None:
    comparison = _comparison([3.0, 1.0, 0.0], [0.0, 1.0, 3.0])

    assert comparison["top1_agreement"] is False
    assert comparison["refresh_top1_action_key"] == "move-0"
    assert comparison["no_refresh_top1_action_key"] == "move-2"
    assert comparison["ranking"]["discordant_pair_rate"] > 0.0
    scheduled = comparison["distributions"]["temperature_0.2"]
    assert scheduled["kl_refresh_to_no_refresh_nats"] > 0.0
    assert scheduled["kl_no_refresh_to_refresh_nats"] > 0.0
    assert scheduled["jensen_shannon_nats"] > 0.0
    assert scheduled["malom"][
        "no_refresh_minus_refresh_preserving_mass"
    ] < 0.0


def test_summary_reports_all_three_phases_without_action_weighting() -> None:
    records = [
        _record("placement", _comparison([2.0, 1.0], [2.0, 1.0])),
        _record("movement", _comparison([2.0, 1.0], [1.0, 2.0])),
        _record("flying", _comparison([2.0, 1.0], [2.0, 1.0])),
    ]

    summary = summarize_state_comparisons(records)

    assert set(summary) == {"all", "placement", "movement", "flying"}
    assert summary["all"]["states"] == 3
    assert summary["all"]["legal_actions"] == 6
    assert summary["all"]["top1_agreement_rate"] == pytest.approx(2 / 3)
    assert summary["movement"]["top1_changed_states"] == 1


def test_final_classification_requires_both_seeds_to_cross_same_gate() -> None:
    identical_records = [
        _record(phase, _comparison([2.0, 1.0], [2.0, 1.0]))
        for phase in ("placement", "movement", "flying")
    ]
    diverged_records = [
        _record(phase, _comparison([4.0, 0.0], [0.0, 4.0]))
        for phase in ("placement", "movement", "flying")
    ]
    identical = summarize_state_comparisons(identical_records)
    diverged = summarize_state_comparisons(diverged_records)

    near = classify_final_policy_divergence(
        {"64": identical, "65": identical}
    )
    material = classify_final_policy_divergence(
        {"64": diverged, "65": diverged}
    )
    mixed = classify_final_policy_divergence(
        {"64": identical, "65": diverged}
    )

    assert near["classification"] == "near_identical"
    assert near["next_design"] == "longer_equal_transition_paired_diagnostic"
    assert material["classification"] == "materially_diverged"
    assert material["next_design"] == (
        "non_flooring_multi_start_outcome_measurement"
    )
    assert mixed["classification"] == "inconclusive"


def test_misaligned_or_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(
        CommonAnchorPolicyDistributionError,
        match="different shapes",
    ):
        compare_action_logits(
            refresh_logits=[1.0, 0.0],
            no_refresh_logits=[1.0],
            action_keys=["a", "b"],
            malom_qualities=[0.0, -1.0],
        )

    with pytest.raises(
        CommonAnchorPolicyDistributionError,
        match="non-finite",
    ):
        compare_action_logits(
            refresh_logits=[1.0, float("nan")],
            no_refresh_logits=[1.0, 0.0],
            action_keys=["a", "b"],
            malom_qualities=[0.0, -1.0],
        )
