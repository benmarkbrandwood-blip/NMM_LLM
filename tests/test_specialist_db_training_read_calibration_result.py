"""Focused tests for SpecialistDB read-calibration result analysis."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from learned_ai.evaluation import (
    specialist_db_training_read_calibration_result as result_module,
)
from learned_ai.evaluation.specialist_db_training_read_calibration_result import (
    SpecialistReadCalibrationResultError,
    decide_specialist_read_calibration_result,
    publish_result,
    summarize_specialist_read_game_rows,
)


def _read_row(
    game: int,
    *,
    mode: str,
    empirical: int = 2,
    suppressed: int = 0,
) -> dict:
    return {
        "game": game,
        "specialist_read_mode": mode,
        "specialist_read_queries": 5,
        "specialist_read_rows_present": 4,
        "specialist_read_theoretical_available": 1,
        "specialist_read_empirical_available": empirical,
        "specialist_read_projections_returned": 3,
        "specialist_read_empirical_suppressed": suppressed,
    }


def _summarize(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict],
    *,
    mode: str,
) -> dict:
    monkeypatch.setattr(
        result_module,
        "summarize_game_rows",
        lambda *args, **kwargs: {"base_metrics": True},
    )
    return summarize_specialist_read_game_rows(
        rows,
        mode=mode,
        expected_games=len(rows),
        expected_schedule_counts={},
    )


def test_full_read_telemetry_is_preserved_and_engaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _summarize(
        monkeypatch,
        [_read_row(1, mode="full"), _read_row(2, mode="full")],
        mode="full",
    )

    intervention = report["specialist_read_intervention"]
    assert intervention["engaged"] is True
    assert intervention["totals"]["queries"] == 10
    assert intervention["totals"]["empirical_available"] == 4
    assert intervention["totals"]["empirical_suppressed"] == 0
    assert intervention["curves"]["rolling_50_complete_windows_only"] == []


def test_theoretical_only_requires_every_empirical_read_to_be_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _summarize(
        monkeypatch,
        [
            _read_row(
                1,
                mode="theoretical-only",
                empirical=3,
                suppressed=3,
            )
        ],
        mode="theoretical-only",
    )
    assert report["specialist_read_intervention"]["engaged"] is True

    with pytest.raises(
        SpecialistReadCalibrationResultError,
        match="did not suppress every empirical read",
    ):
        _summarize(
            monkeypatch,
            [
                _read_row(
                    1,
                    mode="theoretical-only",
                    empirical=3,
                    suppressed=2,
                )
            ],
            mode="theoretical-only",
        )


def test_read_counter_relations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _read_row(1, mode="full")
    row["specialist_read_rows_present"] = 6

    with pytest.raises(
        SpecialistReadCalibrationResultError,
        match="counters contradict",
    ):
        _summarize(monkeypatch, [row], mode="full")


def _arm(seed: int, condition: str, *, engaged: bool = True) -> dict:
    return {
        "arm_id": f"seed{seed}-{condition}",
        "seed": seed,
        "condition": condition,
        "metrics": {
            "specialist_read_intervention": {"engaged": engaged},
        },
        "policy_health": {"passed": True},
    }


def _endpoint(seed: int, *, argmax: int, tv: float) -> dict:
    return {
        "seed": seed,
        "route": {
            "scratch_initialization_shared_exactly": True,
            "specialist_db_projection": "disabled",
        },
        "aggregate": {
            "all": {
                "argmax_changes": argmax,
                "mean_policy_total_variation": tv,
            }
        },
    }


def _decision_rule() -> dict:
    return {
        "minimum_argmax_changes_for_detectable_pair": 3,
        "minimum_mean_total_variation_for_detectable_pair": 0.01,
        "minimum_reproducible_seed_pairs": 2,
        "training_wdl_is_not_a_selection_metric": True,
    }


def _arms() -> list[dict]:
    return [
        _arm(seed, condition)
        for seed in (61, 62, 63)
        for condition in ("full", "theoretical-only")
    ]


def test_paired_decision_requires_two_detectable_seed_pairs() -> None:
    decision = decide_specialist_read_calibration_result(
        _arms(),
        [
            _endpoint(61, argmax=3, tv=0.002),
            _endpoint(62, argmax=0, tv=0.011),
            _endpoint(63, argmax=1, tv=0.005),
        ],
        decision_rule=_decision_rule(),
    )

    assert decision["eligible"] is True
    assert decision["detectable_seed_pairs"] == 2
    assert decision["selected_read_mode"] is None
    assert decision["training_wdl_used_for_selection"] is False
    assert decision["verdict"] == (
        "reproducible_read_effect_eligible_for_heldout_design"
    )


def test_unengaged_intervention_blocks_an_otherwise_detectable_result() -> None:
    arms = _arms()
    arms[3]["metrics"]["specialist_read_intervention"]["engaged"] = False
    decision = decide_specialist_read_calibration_result(
        arms,
        [
            _endpoint(61, argmax=3, tv=0.02),
            _endpoint(62, argmax=3, tv=0.02),
            _endpoint(63, argmax=3, tv=0.02),
        ],
        decision_rule=_decision_rule(),
    )

    assert decision["eligible"] is False
    assert decision["all_identity_and_safety_gates_passed"] is False
    assert decision["verdict"] == "inconclusive_no_retained_mode_selection"


def test_result_publisher_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    publish_result(path, {"result_identity": "test"})

    with pytest.raises(
        SpecialistReadCalibrationResultError,
        match="already exists",
    ):
        publish_result(path, {"result_identity": "test"})


def test_scratch_policy_reconstruction_is_exact_for_a_seed() -> None:
    config = {
        "move_feat_dim": 134,
        "value_input_dim": 80,
        "policy_hidden": (256, 128),
        "value_hidden": (256, 128, 64),
        "dropout": 0.0,
    }

    first = result_module._scratch_policy(config, 61)
    second = result_module._scratch_policy(config, 61)

    assert first.get_config() == config
    assert second.get_config() == config
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
