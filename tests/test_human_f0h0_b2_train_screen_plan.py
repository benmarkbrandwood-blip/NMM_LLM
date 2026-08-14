from pathlib import Path

from learned_ai.evaluation.human_f0h0_b2_train_screen import (
    EXPECTED_SAMPLE_COMPOSITION,
    load_screen_plan,
    verify_implementation_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/experiments/f0-h0-b2-train-rejection-screen-v1.json"
PLAN_V2 = ROOT / "docs/experiments/f0-h0-b2-train-rejection-screen-v2.json"


def test_historical_frozen_train_screen_plan_remains_sealed() -> None:
    plan, file_sha = load_screen_plan(PLAN)

    assert plan["plan_identity"] == (
        "dd87175dc950cbcde4b0b44cd5d4a8da0b039dcbd3cacaf198ba43ec00de0bdc"
    )
    assert len(file_sha) == 64
    assert plan["repository_base_commit"] == (
        "32b1386b0f87ddbde0a052c2d95ad59b958838d3"
    )
    assert plan["implementation_artifacts"][0]["sha256"] == (
        "588e6bdf73bc00cff60966926f496f03c4ec11719090878b068bac6d9e1ef44c"
    )


def test_frozen_plan_uses_only_the_preregistered_train_intersection() -> None:
    plan, _file_sha = load_screen_plan(PLAN)

    assert plan["sample"]["membership_composition"] == EXPECTED_SAMPLE_COMPOSITION
    assert plan["sample"]["analysis_games"] == 9_113
    assert plan["sample"]["resampling_allowed"] is False
    assert plan["statistics_partitions"] == ["train"]
    assert set(plan["protected_partitions"]) == {
        "selection",
        "confirmation",
        "final-test",
    }


def test_frozen_plan_keeps_rejection_and_positional_boundaries() -> None:
    plan, _file_sha = load_screen_plan(PLAN)

    assert plan["safe_set"] == "A_pos"
    assert plan["a_allow_claim"] is False
    assert plan["four_b_execution_rule"] == "run_only_if_four_a_passes"
    assert plan["thresholds"]["estimability"]["k"] == 20
    assert plan["thresholds"]["estimability"]["m"] == 5
    assert (
        plan["thresholds"]["product_effect"]["minimum_signable_absolute_effect"] == 0.01
    )


def test_technical_replay_changes_code_only_not_thresholds_or_sample() -> None:
    original, _original_sha = load_screen_plan(PLAN)
    replay, _replay_sha = load_screen_plan(PLAN_V2)

    assert replay["plan_identity"] == (
        "a6972c3dae62ec249ccf6ea7bc7bf46132288a15db41b1c33b347b75615a9d0c"
    )
    assert replay["technical_correction"]["thresholds_changed"] is False
    assert replay["technical_correction"]["sample_changed"] is False
    assert replay["thresholds"] == original["thresholds"]
    assert (
        replay["sample"]["train_session_ids_identity"]
        == original["sample"]["train_session_ids_identity"]
    )
    verify_implementation_artifacts(ROOT, replay)
