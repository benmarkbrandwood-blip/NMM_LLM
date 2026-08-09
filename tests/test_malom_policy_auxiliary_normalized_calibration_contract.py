"""Contract tests for the paired normalized auxiliary calibration."""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_sha256
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-malom-policy-auxiliary-normalized-calibration-v1.json"
)


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_contract_freezes_three_paired_seeds_and_one_training_factor() -> None:
    plan = _plan()
    arms = sorted(plan["arms"], key=lambda arm: arm["launch_order"])

    assert [arm["seed"] for arm in arms] == [55, 55, 56, 56, 57, 57]
    assert [arm["condition"] for arm in arms] == [
        "control",
        "normalized-0.25",
        "control",
        "normalized-0.25",
        "control",
        "normalized-0.25",
    ]
    assert [arm["malom_policy_aux_mode"] for arm in arms] == [
        "fixed",
        "policy-head-normalized",
        "fixed",
        "policy-head-normalized",
        "fixed",
        "policy-head-normalized",
    ]
    assert {arm["malom_policy_aux_coef"] for arm in arms} == {0.0}
    assert {arm["malom_policy_aux_target_ratio"] for arm in arms} == {0.25}
    assert {arm["malom_policy_aux_coef_cap"] for arm in arms} == {0.25}
    assert {arm["malom_policy_aux_denominator_floor"] for arm in arms} == {
        1e-12
    }
    assert plan["pairing"]["single_changed_training_factor"] == (
        "malom_policy_aux_mode"
    )


def test_contract_arm_outputs_and_order_are_unique() -> None:
    arms = _plan()["arms"]

    for field in (
        "arm_id",
        "control_dir",
        "experiment_id",
        "launch_order",
        "plan_id",
        "specialist_db",
    ):
        values = [arm[field] for arm in arms]
        assert len(values) == len(set(values))
    assert sorted(arm["launch_order"] for arm in arms) == list(range(1, 7))


def test_contract_schedule_counts_and_resource_ceiling_recompute() -> None:
    plan = _plan()
    common = plan["common_training_contract"]
    resources = plan["resources"]
    requested_nodes = 0

    for seed in plan["pairing"]["seeds"]:
        actual = {
            "frozen_black": 0,
            "frozen_white": 0,
            "sanmill_black": 0,
            "sanmill_white": 0,
        }
        for scheduled_index in range(resources["completed_games_per_arm"]):
            _, torch_seed = trainer._derive_game_identity(
                seed,
                scheduled_index,
                "primary",
            )
            config_rng = random.Random(torch_seed)
            learner_colour = "white" if config_rng.random() < 0.5 else "black"
            opponent = (
                "frozen"
                if config_rng.random() < common["frozen_target_ratio"]
                else "sanmill"
            )
            actual[f"{opponent}_{learner_colour}"] += 1

        assert actual == resources["schedule_counts_by_seed"][str(seed)]
        sanmill_games = actual["sanmill_black"] + actual["sanmill_white"]
        requested_nodes += (
            sanmill_games
            * 2
            * (common["max_logical_plies"] // 2)
            * common["sanmill_node_ladder"][0]
        )

    assert resources["maximum_completed_games_total"] == 600
    assert resources["maximum_active_wall_hours_total"] == 2.0
    assert resources["active_wall_hours_per_arm"] * 6 == pytest.approx(2.0)
    assert resources["maximum_requested_sanmill_nodes_total"] == requested_nodes


def test_contract_is_fresh_unlaunched_and_not_strength_evidence() -> None:
    plan = _plan()

    assert plan["status"] == "designed_unlaunched_needs_publication"
    assert plan["common_training_contract"]["start_mode"] == "fresh"
    assert plan["lineage"]["resume_source"] is None
    assert plan["lineage"]["fresh_random_weights_per_arm"] is True
    assert plan["authorization"] == {
        "authorized_segments_per_arm": 0,
        "launch_authorized": False,
        "promotion_allowed": False,
        "publication_allowed": False,
    }
    assert "not held-out validation" in plan["claim_boundary"]


def test_contract_binds_the_no_update_result_and_interpretation() -> None:
    evidence = _plan()["preparation_evidence"]["no_update_batch_capture"]

    assert evidence["result_identity"] == (
        "b0dfd3415c55196c59e71cf67e45b00ab5844e9f62fbc9f3bdc31b09a694bd86"
    )
    assert evidence["result_sha256"] == (
        "2e310ecccf869f16b314093c6f50395e91019839b603d805ad3e0cab9d651fee"
    )
    assert evidence["interpretation_commit"] == (
        "bcceab547cc7de9177a24964bd021816d656bd7c"
    )
    assert evidence["result_summary"]["fresh_seeds"] == [52, 53, 54]
    assert evidence["result_summary"]["target_025_effective_coefficient_max"] < (
        0.25
    )


def test_contract_identity_and_temperature_schedule_are_canonical() -> None:
    plan = _plan()
    identity = plan.pop("plan_identity")
    common = plan["common_training_contract"]

    assert identity == canonical_sha256(plan)
    assert trainer._compute_temperature(
        0,
        common["max_games_schedule"],
        common["temperature_start"],
    ) == common["temperature_start"]
    assert trainer._compute_temperature(
        4000,
        common["max_games_schedule"],
        common["temperature_start"],
    ) == pytest.approx(common["temperature_end"])


def test_contract_pins_the_pre_result_analyzer_and_publisher() -> None:
    implementation = _plan()["analysis"]["result_implementation"]

    assert implementation["result_schema"] == (
        "nmm.sanmill-malom-policy-auxiliary-normalized-calibration-result.v1"
    )
    for name in ("module", "publisher"):
        record = implementation[name]
        path = ROOT / record["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
