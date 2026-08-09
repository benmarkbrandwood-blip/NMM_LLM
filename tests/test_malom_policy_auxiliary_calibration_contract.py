"""Contract checks for the bounded Malom policy-auxiliary calibration."""

from __future__ import annotations

import json
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
    / "sanmill-malom-policy-auxiliary-calibration-smoke-v1.json"
)


def _load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_calibration_freezes_four_same_seed_coefficient_arms() -> None:
    plan = _load_plan()
    arms = sorted(plan["arms"], key=lambda arm: arm["launch_order"])

    assert [arm["arm_id"] for arm in arms] == [
        "control-c000",
        "low-c003",
        "medium-c010",
        "high-c030",
    ]
    assert [arm["malom_policy_aux_coef"] for arm in arms] == [
        0.0,
        0.03,
        0.1,
        0.3,
    ]
    assert {arm["seed"] for arm in arms} == {51}
    assert {arm["mill_bonus_mode"] for arm in arms} == {
        "malom-preserving-only"
    }
    assert plan["pairing"]["single_changed_factor"] == (
        "malom_policy_aux_coef"
    )
    assert plan["pairing"]["single_process_at_a_time"] is True


def test_calibration_arm_outputs_and_order_are_unique() -> None:
    arms = _load_plan()["arms"]

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
    assert sorted(arm["launch_order"] for arm in arms) == [1, 2, 3, 4]


def test_calibration_schedule_and_resource_envelope_are_exact() -> None:
    plan = _load_plan()
    common = plan["common_training_contract"]
    resources = plan["resources"]
    expected = resources["schedule_counts_per_arm"]
    actual = {
        "frozen_black": 0,
        "frozen_white": 0,
        "sanmill_black": 0,
        "sanmill_white": 0,
    }
    for scheduled_index in range(resources["completed_games_per_arm"]):
        _, torch_seed = trainer._derive_game_identity(
            plan["pairing"]["seed"], scheduled_index, "primary"
        )
        config_rng = random.Random(torch_seed)
        learner_colour = "white" if config_rng.random() < 0.5 else "black"
        opponent = (
            "frozen"
            if config_rng.random() < common["frozen_target_ratio"]
            else "sanmill"
        )
        actual[f"{opponent}_{learner_colour}"] += 1

    assert actual == expected
    assert sum(expected.values()) == 100
    sanmill_games = (
        expected["sanmill_black"] + expected["sanmill_white"]
    ) * len(plan["arms"])
    max_searches_per_game = common["max_logical_plies"] // 2
    assert resources["maximum_requested_sanmill_nodes_total"] == (
        sanmill_games
        * max_searches_per_game
        * common["sanmill_node_ladder"][0]
    )
    assert resources["maximum_completed_games_total"] == 400
    assert resources["maximum_active_wall_hours_total"] == 2.0
    assert common["one_segment_games"] == 100
    assert common["only_observed_node_level"] == 1


def test_calibration_keeps_the_training_lineage_fresh_and_bounded() -> None:
    plan = _load_plan()
    common = plan["common_training_contract"]

    assert common["algorithm"] == "A2C"
    assert common["start_mode"] == "fresh"
    assert common["max_games_schedule"] == 5000
    assert common["minimal_rollouts"] is True
    assert common["recovery"] is False
    assert plan["lineage"]["resume_source"] is None
    assert plan["lineage"]["fresh_random_weights_per_arm"] is True
    assert not plan["authorization"]["launch_authorized"]
    assert plan["authorization"]["authorized_segments_per_arm"] == 0
    assert not plan["authorization"]["publication_allowed"]
    assert not plan["authorization"]["promotion_allowed"]


def test_calibration_binds_the_exact_gradient_evidence() -> None:
    evidence = _load_plan()["preparation_evidence"]["tracked_manifest"]

    assert evidence == {
        "path": (
            "docs/evidence/"
            "sanmill-malom-policy-auxiliary-gradient-probe-2026-08-09.json"
        ),
        "probe_identity": (
            "5ea60e2955a7a9b878ec4119648ed91ddcffd94687bb3c0976571e96048daa9c"
        ),
        "probe_path": (
            "out/diagnostics/malom-policy-auxiliary-gradient-probe-v2.json"
        ),
        "probe_schema_version": (
            "nmm.malom-policy-auxiliary-gradient-probe.v2"
        ),
        "probe_sha256": (
            "ad1e6e3ee7596a872d3129e623d377e083c439f6bcbee23705a10bf8ced1b003"
        ),
        "probe_size_bytes": 34974,
        "probe_source_commit": (
            "b2ccecf2a7518adc99d1e1b8c38887ab3938b8ec"
        ),
        "schema_version": (
            "nmm.malom-policy-auxiliary-gradient-evidence.v1"
        ),
        "sha256": (
            "1d7784dfabf8aa59d70adc310d0279b03a08863e69e2a5a009339d9f13394092"
        ),
    }


def test_calibration_plan_identity_and_temperature_are_canonical() -> None:
    plan = _load_plan()
    common = plan["common_training_contract"]
    identity = plan.pop("plan_identity")

    assert identity == canonical_sha256(plan)
    assert trainer.TEMP_END == common["temperature_end"]
    assert trainer._compute_temperature(
        0, common["max_games_schedule"], common["temperature_start"]
    ) == common["temperature_start"]
    assert trainer._compute_temperature(
        4000, common["max_games_schedule"], common["temperature_start"]
    ) == pytest.approx(common["temperature_end"])
