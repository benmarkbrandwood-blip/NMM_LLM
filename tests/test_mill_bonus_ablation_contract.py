"""Contract checks for the bounded multi-seed mill-bonus ablation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-mill-bonus-ablation-smoke-v1.json"
)


def _load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_ablation_has_three_complete_seed_pairs() -> None:
    plan = _load_plan()
    arms = plan["arms"]

    assert len(arms) == 6
    assert plan["pairing"]["paired_seeds"] == [42, 43, 44]
    for seed in (42, 43, 44):
        pair = [arm for arm in arms if arm["seed"] == seed]
        assert {arm["mill_bonus_mode"] for arm in pair} == {
            "legacy-unconditional",
            "malom-preserving-only",
        }


def test_ablation_arm_isolation_and_launch_order_are_unique() -> None:
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
    assert sorted(arm["launch_order"] for arm in arms) == list(range(1, 7))


def test_ablation_freezes_one_factor_and_bounded_resources() -> None:
    plan = _load_plan()
    common = plan["common_training_contract"]

    assert plan["pairing"]["arm_difference_allowlist"] == [
        "experiment_id",
        "plan_id",
        "control_dir",
        "specialist_db",
        "mill_bonus_mode",
    ]
    assert common["algorithm"] == "A2C"
    assert common["start_mode"] == "fresh"
    assert common["max_games_schedule"] == 5000
    assert common["one_segment_games"] == 500
    assert common["only_observed_node_level"] == 1
    assert plan["resources"]["maximum_completed_games_total"] == 3000
    assert plan["resources"]["maximum_active_wall_hours_total"] == 6.0
    per_arm = plan["resources"]["schedule_counts_per_arm"]
    assert all(sum(counts.values()) == 500 for counts in per_arm.values())
    sanmill_games = sum(
        counts["sanmill_black"] + counts["sanmill_white"]
        for counts in per_arm.values()
    )
    assert sanmill_games * 2 == 1220
    assert plan["resources"]["maximum_requested_sanmill_nodes_total"] == (
        sanmill_games * 2 * 60 * 1000
    )
    assert not plan["authorization"]["launch_authorized"]
    assert plan["authorization"]["authorized_segments_per_arm"] == 0


def test_ablation_schedule_counts_match_the_trainer_seed_contract() -> None:
    expected = _load_plan()["resources"]["schedule_counts_per_arm"]

    for seed_text, frozen_counts in expected.items():
        actual = {
            "frozen_black": 0,
            "frozen_white": 0,
            "sanmill_black": 0,
            "sanmill_white": 0,
        }
        for scheduled_index in range(500):
            _, torch_seed = trainer._derive_game_identity(
                int(seed_text), scheduled_index, "primary"
            )
            config_rng = random.Random(torch_seed)
            learner_colour = "white" if config_rng.random() < 0.5 else "black"
            opponent_source = (
                "frozen" if config_rng.random() < 0.60 else "sanmill"
            )
            actual[f"{opponent_source}_{learner_colour}"] += 1

        assert actual == frozen_counts


def test_ablation_uses_one_closed_empty_specialist_template() -> None:
    template = _load_plan()["data_contract"]["specialist_db_initial_template"]

    assert template["label_version"] == "sector-corrected-v1"
    assert template["quick_check"] == "ok"
    assert template["positions"] == 0
    assert template["winning_lines"] == 0
    assert template["preferred_plays"] == 0
    assert template["sidecars"] == "absent"
    assert len(template["sha256"]) == 64


def test_ablation_plan_identity_is_canonical() -> None:
    plan = _load_plan()
    identity = plan.pop("plan_identity")

    assert identity == canonical_sha256(plan)
