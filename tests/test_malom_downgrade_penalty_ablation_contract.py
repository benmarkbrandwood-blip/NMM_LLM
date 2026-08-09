"""Contract checks for the exact-Malom downgrade-penalty ablation."""

from __future__ import annotations

import json
import random
from pathlib import Path

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.validation.malom_downgrade_penalty_probe import (
    CONTROL_MODE,
    MALOM_DOWNGRADE_PENALTY_PROBE_SCHEMA,
    TREATMENT_MODE,
)
from learned_ai.validation.mill_bonus_ablation_readiness import (
    load_ablation_contract,
)
from scripts import train_s_gen_v2 as trainer


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "docs"
    / "experiments"
    / "sanmill-malom-downgrade-penalty-ablation-smoke-v1.json"
)


def _load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_penalty_contract_has_three_complete_fresh_pairs() -> None:
    plan = load_ablation_contract(PLAN)

    assert plan["pairing"]["paired_seeds"] == [45, 46, 47]
    assert len(plan["arms"]) == 6
    for seed in (45, 46, 47):
        pair = [arm for arm in plan["arms"] if arm["seed"] == seed]
        assert {arm["mill_bonus_mode"] for arm in pair} == {
            CONTROL_MODE,
            TREATMENT_MODE,
        }
    assert plan["common_training_contract"]["start_mode"] == "fresh"
    assert plan["lineage"]["resume_source"] is None


def test_penalty_contract_freezes_the_dense_primary_metric() -> None:
    rule = _load_plan()["analysis"]["decision_rule"]

    assert rule["material_absolute_reduction"] == 0.02
    assert rule["maximum_allowed_seed_harm"] == 0.02
    assert rule["minimum_tail_known_actions_per_arm"] == 2000
    assert "all tail learner actions" in rule["primary_metric"]


def test_penalty_contract_requires_the_exact_no_update_probe() -> None:
    spec = _load_plan()["preparation_evidence"][
        "downgrade_penalty_no_update_probe"
    ]

    assert spec["schema_version"] == MALOM_DOWNGRADE_PENALTY_PROBE_SCHEMA
    assert spec["expected_summary"] == {
        "affected_states": 19,
        "control_reward_total": 0.0,
        "mill_forming_states": 16,
        "non_mill_states": 3,
        "phase_counts": {"fly": 9, "move": 4, "place": 6},
        "quality_rank_counts": {"1": 19},
        "states": 19,
        "stratum_counts": {"book": 6, "human_db": 2, "perfect_db": 11},
        "treatment_minus_control": -4.75,
        "treatment_reward_total": -4.75,
    }


def test_penalty_schedule_counts_match_trainer_derivation() -> None:
    plan = _load_plan()
    expected = plan["resources"]["schedule_counts_per_arm"]

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
            opponent = "frozen" if config_rng.random() < 0.60 else "sanmill"
            actual[f"{opponent}_{learner_colour}"] += 1
        assert actual == frozen_counts

    sanmill_games = sum(
        counts["sanmill_black"] + counts["sanmill_white"]
        for counts in expected.values()
    ) * 2
    assert sanmill_games == 1230
    assert plan["resources"]["maximum_requested_sanmill_nodes_total"] == (
        sanmill_games * 60 * 1000
    )


def test_penalty_contract_is_unlaunched_and_canonical() -> None:
    plan = _load_plan()
    identity = plan.pop("plan_identity")

    assert identity == canonical_sha256(plan)
    assert not plan["authorization"]["launch_authorized"]
    assert plan["authorization"]["authorized_segments_per_arm"] == 0
    assert not plan["authorization"]["publication_allowed"]
    assert not plan["authorization"]["promotion_allowed"]
