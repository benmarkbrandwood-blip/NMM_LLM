from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.training.run_contract import canonical_json_bytes, canonical_sha256
from scripts.run_retained_heldout_score import load_plan
from tools import freeze_retained_heldout_score_plan as freezer


def test_committed_plan_is_reproducible_canonical_and_runtime_valid() -> None:
    expected = freezer.build_plan()
    raw = freezer.OUTPUT.read_bytes()
    persisted = json.loads(raw)

    assert canonical_json_bytes(persisted) == raw
    assert persisted == expected
    assert load_plan(freezer.OUTPUT) == expected
    body = {key: value for key, value in persisted.items() if key != "plan_identity"}
    assert persisted["plan_identity"] == canonical_sha256(body)


def test_plan_binds_selected_prefix_primary_and_resource_ceiling() -> None:
    plan = freezer.build_plan()

    assert plan["implementation"]["commit"] == freezer.IMPLEMENTATION_COMMIT
    assert plan["corpus"]["pool_identity"].startswith("2eb04f54")
    assert plan["corpus"]["prefix_records_identity"].startswith("99951a69")
    assert plan["corpus"]["records"] == 253
    assert plan["corpus"]["phase_counts"] == {
        "flying": 56,
        "movement": 98,
        "placement": 99,
    }
    assert (
        plan["analysis"]["engineering_interval"]["maximum_primary_half_width"] == 0.015
    )
    assert plan["analysis"]["primary_decision_rule"] == {
        "v4_higher_fixed_heldout_score": (
            "lower_bound > 0 and half_width <= 0.015 and all games reach "
            "strict rules terminals"
        ),
        "v3_higher_fixed_heldout_score": (
            "upper_bound < 0 and half_width <= 0.015 and all games reach "
            "strict rules terminals"
        ),
        "inconclusive": (
            "interval includes zero and half_width <= 0.015 and all games "
            "reach strict rules terminals"
        ),
        "inconclusive_precision": "half_width > 0.015",
        "inconclusive_incomplete_safety_cap": (
            "one or more games lack a strict rules terminal"
        ),
    }
    assert plan["workload"]["games"] == 1012
    assert plan["workload"]["max_active_hours"] == 4.0
    assert plan["workload"]["max_sanmill_search_turns"] == 777216
    assert plan["workload"]["max_summed_node_ceiling"] == 388608000000
    assert plan["protocol"]["automatic_retry"] is False
    assert plan["protocol"]["semantic_failure_recovery"] is False
    assert plan["claim_boundary"]["equivalence_claim"] is False
    assert plan["claim_boundary"]["refresh_causal_claim"] is False
    assert plan["claim_boundary"]["automatic_promotion"] is False


def test_plan_freezer_never_overwrites_an_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    target.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        freezer.write_plan(freezer.build_plan(), target)

    assert target.read_text(encoding="utf-8") == "preserve"
