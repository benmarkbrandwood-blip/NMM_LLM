"""Tests for the immutable retained phase-process machine plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_retained_phase_process_generalization as runner
import tools.freeze_retained_phase_process_plan as freeze


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / (
    "docs/experiments/"
    "sanmill-retained-v3-v4-phase-process-generalization-v1.json"
)


def test_committed_plan_is_reproducible_canonical_and_runtime_valid() -> None:
    persisted = json.loads(PLAN.read_text(encoding="utf-8"))
    rebuilt = freeze.build_plan()
    assert rebuilt == persisted
    assert runner.load_plan(PLAN) == persisted
    assert persisted["plan_identity"] == (
        "4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256"
    )
    assert persisted["implementation"]["commit"] == (
        "5a318a063b561b12bafe5e72e44ff6fdc9426f1e"
    )


def test_plan_binds_successor_inputs_relative_horizon_and_claim_boundary() -> None:
    plan = runner.load_plan(PLAN)
    assert plan["inputs"]["snapshot_identity"] == (
        "b35ecc061e53a35e227c69ff886a7c6534e707bd124abdbe13acbbf9647f48ac"
    )
    assert all(
        "phase-process-generalization-v1/inputs" in candidate["bundle"]["path"]
        and "phase-process-generalization-v1/inputs"
        in candidate["specialist_db"]["path"]
        for candidate in plan["candidates"]
    )
    assert plan["protocol"]["horizon_post_start_logical_plies"] == 108
    assert plan["protocol"]["max_post_start_logical_plies"] == 1536
    assert plan["workload"] == {
        "automatic_retry_or_recovery": False,
        "candidate_colors_per_start": 2,
        "candidates_per_color_unit": 2,
        "expansion": False,
        "games": 156,
        "host_interruption_exact_resume_only": True,
        "matched_color_units": 78,
        "max_active_hours": 2.0,
        "max_sanmill_search_turns": 119808,
        "max_summed_node_ceiling": 59904000000,
        "mechanism_reanalysis_new_games": 0,
        "safe_exact_resume_same_spec": True,
        "semantic_failure_recovery": False,
        "unique_starts": 39,
    }
    assert plan["claim_boundary"] == {
        "corpus_previously_project_visible": True,
        "equivalence_claim": False,
        "held_out": False,
        "held_out_strength_claim": False,
        "playing_strength_claim": False,
        "promotion": False,
        "publication": False,
        "refresh_causal_claim": False,
        "release": False,
        "training_or_update": False,
    }
    assert plan["analysis"]["engineering_interval"][
        "method"
    ] == "normal-interval-on-start-clustered-difference"
    assert plan["analysis"]["mechanism_metrics"].startswith("exploratory")


def test_plan_freezer_never_overwrites_an_existing_target(tmp_path) -> None:
    output = tmp_path / "plan.json"
    freeze.write_plan(freeze.build_plan(), output)
    with pytest.raises(FileExistsError):
        freeze.write_plan(freeze.build_plan(), output)
