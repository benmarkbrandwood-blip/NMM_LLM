from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from learned_ai.evaluation.sanmill_trained_model_lightweight import (
    CANDIDATE_ARMS,
    EXPECTED_CANDIDATE_GAMES,
    EXPECTED_REPRODUCTION_GAMES,
    EXPECTED_TOTAL_GAMES,
    LightweightMeasurementError,
    candidate_schedule,
    exact_reproduction_gate,
    load_plan,
    reproduction_schedule,
)


ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = (
    ROOT / "docs/experiments/sanmill-safe-guidance-gameplay-start-pool-v1.json"
)
REFERENCE_PATH = (
    ROOT
    / "docs/evidence/"
    "sanmill-safe-guidance-gameplay-attempt-002-manifest-2026-08-16.json"
)
EXCLUDED = [
    "00092c974cabf05874f066b8948e791f9fdc82d84a65759da1ba78f212a643b0"
]


@pytest.fixture(scope="module")
def pool() -> dict:
    return json.loads(POOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def test_reproduction_schedule_preserves_attempt_002_game_identities(
    pool: dict, reference: dict
) -> None:
    rows = reproduction_schedule(pool, excluded_start_ids=EXCLUDED)
    expected = [row for row in reference["games"] if row["arm"] == "random-safe"]
    assert len(rows) == EXPECTED_REPRODUCTION_GAMES
    assert {
        (row["start_id"], row["candidate_color"], row["game_id"])
        for row in rows
    } == {
        (row["start_id"], row["candidate_color"], row["game_id"])
        for row in expected
    }


def test_known_answer_gate_accepts_exact_reference_and_rejects_one_wrong_game(
    reference: dict,
) -> None:
    rows = [row for row in reference["games"] if row["arm"] == "random-safe"]
    gate = exact_reproduction_gate(rows, reference)
    assert gate["passed"] is True
    assert gate["mismatch_count"] == 0
    assert gate["summary"]["score_rate"] == 0.44881889763779526

    changed = copy.deepcopy(rows)
    changed[0]["outcome_reason"] = "drawThreefoldRepetition"
    failed = exact_reproduction_gate(changed, reference)
    assert failed["passed"] is False
    assert failed["candidate_measurement_allowed"] is False
    assert failed["mismatch_count"] == 1


def test_known_answer_gate_rejects_missing_game(reference: dict) -> None:
    rows = [row for row in reference["games"] if row["arm"] == "random-safe"]
    with pytest.raises(LightweightMeasurementError, match="count"):
        exact_reproduction_gate(rows[:-1], reference)


def test_candidate_schedule_is_exactly_four_arms_per_color(pool: dict) -> None:
    rows = candidate_schedule(
        pool,
        excluded_start_ids=EXCLUDED,
        namespace="test-lightweight-candidate-schedule",
    )
    assert len(rows) == EXPECTED_CANDIDATE_GAMES
    assert {row["arm"] for row in rows} == set(CANDIDATE_ARMS)
    assert len({(row["start_id"], row["candidate_color"]) for row in rows}) == 508
    assert len(rows) + EXPECTED_REPRODUCTION_GAMES == EXPECTED_TOTAL_GAMES


def test_frozen_plan_is_lightweight_and_does_not_bind_old_gates() -> None:
    plan_path = (
        ROOT / "docs/experiments/sanmill-trained-model-lightweight-v1.json"
    )
    if not plan_path.exists():
        pytest.skip("plan is generated after implementation freeze")
    plan, _digest = load_plan(plan_path)
    rendered = json.dumps(plan, sort_keys=True)
    assert "boundary-registry" not in rendered
    assert "coverage_ledger" not in rendered
    assert "old_authorization" not in rendered
    assert plan["experiment"]["planned_total_games"] == 2540
    assert plan["resource_envelope"]["authorized_literal_maximum_complete_games"] == 3048
    assert plan["primary_decision"]["maximum_95_half_width"] == 0.015


def test_runner_has_no_old_boundary_or_authorization_dependency() -> None:
    source = (
        ROOT / "scripts/run_sanmill_trained_model_lightweight.py"
    ).read_text(encoding="utf-8")
    assert "sanmill_trained_model_boundary_registry" not in source
    assert "load_attempt_authorization" not in source
    assert "load_rehearsal" not in source
    assert "load_preflight" not in source
