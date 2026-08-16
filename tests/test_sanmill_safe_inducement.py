from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
    EstimatorReadinessError,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    PLAN_SCHEMA,
    SafeInducementError,
    classify_preprobe,
    load_plan,
    summarize_measurements,
)
from learned_ai.training.run_contract import canonical_sha256


def _plan() -> dict:
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "frozen_preprobe_authorized_main_unlaunched",
        "claim_boundary": {
            "safe_set": "A_pos",
            "positional_only": True,
            "human_trap_claim": False,
        },
        "preprobe": {
            "node_budgets": [1_000, 10_000, 100_000, 500_000],
            "measurement_order_seed": "order",
            "resource_envelope": {
                "maximum_engine_single_step_queries": 100_000,
                "maximum_active_seconds": 7_200,
                "maximum_concurrent_evaluators": 1,
                "maximum_concurrent_sanmill_processes": 1,
            },
            "uncertainty": {
                "bootstrap_repetitions": 200,
                "bootstrap_seed": "bootstrap",
            },
            "signal_gate": {
                "minimum_evaluable_states": 2,
                "minimum_downgrade_actions": 1,
                "minimum_baseline_rate": 0.01,
                "maximum_baseline_rate": 0.30,
                "minimum_oracle_gain": 0.05,
            },
        },
    }


def _row(state: str, *, downgraded: bool, action: int) -> dict:
    return {
        "state_id": state,
        "phase": "movement",
        "engine_reply_state_tier": "D",
        "a_pos_cardinality": 2,
        "node_budget": 1_000,
        "a_pos_index": action,
        "searched": True,
        "strict_terminal": False,
        "downgrade_transition": "D->L" if downgraded else None,
        "search_elapsed_seconds": 0.01,
        "abstained": False,
    }


def test_state_uniform_estimands_do_not_pool_actions() -> None:
    plan = _plan()
    rows = [
        _row("a", downgraded=True, action=0),
        _row("a", downgraded=False, action=1),
        _row("b", downgraded=False, action=0),
        _row("b", downgraded=False, action=1),
    ]
    for budget in (10_000, 100_000, 500_000):
        rows.extend([{**row, "node_budget": budget} for row in rows[:4]])
    summary = summarize_measurements(rows, plan=plan)["1000"]["overall"]
    assert summary["estimates"]["b"]["mean"] == pytest.approx(0.25)
    assert summary["estimates"]["o"]["mean"] == pytest.approx(0.5)
    assert summary["estimates"]["o_minus_b"]["mean"] == pytest.approx(0.25)


def test_signal_gate_selects_highest_passing_budget() -> None:
    plan = _plan()
    template = {
        "states_evaluable": 3,
        "downgrade_actions": 2,
        "estimates": {
            "b": {"mean": 0.1},
            "o_minus_b": {"mean": 0.2},
        },
    }
    summaries = {
        str(budget): {"overall": dict(template)}
        for budget in plan["preprobe"]["node_budgets"]
    }
    decision = classify_preprobe(
        summaries,
        plan=plan,
        determinism_passed=True,
    )
    assert decision["conclusion"].startswith("A_signal_region")
    assert decision["recommended_main_node_budget"] == 500_000


def test_failed_determinism_is_design_conclusion_c() -> None:
    decision = classify_preprobe({}, plan=_plan(), determinism_passed=False)
    assert decision == {
        "conclusion": "C_design_invalid",
        "reason": "determinism_gate_failed",
        "recommended_main_node_budget": None,
    }


def test_plan_loader_rejects_identity_drift(tmp_path: Path) -> None:
    plan = _plan()
    plan["plan_identity"] = canonical_sha256(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded, _file_sha = load_plan(path)
    assert loaded["plan_identity"] == plan["plan_identity"]
    plan["preprobe"]["node_budgets"][0] = 999
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(SafeInducementError, match="identity differs"):
        load_plan(path)


def test_protected_final_test_access_fails_before_producer() -> None:
    official = {
        "partitions": {
            "train": {"session_ids": ["train"]},
            "selection": {"session_ids": []},
            "confirmation": {"session_ids": []},
            "final-test": {"session_ids": ["sealed"]},
        }
    }
    research = {
        "partitions": {
            "research-exploration": {"session_ids": ["train"]},
            "research-confirmation": {"session_ids": []},
            "cross-player-discard": {"session_ids": []},
        }
    }
    access = EstimatorAccess.from_memberships(
        official,
        research,
        allowed_sessions=["train"],
    )
    called = False

    def producer() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(EstimatorReadinessError, match="access denied"):
        access.derive("sealed", access_kind="raw_game", producer=producer)
    assert called is False


def test_protected_research_confirmation_cannot_enter_allowlist() -> None:
    official = {
        "partitions": {
            "train": {"session_ids": ["explore", "confirm"]},
            "selection": {"session_ids": []},
            "confirmation": {"session_ids": []},
            "final-test": {"session_ids": []},
        }
    }
    research = {
        "partitions": {
            "research-exploration": {"session_ids": ["explore"]},
            "research-confirmation": {"session_ids": ["confirm"]},
            "cross-player-discard": {"session_ids": []},
        }
    }
    with pytest.raises(EstimatorReadinessError, match="crosses protection"):
        EstimatorAccess.from_memberships(
            official,
            research,
            allowed_sessions=["confirm"],
        )
