from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from learned_ai.evaluation.human_feature_deviation_estimator_readiness import (
    EstimatorAccess,
    EstimatorReadinessError,
)
from learned_ai.evaluation.sanmill_safe_inducement import (
    MAIN_PLAN_SCHEMA,
    PLAN_SCHEMA,
    SafeInducementError,
    classify_main,
    classify_preprobe,
    decompose_budget_stability,
    frequency_weighted_gain,
    load_main_plan,
    load_plan,
    summarize_measurements,
)
from learned_ai.training.run_contract import canonical_sha256
from scripts import run_sanmill_safe_inducement_main_v2 as main_runner


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


def _main_plan() -> dict:
    return {
        "schema_version": MAIN_PLAN_SCHEMA,
        "status": "frozen_protocol_v2_execution_unlaunched",
        "claim_boundary": {
            "safe_set": "A_pos",
            "positional_only": True,
            "A_allow_claim": False,
            "human_trap_claim": False,
        },
        "main_experiment": {
            "node_budgets": [1_000, 100_000, 500_000],
            "primary_node_budget": 100_000,
            "interval": {"repetitions": 100, "seed": "main"},
            "mechanism_success_gate": {
                "minimum_point_o_minus_b": 0.05,
                "minimum_lower_95_o_minus_b": 0.05,
                "minimum_evaluable_states": 330,
                "determinism_gate": True,
                "all_conditions_conjunctive": True,
            },
            "budget_decomposition": {
                "fixed_blind_spot_interpretation_threshold": 0.80,
            },
            "frequency_weighted_secondary": {
                "phase_counts": {
                    "placement": 2,
                    "movement": 1,
                    "flying": 1,
                },
                "weights": {
                    "placement": 0.5,
                    "movement": 0.25,
                    "flying": 0.25,
                },
            },
            "resource_envelope": {
                "maximum_states": 360,
                "maximum_engine_single_step_queries": 40_000,
                "maximum_malom_queries": 250_000,
                "maximum_active_seconds": 14_400,
                "maximum_concurrent_evaluators": 1,
                "maximum_concurrent_sanmill_processes": 1,
                "maximum_complete_games": 0,
                "maximum_model_loads": 0,
                "maximum_training_updates": 0,
                "stop_at_any_limit": True,
                "automatic_retry_or_extension": False,
            },
        },
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


def test_budget_decomposition_separates_invariant_and_sensitive_states() -> None:
    plan = _main_plan()
    rows = []
    for state, phase in (
        ("invariant", "placement"),
        ("sensitive", "flying"),
        ("never", "movement"),
    ):
        for action in range(2):
            for budget in plan["main_experiment"]["node_budgets"]:
                downgraded = state == "invariant" and action == 0
                if state == "sensitive" and action == 1 and budget == 100_000:
                    downgraded = True
                rows.append(
                    {
                        "state_id": state,
                        "phase": phase,
                        "a_pos_index": action,
                        "node_budget": budget,
                        "downgrade_transition": "D->L" if downgraded else None,
                        "abstained": False,
                    }
                )
    result = decompose_budget_stability(rows, plan=plan)
    assert result["overall"]["o_inv"] == pytest.approx(1 / 3)
    assert result["overall"]["o_sens"] == pytest.approx(1 / 3)
    assert result["overall"]["o_union"] == pytest.approx(2 / 3)
    assert result["overall"]["invariant_share_of_induced_states"] == pytest.approx(
        0.5
    )
    assert result["overall"][
        "identity_check_o_union_equals_o_inv_plus_o_sens"
    ]
    assert result["interpretation"] == "budget_sensitive_component_not_negligible"


def test_frequency_weighted_gain_is_secondary_and_cannot_flip_primary() -> None:
    plan = _main_plan()
    summaries = {
        "100000": {
            "by_phase": {
                "placement": {"estimates": {"o_minus_b": {"mean": 0.10}}},
                "movement": {"estimates": {"o_minus_b": {"mean": 0.02}}},
                "flying": {"estimates": {"o_minus_b": {"mean": 0.30}}},
            }
        }
    }
    result = frequency_weighted_gain(summaries, plan=plan)
    assert result["weighted_o_minus_b"] == pytest.approx(0.13)
    assert result["threshold"] is None
    assert result["can_flip_primary_decision"] is False


def test_main_gate_remains_conjunctive_at_100k() -> None:
    plan = _main_plan()
    summaries = {
        "100000": {
            "overall": {
                "states_evaluable": 350,
                "estimates": {
                    "o_minus_b": {
                        "mean": 0.08,
                        "state_bootstrap_percentile_95": {
                            "lower_95": 0.049,
                            "upper_95": 0.12,
                        },
                    }
                },
            }
        }
    }
    result = classify_main(summaries, plan=plan, determinism_passed=True)
    assert result["decision"] == "mechanism_gate_failed"
    assert result["failures"] == ["lower_95_o_minus_b_below_5pp"]


def test_main_plan_loader_rejects_threshold_drift(tmp_path: Path) -> None:
    plan = _main_plan()
    plan["plan_identity"] = canonical_sha256(plan)
    path = tmp_path / "plan-v2.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded, _file_sha = load_main_plan(path)
    assert loaded["plan_identity"] == plan["plan_identity"]

    changed = _main_plan()
    changed["main_experiment"]["mechanism_success_gate"][
        "minimum_point_o_minus_b"
    ] = 0.049
    changed["plan_identity"] = canonical_sha256(changed)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SafeInducementError, match="mechanism gate differs"):
        load_main_plan(path)


def test_sanmill_process_check_accepts_windows_zero_count(monkeypatch) -> None:
    monkeypatch.setattr(
        main_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="0\n",
            stderr="",
        ),
    )
    assert main_runner._running_tgf_processes() == 0


def test_sanmill_process_check_fails_closed_on_malformed_count(monkeypatch) -> None:
    monkeypatch.setattr(
        main_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="no process\n",
            stderr="",
        ),
    )
    with pytest.raises(SafeInducementError, match="count is malformed"):
        main_runner._running_tgf_processes()
