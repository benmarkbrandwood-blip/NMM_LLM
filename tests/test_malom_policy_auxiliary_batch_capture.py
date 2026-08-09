"""Focused tests for the no-update multi-seed batch capture."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from learned_ai.training.run_contract import canonical_sha256
from learned_ai.training.scaffolded_a2c import ScaffoldedStep
from learned_ai.validation import malom_policy_auxiliary_batch_capture as capture
from scripts import train_s_gen_v2 as trainer


_ROOT = Path(__file__).resolve().parents[1]


def _step() -> ScaffoldedStep:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return ScaffoldedStep(
        move_features=features,
        value_input=np.zeros(1, dtype=np.float32),
        chosen_idx=0,
        log_prob_old=-0.5,
        reward=0.0,
        next_move_features=features.copy(),
        next_value_input=np.zeros(1, dtype=np.float32),
        done=False,
        behaviour_temperature=0.9,
        malom_preserving_mask=np.asarray([True, False]),
    )


def test_tracked_plan_freezes_balanced_three_seed_schedule() -> None:
    plan = capture.load_batch_capture_plan(_ROOT / capture.DEFAULT_PLAN_RELATIVE)

    assert plan.identity == capture.EXPECTED_PLAN_IDENTITY
    assert plan.seeds == (52, 53, 54)
    assert len(plan.schedule) == 60
    assert len({game.game_id for game in plan.schedule}) == 60
    assert sum(game.opponent_kind == "sanmill" for game in plan.schedule) == 24
    assert sum(game.opponent_kind == "frozen_target" for game in plan.schedule) == 36
    for seed in plan.seeds:
        games = [game for game in plan.schedule if game.seed == seed]
        assert len(games) == 20
        assert sum(game.route_depth == "deep" for game in games) == 1
        for opponent, count in (("sanmill", 8), ("frozen_target", 12)):
            selected = [game for game in games if game.opponent_kind == opponent]
            assert len(selected) == count
            assert sum(game.learner_color == "W" for game in selected) == count // 2


def test_plan_identity_rejects_schedule_mutation(tmp_path: Path) -> None:
    source = _ROOT / capture.DEFAULT_PLAN_RELATIVE
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["schedule_contract"]["pattern"][0]["learner_color"] = "B"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        capture.MalomPolicyAuxiliaryBatchCaptureError,
        match="identity mismatch",
    ):
        capture.load_batch_capture_plan(changed)


def test_source_evidence_matches_ignored_calibration_artifacts() -> None:
    plan = capture.load_batch_capture_plan(_ROOT / capture.DEFAULT_PLAN_RELATIVE)
    expected = plan.payload["source_evidence"]
    if not all(
        (_ROOT / expected[key]).is_file()
        for key in (
            "calibration_result_path",
            "gradient_interaction_report_path",
        )
    ):
        pytest.skip("requires ignored calibration result artifacts")

    record = capture.verify_source_evidence(plan)

    assert (
        record["calibration_result"]["result_identity"]
        == expected["calibration_result_identity"]
    )
    assert (
        record["gradient_interaction_report"]["audit_identity"]
        == expected["gradient_interaction_audit_identity"]
    )


def test_batch_accumulator_preserves_whole_games_and_final_minimum() -> None:
    accumulator = capture.ProductionBatchAccumulator(
        threshold=64,
        final_minimum=8,
    )
    first = {"game_id": "one"}
    second = {"game_id": "two"}
    for value in (first, second):
        value.update(
            {
                "opponent_kind": "sanmill",
                "learner_color": "W",
                "termination_reason": "test",
            }
        )

    assert accumulator.append_game(first, [_step()] * 40) is None
    periodic = accumulator.append_game(second, [_step()] * 30)

    assert periodic is not None
    assert periodic.reason == "periodic"
    assert len(periodic.steps) == 70
    assert [item.steps for item in periodic.contributions] == [40, 30]
    assert accumulator.pending_steps == 0

    assert accumulator.append_game(first, [_step()] * 7) is None
    final, excluded = accumulator.finish()
    assert final is None
    assert excluded is not None
    assert excluded["steps"] == 7
    assert excluded["minimum_steps"] == 8


class _FakeRuntime:
    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(2, 1)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.frozen_opponent = SimpleNamespace(_model=copy.deepcopy(self.model))
        self.lookahead_advisor = object()
        self.human_route = object()
        self.specialist_route = object()
        self.malom_route = object()
        self.installation = object()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeGame:
    def __init__(self, _installation, *, seed: int) -> None:
        self.seed = seed
        self.state = SimpleNamespace(
            fen="fake",
            history_sha256="a" * 64,
            logical_ply_count=8,
            terminal=False,
            winner=None,
            outcome_reason_code="ongoing",
        )

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None


def test_execution_collects_production_batches_without_updates(monkeypatch) -> None:
    plan = capture.load_batch_capture_plan(_ROOT / capture.DEFAULT_PLAN_RELATIVE)
    runtimes: list[_FakeRuntime] = []
    rollout_calls: list[dict] = []
    measurement_sizes: list[int] = []
    rescore_calls = 0

    def runtime_factory(_plan, _inputs, seed):
        runtime = _FakeRuntime(seed)
        runtimes.append(runtime)
        return runtime

    def opponent_factory(_game, *, node_budget, depth):
        return SimpleNamespace(node_budget=node_budget, depth=depth)

    def rollout_fn(**kwargs):
        rollout_calls.append(kwargs)
        budget = getattr(kwargs["opponent"], "node_budget", None)
        observations = [] if budget is None else [{"nodes": 17, "depth": 3}]
        rewards = [SimpleNamespace(retro=0.0, total=0.0) for _ in range(4)]
        return SimpleNamespace(
            trajectory=[_step() for _ in range(4)],
            step_diags=[
                SimpleNamespace(reward=reward, was_top1_heuristic=0)
                for reward in rewards
            ],
            outcome=trainer.DRAW_LONG,
            termination_reason="max-ply-truncation",
            ply=8,
            phase_ply_counts={"place": 8},
            compound_turn_count=0,
            opponent_search_observations=observations,
            opponent_search_calls=len(observations),
            opponent_search_nodes=sum(item["nodes"] for item in observations),
            opponent_search_depth_sum=sum(item["depth"] for item in observations),
        )

    def rescore(*_args, **_kwargs):
        nonlocal rescore_calls
        rescore_calls += 1

    def measurement(_model, steps, **kwargs):
        measurement_sizes.append(len(steps))
        return {
            "support": {"steps": len(steps), "informative_steps": len(steps)},
            "candidate_scales": [
                {
                    "target_policy_head_ratio": value,
                    "status": "measured",
                    "effective_coefficient": value,
                }
                for value in kwargs["target_policy_head_ratios"]
            ],
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }

    monkeypatch.setattr(trainer, "_retroactive_rescore", rescore)
    result = capture.execute_batch_capture(
        plan,
        cast(Any, SimpleNamespace()),
        runtime_factory=cast(Any, runtime_factory),
        game_factory=_FakeGame,
        opponent_factory=opponent_factory,
        rollout_fn=rollout_fn,
        measurement_fn=measurement,
    )

    assert len(result["samples"]) == 60
    assert len(result["batches"]) == 6
    assert measurement_sizes == [64, 16, 64, 16, 64, 16]
    assert rescore_calls == 60
    assert all(runtime.closed for runtime in runtimes)
    assert all(call["persist_rollout_evidence"] is False for call in rollout_calls)
    assert all(call["record_branches"] is False for call in rollout_calls)
    assert all(call["retry_ply"] == 0 for call in rollout_calls)
    assert all(call["malom_policy_aux_coef"] == 1.0 for call in rollout_calls)
    assert all(
        record["learner_unchanged"] and record["frozen_unchanged"]
        for record in result["models"]
    )


def test_summary_disaggregates_observed_support_without_selecting() -> None:
    samples = [
        {
            "seed": 52,
            "opponent_kind": "sanmill",
            "learner_color": "W",
            "outcome": "loss",
            "termination_reason": "material-loss",
            "learner_steps": 40,
            "malom_informative_steps": 5,
        },
        {
            "seed": 52,
            "opponent_kind": "frozen_target",
            "learner_color": "B",
            "outcome": "draw",
            "termination_reason": "draw_threefold_repetition",
            "learner_steps": 30,
            "malom_informative_steps": 3,
        },
    ]
    batches = [
        {
            "seed": 52,
            "steps": 70,
            "gradient_measurement": {
                "support": {
                    "informative_steps": 8,
                    "labelled_by_phase": {
                        "placement": 40,
                        "movement": 30,
                        "flying": 0,
                    },
                    "informative_by_phase": {
                        "placement": 5,
                        "movement": 3,
                        "flying": 0,
                    },
                },
                "ordinary_policy_head_gradient_l2": 0.1,
                "raw_auxiliary_gradient_l2": 0.2,
                "raw_auxiliary_to_ordinary_policy_head_cosine": -0.25,
                "candidate_scales": [
                    {
                        "target_policy_head_ratio": 0.5,
                        "status": "measured",
                        "effective_coefficient": 0.25,
                    }
                ],
            },
        }
    ]

    summary = capture.summarize_batch_capture(samples, batches)

    assert summary["games"] == 2
    assert summary["batches"] == 1
    assert summary["labelled_steps_by_phase"] == {
        "flying": 0,
        "movement": 30,
        "placement": 40,
    }
    assert summary["informative_steps_by_phase"]["movement"] == 3
    assert summary["candidate_scale_distributions"][0]["effective_coefficient"][
        "median"
    ] == pytest.approx(0.25)
    assert summary["selection_made"] is False


def test_readiness_validation_binds_plan_source_and_identity(monkeypatch) -> None:
    plan = capture.load_batch_capture_plan(_ROOT / capture.DEFAULT_PLAN_RELATIVE)
    source = {"branch": "dev", "commit": "a" * 40}
    source_evidence = {"calibration_result": {"sha256": "c" * 64}}
    monkeypatch.setattr(
        capture,
        "verify_source_evidence",
        lambda _plan: source_evidence,
    )
    body = {
        "schema_version": capture.PREFLIGHT_SCHEMA,
        "status": "ready_for_explicit_one_run_authorization",
        "launch_authorized": False,
        "plan": {
            "relative_path": capture.DEFAULT_PLAN_RELATIVE.as_posix(),
            "raw_sha256": plan.raw_sha256,
            "identity": plan.identity,
        },
        "source": source,
        "parent_route_preflight": {},
        "data": {},
        "source_evidence": source_evidence,
        "fresh_initializations": [
            {
                "seed": seed,
                "learner_sha256": f"{seed:064x}",
                "frozen_target_sha256": f"{seed:064x}",
            }
            for seed in plan.seeds
        ],
        "gradient_contract": {
            "target_policy_head_ratios": list(plan.target_ratios),
            "denominator_floor": plan.denominator_floor,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "bounded_work": dict(plan.payload["bounded_work"]),
        "claim_boundary": dict(plan.payload["claim_boundary"]),
        "next_gate": "test",
    }
    report = {**body, "readiness_identity": canonical_sha256(body)}

    capture.validate_readiness(
        report,
        plan,
        expected_identity=report["readiness_identity"],
        source=source,
    )
    changed = {**source, "commit": "b" * 40}
    with pytest.raises(
        capture.MalomPolicyAuxiliaryBatchCaptureError,
        match="source commit drifted",
    ):
        capture.validate_readiness(
            report,
            plan,
            expected_identity=report["readiness_identity"],
            source=changed,
        )
