"""Focused tests for additive trainer search observability."""

from __future__ import annotations

from collections import Counter, deque
from types import SimpleNamespace

import torch

from learned_ai.agents.heuristic_agent import HeuristicAgent
from scripts import train_s_gen_v2 as trainer


def test_heuristic_agent_exposes_completed_search_depth() -> None:
    inner = SimpleNamespace(
        last_depth_reached=7,
        _nodes=123,
        _override_node_budget=500,
    )
    agent = HeuristicAgent(game_ai=inner)

    assert agent.last_search_depth == 7
    assert agent.last_search_nodes == 123


def test_opponent_search_observation_uses_public_wrapper_properties() -> None:
    opponent = SimpleNamespace(last_search_nodes=321, last_search_depth=6)

    assert trainer._opponent_search_observation(opponent) == (321, 6)
    assert trainer._opponent_search_observation(object()) is None


def test_game_diag_reports_mean_realised_opponent_depth() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    result = trainer.RolloutResult(
        trajectory=[],
        step_diags=[],
        outcome=0.0,
        ply=0,
        termination_reason="max-ply-truncation",
        branch_candidates=[],
        opponent_search_nodes=1000,
        opponent_search_calls=2,
        opponent_search_depth_sum=7,
        opponent_node_budget=500,
    )

    diag = trainer._build_game_diag(
        game_id="game:test",
        game_count=1,
        difficulty=2,
        learner_color="W",
        temperature=0.9,
        result=result,
        best_win_rate=0.0,
        win_history=deque(),
        last_update_pl=None,
        last_update_vl=None,
        last_update_ent=None,
        opt=optimizer,
        temp_frozen=False,
        source_ckpt="",
        game_type="vs_heuristic",
        phase_bucket="main",
        is_branch=False,
        branch_ply_start=0,
        target_age=0,
        bucket_counts=Counter(),
    )

    assert diag.opponent_search_depth_mean == 3.5
    assert diag.opponent_search_calls == 2
    assert diag.opponent_search_nodes == 1000
