"""Tests for learned_ai/sentinel/infer.py (SentinelAdvisor)."""

from __future__ import annotations

import time

import torch

from game.board import BoardState
from game.rules import get_all_legal_moves
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.infer import SentinelAdvice, SentinelAdvisor, load_advisor
from learned_ai.sentinel.model import SentinelNet


def _board():
    return BoardState.from_fen_string("BBW....B.W.W............|W|3|3")


def _candidates(board):
    """Return legal moves as candidate dicts."""
    return [{"from": m.get("from"), "to": m["to"], "capture": m.get("capture")}
            for m in get_all_legal_moves(board)]


def _trained_advisor(tmp_path):
    cfg = SentinelConfig(hidden_dims=[64, 32], dropout=0.0)
    net = SentinelNet(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims, dropout=0.0, aux_wdl=False)
    ckpt = tmp_path / "sentinel.pt"
    torch.save({"state_dict": net.state_dict(), "config": cfg.to_dict(), "aux_wdl": False}, ckpt)
    return SentinelAdvisor(str(ckpt), config=cfg, device="cpu")


def test_advise_returns_sentinel_advice(tmp_path):
    advisor = _trained_advisor(tmp_path)
    assert advisor.is_loaded()
    board = _board()
    cands = _candidates(board)
    advice = advisor.advise(board, cands, "W", 0)
    assert isinstance(advice, SentinelAdvice)
    assert len(advice.move_scores) == len(cands)
    assert all(0.0 <= s <= 1.0 for s in advice.move_scores)
    assert 0.0 <= advice.played_move_quality <= 1.0
    assert 0.0 <= advice.best_available_quality <= 1.0
    assert advice.opportunity_gap >= 0.0
    assert advice.advisory_message in ("safe", "possible_mistake", "missed_opportunity", "critical")


def test_advise_fast(tmp_path):
    advisor = _trained_advisor(tmp_path)
    board = _board()
    cands = _candidates(board)
    advisor.advise(board, cands, "W", 0)   # warmup
    t0 = time.perf_counter()
    advisor.advise(board, cands, "W", 0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"advise took {elapsed_ms:.2f}ms"


def test_advise_no_crash_empty_candidates(tmp_path):
    advisor = _trained_advisor(tmp_path)
    result = advisor.advise(_board(), [], "W", 0)
    assert result is None


def test_unloaded_advisor_returns_none():
    advisor = SentinelAdvisor()   # no checkpoint
    assert not advisor.is_loaded()
    result = advisor.advise(_board(), _candidates(_board()), "W", 0)
    assert result is None


def test_load_advisor_helper(tmp_path):
    cfg = SentinelConfig(hidden_dims=[32], dropout=0.0)
    net = SentinelNet(input_dim=cfg.input_dim, hidden_dims=cfg.hidden_dims, dropout=0.0, aux_wdl=False)
    ckpt = tmp_path / "sentinel.pt"
    torch.save({"state_dict": net.state_dict(), "config": cfg.to_dict(), "aux_wdl": False}, ckpt)
    adv = load_advisor(str(ckpt))
    assert adv is not None and adv.is_loaded()
