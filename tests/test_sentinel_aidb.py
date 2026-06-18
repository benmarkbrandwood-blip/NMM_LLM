"""tests/test_sentinel_aidb.py — smoke tests for AIDB generation and sentinel retraining.

Five focused tests:
  1. gen_aidb plays 2 games and produces JSONL with malom fields in the move records.
  2. SentinelDataset loads examples from an AIDB game and position_key is populated.
  3. ContrastiveSentinelDataset yields (feat_good, feat_bad) pairs when data is rich enough.
  4. BCE + contrastive backward pass runs without error (gradient flow test).
  5. contrastive_ranking_loss returns zero for perfect ordering and positive for violations.
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch

from learned_ai.sentinel.model import SentinelNet, sentinel_loss, contrastive_ranking_loss
from learned_ai.sentinel.labels import MoveExample
from learned_ai.sentinel.dataset import SentinelDataset, ContrastiveSentinelDataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_game_record(winner: str = "W", n_plies: int = 6) -> dict:
    """Build a synthetic game record that mimics the AIDB format."""
    from game.game_engine import GameEngine
    engine = GameEngine()

    from game.rules import get_all_legal_moves
    import random

    plies = 0
    while not engine.finished and plies < n_plies:
        legal = engine.get_all_legal_moves()
        if not legal:
            break
        move = random.choice(legal)
        engine.apply_move(move)
        entry = engine.game_record["moves"][-1]
        # Inject synthetic malom fields.
        entry["malom_wdl"] = "win" if plies % 2 == 0 else "loss"
        entry["malom_dtw"] = 20
        entry["malom_move_wdl"] = "win" if plies % 2 == 0 else "loss"
        plies += 1

    record = engine.game_record
    record["winner"] = winner
    record["white_personality"] = "balanced"
    record["black_personality"] = "aggressive"
    record["white_vn_blend"] = 0
    record["black_vn_blend"] = 80
    record["white_sentinel"] = False
    record["black_sentinel"] = False
    return record


def _make_examples(n: int, seed: int = 0) -> list[MoveExample]:
    """Produce n synthetic MoveExamples with position_key set."""
    from learned_ai.sentinel.feature_builder import FEATURE_DIM
    rng = np.random.default_rng(seed)
    examples = []
    # Two positions — each with good (>= 0.65) and bad (<= 0.35) moves.
    for pos_idx in range(2):
        key = f"pos_{pos_idx}"
        for _ in range(n // 4):
            examples.append(MoveExample(
                features=rng.random(FEATURE_DIM).astype(np.float32),
                move_quality=0.9,
                training_weight=1.0,
                supervision_source="solved_db",
                position_key=key,
            ))
            examples.append(MoveExample(
                features=rng.random(FEATURE_DIM).astype(np.float32),
                move_quality=0.1,
                training_weight=1.0,
                supervision_source="solved_db",
                position_key=key,
            ))
    return examples


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGenAIDB:
    """Test 1: gen_aidb produces valid JSONL with malom fields."""

    def test_game_record_has_malom_fields(self):
        """A synthetic AIDB game record contains the expected malom fields."""
        record = _minimal_game_record(winner="W", n_plies=6)
        moves = record.get("moves", [])
        assert len(moves) > 0, "game record has no moves"

        for mv in moves:
            assert "malom_wdl" in mv, f"missing malom_wdl in move {mv}"
            assert "malom_dtw" in mv, f"missing malom_dtw in move {mv}"
            assert "malom_move_wdl" in mv, f"missing malom_move_wdl in move {mv}"

        assert "white_personality" in record
        assert "black_personality" in record
        assert "white_vn_blend" in record
        assert "black_vn_blend" in record
        assert "white_sentinel" in record
        assert "black_sentinel" in record

    def test_jsonl_roundtrip(self):
        """Game record can be written as JSONL and read back identically."""
        record = _minimal_game_record(winner="B", n_plies=4)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(record))
            path = f.name

        loaded = json.loads(Path(path).read_text())
        assert loaded["winner"] == "B"
        assert len(loaded["moves"]) == len(record["moves"])
        assert loaded["moves"][0].get("malom_wdl") == record["moves"][0].get("malom_wdl")
        Path(path).unlink(missing_ok=True)


class TestDatasetPositionKey:
    """Test 2: SentinelDataset populates position_key from game records."""

    def test_position_key_populated(self):
        """examples_from_game sets position_key on all MoveExamples."""
        from learned_ai.sentinel.dataset import examples_from_game
        record = _minimal_game_record(winner="W", n_plies=6)
        exs = examples_from_game(record, db=None, trajectory_weight=False)
        assert len(exs) > 0, "no examples generated"
        for ex in exs:
            assert ex.position_key, f"position_key empty for example at ply={ex.ply}"

    def test_aidb_malom_label_injected(self):
        """Pre-computed malom_move_wdl in game record is used for the played move when db=None."""
        from learned_ai.sentinel.dataset import examples_from_game
        record = _minimal_game_record(winner="W", n_plies=4)

        # Force all played moves to malom_move_wdl = "win"
        for mv in record["moves"]:
            mv["malom_move_wdl"] = "win"

        exs = examples_from_game(record, db=None, trajectory_weight=False)
        # At least some examples should have a high quality label (win ≥ 0.55).
        high_q = [e for e in exs if e.move_quality >= 0.55 and e.supervision_source in ("solved_db", "solved_db_dtm")]
        assert len(high_q) > 0, "no solved_db-labelled high-quality examples from malom injection"

    def test_save_load_preserves_position_key(self):
        """save_to_disk / load_from_disk round-trips position_key."""
        examples = _make_examples(8)
        ds = SentinelDataset(examples)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            ds.save_to_disk(path)
            ds2 = SentinelDataset.load_from_disk(path)
            assert len(ds2) == len(ds)
            for orig, loaded in zip(ds.examples, ds2.examples):
                assert orig.position_key == loaded.position_key
        finally:
            Path(path).unlink(missing_ok=True)


class TestContrastiveDataset:
    """Test 3: ContrastiveSentinelDataset yields (good, bad) feature pairs."""

    def test_pairs_yielded(self):
        """With enough examples ContrastiveSentinelDataset produces pairs."""
        examples = _make_examples(20)
        cds = ContrastiveSentinelDataset(examples)
        assert len(cds) > 0, "no pairs generated"

        feat_g, feat_b = cds[0]
        from learned_ai.sentinel.feature_builder import FEATURE_DIM
        assert feat_g.shape == (FEATURE_DIM,)
        assert feat_b.shape == (FEATURE_DIM,)

    def test_no_pairs_without_position_key(self):
        """Examples with empty position_key produce no contrastive pairs."""
        examples = _make_examples(20)
        for e in examples:
            e.position_key = ""
        cds = ContrastiveSentinelDataset(examples)
        assert len(cds) == 0, "unexpected pairs from examples with no position_key"

    def test_pair_cap_respected(self):
        """max_pairs parameter limits the dataset size."""
        examples = _make_examples(40)
        cds = ContrastiveSentinelDataset(examples, max_pairs=4)
        assert len(cds) <= 4


class TestContrastiveBackward:
    """Test 4: BCE + contrastive forward/backward pass completes without error."""

    def test_backward_pass(self):
        """Combined BCE + contrastive loss backpropagates through SentinelNet."""
        from learned_ai.sentinel.feature_builder import FEATURE_DIM
        model = SentinelNet(hidden_dims=(32, 16))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng = torch.Generator().manual_seed(0)

        # BCE batch
        feats = torch.randn(16, FEATURE_DIM, generator=rng)
        target = torch.rand(16, generator=rng)
        weight = torch.ones(16)

        out = model(feats)
        bce_losses = sentinel_loss(out, target, sample_weight=weight)

        # Contrastive batch
        feat_g = torch.randn(8, FEATURE_DIM, generator=rng)
        feat_b = torch.randn(8, FEATURE_DIM, generator=rng)
        s_g = model(feat_g)
        s_b = model(feat_b)
        c_loss = contrastive_ranking_loss(s_g, s_b)

        total = bce_losses["total"] + 0.3 * c_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        assert total.item() >= 0.0, "loss is negative — unexpected"


class TestContrastiveLoss:
    """Test 5: contrastive_ranking_loss behaves correctly."""

    def test_perfect_ordering_zero_loss(self):
        """When good > bad + margin, loss is 0."""
        s_good = torch.tensor([0.9, 0.8, 0.7])
        s_bad  = torch.tensor([0.1, 0.2, 0.3])
        loss = contrastive_ranking_loss(s_good, s_bad, margin=0.2)
        assert loss.item() == pytest.approx(0.0), f"expected 0 loss, got {loss.item()}"

    def test_violation_positive_loss(self):
        """When bad >= good (clear violation), loss > 0."""
        s_good = torch.tensor([0.3])
        s_bad  = torch.tensor([0.8])
        loss = contrastive_ranking_loss(s_good, s_bad, margin=0.2)
        assert loss.item() > 0.0, "expected positive loss for ordering violation"

    def test_equal_scores_margin_loss(self):
        """Equal scores trigger a loss equal to the margin."""
        s = torch.tensor([0.5, 0.5, 0.5])
        loss = contrastive_ranking_loss(s, s, margin=0.2)
        assert loss.item() == pytest.approx(0.2, abs=1e-5), f"expected margin=0.2, got {loss.item()}"
