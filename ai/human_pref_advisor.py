"""ai/human_pref_advisor.py — Inference wrapper around HumanPrefNet weights.

Loads the .npz emitted by tools/train_human_pref_net.py and exposes two calls
used by both GameAI (humanlike-play mode, Step 3 of retrain_v2_plan.md) and
scripts/build_gap_dataset.py (label enrichment).

Layer layout in the .npz:
    input_dim   → int64 array of shape (1,)
    layer_count → int64 array of shape (1,)
    w{i}, b{i}  → float32 weights + biases for each Linear layer, ReLU
                  between all but the last.

Forward pass is pure numpy; no torch dependency at inference time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from game.board import BoardState
from ai.value_net import board_to_features, _INPUT_DIM


class HumanPrefAdvisor:
    """Score / rank / sample candidate moves by predicted human likelihood."""

    def __init__(self, npz_path: str | Path, temperature: float = 1.0):
        data = np.load(str(npz_path))
        if "layer_count" in data.files:
            n_layers = int(data["layer_count"][0])
        else:
            n_layers = sum(1 for k in data.files if k.startswith("w"))
        self._layers = [
            (
                data[f"w{i}"].astype(np.float32),
                data[f"b{i}"].astype(np.float32),
            )
            for i in range(n_layers)
        ]
        expected_in = self._layers[0][0].shape[1]
        if expected_in != _INPUT_DIM:
            raise ValueError(
                f"HumanPrefNet at {npz_path!s} expects input_dim {expected_in}, "
                f"but board_to_features emits {_INPUT_DIM}."
            )
        self.input_dim = expected_in
        self.temperature = max(1e-3, float(temperature))

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _score_batch(self, feats: np.ndarray) -> np.ndarray:
        """Run the MLP on a (N, input_dim) feature matrix → (N,) scores."""
        x = feats.astype(np.float32, copy=False)
        n_layers = len(self._layers)
        for i, (w, b) in enumerate(self._layers):
            x = x @ w.T + b
            if i < n_layers - 1:
                x = np.maximum(x, 0.0, out=x)
        return x.squeeze(-1)

    def _successor_features(
        self, board: BoardState, legal_moves: list[dict]
    ) -> np.ndarray:
        """Extract the 79-float feature for the successor produced by each move.

        Silent skips (e.g. a move that raises during apply_move) yield a zero
        row so the caller gets a stable ordering that matches `legal_moves`.
        """
        rows: list[np.ndarray] = []
        for m in legal_moves:
            try:
                succ = board.apply_move(m)
                rows.append(
                    np.asarray(board_to_features(succ, succ.turn), dtype=np.float32)
                )
            except Exception:
                rows.append(np.zeros(self.input_dim, dtype=np.float32))
        if not rows:
            return np.zeros((0, self.input_dim), dtype=np.float32)
        return np.stack(rows)

    # ── Public API (matches retrain_v2_plan.md Step 3a signature) ───────────

    def rank(self, board: BoardState, legal_moves: list[dict]) -> list[float]:
        """Return one scalar per move; larger = more human-likely.

        Values are unbounded (ranking loss trains only relative order).
        """
        if not legal_moves:
            return []
        feats = self._successor_features(board, legal_moves)
        return [float(s) for s in self._score_batch(feats)]

    def probs(self, board: BoardState, legal_moves: list[dict]) -> np.ndarray:
        """Softmax over rank() outputs — for sampling in human-play mode."""
        if not legal_moves:
            return np.zeros(0, dtype=np.float32)
        feats  = self._successor_features(board, legal_moves)
        scores = self._score_batch(feats) / self.temperature
        # Numerical stability: subtract max before exp.
        scores -= scores.max(initial=0.0)
        exp_s = np.exp(scores).astype(np.float32)
        total = float(exp_s.sum())
        if total <= 0.0 or not np.isfinite(total):
            # Fall back to uniform distribution if the score head is degenerate.
            return np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        return exp_s / total


def try_load(
    npz_path: str | Path, temperature: float = 1.0
) -> HumanPrefAdvisor | None:
    """Return a HumanPrefAdvisor if the file exists + loads cleanly, else None.

    Used by opt-in callers (GameAI, bench scripts) so a missing HumanPrefNet
    is a graceful no-op rather than a crash.
    """
    path = Path(npz_path)
    if not path.exists():
        return None
    try:
        return HumanPrefAdvisor(path, temperature=temperature)
    except Exception:
        return None
