"""ai/human_move_policy_advisor.py — Inference wrapper around HumanMovePolicyNet.

Loads the .npz emitted by `tools/train_human_move_policy_net.py` and exposes
`rank(board, legal_moves, elo_band)` and `probs(board, legal_moves,
elo_band)` — the calibrated per-move human probability distribution
that GapNet v2 will consume in the expected-regret formula:

    expected_regret(pos) = Σ_m  probs(m | pos, band) · malom_regret(m)

Feature contract (locked, tested):
  * Board features are computed from `board.apply_move(m)` per legal
    move, encoded via `board_to_features(succ, original_mover_colour)`
    — i.e. the ORIGINAL MOVER'S perspective, not the successor's turn.
  * The 3-way Elo band one-hot is appended to each row → (legal, 82).
  * Softmax is taken over EVERY legal move at the position (not just
    observed).  Sums to 1.

Pure numpy at inference so callers do not need torch.

Layer layout in the .npz:
    input_dim         → int64 (1,)  == board_feature_dim + n_bands
    board_feature_dim → int64 (1,)  == 79
    n_bands           → int64 (1,)  == 3
    layer_count       → int64 (1,)  == 4
    w{i}, b{i}        → float32 weights (rows-out × rows-in), biases
    provenance_json   → object array, single JSON string
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from game.board import BoardState
from ai.value_net import board_to_features, _INPUT_DIM


# Must match tools/train_human_move_policy_net.py: 0=lower, 1=middle, 2=upper.
_BAND_TO_IDX = {"lower": 0, "middle": 1, "upper": 2}


class HumanMovePolicyAdvisor:
    """Score / rank / sample candidate moves by predicted human likelihood,
    conditioned on Elo band.  Sibling to `HumanPrefAdvisor`."""

    def __init__(self, npz_path: str | Path, temperature: float = 1.0):
        data = np.load(str(npz_path), allow_pickle=True)
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
        board_dim = int(data["board_feature_dim"][0]) if "board_feature_dim" in data.files else _INPUT_DIM
        n_bands   = int(data["n_bands"][0]) if "n_bands" in data.files else 3
        if expected_in != board_dim + n_bands:
            raise ValueError(
                f"HumanMovePolicyNet at {npz_path!s} expects input_dim {expected_in}, "
                f"but board_feature_dim + n_bands = {board_dim + n_bands}."
            )
        if board_dim != _INPUT_DIM:
            raise ValueError(
                f"HumanMovePolicyNet at {npz_path!s} expects board_feature_dim "
                f"{board_dim}, but board_to_features emits {_INPUT_DIM}."
            )
        self.input_dim = expected_in
        self.board_feature_dim = board_dim
        self.n_bands = n_bands
        self.temperature = max(1e-3, float(temperature))

        # Provenance is retained but not required for inference.
        prov_key = "provenance_json"
        if prov_key in data.files:
            try:
                self.provenance = json.loads(str(data[prov_key].item()))
            except Exception:
                self.provenance = None
        else:
            self.provenance = None

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
        """Return (N, board_feature_dim) successor encodings from the
        ORIGINAL MOVER'S perspective — i.e. `board.turn` at call time.

        A move that raises during apply_move yields a zero row so
        `legal_moves`/rows stay aligned.
        """
        mover_color = board.turn
        rows: list[np.ndarray] = []
        for m in legal_moves:
            try:
                succ = board.apply_move(m)
                rows.append(
                    np.asarray(board_to_features(succ, mover_color),
                               dtype=np.float32)
                )
            except Exception:
                rows.append(np.zeros(self.board_feature_dim, dtype=np.float32))
        if not rows:
            return np.zeros((0, self.board_feature_dim), dtype=np.float32)
        return np.stack(rows)

    def _band_row(self, elo_band: str, n_rows: int) -> np.ndarray:
        if elo_band not in _BAND_TO_IDX:
            raise ValueError(
                f"Unknown elo_band {elo_band!r}; expected one of "
                f"{sorted(_BAND_TO_IDX)}"
            )
        idx = _BAND_TO_IDX[elo_band]
        row = np.zeros((n_rows, self.n_bands), dtype=np.float32)
        row[:, idx] = 1.0
        return row

    # ── Public API ────────────────────────────────────────────────────────────

    def rank(
        self, board: BoardState, legal_moves: list[dict], elo_band: str
    ) -> list[float]:
        """Return one scalar per legal move; larger = more human-likely for
        the given `elo_band`.  Values are unbounded — softmax them for
        calibrated probabilities via `probs()`."""
        if not legal_moves:
            return []
        board_feats = self._successor_features(board, legal_moves)
        band_bits   = self._band_row(elo_band, board_feats.shape[0])
        x           = np.concatenate([board_feats, band_bits], axis=1)
        return [float(s) for s in self._score_batch(x)]

    def probs(
        self, board: BoardState, legal_moves: list[dict], elo_band: str
    ) -> np.ndarray:
        """Softmax over every legal move → calibrated
        `p(m | position, elo_band)`.  Sums to 1 across `legal_moves`.

        Falls back to a uniform distribution if the score head is
        degenerate (all inf / nan / equal / etc).  Use `probs_for_display`
        for overlays where a silent fallback is misleading."""
        if not legal_moves:
            return np.zeros(0, dtype=np.float32)
        board_feats = self._successor_features(board, legal_moves)
        band_bits   = self._band_row(elo_band, board_feats.shape[0])
        x           = np.concatenate([board_feats, band_bits], axis=1)
        scores      = self._score_batch(x) / self.temperature
        if not np.isfinite(scores).all():
            return np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        # Numerical stability: subtract max before exp.
        scores -= scores.max(initial=0.0)
        exp_s = np.exp(scores).astype(np.float32)
        total = float(exp_s.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        return exp_s / total

    def probs_for_display(
        self, board: BoardState, legal_moves: list[dict], elo_band: str
    ) -> np.ndarray:
        """Like `probs()` but raises ValueError instead of silently degrading.

        A failed successor encoding or non-finite model output are technical
        failures, not a uniform prior.  Callers that use these probabilities
        for an explanatory overlay should call this method and treat any
        exception as visibly unavailable rather than letting a zero-row or
        uniform distribution masquerade as a real prediction."""
        if not legal_moves:
            return np.zeros(0, dtype=np.float32)
        mover_color = board.turn
        rows: list[np.ndarray] = []
        for m in legal_moves:
            succ = board.apply_move(m)   # raises on failure — intentional, no catch
            rows.append(
                np.asarray(board_to_features(succ, mover_color), dtype=np.float32)
            )
        board_feats = np.stack(rows)
        band_bits   = self._band_row(elo_band, board_feats.shape[0])
        x           = np.concatenate([board_feats, band_bits], axis=1)
        scores      = self._score_batch(x) / self.temperature
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite policy scores — model output unusable for display")
        scores -= scores.max(initial=0.0)
        exp_s = np.exp(scores).astype(np.float32)
        total = float(exp_s.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("Policy score exponents underflowed — output unusable")
        return exp_s / total


def try_load(
    npz_path: str | Path, temperature: float = 1.0
) -> Optional[HumanMovePolicyAdvisor]:
    """Return a HumanMovePolicyAdvisor if the file exists + loads cleanly,
    else None.  Mirrors `ai.human_pref_advisor.try_load` so callers can
    fall back gracefully when the move-policy net isn't available."""
    path = Path(npz_path)
    if not path.exists():
        return None
    try:
        return HumanMovePolicyAdvisor(path, temperature=temperature)
    except Exception:
        return None
