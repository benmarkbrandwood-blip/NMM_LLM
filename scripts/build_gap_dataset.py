#!/usr/bin/env python3
"""
scripts/build_gap_dataset.py — Build gap_net training data using sentinel + heuristics.

Three balanced categories (equal sample count):
  LOSING    — human played a Malom-losing move (malom_wdl_after='W', ABS(dtw)<15),
               AND a better move existed. These are genuine blunders.
  WINNING   — human only played Malom-winning moves (malom_wdl_after='L').
               Represents good play that the AI should not "exploit".
  NEUTRAL   — everything else (draw moves, long-horizon losses, no Malom data).

Training target per position:
  gap = best_composite_quality - played_composite_quality   ∈ [0, 1]
  y   = 2 * gap - 1                                         ∈ [-1, 1] (tanh range)
  where composite_quality = SENTINEL_WEIGHT * sentinel_q + (1-SENTINEL_WEIGHT) * heuristic_q_norm

Deduplication: each state_key appears once only — prevents opening positions from
               dominating the dataset.

Fallback: if LOSING category is underpopulated, generate synthetic games using
          GameAI at difficulty 5 with 80% value_net blend (human-like play).

Usage:
    .venv/bin/python scripts/build_gap_dataset.py [options]
    .venv/bin/python scripts/build_gap_dataset.py --samples-per-category 20000
    .venv/bin/python scripts/build_gap_dataset.py --dtw-threshold 30
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import sqlite3

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves
from ai.value_net import board_to_features, _INPUT_DIM
from ai.heuristics import evaluate_v2
from learned_ai.data.malom_label_provenance import (
    require_current_human_db_malom_labels,
)

SENTINEL_WEIGHT = 0.6    # fraction of composite quality from sentinel (rest from heuristics)
MAX_SYNTHETIC_GAMES = 200  # games to self-play if LOSING category under-represented


# ── HumanPrefNet loader (numpy-only, matches train_human_pref_net.py .npz layout) ──

class HumanPrefLoader:
    """Load a HumanPrefNet .npz written by tools/train_human_pref_net.py and
    run its forward pass in pure numpy.  Used to compute per-move ranking
    scores so we can derive an hp_disagreement label for each position.
    """
    def __init__(self, npz_path: Path):
        data = np.load(str(npz_path))
        n_layers = int(data["layer_count"][0]) if "layer_count" in data.files else 0
        if n_layers <= 0:
            # Fall back: count w{i} arrays
            n_layers = sum(1 for k in data.files if k.startswith("w"))
        self.layers = [(data[f"w{i}"].astype(np.float32),
                        data[f"b{i}"].astype(np.float32))
                       for i in range(n_layers)]
        self.input_dim = int(data["input_dim"][0]) if "input_dim" in data.files else self.layers[0][0].shape[1]
        if self.layers[0][0].shape[1] != _INPUT_DIM:
            raise ValueError(
                f"HumanPrefNet input_dim {self.layers[0][0].shape[1]} does not "
                f"match board_to_features dim {_INPUT_DIM}"
            )

    def score_batch(self, feats: np.ndarray) -> np.ndarray:
        """Return one score per row of `feats` (shape (N, input_dim))."""
        x = feats.astype(np.float32)
        n = len(self.layers)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < n - 1:
                x = np.maximum(x, 0.0, out=x)
        return x.squeeze(-1)


# ── Board reconstruction ──────────────────────────────────────────────────────

def _board_from_state_key(state_key: str) -> BoardState | None:
    """Reconstruct a BoardState from a canonical state_key string."""
    parts = state_key.split('|')
    if len(parts) != 7:
        return None
    board_str, turn, phase, placed_w, placed_b, on_w, on_b = parts
    if len(board_str) != 24:
        return None
    try:
        positions = {pos: (board_str[i] if board_str[i] != '.' else '')
                     for i, pos in enumerate(POSITIONS)}
        return BoardState(
            positions=positions,
            turn=turn,
            pieces_on_board={'W': int(on_w), 'B': int(on_b)},
            pieces_placed={'W': int(placed_w), 'B': int(placed_b)},
            pieces_captured={
                'W': max(0, int(placed_b) - int(on_b)),
                'B': max(0, int(placed_w) - int(on_w)),
            },
        )
    except Exception:
        return None


# ── Notation parsing ──────────────────────────────────────────────────────────

def _parse_notation(notation: str) -> dict | None:
    """Parse a DB notation string to a move dict: {from, to, capture}."""
    n = notation.replace('×', 'x').replace('✕', 'x').strip()
    if '-' in n:
        # Movement: 'e3-d3' or 'e3-d3xb4'
        parts = n.split('-', 1)
        from_sq = parts[0].strip()
        rest = parts[1].strip()
        if 'x' in rest:
            to_sq, cap_sq = rest.split('x', 1)
        else:
            to_sq, cap_sq = rest, None
        return {'from': from_sq.strip(), 'to': to_sq.strip(),
                'capture': cap_sq.strip() if cap_sq else None}
    elif 'x' in n:
        # Placement with capture: 'c4xd2'
        to_sq, cap_sq = n.split('x', 1)
        return {'from': None, 'to': to_sq.strip(), 'capture': cap_sq.strip()}
    else:
        # Simple placement: 'c4'
        return {'from': None, 'to': n, 'capture': None}


def _find_move_in_legal(parsed: dict | None, legal: list[dict]) -> dict | None:
    """Match a parsed notation dict to the actual legal move dict."""
    if parsed is None:
        return None
    for m in legal:
        if (m.get('to') == parsed.get('to') and
                m.get('from') == parsed.get('from') and
                m.get('capture') == parsed.get('capture')):
            return m
    # Fallback: match without capture (move may have a different capture choice)
    for m in legal:
        if m.get('to') == parsed.get('to') and m.get('from') == parsed.get('from'):
            return m
    return None


# ── Composite quality scoring ─────────────────────────────────────────────────

def _score_moves(board: BoardState, legal_moves: list[dict],
                 sentinel_advisor) -> dict[tuple, float]:
    """Return {(from, to, capture): composite_quality} for all legal moves.

    composite_quality ∈ [0, 1]:
      SENTINEL_WEIGHT * sentinel_q + (1-SENTINEL_WEIGHT) * normalized_heuristic
    """
    if not legal_moves:
        return {}

    # Heuristic scores for each move (eval of successor from current player's view)
    color = board.turn
    h_scores = []
    for m in legal_moves:
        try:
            succ = board.apply_move(m)
            h_scores.append(float(evaluate_v2(succ, color)))
        except Exception:
            h_scores.append(0.0)

    # Minmax-normalise heuristic scores to [0, 1] within this position
    h_min, h_max = min(h_scores), max(h_scores)
    span = h_max - h_min
    if span < 1e-6:
        h_norms = [0.5] * len(legal_moves)
    else:
        h_norms = [(h - h_min) / span for h in h_scores]

    # Sentinel scores for each move
    s_scores = [0.5] * len(legal_moves)
    if sentinel_advisor is not None:
        try:
            advice = sentinel_advisor.advise(board, legal_moves, board.turn, played_move_idx=0)
            if advice is not None and len(advice.move_scores) == len(legal_moves):
                s_scores = list(advice.move_scores)
        except Exception:
            pass

    composite = {}
    for i, m in enumerate(legal_moves):
        key = (m.get('from'), m.get('to'), m.get('capture'))
        composite[key] = SENTINEL_WEIGHT * s_scores[i] + (1 - SENTINEL_WEIGHT) * h_norms[i]
    return composite


def _compute_gap(board: BoardState, played_moves_freq: dict[str, int],
                 composite: dict[tuple, float]) -> float | None:
    """Compute gap = best_quality - freq_weighted_played_quality.

    played_moves_freq: {notation_string: total_plays_by_humans}
    composite: {(from, to, capture): quality}
    """
    if not composite:
        return None

    best_quality = max(composite.values())
    legal_moves = list(get_all_legal_moves(board))

    # Build notation → quality mapping for played moves
    played_q_weighted = 0.0
    played_total = 0
    for notation, freq in played_moves_freq.items():
        parsed = _parse_notation(notation)
        if parsed is None:
            continue
        matched = _find_move_in_legal(parsed, legal_moves)
        if matched is None:
            continue
        key = (matched.get('from'), matched.get('to'), matched.get('capture'))
        q = composite.get(key, 0.5)
        played_q_weighted += q * freq
        played_total += freq

    if played_total == 0:
        return None

    played_quality = played_q_weighted / played_total
    return best_quality - played_quality


# ── Category queries ──────────────────────────────────────────────────────────

def _query_categories(conn: sqlite3.Connection, dtw_threshold: int,
                      n: int) -> tuple[list[str], list[str], list[str]]:
    """Return (losing_keys, winning_keys, neutral_keys) state_key lists, each ≤ n."""

    # LOSING: human played a W move with ABS(dtw)<threshold (losing trajectory move)
    # No requirement that an L move also exists — the sentinel gap vs all legal moves
    # tells us how well they navigated, even in an already-lost position.
    losing = [row[0] for row in conn.execute(f"""
        SELECT DISTINCT m.state_key FROM moves m
        WHERE m.malom_wdl_after = 'W'
          AND m.malom_dtw_after IS NOT NULL
          AND ABS(m.malom_dtw_after) < {dtw_threshold}
        LIMIT {n * 3}
    """).fetchall()]

    winning = [row[0] for row in conn.execute(f"""
        SELECT DISTINCT m.state_key FROM moves m
        WHERE m.malom_wdl_after = 'L'
          AND NOT EXISTS (
              SELECT 1 FROM moves m2
              WHERE m2.state_key = m.state_key
                AND m2.malom_wdl_after = 'W'
                AND m2.malom_dtw_after IS NOT NULL
                AND ABS(m2.malom_dtw_after) < {dtw_threshold}
          )
        LIMIT {n * 2}
    """).fetchall()]

    losing_set = set(losing)
    winning_set = set(winning)
    neutral = [row[0] for row in conn.execute(f"""
        SELECT DISTINCT state_key FROM positions
        WHERE state_key NOT IN (
            SELECT DISTINCT state_key FROM moves
            WHERE malom_wdl_after = 'W'
              AND malom_dtw_after IS NOT NULL
              AND ABS(malom_dtw_after) < {dtw_threshold}
        )
        LIMIT {n * 2}
    """).fetchall()]
    neutral = [k for k in neutral if k not in winning_set]

    return losing, winning, neutral


def _fetch_played_moves(conn: sqlite3.Connection,
                        state_keys: list[str]) -> dict[str, dict[str, int]]:
    """Return {state_key: {notation: total}} for all given state_keys."""
    if not state_keys:
        return {}
    placeholders = ','.join('?' * len(state_keys))
    rows = conn.execute(
        f"SELECT state_key, notation, total FROM moves WHERE state_key IN ({placeholders})",
        state_keys
    ).fetchall()
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for sk, notation, total in rows:
        result[sk][notation] = total
    return dict(result)


# ── Synthetic game generation (fallback) ─────────────────────────────────────

def _generate_synthetic_positions(n_games: int, value_net_path: Path) -> list[BoardState]:
    """Generate board positions via AI self-play (diff=5, 80% VN blend)."""
    from ai.game_ai import GameAI
    from ai.heuristics import HeuristicWeights
    from ai.value_net import ValueNet
    from game.rules import is_terminal

    vn = ValueNet.load_if_exists(value_net_path)
    if vn is None:
        print("  No value_net found — skipping synthetic generation")
        return []

    w = HeuristicWeights()
    w.value_net_blend = 80

    positions = []
    for g in range(n_games):
        board = BoardState.new_game()
        ai_w = GameAI(color='W', difficulty=5, weights=w, value_net=vn)
        ai_b = GameAI(color='B', difficulty=5, weights=w, value_net=vn)
        for _ in range(80):
            if is_terminal(board):
                break
            ai = ai_w if board.turn == 'W' else ai_b
            move = ai.choose_move(board)
            if move is None:
                break
            positions.append(board)
            board = board.apply_move(move)
        if g % 20 == 0:
            print(f"    synthetic game {g+1}/{n_games}, {len(positions)} positions so far")
    return positions


# ── Main dataset builder ──────────────────────────────────────────────────────

def build_dataset(db_path: Path, sentinel_path: Path, value_net_path: Path,
                  n_per_category: int, dtw_threshold: int,
                  human_pref_path: Path | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conn = sqlite3.connect(str(db_path))
    try:
        require_current_human_db_malom_labels(conn, db_path)
    except Exception:
        conn.close()
        raise

    # Load sentinel
    sentinel_advisor = None
    if sentinel_path.exists():
        try:
            from learned_ai.sentinel.infer import SentinelAdvisor
            sentinel_advisor = SentinelAdvisor(checkpoint_path=str(sentinel_path))
            # Trigger lazy load
            board_tmp = BoardState.new_game()
            _dummy = [{'from': None, 'to': 'a1', 'capture': None}]
            sentinel_advisor.advise(board_tmp, _dummy, board_tmp.turn, played_move_idx=0)
            print("Sentinel loaded from", sentinel_path)
        except Exception as e:
            print(f"Sentinel load failed ({e}) — using heuristics only")
            sentinel_advisor = None
    else:
        print(f"Sentinel checkpoint not found at {sentinel_path} — using heuristics only")

    # Optional HumanPrefNet loader (Step 4 label enrichment).
    hp_loader: HumanPrefLoader | None = None
    if human_pref_path is not None and human_pref_path.exists():
        try:
            hp_loader = HumanPrefLoader(human_pref_path)
            print(f"HumanPrefNet loaded from {human_pref_path}")
        except Exception as e:
            print(f"HumanPrefNet load failed ({e}) — hp_disagreement label will be NaN")
    elif human_pref_path is not None:
        print(f"HumanPrefNet path {human_pref_path} does not exist — hp_disagreement will be NaN")

    rng = np.random.default_rng(42)

    print("Querying categories...")
    losing_keys, winning_keys, neutral_keys = _query_categories(conn, dtw_threshold, n_per_category)
    print(f"  Losing: {len(losing_keys)}, Winning: {len(winning_keys)}, Neutral: {len(neutral_keys)}")

    # Sample down to n_per_category each
    def _sample(keys, n):
        if len(keys) <= n:
            return keys
        idx = rng.choice(len(keys), size=n, replace=False)
        return [keys[i] for i in idx]

    n_target = min(n_per_category, len(losing_keys))
    losing_keys  = _sample(losing_keys, n_target)
    winning_keys = _sample(winning_keys, n_target)
    neutral_keys = _sample(neutral_keys, n_target)
    print(f"  Sampled: {len(losing_keys)} per category = {3*len(losing_keys)} total")

    all_keys = losing_keys + winning_keys + neutral_keys
    played_moves_map = _fetch_played_moves(conn, all_keys)
    conn.close()

    # Score each position
    X_list, y_list = [], []
    y_hp_list: list[float] = []   # per-plan: |malom_optimal_q - malom_q_of_hp_top|
    skipped = 0
    t0 = time.time()

    for i, state_key in enumerate(all_keys):
        board = _board_from_state_key(state_key)
        if board is None:
            skipped += 1
            continue

        legal = list(get_all_legal_moves(board))
        if not legal:
            skipped += 1
            continue

        composite = _score_moves(board, legal, sentinel_advisor)
        played_freq = played_moves_map.get(state_key, {})
        gap = _compute_gap(board, played_freq, composite)
        if gap is None:
            skipped += 1
            continue

        try:
            feats = board_to_features(board, board.turn)
        except Exception:
            skipped += 1
            continue

        # Auxiliary target: hp_disagreement.  For each legal move we already
        # have composite[key].  Score each legal successor via the HumanPrefNet
        # (numpy forward pass), find the HP-top move, and take the malom
        # composite gap between that and the malom-top move.
        hp_disagreement = float("nan")
        if hp_loader is not None and legal:
            try:
                succ_feats = []
                keys       = []
                for m in legal:
                    succ_b = board.apply_move(m)
                    succ_feats.append(board_to_features(succ_b, succ_b.turn))
                    keys.append((m.get("from"), m.get("to"), m.get("capture")))
                succ_feats_arr = np.stack(succ_feats).astype(np.float32)
                hp_scores      = hp_loader.score_batch(succ_feats_arr)
                hp_top_idx     = int(np.argmax(hp_scores))
                # Composite mapping was keyed by move-tuple; pick out per-move q.
                q_by_key       = composite
                malom_qs       = np.array([q_by_key.get(k, 0.5) for k in keys], dtype=np.float32)
                malom_top_q    = float(malom_qs.max())
                hp_top_q       = float(malom_qs[hp_top_idx])
                # Positive when HP prefers a worse move; zero when they agree.
                hp_disagreement = malom_top_q - hp_top_q
            except Exception:
                hp_disagreement = float("nan")

        y_scaled = float(2.0 * gap - 1.0)
        X_list.append(feats)
        y_list.append(y_scaled)
        y_hp_list.append(hp_disagreement)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(all_keys) - i - 1) / max(rate, 0.01)
            print(f"  {i+1}/{len(all_keys)}  rate={rate:.0f}/s  eta={remaining:.0f}s  "
                  f"skipped={skipped}")

    print(f"  Finished: {len(X_list)} samples, {skipped} skipped")

    # Fallback: supplement LOSING category with synthetic positions if under target
    if len(losing_keys) < n_target * 0.5 and value_net_path.exists():
        print(f"LOSING category under-represented ({len(losing_keys)} < {n_target//2}). "
              f"Generating synthetic positions...")
        syn_boards = _generate_synthetic_positions(MAX_SYNTHETIC_GAMES, value_net_path)
        print(f"  Generated {len(syn_boards)} synthetic positions")
        for board in syn_boards:
            legal = list(get_all_legal_moves(board))
            if not legal:
                continue
            composite = _score_moves(board, legal, sentinel_advisor)
            if not composite:
                continue
            best_q = max(composite.values())
            avg_q = sum(composite.values()) / len(composite)
            gap = best_q - avg_q  # no human play data → use best vs average
            try:
                feats = board_to_features(board, board.turn)
            except Exception:
                continue
            X_list.append(feats)
            y_list.append(float(2.0 * gap - 1.0))
            # Synthetic positions have no HP disagreement signal — mark NaN so
            # downstream training can mask/weight them.
            y_hp_list.append(float("nan"))

    X    = np.stack(X_list).astype(np.float32)
    y    = np.array(y_list,    dtype=np.float32)
    y_hp = np.array(y_hp_list, dtype=np.float32)

    n_blunder = int(np.sum(y > -0.5))
    print(f"Dataset: {len(X)} samples, blunder zones (gap>0.25): {n_blunder} ({100*n_blunder/len(y):.1f}%)")
    print(f"y stats: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")
    hp_valid_mask = ~np.isnan(y_hp)
    n_hp_valid = int(hp_valid_mask.sum())
    if n_hp_valid:
        hp_valid = y_hp[hp_valid_mask]
        print(f"y_hp: {n_hp_valid} valid / {len(y_hp)} total  "
              f"min={hp_valid.min():.3f} max={hp_valid.max():.3f} mean={hp_valid.mean():.3f}")
    else:
        print("y_hp: no valid samples (HumanPrefNet not loaded).")
    return X, y, y_hp


def main():
    parser = argparse.ArgumentParser(description="Build gap_net training dataset")
    parser.add_argument("--db",  default="data/human_db.sqlite",
                        help="Human DB path")
    parser.add_argument("--sentinel", default="learned_ai/sentinel/checkpoints/best.pt",
                        help="Sentinel checkpoint path (or use --sentinel-ckpt as an alias)")
    parser.add_argument("--sentinel-ckpt", default=None,
                        help="Alias for --sentinel; matches retrain_v2_plan.md wording")
    parser.add_argument("--human-pref-ckpt", default=None,
                        help="HumanPrefNet .npz for auxiliary hp_disagreement label "
                             "(Step 4 label enrichment; leave unset to skip)")
    parser.add_argument("--value-net", default="data/value_net.npz",
                        help="Value net path (for synthetic fallback)")
    parser.add_argument("--out", default="data/gap_net_training.npz",
                        help="Output dataset path")
    parser.add_argument("--samples-per-category", type=int, default=15000,
                        help="Target samples per category (losing/winning/neutral)")
    parser.add_argument("--dtw-threshold", type=int, default=15,
                        help="ABS(malom_dtw) < threshold to qualify as decisive loss")
    args = parser.parse_args()

    db_path       = _ROOT / args.db
    sentinel_path = _ROOT / (args.sentinel_ckpt or args.sentinel)
    vn_path       = _ROOT / args.value_net
    out_path      = _ROOT / args.out
    hp_path       = (_ROOT / args.human_pref_ckpt) if args.human_pref_ckpt else None

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    X, y, y_hp = build_dataset(db_path, sentinel_path, vn_path,
                               args.samples_per_category, args.dtw_threshold,
                               human_pref_path=hp_path)
    np.savez(str(out_path), X=X, y=y, y_hp=y_hp)
    print(f"Saved {len(X)} samples → {out_path}")


if __name__ == "__main__":
    main()
