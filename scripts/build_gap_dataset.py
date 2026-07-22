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


class RequiredSentinelError(RuntimeError):
    """The GapNet teacher cannot use its required Sentinel signal."""


def _required_sentinel_scores(advice, expected_count: int) -> list[float]:
    """Validate one required Sentinel response without a neutral fallback."""
    if advice is None:
        raise RequiredSentinelError("required Sentinel returned no advice")

    raw_scores = getattr(advice, "move_scores", None)
    if raw_scores is None:
        raise RequiredSentinelError(
            "required Sentinel advice has no move_scores"
        )

    try:
        scores = [float(score) for score in raw_scores]
    except (TypeError, ValueError) as exc:
        raise RequiredSentinelError(
            "required Sentinel returned non-numeric move scores"
        ) from exc

    if len(scores) != expected_count:
        raise RequiredSentinelError(
            "required Sentinel returned "
            f"{len(scores)} scores for {expected_count} legal moves"
        )
    if not np.all(np.isfinite(scores)):
        raise RequiredSentinelError(
            "required Sentinel returned non-finite move scores"
        )
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise RequiredSentinelError(
            "required Sentinel returned a move score outside [0, 1]"
        )
    return scores


def _load_required_sentinel(sentinel_path: Path):
    """Load and probe the Sentinel required by the GapNet label contract."""
    if not sentinel_path.is_file():
        raise FileNotFoundError(
            f"required Sentinel checkpoint not found: {sentinel_path}"
        )

    try:
        from learned_ai.sentinel.infer import SentinelAdvisor

        advisor = SentinelAdvisor(checkpoint_path=str(sentinel_path))
        if not advisor.is_loaded():
            raise RequiredSentinelError(
                "required Sentinel did not report a loaded model"
            )

        probe_board = BoardState.new_game()
        probe_moves = list(get_all_legal_moves(probe_board))[:1]
        if not probe_moves:
            raise RequiredSentinelError(
                "could not construct the required Sentinel load probe"
            )
        advice = advisor.advise(
            probe_board,
            probe_moves,
            probe_board.turn,
            played_move_idx=0,
        )
        _required_sentinel_scores(advice, len(probe_moves))
    except RequiredSentinelError:
        raise
    except Exception as exc:
        raise RequiredSentinelError(
            f"required Sentinel checkpoint is unusable: {sentinel_path}: {exc}"
        ) from exc

    print("Sentinel loaded from", sentinel_path)
    return advisor


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

    if sentinel_advisor is None:
        raise RequiredSentinelError(
            "required Sentinel advisor is absent during GapNet scoring"
        )
    try:
        advice = sentinel_advisor.advise(
            board,
            legal_moves,
            board.turn,
            played_move_idx=0,
        )
    except Exception as exc:
        raise RequiredSentinelError(
            "required Sentinel failed during GapNet scoring"
        ) from exc
    s_scores = _required_sentinel_scores(advice, len(legal_moves))

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
                  n_per_category: int, dtw_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    sentinel_advisor = _load_required_sentinel(sentinel_path)

    conn = sqlite3.connect(str(db_path))
    try:
        require_current_human_db_malom_labels(conn, db_path)
        rng = np.random.default_rng(42)

        print("Querying categories...")
        losing_keys, winning_keys, neutral_keys = _query_categories(
            conn,
            dtw_threshold,
            n_per_category,
        )
        print(
            f"  Losing: {len(losing_keys)}, Winning: {len(winning_keys)}, "
            f"Neutral: {len(neutral_keys)}"
        )

        # Sample down to n_per_category each
        def _sample(keys, n):
            if len(keys) <= n:
                return keys
            idx = rng.choice(len(keys), size=n, replace=False)
            return [keys[i] for i in idx]

        n_target = min(n_per_category, len(losing_keys))
        losing_keys = _sample(losing_keys, n_target)
        winning_keys = _sample(winning_keys, n_target)
        neutral_keys = _sample(neutral_keys, n_target)
        print(
            f"  Sampled: {len(losing_keys)} per category = "
            f"{3 * len(losing_keys)} total"
        )

        all_keys = losing_keys + winning_keys + neutral_keys
        played_moves_map = _fetch_played_moves(conn, all_keys)
    finally:
        conn.close()

    # Score each position
    X_list, y_list = [], []
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

        y_scaled = float(2.0 * gap - 1.0)
        X_list.append(feats)
        y_list.append(y_scaled)

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

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_blunder = int(np.sum(y > -0.5))
    print(f"Dataset: {len(X)} samples, blunder zones (gap>0.25): {n_blunder} ({100*n_blunder/len(y):.1f}%)")
    print(f"y stats: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Build gap_net training dataset")
    parser.add_argument("--db",  default="data/human_db.sqlite",
                        help="Human DB path")
    parser.add_argument("--sentinel", default="learned_ai/sentinel/checkpoints/best.pt",
                        help="Sentinel checkpoint path")
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
    sentinel_path = _ROOT / args.sentinel
    vn_path       = _ROOT / args.value_net
    out_path      = _ROOT / args.out

    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    try:
        X, y = build_dataset(
            db_path,
            sentinel_path,
            vn_path,
            args.samples_per_category,
            args.dtw_threshold,
        )
    except (FileNotFoundError, RequiredSentinelError) as exc:
        parser.error(str(exc))
    np.savez(str(out_path), X=X, y=y)
    print(f"Saved {len(X)} samples → {out_path}")


if __name__ == "__main__":
    main()
