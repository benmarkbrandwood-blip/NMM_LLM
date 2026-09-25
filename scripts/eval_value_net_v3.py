#!/usr/bin/env python3
"""scripts/eval_value_net_v3.py — Evaluate ValueNet v3 vs v2 two ways:

1. STATIC TEST-SET METRICS
   Runs both nets over the held-out test slice (buckets 0-4, never seen during
   training) from human_db_candidate_new.sqlite and reports MSE + sign accuracy
   per phase and overall.

2. HEAD-TO-HEAD GAME PLAY
   Runs N games of heuristic engine + v3 net vs heuristic engine + v2 net at a
   given difficulty and VN blend percentage.  Reports win/draw/loss counts and
   win rate.

Usage
-----
    # Metrics only (fast):
    .venv/bin/python scripts/eval_value_net_v3.py --mode metrics

    # Games only:
    .venv/bin/python scripts/eval_value_net_v3.py --mode games --n-games 100

    # Both:
    .venv/bin/python scripts/eval_value_net_v3.py --mode both --n-games 200

    # Custom paths:
    .venv/bin/python scripts/eval_value_net_v3.py \\
        --v3-base data/value_net_v3 \\
        --v2      data/value_net_v2.npz \\
        --db      data/human_db_candidate_new.sqlite \\
        --n-games 200 --difficulty 5 --vn-blend 60
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState, POSITIONS
from game.rules import get_all_legal_moves, is_terminal
from ai.value_net import ValueNet, PhaseValueNet, board_to_features, _INPUT_DIM
from ai.heuristics import HeuristicWeights, DEFAULT_WEIGHTS
from ai.game_ai import GameAI


# ── Board reconstruction (shared with train_value_net_v3) ─────────────────────

def _board_and_phase_from_state_key(state_key: str) -> tuple[BoardState, str] | None:
    parts = state_key.split("|")
    if len(parts) != 7:
        return None
    canon24, turn, phase, placed_w_s, placed_b_s, on_w_s, on_b_s = parts
    if len(canon24) != len(POSITIONS) or phase not in ("place", "move", "fly"):
        return None
    try:
        placed_w, placed_b = int(placed_w_s), int(placed_b_s)
        on_w,     on_b     = int(on_w_s),     int(on_b_s)
    except ValueError:
        return None
    positions = {pos: ("" if c == "." else c) for pos, c in zip(POSITIONS, canon24)}
    board = BoardState(
        positions=positions, turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": max(0, placed_b - on_b), "B": max(0, placed_w - on_w)},
    )
    return board, phase


def dtw_target(wdl: str, dtw: int | None, dtw_scale: float = 7.0) -> float | None:
    if wdl == "D":
        return 0.0
    if wdl not in ("W", "L") or dtw is None or dtw <= 0:
        return None
    mag = math.tanh(dtw_scale / dtw)
    return mag if wdl == "W" else -mag


_PHASES = ("place", "move", "fly")


# ── 1. Static test-set metrics ─────────────────────────────────────────────────

def run_metrics(
    v3_base: Path,
    v2_path: Path,
    db_path: Path,
    dtw_scale: float,
) -> None:
    from learned_ai.data.human_db_split import three_way_split

    print("[eval] Loading nets …")
    v3 = PhaseValueNet.load_if_exists(v3_base)
    if v3 is None:
        raise SystemExit(f"ValueNet v3 not found at {v3_base}_{{place,move,fly}}.npz")
    v2 = ValueNet.load_if_exists(v2_path)
    if v2 is None:
        raise SystemExit(f"ValueNet v2 not found at {v2_path}")

    print(f"[eval] Scanning test slice from {db_path} …")
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    phase_rows: dict[str, tuple[list, list]] = {p: ([], []) for p in _PHASES}
    n_seen = n_test = n_skip = 0
    conn = sqlite3.connect(str(db_path))
    for sk, wdl, dtw in conn.execute(
        "SELECT state_key, malom_wdl, malom_dtw FROM positions WHERE malom_wdl IS NOT NULL"
    ):
        n_seen += 1
        if three_way_split(sk) != "test":
            n_skip += 1
            continue
        n_test += 1
        label = dtw_target(wdl, dtw, dtw_scale)
        if label is None:
            continue
        res = _board_and_phase_from_state_key(sk)
        if res is None:
            continue
        board, phase = res
        try:
            feat = board_to_features(board, board.turn).astype(np.float32)
        except Exception:
            continue
        phase_rows[phase][0].append(feat)
        phase_rows[phase][1].append(label)
    conn.close()

    print(f"[eval] Seen={n_seen:,}  test={n_test:,}")
    print()
    print(f"{'Phase':>6}  {'N':>8}  {'v2 MSE':>8}  {'v3 MSE':>8}  "
          f"{'v2 sign%':>9}  {'v3 sign%':>9}  {'v3 W-mean':>10}  {'v3 D-mean':>10}  {'v3 L-mean':>10}")
    print("-" * 100)

    all_v2_preds: list[np.ndarray] = []
    all_v3_preds: list[np.ndarray] = []
    all_y:        list[np.ndarray] = []

    for phase in _PHASES:
        feats, labels = phase_rows[phase]
        if not feats:
            print(f"  {phase:>6}  (no test data)")
            continue
        X  = np.stack(feats)
        y  = np.array(labels, dtype=np.float32)

        p2 = v2.predict_batch(X)
        p3 = v3.predict(None, None) if False else None  # dispatch via phase net

        # Predict with phase-specific v3 net directly
        phase_net = v3._nets[phase]
        p3 = phase_net.predict_batch(X)

        mse2 = float(np.mean((p2 - y) ** 2))
        mse3 = float(np.mean((p3 - y) ** 2))
        nondraw = y != 0.0
        def _sacc(preds: np.ndarray) -> str:
            if not nondraw.any():
                return "   n/a"
            return f"{100*float(np.mean(np.sign(preds[nondraw]) == np.sign(y[nondraw]))):.1f}%"

        wins  = y > 0; draws_ = y == 0; losses_ = y < 0
        w3 = f"{p3[wins].mean():.3f}"   if wins.any()    else "  n/a"
        d3 = f"{p3[draws_].mean():.3f}" if draws_.any()  else "  n/a"
        l3 = f"{p3[losses_].mean():.3f}"if losses_.any() else "  n/a"

        print(f"  {phase:>6}  {len(y):>8,}  {mse2:>8.5f}  {mse3:>8.5f}  "
              f"{_sacc(p2):>9}  {_sacc(p3):>9}  {w3:>10}  {d3:>10}  {l3:>10}")

        all_v2_preds.append(p2); all_v3_preds.append(p3); all_y.append(y)

    if all_y:
        p2a = np.concatenate(all_v2_preds)
        p3a = np.concatenate(all_v3_preds)
        ya  = np.concatenate(all_y)
        mse2a = float(np.mean((p2a - ya) ** 2))
        mse3a = float(np.mean((p3a - ya) ** 2))
        nd = ya != 0.0
        sa2 = f"{100*float(np.mean(np.sign(p2a[nd]) == np.sign(ya[nd]))):.1f}%" if nd.any() else "n/a"
        sa3 = f"{100*float(np.mean(np.sign(p3a[nd]) == np.sign(ya[nd]))):.1f}%" if nd.any() else "n/a"
        print("-" * 100)
        print(f"  {'total':>6}  {len(ya):>8,}  {mse2a:>8.5f}  {mse3a:>8.5f}  "
              f"{sa2:>9}  {sa3:>9}")
    print()


# ── 2. Head-to-head game play ─────────────────────────────────────────────────

def _make_ai(color: str, vnet, difficulty: int, vn_blend: int) -> GameAI:
    weights = HeuristicWeights(**{
        **vars(DEFAULT_WEIGHTS),
        "value_net_blend": vn_blend,
    })
    return GameAI(
        color=color,
        difficulty=difficulty,
        value_net=vnet,
        weights=weights,
    )


def _play_game(
    ai_w: GameAI, ai_b: GameAI, max_plies: int = 400
) -> str | None:
    """Play one game; return 'W', 'B', or None (draw/timeout)."""
    board = BoardState.new_game()
    for _ in range(max_plies):
        ai = ai_w if board.turn == "W" else ai_b
        move = ai.choose_move(board)
        if move is None:
            return "B" if board.turn == "W" else "W"
        board = board.apply_move(move)
        terminal, winner = is_terminal(board)
        if terminal:
            return winner  # None = draw by rule
    return None


def run_games(
    v3_base: Path,
    v2_path: Path,
    n_games: int,
    difficulty: int,
    vn_blend: int,
) -> None:
    print("[eval] Loading nets for game play …")
    v3 = PhaseValueNet.load_if_exists(v3_base)
    if v3 is None:
        raise SystemExit(f"ValueNet v3 not found at {v3_base}_{{phase}}.npz")
    v2 = ValueNet.load_if_exists(v2_path)
    if v2 is None:
        raise SystemExit(f"ValueNet v2 not found at {v2_path}")

    print(f"[eval] Head-to-head: v3 vs v2  n={n_games}  difficulty={difficulty}  vn_blend={vn_blend}%")
    print(f"       First half: v3=White v2=Black  |  Second half: v3=Black v2=White")
    print()

    results: dict[str, int] = {"v3": 0, "v2": 0, "draw": 0}
    half = n_games // 2
    t0 = time.time()

    for i in range(n_games):
        # Alternate sides to remove first-mover bias
        if i < half:
            ai_v3 = _make_ai("W", v3, difficulty, vn_blend)
            ai_v2 = _make_ai("B", v2, difficulty, vn_blend)
            v3_color = "W"
        else:
            ai_v3 = _make_ai("B", v3, difficulty, vn_blend)
            ai_v2 = _make_ai("W", v2, difficulty, vn_blend)
            v3_color = "B"

        winner = _play_game(ai_v3 if v3_color == "W" else ai_v2,
                            ai_v2 if v3_color == "W" else ai_v3)
        if winner == v3_color:
            results["v3"] += 1
        elif winner is not None:
            results["v2"] += 1
        else:
            results["draw"] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n_games - i - 1) / rate
            print(f"  game {i+1:>4}/{n_games}  "
                  f"v3={results['v3']}  v2={results['v2']}  draw={results['draw']}  "
                  f"[{rate:.1f} g/s  ETA {eta:.0f}s]")

    total = n_games
    v3_pct  = 100 * results["v3"]  / total
    v2_pct  = 100 * results["v2"]  / total
    draw_pct = 100 * results["draw"] / total
    print()
    print(f"[eval] ── RESULT ({n_games} games, difficulty={difficulty}, vn_blend={vn_blend}%) ──")
    print(f"  v3 wins  : {results['v3']:>4}  ({v3_pct:.1f}%)")
    print(f"  v2 wins  : {results['v2']:>4}  ({v2_pct:.1f}%)")
    print(f"  draws    : {results['draw']:>4}  ({draw_pct:.1f}%)")
    non_draw = results["v3"] + results["v2"]
    if non_draw > 0:
        decisive_v3 = 100 * results["v3"] / non_draw
        print(f"  decisive v3 win rate: {decisive_v3:.1f}%  (excluding draws)")
    elapsed = time.time() - t0
    print(f"  total time: {elapsed:.0f}s  ({elapsed/n_games:.1f}s/game)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate ValueNet v3 vs v2.")
    p.add_argument("--mode",       choices=["metrics", "games", "both"], default="both")
    p.add_argument("--v3-base",    type=Path, default=Path("data/value_net_v3"),
                   help="Base path for v3 phase nets (without _phase.npz suffix).")
    p.add_argument("--v2",         type=Path, default=Path("data/value_net_v2.npz"))
    p.add_argument("--db",         type=Path, default=Path("data/human_db_candidate_new.sqlite"))
    p.add_argument("--dtw-scale",  type=float, default=7.0)
    p.add_argument("--n-games",    type=int,  default=100)
    p.add_argument("--difficulty", type=int,  default=5,
                   help="AI search depth/difficulty for both sides (default 5).")
    p.add_argument("--vn-blend",   type=int,  default=60,
                   help="Value-net blend %% applied to both AIs (default 60).")
    args = p.parse_args()

    if args.mode in ("metrics", "both"):
        run_metrics(args.v3_base, args.v2, args.db, args.dtw_scale)

    if args.mode in ("games", "both"):
        run_games(args.v3_base, args.v2, args.n_games, args.difficulty, args.vn_blend)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
