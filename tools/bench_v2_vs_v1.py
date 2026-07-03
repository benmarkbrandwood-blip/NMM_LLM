"""tools/bench_v2_vs_v1.py — V2 (Rust+evaluate_v2) vs V1 (Python+evaluate) match.

Plays N games between:
  V2 — current engine: Rust search with evaluate_v2, EvalScale, FGOP.
  V1 — old engine:     Python negamax with evaluate() (use_v2_heuristics=False).

Both engines receive the same per-move time budget. Colours alternate each game.
Results are printed after each game and summarised at the end. Nothing is saved
to the openings book or training data — read-only benchmark.

Usage:
    .venv/bin/python tools/bench_v2_vs_v1.py [--games 100] [--budget 3.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.board import BoardState
from game.game_engine import GameEngine
from game.rules import get_all_legal_moves
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights


# ── Engine factory ────────────────────────────────────────────────────────────

def make_v2(color: str, budget: float) -> GameAI:
    """Current engine: Rust search, evaluate_v2, EvalScale, FGOP."""
    ai = GameAI(
        color=color,
        difficulty=6,
        override_time_budget=budget,
    )
    ai.use_v2_heuristics = True
    return ai


def make_v1(color: str, budget: float) -> GameAI:
    """Old engine: Python negamax with evaluate(), Rust search bypassed."""
    ai = GameAI(
        color=color,
        difficulty=6,
        override_time_budget=budget,
    )
    ai.use_v2_heuristics = False
    # Bypass Rust search so Python _iterative_deepen + evaluate() is used.
    ai._choose_rust_scored = lambda *a, **kw: None  # type: ignore[method-assign]
    return ai


# ── Game runner ───────────────────────────────────────────────────────────────

MAX_MOVES = 300   # draw after this many half-moves

def play_game(
    white_ai: GameAI,
    black_ai: GameAI,
) -> str | None:
    """Play one game. Returns 'W', 'B', or None (draw/max-moves)."""
    engine = GameEngine(human_color="W")   # human_color ignored; we drive both sides
    move_count = 0

    while not engine.finished and move_count < MAX_MOVES:
        board = engine.board
        ai = white_ai if board.turn == "W" else black_ai
        move = ai.choose_move(board)
        if not move:
            break
        engine.apply_move(move)
        move_count += 1

    return engine.winner   # None = draw or max-moves


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="V2 vs V1 heuristics match")
    ap.add_argument("--games",  type=int,   default=100, help="number of games (default 100)")
    ap.add_argument("--budget", type=float, default=3.0, help="per-move time budget in seconds (default 3)")
    ap.add_argument("--out",    type=str,   default=str(ROOT / "eval_results.json"),
                    help="JSON output file for raw results")
    args = ap.parse_args()

    n_games  = args.games
    budget   = args.budget
    out_path = Path(args.out)

    print(f"V2 vs V1 — {n_games} games, {budget:.1f}s/move budget")
    print(f"V2: Rust search + evaluate_v2 + EvalScale + FGOP")
    print(f"V1: Python negamax + evaluate() (old heuristics)")
    print("-" * 60)

    results: list[dict] = []
    v2_wins = v1_wins = draws = 0
    t_start = time.time()

    for game_idx in range(n_games):
        # Alternate which engine plays White
        v2_is_white = (game_idx % 2 == 0)
        v2_color = "W" if v2_is_white else "B"
        v1_color = "B" if v2_is_white else "W"

        white_ai = make_v2("W", budget) if v2_is_white else make_v1("W", budget)
        black_ai = make_v1("B", budget) if v2_is_white else make_v2("B", budget)

        g_start = time.time()
        winner = play_game(white_ai, black_ai)
        g_elapsed = time.time() - g_start

        if winner == v2_color:
            outcome = "V2"
            v2_wins += 1
        elif winner == v1_color:
            outcome = "V1"
            v1_wins += 1
        else:
            outcome = "draw"
            draws += 1

        total = v2_wins + v1_wins + draws
        elapsed = time.time() - t_start
        eta = (elapsed / total) * (n_games - total) if total else 0

        print(
            f"Game {game_idx+1:>3}/{n_games}  "
            f"V2={'W' if v2_is_white else 'B'}  "
            f"winner={winner or 'draw':<4}  [{outcome}]  "
            f"{g_elapsed:.0f}s  |  "
            f"V2 {v2_wins}-{draws}-{v1_wins} V1  "
            f"({100*v2_wins/total:.0f}%/{100*draws/total:.0f}%/{100*v1_wins/total:.0f}%)  "
            f"ETA {eta/60:.0f}m",
            flush=True,
        )

        results.append({
            "game": game_idx + 1,
            "v2_color": v2_color,
            "winner": winner,
            "outcome": outcome,
            "elapsed_s": round(g_elapsed, 1),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    total = v2_wins + v1_wins + draws
    total_time = time.time() - t_start
    print()
    print("=" * 60)
    print(f"RESULT  V2 {v2_wins} — {draws} draws — {v1_wins} V1  ({n_games} games)")
    print(f"V2 score: {v2_wins + 0.5*draws:.1f}/{n_games}  "
          f"({100*(v2_wins + 0.5*draws)/n_games:.1f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("=" * 60)

    summary = {
        "engine_v2": "Rust search + evaluate_v2 + EvalScale + FGOP",
        "engine_v1": "Python negamax + evaluate()",
        "budget_s":  budget,
        "n_games":   n_games,
        "v2_wins":   v2_wins,
        "v1_wins":   v1_wins,
        "draws":     draws,
        "v2_score":  v2_wins + 0.5 * draws,
        "v2_pct":    round(100 * (v2_wins + 0.5 * draws) / n_games, 1),
        "total_time_min": round(total_time / 60, 1),
        "games": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
