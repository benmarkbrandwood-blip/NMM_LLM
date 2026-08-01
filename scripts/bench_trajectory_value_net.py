"""scripts/bench_trajectory_value_net.py — Round-robin benchmark for trajectory value net.

Compares five configurations:
  Baseline  — pure heuristics, no value net
  TrajVN-10 — heuristics + trajectory value net at 10% blend
  TrajVN-30 — heuristics + trajectory value net at 30% blend
  TrajVN-60 — heuristics + trajectory value net at 60% blend
  GapNet    — heuristics + gap net

Default mode is round-robin (all 10 pairs). Use --matchup A B for a single pair.

Usage
-----
# Full round-robin (10 pairs × --games each)
.venv/bin/python scripts/bench_trajectory_value_net.py --games 20 --difficulty 4

# Single matchup
.venv/bin/python scripts/bench_trajectory_value_net.py --matchup TrajVN-30 Baseline --games 40

# Custom VN path
.venv/bin/python scripts/bench_trajectory_value_net.py --vn-path data/value_net_trajectory.npz

Colours alternate first-mover across games to cancel first-move advantage.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from itertools import combinations
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BoardState
from ai.game_ai import GameAI

_TRAJ_VN_PATH = "data/value_net_trajectory.npz"
_GAP_NET_PATH  = "data/gap_net.npz"


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_value_net(path: str):
    try:
        from ai.value_net import ValueNet
        vn = ValueNet.load(path)
        print(f"  Trajectory value net loaded: {path}")
        return vn
    except Exception as e:
        print(f"  Trajectory value net load failed: {e}")
        return None


def _load_gap_net(path: str = _GAP_NET_PATH):
    try:
        from ai.value_net import ValueNet
        gn = ValueNet.load(path)
        print(f"  Gap net loaded: {path}")
        return gn
    except Exception as e:
        print(f"  Gap net load failed: {e}")
        return None


# ── AI factory ─────────────────────────────────────────────────────────────────

def _make_ai(color: str, difficulty: int, value_net=None, gap_net=None,
             vn_blend: int = 0, time_budget: float | None = None) -> GameAI:
    from ai.heuristics import HeuristicWeights
    weights = HeuristicWeights(value_net_blend=vn_blend) if vn_blend > 0 else None
    return GameAI(color=color, difficulty=difficulty, value_net=value_net,
                  gap_net=gap_net, weights=weights, override_time_budget=time_budget)


# ── Game runner ────────────────────────────────────────────────────────────────

def play_game(white_ai: GameAI, black_ai: GameAI, max_plies: int = 400) -> Optional[str]:
    """Return 'W', 'B', or None (draw/stalemate)."""
    from game.game_engine import GameEngine
    from game.rules import is_terminal

    engine = GameEngine(human_color=None)
    for _ in range(max_plies):
        if engine.winner is not None:
            return engine.winner
        terminal, winner = is_terminal(engine.board)
        if terminal:
            return winner
        color = engine.board.turn
        ai = white_ai if color == "W" else black_ai
        try:
            move = ai.choose_move(engine.board)
        except Exception:
            return None
        if move is None:
            return "B" if color == "W" else "W"
        try:
            engine.apply_move(move)
        except Exception:
            return None
    return None


# ── Matchup runner ─────────────────────────────────────────────────────────────

def run_matchup(
    label_a: str, label_b: str,
    n_games: int, difficulty: int, time_budget: float | None,
    value_net_a=None, gap_net_a=None, vn_blend_a: int = 0,
    value_net_b=None, gap_net_b=None, vn_blend_b: int = 0,
) -> dict:
    """Run n_games between config A and config B, alternating colours."""
    results = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for g in range(n_games):
        if g % 2 == 0:
            w_ai = _make_ai("W", difficulty, value_net_a, gap_net_a, vn_blend_a, time_budget)
            b_ai = _make_ai("B", difficulty, value_net_b, gap_net_b, vn_blend_b, time_budget)
            a_color = "W"
        else:
            w_ai = _make_ai("W", difficulty, value_net_b, gap_net_b, vn_blend_b, time_budget)
            b_ai = _make_ai("B", difficulty, value_net_a, gap_net_a, vn_blend_a, time_budget)
            a_color = "B"

        winner = play_game(w_ai, b_ai)
        if winner is None:
            results["draw"] += 1
        elif winner == a_color:
            results["A"] += 1
        else:
            results["B"] += 1

        elapsed = time.time() - t0
        done = g + 1
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_games - done) / rate if rate > 0 else 0
        print(f"\r  {done}/{n_games}  A:{results['A']}  B:{results['B']}  "
              f"D:{results['draw']}  {rate:.1f}g/s  ETA {eta:.0f}s    ", end="", flush=True)

    print()
    elapsed = time.time() - t0
    a_pct  = 100 * results["A"] / n_games
    b_pct  = 100 * results["B"] / n_games
    d_pct  = 100 * results["draw"] / n_games
    edge   = a_pct - b_pct
    verdict = "A+" if edge > 2 else ("B+" if edge < -2 else "~=")
    print(f"\n{'='*62}")
    print(f"  {n_games} games  ({elapsed:.0f}s)  difficulty={difficulty}")
    print(f"{'='*62}")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"  (each config plays White in half the games, Black in the other half)")
    print(f"{'='*62}")
    print(f"  A wins : {results['A']:4d}  ({a_pct:.1f}%)")
    print(f"  B wins : {results['B']:4d}  ({b_pct:.1f}%)")
    print(f"  Draws  : {results['draw']:4d}  ({d_pct:.1f}%)")
    print(f"  A edge : {edge:+.1f}pp  ({verdict})")
    print(f"{'='*62}\n")
    return results


# ── Round-robin ────────────────────────────────────────────────────────────────

def run_round_robin(n_games: int, difficulty: int, time_budget: float | None,
                    vn_path: str, gap_path: str,
                    blends: list[int]) -> None:
    """Run all config pairs against each other and print standings."""
    print("Loading components...")
    traj_vn  = _load_value_net(vn_path)
    gap_net  = _load_gap_net(gap_path)
    print()

    # Build config list
    configs: list[dict] = [
        {"name": "Baseline",
         "value_net": None,  "gap_net": None,    "vn_blend": 0},
    ]
    for pct in blends:
        configs.append({
            "name":      f"TrajVN-{pct}%",
            "value_net": traj_vn, "gap_net": None, "vn_blend": pct,
        })
    configs.append({
        "name":      "GapNet",
        "value_net": None,  "gap_net": gap_net, "vn_blend": 0,
    })

    n = len(configs)
    n_pairs = n * (n - 1) // 2
    print(f"Round-robin: {n} configs  {n_pairs} pairs × {n_games} games = {n_pairs * n_games} total\n")
    print("Configs:")
    for c in configs:
        print(f"  {c['name']}")
    print()

    wins  = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]

    pairs = list(combinations(range(n), 2))
    for match_num, (i, j) in enumerate(pairs, 1):
        ca, cb = configs[i], configs[j]
        print(f"Matchup {match_num}/{n_pairs}: {ca['name']} vs {cb['name']}")
        r = run_matchup(
            ca["name"], cb["name"], n_games, difficulty, time_budget,
            value_net_a=ca["value_net"], gap_net_a=ca["gap_net"], vn_blend_a=ca["vn_blend"],
            value_net_b=cb["value_net"], gap_net_b=cb["gap_net"], vn_blend_b=cb["vn_blend"],
        )
        wins[i][j] += r["A"]
        wins[j][i] += r["B"]
        draws[i][j] += r["draw"]
        draws[j][i] += r["draw"]

    # Standings
    total_w = [sum(wins[i]) for i in range(n)]
    total_l = [sum(wins[j][i] for j in range(n)) for i in range(n)]
    total_d = [sum(draws[i]) // 2 for i in range(n)]
    ranking = sorted(range(n), key=lambda x: total_w[x], reverse=True)

    print(f"\n{'='*66}")
    print(f"  ROUND-ROBIN STANDINGS  ({n_games} games/pair, d{difficulty})")
    print(f"{'='*66}")
    print(f"  {'Config':<18}  {'W':>4}  {'L':>4}  {'D':>4}  {'Win%':>6}")
    print(f"  {'-'*40}")
    for idx in ranking:
        played = total_w[idx] + total_l[idx] + total_d[idx]
        win_pct = 100 * total_w[idx] / played if played > 0 else 0.0
        print(f"  {configs[idx]['name']:<18}  {total_w[idx]:>4}  {total_l[idx]:>4}  "
              f"{total_d[idx]:>4}  {win_pct:>6.1f}%")

    # Head-to-head matrix
    names = [c["name"] for c in configs]
    col_w = max(len(nm) for nm in names)
    header_col = 14
    print(f"\n  Head-to-head wins (row vs col, out of {n_games}):")
    header = f"  {'':{col_w}}"
    for nm in names:
        header += f"  {nm[:header_col]:>{header_col}}"
    print(header)
    for i, nm in enumerate(names):
        row = f"  {nm:{col_w}}"
        for j in range(n):
            row += f"  {'---':>{header_col}}" if i == j else f"  {wins[i][j]:>{header_col}}"
        print(row)
    print(f"{'='*66}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Round-robin benchmark: trajectory value net vs gap net vs baseline"
    )
    p.add_argument("--games",      type=int,   default=20,
                   help="Games per matchup pair (default 20; must be even)")
    p.add_argument("--difficulty", type=int,   default=4,
                   help="AI difficulty for all configs (default 4)")
    p.add_argument("--time-budget", type=float, default=None,
                   help="Per-move time budget in seconds (alias: --time-limit)")
    p.add_argument("--time-limit", type=float, default=None,
                   help="Per-move time limit in seconds (alias: --time-budget)")
    p.add_argument("--vn-path",    default=_TRAJ_VN_PATH,
                   help=f"Trajectory value net .npz path (default: {_TRAJ_VN_PATH})")
    p.add_argument("--gap-path",   default=_GAP_NET_PATH,
                   help=f"Gap net .npz path (default: {_GAP_NET_PATH})")
    p.add_argument("--blends",     type=int,   nargs="+", default=[10, 30, 60],
                   metavar="PCT",
                   help="Value-net blend percentages to test (default: 10 30 60)")
    p.add_argument("--matchup",    nargs=2,    metavar=("A", "B"),
                   help="Run a single matchup between two named configs instead of round-robin. "
                        "Config names: Baseline, TrajVN-10%%, TrajVN-30%%, TrajVN-60%%, GapNet "
                        "(or TrajVN-N%% for a custom blend from --blends)")
    args = p.parse_args()

    time_budget = args.time_limit or args.time_budget

    if args.games % 2 != 0:
        print("Warning: --games should be even so each config plays equal White/Black.", flush=True)

    if args.matchup:
        # Single matchup mode
        label_a, label_b = args.matchup

        print("Loading components...")
        traj_vn = _load_value_net(args.vn_path)
        gap_net = _load_gap_net(args.gap_path)
        print()

        def _resolve(name: str):
            nl = name.lower()
            if nl == "baseline":
                return {"value_net": None, "gap_net": None, "vn_blend": 0}
            if nl == "gapnet":
                return {"value_net": None, "gap_net": gap_net, "vn_blend": 0}
            if nl.startswith("trajvn-"):
                try:
                    pct = int(nl.removeprefix("trajvn-").rstrip("%"))
                    return {"value_net": traj_vn, "gap_net": None, "vn_blend": pct}
                except ValueError:
                    pass
            print(f"Unknown config name '{name}'. "
                  f"Use: Baseline, TrajVN-10%, TrajVN-30%, TrajVN-60%, GapNet")
            sys.exit(1)

        ca = _resolve(label_a)
        cb = _resolve(label_b)
        run_matchup(
            label_a, label_b, args.games, args.difficulty, time_budget,
            value_net_a=ca["value_net"], gap_net_a=ca["gap_net"], vn_blend_a=ca["vn_blend"],
            value_net_b=cb["value_net"], gap_net_b=cb["gap_net"], vn_blend_b=cb["vn_blend"],
        )
        return 0

    # Default: full round-robin
    run_round_robin(
        n_games=args.games,
        difficulty=args.difficulty,
        time_budget=time_budget,
        vn_path=args.vn_path,
        gap_path=args.gap_path,
        blends=args.blends,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
