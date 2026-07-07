"""scripts/bench_sentinel.py — headless AI vs AI benchmark.

Runs N games between two GameAI configurations and reports win/draw/loss rates.
Use this to measure whether sentinel, value_net, or gap_net improve the heuristic engine.

Usage examples
--------------
# Baseline vs Baseline (sanity check — should be near 50/50)
python scripts/bench_sentinel.py --games 200 --difficulty 4

# Sentinel (score_adjust) vs Baseline
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust

# Sentinel + value_net vs Baseline
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust --white-value-net

# Gap net vs Baseline
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-gap-net

# Sentinel + gap net vs Baseline
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust --white-gap-net

# Sentinel vs Sentinel (should be ~50/50)
python scripts/bench_sentinel.py --games 200 --difficulty 4 \
  --white-sentinel score_adjust --black-sentinel score_adjust

# V3a suite: three matchups back to back (4 games each, difficulty 5)
python scripts/bench_sentinel.py --suite --games 4 --difficulty 5

Colours alternate first-mover across games to cancel first-move advantage.
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BoardState
from ai.game_ai import GameAI

_SENTINEL_CKPT = "learned_ai/sentinel/checkpoints/best.pt"
_VALUE_NET_PATH = "data/value_net.npz"
_GAP_NET_PATH = "data/gap_net.npz"


def _load_sentinel(path: str = _SENTINEL_CKPT):
    try:
        from learned_ai.sentinel.infer import load_advisor
        advisor = load_advisor(path)
        if advisor:
            print(f"  Sentinel loaded: {path}")
        else:
            print("  Sentinel load returned None.")
        return advisor
    except Exception as e:
        print(f"  Sentinel load failed: {e}")
        return None


def _load_value_net():
    try:
        from ai.value_net import ValueNet
        vn = ValueNet.load(_VALUE_NET_PATH)
        print(f"  Value net loaded: {_VALUE_NET_PATH}")
        return vn
    except Exception as e:
        print(f"  Value net load failed: {e}")
        return None


def _load_gap_net():
    try:
        from ai.value_net import ValueNet
        gn = ValueNet.load(_GAP_NET_PATH)
        print(f"  Gap net loaded: {_GAP_NET_PATH}")
        return gn
    except Exception as e:
        print(f"  Gap net load failed: {e}")
        return None


def _make_ai(color: str, difficulty: int, sentinel=None, sentinel_mode: str = "advisory",
             value_net=None, gap_net=None, time_budget: float | None = None,
             vn_blend: int = 0, sentinel_scale: float | None = None) -> GameAI:
    from ai.heuristics import HeuristicWeights
    weights = HeuristicWeights(value_net_blend=vn_blend) if vn_blend else None
    ai = GameAI(color=color, difficulty=difficulty, value_net=value_net,
                gap_net=gap_net, weights=weights, override_time_budget=time_budget)
    if sentinel is not None:
        ai.set_sentinel(sentinel, mode=sentinel_mode)
        if sentinel_scale is not None:
            ai._sentinel_score_scale = sentinel_scale
    return ai


def play_game(white_ai: GameAI, black_ai: GameAI, max_plies: int = 400) -> Optional[str]:
    """Return 'W', 'B', or None (draw/stalemate)."""
    from game.game_engine import GameEngine
    from game.rules import is_terminal

    engine = GameEngine(human_color=None)  # no human — both sides AI
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

    return None  # draw by length


def run_matchup(
    label_a: str,
    label_b: str,
    n_games: int,
    difficulty: int,
    time_budget: float | None,
    # Config A
    sentinel_a=None, sentinel_mode_a: str = "advisory", sentinel_scale_a: float | None = None,
    value_net_a=None, gap_net_a=None, vn_blend_a: int = 0,
    # Config B (baseline has all None/0)
    sentinel_b=None, sentinel_mode_b: str = "advisory", sentinel_scale_b: float | None = None,
    value_net_b=None, gap_net_b=None, vn_blend_b: int = 0,
) -> dict:
    """Run n_games between config A and config B, alternating colours. Returns results dict."""
    results = {"A": 0, "B": 0, "draw": 0}
    t0 = time.time()

    for g in range(n_games):
        if g % 2 == 0:
            w_ai = _make_ai("W", difficulty, sentinel_a, sentinel_mode_a, value_net_a,
                            gap_net_a, time_budget, vn_blend_a, sentinel_scale_a)
            b_ai = _make_ai("B", difficulty, sentinel_b, sentinel_mode_b, value_net_b,
                            gap_net_b, time_budget, vn_blend_b, sentinel_scale_b)
            a_color = "W"
        else:
            w_ai = _make_ai("W", difficulty, sentinel_b, sentinel_mode_b, value_net_b,
                            gap_net_b, time_budget, vn_blend_b, sentinel_scale_b)
            b_ai = _make_ai("B", difficulty, sentinel_a, sentinel_mode_a, value_net_a,
                            gap_net_a, time_budget, vn_blend_a, sentinel_scale_a)
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
        rate = done / elapsed
        eta = (n_games - done) / rate if rate > 0 else 0
        print(f"\r  {done}/{n_games}  A:{results['A']}  B:{results['B']}  "
              f"D:{results['draw']}  {rate:.1f} g/s  ETA {eta:.0f}s    ", end="", flush=True)

    print()
    elapsed = time.time() - t0
    total = n_games
    a_rate = 100 * results["A"] / total
    b_rate = 100 * results["B"] / total
    d_rate = 100 * results["draw"] / total
    edge = a_rate - b_rate
    print(f"\n{'='*62}")
    print(f"  RESULTS after {total} games  ({elapsed:.0f}s)")
    print(f"{'='*62}")
    print(f"  Config A: {label_a}")
    print(f"  Config B: {label_b}")
    print(f"  (each config plays White in half the games, Black in the other half)")
    print(f"{'='*62}")
    print(f"  A wins : {results['A']:4d}  ({a_rate:.1f}%)")
    print(f"  B wins : {results['B']:4d}  ({b_rate:.1f}%)")
    print(f"  Draws  : {results['draw']:4d}  ({d_rate:.1f}%)")
    print(f"  A edge : {edge:+.1f}pp  ({'A better' if edge > 2 else 'B better' if edge < -2 else 'roughly equal'})")
    print(f"{'='*62}\n")
    return results


def run_suite(n_games: int, difficulty: int, time_budget: float | None,
              sentinel_scale: float, sentinel_path: str) -> None:
    """Run all three V3a matchups: sentinel / gap_net / both vs baseline."""
    print(f"Running 3 matchups × {n_games} games = {3 * n_games} total\n")
    print("Loading components...")
    sentinel = _load_sentinel(sentinel_path)
    gap_net = _load_gap_net()
    print()

    baseline_label = f"Baseline[d{difficulty}]"
    scale_pct = int(sentinel_scale * 100)

    suite_results = []

    # Matchup 1: Sentinel (score_adjust @ sentinel_scale) vs Baseline
    label_a = f"Sentinel[score_adjust,{scale_pct}%,d{difficulty}]"
    print(f"Matchup 1/3: {label_a} vs {baseline_label}")
    r1 = run_matchup(
        label_a, baseline_label, n_games, difficulty, time_budget,
        sentinel_a=sentinel, sentinel_mode_a="score_adjust", sentinel_scale_a=sentinel_scale,
    )
    suite_results.append((label_a, r1))

    # Matchup 2: GapNet vs Baseline
    label_a = f"GapNet[d{difficulty}]"
    print(f"Matchup 2/3: {label_a} vs {baseline_label}")
    r2 = run_matchup(
        label_a, baseline_label, n_games, difficulty, time_budget,
        gap_net_a=gap_net,
    )
    suite_results.append((label_a, r2))

    # Matchup 3: Sentinel + GapNet vs Baseline
    label_a = f"Sentinel+GapNet[score_adjust,{scale_pct}%,d{difficulty}]"
    print(f"Matchup 3/3: {label_a} vs {baseline_label}")
    r3 = run_matchup(
        label_a, baseline_label, n_games, difficulty, time_budget,
        sentinel_a=sentinel, sentinel_mode_a="score_adjust", sentinel_scale_a=sentinel_scale,
        gap_net_a=gap_net,
    )
    suite_results.append((label_a, r3))

    # Summary table
    print(f"\n{'='*62}")
    print(f"  SUITE SUMMARY  ({n_games} games each, d{difficulty})")
    print(f"{'='*62}")
    print(f"  {'Config A':<42}  {'A%':>5}  {'B%':>5}  {'Edge':>6}")
    print(f"  {'-'*58}")
    for label, r in suite_results:
        total = n_games
        a_pct = 100 * r["A"] / total
        b_pct = 100 * r["B"] / total
        edge = a_pct - b_pct
        verdict = "A+" if edge > 2 else ("B+" if edge < -2 else "~=")
        print(f"  {label:<42}  {a_pct:>5.1f}  {b_pct:>5.1f}  {edge:>+6.1f}  {verdict}")
    print(f"{'='*62}")


def run_round_robin(n_games: int, difficulty: int, time_budget: float | None,
                    sentinel_scale: float, sentinel_path: str) -> None:
    """Run all 4 configs against each other (6 pairs, n_games each)."""
    from itertools import combinations
    n_pairs = 6
    print(f"Running round-robin: {n_pairs} pairs × {n_games} games = {n_pairs * n_games} total\n")
    print("Loading components...")
    sentinel = _load_sentinel(sentinel_path)
    gap_net = _load_gap_net()
    print()

    scale_pct = int(sentinel_scale * 100)
    configs = [
        {"name": "Baseline",            "sentinel": None,     "sentinel_mode": "advisory", "sentinel_scale": None,           "gap_net": None},
        {"name": f"Sentinel@{scale_pct}%", "sentinel": sentinel, "sentinel_mode": "score_adjust", "sentinel_scale": sentinel_scale, "gap_net": None},
        {"name": "GapNet",               "sentinel": None,     "sentinel_mode": "advisory", "sentinel_scale": None,           "gap_net": gap_net},
        {"name": f"Sen+Gap@{scale_pct}%",  "sentinel": sentinel, "sentinel_mode": "score_adjust", "sentinel_scale": sentinel_scale, "gap_net": gap_net},
    ]

    # wins[i][j] = number of times config i beat config j
    n = len(configs)
    wins = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]

    pairs = list(combinations(range(n), 2))
    for matchup_num, (i, j) in enumerate(pairs, 1):
        ca, cb = configs[i], configs[j]
        print(f"Matchup {matchup_num}/{n_pairs}: {ca['name']} vs {cb['name']}")
        r = run_matchup(
            ca["name"], cb["name"], n_games, difficulty, time_budget,
            sentinel_a=ca["sentinel"], sentinel_mode_a=ca["sentinel_mode"], sentinel_scale_a=ca["sentinel_scale"],
            gap_net_a=ca["gap_net"],
            sentinel_b=cb["sentinel"], sentinel_mode_b=cb["sentinel_mode"], sentinel_scale_b=cb["sentinel_scale"],
            gap_net_b=cb["gap_net"],
        )
        wins[i][j] += r["A"]
        wins[j][i] += r["B"]
        draws[i][j] += r["draw"]
        draws[j][i] += r["draw"]

    # Standings table
    total_w = [sum(wins[i]) for i in range(n)]
    total_l = [sum(wins[j][i] for j in range(n)) for i in range(n)]
    total_d = [sum(draws[i]) // 2 for i in range(n)]   # each draw counted once per side above
    ranking = sorted(range(n), key=lambda x: total_w[x], reverse=True)

    print(f"\n{'='*66}")
    print(f"  ROUND-ROBIN STANDINGS  ({n_games} games/pair, d{difficulty})")
    print(f"{'='*66}")
    print(f"  {'Config':<22}  {'W':>4}  {'L':>4}  {'D':>4}  {'Win%':>6}")
    print(f"  {'-'*44}")
    for idx in ranking:
        played = total_w[idx] + total_l[idx] + total_d[idx]
        win_pct = 100 * total_w[idx] / played if played > 0 else 0.0
        print(f"  {configs[idx]['name']:<22}  {total_w[idx]:>4}  {total_l[idx]:>4}  {total_d[idx]:>4}  {win_pct:>6.1f}%")

    # Head-to-head matrix (row beats col, count / n_games)
    names = [c["name"] for c in configs]
    col_w = max(len(nm) for nm in names)
    print(f"\n  Head-to-head wins (row vs col out of {n_games}):")
    header = f"  {'':{col_w}}"
    for nm in names:
        header += f"  {nm[:10]:>10}"
    print(header)
    for i, nm in enumerate(names):
        row = f"  {nm:{col_w}}"
        for j in range(n):
            if i == j:
                row += f"  {'---':>10}"
            else:
                row += f"  {wins[i][j]:>10}"
        print(row)
    print(f"{'='*66}")


def main() -> int:
    p = argparse.ArgumentParser(description="Headless sentinel/gap-net benchmark")
    p.add_argument("--games", type=int, default=4,
                   help="Games per matchup (default 4; suite recommends 2-6)")
    p.add_argument("--difficulty", type=int, default=4)
    p.add_argument("--suite", action="store_true",
                   help="Run all three V3a matchups: sentinel / gap_net / both vs baseline")
    p.add_argument("--round-robin", action="store_true",
                   help="Run all 4 configs against each other (6 pairs × --games each)")
    p.add_argument("--sentinel-scale", type=float, default=0.20,
                   help="score_adjust scale override (default 0.20 = 20%%)")
    p.add_argument("--white-sentinel", default=None,
                   choices=["advisory", "score_adjust", "reconsider"],
                   help="Sentinel mode for White; omit for pure heuristic")
    p.add_argument("--black-sentinel", default=None,
                   choices=["advisory", "score_adjust", "reconsider"])
    p.add_argument("--sentinel-path", default=_SENTINEL_CKPT,
                   help="Path to sentinel checkpoint (default: best.pt)")
    p.add_argument("--white-value-net", action="store_true")
    p.add_argument("--black-value-net", action="store_true")
    p.add_argument("--vn-blend", type=int, default=0,
                   help="value_net_blend %% (0=off; e.g. 80 blends 80%% VN into leaf eval)")
    p.add_argument("--white-gap-net", action="store_true",
                   help="Enable gap_net blunder-zone correction for White (V3a)")
    p.add_argument("--black-gap-net", action="store_true",
                   help="Enable gap_net blunder-zone correction for Black (V3a)")
    p.add_argument("--time-budget", type=float, default=None,
                   help="Seconds per move override (omit to use difficulty's natural time budget)")
    args = p.parse_args()

    if args.suite:
        run_suite(args.games, args.difficulty, args.time_budget,
                  args.sentinel_scale, args.sentinel_path)
        return 0

    if args.round_robin:
        run_round_robin(args.games, args.difficulty, args.time_budget,
                        args.sentinel_scale, args.sentinel_path)
        return 0

    # --- manual single-matchup mode (unchanged behaviour) ---
    need_sentinel = args.white_sentinel or args.black_sentinel
    need_vn = args.white_value_net or args.black_value_net
    need_gn = args.white_gap_net or args.black_gap_net

    print("Loading components...")
    sentinel = _load_sentinel(args.sentinel_path) if need_sentinel else None
    value_net = _load_value_net() if need_vn else None
    gap_net = _load_gap_net() if need_gn else None
    print()

    white_label = f"White[d{args.difficulty}"
    black_label = f"Black[d{args.difficulty}"
    if args.white_sentinel:
        scale_pct = int(args.sentinel_scale * 100)
        white_label += f"+sentinel:{args.white_sentinel}@{scale_pct}%"
    if args.white_value_net:
        white_label += f"+vn{args.vn_blend}%"
    if args.white_gap_net:
        white_label += "+gap_net"
    white_label += "]"
    if args.black_sentinel:
        scale_pct = int(args.sentinel_scale * 100)
        black_label += f"+sentinel:{args.black_sentinel}@{scale_pct}%"
    if args.black_value_net:
        black_label += f"+vn{args.vn_blend}%"
    if args.black_gap_net:
        black_label += "+gap_net"
    black_label += "]"

    w_scale = args.sentinel_scale if args.white_sentinel else None
    b_scale = args.sentinel_scale if args.black_sentinel else None

    run_matchup(
        white_label, black_label, args.games, args.difficulty, args.time_budget,
        sentinel_a=sentinel if args.white_sentinel else None,
        sentinel_mode_a=args.white_sentinel or "advisory",
        sentinel_scale_a=w_scale,
        value_net_a=value_net if args.white_value_net else None,
        gap_net_a=gap_net if args.white_gap_net else None,
        vn_blend_a=args.vn_blend if args.white_value_net else 0,
        sentinel_b=sentinel if args.black_sentinel else None,
        sentinel_mode_b=args.black_sentinel or "advisory",
        sentinel_scale_b=b_scale,
        value_net_b=value_net if args.black_value_net else None,
        gap_net_b=gap_net if args.black_gap_net else None,
        vn_blend_b=args.vn_blend if args.black_value_net else 0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
