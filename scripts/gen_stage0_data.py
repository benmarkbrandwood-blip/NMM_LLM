"""scripts/gen_stage0_data.py — Generate Stage 0 supervised pre-training data.

Runs heuristic self-play (diff 5, vn_blend=80%) in parallel across CPU cores
and labels every board position with the value net score from the side-to-move's
perspective.  Output is a .npz file consumed by train_stage0.py.

Usage:
    .venv/bin/python scripts/gen_stage0_data.py [--games N] [--workers N]
                                                 [--out PATH] [--difficulty D]
                                                 [--budget S]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ── Per-worker globals (set once in initializer, reused across games) ─────────
_value_net = None
_weights    = None
_difficulty = None
_budget     = None


def _worker_init(vn_blob: dict, difficulty: int, budget: float, vn_blend: int) -> None:
    """Initialise per-worker state — called once per process in the pool."""
    global _value_net, _weights, _difficulty, _budget

    # Re-import inside the worker so each process has its own copies
    import sys, os
    sys.path.insert(0, str(_ROOT))

    from ai.value_net import ValueNet
    from ai.heuristics import HeuristicWeights

    # Reconstruct value net from raw numpy arrays (avoids repeated pickle overhead)
    vn = ValueNet()
    vn.W1 = vn_blob["W1"]; vn.b1 = vn_blob["b1"]
    vn.W2 = vn_blob["W2"]; vn.b2 = vn_blob["b2"]
    vn.W3 = vn_blob["W3"]; vn.b3 = vn_blob["b3"]

    _value_net  = vn
    _weights    = HeuristicWeights(value_net_blend=vn_blend)
    _difficulty = difficulty
    _budget     = budget


def _run_one_game(_game_idx: int) -> tuple[list, list, list]:
    """Run a single game and return (states, phase_ids, values)."""
    from ai.game_ai import GameAI
    from game.board import BoardState
    from game.rules import get_all_legal_moves, is_terminal
    from learned_ai.models.state_encoder import detect_phase, encode_state

    white_ai = GameAI(color="W", difficulty=_difficulty, weights=_weights,
                      value_net=_value_net, override_time_budget=_budget)
    black_ai = GameAI(color="B", difficulty=_difficulty, weights=_weights,
                      value_net=_value_net, override_time_budget=_budget)

    states, phase_ids, values = [], [], []
    board = BoardState.new_game()
    game_seen: set = set()

    for _ in range(400):
        done, _ = is_terminal(board)
        if done:
            break
        if not get_all_legal_moves(board):
            break

        fp = tuple(board.positions[p] for p in sorted(board.positions)) + (board.turn,)
        if fp not in game_seen:
            game_seen.add(fp)
            score = _value_net.predict(board, board.turn)
            states.append(encode_state(board).numpy())
            phase_ids.append(detect_phase(board))
            values.append(float(score))

        ai = white_ai if board.turn == "W" else black_ai
        move = ai.choose_move(board)
        if move is None:
            break
        board = board.apply_move(move)

    return states, phase_ids, values


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Stage 0 supervised value pre-training data")
    p.add_argument("--games",      type=int,   default=500)
    p.add_argument("--workers",    type=int,   default=max(1, os.cpu_count() - 2),
                   help="Parallel workers (default: nproc-2)")
    p.add_argument("--difficulty", type=int,   default=5)
    p.add_argument("--budget",     type=float, default=0.1,
                   help="Seconds per move per worker")
    p.add_argument("--vn-blend",   type=int,   default=80)
    p.add_argument("--out",        type=str,
                   default=str(_ROOT / "learned_ai" / "data" / "stage0_positions.npz"))
    args = p.parse_args()

    from ai.value_net import ValueNet
    vn_path = _ROOT / "data" / "value_net.npz"
    if not vn_path.exists():
        print(f"ERROR: value net not found at {vn_path}")
        sys.exit(1)
    vn = ValueNet.load(str(vn_path))
    # Pack raw arrays for cheap pickle transfer to workers
    vn_blob = {"W1": vn.W1, "b1": vn.b1, "W2": vn.W2,
               "b2": vn.b2, "W3": vn.W3, "b3": vn.b3}
    print(f"Value net loaded from {vn_path}")
    print(f"Running {args.games} games on {args.workers} workers "
          f"(diff={args.difficulty}, budget={args.budget}s, vn_blend={args.vn_blend}%)\n")

    all_states:  list[np.ndarray] = []
    all_phases:  list[int]        = []
    all_values:  list[float]      = []
    t0 = time.time()

    init_args = (vn_blob, args.difficulty, args.budget, args.vn_blend)
    with mp.Pool(processes=args.workers,
                 initializer=_worker_init,
                 initargs=init_args) as pool:
        for done, (s, ph, v) in enumerate(
            pool.imap_unordered(_run_one_game, range(args.games), chunksize=1), 1
        ):
            all_states.extend(s)
            all_phases.extend(ph)
            all_values.extend(v)

            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (args.games - done) / rate if rate > 0 else 0
            print(f"\r  {done}/{args.games}  positions={len(all_states)}"
                  f"  {rate:.2f} g/s  ETA {eta:.0f}s    ", end="", flush=True)

    print()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        states=np.array(all_states,  dtype=np.float32),
        phase_ids=np.array(all_phases, dtype=np.int8),
        values=np.array(all_values,  dtype=np.float32),
    )
    total = time.time() - t0
    print(f"Saved {len(all_states)} positions to {out_path}  ({total:.0f}s total)")
    phase_counts: dict[int, int] = {}
    for ph in all_phases:
        phase_counts[ph] = phase_counts.get(ph, 0) + 1
    print(f"Phase distribution: {dict(sorted(phase_counts.items()))}")
    print(f"Value range: [{min(all_values):.3f}, {max(all_values):.3f}]  "
          f"mean={sum(all_values)/len(all_values):.3f}")


if __name__ == "__main__":
    main()
