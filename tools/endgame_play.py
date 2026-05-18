"""tools/endgame_play.py — Endgame self-play for rapid EndgameDB enrichment.

Instead of running full games from the start, this tool generates (or
extracts) endgame positions and plays them out from there.  Each completed
game lands in data/games/ in the standard JSONL format so EndgameDB picks
it up on the next server restart (or incremental reload).

Why endgame-only?
  Full-game self-play produces only a handful of sub-11-piece positions per
  game.  Starting from endgame positions generates hundreds per minute,
  making EndgameDB far more useful for move guidance without the overhead
  of the placement and mid-game phases.

Two position sources:
  random (default)      Generate random valid endgame boards (both sides ≥3
                        pieces, total ≤ threshold, both sides have legal moves).
  --seed-from-games     Extract real endgame positions from existing JSONL
                        game records — guarantees positions are reachable.

Usage:
  # 200 random endgame positions, 4 parallel workers
  python tools/endgame_play.py --positions 200 --parallel 4

  # Seed from real games, difficulty 6
  python tools/endgame_play.py --seed-from-games --positions 500 --difficulty 6

  # Narrow piece range, verbose
  python tools/endgame_play.py --positions 20 --min-pieces 5 --max-pieces 8 -v

  # Alternate AI personalities to reduce draws
  python tools/endgame_play.py --positions 100 --personalities balanced,positional,defensive
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.board import POSITIONS, BoardState
from game.game_engine import GameEngine
from game.rules import get_all_legal_moves, is_terminal
from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights

_GAMES_DIR = ROOT / "data" / "games"

_MAX_MOVES   = 200   # cap per endgame game (endgames are short)
_REPEAT_DRAW = 3     # draw after 3-fold board repetition

# Re-use personality presets from self_play.py — copied here so this tool
# is self-contained as a subprocess worker.
_PERSONALITIES: dict[str, dict] = {
    "balanced": {
        "close_mill": 500, "cycling_mill": 50, "block_opponent_mill": 400,
        "stop_opponent_mills": 450, "feeder_diamond": 200, "mill_wrapping": 150,
        "cardinal_block": 400, "scatter_placement": 100, "long_term_position": 100,
        "mill_count_scale": 100, "mobility_scale": 100, "blocked_scale": 100,
        "make_mistakes": 0, "opening_adherence": 30,
    },
    "aggressive": {
        "close_mill": 900, "cycling_mill": 75, "block_opponent_mill": 150,
        "stop_opponent_mills": 150, "feeder_diamond": 350, "mill_wrapping": 50,
        "cardinal_block": 500, "scatter_placement": 25, "long_term_position": 70,
        "mill_count_scale": 180, "mobility_scale": 50, "blocked_scale": 80,
        "make_mistakes": 0, "opening_adherence": 15,
    },
    "defensive": {
        "close_mill": 300, "cycling_mill": 25, "block_opponent_mill": 850,
        "stop_opponent_mills": 800, "feeder_diamond": 350, "mill_wrapping": 350,
        "cardinal_block": 275, "scatter_placement": 100, "long_term_position": 150,
        "mill_count_scale": 75, "mobility_scale": 200, "blocked_scale": 250,
        "make_mistakes": 0, "opening_adherence": 25,
    },
    "positional": {
        "close_mill": 400, "cycling_mill": 60, "block_opponent_mill": 350,
        "stop_opponent_mills": 350, "feeder_diamond": 300, "mill_wrapping": 250,
        "cardinal_block": 500, "scatter_placement": 450, "long_term_position": 200,
        "mill_count_scale": 80, "mobility_scale": 300, "blocked_scale": 150,
        "make_mistakes": 0, "opening_adherence": 40,
    },
    "scholar": {
        "close_mill": 450, "cycling_mill": 50, "block_opponent_mill": 400,
        "stop_opponent_mills": 400, "feeder_diamond": 250, "mill_wrapping": 200,
        "cardinal_block": 450, "scatter_placement": 400, "long_term_position": 175,
        "mill_count_scale": 100, "mobility_scale": 200, "blocked_scale": 125,
        "make_mistakes": 0, "opening_adherence": 50,
    },
    "chaos": {
        "close_mill": 150, "cycling_mill": 25, "block_opponent_mill": 150,
        "stop_opponent_mills": 150, "feeder_diamond": 75, "mill_wrapping": 25,
        "cardinal_block": 0, "scatter_placement": 500, "long_term_position": 10,
        "mill_count_scale": 50, "mobility_scale": 50, "blocked_scale": 50,
        "make_mistakes": 45, "opening_adherence": 0,
    },
}
_DEFAULT_POOL = [p for p in _PERSONALITIES if p != "chaos"]


def _hw_from_preset(name: str) -> HeuristicWeights:
    p = _PERSONALITIES[name]
    return HeuristicWeights(**{k: p[k] for k in p})


def _make_ai(color: str, difficulty: int, personality: str) -> GameAI:
    hw = _hw_from_preset(personality)
    return GameAI(color=color, difficulty=difficulty, weights=hw,
                  blunder_probability=hw.make_mistakes / 100.0)


# ── Position generation ───────────────────────────────────────────────────────

def _gen_random_position(
    min_pieces: int,
    max_pieces: int,
    rng: random.Random,
    max_attempts: int = 50,
) -> dict | None:
    """
    Return a position dict {"positions": {...}, "turn": "W"|"B", "start_fen": str}
    or None if no valid position could be generated within max_attempts.

    Valid means:
      - Each side has between 3 and (total-3) pieces, both ≥ 3.
      - Neither side has already won (is_terminal returns False).
      - The side to move has at least one legal move.
    """
    for _ in range(max_attempts):
        total = rng.randint(min_pieces, max_pieces)
        # Both sides must have ≥ 3 pieces.
        n_w = rng.randint(3, total - 3)
        n_b = total - n_w

        chosen = rng.sample(POSITIONS, total)
        pos_dict = {p: "" for p in POSITIONS}
        for p in chosen[:n_w]:
            pos_dict[p] = "W"
        for p in chosen[n_w:]:
            pos_dict[p] = "B"

        turn = rng.choice(["W", "B"])
        board = BoardState.from_setup(pos_dict, turn, phase="move")

        # Reject terminal positions (already won/lost).
        terminal, _ = is_terminal(board)
        if terminal:
            continue

        # Reject positions where the side to move has no moves.
        if not get_all_legal_moves(board):
            continue

        return {
            "positions": pos_dict,
            "turn":      turn,
            "start_fen": board.to_fen_string(),
        }
    return None


def _extract_from_games(
    games_dir: Path,
    max_pieces: int,
    limit: int,
    rng: random.Random,
) -> list[dict]:
    """
    Read all JSONL game records and collect unique endgame positions where
    both sides have placed all 9 pieces and total ≤ max_pieces.
    """
    seen_fens: set[str] = set()
    results: list[dict] = []

    for path in sorted(games_dir.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            record = json.loads(text)
        except Exception:
            continue

        for move in record.get("moves", []):
            fen = move.get("board_fen_before", "")
            if not fen:
                continue
            # Placement must be done (both sides placed all 9)
            parts = fen.split("|")
            if len(parts) < 4:
                continue
            try:
                if int(parts[2]) < 9 or int(parts[3]) < 9:
                    continue
            except ValueError:
                continue
            # Total pieces must be within threshold
            board_str = parts[0]
            total = board_str.count("W") + board_str.count("B")
            if total > max_pieces or total < 6:
                continue
            if fen in seen_fens:
                continue
            seen_fens.add(fen)

            # Reconstruct position dict and turn
            turn = parts[1] if len(parts[1]) == 1 else "W"
            pos_dict = {p: "" for p in POSITIONS}
            for i, sq in enumerate(POSITIONS):
                ch = board_str[i] if i < len(board_str) else "."
                if ch in ("W", "B"):
                    pos_dict[sq] = ch

            board = BoardState.from_setup(pos_dict, turn, phase="move")
            terminal, _ = is_terminal(board)
            if terminal:
                continue
            if not get_all_legal_moves(board):
                continue

            results.append({
                "positions": pos_dict,
                "turn":      turn,
                "start_fen": fen,
            })
            if len(results) >= limit * 3:   # gather extra, shuffle later
                break
        if len(results) >= limit * 3:
            break

    rng.shuffle(results)
    return results[:limit]


# ── Single game (module-level for ProcessPoolExecutor pickling) ───────────────

def _play_endgame(params: dict) -> dict:
    """
    Play one endgame from the given start position.
    Returns a JSONL-ready game record dict.
    """
    sys.path.insert(0, str(ROOT))

    pos_entry        = params["pos_entry"]
    difficulty       = params["difficulty"]
    white_personality = params["white_personality"]
    black_personality = params["black_personality"]
    verbose          = params["verbose"]
    game_num         = params["game_num"]

    board_start = BoardState.from_setup(
        pos_entry["positions"], pos_entry["turn"], phase="move"
    )

    engine = GameEngine(human_color="B")
    engine.board = board_start

    white_ai = _make_ai("W", difficulty, white_personality)
    black_ai = _make_ai("B", difficulty, black_personality)

    session_id = str(uuid.uuid4())
    moves_log: list[dict] = []
    fen_counts: Counter = Counter()
    moves_since_capture = 0
    move_count = 0
    draw_by_repetition = False

    label = f"[Game {game_num}] " if verbose else ""

    while not engine.finished and move_count < _MAX_MOVES:
        board = engine.board
        fen   = board.to_fen_string()
        fen_counts[fen] += 1
        if fen_counts[fen] >= _REPEAT_DRAW:
            draw_by_repetition = True
            break
        if moves_since_capture >= 80:
            draw_by_repetition = True
            break

        color = board.turn
        ai    = white_ai if color == "W" else black_ai

        t0   = time.perf_counter()
        move = ai.choose_move(board, fast_early_game=True)
        elapsed = time.perf_counter() - t0

        frm = move.get("from")
        to  = move.get("to")
        cap = move.get("capture")
        notation = (f"{frm}-{to}" if frm else to) + (f"x{cap}" if cap else "")

        moves_log.append({
            "turn":             move_count + 1,
            "color":            color,
            "type":             "move",
            "from":             frm,
            "to":               to,
            "capture":          cap,
            "notation":         notation,
            "board_fen_before": fen,
        })

        engine.apply_move(move)
        moves_since_capture = 0 if cap else moves_since_capture + 1
        move_count += 1

        if verbose:
            color_name = "White" if color == "W" else "Black"
            print(f"{label}Move {move_count:3d}: {color_name} {notation}  ({elapsed:.2f}s)")

    if verbose:
        result = (f"{'White' if engine.winner == 'W' else 'Black'} wins"
                  if engine.winner else "Draw")
        print(f"{label}→ {result} in {move_count} moves  (start: {pos_entry['start_fen'][:24]})")

    return {
        "session_id":        session_id,
        "date":              datetime.now().isoformat(),
        "human_color":       "endgame_play",
        "winner":            engine.winner,
        "move_count":        move_count,
        "white_difficulty":  difficulty,
        "black_difficulty":  difficulty,
        "white_personality": white_personality,
        "black_personality": black_personality,
        "endgame_play":      True,
        "start_fen":         pos_entry["start_fen"],
        "draw_repetition":   draw_by_repetition,
        "moves":             moves_log,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_record(record: dict) -> Path:
    _GAMES_DIR.mkdir(parents=True, exist_ok=True)
    date_str = record.get("date", datetime.now().isoformat())[:10]
    sid      = record.get("session_id", str(uuid.uuid4()))[:8]
    path     = _GAMES_DIR / f"game_{date_str}_{sid}.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Endgame self-play: generate endgame positions and play them out"
    )
    parser.add_argument("--positions",     type=int, default=100,
                        help="Number of endgame positions to play (default: 100)")
    parser.add_argument("--difficulty",    type=int, default=5,
                        help="AI difficulty for both sides (1–10, default: 5)")
    parser.add_argument("--parallel",      type=int, default=1,
                        help="Number of parallel game workers (default: 1)")
    parser.add_argument("--min-pieces",    type=int, default=6,
                        help="Minimum total pieces on board when starting (default: 6)")
    parser.add_argument("--max-pieces",    type=int, default=11,
                        help="Maximum total pieces on board when starting (default: 11)")
    parser.add_argument("--personalities", type=str, default="",
                        help="Comma-separated personality pool for random pairing "
                             "(e.g. balanced,positional,defensive). Default: all except chaos.")
    parser.add_argument("--seed-from-games", action="store_true",
                        help="Seed positions from existing game records in data/games/ "
                             "instead of generating random positions")
    parser.add_argument("--seed",          type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each move to stdout")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    pool = (
        [p.strip() for p in args.personalities.split(",") if p.strip() in _PERSONALITIES]
        if args.personalities else list(_DEFAULT_POOL)
    )
    if not pool:
        print("No valid personalities specified. Using defaults.")
        pool = list(_DEFAULT_POOL)

    min_p = max(6, args.min_pieces)
    max_p = min(11, args.max_pieces)
    if min_p > max_p:
        parser.error(f"--min-pieces ({min_p}) must be ≤ --max-pieces ({max_p})")

    n = args.positions

    print(f"\nEndgame Self-Play")
    print(f"  Positions : {n}")
    print(f"  Pieces    : {min_p}–{max_p}  |  Difficulty: {args.difficulty}")
    print(f"  Workers   : {args.parallel}  |  Pool: {', '.join(pool)}")
    source = "extracted from data/games/" if args.seed_from_games else "randomly generated"
    print(f"  Source    : {source}")
    print()

    # Build position list
    if args.seed_from_games:
        print("Extracting positions from existing game records…")
        positions = _extract_from_games(_GAMES_DIR, max_p, n, rng)
        if not positions:
            print("No endgame positions found in data/games/. "
                  "Run self-play first, or use random mode (omit --seed-from-games).")
            return
        if len(positions) < n:
            print(f"  Only {len(positions)} unique positions found "
                  f"(requested {n}). Proceeding with {len(positions)}.")
            n = len(positions)
    else:
        print(f"Generating {n} random endgame positions…")
        positions = []
        attempts  = 0
        max_total = n * 20
        while len(positions) < n and attempts < max_total:
            p = _gen_random_position(min_p, max_p, rng)
            if p:
                positions.append(p)
            attempts += 1
        if len(positions) < n:
            print(f"  Warning: only {len(positions)} valid positions generated "
                  f"(tried {attempts}). Proceeding.")
            n = len(positions)
    print(f"  Generated {n} starting positions.\n")

    # Build game params
    game_params = []
    for i, pos_entry in enumerate(positions, 1):
        wp = rng.choice(pool)
        bp = rng.choice(pool)
        game_params.append({
            "pos_entry":         pos_entry,
            "difficulty":        args.difficulty,
            "white_personality": wp,
            "black_personality": bp,
            "verbose":           args.verbose,
            "game_num":          i,
        })

    t_start = time.perf_counter()
    wins    = {"W": 0, "B": 0, "D": 0}
    saved   = 0

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool_ex:
            futs = {pool_ex.submit(_play_endgame, p): p["game_num"] for p in game_params}
            for fut, gnum in futs.items():
                try:
                    record = fut.result()
                    _save_record(record)
                    saved += 1
                    w = record["winner"] or "D"
                    wins[w] += 1
                    if not args.verbose:
                        pct = saved / n * 100
                        print(f"\r  {saved}/{n} ({pct:.0f}%)  "
                              f"W:{wins['W']} B:{wins['B']} D:{wins['D']}", end="", flush=True)
                except Exception as exc:
                    print(f"\n  Game {gnum} failed: {exc}")
    else:
        for params in game_params:
            try:
                record = _play_endgame(params)
                _save_record(record)
                saved += 1
                w = record["winner"] or "D"
                wins[w] += 1
                if not args.verbose:
                    pct = saved / n * 100
                    print(f"\r  {saved}/{n} ({pct:.0f}%)  "
                          f"W:{wins['W']} B:{wins['B']} D:{wins['D']}", end="", flush=True)
            except Exception as exc:
                print(f"\n  Game {params['game_num']} failed: {exc}")

    elapsed = time.perf_counter() - t_start
    print(f"\n\nDone — {saved} games saved to data/games/  ({elapsed:.1f}s)")
    print(f"  White wins: {wins['W']}  Black wins: {wins['B']}  Draws: {wins['D']}")
    print(f"  Avg: {elapsed/max(1,saved):.2f}s per game")
    print()
    print("Restart the server to reload EndgameDB with the new positions.")


if __name__ == "__main__":
    main()
