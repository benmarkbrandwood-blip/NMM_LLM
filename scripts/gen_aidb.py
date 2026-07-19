"""scripts/gen_aidb.py — generate AI-vs-AI games for sentinel AIDB training.

Each game is annotated with Malom DB win/draw/loss labels per move.
Games are saved to data/ai_games/ as JSONL (one game per file), matching
the format read by SentinelDataset.load_from_games().

Agent configuration pool:
  6 personalities × 2 value-net-blend variants × 2 sentinel variants = 24 configs.
  White and black are drawn independently (with replacement) from the pool.

The first RANDOM_PLIES (4) plies are played randomly so the opening position
varies across games.  After that, each side's GameAI drives the game.

Usage:
    .venv/bin/python scripts/gen_aidb.py --games 5000 \\
        [--db-path "/mnt/windows/NMM_DB/Entire DB"] \\
        [--sentinel learned_ai/sentinel/checkpoints/best.pt] \\
        [--out-dir data/ai_games] [--smoke-test]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from ai.game_ai import GameAI
from ai.heuristics import HeuristicWeights
from ai.value_net import ValueNet
from game.game_engine import GameEngine
from game.rules import terminal_wdl
from learned_ai.sentinel.db_teacher import ExternalSolvedDB

# ── Constants ─────────────────────────────────────────────────────────────────

PERSONALITIES: List[str] = ["balanced", "aggressive", "defensive", "positional", "scholar", "chaos"]
VN_BLENDS: List[int] = [0, 80]
SENTINEL_VARIANTS: List[bool] = [False, True]
RANDOM_PLIES = 4
MAX_PLIES = 400
DIFFICULTY = 5

# Malom outcome strings → label strings (mover's perspective, before-move).
_OUTCOME_MAP: Dict[str, str] = {"W": "win", "L": "loss", "D": "draw"}
# Flip: after applying a move it becomes the opponent's turn, so we negate.
_NEGATE_WDL: Dict[str, str] = {"W": "loss", "L": "win", "D": "draw"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    path = _ROOT / "data" / "settings.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_personality(name: str) -> dict:
    path = _ROOT / "data" / "personalities" / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _build_agent_configs() -> List[dict]:
    """All combinations of personality × vn_blend × sentinel."""
    return [
        {"personality": p, "vn_blend": v, "use_sentinel": s}
        for p in PERSONALITIES
        for v in VN_BLENDS
        for s in SENTINEL_VARIANTS
    ]


def _make_game_ai(
    color: str,
    cfg: dict,
    sentinel_advisor,
    value_net: Optional[ValueNet],
    malom_wrapper: Optional[ExternalSolvedDB],
) -> GameAI:
    """Build a GameAI from an agent config dict."""
    personality = cfg["personality"]
    vn_blend = cfg["vn_blend"]

    d = _load_personality(personality)
    d["value_net_blend"] = vn_blend

    hw = HeuristicWeights(**d)
    ai = GameAI(
        color=color,
        difficulty=DIFFICULTY,
        weights=hw,
        blunder_probability=hw.make_mistakes / 100.0,
        value_net=value_net if vn_blend > 0 else None,
        malom_db=malom_wrapper,
        override_time_budget=0.5,
    )

    if cfg["use_sentinel"] and sentinel_advisor is not None:
        ai.set_sentinel(sentinel_advisor, mode="advisory")

    return ai


def _annotate_move_malom(
    board,
    move: dict,
    malom: Optional["_MalomDB"],
) -> tuple:
    """Return (malom_wdl, malom_dtw, malom_move_wdl) for the pre/post-move position.

    All three are None when the Malom DB is unavailable.
    malom_wdl / malom_dtw: quality of the position BEFORE the move (mover's perspective).
    malom_move_wdl: quality of the move itself (mover's perspective, flip after-move outcome).
    """
    if malom is None:
        return None, None, None

    malom_wdl = None
    malom_dtw = None
    malom_move_wdl = None

    try:
        pre = malom.query(board)
        if pre:
            malom_wdl = _OUTCOME_MAP.get(pre.get("outcome"))
            malom_dtw = pre.get("dtw")
    except Exception:
        pass

    try:
        after_board = board.apply_move(move)
        post_outcome = terminal_wdl(after_board)
        if post_outcome is None:
            post = malom.query(after_board)
            post_outcome = post.get("outcome") if post else None
        malom_move_wdl = _NEGATE_WDL.get(post_outcome)
    except Exception:
        pass

    return malom_wdl, malom_dtw, malom_move_wdl


def _get_malom_direct(malom_wrapper: Optional[ExternalSolvedDB]):
    """Extract the underlying MalomDB instance from an ExternalSolvedDB wrapper."""
    if malom_wrapper is None:
        return None
    return getattr(malom_wrapper, "_malom", None)


# ── Core game loop ─────────────────────────────────────────────────────────────

def play_annotated_game(
    white_cfg: dict,
    black_cfg: dict,
    sentinel_advisor,
    value_net: Optional[ValueNet],
    malom_wrapper: Optional[ExternalSolvedDB],
    random_plies: int = RANDOM_PLIES,
    max_plies: int = MAX_PLIES,
) -> dict:
    """Play one annotated game; return the game record dict."""
    engine = GameEngine()
    white_ai = _make_game_ai("W", white_cfg, sentinel_advisor, value_net, malom_wrapper)
    black_ai = _make_game_ai("B", black_cfg, sentinel_advisor, value_net, malom_wrapper)
    malom = _get_malom_direct(malom_wrapper)

    ply = 0
    while not engine.finished and ply < max_plies:
        board = engine.board
        player = board.turn
        legal = engine.get_all_legal_moves()

        if not legal:
            break

        # Choose move.
        if ply < random_plies:
            move = random.choice(legal)
        else:
            ai = white_ai if player == "W" else black_ai
            try:
                move = ai.choose_move(board, fast_early_game=True)
            except Exception:
                move = None
            if move is None:
                move = random.choice(legal)

        malom_wdl, malom_dtw, malom_move_wdl = _annotate_move_malom(
            board,
            move,
            malom,
        )

        # Apply and amend the game-record entry with Malom data.
        engine.apply_move(move)
        entry = engine.game_record["moves"][-1]
        entry["malom_wdl"] = malom_wdl
        entry["malom_dtw"] = malom_dtw
        entry["malom_move_wdl"] = malom_move_wdl

        ply += 1

    record = engine.game_record
    record["white_personality"] = white_cfg["personality"]
    record["black_personality"] = black_cfg["personality"]
    record["white_vn_blend"] = white_cfg["vn_blend"]
    record["black_vn_blend"] = black_cfg["vn_blend"]
    record["white_sentinel"] = white_cfg["use_sentinel"]
    record["black_sentinel"] = black_cfg["use_sentinel"]

    return record


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    pa = argparse.ArgumentParser(description="Generate AI-vs-AI games for sentinel training")
    pa.add_argument("--games",     type=int, default=5000, help="Number of games to generate")
    pa.add_argument("--out-dir",   default=str(_ROOT / "data" / "ai_games"))
    pa.add_argument("--db-path",   default="", help="Path to Malom DB directory")
    pa.add_argument("--sentinel",  default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"),
                    help="Sentinel checkpoint for advisory mode (omit to disable)")
    pa.add_argument("--value-net", default=str(_ROOT / "data" / "value_net.npz"))
    pa.add_argument("--seed",      type=int, default=0)
    pa.add_argument("--smoke-test", action="store_true",
                    help="Run 2 games quickly and exit (DB annotation optional)")
    args = pa.parse_args()

    if args.smoke_test:
        args.games = 2

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load optional resources ───────────────────────────────────────────────
    settings = _load_settings()

    db_path = args.db_path or settings.get("malom_db_path", "")
    malom_wrapper: Optional[ExternalSolvedDB] = None
    if db_path:
        malom_wrapper = ExternalSolvedDB(db_path)
        if not malom_wrapper.is_available():
            print(f"Warning: Malom DB not available at {db_path!r} — games will have no malom labels")
            malom_wrapper = None
        else:
            print(f"Malom DB: available at {db_path}")
    else:
        print("Malom DB: not configured (set --db-path or malom_db_path in settings.json)")

    value_net: Optional[ValueNet] = None
    vn_path = Path(args.value_net)
    if vn_path.exists():
        value_net = ValueNet.load_if_exists(vn_path)
        if value_net is not None:
            print(f"ValueNet: loaded from {vn_path}")
    else:
        print("ValueNet: not found — vn_blend=80 games will use vn_blend=0 behaviour")

    sentinel_advisor = None
    sentinel_path = Path(args.sentinel)
    if sentinel_path.exists():
        try:
            from learned_ai.sentinel.infer import SentinelAdvisor
            sentinel_advisor = SentinelAdvisor()
            if not sentinel_advisor.load(str(sentinel_path)):
                print(f"Warning: could not load sentinel from {sentinel_path}")
                sentinel_advisor = None
            else:
                print(f"Sentinel: loaded from {sentinel_path}")
        except Exception as exc:
            print(f"Warning: sentinel load failed ({exc}) — sentinel=False games only")
            sentinel_advisor = None
    else:
        print(f"Sentinel: not found at {sentinel_path} — sentinel=False games only")

    # ── Game generation ───────────────────────────────────────────────────────
    agent_configs = _build_agent_configs()
    n_configs = len(agent_configs)
    today = date.today().isoformat()

    malom_count = 0
    print(f"\nGenerating {args.games} games → {out_dir}")

    for game_idx in range(args.games):
        white_cfg = agent_configs[game_idx % n_configs]
        black_cfg = agent_configs[(game_idx + n_configs // 2) % n_configs]

        try:
            record = play_annotated_game(
                white_cfg, black_cfg,
                sentinel_advisor=sentinel_advisor,
                value_net=value_net,
                malom_wrapper=malom_wrapper,
            )
        except Exception as exc:
            print(f"  game {game_idx}: ERROR — {exc}")
            continue

        # Count malom-annotated moves.
        n_moves = len(record.get("moves", []))
        n_malom = sum(1 for m in record.get("moves", []) if m.get("malom_move_wdl") is not None)
        malom_count += n_malom

        # Save as JSONL (one JSON object per file, compatible with _iter_game_records).
        game_id = uuid.uuid4().hex[:8]
        fname = out_dir / f"game_{today}_{game_id}.jsonl"
        fname.write_text(json.dumps(record))

        if (game_idx + 1) % max(1, args.games // 20) == 0 or (game_idx + 1) == args.games:
            print(
                f"  {game_idx + 1}/{args.games}  winner={record.get('winner')}  "
                f"plies={n_moves}  malom={n_malom}/{n_moves}"
            )

    total_moves = sum(len(json.loads(p.read_text()).get("moves", []))
                      for p in out_dir.glob("*.jsonl"))
    print(f"\nDone. {args.games} games, {malom_count} malom-annotated moves total.")
    print(f"Output: {out_dir}  ({len(list(out_dir.glob('*.jsonl')))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
