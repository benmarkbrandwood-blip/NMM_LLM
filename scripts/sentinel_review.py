"""scripts/sentinel_review.py — replay games and annotate sentinel turning points.

Loads a trained sentinel checkpoint and replays game files, printing a
move-by-move table with sentinel scores and ASCII board displays at every
position the sentinel flags as a turning point.

Usage:
    # Review all games in a directory (summary only)
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-dir learned_ai/self_play_games

    # Review a single game with full move table
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-file learned_ai/self_play_games/game_2026-06-06_015a34db.jsonl

    # Limit to top-N most flagged games
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-dir learned_ai/self_play_games --top 5

    # Show all moves (not just turning points) for a single game
    python scripts/sentinel_review.py --checkpoint learned_ai/sentinel/checkpoints/best.pt \
        --game-file my_game.jsonl --all-moves
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import BoardState
from learned_ai.sentinel.config import load_config
from learned_ai.sentinel.feature_builder import build_features
from learned_ai.sentinel.infer import SentinelAdvisor, load_advisor

# ANSI colours (disabled when not a tty)
_TTY = sys.stdout.isatty()
_RED    = "\033[91m" if _TTY else ""
_YELLOW = "\033[93m" if _TTY else ""
_GREEN  = "\033[92m" if _TTY else ""
_CYAN   = "\033[96m" if _TTY else ""
_BOLD   = "\033[1m"  if _TTY else ""
_RESET  = "\033[0m"  if _TTY else ""


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def _colour_risk(v: float) -> str:
    if v >= 0.7:
        return f"{_RED}{v:.2f}{_RESET}"
    if v >= 0.5:
        return f"{_YELLOW}{v:.2f}{_RESET}"
    return f"{v:.2f}"


def _colour_tp(v: float) -> str:
    if v >= 0.7:
        return f"{_RED}{_BOLD}{v:.2f}{_RESET}"
    if v >= 0.5:
        return f"{_YELLOW}{v:.2f}{_RESET}"
    return f"{v:.2f}"


def _board_from_fen(fen: str):
    try:
        return BoardState.from_fen_string(fen)
    except Exception:
        return None


def _replay_game(record: dict, advisor: SentinelAdvisor, tp_threshold: float,
                 show_all: bool = False) -> list[dict]:
    """Replay one game record; return list of annotated move dicts."""
    moves = record.get("moves") or []
    results = []
    traj_scores: list[float] = []

    for ply, log_move in enumerate(moves):
        fen = log_move.get("board_fen_before")
        if not fen:
            continue
        board = _board_from_fen(fen)
        if board is None:
            continue

        mv = {"from": log_move.get("from"), "to": log_move.get("to"),
              "capture": log_move.get("capture")}
        ctx = {
            "candidates": [{"move": mv, "score": log_move.get("game_ai_score", 0.0)}],
            "chosen_rank": 0,
            "closes_mill": False,
            "opens_mill_threat": False,
            "reduces_own_mobility": False,
            "trajectory_scores": list(traj_scores[-4:]),
            "game_source": "ai_vs_ai",
            "color": log_move.get("color", board.turn),
            "was_blunder": False,
            "game_ai_score": log_move.get("game_ai_score", 0.0),
        }

        try:
            feats = build_features(board, ctx)
            advice = advisor.advise(board, ctx)
        except Exception:
            continue

        sc = log_move.get("game_ai_score")
        if isinstance(sc, (int, float)):
            traj_scores.append(float(sc))

        is_tp = advice.turning_point_confidence >= tp_threshold
        results.append({
            "ply": ply,
            "color": log_move.get("color", board.turn),
            "notation": log_move.get("notation", f"{mv.get('from','?')}→{mv.get('to','?')}"),
            "type": log_move.get("type", ""),
            "board": board,
            "advice": advice,
            "is_tp": is_tp,
        })

    return results


def _print_game_header(record: dict, game_path: str) -> None:
    winner = record.get("winner") or "Draw"
    wp = record.get("white_personality", "?")
    bp = record.get("black_personality", "?")
    wd = record.get("white_difficulty", "?")
    bd = record.get("black_difficulty", "?")
    mc = record.get("move_count", "?")
    print(f"\n{_BOLD}{'─'*64}{_RESET}")
    print(f"{_BOLD}{os.path.basename(game_path)}{_RESET}")
    print(f"  Winner: {_BOLD}{winner}{_RESET}  |  Moves: {mc}")
    print(f"  White: {wp} (diff {wd})    Black: {bp} (diff {bd})")
    print(f"{'─'*64}")


def _print_move_row(entry: dict, show_board: bool = False) -> None:
    adv = entry["advice"]
    tp_str = _colour_tp(adv.turning_point_confidence)
    risk_str = _colour_risk(adv.mistake_risk)
    flag = f" {_RED}{_BOLD}◀ TURNING POINT{_RESET}" if entry["is_tp"] else ""
    print(
        f"  ply {entry['ply']:>3}  {entry['color']}  {entry['notation']:<14}"
        f"  tp={tp_str} {_bar(adv.turning_point_confidence, 8)}"
        f"  risk={risk_str}"
        f"  opp={adv.opportunity_score:.2f}"
        f"  [{adv.advisory_message}]{flag}"
    )
    if show_board and entry["board"] is not None:
        for line in entry["board"].to_display_grid().splitlines():
            print(f"      {line}")
        print()


def _review_single(record: dict, game_path: str, advisor: SentinelAdvisor,
                   tp_threshold: float, show_all: bool) -> list[dict]:
    entries = _replay_game(record, advisor, tp_threshold, show_all)
    turning_points = [e for e in entries if e["is_tp"]]

    _print_game_header(record, game_path)
    print(f"  {_CYAN}Sentinel turning points: {len(turning_points)} / {len(entries)} plies{_RESET}\n")

    if show_all:
        print(f"  {'ply':>3}  col  {'notation':<14}  {'tp-conf':^14}  risk    opp")
        for e in entries:
            _print_move_row(e, show_board=e["is_tp"])
    else:
        if not turning_points:
            print(f"  {_GREEN}No turning points above threshold {tp_threshold:.2f}{_RESET}")
        else:
            print(f"  {'ply':>3}  col  {'notation':<14}  {'tp-conf':^14}  risk    opp")
            for e in turning_points:
                _print_move_row(e, show_board=True)

    return turning_points


def main() -> int:
    p = argparse.ArgumentParser(description="Replay games with sentinel turning-point annotations")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--game-dir", default=None)
    p.add_argument("--game-file", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Turning-point confidence threshold (default: from config)")
    p.add_argument("--top", type=int, default=None,
                   help="Show only top-N games by turning-point count")
    p.add_argument("--all-moves", action="store_true",
                   help="Print every move (not just turning points)")
    p.add_argument("--limit", type=int, default=None, help="Max game files to scan")
    args = p.parse_args()

    if not args.game_dir and not args.game_file:
        p.error("Provide --game-dir or --game-file")

    config = load_config(args.config)
    tp_threshold = args.threshold if args.threshold is not None else config.turning_point_threshold

    advisor = load_advisor(args.checkpoint, config)
    if advisor is None or not advisor.is_loaded():
        print(f"Failed to load checkpoint: {args.checkpoint}")
        return 1
    print(f"Sentinel loaded  |  turning-point threshold: {tp_threshold:.2f}\n")

    # Collect game files
    if args.game_file:
        paths = [args.game_file]
    else:
        paths = sorted(glob.glob(os.path.join(args.game_dir, "**", "*.jsonl"), recursive=True))
    if args.limit:
        paths = paths[:args.limit]

    # Process games
    game_summaries: list[tuple[str, dict, int]] = []  # (path, record, tp_count)
    for path in paths:
        try:
            with open(path) as f:
                content = f.read().strip()
            record = json.loads(content)
            if not isinstance(record, dict):
                continue
        except Exception:
            continue

        entries = _replay_game(record, advisor, tp_threshold)
        tp_count = sum(1 for e in entries if e["is_tp"])
        game_summaries.append((path, record, tp_count))

    if not game_summaries:
        print("No games found.")
        return 1

    # Single file: full review
    if args.game_file:
        path, record, _ = game_summaries[0]
        _review_single(record, path, advisor, tp_threshold, args.all_moves)
        return 0

    # Directory: sort by tp_count, optionally limit to top-N
    game_summaries.sort(key=lambda x: x[2], reverse=True)
    if args.top:
        to_show = game_summaries[:args.top]
    else:
        to_show = game_summaries

    # Print summary table
    total_games = len(game_summaries)
    total_tps = sum(s[2] for s in game_summaries)
    games_with_tp = sum(1 for s in game_summaries if s[2] > 0)
    print(f"{'─'*64}")
    print(f"Games scanned: {total_games}  |  "
          f"Games with turning points: {games_with_tp}  |  "
          f"Total TPs flagged: {total_tps}")
    print(f"{'─'*64}")
    print(f"  {'Game file':<42}  {'winner':<6}  TPs")
    print(f"  {'─'*42}  {'─'*6}  ───")
    for path, record, tp_count in game_summaries[:40]:
        winner = (record.get("winner") or "draw")[:6]
        tp_col = f"{_RED}{tp_count:>3}{_RESET}" if tp_count >= 3 else f"{tp_count:>3}"
        print(f"  {os.path.basename(path):<42}  {winner:<6}  {tp_col}")
    if len(game_summaries) > 40:
        print(f"  ... ({len(game_summaries) - 40} more games)")

    # Detailed review for selected games
    if args.top or args.all_moves:
        for path, record, tp_count in to_show:
            _review_single(record, path, advisor, tp_threshold, args.all_moves)

    print(f"\n{_CYAN}Tip: use --game-file <path> to review a specific game in detail,")
    print(f"or --top 5 to see the 5 games with the most turning points.{_RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
