"""
main.py — Console entry point for Nine Men's Morris.

Usage:
    python main.py                        # Human (W) vs AI difficulty 3
    python main.py --difficulty 5         # harder AI
    python main.py --human B              # Human plays Black, AI plays White
    python main.py --blunder 0.3          # AI blunders ~30% of moves (training mode)
    python main.py --hvh                  # Human vs Human
"""

from __future__ import annotations

import argparse

from game.board import BOARD_REFERENCE
from game.game_engine import GameEngine
from game.rules import get_all_legal_moves, get_game_phase
from ai.game_ai import GameAI


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _prompt_placement(engine: GameEngine) -> dict:
    board = engine.board
    color = board.turn
    legal = board.legal_placements(color)
    while True:
        raw = input("  Place piece at (e.g. d2): ").strip().lower()
        if raw not in legal:
            print(f"  ! '{raw}' is not a legal placement. Legal: {sorted(legal)}")
            continue
        return {"from": None, "to": raw, "capture": None}


def _prompt_movement(engine: GameEngine) -> dict:
    board = engine.board
    color = board.turn
    phase = get_game_phase(board, color)
    legal_pairs = set(board.legal_moves(color))
    legal_srcs = sorted({s for s, _ in legal_pairs})
    while True:
        raw = input(
            f"  {'Fly' if phase == 'fly' else 'Move'} piece (e.g. c5-c4): "
        ).strip().lower()
        if "-" not in raw:
            print("  ! Format must be src-dst, e.g. c5-c4")
            continue
        src, dst = raw.split("-", 1)
        if (src, dst) not in legal_pairs:
            print(f"  ! '{raw}' is not a legal move. Legal sources: {legal_srcs}")
            continue
        return {"from": src, "to": dst, "capture": None}


def _prompt_capture(engine: GameEngine) -> str:
    board = engine.board
    color = board.turn
    legal = board.legal_captures(color)
    print(f"  Mill formed! Legal captures: {sorted(legal)}")
    while True:
        raw = input("  Capture: ").strip().lower()
        if raw not in legal:
            print(f"  ! '{raw}' is not a legal capture.")
            continue
        return raw


# ── Game loop ─────────────────────────────────────────────────────────────────

def run_game(
    human_color: str = "W",
    difficulty: int = 3,
    blunder_probability: float = 0.0,
    vs_human: bool = False,
) -> None:
    ai_color = "B" if human_color == "W" else "W"
    engine = GameEngine(human_color=human_color)

    ai: GameAI | None = None
    if not vs_human:
        ai = GameAI(
            color=ai_color,
            difficulty=difficulty,
            blunder_probability=blunder_probability,
        )

    print("\n═══ Nine Men's Morris ═══\n")
    print(BOARD_REFERENCE)
    if vs_human:
        print("\nHuman vs Human")
    else:
        mode = f"difficulty {difficulty}"
        if blunder_probability > 0:
            mode += f", blunder rate {blunder_probability:.0%}"
        print(f"\nYou are {'White (W)' if human_color == 'W' else 'Black (B)'}  |  AI: {mode}")
    print()

    while not engine.finished:
        board = engine.board
        color = board.turn
        phase = get_game_phase(board, color)
        name = "White" if color == "W" else "Black"
        is_human_turn = vs_human or (color == human_color)

        print(engine.status_line())
        print(board.to_display_grid())

        if is_human_turn:
            print(f"\n{name}'s turn [{phase}]")
            if phase == "place":
                move = _prompt_placement(engine)
            else:
                move = _prompt_movement(engine)
            if engine.move_forms_mill(move):
                cap = _prompt_capture(engine)
                move["capture"] = cap
        else:
            assert ai is not None
            print(f"\nAI ({name}) thinking... [{phase}]")
            move = ai.choose_move(board)
            move_str = (
                f"{move.get('from')}-{move['to']}"
                if move.get("from")
                else move["to"]
            )
            if move.get("capture"):
                move_str += f"x{move['capture']}"
            if ai.last_was_blunder:
                print(f"  AI plays: {move_str}  ← deliberate mistake!")
            else:
                print(f"  AI plays: {move_str}")

        engine.apply_move(move)
        print()

    print(engine.board.to_display_grid())
    winner_name = "White" if engine.winner == "W" else "Black"
    print(f"\n{'═' * 40}")
    print(f"  Game over — {winner_name} wins!")
    print(f"{'═' * 40}\n")
    print(engine.export())


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nine Men's Morris console")
    p.add_argument("--difficulty", "-d", type=int, default=3, choices=range(1, 6),
                   help="AI difficulty 1-5 (default 3)")
    p.add_argument("--human", "-p", default="W", choices=["W", "B"],
                   help="Human plays W or B (default W)")
    p.add_argument("--blunder", "-b", type=float, default=0.0, metavar="PROB",
                   help="AI blunder probability 0.0-1.0 (default 0, training mode)")
    p.add_argument("--hvh", action="store_true",
                   help="Human vs Human (no AI)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_game(
        human_color=args.human,
        difficulty=args.difficulty,
        blunder_probability=args.blunder,
        vs_human=args.hvh,
    )
