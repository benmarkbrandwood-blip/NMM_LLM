"""scripts/train_stage2.py — Stage 2: A2C self-play with GNN model.

Replaces the REINFORCE approach (v1/v2/v3) with Actor-Critic (A2C) to solve
the high-variance gradient problem in long-horizon NMM games.

Three root-cause bug fixes vs REINFORCE v3:
  1. win_reward = 1.0 — matches Stage 0's [-1,+1] value-head training range.
     (v3 used 2.0, causing the value head to oscillate.)
  2. temperature_start = 0.2, annealed to 0.6 — preserves the Stage 1
     imitation prior. (v3 used 0.5 flat, too noisy for a pre-trained model.)
  3. lr = 5e-6 — safe fine-tuning rate for a Stage 1 checkpoint.
     (v3 used 1e-4, which destroyed the prior within the first 50 games.)

Algorithm: A2C (default) or PPO (--ppo flag).
  A2C: per-step TD bootstrapping, advantage = r + γV(s') - V(s).
  PPO: same collection, clipped surrogate, 4 epochs per batch.

Malom shaping:
  - Move quality: r += malom_weight * Δ(WDL) for each learner move.
  - Trap reward:  r += malom_weight when opponent's post-move position = "L".
  malom_weight=0.1 (down from 0.3 in REINFORCE v3) to prevent accumulated
  shaping from swamping the ±1.0 terminal signal in A2C's per-step gradients.

Curriculum: diff 2 (no vn_blend) → diff 3 when rolling-200 win rate ≥ 60%.
Exit: rolling-200 ≥ 60% at diff 3.

Usage:
    .venv/bin/python scripts/train_stage2.py [--resume CKPT] [--out-dir DIR]
                                             [--ppo] [--no-malom] [--no-gnn]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
import learned_ai.agents.heuristic_agent as _ha_mod
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.models.action_encoder import (
    CAPTURE_OFFSET,
    PLACE_OFFSET,
    decode_action,
    get_legal_mask,
    move_requires_capture,
    ACTION_DIM,
)
from learned_ai.models.gnn_backbone import NMMGNNNet
from learned_ai.models.backbone import NMMNet
from learned_ai.models.state_encoder import encode_state_with_phase, PHASE_NAMES
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.a2c import A2CStep, a2c_update
from learned_ai.training.ppo import ppo_update

# ── Defaults (bug-fixed vs REINFORCE v3) ─────────────────────────────────────

LR              = 5e-6         # Bug fix 3: safe fine-tune LR (was 1e-4)
GAMMA           = 0.99
TEMP_START      = 0.2          # Bug fix 2: low start temperature (was 0.5)
TEMP_END        = 0.6          # annealed upward as training progresses
ENTROPY_COEF    = 0.01
UPDATE_EVERY    = 16
MIN_BATCH       = 8

WIN_REWARD      = 1.0          # Bug fix 1: matches Stage 0 [-1,+1] scale (was 2.0)
MALOM_WEIGHT    = 0.1          # Lower than REINFORCE v3 (0.3) — A2C per-step grads
                               # amplify shaping; keep budget below ±1 terminal
MALOM_FRAC      = 0.50         # fraction of max_games with Malom shaping active

ROLLING_WINDOW  = 200
WIN_RATE_TARGET = 0.60
DIFF_START      = 2
DIFF_TARGET     = 3
MAX_PLIES       = 400
TIME_BUDGET     = 0.05         # seconds per opponent move

DEFAULT_MALOM_DB = (
    "/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_opponent(difficulty: int) -> HeuristicAgent:
    inner = _ha_mod.GameAI(color="B", difficulty=difficulty,
                           override_time_budget=TIME_BUDGET)
    return HeuristicAgent(color="B", difficulty=difficulty, game_ai=inner)


def _sample_action(
    model: torch.nn.Module,
    state: torch.Tensor,
    phase_id: int,
    legal_mask: torch.Tensor,
    board: BoardState,
    device: torch.device,
    temperature: float,
) -> tuple[int, Optional[int], float, dict]:
    """Forward the model, sample primary + capture index, return (primary_idx,
    capture_idx_or_None, log_prob_detached, move_dict)."""
    state_d  = state.unsqueeze(0).to(device)
    mask_d   = legal_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        out     = model.forward(state_d, phase_id, mask_d)
        logits  = out["logits"].squeeze(0)

    # Primary action — use mask_d (already on device) for masking
    mask_1d    = mask_d.squeeze(0)
    pri_logits = logits[PLACE_OFFSET:CAPTURE_OFFSET]
    pri_mask   = mask_1d[PLACE_OFFSET:CAPTURE_OFFSET]
    scaled     = pri_logits / max(temperature, 1e-6)
    scaled     = scaled.masked_fill(~pri_mask, float("-inf"))
    log_probs  = F.log_softmax(scaled, dim=-1)
    probs      = log_probs.exp()
    pri_rel    = int(torch.multinomial(probs, 1).item())
    pri_idx    = PLACE_OFFSET + pri_rel
    log_prob   = float(log_probs[pri_rel].item())

    # Capture if needed
    cap_idx: Optional[int] = None
    if move_requires_capture(board, pri_idx):
        cap_logits = logits[CAPTURE_OFFSET:]
        cap_mask   = mask_1d[CAPTURE_OFFSET:]
        if cap_mask.any():
            c_scaled  = cap_logits / max(temperature, 1e-6)
            c_scaled  = c_scaled.masked_fill(~cap_mask, float("-inf"))
            c_lp      = F.log_softmax(c_scaled, dim=-1)
            c_probs   = c_lp.exp()
            c_rel     = int(torch.multinomial(c_probs, 1).item())
            cap_idx   = CAPTURE_OFFSET + c_rel
        else:
            # Fallback: first legal capture
            from game.board import POSITIONS
            from learned_ai.models.action_encoder import POS_INDEX
            first = board.legal_captures(board.turn)[0]
            cap_idx = CAPTURE_OFFSET + POS_INDEX[first]

    move = decode_action(pri_idx, board, capture_index=cap_idx)
    return pri_idx, cap_idx, log_prob, move


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    model: torch.nn.Module,
    learner_color: str,
    opponent: HeuristicAgent,
    malom_db: Optional[ExternalSolvedDB],
    device: torch.device,
    use_malom: bool,
    temperature: float,
    win_reward: float = WIN_REWARD,
    malom_weight: float = MALOM_WEIGHT,
    gamma: float = GAMMA,
    max_plies: int = MAX_PLIES,
) -> tuple[Optional[str], list[A2CStep]]:
    """Play one game and return (winner, list_of_A2CSteps).

    Each A2CStep corresponds to one learner turn. next_state is the board at the
    *next learner turn* (after opponent responds), or a dummy tensor when done.
    """
    board = BoardState.new_game()
    steps: list[A2CStep] = []
    winner: Optional[str] = None
    plies = 0
    opp_moves = 0

    # Pending info from the most recent learner move (waiting for opponent response)
    pending: Optional[tuple] = None  # (state, phase_id, pri_idx, legal_mask, malom_r, log_prob)

    while plies < max_plies:
        terminal, winner = is_terminal(board)
        if terminal:
            break
        legal = get_all_legal_moves(board)
        if not legal:
            winner = "B" if board.turn == "W" else "W"
            break

        if board.turn == learner_color:
            # Flush any open pending step: this means the game got to learner's
            # next turn (previous opponent move didn't end the game).
            if pending is not None:
                s, ph, ai, lm, mr, lp = pending
                ns, nph = encode_state_with_phase(board)
                nlm     = get_legal_mask(board)
                steps.append(A2CStep(s, ph, ai, lm, mr, ns, nph, nlm, False, lp))
                pending = None

            state, phase_id = encode_state_with_phase(board)
            legal_mask = get_legal_mask(board)

            pri_idx, cap_idx, log_prob, move = _sample_action(
                model, state, phase_id, legal_mask, board, device, temperature
            )

            # Malom signal 1: move quality
            malom_r = 0.0
            if use_malom and malom_db and malom_db.is_available():
                try:
                    q = malom_db.query_move_quality(board, move)
                    if q is not None:
                        malom_r += malom_weight * float(q)
                except Exception:
                    pass

            board = board.apply_move(move)
            plies += 1

            # Malom signal 2: trap reward
            if use_malom and malom_db and malom_db.is_available():
                try:
                    q_trap = malom_db.query(board)
                    if q_trap == "L":
                        malom_r += malom_weight
                except Exception:
                    pass

            # Check terminal after learner move
            terminal, winner = is_terminal(board)
            if not terminal:
                post_legal = get_all_legal_moves(board)
                if not post_legal:
                    winner = learner_color
                    terminal = True

            if terminal:
                r_term = _terminal_r(winner, learner_color, win_reward)
                dummy = torch.zeros(state.shape)
                steps.append(A2CStep(state, phase_id, pri_idx, legal_mask,
                                     malom_r + r_term, dummy, 0, legal_mask, True, log_prob))
                pending = None
                break

            # Game continues — save pending (opponent yet to respond)
            pending = (state, phase_id, pri_idx, legal_mask, malom_r, log_prob)

        else:
            # Opponent's turn
            if opp_moves == 0:
                move = random.choice(legal)  # random first move for variety
            else:
                move = opponent.choose_move(board)
            opp_moves += 1

            if not move:
                winner = learner_color
                if pending is not None:
                    s, ph, ai, lm, mr, lp = pending
                    dummy = torch.zeros(s.shape)
                    steps.append(A2CStep(s, ph, ai, lm, mr + WIN_REWARD,
                                         dummy, 0, lm, True, lp))
                    pending = None
                break

            board = board.apply_move(move)
            plies += 1

            # Check terminal after opponent
            terminal, winner = is_terminal(board)
            if not terminal:
                post_legal = get_all_legal_moves(board)
                if not post_legal:
                    if board.turn == learner_color:
                        winner = "B" if learner_color == "W" else "W"
                    else:
                        winner = learner_color
                    terminal = True

            if terminal and pending is not None:
                s, ph, ai, lm, mr, lp = pending
                r_term = _terminal_r(winner, learner_color, win_reward)
                dummy = torch.zeros(s.shape)
                steps.append(A2CStep(s, ph, ai, lm, mr + r_term,
                                     dummy, 0, lm, True, lp))
                pending = None
                break

    else:
        # Ply cap — treat as draw
        winner = None
        if pending is not None:
            s, ph, ai, lm, mr, lp = pending
            dummy = torch.zeros(s.shape)
            steps.append(A2CStep(s, ph, ai, lm, mr, dummy, 0, lm, True, lp))

    return winner, steps


def _terminal_r(winner: Optional[str], learner_color: str, win_reward: float) -> float:
    if winner is None:
        return 0.0
    return win_reward if winner == learner_color else -win_reward


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(description="Stage 2: A2C/PPO self-play (GNN)")
    pa.add_argument("--resume",     default=str(_ROOT / "learned_ai/checkpoints/stage1/best.pt"))
    pa.add_argument("--out-dir",    default=str(_ROOT / "learned_ai/checkpoints/stage2"))
    pa.add_argument("--malom-db",   default=DEFAULT_MALOM_DB)
    pa.add_argument("--max-games",  type=int,   default=10_000)
    pa.add_argument("--ppo",        action="store_true", help="Use PPO instead of A2C")
    pa.add_argument("--no-malom",   action="store_true")
    pa.add_argument("--no-gnn",     action="store_true", help="Use MLP backbone (NMMNet)")
    pa.add_argument("--lr",         type=float, default=LR)
    pa.add_argument("--temp-start", type=float, default=TEMP_START)
    pa.add_argument("--temp-end",   type=float, default=TEMP_END)
    pa.add_argument("--win-reward", type=float, default=WIN_REWARD)
    pa.add_argument("--malom-weight", type=float, default=MALOM_WEIGHT)
    pa.add_argument("--diff-start", type=int,   default=DIFF_START)
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    malom_games = int(args.max_games * MALOM_FRAC)

    # ── Model ─────────────────────────────────────────────────────────────────
    ModelClass = NMMNet if args.no_gnn else NMMGNNNet
    model_type = "mlp" if args.no_gnn else "gnn"

    resume = Path(args.resume)
    if resume.exists():
        ckpt = torch.load(str(resume), map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Load into matching architecture
        ckpt_type = ckpt.get("model_type", "mlp") if isinstance(ckpt, dict) else "mlp"
        if ckpt_type != model_type:
            print(f"WARNING: checkpoint model_type={ckpt_type!r} but requested {model_type!r}")
        model = NMMGNNNet() if model_type == "gnn" else NMMNet()
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys (init from scratch): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")
        print(f"Resumed from {resume}  (model_type={model_type})")
    else:
        print(f"WARNING: no checkpoint at {resume} — using random weights")
        model = NMMGNNNet() if model_type == "gnn" else NMMNet()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Malom DB ──────────────────────────────────────────────────────────────
    malom_db: Optional[ExternalSolvedDB] = None
    if not args.no_malom:
        malom_db = ExternalSolvedDB(db_path=args.malom_db)
        if malom_db.is_available():
            print(f"Malom DB ready  (shaping for games 0–{malom_games})")
        else:
            print("Malom DB unavailable — no reward shaping")

    # ── Curriculum ────────────────────────────────────────────────────────────
    algo    = "PPO" if args.ppo else "A2C"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_diff = args.diff_start
    opponent     = make_opponent(current_diff)
    results: deque[str] = deque(maxlen=ROLLING_WINDOW)
    accumulated: list[A2CStep] = []
    best_win_rate = 0.0

    print(f"\nStage 2 {algo}  lr={args.lr}  γ={GAMMA}  "
          f"T={args.temp_start}→{args.temp_end}  win_reward={args.win_reward}"
          f"  malom_weight={args.malom_weight}  model={model_type}")
    print(f"  update_every={UPDATE_EVERY}  min_batch={MIN_BATCH}")
    print(f"  diff {current_diff}→{DIFF_TARGET}  "
          f"exit: {WIN_RATE_TARGET:.0%} rolling-{ROLLING_WINDOW}\n")

    t0 = time.time()
    for game in range(args.max_games):
        learner_color = "W" if game % 2 == 0 else "B"

        # Anneal temperature linearly over all games
        frac        = min(1.0, game / max(args.max_games - 1, 1))
        temperature = args.temp_start + frac * (args.temp_end - args.temp_start)

        use_malom = (malom_db is not None and malom_db.is_available()
                     and game < malom_games)

        winner, steps = run_episode(
            model, learner_color, opponent, malom_db, device,
            use_malom=use_malom,
            temperature=temperature,
            win_reward=args.win_reward,
            malom_weight=args.malom_weight,
        )

        accumulated.extend(steps)
        r_str = "D" if winner is None else ("W" if winner == learner_color else "L")
        results.append(r_str)

        n_res    = len(results)
        win_rate = results.count("W") / n_res
        elapsed  = time.time() - t0
        malom_tag = "[M]" if use_malom else "   "

        print(f"  game {game+1:5d}  {r_str}  diff={current_diff}  "
              f"wr={win_rate:.1%} ({n_res:3d})  "
              f"T={temperature:.2f}  steps={len(steps):3d}  "
              f"{malom_tag}  t={elapsed:.0f}s")

        # ── A2C / PPO update ──────────────────────────────────────────────────
        if (game + 1) % UPDATE_EVERY == 0 and accumulated:
            batch_size = len(accumulated)
            if args.ppo:
                pl, vl, ent = ppo_update(model, optimizer, accumulated, device)
            else:
                pl, vl, ent = a2c_update(model, optimizer, accumulated, device)
            accumulated.clear()
            if pl != 0.0:
                print(f"    → update  policy_loss={pl:.4f}  value_loss={vl:.4f}"
                      f"  entropy={ent:.4f}  batch={batch_size}")

        # ── Best checkpoint ────────────────────────────────────────────────
        if n_res >= 50 and win_rate > best_win_rate:
            best_win_rate = win_rate
            torch.save({"model": model.state_dict(),
                        "model_type": model_type}, out_dir / "best.pt")

        # ── Curriculum bump ────────────────────────────────────────────────
        if (current_diff < DIFF_TARGET
                and n_res >= ROLLING_WINDOW
                and win_rate >= WIN_RATE_TARGET):
            current_diff += 1
            opponent = make_opponent(current_diff)
            results.clear()
            print(f"\n  ★ difficulty → {current_diff}\n")

        # ── Exit criterion ─────────────────────────────────────────────────
        if (current_diff >= DIFF_TARGET
                and n_res >= ROLLING_WINDOW
                and win_rate >= WIN_RATE_TARGET):
            print(f"\n  ★ EXIT: {win_rate:.1%} at diff {current_diff} (game {game+1})")
            break

    torch.save({"model": model.state_dict(), "model_type": model_type},
               out_dir / "latest.pt")
    n_res    = len(results)
    win_rate = results.count("W") / n_res if n_res else 0.0
    print(f"\nStage 2 done.  win_rate={win_rate:.1%}  best={best_win_rate:.1%}")
    print(f"Checkpoints → {out_dir}")


if __name__ == "__main__":
    main()
