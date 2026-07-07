"""scripts/train_scaffolded_s2b.py — Stage 2b: self-play with branched mid-game rollouts.

Extends s2_diagnostic with two additions:

1. SELF-PLAY: Half of main games pit the live model (temperature-sampled) against a
   periodically-frozen copy of itself.  The other half use the heuristic opponent from
   s2.  This provides trajectory diversity without the training-vs-inference gap of an
   undo mechanism.

2. BRANCHED ROLLOUTS: During every main game, the board state is snapshotted every
   BRANCH_EVERY learner turns.  After the main game ends, up to MAX_BRANCHES_PER_GAME
   of those snapshots are selected as starting points for fresh independent rollouts
   (model vs frozen copy).  Each branch is recorded as a completely separate trajectory
   — it never shares a gradient batch with the game it was spawned from, so there is no
   gradient contamination for shared positions.

GAME-STAGE DIVERSITY: Branch points are bucketed by phase:
   "opening"  — placement phase OR first 3 moves (6 plies) of movement phase
   "midgame"  — movement phase, > 3 moves past placement, 12+ pieces on board
   "endgame"  — movement phase with fewer than 12 pieces on board

A rolling counter (BUCKET_WINDOW games) caps how many branches can come from any
single bucket (MAX_PER_BUCKET).  This prevents the training set from flooding with
one phase type while ensuring beginning, middle, and end-game positions all appear.

All other mechanics (reward shape, diagnostics, temperature schedule, LR backoff,
checkpoint logic) are identical to s2_diagnostic.

Checkpoints are saved to learned_ai/checkpoints/scaffolded/s2b/ by default.
Resume chain: explicit --resume → s2b/best.pt → s2/best.pt → s1b/best.pt → s1/best.pt

Usage
-----
# Quick smoke test (20 main games, no branches)
.venv/bin/python scripts/train_scaffolded_s2b.py --max-games 20 --max-branches-per-game 0

# Normal run from s2 checkpoint
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-s2

# Full run with PPO update
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-s2 --ppo --max-games 5000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState, MILLS
from game.rules import is_terminal
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.models.scaffolded_encoder import encode_position
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
    scaffolded_ppo_update,
)

# ── Opening book (combined learned + curated) ────────────────────────────────

def _load_opening_book() -> list[list[str]]:
    """Load all line_moves sequences from both opening book files."""
    lines: list[list[str]] = []
    for fname in ("book_openings.json", "learned_openings.json"):
        fpath = _ROOT / "data" / "openings" / fname
        if not fpath.exists():
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                entries = json.load(f)
            for entry in entries:
                moves = entry.get("line_moves", [])
                if isinstance(moves, list) and len(moves) >= 2:
                    lines.append(moves)
        except Exception:
            pass
    return lines

_OPENING_LINES: list[list[str]] = _load_opening_book()
BOOK_GAME_PROB = 0.50   # fraction of games that follow the opening book


def _sample_forced_placements(line_moves: list[str], learner_color: str) -> list[str]:
    """Extract up to 4 placement positions for the learner's side from a line."""
    start = 0 if learner_color == "W" else 1
    return [line_moves[i] for i in range(start, len(line_moves), 2)][:4]


# ── Reward weights (same as s2) ───────────────────────────────────────────────

ALPHA   = 0.15   # sentinel relative score
BETA    = 0.10   # heuristic delta
GAMMA   = 0.25   # malom win quality
DELTA   = 0.15   # malom trap bonus
LAMBDA  = 0.50   # retro-active outcome weight
DECAY   = 0.98   # retro decay per ply remaining
VN_BETA = 0.10   # value-net delta

WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_SHORT  =  0.15   # draw in < 100 plies
DRAW_LONG   = -0.05   # draw by exhaustion
MILL_BONUS  =  0.20   # per new mill closed by the learner

# ── Optimiser / schedule ──────────────────────────────────────────────────────

LR            = 1e-4
GAMMA_TD      = 0.99
TEMP_START    = 0.50
TEMP_MIN      = 0.45
TEMP_MAX      = 0.90
ENTROPY_COEF  = 0.01
UPDATE_EVERY  = 16
ROLLING_WIN   = 200
DIFF_START = 3    # warm-up at diff 3 before tackling diff 4+
DIFF_MAX   = 7
ADVANCE_THRESHOLDS = {3: 0.60, 4: 0.55, 5: 0.50, 6: 0.45}
EXIT_THRESHOLD = 0.50   # win rate vs diff 7 considered done

S1B_REFRESHER_EPOCHS = 3
S1B_REFRESHER_LR     = 3e-4
S1B_REFRESHER_BATCH  = 32
MAX_PLY       = 60
MAX_PLY_BRANCH = 60
TIME_BUDGET   = 0.05

LOG_EVERY     = 50
# LR adaptation: scaled by win rate, never floored hard
LR_SCALE_WIN  = 0.35   # win rate at which LR = LR_BASE (1.0x)
LR_SCALE_MIN  = 0.50   # minimum LR multiplier (win_rate=0 → 0.5x LR_BASE)
LR_SCALE_MAX  = 2.00   # maximum LR multiplier
# Recovery: reload best checkpoint when win rate is very poor
RECOVERY_THRESHOLD  = 0.12
RECOVERY_MIN_GAMES  = 30

# ── s2b-specific knobs ────────────────────────────────────────────────────────

UPDATE_TARGET_EVERY    = 50    # games between frozen-model refreshes
SELF_PLAY_RATIO        = 0.5   # fraction of main games vs frozen model (rest vs heuristic)
BRANCH_EVERY           = 10    # save branch candidate every N learner moves in a main game
MAX_BRANCHES_PER_GAME  = 2     # max branch games spawned per main game
BUCKET_WINDOW          = 300   # rolling window size for saturation tracking
MAX_PER_BUCKET         = 80    # max branch games from any single bucket in that window

# Placement ends when both players have placed all 9 pieces (18 total plies).
# "Opening" extends 3 moves past placement — each move = one turn per player = 2 plies,
# so OPENING_EXTENSION_PLY = 6 total plies after movement phase begins.
OPENING_EXTENSION_PLY  = 6

PHASE_BUCKETS = ("opening", "midgame", "endgame")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    total:      float = 0.0
    sentinel:   float = 0.0
    heuristic:  float = 0.0
    value_net:  float = 0.0
    malom_win:  float = 0.0
    malom_trap: float = 0.0
    mill_formed: float = 0.0
    retro:      float = 0.0


@dataclass
class StepDiag:
    reward:           RewardBreakdown
    legal_moves:      int
    chosen_idx:       int
    chosen_prob:      float
    entropy:          float
    top1_prob:        float
    sentinel_mean:    float
    sentinel_chosen:  float
    h_before:         float
    h_after:          float
    h_delta:          float
    vn_before:        float
    vn_after:         float
    vn_delta:         float
    malom_chosen_wdl: str
    malom_chosen_dtm: Optional[float]
    was_top1_policy:  int
    was_top1_heuristic: int


@dataclass
class GameDiag:
    game:                   int
    difficulty:             int
    learner_color:          str
    temperature:            float
    outcome:                float
    win_rate_200:           float
    ply:                    int
    steps:                  int
    update_policy_loss:     Optional[float]
    update_value_loss:      Optional[float]
    update_entropy:         Optional[float]
    reward_total_mean:      float
    reward_sentinel_mean:   float
    reward_heuristic_mean:  float
    reward_value_mean:      float
    reward_malom_win_mean:  float
    reward_malom_trap_mean: float
    reward_retro_mean:      float
    sentinel_mean:          float
    sentinel_chosen_mean:   float
    h_delta_mean:           float
    vn_delta_mean:          float
    chosen_prob_mean:       float
    entropy_mean:           float
    top1_prob_mean:         float
    legal_moves_mean:       float
    policy_top1_rate:       float
    heuristic_top1_rate:    float
    malom_win_move_rate:    float
    malom_unknown_rate:     float
    best_win_rate:          float
    temp_frozen:            int
    lr:                     float
    source_checkpoint:      str
    # s2b additions
    game_type:              str    # "vs_heuristic" | "vs_frozen" | "branch"
    phase_bucket:           str    # "opening" | "midgame" | "endgame" | "main"
    is_branch:              int    # 0 = main game, 1 = branch
    branch_ply_start:       int    # ply offset where branch was spawned (0 for main)
    target_age:             int    # games since last frozen-model update
    bucket_opening:         int    # current bucket counts in rolling window
    bucket_midgame:         int
    bucket_endgame:         int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _phase_bucket(board: BoardState, moves_into_movement: Optional[int] = None) -> str:
    """Classify board into training phase bucket for saturation tracking.

    moves_into_movement: plies elapsed since movement phase began (None = unknown).
    Opening extends OPENING_EXTENSION_PLY plies past placement end.
    """
    total_on_board = board.pieces_on_board["W"] + board.pieces_on_board["B"]
    if board.phase == "place":
        return "opening"
    # movement / fly phase
    if total_on_board < 12:
        return "endgame"
    if moves_into_movement is not None and moves_into_movement < OPENING_EXTENSION_PLY:
        return "opening"
    return "midgame"


def _run_s1b_refresher(
    model: ScaffoldedPolicyNet,
    device: torch.device,
    data_path: str,
    epochs: int = S1B_REFRESHER_EPOCHS,
    lr: float = S1B_REFRESHER_LR,
    batch: int = S1B_REFRESHER_BATCH,
    deviate_bonus: float = 1.5,
) -> None:
    """Inline s1b human-imitation refresher. Modifies model in-place.

    Freezes the value head during fine-tuning then restores requires_grad
    so A2C training continues to update it afterwards.
    """
    p = Path(data_path)
    if not p.exists():
        print(f"[s2b] s1b refresher: data not found ({data_path}) — skipping")
        return

    data          = np.load(str(p), allow_pickle=True)
    feat_matrices = data["feat_matrices"]
    label_dists   = data["label_dists"]
    h_top1_idxs   = data["h_top1_idxs"]
    weights       = data["weights"]
    deviates      = data["deviates"]
    # is_winner: True = human's moves in won games; False = loser moves + draw moves
    is_winner     = data["is_winner"] if "is_winner" in data else np.ones(len(weights), dtype=bool)
    N             = len(weights)

    effective_weights = weights.copy()
    bonus_mask        = (is_winner) & deviates
    effective_weights[bonus_mask] *= deviate_bonus

    loser_idxs  = [i for i in range(N) if not is_winner[i]]
    winner_idxs = [i for i in range(N) if is_winner[i]]

    # Freeze value head
    for param in model.value_mlp.parameters():
        param.requires_grad = False

    opt_s1b = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()), lr=lr
    )

    model.train()
    print(f"[s2b] s1b refresher: loser={len(loser_idxs)} winner={len(winner_idxs)} positions  lr={lr:.2e}")

    def _run_phase(phase_idxs: list[int], phase_label: str, use_heuristic_target: bool) -> None:
        if not phase_idxs:
            return
        for epoch in range(1, epochs + 1):
            random.shuffle(phase_idxs)
            ep_loss  = 0.0
            ep_w_sum = 0.0
            for b_start in range(0, len(phase_idxs), batch):
                b = phase_idxs[b_start : b_start + batch]
                if not b:
                    continue
                terms    = []
                bweights = []
                for i in b:
                    feat = torch.tensor(feat_matrices[i], dtype=torch.float32).to(device)
                    if use_heuristic_target:
                        k     = feat.shape[0]
                        h_idx = int(h_top1_idxs[i])
                        tgt   = np.full(k, 0.05 / max(k - 1, 1), dtype=np.float32)
                        if 0 <= h_idx < k:
                            tgt[h_idx] = 0.95
                        else:
                            tgt[:] = 1.0 / k
                        target = torch.tensor(tgt, dtype=torch.float32).to(device)
                    else:
                        target = torch.tensor(label_dists[i], dtype=torch.float32).to(device)
                    logits = model.policy_logits(feat)
                    log_p  = F.log_softmax(logits, dim=-1)
                    terms.append(-(target * log_p).sum())
                    bweights.append(float(effective_weights[i]))
                w_t  = torch.tensor(bweights, dtype=torch.float32).to(device)
                loss = (w_t * torch.stack(terms)).sum() / w_t.sum().clamp(min=1e-9)
                opt_s1b.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_s1b.step()
                ep_loss  += float(loss.item()) * float(w_t.sum())
                ep_w_sum += float(w_t.sum())
            print(f"[s2b]   refresher [{phase_label}] epoch {epoch}/{epochs}  loss={ep_loss / max(ep_w_sum, 1e-9):.4f}")

    # Phase 1: loser positions — teach model to correct losing-side mistakes
    _run_phase(loser_idxs, "loser→heuristic", use_heuristic_target=True)
    # Phase 2: winner positions — reinforce winning human behaviour
    _run_phase(winner_idxs, "winner", use_heuristic_target=False)

    # Unfreeze value head
    for param in model.value_mlp.parameters():
        param.requires_grad = True

    model.eval()
    print("[s2b] s1b refresher done")


def _choose_resume_path(args: argparse.Namespace) -> tuple[Optional[Path], str]:
    if args.resume:
        p = Path(args.resume)
        if p.exists():
            return p, "explicit_resume"
    s2b_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b" / "best.pt"
    s2b_latest = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b" / "latest.pt"
    s2_best    = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2"  / "best.pt"
    s1b_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1b" / "best.pt"
    s1_best    = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1"  / "best.pt"
    candidates = []
    if args.auto_resume_best:
        candidates.append((s2b_best,   "s2b_best"))
    if args.auto_resume_latest:
        candidates.append((s2b_latest, "s2b_latest"))
    if args.auto_resume_s2:
        candidates.append((s2_best,    "s2_best"))
    candidates += [(s1b_best, "s1b_best"), (s1_best, "s1_best")]
    for p, tag in candidates:
        if p.exists():
            return p, tag
    return None, "scratch"


def _load_model(device: torch.device, resume_path: Optional[Path], force_start_diff: bool = False) -> tuple[ScaffoldedPolicyNet, int, float, int, str]:
    if resume_path is None:
        return ScaffoldedPolicyNet().to(device), 0, 0.0, DIFF_START, "scratch"
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = ScaffoldedPolicyNet.from_config(cfg).to(device)
    sd_key = "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    stage       = ckpt.get("stage", "unknown")
    is_s2b      = stage == "s2b"
    start_game  = int(ckpt.get("game_count", 0)) if is_s2b else 0
    best_wr     = float(ckpt.get("best_win_rate", 0.0)) if is_s2b else 0.0
    difficulty  = int(ckpt.get("difficulty", DIFF_START)) if is_s2b else DIFF_START
    if force_start_diff:
        difficulty = max(difficulty, DIFF_START)
    return model, start_game, best_wr, difficulty, str(resume_path)


def _apply_diff_start_override(difficulty: int, args: argparse.Namespace) -> int:
    if args.diff_start is not None:
        return max(1, min(args.diff_start, DIFF_MAX))
    return difficulty


def _compute_temperature(game_count: int, max_games: int) -> float:
    """Linear anneal from TEMP_START to TEMP_MAX over the first 80% of training."""
    progress = min(1.0, game_count / max(max_games * 0.8, 1))
    return float(TEMP_START + (TEMP_MAX - TEMP_START) * progress)



def _adapt_lr(opt: torch.optim.Optimizer, win_rate: float, lr_base: float) -> None:
    """Scale LR proportionally to win rate — higher win rate → higher LR."""
    scale  = max(LR_SCALE_MIN, min(LR_SCALE_MAX, win_rate / LR_SCALE_WIN))
    new_lr = lr_base * scale
    for g in opt.param_groups:
        g["lr"] = new_lr


def _compute_per_move_reward(enc, chosen_idx: int, enc_after, db_moves=None) -> tuple[float, RewardBreakdown, dict[str, Any]]:
    rb = RewardBreakdown()
    extra: dict[str, Any] = {"malom_chosen_wdl": "unknown", "malom_chosen_dtm": None}

    if getattr(enc, "sentinel_scores", None):
        mean_s   = float(sum(enc.sentinel_scores) / len(enc.sentinel_scores))
        played_s = float(enc.sentinel_scores[chosen_idx])
        rb.sentinel = ALPHA * (played_s - mean_s)

    if enc_after is not None:
        h_before = float(getattr(enc, "h_before", 0.0))
        h_after  = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
        rb.heuristic = BETA * math.tanh(h_after - h_before)

    if getattr(enc, "vn_scores_abs", None):
        vn_before = float(getattr(enc, "vn_before", 0.0))
        vn_after  = float(enc.vn_scores_abs[chosen_idx])
        rb.value_net = VN_BETA * math.tanh(vn_after - vn_before)

    if db_moves:
        mv_key   = _move_key(enc.legal_moves[chosen_idx])
        db_entry = next((m for m in db_moves if _move_key(m.get("move", {})) == mv_key), None)
        if db_entry:
            wdl = str(db_entry.get("wdl", "unknown"))
            dtm = db_entry.get("dtm")
            extra["malom_chosen_wdl"] = wdl
            extra["malom_chosen_dtm"] = dtm
            if wdl == "win":
                rb.malom_win = GAMMA * float(dtm_quality("win", dtm))

    rb.total = rb.sentinel + rb.heuristic + rb.value_net + rb.malom_win + rb.malom_trap + rb.retro
    return float(rb.total), rb, extra


def _retroactive_rescore(trajectory: list[ScaffoldedStep], step_diags: list[StepDiag], outcome: float) -> None:
    n = len(trajectory)
    for t_idx, step in enumerate(trajectory):
        plies_remaining  = n - t_idx - 1
        delta            = LAMBDA * outcome * (DECAY ** plies_remaining)
        step.reward     += delta
        step_diags[t_idx].reward.retro += float(delta)
        step_diags[t_idx].reward.total += float(delta)


# ── Frozen-model opponent ─────────────────────────────────────────────────────

class FrozenModelOpponent:
    """Plays argmax from a deep-copied, frozen snapshot of the live model."""

    def __init__(self, model: ScaffoldedPolicyNet, device: torch.device, sentinel=None, value_net=None):
        self._model     = copy.deepcopy(model).to(device)
        self._model.eval()
        self._device    = device
        self._sentinel  = sentinel
        self._value_net = value_net
        self.last_was_blunder = False
        self.last_thinking    = "frozen"

    def refresh(self, model: ScaffoldedPolicyNet) -> None:
        self._model.load_state_dict(copy.deepcopy(model).state_dict())
        self._model.eval()

    def choose_move(self, board: BoardState) -> dict:
        player = board.turn
        enc = encode_position(board, player,
                              sentinel_advisor=self._sentinel,
                              db=None,
                              value_net=self._value_net)
        if enc is None or not enc.legal_moves:
            return {}
        feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model.policy_logits(feat_t)
            idx    = int(torch.argmax(logits).item())
        return enc.legal_moves[idx]


# ── Single-game rollout (shared by main and branch games) ─────────────────────

RETRY_PLY_MIN  =  5   # random retry ply range (inclusive)
RETRY_PLY_MAX  = 15

@dataclass
class RolloutResult:
    trajectory: list[ScaffoldedStep]
    step_diags: list[StepDiag]
    outcome:    float
    ply:        int
    branch_candidates: list[tuple[int, BoardState, str]]  # (ply, board, phase_bucket)
    retry_board: Optional[BoardState] = None   # board at random ply RETRY_PLY_MIN..RETRY_PLY_MAX


def _rollout(
    model:         ScaffoldedPolicyNet,
    device:        torch.device,
    start_board:   BoardState,
    learner_color: str,
    opponent,               # HeuristicAgent | FrozenModelOpponent
    opp_color:     str,
    sentinel,
    db,
    value_net,
    temperature:   float,
    max_ply:       int,
    record_branches: bool,
    branch_every:  int,
    retry_ply:     int,
    forced_placements: Optional[list[str]] = None,
) -> RolloutResult:
    """
    Run a single game rollout from start_board.

    record_branches — if True, snapshot (ply, board, bucket) every branch_every
    learner turns for later branch-game spawning.
    forced_placements — if set, override learner's first N placement moves with
    these position strings (from the opening book).
    """
    board                  = start_board
    ply                    = 0
    move_phase_start_ply:  Optional[int] = None
    game_trajectory:       list[ScaffoldedStep] = []
    step_diags:            list[StepDiag]       = []
    branch_candidates:     list[tuple[int, BoardState, str]] = []
    done                   = False
    outcome                = 0.0
    learner_move_count     = 0
    learner_placement_count = 0
    retry_board: Optional[BoardState] = None

    while ply < max_ply:
        if ply == retry_ply:
            retry_board = board
        if board.phase != "place" and move_phase_start_ply is None:
            move_phase_start_ply = ply
        terminal, winner = is_terminal(board)
        if terminal:
            if winner == learner_color:
                outcome = WIN_REWARD
            elif winner is not None:
                outcome = LOSS_REWARD
            else:
                outcome = DRAW_SHORT if ply < MAX_PLY else DRAW_LONG
            done = True
            break

        player = board.turn

        if player == learner_color:
            enc = encode_position(board, player, sentinel_advisor=sentinel, db=None, value_net=value_net)
            if enc is None or not enc.legal_moves:
                outcome = LOSS_REWARD
                done = True
                break

            feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits     = model.policy_logits(feat_t)
                scaled     = logits / max(temperature, 1e-6)
                log_probs  = F.log_softmax(scaled, dim=-1)
                probs      = log_probs.exp()
                if not torch.isfinite(probs).all():
                    probs  = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs      = probs / probs.sum().clamp(min=1e-9)
                entropy    = float((-(probs * log_probs).sum()).item())

                # Forced opening: override learner's placement choice if book move is legal
                forced_idx = None
                if (forced_placements
                        and board.phase == "place"
                        and learner_placement_count < len(forced_placements)):
                    book_pos = forced_placements[learner_placement_count]
                    for _fi, _m in enumerate(enc.legal_moves):
                        if _m.get("to") == book_pos:
                            forced_idx = _fi
                            break

                if forced_idx is not None:
                    chosen_idx = forced_idx
                else:
                    chosen_idx = int(torch.multinomial(probs.cpu(), 1).item())
                chosen_prob = float(probs[chosen_idx].item())
                top1_prob  = float(probs.max().item())
                was_top1_policy = int(chosen_idx == int(torch.argmax(probs).item()))
                log_prob_old    = float(log_probs[chosen_idx].item())

            move       = enc.legal_moves[chosen_idx]
            if board.phase == "place":
                learner_placement_count += 1
            board_after = board.apply_move(move)
            enc_after  = encode_position(board_after, opp_color, sentinel_advisor=sentinel, db=None, value_net=value_net)

            db_moves = []
            if db is not None:
                try:
                    db_moves = db.query_all_moves(board, player) or []
                except Exception:
                    pass

            reward, rb, extra = _compute_per_move_reward(enc, chosen_idx, enc_after, db_moves=db_moves)

            # Mill formation bonus
            mills_before = sum(1 for m in MILLS if all(board.positions.get(p) == learner_color for p in m))
            mills_after  = sum(1 for m in MILLS if all(board_after.positions.get(p) == learner_color for p in m))
            if mills_after > mills_before:
                mill_bonus = MILL_BONUS * (mills_after - mills_before)
                reward += mill_bonus
                rb.mill_formed += mill_bonus
                rb.total += mill_bonus

            if db is not None:
                try:
                    opp_state_wdl = db.query_state(board_after)
                    if opp_state_wdl == "L":
                        reward      += DELTA
                        rb.malom_trap += DELTA
                        rb.total    += DELTA
                except Exception:
                    pass

            if enc_after is not None and enc_after.legal_moves:
                next_mf = enc_after.feat_matrix
                next_vi = enc_after.value_input
            else:
                next_mf = np.zeros((1, enc.feat_matrix.shape[1]), dtype=np.float32)
                next_vi = np.zeros(enc.value_input.shape, dtype=np.float32)

            terminal_next, _ = is_terminal(board_after)
            step = ScaffoldedStep(
                move_features=enc.feat_matrix,
                value_input=enc.value_input,
                chosen_idx=chosen_idx,
                log_prob_old=log_prob_old,
                reward=reward,
                next_move_features=next_mf,
                next_value_input=next_vi,
                done=terminal_next,
            )
            game_trajectory.append(step)

            sentinel_scores   = list(getattr(enc, "sentinel_scores", []) or [])
            sentinel_mean     = float(sum(sentinel_scores) / len(sentinel_scores)) if sentinel_scores else 0.0
            sentinel_chosen   = float(sentinel_scores[chosen_idx]) if sentinel_scores else 0.0
            h_before  = float(getattr(enc, "h_before", 0.0))
            h_after   = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
            vn_before = float(getattr(enc, "vn_before", 0.0))
            vn_after  = float(enc.vn_scores_abs[chosen_idx]) if getattr(enc, "vn_scores_abs", None) else vn_before
            heuristic_top1 = 0
            if getattr(enc, "h_scores_abs", None):
                heuristic_top1 = int(chosen_idx == int(np.argmax(np.asarray(enc.h_scores_abs))))

            step_diags.append(StepDiag(
                reward=rb,
                legal_moves=len(enc.legal_moves),
                chosen_idx=chosen_idx,
                chosen_prob=chosen_prob,
                entropy=entropy,
                top1_prob=top1_prob,
                sentinel_mean=sentinel_mean,
                sentinel_chosen=sentinel_chosen,
                h_before=h_before,
                h_after=h_after,
                h_delta=h_after - h_before,
                vn_before=vn_before,
                vn_after=vn_after,
                vn_delta=vn_after - vn_before,
                malom_chosen_wdl=extra["malom_chosen_wdl"],
                malom_chosen_dtm=extra["malom_chosen_dtm"],
                was_top1_policy=was_top1_policy,
                was_top1_heuristic=heuristic_top1,
            ))

            # Record branch candidate every branch_every learner moves
            learner_move_count += 1
            if record_branches and branch_every > 0 and (learner_move_count % branch_every == 0):
                moves_into_movement = (ply - move_phase_start_ply) if move_phase_start_ply is not None else None
                branch_candidates.append((ply, board, _phase_bucket(board, moves_into_movement)))

            board = board_after

        else:
            # Opponent's turn
            try:
                opp_move = opponent.choose_move(board)
            except Exception:
                opp_move = None
            if not opp_move:
                outcome = WIN_REWARD
                done    = True
                break
            board = board.apply_move(opp_move)

        ply += 1

    if not done:
        outcome = DRAW_LONG

    return RolloutResult(
        trajectory=game_trajectory,
        step_diags=step_diags,
        outcome=outcome,
        ply=ply,
        branch_candidates=branch_candidates,
        retry_board=retry_board,
    )


# ── Diagnostic logging ────────────────────────────────────────────────────────

def _build_game_diag(
    game_count:      int,
    difficulty:      int,
    learner_color:   str,
    temperature:     float,
    result:          RolloutResult,
    best_win_rate:   float,
    win_history:     deque,
    last_update_pl:  Optional[float],
    last_update_vl:  Optional[float],
    last_update_ent: Optional[float],
    opt:             torch.optim.Optimizer,
    temp_frozen:     bool,
    source_ckpt:     str,
    game_type:       str,
    phase_bucket:    str,
    is_branch:       bool,
    branch_ply_start: int,
    target_age:      int,
    bucket_counts:   Counter,
) -> GameDiag:
    sd = result.step_diags
    win_rate = sum(win_history) / max(len(win_history), 1)
    return GameDiag(
        game=game_count,
        difficulty=difficulty,
        learner_color=learner_color,
        temperature=round(temperature, 4),
        outcome=float(result.outcome),
        win_rate_200=round(win_rate, 4),
        ply=int(result.ply),
        steps=len(sd),
        update_policy_loss=None if last_update_pl is None else float(last_update_pl),
        update_value_loss=None if last_update_vl is None else float(last_update_vl),
        update_entropy=None if last_update_ent is None else float(last_update_ent),
        reward_total_mean=_safe_mean([d.reward.total for d in sd]),
        reward_sentinel_mean=_safe_mean([d.reward.sentinel for d in sd]),
        reward_heuristic_mean=_safe_mean([d.reward.heuristic for d in sd]),
        reward_value_mean=_safe_mean([d.reward.value_net for d in sd]),
        reward_malom_win_mean=_safe_mean([d.reward.malom_win for d in sd]),
        reward_malom_trap_mean=_safe_mean([d.reward.malom_trap for d in sd]),
        reward_retro_mean=_safe_mean([d.reward.retro for d in sd]),
        sentinel_mean=_safe_mean([d.sentinel_mean for d in sd]),
        sentinel_chosen_mean=_safe_mean([d.sentinel_chosen for d in sd]),
        h_delta_mean=_safe_mean([d.h_delta for d in sd]),
        vn_delta_mean=_safe_mean([d.vn_delta for d in sd]),
        chosen_prob_mean=_safe_mean([d.chosen_prob for d in sd]),
        entropy_mean=_safe_mean([d.entropy for d in sd]),
        top1_prob_mean=_safe_mean([d.top1_prob for d in sd]),
        legal_moves_mean=_safe_mean([float(d.legal_moves) for d in sd]),
        policy_top1_rate=_safe_mean([float(d.was_top1_policy) for d in sd]),
        heuristic_top1_rate=_safe_mean([float(d.was_top1_heuristic) for d in sd]),
        malom_win_move_rate=_safe_mean([1.0 if d.malom_chosen_wdl == "win" else 0.0 for d in sd]),
        malom_unknown_rate=_safe_mean([1.0 if d.malom_chosen_wdl == "unknown" else 0.0 for d in sd]),
        best_win_rate=float(best_win_rate),
        temp_frozen=int(temp_frozen),
        lr=float(opt.param_groups[0]["lr"]),
        source_checkpoint=source_ckpt,
        game_type=game_type,
        phase_bucket=phase_bucket,
        is_branch=int(is_branch),
        branch_ply_start=branch_ply_start,
        target_age=target_age,
        bucket_opening=bucket_counts.get("opening", 0),
        bucket_midgame=bucket_counts.get("midgame", 0),
        bucket_endgame=bucket_counts.get("endgame", 0),
    )


# ── Main training loop ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s2b] Device: {device}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load components ────────────────────────────────────────────────────────
    sentinel = None
    sent_path = args.sentinel or str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
    if Path(sent_path).exists():
        sentinel = load_advisor(sent_path)
        if sentinel and sentinel.is_loaded():
            print(f"[s2b] Sentinel loaded: {sent_path}")
        else:
            sentinel = None
    if sentinel is None:
        print("[s2b] Sentinel unavailable — sentinel reward = 0")

    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s2b] Malom DB loaded: {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s2b] Malom DB failed ({e})")
    if db is None:
        print("[s2b] Malom DB unavailable — Malom rewards = 0")

    value_net = None
    vn_path = args.value_net or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[s2b] Value net loaded: {vn_path}")
        except Exception as e:
            print(f"[s2b] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[s2b] No value net — VN features will be 0")

    # ── Load model ─────────────────────────────────────────────────────────────
    resume_path, source_tag = _choose_resume_path(args)
    model, start_game, best_win_rate, difficulty, source_checkpoint = _load_model(
        device, resume_path, force_start_diff=args.force_start_diff
    )
    difficulty = _apply_diff_start_override(difficulty, args)
    if resume_path is None:
        print("[s2b] No checkpoint found — starting from scratch")
    else:
        print(f"[s2b] Resuming from ({source_tag}): {resume_path}")
    print(f"[s2b] Starting at game {start_game}, difficulty {difficulty}")

    # ── Frozen opponent (self-play target network) ─────────────────────────────
    frozen_opp = FrozenModelOpponent(model, device, sentinel=sentinel, value_net=value_net)
    games_since_target_update = 0

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    game_count              = start_game
    temperature             = args.temp_start
    win_history:              deque[float] = deque(maxlen=args.rolling_win)
    win_history_heuristic:    deque[float] = deque(maxlen=args.rolling_win)
    malom_win_rate_history:   deque[float] = deque(maxlen=10)
    ep_steps:  list[ScaffoldedStep] = []
    last_update_pl   = None
    last_update_vl   = None
    last_update_ent  = None
    best_win_rate_at_diff = 0.0   # resets each difficulty level

    # Rolling bucket saturation tracker
    branch_bucket_history: deque[str] = deque(maxlen=args.bucket_window)

    log_path        = out_dir / "train_log.jsonl"
    update_log_path = out_dir / "update_log.jsonl"

    print(f"[s2b] Starting at game {game_count}, difficulty {difficulty}")
    print(f"[s2b] Self-play ratio {args.self_play_ratio:.0%}, "
          f"branch every {args.branch_every} turns, "
          f"max {args.max_branches_per_game} branches/game")

    # ── Initial s1b refresher ──────────────────────────────────────────────────
    if not args.no_s1b_refresher:
        print(f"[s2b] Running s1b refresher before diff {difficulty} training")
        _run_s1b_refresher(model, device, args.s1b_data,
                           epochs=args.s1b_refresher_epochs,
                           lr=args.s1b_refresher_lr)

    diag_buffer: list[GameDiag] = []

    while game_count < args.max_games:
        temperature = _compute_temperature(game_count, args.max_games)

        # Refresh frozen model periodically
        if games_since_target_update >= args.update_target_every:
            frozen_opp.refresh(model)
            games_since_target_update = 0
            print(f"[s2b] Frozen model updated at game {game_count}")

        learner_color = "W" if rng.random() < 0.5 else "B"
        opp_color     = "B" if learner_color == "W" else "W"

        # Choose opponent type for this main game
        use_self_play = rng.random() < args.self_play_ratio
        if use_self_play:
            opponent = frozen_opp
            game_type = "vs_frozen"
        else:
            from learned_ai.agents.heuristic_agent import GameAI as _GA
            _h = HeuristicAgent(color=opp_color, difficulty=difficulty, game_ai=None)
            _h._inner = _GA(color=opp_color, difficulty=difficulty, override_time_budget=args.time_budget)
            opponent  = _h
            game_type = "vs_heuristic"

        # ── Opening book: 50% of games follow a book/learned opening line ────────
        game_forced_placements: Optional[list[str]] = None
        if _OPENING_LINES and rng.random() < BOOK_GAME_PROB:
            line = _OPENING_LINES[rng.randint(0, len(_OPENING_LINES) - 1)]
            game_forced_placements = _sample_forced_placements(line, learner_color)

        # ── Main game rollout ──────────────────────────────────────────────────
        game_retry_ply = rng.randint(RETRY_PLY_MIN, RETRY_PLY_MAX)
        result = _rollout(
            model=model,
            device=device,
            start_board=BoardState.new_game(),
            learner_color=learner_color,
            opponent=opponent,
            opp_color=opp_color,
            sentinel=sentinel,
            db=db,
            value_net=value_net,
            temperature=temperature,
            max_ply=args.max_ply,
            record_branches=(args.max_branches_per_game > 0),
            branch_every=args.branch_every,
            retry_ply=game_retry_ply,
            forced_placements=game_forced_placements,
        )

        if result.trajectory:
            _retroactive_rescore(result.trajectory, result.step_diags, result.outcome)

        # WIN: always learn. LOSS/DRAW_SHORT: run confirmation retry from random ply.
        # Only add original loss/draw if retry confirms same outcome class.
        if result.outcome == WIN_REWARD:
            ep_steps.extend(result.trajectory)
        elif result.outcome in (LOSS_REWARD, DRAW_SHORT) and result.retry_board is not None:
            confirm_result = _rollout(
                model=model,
                device=device,
                start_board=result.retry_board,
                learner_color=learner_color,
                opponent=opponent,
                opp_color=opp_color,
                sentinel=sentinel,
                db=db,
                value_net=value_net,
                temperature=temperature,
                max_ply=args.max_ply,
                record_branches=False,
                branch_every=0,
                retry_ply=0,
            )
            if confirm_result.trajectory:
                _retroactive_rescore(confirm_result.trajectory, confirm_result.step_diags,
                                     confirm_result.outcome)
            confirmed = (
                (result.outcome == LOSS_REWARD  and confirm_result.outcome == LOSS_REWARD) or
                (result.outcome == DRAW_SHORT   and confirm_result.outcome == DRAW_SHORT)
            )
            if confirmed and result.trajectory:
                ep_steps.extend(result.trajectory)
            if confirm_result.outcome in (WIN_REWARD, DRAW_SHORT):
                ep_steps.extend(confirm_result.trajectory)
            game_count += 1
            games_since_target_update += 1
            win_history.append(1.0 if confirm_result.outcome == WIN_REWARD else 0.0)
            if game_type == "vs_heuristic":
                win_history_heuristic.append(1.0 if confirm_result.outcome == WIN_REWARD else 0.0)
            _coc = "W" if confirm_result.outcome == WIN_REWARD else ("L" if confirm_result.outcome == LOSS_REWARD else "D")
            if game_count % 10 == 0:
                print(f"[s2b] {game_count:6d}  r{game_retry_ply:2d} {learner_color} |          | {_coc} ply={confirm_result.ply:3d} | (from ply {game_retry_ply}) {'[learn]' if confirmed else '[skip]'}")

        win_history.append(1.0 if result.outcome == WIN_REWARD else 0.0)
        if game_type == "vs_heuristic":
            win_history_heuristic.append(1.0 if result.outcome == WIN_REWARD else 0.0)
        game_count += 1
        games_since_target_update += 1

        bucket_counts = Counter(branch_bucket_history)
        _diag = _build_game_diag(
            game_count, difficulty, learner_color, temperature, result,
            best_win_rate, win_history, last_update_pl, last_update_vl, last_update_ent,
            opt, False, source_checkpoint,
            game_type=game_type, phase_bucket="main", is_branch=False,
            branch_ply_start=0, target_age=games_since_target_update,
            bucket_counts=bucket_counts,
        )
        diag_buffer.append(_diag)
        malom_win_rate_history.append(_diag.malom_win_move_rate)

        if game_count % 10 == 0:
            _hwr  = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
            _awr  = sum(win_history) / max(len(win_history), 1)
            _mwr  = sum(malom_win_rate_history) / max(len(malom_win_rate_history), 1)
            _oc   = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
            _gt   = "heur" if game_type == "vs_heuristic" else "self"
            print(f"[s2b] {game_count:6d} {_gt:4s} {learner_color} | diff {difficulty} | {_oc} ply={result.ply:3d} | hwr={_hwr:.3f} awr={_awr:.3f} malom={_mwr:.1%} | temp={temperature:.2f} lr={opt.param_groups[0]['lr']:.5f}")

        # ── Loss/draw retry from random ply ───────────────────────────────────
        if result.outcome != WIN_REWARD and result.retry_board is not None:
            retry_result = _rollout(
                model=model,
                device=device,
                start_board=result.retry_board,
                learner_color=learner_color,
                opponent=opponent,
                opp_color=opp_color,
                sentinel=sentinel,
                db=db,
                value_net=value_net,
                temperature=temperature,
                max_ply=args.max_ply,
                record_branches=False,
                branch_every=0,
                retry_ply=0,
            )
            if retry_result.trajectory:
                _retroactive_rescore(retry_result.trajectory, retry_result.step_diags, retry_result.outcome)
                if retry_result.outcome in (WIN_REWARD, DRAW_SHORT):
                    ep_steps.extend(retry_result.trajectory)
            win_history.append(1.0 if retry_result.outcome == WIN_REWARD else 0.0)
            if game_type == "vs_heuristic":
                win_history_heuristic.append(1.0 if retry_result.outcome == WIN_REWARD else 0.0)
            game_count += 1
            games_since_target_update += 1
            _roc = "W" if retry_result.outcome == WIN_REWARD else ("L" if retry_result.outcome == LOSS_REWARD else "D")
            if game_count % 10 == 0:
                print(f"[s2b] {game_count:6d} retry {learner_color} |          | {_roc} ply={retry_result.ply:3d} | (from ply {game_retry_ply})")

        # ── Spawn branch games ─────────────────────────────────────────────────
        branches_spawned = 0
        # Shuffle candidates so we don't always pick early-game branches
        candidates = list(result.branch_candidates)
        rng.shuffle(candidates)
        # Try to pick one from each bucket first, then fill remaining slots
        seen_buckets: set[str] = set()
        ordered_candidates: list[tuple[int, BoardState, str]] = []
        for cand in candidates:
            if cand[2] not in seen_buckets:
                ordered_candidates.insert(0, cand)   # prioritise diverse buckets
                seen_buckets.add(cand[2])
            else:
                ordered_candidates.append(cand)

        for branch_ply, branch_board, bucket in ordered_candidates:
            if branches_spawned >= args.max_branches_per_game:
                break
            # Saturation check
            bucket_counts = Counter(branch_bucket_history)
            if bucket_counts.get(bucket, 0) >= args.max_per_bucket:
                continue

            # Branch game: model vs frozen copy from mid-game state
            # Learner color stays the same so reward signs are consistent
            branch_result = _rollout(
                model=model,
                device=device,
                start_board=branch_board,
                learner_color=learner_color,
                opponent=frozen_opp,
                opp_color=opp_color,
                sentinel=sentinel,
                db=db,
                value_net=value_net,
                temperature=temperature,
                max_ply=args.max_ply_branch,
                record_branches=False,   # no nested branching
                branch_every=0,
                retry_ply=0,
            )

            if branch_result.trajectory:
                _retroactive_rescore(branch_result.trajectory, branch_result.step_diags, branch_result.outcome)
                if branch_result.outcome in (WIN_REWARD, DRAW_SHORT):
                    ep_steps.extend(branch_result.trajectory)
                branch_bucket_history.append(bucket)
                branches_spawned += 1
                game_count += 1
                games_since_target_update += 1
                win_history.append(1.0 if branch_result.outcome == WIN_REWARD else 0.0)

                bucket_counts = Counter(branch_bucket_history)
                diag_buffer.append(_build_game_diag(
                    game_count, difficulty, learner_color, temperature, branch_result,
                    best_win_rate, win_history, last_update_pl, last_update_vl, last_update_ent,
                    opt, False, source_checkpoint,
                    game_type="branch", phase_bucket=bucket, is_branch=True,
                    branch_ply_start=branch_ply, target_age=games_since_target_update,
                    bucket_counts=bucket_counts,
                ))

                if game_count % 10 == 0:
                    _boc = "W" if branch_result.outcome == WIN_REWARD else ("L" if branch_result.outcome == LOSS_REWARD else "D")
                    print(f"[s2b] {game_count:6d}  +b  {learner_color} | {bucket:7s} | {_boc} ply={branch_result.ply:3d} | (from ply {branch_ply})")

        # ── Update ────────────────────────────────────────────────────────────
        if len(ep_steps) >= args.update_every:
            last_update_pl, last_update_vl, last_update_ent = update_fn(
                model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef
            )
            upd_entry = {
                "game":         game_count,
                "policy_loss":  None if last_update_pl  is None else float(last_update_pl),
                "value_loss":   None if last_update_vl  is None else float(last_update_vl),
                "entropy":      None if last_update_ent is None else float(last_update_ent),
                "lr":           float(opt.param_groups[0]["lr"]),
                "batch_steps":  len(ep_steps),
            }
            with open(update_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(upd_entry) + "\n")
            ep_steps.clear()

        # ── Periodic log + checkpoint ──────────────────────────────────────────
        if game_count % args.log_every == 0 and diag_buffer:
            win_rate     = sum(win_history_heuristic) / max(len(win_history_heuristic), 1)
            win_rate_all = sum(win_history) / max(len(win_history), 1)

            # Adaptive LR based on win rate
            _adapt_lr(opt, win_rate, args.lr)

            # Recovery: reload best checkpoint if win rate is very poor
            if (len(win_history_heuristic) >= RECOVERY_MIN_GAMES
                    and win_rate < RECOVERY_THRESHOLD):
                best_ckpt = out_dir / f"best{difficulty}.pt"
                if best_ckpt.exists():
                    ckpt_r = torch.load(str(best_ckpt), map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_r["model"])
                    model.to(device)
                    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                    frozen_opp.refresh(model)
                    win_history.clear()
                    win_history_heuristic.clear()
                    temperature = TEMP_START
                    print(f"[s2b] Recovery: reloaded best{difficulty}.pt (win rate was {win_rate:.2f}, temp reset to {TEMP_START})")

            main_diags   = [d for d in diag_buffer if not d.is_branch]
            branch_diags = [d for d in diag_buffer if d.is_branch]
            bc = Counter(branch_bucket_history)

            # Write JSONL
            with open(log_path, "a", encoding="utf-8") as f:
                for d in diag_buffer:
                    f.write(json.dumps(asdict(d)) + "\n")
            diag_buffer.clear()

            # Console summary
            last_main = next((d for d in reversed(main_diags) if main_diags), None)
            if last_main:
                d = last_main
                _sign = lambda v: f"{'+' if v >= 0 else ''}{v:.3f}"
                print(
                    f"[s2b] game {game_count:6d} | diff {difficulty} | "
                    f"win-200={win_rate:.3f} | all={win_rate_all:.3f} | "
                    f"temp={temperature:.2f} | "
                    f"outcome={d.outcome:+.2f} | lr={opt.param_groups[0]['lr']:.5f} | "
                    f"rew={_sign(d.reward_total_mean)} | "
                    f"sent={_sign(d.reward_sentinel_mean)} "
                    f"h={_sign(d.reward_heuristic_mean)} "
                    f"vn={_sign(d.reward_value_mean)} "
                    f"mw={_sign(d.reward_malom_win_mean)} "
                    f"mt={_sign(d.reward_malom_trap_mean)} | "
                    f"p_top1={d.policy_top1_rate:.2f} h_top1={d.heuristic_top1_rate:.2f} | "
                    f"branches={len(branch_diags)} "
                    f"[op={bc.get('opening',0)} mid={bc.get('midgame',0)} end={bc.get('endgame',0)}]"
                )

            ckpt = {
                "model":             model.state_dict(),
                "model_config":      model.get_config(),
                "stage":             "s2b",
                "game_count":        game_count,
                "best_win_rate":     best_win_rate,
                "difficulty":        difficulty,
                "source_checkpoint": source_checkpoint,
                "lr":                float(opt.param_groups[0]["lr"]),
                "temperature":       float(temperature),
            }
            torch.save(ckpt, out_dir / "latest.pt")

            if win_rate > best_win_rate_at_diff and len(win_history_heuristic) >= min(100, args.rolling_win):
                best_win_rate_at_diff = win_rate
                ckpt["best_win_rate"] = best_win_rate_at_diff
                torch.save(ckpt, out_dir / f"best{difficulty}.pt")
                torch.save(ckpt, out_dir / "best.pt")
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                print(f"[s2b]  → best diff-{difficulty} win rate: {best_win_rate_at_diff:.3f}  (saved best{difficulty}.pt)")

        # ── Difficulty advancement ─────────────────────────────────────────────
        if len(win_history_heuristic) >= args.rolling_win:
            win_rate = sum(win_history_heuristic) / len(win_history_heuristic)
            advance_thr = ADVANCE_THRESHOLDS.get(difficulty, args.advance_threshold)
            if difficulty >= args.diff_max:
                if win_rate >= args.exit_threshold:
                    print(f"[s2b] *** {win_rate:.3f} win rate vs difficulty {difficulty} — done! ***")
                    break
            elif win_rate >= advance_thr:
                prev_diff = difficulty
                difficulty += 1
                win_history.clear()
                win_history_heuristic.clear()
                print(f"[s2b] *** Advanced to difficulty {difficulty} (was {win_rate:.3f} vs diff {prev_diff}) ***")

                # Load best checkpoint for the completed level as clean starting point
                prev_best = out_dir / f"best{prev_diff}.pt"
                if prev_best.exists():
                    ckpt_prev = torch.load(str(prev_best), map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_prev["model"])
                    model.to(device)
                    print(f"[s2b] Loaded best{prev_diff}.pt as starting point for diff {difficulty}")
                else:
                    print(f"[s2b] best{prev_diff}.pt not found — continuing from current weights")

                # Reset per-level tracking and optimizer
                best_win_rate_at_diff = 0.0
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                frozen_opp.refresh(model)

                # s1b refresher before new difficulty
                if not args.no_s1b_refresher:
                    print(f"[s2b] Running s1b refresher before diff {difficulty} training")
                    _run_s1b_refresher(model, device, args.s1b_data,
                                       epochs=args.s1b_refresher_epochs,
                                       lr=args.s1b_refresher_lr)

    # ── Final flush ───────────────────────────────────────────────────────────
    if ep_steps:
        update_fn(model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef)
    if diag_buffer:
        with open(log_path, "a", encoding="utf-8") as f:
            for d in diag_buffer:
                f.write(json.dumps(asdict(d)) + "\n")

    ckpt = {
        "model":             model.state_dict(),
        "model_config":      model.get_config(),
        "stage":             "s2b",
        "game_count":        game_count,
        "best_win_rate":     best_win_rate,
        "difficulty":        difficulty,
        "source_checkpoint": source_checkpoint,
        "lr":                float(opt.param_groups[0]["lr"]),
        "temperature":       float(temperature),
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s2b] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s2b] Checkpoint: {out_dir / 'best.pt'}")
    print(f"[s2b] Logs: {log_path} and {update_log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2b: self-play + branched mid-game rollouts")
    p.add_argument("--resume",              default="",    type=str, help="Explicit checkpoint path")
    p.add_argument("--auto-resume-best",    action="store_true", help="Prefer s2b/best.pt in resume chain")
    p.add_argument("--auto-resume-latest",  action="store_true", help="Prefer s2b/latest.pt in resume chain")
    p.add_argument("--auto-resume-s2",      action="store_true", help="Start from s2/best.pt")
    p.add_argument("--out-dir",  default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b"))
    p.add_argument("--sentinel", default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--malom",    default="", type=str)
    p.add_argument("--value-net",default=str(_ROOT / "data" / "value_net.npz"), type=str)
    p.add_argument("--ppo",      action="store_true")
    p.add_argument("--max-games",           type=int,   default=5000)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--lr",                  type=float, default=LR)
    p.add_argument("--gamma-td",            type=float, default=GAMMA_TD)
    p.add_argument("--entropy-coef",        type=float, default=ENTROPY_COEF)
    p.add_argument("--update-every",        type=int,   default=UPDATE_EVERY)
    p.add_argument("--rolling-win",         type=int,   default=ROLLING_WIN)
    p.add_argument("--force-start-diff",   action="store_true",
                   help="When resuming, clamp starting difficulty to at least DIFF_START (default 4)")
    p.add_argument("--diff-start",          type=int,   default=None,
                   help="Override starting difficulty (e.g. 1 to start from easiest)")
    p.add_argument("--diff-max",            type=int,   default=DIFF_MAX,
                   help="Highest difficulty to train against (default 7)")
    p.add_argument("--advance-threshold",   type=float, default=0.50,
                   help="Fallback win rate to advance difficulty (per-level defaults in ADVANCE_THRESHOLDS)")
    p.add_argument("--exit-threshold",      type=float, default=EXIT_THRESHOLD,
                   help="Win rate vs diff-max considered done (default 0.30)")
    p.add_argument("--temp-start",          type=float, default=TEMP_START)
    p.add_argument("--log-every",           type=int,   default=LOG_EVERY)
    p.add_argument("--max-ply",             type=int,   default=MAX_PLY)
    p.add_argument("--max-ply-branch",      type=int,   default=MAX_PLY_BRANCH)
    p.add_argument("--time-budget",         type=float, default=TIME_BUDGET)
    # s2b-specific
    p.add_argument("--self-play-ratio",     type=float, default=SELF_PLAY_RATIO,
                   help="Fraction of main games vs frozen model (default 0.5)")
    p.add_argument("--update-target-every", type=int,   default=UPDATE_TARGET_EVERY,
                   help="Games between frozen-model refreshes (default 50)")
    p.add_argument("--branch-every",        type=int,   default=BRANCH_EVERY,
                   help="Snapshot branch candidate every N learner moves (default 10)")
    p.add_argument("--max-branches-per-game", type=int, default=0,
                   help="Max branch games spawned per main game (default 0; 0 disables)")
    p.add_argument("--bucket-window",       type=int,   default=BUCKET_WINDOW,
                   help="Rolling window for bucket saturation (default 300)")
    p.add_argument("--max-per-bucket",      type=int,   default=MAX_PER_BUCKET,
                   help="Max branch games from any bucket in window (default 80)")
    # s1b refresher
    p.add_argument("--s1b-data",             type=str,   default=str(_ROOT / "learned_ai" / "data" / "human_imitation.npz"),
                   help="Path to human_imitation.npz for s1b refresher")
    p.add_argument("--s1b-refresher-epochs", type=int,   default=S1B_REFRESHER_EPOCHS,
                   help="Epochs per s1b refresher run (default 3)")
    p.add_argument("--s1b-refresher-lr",     type=float, default=S1B_REFRESHER_LR,
                   help="Learning rate for s1b refresher (default 3e-4)")
    p.add_argument("--no-s1b-refresher",     action="store_true",
                   help="Disable s1b refresher at start and on difficulty advance")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
