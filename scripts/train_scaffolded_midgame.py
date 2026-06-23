"""scripts/train_scaffolded_midgame.py — Midgame specialist: movement phase when >= 12 pieces.

Trains on the movement phase when 12 or more total pieces are on the board.
Rewards are gated: heuristic delta, mobility reward, and Malom trap bonus only
fire when the board is in midgame (>= 12 pieces, not placement phase).
Mill bonus is un-gated.  No sentinel reward in this specialist.

Resume chain: explicit --resume → s_mid/best.pt → s1c/best.pt → s1b/best.pt → s1/best.pt

Usage
-----
# Quick smoke test (20 games)
.venv/bin/python scripts/train_scaffolded_midgame.py --max-games 20

# Normal run from best available checkpoint
.venv/bin/python scripts/train_scaffolded_midgame.py --auto-resume-best
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
from learned_ai.models.lookahead_advisor import LookaheadAdvisor
from learned_ai.models.scaffolded_encoder import encode_position_with_lookahead, MOVE_FEAT_DIM_WITH_LOOKAHEAD
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
    scaffolded_ppo_update,
)

# ── Opening book ──────────────────────────────────────────────────────────────

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
BOOK_GAME_PROB = 0.50   # 50% of games follow opening book


def _sample_forced_placements(line_moves: list[str], learner_color: str) -> list[str]:
    """Extract up to 4 placement positions for the learner's side from a line."""
    start = 0 if learner_color == "W" else 1
    return [line_moves[i] for i in range(start, len(line_moves), 2)][:4]


# ── Stage tag ─────────────────────────────────────────────────────────────────

STAGE_TAG = "s_mid"
OUT_DIR   = "learned_ai/checkpoints/scaffolded/s_mid"

# ── Reward weights ────────────────────────────────────────────────────────────

ALPHA         = 0.20   # sentinel quality delta (chosen vs mean)
BETA          = 0.15   # heuristic delta
MILL_BONUS    = 0.25   # larger mill bonus — midgame mills more decisive
MOBILITY_COEF = 0.10   # mobility reward
GAMMA              = 0.0    # no separate Malom win reward (covered by DELTA)
DELTA              = 0.15   # Malom reward: opponent in losing state after our move
MALOM_LOSS_PENALTY = 0.15   # Malom penalty: opponent in winning state after our move
VN_BETA       = 0.0    # no value-net reward
LAMBDA        = 0.50   # retro-active outcome weight
DECAY         = 0.98   # retro decay per ply remaining

OPP_MILL_PENALTY       = 0.25   # opponent closes a new mill
OPP_MILL_DANCE_PENALTY = 0.20   # opponent opens their own mill (dance start)
MILL_DANCE_REWARD      = 0.40   # own piece: source in completed mill, dest closes different mill
BLOCK_OPP_DANCE_REWARD = 0.15   # own move occupies a square opponent would use to close a mill
DUAL_MILL_CONFIG_REWARD = 0.45  # newly created double-mill (piece shared by 2 completed mills)

WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_SHORT  =  0.15
DRAW_LONG   = -0.05

# ── Optimiser / schedule ──────────────────────────────────────────────────────

LR            = 1e-4
GAMMA_TD      = 0.99
TEMP_START    = 0.50
TEMP_MIN      = 0.45
TEMP_MAX      = 0.90
ENTROPY_COEF  = 0.01
UPDATE_EVERY  = 16
ROLLING_WIN   = 50
DIFF_START    = 1
DIFF_MAX      = 7
EXIT_THRESHOLD = 0.50

S1B_REFRESHER_EPOCHS = 3
S1B_REFRESHER_LR     = 3e-4
S1B_REFRESHER_BATCH  = 32
MAX_PLY        = 60
MAX_PLY_BRANCH = 60
TIME_BUDGET    = 0.05

LOG_EVERY    = 50
LR_SCALE_WIN = 0.35
LR_SCALE_MIN = 0.50
LR_SCALE_MAX = 2.00
RECOVERY_THRESHOLD  = 0.12
RECOVERY_MIN_GAMES  = 30

# ── s2b-compatible knobs ──────────────────────────────────────────────────────

UPDATE_TARGET_EVERY   = 50
SELF_PLAY_RATIO       = 0.5
BRANCH_EVERY          = 10
MAX_BRANCHES_PER_GAME = 2
BUCKET_WINDOW         = 300
MAX_PER_BUCKET        = 80

OPENING_EXTENSION_PLY = 6
MIDGAME_PLY_GATE      = 0    # pool positions already start at movement turn 10

PHASE_BUCKETS = ("opening", "midgame", "endgame")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    total:       float = 0.0
    sentinel:    float = 0.0
    heuristic:   float = 0.0
    value_net:   float = 0.0
    malom_win:   float = 0.0
    malom_trap:  float = 0.0
    mill_formed: float = 0.0
    mill_dance:  float = 0.0
    opp_penalty: float = 0.0
    retro:       float = 0.0


@dataclass
class StepDiag:
    reward:            RewardBreakdown
    legal_moves:       int
    chosen_idx:        int
    chosen_prob:       float
    entropy:           float
    top1_prob:         float
    sentinel_mean:     float
    sentinel_chosen:   float
    h_before:          float
    h_after:           float
    h_delta:           float
    vn_before:         float
    vn_after:          float
    vn_delta:          float
    malom_chosen_wdl:  str
    malom_chosen_dtm:  Optional[float]
    was_top1_policy:   int
    was_top1_heuristic: int


@dataclass
class GameDiag:
    game:                    int
    difficulty:              int
    learner_color:           str
    temperature:             float
    outcome:                 float
    win_rate_200:            float
    ply:                     int
    steps:                   int
    update_policy_loss:      Optional[float]
    update_value_loss:       Optional[float]
    update_entropy:          Optional[float]
    reward_total_mean:       float
    reward_sentinel_mean:    float
    reward_heuristic_mean:   float
    reward_value_mean:       float
    reward_malom_win_mean:   float
    reward_malom_trap_mean:  float
    reward_retro_mean:       float
    sentinel_mean:           float
    sentinel_chosen_mean:    float
    h_delta_mean:            float
    vn_delta_mean:           float
    chosen_prob_mean:        float
    entropy_mean:            float
    top1_prob_mean:          float
    legal_moves_mean:        float
    policy_top1_rate:        float
    heuristic_top1_rate:     float
    malom_win_move_rate:     float
    malom_unknown_rate:      float
    best_win_rate:           float
    temp_frozen:             int
    lr:                      float
    source_checkpoint:       str
    game_type:               str
    phase_bucket:            str
    is_branch:               int
    branch_ply_start:        int
    target_age:              int
    bucket_opening:          int
    bucket_midgame:          int
    bucket_endgame:          int
    reward_mill_dance_mean:  float
    reward_opp_penalty_mean: float


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
    """Classify board into training phase bucket for saturation tracking."""
    total_on_board = board.pieces_on_board["W"] + board.pieces_on_board["B"]
    if board.phase == "place":
        return "opening"
    if total_on_board < 12:
        return "endgame"
    if moves_into_movement is not None and moves_into_movement < OPENING_EXTENSION_PLY:
        return "opening"
    return "midgame"


def _count_mills(board: BoardState, color: str) -> int:
    return sum(1 for m in MILLS if all(board.positions.get(p) == color for p in m))


def _piece_in_any_mill(board: BoardState, pos: str, color: str) -> bool:
    return any(pos in m and all(board.positions.get(p) == color for p in m) for m in MILLS)


def _would_close_mill(board: BoardState, pos: str, color: str) -> bool:
    """True if placing color's piece at pos (currently empty) would close a mill."""
    return any(
        pos in m and all(board.positions.get(p) == color for p in m if p != pos)
        for m in MILLS
    )


def _pieces_in_double_mill(board: BoardState, color: str) -> set:
    """Return positions that are simultaneously part of 2+ completed mills for color."""
    closed = [set(m) for m in MILLS if all(board.positions.get(p) == color for p in m)]
    if len(closed) < 2:
        return set()
    result: set = set()
    for i in range(len(closed)):
        for j in range(i + 1, len(closed)):
            result |= closed[i] & closed[j]
    return result


def _run_s1b_refresher(
    model: ScaffoldedPolicyNet,
    device: torch.device,
    data_path: str,
    epochs: int = S1B_REFRESHER_EPOCHS,
    lr: float = S1B_REFRESHER_LR,
    batch: int = S1B_REFRESHER_BATCH,
    deviate_bonus: float = 1.5,
) -> None:
    """Inline s1b human-imitation refresher. Modifies model in-place."""
    p = Path(data_path)
    if not p.exists():
        print(f"[s_mid] s1b refresher: data not found ({data_path}) — skipping")
        return

    data          = np.load(str(p), allow_pickle=True)
    feat_matrices = data["feat_matrices"]
    label_dists   = data["label_dists"]
    h_top1_idxs   = data["h_top1_idxs"]
    weights       = data["weights"]
    deviates      = data["deviates"]
    is_winner     = data["is_winner"] if "is_winner" in data else np.ones(len(weights), dtype=bool)
    N             = len(weights)

    effective_weights = weights.copy()
    bonus_mask        = (is_winner) & deviates
    effective_weights[bonus_mask] *= deviate_bonus

    loser_idxs  = [i for i in range(N) if not is_winner[i]]
    winner_idxs = [i for i in range(N) if is_winner[i]]

    for param in model.value_mlp.parameters():
        param.requires_grad = False

    opt_s1b = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()), lr=lr
    )

    model.train()
    print(f"[s_mid] s1b refresher: loser={len(loser_idxs)} winner={len(winner_idxs)} positions  lr={lr:.2e}")

    def _pad_feat(fm: np.ndarray) -> np.ndarray:
        """Pad (k, d) feat matrix to (k, MOVE_FEAT_DIM_WITH_LOOKAHEAD) with zeros."""
        k, d = fm.shape
        if d >= MOVE_FEAT_DIM_WITH_LOOKAHEAD:
            return fm[:, :MOVE_FEAT_DIM_WITH_LOOKAHEAD]
        pad = np.zeros((k, MOVE_FEAT_DIM_WITH_LOOKAHEAD - d), dtype=np.float32)
        return np.concatenate([fm, pad], axis=1)

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
                    feat = torch.tensor(_pad_feat(feat_matrices[i]), dtype=torch.float32).to(device)
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
            print(f"[s_mid]   refresher [{phase_label}] epoch {epoch}/{epochs}  loss={ep_loss / max(ep_w_sum, 1e-9):.4f}")

    _run_phase(loser_idxs, "loser→heuristic", use_heuristic_target=True)
    _run_phase(winner_idxs, "winner", use_heuristic_target=False)

    for param in model.value_mlp.parameters():
        param.requires_grad = True

    model.eval()
    print("[s_mid] s1b refresher done")


def _choose_resume_path(args: argparse.Namespace) -> tuple[Optional[Path], str]:
    if args.resume:
        p = Path(args.resume)
        if p.exists():
            return p, "explicit_resume"
    s_mid_best = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_mid" / "best.pt"
    s1c_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1c"   / "best.pt"
    s1b_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1b"   / "best.pt"
    s1_best    = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1"    / "best.pt"
    candidates = []
    if args.auto_resume_best:
        candidates.append((s_mid_best, "s_mid_best"))
    candidates += [
        (s1c_best, "s1c_best"),
        (s1b_best, "s1b_best"),
        (s1_best,  "s1_best"),
    ]
    for p, tag in candidates:
        if p.exists():
            return p, tag
    return None, "scratch"


def _load_model(device: torch.device, resume_path: Optional[Path]) -> tuple[ScaffoldedPolicyNet, int, float, int, str]:
    if resume_path is None:
        return ScaffoldedPolicyNet(move_feat_dim=MOVE_FEAT_DIM_WITH_LOOKAHEAD).to(device), 0, 0.0, DIFF_START, "scratch"
    ckpt   = torch.load(resume_path, map_location=device, weights_only=False)
    cfg    = ckpt.get("model_config", {})
    model  = ScaffoldedPolicyNet.from_config(cfg).to(device)
    sd_key = "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    stage      = ckpt.get("stage", "unknown")
    is_s_mid   = (stage == STAGE_TAG)
    start_game = int(ckpt.get("game_count",   0))         if is_s_mid else 0
    best_wr    = float(ckpt.get("best_win_rate", 0.0))    if is_s_mid else 0.0
    difficulty = int(ckpt.get("difficulty",   DIFF_START)) if is_s_mid else DIFF_START
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
    """Scale LR proportionally to win rate."""
    scale  = max(LR_SCALE_MIN, min(LR_SCALE_MAX, win_rate / LR_SCALE_WIN))
    new_lr = lr_base * scale
    for g in opt.param_groups:
        g["lr"] = new_lr


def _compute_per_move_reward(
    enc,
    chosen_idx: int,
    enc_after,
    db_moves=None,
    board=None,
    in_midgame: bool = False,
) -> tuple[float, RewardBreakdown, dict[str, Any]]:
    rb    = RewardBreakdown()
    extra: dict[str, Any] = {"malom_chosen_wdl": "unknown", "malom_chosen_dtm": None}

    if in_midgame:
        if getattr(enc, "sentinel_scores", None):
            mean_s   = float(sum(enc.sentinel_scores) / len(enc.sentinel_scores))
            played_s = float(enc.sentinel_scores[chosen_idx])
            rb.sentinel = ALPHA * (played_s - mean_s)

        if enc_after is not None:
            h_before = float(getattr(enc, "h_before", 0.0))
            h_after  = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
            rb.heuristic = BETA * math.tanh(h_after - h_before)

    rb.total = rb.sentinel + rb.heuristic
    return float(rb.total), rb, extra


def _retroactive_rescore(trajectory: list[ScaffoldedStep], step_diags: list[StepDiag], outcome: float) -> None:
    n = len(trajectory)
    for t_idx, step in enumerate(trajectory):
        plies_remaining  = n - t_idx - 1
        delta            = LAMBDA * outcome * (DECAY ** plies_remaining)
        step.reward     += delta
        step_diags[t_idx].reward.retro += float(delta)
        step_diags[t_idx].reward.total += float(delta)


def _outcome_to_history_float(outcome: float) -> float:
    """Map rollout outcome to 1.0 (win) / 0.5 (draw) / 0.0 (loss)."""
    if outcome == WIN_REWARD:
        return 1.0
    if outcome in (DRAW_SHORT, DRAW_LONG):
        return 0.5
    return 0.0


def _check_advance(win_history_heuristic: deque, rolling_win: int) -> bool:
    """Return True when advancement criterion is met."""
    if len(win_history_heuristic) < rolling_win:
        return False
    recent = list(win_history_heuristic)[-rolling_win:]
    wr = sum(1 for x in recent if x == 1.0) / len(recent)
    dr = sum(1 for x in recent if x == 0.5) / len(recent)
    return (wr >= 0.30 and dr >= 0.30) or wr >= 0.50


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
        enc = encode_position_with_lookahead(board, player,
                                             sentinel_advisor=self._sentinel,
                                             db=None,
                                             value_net=self._value_net,
                                             lookahead_advisor=None)
        if enc is None or not enc.legal_moves:
            return {}
        feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model.policy_logits(feat_t)
            idx    = int(torch.argmax(logits).item())
        return enc.legal_moves[idx]


# ── Single-game rollout ────────────────────────────────────────────────────────

RETRY_PLY_MIN =  5
RETRY_PLY_MAX = 15

@dataclass
class RolloutResult:
    trajectory:        list[ScaffoldedStep]
    step_diags:        list[StepDiag]
    outcome:           float
    ply:               int
    branch_candidates: list[tuple[int, BoardState, str]]
    retry_board:       Optional[BoardState] = None


def _rollout(
    model:          ScaffoldedPolicyNet,
    device:         torch.device,
    start_board:    BoardState,
    learner_color:  str,
    opponent,
    opp_color:      str,
    sentinel,
    db,
    value_net,
    temperature:    float,
    max_ply:        int,
    record_branches: bool,
    branch_every:   int,
    retry_ply:      int,
    forced_placements: Optional[list[str]] = None,
    lookahead_advisor=None,
) -> RolloutResult:
    board                   = start_board
    ply                     = 0
    move_phase_start_ply:   Optional[int] = None
    game_trajectory:        list[ScaffoldedStep] = []
    step_diags:             list[StepDiag]       = []
    branch_candidates:      list[tuple[int, BoardState, str]] = []
    done                    = False
    outcome                 = 0.0
    learner_move_count      = 0
    learner_placement_count = 0
    retry_board: Optional[BoardState] = None
    deferred_opp_penalty:   float = 0.0

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
            enc = encode_position_with_lookahead(board, player,
                                                  sentinel_advisor=sentinel, db=None,
                                                  value_net=value_net,
                                                  lookahead_advisor=lookahead_advisor)
            if enc is None or not enc.legal_moves:
                outcome = LOSS_REWARD
                done    = True
                break

            feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits    = model.policy_logits(feat_t)
                scaled    = logits / max(temperature, 1e-6)
                log_probs = F.log_softmax(scaled, dim=-1)
                probs     = log_probs.exp()
                if not torch.isfinite(probs).all():
                    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs     = probs / probs.sum().clamp(min=1e-9)
                entropy   = float((-(probs * log_probs).sum()).item())

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
                chosen_prob     = float(probs[chosen_idx].item())
                top1_prob       = float(probs.max().item())
                was_top1_policy = int(chosen_idx == int(torch.argmax(probs).item()))
                log_prob_old    = float(log_probs[chosen_idx].item())

            move = enc.legal_moves[chosen_idx]
            if board.phase == "place":
                learner_placement_count += 1
            board_after = board.apply_move(move)
            enc_after   = encode_position_with_lookahead(board_after, opp_color,
                                                         sentinel_advisor=sentinel, db=None,
                                                         value_net=value_net,
                                                         lookahead_advisor=None)

            # Compute midgame gate with ply guard — rewards only after movement turn 10
            total_pieces = board.pieces_on_board.get("W", 0) + board.pieces_on_board.get("B", 0)
            _moves_into_mv = (ply - move_phase_start_ply) if move_phase_start_ply is not None else 0
            in_midgame = (
                total_pieces >= 12
                and board.phase != "place"
                and _moves_into_mv >= MIDGAME_PLY_GATE
            )

            reward, rb, extra = _compute_per_move_reward(
                enc, chosen_idx, enc_after,
                db_moves=None,
                board=board,
                in_midgame=in_midgame,
            )

            # Apply deferred penalty from opponent's previous mill activity
            if deferred_opp_penalty > 0.0:
                reward              -= deferred_opp_penalty
                rb.opp_penalty      -= deferred_opp_penalty
                rb.total            -= deferred_opp_penalty
                deferred_opp_penalty = 0.0

            # Mill formation bonus — un-gated (mills matter everywhere)
            mills_before = sum(1 for m in MILLS if all(board.positions.get(p) == learner_color for p in m))
            mills_after  = sum(1 for m in MILLS if all(board_after.positions.get(p) == learner_color for p in m))
            if mills_after > mills_before:
                mill_bonus = MILL_BONUS * (mills_after - mills_before)
                reward    += mill_bonus
                rb.mill_formed += mill_bonus
                rb.total  += mill_bonus

            # Mill dance / block / dual-mill rewards — un-gated
            src = move.get("from")
            dst = move.get("to")
            if src and dst and board.phase != "place":
                # Dance: source piece was in a completed mill, dest closes a different mill
                if _piece_in_any_mill(board, src, learner_color):
                    closes_other = any(
                        dst in m and src not in m and
                        all(board.positions.get(p) == learner_color for p in m if p != dst)
                        for m in MILLS
                    )
                    if closes_other:
                        reward        += MILL_DANCE_REWARD
                        rb.mill_dance += MILL_DANCE_REWARD
                        rb.total      += MILL_DANCE_REWARD

                # Block: learner occupies a square the opponent would use to close a mill
                if _would_close_mill(board, dst, opp_color):
                    reward        += BLOCK_OPP_DANCE_REWARD
                    rb.mill_dance += BLOCK_OPP_DANCE_REWARD
                    rb.total      += BLOCK_OPP_DANCE_REWARD

            # Dual-mill toggle: learner creates a new double-mill configuration
            if board.phase != "place":
                double_before = _pieces_in_double_mill(board, learner_color)
                double_after  = _pieces_in_double_mill(board_after, learner_color)
                if double_after - double_before:
                    reward        += DUAL_MILL_CONFIG_REWARD
                    rb.mill_dance += DUAL_MILL_CONFIG_REWARD
                    rb.total      += DUAL_MILL_CONFIG_REWARD

            # Malom per-move reward/penalty — gated to midgame
            if db is not None and in_midgame:
                try:
                    opp_state_wdl = db.query_state(board_after)
                    if opp_state_wdl:
                        extra["malom_chosen_wdl"] = opp_state_wdl
                    if opp_state_wdl == "L":      # opponent losing → we are winning
                        reward        += DELTA
                        rb.malom_trap += DELTA
                        rb.total      += DELTA
                    elif opp_state_wdl == "W":    # opponent winning → we are losing
                        reward        -= MALOM_LOSS_PENALTY
                        rb.malom_trap -= MALOM_LOSS_PENALTY
                        rb.total      -= MALOM_LOSS_PENALTY
                except Exception:
                    pass

            # Mobility reward — gated to midgame, requires enc_after
            if in_midgame and enc_after is not None:
                learner_legal = len(enc.legal_moves)
                opp_legal     = len(enc_after.legal_moves) if enc_after.legal_moves else 0
                mobility_delta = learner_legal - opp_legal
                mob_reward = MOBILITY_COEF * math.tanh(mobility_delta / 5.0)
                reward        += mob_reward
                rb.heuristic  += mob_reward   # fold into heuristic slot for logging
                rb.total      += mob_reward

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

            sentinel_scores = list(getattr(enc, "sentinel_scores", []) or [])
            sentinel_mean   = float(sum(sentinel_scores) / len(sentinel_scores)) if sentinel_scores else 0.0
            sentinel_chosen = float(sentinel_scores[chosen_idx]) if sentinel_scores else 0.0
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

            learner_move_count += 1
            if record_branches and branch_every > 0 and (learner_move_count % branch_every == 0):
                moves_into_movement = (ply - move_phase_start_ply) if move_phase_start_ply is not None else None
                branch_candidates.append((ply, board, _phase_bucket(board, moves_into_movement)))

            board = board_after

        else:
            try:
                opp_move = opponent.choose_move(board)
            except Exception:
                opp_move = None
            if not opp_move:
                outcome = WIN_REWARD
                done    = True
                break
            opp_mills_before = _count_mills(board, opp_color)
            board = board.apply_move(opp_move)
            opp_mills_after  = _count_mills(board, opp_color)
            if opp_mills_after > opp_mills_before:
                deferred_opp_penalty += OPP_MILL_PENALTY
            elif opp_mills_after < opp_mills_before:
                deferred_opp_penalty += OPP_MILL_DANCE_PENALTY

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
    sd       = result.step_diags
    win_rate = sum(1 for x in win_history if x == 1.0) / max(len(win_history), 1)
    return GameDiag(
        game=game_count,
        difficulty=difficulty,
        learner_color=learner_color,
        temperature=round(temperature, 4),
        outcome=float(result.outcome),
        win_rate_200=round(win_rate, 4),
        ply=int(result.ply),
        steps=len(sd),
        update_policy_loss=None if last_update_pl  is None else float(last_update_pl),
        update_value_loss =None if last_update_vl  is None else float(last_update_vl),
        update_entropy    =None if last_update_ent is None else float(last_update_ent),
        reward_total_mean    =_safe_mean([d.reward.total      for d in sd]),
        reward_sentinel_mean =_safe_mean([d.reward.sentinel   for d in sd]),
        reward_heuristic_mean=_safe_mean([d.reward.heuristic  for d in sd]),
        reward_value_mean    =_safe_mean([d.reward.value_net  for d in sd]),
        reward_malom_win_mean=_safe_mean([d.reward.malom_win  for d in sd]),
        reward_malom_trap_mean=_safe_mean([d.reward.malom_trap for d in sd]),
        reward_retro_mean    =_safe_mean([d.reward.retro      for d in sd]),
        sentinel_mean        =_safe_mean([d.sentinel_mean     for d in sd]),
        sentinel_chosen_mean =_safe_mean([d.sentinel_chosen   for d in sd]),
        h_delta_mean         =_safe_mean([d.h_delta           for d in sd]),
        vn_delta_mean        =_safe_mean([d.vn_delta          for d in sd]),
        chosen_prob_mean     =_safe_mean([d.chosen_prob       for d in sd]),
        entropy_mean         =_safe_mean([d.entropy           for d in sd]),
        top1_prob_mean       =_safe_mean([d.top1_prob         for d in sd]),
        legal_moves_mean     =_safe_mean([float(d.legal_moves) for d in sd]),
        policy_top1_rate     =_safe_mean([float(d.was_top1_policy)     for d in sd]),
        heuristic_top1_rate  =_safe_mean([float(d.was_top1_heuristic)  for d in sd]),
        malom_win_move_rate  =_safe_mean([1.0 if d.malom_chosen_wdl == "win" else 0.0 for d in sd]),
        malom_unknown_rate   =_safe_mean([1.0 if d.malom_chosen_wdl == "unknown" else 0.0 for d in sd]),
        best_win_rate  =float(best_win_rate),
        temp_frozen    =int(temp_frozen),
        lr             =float(opt.param_groups[0]["lr"]),
        source_checkpoint=source_ckpt,
        game_type      =game_type,
        phase_bucket   =phase_bucket,
        is_branch      =int(is_branch),
        branch_ply_start=branch_ply_start,
        target_age     =target_age,
        bucket_opening =bucket_counts.get("opening",  0),
        bucket_midgame =bucket_counts.get("midgame",  0),
        bucket_endgame =bucket_counts.get("endgame",  0),
        reward_mill_dance_mean  =_safe_mean([d.reward.mill_dance   for d in sd]),
        reward_opp_penalty_mean =_safe_mean([d.reward.opp_penalty  for d in sd]),
    )


# ── Main training loop ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s_mid] Device: {device}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load components ────────────────────────────────────────────────────────
    sentinel = None
    sentinel_path = args.sentinel
    if sentinel_path and Path(sentinel_path).exists():
        try:
            sentinel = load_advisor(sentinel_path)
            print(f"[s_mid] Sentinel loaded (features only, no reward): {sentinel_path}")
        except Exception as e:
            print(f"[s_mid] Sentinel load failed ({e}) — sentinel features will be 0.5")
    else:
        print("[s_mid] No sentinel found — sentinel features will be 0.5")

    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s_mid] Malom DB loaded: {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s_mid] Malom DB failed ({e})")
    if db is None:
        print("[s_mid] Malom DB unavailable — Malom trap reward = 0")

    value_net = None
    vn_path = args.value_net or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[s_mid] Value net loaded: {vn_path}")
        except Exception as e:
            print(f"[s_mid] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[s_mid] No value net — VN features will be 0")

    # ── LookaheadAdvisor ───────────────────────────────────────────────────────
    lookahead_advisor: Optional[LookaheadAdvisor] = None
    if not getattr(args, "no_lookahead", False):
        from learned_ai.agents.heuristic_agent import get_heuristic_evaluate as _get_eval
        _evaluate_fn = _get_eval()
        lookahead_advisor = LookaheadAdvisor(
            sentinel=sentinel,
            value_net=value_net,
            evaluate_fn=_evaluate_fn,
            use_sentinel=True,
        )
        print("[s_mid] LookaheadAdvisor enabled (5-ply heuristic+sentinel+VN)")
    else:
        print("[s_mid] LookaheadAdvisor disabled (--no-lookahead)")

    # ── Load model ─────────────────────────────────────────────────────────────
    resume_path, source_tag = _choose_resume_path(args)
    model, start_game, best_win_rate, difficulty, source_checkpoint = _load_model(
        device, resume_path
    )
    difficulty = _apply_diff_start_override(difficulty, args)
    if resume_path is None:
        print("[s_mid] No checkpoint found — starting from scratch")
    else:
        print(f"[s_mid] Resuming from ({source_tag}): {resume_path}")
    print(f"[s_mid] Starting at game {start_game}, difficulty {difficulty}")

    # ── Frozen opponent ────────────────────────────────────────────────────────
    frozen_opp = FrozenModelOpponent(model, device, sentinel=sentinel, value_net=value_net)
    games_since_target_update = 0

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    game_count             = start_game
    temperature            = args.temp_start
    win_history:             deque[float] = deque(maxlen=args.rolling_win)
    win_history_heuristic:   deque[float] = deque(maxlen=args.rolling_win)
    mixed_win_history:       deque[float] = deque(maxlen=50)
    malom_win_rate_history:  deque[float] = deque(maxlen=10)
    ep_steps: list[ScaffoldedStep] = []
    last_update_pl  = None
    last_update_vl  = None
    last_update_ent = None
    best_win_rate_at_diff = 0.0

    branch_bucket_history: deque[str] = deque(maxlen=args.bucket_window)

    log_path        = out_dir / "train_log.jsonl"
    update_log_path = out_dir / "update_log.jsonl"

    print(f"[s_mid] Starting at game {game_count}, difficulty {difficulty}")
    print(f"[s_mid] Self-play ratio {args.self_play_ratio:.0%}, "
          f"branch every {args.branch_every} turns, "
          f"max {args.max_branches_per_game} branches/game")

    # ── Position pool ──────────────────────────────────────────────────────────
    midgame_pool: list = []
    if not args.no_position_pool:
        try:
            from learned_ai.training.position_pool import load_position_pool
            midgame_pool = load_position_pool(
                _ROOT, phase="midgame", movement_turn=10, window=2,
            )
            print(f"[s_mid] Position pool: {len(midgame_pool)} midgame positions (movement turn 8-12)")
        except Exception as e:
            print(f"[s_mid] Position pool load failed ({e}) — starting all games from new_game()")

    if not args.no_s1b_refresher:
        print(f"[s_mid] Running s1b refresher before diff {difficulty} training")
        _run_s1b_refresher(model, device, args.s1b_data,
                           epochs=args.s1b_refresher_epochs,
                           lr=args.s1b_refresher_lr)

    diag_buffer: list[GameDiag] = []

    while game_count < args.max_games:
        temperature = _compute_temperature(game_count, args.max_games)

        if games_since_target_update >= args.update_target_every:
            frozen_opp.refresh(model)
            games_since_target_update = 0
            print(f"[s_mid] Frozen model updated at game {game_count}")

        if midgame_pool and rng.random() < 0.70:
            start_board   = midgame_pool[rng.randint(0, len(midgame_pool) - 1)]
            learner_color = start_board.turn
        else:
            start_board   = BoardState.new_game()
            learner_color = "W" if rng.random() < 0.5 else "B"
        opp_color = "B" if learner_color == "W" else "W"

        use_self_play = rng.random() < args.self_play_ratio
        if use_self_play:
            opponent      = frozen_opp
            game_type     = "vs_frozen"
            game_difficulty = difficulty
        else:
            from learned_ai.agents.heuristic_agent import GameAI as _GA
            game_difficulty = difficulty
            # 15% of non-self-play games: use a randomly lower difficulty
            if difficulty > 1 and rng.random() < 0.15:
                game_difficulty = rng.randint(1, difficulty - 1)
            _h = HeuristicAgent(color=opp_color, difficulty=game_difficulty, game_ai=None)
            _h._inner = _GA(color=opp_color, difficulty=game_difficulty, override_time_budget=args.time_budget)
            opponent  = _h
            game_type = "vs_heuristic"

        game_forced_placements: Optional[list[str]] = None
        if _OPENING_LINES and rng.random() < BOOK_GAME_PROB:
            line = _OPENING_LINES[rng.randint(0, len(_OPENING_LINES) - 1)]
            game_forced_placements = _sample_forced_placements(line, learner_color)

        game_retry_ply = rng.randint(RETRY_PLY_MIN, RETRY_PLY_MAX)
        result = _rollout(
            model=model,
            device=device,
            start_board=start_board,
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
            lookahead_advisor=lookahead_advisor,
        )

        if result.trajectory:
            _retroactive_rescore(result.trajectory, result.step_diags, result.outcome)

        is_full_diff = (game_type == "vs_heuristic" and game_difficulty == difficulty)

        # WIN: always learn. LOSS/DRAW_SHORT: run confirmation retry.
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
                lookahead_advisor=lookahead_advisor,
            )
            if confirm_result.trajectory:
                _retroactive_rescore(confirm_result.trajectory, confirm_result.step_diags,
                                     confirm_result.outcome)
            confirmed = (
                (result.outcome == LOSS_REWARD and confirm_result.outcome == LOSS_REWARD) or
                (result.outcome == DRAW_SHORT  and confirm_result.outcome == DRAW_SHORT)
            )
            if confirmed and result.trajectory:
                ep_steps.extend(result.trajectory)
            if confirm_result.outcome in (WIN_REWARD, DRAW_SHORT):
                ep_steps.extend(confirm_result.trajectory)
            game_count += 1
            games_since_target_update += 1
            _hv = _outcome_to_history_float(confirm_result.outcome)
            win_history.append(_hv)
            if is_full_diff:
                win_history_heuristic.append(_hv)
            elif game_type == "vs_heuristic":
                mixed_win_history.append(_hv)
            _coc = "W" if confirm_result.outcome == WIN_REWARD else ("L" if confirm_result.outcome == LOSS_REWARD else "D")
            if game_count % 10 == 0:
                print(f"[s_mid] {game_count:6d}  r{game_retry_ply:2d} {learner_color} |          | {_coc} ply={confirm_result.ply:3d} | (from ply {game_retry_ply}) {'[learn]' if confirmed else '[skip]'}")

        _hv = _outcome_to_history_float(result.outcome)
        win_history.append(_hv)
        if is_full_diff:
            win_history_heuristic.append(_hv)
        elif game_type == "vs_heuristic":
            mixed_win_history.append(_hv)
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
            recent_h = list(win_history_heuristic)
            hwr = sum(1 for x in recent_h if x == 1.0) / max(len(recent_h), 1)
            hdr = sum(1 for x in recent_h if x == 0.5) / max(len(recent_h), 1)
            _awr = sum(1 for x in win_history if x == 1.0) / max(len(win_history), 1)
            _oc  = "W" if result.outcome == WIN_REWARD else ("L" if result.outcome == LOSS_REWARD else "D")
            _gt  = "heur" if game_type == "vs_heuristic" else "self"
            _dif = f"d{game_difficulty}" if game_difficulty != difficulty else f"diff {difficulty}"
            print(f"[s_mid] {game_count:6d} {_gt:4s} {learner_color} | {_dif} | {_oc} ply={result.ply:3d} | hwr={hwr:.3f} hdr={hdr:.3f} awr={_awr:.3f} | temp={temperature:.2f} lr={opt.param_groups[0]['lr']:.5f}")

        # ── Loss/draw retry ────────────────────────────────────────────────────
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
                lookahead_advisor=lookahead_advisor,
            )
            if retry_result.trajectory:
                _retroactive_rescore(retry_result.trajectory, retry_result.step_diags, retry_result.outcome)
                if retry_result.outcome in (WIN_REWARD, DRAW_SHORT):
                    ep_steps.extend(retry_result.trajectory)
            _rv = _outcome_to_history_float(retry_result.outcome)
            win_history.append(_rv)
            if is_full_diff:
                win_history_heuristic.append(_rv)
            elif game_type == "vs_heuristic":
                mixed_win_history.append(_rv)
            game_count += 1
            games_since_target_update += 1
            _roc = "W" if retry_result.outcome == WIN_REWARD else ("L" if retry_result.outcome == LOSS_REWARD else "D")
            if game_count % 10 == 0:
                print(f"[s_mid] {game_count:6d} retry {learner_color} |          | {_roc} ply={retry_result.ply:3d} | (from ply {game_retry_ply})")

        # ── Branch games ───────────────────────────────────────────────────────
        branches_spawned = 0
        candidates = list(result.branch_candidates)
        rng.shuffle(candidates)
        seen_buckets: set[str] = set()
        ordered_candidates: list[tuple[int, BoardState, str]] = []
        for cand in candidates:
            if cand[2] not in seen_buckets:
                ordered_candidates.insert(0, cand)
                seen_buckets.add(cand[2])
            else:
                ordered_candidates.append(cand)

        for branch_ply, branch_board, bucket in ordered_candidates:
            if branches_spawned >= args.max_branches_per_game:
                break
            bucket_counts = Counter(branch_bucket_history)
            if bucket_counts.get(bucket, 0) >= args.max_per_bucket:
                continue

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
                record_branches=False,
                branch_every=0,
                retry_ply=0,
                lookahead_advisor=lookahead_advisor,
            )

            if branch_result.trajectory:
                _retroactive_rescore(branch_result.trajectory, branch_result.step_diags, branch_result.outcome)
                if branch_result.outcome in (WIN_REWARD, DRAW_SHORT):
                    ep_steps.extend(branch_result.trajectory)
                branch_bucket_history.append(bucket)
                branches_spawned += 1
                game_count += 1
                games_since_target_update += 1
                win_history.append(_outcome_to_history_float(branch_result.outcome))

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
                    print(f"[s_mid] {game_count:6d}  +b  {learner_color} | {bucket:7s} | {_boc} ply={branch_result.ply:3d} | (from ply {branch_ply})")

        # ── Update ─────────────────────────────────────────────────────────────
        if len(ep_steps) >= args.update_every:
            last_update_pl, last_update_vl, last_update_ent = update_fn(
                model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef
            )
            upd_entry = {
                "game":        game_count,
                "policy_loss": None if last_update_pl  is None else float(last_update_pl),
                "value_loss":  None if last_update_vl  is None else float(last_update_vl),
                "entropy":     None if last_update_ent is None else float(last_update_ent),
                "lr":          float(opt.param_groups[0]["lr"]),
                "batch_steps": len(ep_steps),
            }
            with open(update_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(upd_entry) + "\n")
            ep_steps.clear()

        # ── Periodic log + checkpoint ──────────────────────────────────────────
        if game_count % args.log_every == 0 and diag_buffer:
            recent_h     = list(win_history_heuristic)
            win_rate     = sum(1 for x in recent_h if x == 1.0) / max(len(recent_h), 1)
            draw_rate    = sum(1 for x in recent_h if x == 0.5) / max(len(recent_h), 1)
            win_rate_all = sum(1 for x in win_history  if x == 1.0) / max(len(win_history), 1)

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
                    print(f"[s_mid] Recovery: reloaded best{difficulty}.pt (win rate was {win_rate:.2f}, temp reset to {TEMP_START})")

            main_diags   = [d for d in diag_buffer if not d.is_branch]
            branch_diags = [d for d in diag_buffer if d.is_branch]
            bc = Counter(branch_bucket_history)

            with open(log_path, "a", encoding="utf-8") as f:
                for d in diag_buffer:
                    f.write(json.dumps(asdict(d)) + "\n")
            diag_buffer.clear()

            last_main = next((d for d in reversed(main_diags) if main_diags), None)
            if last_main:
                d = last_main
                _sign = lambda v: f"{'+' if v >= 0 else ''}{v:.3f}"
                print(
                    f"[s_mid] game {game_count:6d} | diff {difficulty} | "
                    f"win={win_rate:.3f} draw={draw_rate:.3f} all={win_rate_all:.3f} | "
                    f"temp={temperature:.2f} | "
                    f"outcome={d.outcome:+.2f} | lr={opt.param_groups[0]['lr']:.5f} | "
                    f"rew={_sign(d.reward_total_mean)} | "
                    f"h={_sign(d.reward_heuristic_mean)} "
                    f"mt={_sign(d.reward_malom_trap_mean)} | "
                    f"p_top1={d.policy_top1_rate:.2f} h_top1={d.heuristic_top1_rate:.2f} | "
                    f"branches={len(branch_diags)} "
                    f"[op={bc.get('opening',0)} mid={bc.get('midgame',0)} end={bc.get('endgame',0)}]"
                )

            ckpt = {
                "model":             model.state_dict(),
                "model_config":      model.get_config(),
                "stage":             STAGE_TAG,
                "game_count":        game_count,
                "best_win_rate":     best_win_rate,
                "difficulty":        difficulty,
                "source_checkpoint": source_checkpoint,
                "lr":                float(opt.param_groups[0]["lr"]),
                "temperature":       float(temperature),
            }
            torch.save(ckpt, out_dir / "latest.pt")

            if win_rate > best_win_rate_at_diff and len(win_history_heuristic) >= 10:
                best_win_rate_at_diff = win_rate
                ckpt["best_win_rate"] = best_win_rate_at_diff
                torch.save(ckpt, out_dir / f"best{difficulty}.pt")
                torch.save(ckpt, out_dir / "best.pt")
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                print(f"[s_mid]  → best diff-{difficulty} win rate: {best_win_rate_at_diff:.3f}  (saved best{difficulty}.pt)")

        # ── Difficulty advancement ─────────────────────────────────────────────
        if _check_advance(win_history_heuristic, args.rolling_win):
            if difficulty >= args.diff_max:
                recent_h = list(win_history_heuristic)[-args.rolling_win:]
                wr = sum(1 for x in recent_h if x == 1.0) / len(recent_h)
                print(f"[s_mid] *** {wr:.3f} win rate vs difficulty {difficulty} — done! ***")
                break
            else:
                recent_h = list(win_history_heuristic)[-args.rolling_win:]
                wr = sum(1 for x in recent_h if x == 1.0) / len(recent_h)
                prev_diff = difficulty
                difficulty += 1
                win_history.clear()
                win_history_heuristic.clear()
                print(f"[s_mid] *** Advanced to difficulty {difficulty} (was wr={wr:.3f} vs diff {prev_diff}) ***")

                prev_best = out_dir / f"best{prev_diff}.pt"
                if not prev_best.exists():
                    _adv_ckpt = {
                        "model":             model.state_dict(),
                        "model_config":      model.get_config(),
                        "stage":             STAGE_TAG,
                        "game_count":        game_count,
                        "best_win_rate":     wr,
                        "difficulty":        prev_diff,
                        "source_checkpoint": source_checkpoint,
                        "lr":                float(opt.param_groups[0]["lr"]),
                        "temperature":       float(temperature),
                    }
                    torch.save(_adv_ckpt, prev_best)
                    print(f"[s_mid] Saved best{prev_diff}.pt at advancement point (wr={wr:.3f})")
                if prev_best.exists():
                    ckpt_prev = torch.load(str(prev_best), map_location=device, weights_only=False)
                    model.load_state_dict(ckpt_prev["model"])
                    model.to(device)
                    print(f"[s_mid] Loaded best{prev_diff}.pt as starting point for diff {difficulty}")

                best_win_rate_at_diff = 0.0
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                frozen_opp.refresh(model)

                if not args.no_s1b_refresher:
                    print(f"[s_mid] Running s1b refresher before diff {difficulty} training")
                    _run_s1b_refresher(model, device, args.s1b_data,
                                       epochs=args.s1b_refresher_epochs,
                                       lr=args.s1b_refresher_lr)

    # ── Final flush ────────────────────────────────────────────────────────────
    if ep_steps:
        update_fn(model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef)
    if diag_buffer:
        with open(log_path, "a", encoding="utf-8") as f:
            for d in diag_buffer:
                f.write(json.dumps(asdict(d)) + "\n")

    ckpt = {
        "model":             model.state_dict(),
        "model_config":      model.get_config(),
        "stage":             STAGE_TAG,
        "game_count":        game_count,
        "best_win_rate":     best_win_rate,
        "difficulty":        difficulty,
        "source_checkpoint": source_checkpoint,
        "lr":                float(opt.param_groups[0]["lr"]),
        "temperature":       float(temperature),
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s_mid] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s_mid] Checkpoint: {out_dir / 'best.pt'}")
    print(f"[s_mid] Logs: {log_path} and {update_log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Midgame specialist: movement phase >= 12 pieces")
    p.add_argument("--resume",             default="",   type=str, help="Explicit checkpoint path")
    p.add_argument("--auto-resume-best",   action="store_true", help="Prefer s_mid/best.pt in resume chain")
    p.add_argument("--out-dir",  default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s_mid"))
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
    p.add_argument("--diff-start",          type=int,   default=None,
                   help="Override starting difficulty")
    p.add_argument("--diff-max",            type=int,   default=DIFF_MAX,
                   help="Highest difficulty to train against (default 7)")
    p.add_argument("--temp-start",          type=float, default=TEMP_START)
    p.add_argument("--log-every",           type=int,   default=LOG_EVERY)
    p.add_argument("--max-ply",             type=int,   default=MAX_PLY)
    p.add_argument("--max-ply-branch",      type=int,   default=MAX_PLY_BRANCH)
    p.add_argument("--time-budget",         type=float, default=TIME_BUDGET)
    p.add_argument("--self-play-ratio",     type=float, default=SELF_PLAY_RATIO)
    p.add_argument("--update-target-every", type=int,   default=UPDATE_TARGET_EVERY)
    p.add_argument("--branch-every",        type=int,   default=BRANCH_EVERY)
    p.add_argument("--max-branches-per-game", type=int, default=0)
    p.add_argument("--bucket-window",       type=int,   default=BUCKET_WINDOW)
    p.add_argument("--max-per-bucket",      type=int,   default=MAX_PER_BUCKET)
    p.add_argument("--s1b-data",             type=str,  default=str(_ROOT / "learned_ai" / "data" / "human_imitation.npz"))
    p.add_argument("--s1b-refresher-epochs", type=int,  default=S1B_REFRESHER_EPOCHS)
    p.add_argument("--s1b-refresher-lr",     type=float,default=S1B_REFRESHER_LR)
    p.add_argument("--no-s1b-refresher",     action="store_true")
    p.add_argument("--no-position-pool",     action="store_true",
                   help="Disable position pool — start all games from new_game()")
    p.add_argument("--no-lookahead",         action="store_true",
                   help="Disable 5-ply LookaheadAdvisor (faster, no lookahead features)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
