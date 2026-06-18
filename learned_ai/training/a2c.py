"""learned_ai/training/a2c.py — Actor-Critic (A2C) update for NMM self-play.

Solves the REINFORCE variance problem via per-step TD bootstrapping:

    advantage(t) = r(t) + γ·V(s_{t+1})·(1 − done(t)) − V(s(t))

Every learner move gets a dense gradient signal. The value head trains on
bootstrapped TD targets rather than terminal-only discounted returns.

This module is *algorithm only* — it does not contain the episode runner or
the main training loop (those live in train_stage2.py).

Key design notes:
- next_state is always the board state at the *next learner turn* after both
  the learner and opponent have played one move each, except at terminal steps.
- Malom shaping rewards slot into r(t) naturally; they are already included
  in the A2CStep.reward field before this function is called.
- V(s) is estimated from the side-to-move's perspective (consistent with
  Stage 0 pre-training which labeled values as value_net.predict(board, stm)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from learned_ai.models.state_encoder import PHASE_NAMES

ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
GRAD_CLIP    = 1.0


@dataclass
class A2CStep:
    """One learner-turn step collected during self-play."""
    state:          torch.Tensor   # [84] — board when learner moved
    phase_id:       int
    action_idx:     int            # primary action index [0, 624)
    legal_mask:     torch.Tensor   # [624] bool
    reward:         float          # Malom shaping + terminal component
    next_state:     torch.Tensor   # [84] — board at next learner turn (or terminal state)
    next_phase_id:  int
    next_legal_mask: torch.Tensor  # [624] bool for next state
    done:           bool           # True when game ended in this step
    log_prob_old:   float          # detached log_prob at collection time (for PPO ratio)


def a2c_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: List[A2CStep],
    device: torch.device,
    gamma: float = 0.99,
    entropy_coef: float = ENTROPY_COEF,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    min_batch: int = 8,
) -> tuple[float, float, float]:
    """One A2C gradient update over a batch of A2CSteps.

    Returns (policy_loss, value_loss, entropy) as Python floats.
    Returns (0, 0, 0) if batch is too small.
    """
    if len(steps) < min_batch:
        return 0.0, 0.0, 0.0

    states      = torch.stack([s.state      for s in steps]).to(device)       # [B, 84]
    next_states = torch.stack([s.next_state for s in steps]).to(device)       # [B, 84]
    legal_masks = torch.stack([s.legal_mask for s in steps]).to(device)       # [B, 624]
    actions     = torch.tensor([s.action_idx for s in steps],
                               device=device, dtype=torch.long)               # [B]
    rewards     = torch.tensor([s.reward for s in steps],
                               device=device, dtype=torch.float32)            # [B]
    dones       = torch.tensor([s.done for s in steps],
                               device=device, dtype=torch.float32)            # [B]
    phase_ids   = [s.phase_id for s in steps]

    model.train()

    # ── Bootstrap: V(next_state) with no gradient ─────────────────────────────
    with torch.no_grad():
        next_feats = model.backbone(next_states)
        v_next = model.value_head(next_feats).squeeze(-1)
        v_next = v_next * (1.0 - dones)  # zero out terminal bootstraps

    td_targets = rewards + gamma * v_next  # [B]

    # ── Current-state features (with gradient) ────────────────────────────────
    feats = model.backbone(states)
    v_curr = model.value_head(feats).squeeze(-1)  # [B]

    # ── Advantages (detached — policy and value losses are independent) ────────
    advantages = (td_targets - v_curr).detach()
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── Policy loss (phase-routed) ─────────────────────────────────────────────
    pl_sum  = torch.zeros([], device=device)
    ent_sum = torch.zeros([], device=device)
    n_total = 0

    for ph in range(model.num_phases):
        idx = [i for i, p in enumerate(phase_ids) if p == ph]
        if not idx:
            continue
        idx_t   = torch.tensor(idx, device=device)
        logits  = model.phase_heads[PHASE_NAMES[ph]](feats[idx_t])
        logits  = logits.masked_fill(~legal_masks[idx_t], -1e9)
        lp      = F.log_softmax(logits, dim=-1)
        sel_lp  = lp.gather(1, actions[idx_t].unsqueeze(1)).squeeze(1)
        pl_sum  = pl_sum - (sel_lp * advantages[idx_t]).sum()
        probs   = lp.exp()
        ent_sum = ent_sum + (-(probs * lp).sum(dim=-1)).sum()
        n_total += len(idx)

    policy_loss  = pl_sum  / max(n_total, 1)
    entropy_loss = ent_sum / max(n_total, 1)
    value_loss   = F.mse_loss(v_curr, td_targets.detach())

    loss = policy_loss - entropy_coef * entropy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item()), float(entropy_loss.item())
