"""learned_ai/training/ppo.py — PPO update for NMM self-play.

Proximal Policy Optimization over a batch of A2CSteps (same data structure as
the A2C module). PPO extends A2C with:

1. Clipped surrogate objective — prevents large policy updates that cause
   catastrophic forgetting:
     ratio = π_new(a|s) / π_old(a|s)
     L_clip = min(ratio · adv, clip(ratio, 1-ε, 1+ε) · adv)

2. Multiple gradient epochs per data batch — amortises collection cost.

Both algorithms share the same A2CStep collection logic in train_stage2.py.
Pass --ppo to the training script to switch from A2C to PPO.

IMPORTANT: A2CStep.log_prob_old must be stored as a *detached float* at
collection time (not a live tensor) so multi-epoch PPO backprop does not try
to retain the stale computation graph.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from learned_ai.models.state_encoder import PHASE_NAMES
from learned_ai.training.a2c import A2CStep

PPO_CLIP_EPS  = 0.2
PPO_EPOCHS    = 4
ENTROPY_COEF  = 0.01
VALUE_COEF    = 0.5
GRAD_CLIP     = 1.0


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: List[A2CStep],
    device: torch.device,
    gamma: float = 0.99,
    clip_eps: float = PPO_CLIP_EPS,
    n_epochs: int = PPO_EPOCHS,
    entropy_coef: float = ENTROPY_COEF,
    value_coef: float = VALUE_COEF,
    grad_clip: float = GRAD_CLIP,
    min_batch: int = 8,
) -> tuple[float, float, float]:
    """PPO update over a batch of A2CSteps, with n_epochs passes.

    Returns (avg_policy_loss, avg_value_loss, avg_entropy) across all epochs.
    """
    if len(steps) < min_batch:
        return 0.0, 0.0, 0.0

    states      = torch.stack([s.state      for s in steps]).to(device)
    next_states = torch.stack([s.next_state for s in steps]).to(device)
    legal_masks = torch.stack([s.legal_mask for s in steps]).to(device)
    actions     = torch.tensor([s.action_idx for s in steps],
                               device=device, dtype=torch.long)
    rewards     = torch.tensor([s.reward for s in steps],
                               device=device, dtype=torch.float32)
    dones       = torch.tensor([s.done for s in steps],
                               device=device, dtype=torch.float32)
    # old log-probs stored as detached floats at collection time
    old_lp      = torch.tensor([s.log_prob_old for s in steps],
                               device=device, dtype=torch.float32)
    phase_ids   = [s.phase_id for s in steps]

    # ── Compute TD targets and advantages once (fixed across epochs) ──────────
    with torch.no_grad():
        next_feats = model.backbone(next_states)
        v_next = model.value_head(next_feats).squeeze(-1) * (1.0 - dones)
    td_targets = rewards + gamma * v_next

    with torch.no_grad():
        feats_ref = model.backbone(states)
        v_ref = model.value_head(feats_ref).squeeze(-1)
    advantages = (td_targets - v_ref)
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_pl = 0.0
    total_vl = 0.0
    total_ent = 0.0

    for _ in range(n_epochs):
        model.train()
        feats = model.backbone(states)
        v_curr = model.value_head(feats).squeeze(-1)

        pl_sum  = torch.zeros([], device=device)
        ent_sum = torch.zeros([], device=device)
        n_total = 0

        for ph in range(model.num_phases):
            idx = [i for i, p in enumerate(phase_ids) if p == ph]
            if not idx:
                continue
            idx_t  = torch.tensor(idx, device=device)
            logits = model.phase_heads[PHASE_NAMES[ph]](feats[idx_t])
            logits = logits.masked_fill(~legal_masks[idx_t], -1e9)
            lp     = F.log_softmax(logits, dim=-1)
            sel_lp = lp.gather(1, actions[idx_t].unsqueeze(1)).squeeze(1)

            ratio  = (sel_lp - old_lp[idx_t]).exp()
            adv_i  = advantages[idx_t].detach()
            surr1  = ratio * adv_i
            surr2  = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_i
            pl_sum = pl_sum - torch.min(surr1, surr2).sum()

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

        total_pl  += float(policy_loss.item())
        total_vl  += float(value_loss.item())
        total_ent += float(entropy_loss.item())

    return total_pl / n_epochs, total_vl / n_epochs, total_ent / n_epochs
