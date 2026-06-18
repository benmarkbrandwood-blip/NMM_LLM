"""NMMGNNNet: Graph Convolutional Network backbone replacing the MLP in NMMNet.

Board graph: 24 nodes (positions), edges from the union of mill lines (shared
threat structure) and move adjacency (piece mobility). Using both edge types
gives the GCN message passing both the "which squares form mills" prior AND
the "where can pieces escape" structure — both are essential for NMM strategy.

State encoding (84 floats) is unchanged:
  state[:72].view(24, 3)  → node features (empty/W/B one-hot per position)
  state[72:84]            → global features (side-to-move, phase, piece counts)

Architecture:
  node_embed: Linear(3 → 64) + ReLU
  GCNLayer × 2: (64 → 64), each = σ(A_norm · X · W)
  mean pool over 24 nodes → [B, 64]
  global_mlp: Linear(12 → 32) + ReLU
  cat([node_pool, global]) → [B, 96]
  project: Linear(96 → 128) + ReLU
  backbone output: [B, 128]  ← same shape as NMMNet backbone output

NMMGNNNet is a drop-in replacement for NMMNet: same phase_heads, value_head,
and forward() interface. All existing training code that calls
  feats = model.backbone(states)
  logits = model.phase_heads[name](feats)
  value  = model.value_head(feats)
works unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from game.board import ADJACENCY, MILLS, POSITIONS
from learned_ai.models.action_encoder import ACTION_DIM
from learned_ai.models.state_encoder import NUM_PHASES, PHASE_NAMES, STATE_DIM

NEG_INF = -1e9
_FEAT_DIM = 128
_NODE_DIM = 64
_GLOBAL_DIM = 32
_GCN_IN = 3    # node one-hot size
_GLOBAL_IN = 12  # state[72:84]

# ── Build normalised adjacency once at module import ─────────────────────────

def _build_adj_norm() -> torch.Tensor:
    """24×24 symmetrically-normalised adjacency (mill + move edges, with self-loops)."""
    pos_idx = {p: i for i, p in enumerate(POSITIONS)}
    adj = torch.zeros(24, 24)

    # Mill edges: every pair in a mill line shares a structural threat
    for mill in MILLS:
        for a, b in [(mill[0], mill[1]), (mill[1], mill[2]), (mill[0], mill[2])]:
            i, j = pos_idx[a], pos_idx[b]
            adj[i, j] = adj[j, i] = 1.0

    # Move-adjacency edges: physical movement connections
    for pos, neighbors in ADJACENCY.items():
        i = pos_idx[pos]
        for nbr in neighbors:
            j = pos_idx[nbr]
            adj[i, j] = adj[j, i] = 1.0

    adj += torch.eye(24)  # self-loops

    degree = adj.sum(dim=1)
    d_inv_sqrt = degree.pow(-0.5)
    d_inv_sqrt[degree == 0] = 0.0
    # D^{-1/2} A D^{-1/2}
    adj_norm = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
    return adj_norm


class GCNLayer(nn.Module):
    """Single graph convolution: σ(A_norm · X · W)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim], adj: [N, N] → [B, N, out_dim]."""
        agg = torch.matmul(adj, x)      # [B, N, in_dim]
        return F.relu(self.linear(agg))  # [B, N, out_dim]


class GNNBackbone(nn.Module):
    """GCN backbone: [B, 84] → [B, 128]."""

    def __init__(self) -> None:
        super().__init__()
        adj_norm = _build_adj_norm()
        self.register_buffer("adj_norm", adj_norm)  # moves with model.to(device)

        self.node_embed = nn.Linear(_GCN_IN, _NODE_DIM)
        self.gcn1 = GCNLayer(_NODE_DIM, _NODE_DIM)
        self.gcn2 = GCNLayer(_NODE_DIM, _NODE_DIM)
        self.global_mlp = nn.Linear(_GLOBAL_IN, _GLOBAL_DIM)
        self.project = nn.Linear(_NODE_DIM + _GLOBAL_DIM, _FEAT_DIM)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: [B, 84] or [1, 84] → [B, 128]."""
        if states.dim() == 1:
            states = states.unsqueeze(0)
        B = states.shape[0]

        node_feats = states[:, :72].view(B, 24, _GCN_IN)  # [B, 24, 3]
        global_feats = states[:, 72:]                       # [B, 12]

        x = F.relu(self.node_embed(node_feats))  # [B, 24, 64]
        x = self.gcn1(x, self.adj_norm)           # [B, 24, 64]
        x = self.gcn2(x, self.adj_norm)           # [B, 24, 64]

        node_pool = x.mean(dim=1)                           # [B, 64]
        global_proj = F.relu(self.global_mlp(global_feats)) # [B, 32]

        feat = torch.cat([node_pool, global_proj], dim=-1)  # [B, 96]
        return F.relu(self.project(feat))                    # [B, 128]


# ── Full network (backbone + heads) ──────────────────────────────────────────

def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class NMMGNNNet(nn.Module):
    """NMMNet with GNN backbone. Drop-in replacement for NMMNet.

    Identical phase_heads and value_head interface; only the backbone changes.
    Checkpoints saved with model_type='gnn' can be loaded back by specifying
    this class at construction time.
    """

    def __init__(
        self,
        head_hidden: Sequence[int] = (64,),
        dropout: float = 0.0,
        action_dim: int = ACTION_DIM,
        num_phases: int = NUM_PHASES,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_phases = num_phases
        feat_dim = _FEAT_DIM

        self.backbone = GNNBackbone()

        head_input = [feat_dim, *head_hidden]
        self.phase_heads: nn.ModuleDict = nn.ModuleDict(
            {
                PHASE_NAMES[p]: nn.Sequential(
                    _mlp(head_input, dropout=dropout),
                    nn.Linear(head_hidden[-1], action_dim),
                )
                for p in range(num_phases)
            }
        )
        self.value_head: nn.Sequential = nn.Sequential(
            _mlp(head_input, dropout=dropout),
            nn.Linear(head_hidden[-1], 1),
        )

    def _featurise(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.backbone(state)

    def forward(
        self,
        state: torch.Tensor,
        phase_id: int,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feats = self._featurise(state)
        if not (0 <= phase_id < self.num_phases):
            raise ValueError(f"phase_id {phase_id} out of range")
        logits = self.phase_heads[PHASE_NAMES[phase_id]](feats)
        value = self.value_head(feats).squeeze(-1)

        if legal_mask is not None:
            if legal_mask.dim() == 1:
                legal_mask = legal_mask.unsqueeze(0)
            logits = logits.masked_fill(~legal_mask, NEG_INF)

        if logits.shape[0] == 1:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        return {"logits": logits, "value": value}

    @torch.no_grad()
    def policy_probs(
        self,
        state: torch.Tensor,
        phase_id: int,
        legal_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        out = self.forward(state, phase_id, legal_mask)
        logits = out["logits"] / max(temperature, 1e-6)
        return F.softmax(logits, dim=-1)
