# Learned AI — Training Plan

The original `learned_ai/` attempt used REINFORCE self-play from a random initialisation and
never produced a model that could beat the heuristic engine.  Three subsequent REINFORCE
attempts (v1/v2/v3 of Stage 2) also failed to converge.  This document explains why they
failed, what the correct approach is, and the concrete staged plan now in use.

---

## Why REINFORCE Failed (Root Cause Analysis)

| Root cause | Effect |
|------------|--------|
| Terminal-only reward | 60-ply game → one gradient signal.  Model mostly lost → log_probs pushed down on *every* action, destroying the Stage 1 imitation prior |
| No per-step value bootstrap | REINFORCE disadvantage = reward − V(s).  With no TD, V is useless until the model converges — which it can't without a good V |
| win_reward = 2.0 vs Stage 0 trained on [−1,+1] | Value head oscillated; gradients had the wrong scale |
| temperature = 0.5 for a pre-trained checkpoint | Too noisy; erased the imitation prior within 50 games |
| lr = 1e-4 for fine-tuning | Too aggressive; destroyed Stage 1 prior within the first update |
| Policy collapse | ~95% loss rate → REINFORCE maximised −log_prob on all actions → worse than random initialisation |

v3 was killed at game 500 with 2.0% win rate (down from 2.5% at game 200).  The algorithm
was making the model *worse* than the Stage 1 imitation baseline.

---

## What We Now Know Works

| Component | Evidence |
|-----------|----------|
| **Value net at 80% blend** | +17.5 pp vs plain heuristic (8W/1L/31D in 40-game bench) |
| **Malom DB** | Perfect DTM labels for any position; both reward signals work (move quality + trap) |
| **HumanDB** | 22,895 real games, 642,703 positions; quality labels from win/loss outcome |
| **Sentinel** | Reliable move-quality classifier; advisory mode well-calibrated |
| **A2C bootstrapping** | Per-step advantage = r(t) + γV(s') − V(s): dense gradient, no collapse |
| **GNN backbone** | Mill + move adjacency edges give the network positional structure as a prior |

---

## Architecture: NMMGNNNet (GNN backbone)

### Why GNN over flat MLP

The flat MLP (NMMNet) must infer all positional structure — which squares share mill lines,
which squares are adjacent — entirely from data.  A GNN bakes the board graph in as a
structural prior so message passing propagates "this square is adjacent to a mill threat"
from the first episode, not after thousands of games.

For NMM's 24-node graph this is a significant sample-efficiency win: mill patterns are
exactly the edges of the graph.

### Graph construction

- **Nodes:** 24 board positions (same order as `POSITIONS`)
- **Node features:** 3-way one-hot per position (empty / W / B) = state[:72].view(24, 3)
- **Edges:** Union of (a) mill edges — every pair of positions sharing a mill line, and
  (b) move-adjacency edges — physical movement connections from `ADJACENCY`.
  Both edge types matter: mill edges encode strategic threat, adjacency encodes mobility.
- **Self-loops:** added before symmetric normalisation (D⁻¹/²AD⁻¹/²)
- **Global features:** state[72:84] (side-to-move, phase one-hot, piece counts/mills) → Linear(12→32)

### Architecture layers

| Layer | Shape | Note |
|-------|-------|------|
| node_embed | Linear(3→64) + ReLU | per-node input projection |
| GCNLayer × 2 | 64→64 each | σ(A_norm · X · W) |
| mean pool | [B,24,64] → [B,64] | aggregate over nodes |
| global_mlp | Linear(12→32) + ReLU | project global features |
| cat + project | [B,96] → [B,128] + ReLU | backbone output |
| phase_heads | 5× Linear(128→64→624) | same as NMMNet |
| value_head | Linear(128→64→1) | same as NMMNet |

`NMMGNNNet` is a drop-in replacement for `NMMNet`: same `.backbone()`, `.phase_heads`,
`.value_head`, and `.forward()` interface.  All existing training code works unchanged.

File: `learned_ai/models/gnn_backbone.py`

---

## Algorithm: A2C (default) / PPO (optional)

### Why A2C over REINFORCE

REINFORCE accumulates gradients until the terminal reward.  In a 40-ply NMM game where the
model loses 95% of the time, every log_prob in every losing game gets pushed down uniformly —
which is exactly policy collapse.

A2C bootstraps the return at every step:

    advantage(t) = r(t) + γ·V(s_{t+1}) − V(s(t))

This gives a dense gradient signal even in losing games.  If the model chose a good move in
a losing game, the local advantage can still be positive (the position improved, even if the
game was ultimately lost).  The value head trains on TD targets, not terminal outcomes.

### PPO (--ppo flag)

PPO adds a clipped surrogate to A2C:

    L_clip = min(ratio · adv,  clip(ratio, 1−ε, 1+ε) · adv)
    where ratio = π_new(a|s) / π_old(a|s)

This prevents large destructive policy updates.  Use PPO if A2C still shows instability on
longer curricula (Stage 3+).

Files: `learned_ai/training/a2c.py`, `learned_ai/training/ppo.py`

### Three bug fixes (vs REINFORCE v3)

| Bug | Old value | Fixed value | Why |
|-----|-----------|-------------|-----|
| win_reward | 2.0 | **1.0** | Stage 0 trains value head on [−1,+1]; reward must match |
| temperature start | 0.5 | **0.2** (annealed to 0.6) | Preserve Stage 1 imitation prior; anneal exploration gradually |
| learning rate | 1e-4 | **5e-6** | Safe fine-tuning rate for pre-trained checkpoint |

### Malom reward shaping in A2C

Both Malom signals carry over from REINFORCE, but `malom_weight` is reduced from 0.3 to
**0.1** because A2C processes per-step rewards directly (not just at terminal).  Over 40+
learner moves, 0.3 × (up to 0.6 per step) accumulates to ±24 — completely swamping the ±1
terminal signal.  With 0.1, accumulated shaping peaks at ±4 and stays commensurate with the
terminal.

1. **Move quality:** `query_move_quality(board, move)` → δ ∈ [−2,+2] × 0.1 added to r(t)
2. **Trap reward:** `query(board)` after learner move; if opponent position is "L" → +0.1

Both signals slot directly into r(t) in the A2C update.

---

## Training Stages

### Stage 0 — Supervised Value Pre-training  *(re-run with GNN)*

**Goal:** Bootstrap the value head so it is not random noise from episode 1.

**Method:**
1. Generate ~50k positions by running heuristic engine (difficulty 6, vn_blend=80%) self-play.
2. Label each position with `value_net.predict(board, board.turn)` — side-to-move relative.
3. Supervised regression on the value head (frozen backbone first, then full network).

**Exit criterion:** val MSE plateaus (no improvement for 3 epochs).

**Command:**
```
.venv/bin/python scripts/train_stage0.py --gnn --out-dir learned_ai/checkpoints/stage0_gnn
```

#### Stage 0 — MLP baseline results (archived)

| Item | Value |
|------|-------|
| Positions | 28,537 across all phases |
| Phase 1 | frozen backbone, lr=3e-3 — val MSE 0.384 → **0.230** |
| Phase 2 | full network, lr=5e-4 — val MSE 0.295 → **0.012** |
| Checkpoint | `learned_ai/checkpoints/stage0/best.pt` (MLP only — not usable for GNN) |

Stage 0 will be re-run from scratch with `--gnn` to produce a GNN checkpoint.

---

### Stage 1 — Imitation Learning from Human Games  *(re-run with GNN)*

**Goal:** Give the policy head a strong prior over move selection before RL begins.

**Data:** HumanDB (30,256 games, 820,495 positions).  Label: win-rate of each move as CE weight.

**Method:**
- CE loss on primary action (placement/movement slice) weighted by win-rate.
- 8× D4 symmetry augmentation so the policy sees all board orientations.
- Two-phase: frozen backbone (high LR) → full network (low LR, early stop).

**Command:**
```
.venv/bin/python scripts/train_stage1.py --gnn \
    --resume learned_ai/checkpoints/stage0_gnn/best.pt \
    --out-dir learned_ai/checkpoints/stage1_gnn
```

#### Stage 1 — MLP baseline results (archived)

| Item | Value |
|------|-------|
| Samples | 169,048 (8× augmented) |
| Phase 1 | frozen backbone — val_acc 4% → **23.5%** |
| Phase 2 | full network — val_acc 30% → **45.3%** (early stop ep 37) |
| Checkpoint | `learned_ai/checkpoints/stage1/best.pt` (MLP only) |

Stage 1 will be re-run from scratch with `--gnn` to produce the RL starting point.

---

### Pre-Stage 2 Baseline

Stage 1 MLP checkpoint (greedy, temp=0) vs heuristic difficulty 2, vn_blend=0:
W=0, D=25, L=15 over 40 games — 0% win rate, 62.5% draw rate.
Confirms imitation learning gives defensive play but no winning ability without RL.

---

### Stage 2 — A2C Self-play vs Weak Heuristic

**Goal:** First RL stage; model learns to win reliably vs a weak opponent.

**Algorithm:** A2C (or `--ppo` for PPO variant)

**Opponent:** Heuristic engine, difficulty 2 → 3, vn_blend=0%, 0.05s/move.

**Malom shaping:** Both signals active for the first 50% of games, weight=0.1.

**Command:**
```
.venv/bin/python scripts/train_stage2.py \
    --resume learned_ai/checkpoints/stage1_gnn/best.pt \
    --out-dir learned_ai/checkpoints/stage2_gnn \
    --max-games 10000
```

**Exit criterion:** Rolling 200-game win rate ≥ 60% at difficulty 3.

#### Stage 2 — REINFORCE Attempt History (killed)

Three REINFORCE attempts all failed due to policy collapse (see root cause analysis above).

| Version | Algorithm | Killed at | Final win rate | Root cause |
|---------|-----------|-----------|----------------|------------|
| v1 | REINFORCE | game 4372 | 7.5% | sentinel over-filtered, value collapse, T=1.0 |
| v2 | REINFORCE + malom quality | game ~200 | 2.3% | too early to judge; restarted with trap reward |
| v3 | REINFORCE + both malom signals | game 500 | 2.0% (declining) | all three algorithm bugs |

v3 post-mortem: the three concrete bugs (win_reward=2.0, T=0.5, lr=1e-4) compounded the
fundamental REINFORCE variance problem.  Switching to A2C with bug fixes is the correct path.

#### Stage 2 — A2C Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LR | 5e-6 | Bug fix 3: safe fine-tune rate |
| Temperature | 0.2 → 0.6 (linear anneal) | Bug fix 2: preserve prior, gradually explore |
| win_reward | 1.0 | Bug fix 1: match Stage 0 [−1,+1] scale |
| Malom weight | 0.1 | Reduced for A2C per-step amplification |
| γ (discount) | 0.99 | Standard; A2C bootstraps so full discount is ok |
| UPDATE_EVERY | 16 | Batch 16 games before each gradient step |
| entropy_coef | 0.01 | Prevent premature determinism |

---

### Stage 3 — Curriculum vs Heuristic + Value Net

**Goal:** Climb from weak to strong heuristic opponent.

**Opponent:** Heuristic engine, difficulty 3 → 8, vn_blend=80% at difficulties 6+.

**Algorithm:** A2C (or PPO — preferred at this stage since opponent is stronger).

**Difficulty ramp rule:** ≥ 55% win rate over 200-game rolling window before bumping.

**Malom DB reward shaping:** Both Malom signals remain active throughout Stage 3.  The model
does **not** receive raw W/L/D labels as input — reward shaping only.

**Opponent move replay (new):** In every lost game, record opponent moves where
`query_move_quality >= 0` (Malom confirms good for opponent).  Train a small supervised CE
loss (`imitation_weight=0.1`) on these transitions alongside A2C.  Teaches the model what
winning positions look like from the exact board states it failed at.

**Training quality:** Stage 3+ uses longer time budgets (0.3s–1.0s/move), vn_blend=80% at
difficulty 6+, full DB access for the opponent.  Extended wall-clock time is acceptable.

**Exit criterion:** 55% win rate at difficulty 8 + vn_blend=80%.

**Coverage note:** Malom DB coverage drops in midgame (many pieces, complex positions) —
`query()` returns `None` more often.  Monitor per-phase signal frequency to confirm the model
still receives useful reward in midgame, not only in placement/endgame phases.

---

### Stage 4 — Self-play Pool

**Goal:** Open-ended strength improvement through self-play against a pool of past checkpoints.

**Method:** Pool-based self-play (keep N past checkpoints; randomly sample opponents).
Remove sentinel blunder filter — the model should be strong enough that blunders are rare.

**Malom DB shaping:** Both signals remain active.  The model still does not see raw W/L/D
labels — reward shaping only.

**Exit criterion:** ≥ 70% win rate vs heuristic + vn_blend=80%, or episode budget reached.

---

### Stage 5 — Malom Full-game Supervised Distillation

**Goal:** Skill refinement by directly learning from Malom's perfect play across the entire game.

**Why at the end:** Supervised distillation onto a strong generalising model is far more
effective than early injection when the rest of the network is random.  Stages 2–4 build the
strategic intuition; Stage 5 sharpens it to Malom precision.

**Method:**
- Sample positions from all phases / piece counts from the Malom DB.
- For each position, query Malom W/L/D for every legal move.
- Supervised training:
  - **Value head target:** W=+1.0, D=0.0, L=−1.0 (exact WDL for side to move)
  - **Policy head target:** CE toward distribution of Malom-winning moves (uniform over "W"
    moves; fallback to "D"; fallback to "L")
- Light LR (1e-5), few epochs to avoid catastrophic forgetting.

**Full-game scope:** Unlike "endgame only" (≤7 pieces), this stage covers all phases.  The
model trained on Stages 2–4 sees Malom reward shaping throughout the game; Stage 5 closes
the loop by directly showing it the W/L/D labels for every move in every position.

**Policy target refinement (future option):** DTM-weighted targets — faster wins get higher
probability mass.  Implement only if "uniform over W moves" plateau is confirmed.

**Exit criterion:** Policy selects a Malom-winning move (where one exists) in ≥ 85% of
sampled full-game positions; value head WDL accuracy ≥ 80% across all phases.

---

## Integration Points

| Mode | What happens |
|------|-------------|
| **Evaluation** | `bench_sentinel.py`-style A/B: new learned agent vs heuristic+vn80% |
| **Advisory** | Display learned agent's top move alongside heuristic in AI Discussion panel |
| **Hybrid** | Blend learned value head at 20% alongside existing value net (80%) once it reaches parity |

The hybrid mode is the lowest-risk path to a playable improvement.

---

## What to Reuse vs Rewrite

| Component | Status |
|-----------|--------|
| `learned_ai/models/gnn_backbone.py` — NMMGNNNet | **New** — GNN backbone, drop-in for NMMNet |
| `learned_ai/models/backbone.py` — NMMNet | Keep (MLP fallback, `--no-gnn` flag) |
| `learned_ai/models/action_encoder.py`, `state_encoder.py` | **Keep as-is** |
| `learned_ai/agents/` — LearnedAgent, HeuristicAgent | **Keep as-is** |
| `learned_ai/training/a2c.py` | **New** — A2C per-step TD update |
| `learned_ai/training/ppo.py` | **New** — PPO clipped surrogate update |
| `learned_ai/training/replay_buffer.py` | Keep |
| `scripts/train_stage0.py` | Updated — `--gnn` flag, fixed checkpoint save |
| `scripts/train_stage1.py` | Updated — `--gnn` flag |
| `scripts/train_stage2.py` | Rewritten — A2C/PPO + three bug fixes |
| `scripts/train_stage3.py` | Keep (update algorithm in next session) |

---

## Success Metrics

| Milestone | Target |
|-----------|--------|
| After Stage 0 (GNN) | Value-head MSE < 0.08 on held-out positions |
| After Stage 1 (GNN) | Top-1 move accuracy > 30% on held-out human games |
| After Stage 2 | ≥ 60% win rate vs heuristic difficulty 3 (A2C, no policy collapse) |
| After Stage 3 | ≥ 55% win rate vs heuristic difficulty 8 + vn80% |
| After Stage 4 | ≥ 70% win rate vs heuristic + vn80% baseline |
| After Stage 5 | Malom-winning move selected in ≥ 85% of positions; WDL accuracy ≥ 80% |

---

## Contingency: If GNN + A2C Also Plateaus

The GNN + A2C restart is the current primary plan.  If it also fails to produce a learning
signal, the following options should be checked in order before concluding the architecture is
the bottleneck.

1. **Confirm A2C is actually learning:** After 500 games, entropy should decrease (model
   becoming less random), value loss should decrease (value head improving).  If both are
   diverging, something is wrong at the implementation level before worrying about the model.

2. **Structural feature enrichment (no restart):** Add pre-computed structural features to the
   state encoder — mill threat count, open triples, mobility — before Stage 3 or a second
   Stage 2.  The sentinel's `feature_builder.py` already computes these.  Low-risk additive
   change; expected 8–10% sample efficiency gain.

3. **PPO for instability:** If A2C shows oscillating policy loss, switch to PPO with the
   `--ppo` flag.  The clipped surrogate prevents large destructive updates.

4. **Increase Stage 1 coverage:** The MLP achieved 45.3% val accuracy.  If the GNN achieves
   significantly lower imitation accuracy, the RL starting point is worse — extend Stage 1
   training or generate more HumanDB data.
