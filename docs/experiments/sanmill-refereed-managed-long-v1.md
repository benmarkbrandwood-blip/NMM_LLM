# Sanmill-Refereed Managed Long Run v1

## Status and authority

Status: `launch_authorized_pending_local_plan_and_preflight`

Parent experiment: `dev-v4-sanmill-refereed-fresh-v1`

Plan ID: `managed-sanmill-v4-fresh-v1`

On 8 August 2026 the product owner approved the successor-v2 objective and a
maximum envelope of 5,000 games, 12 active hours, 120 logical plies per game,
and 250 games per process segment. After the exact-resume and resource-schedule
gates passed, the owner delegated the remaining technical launch decisions to
Codex through entry into long training.

This document records that authorization. It permits creation of the exact
local managed plan and authorization below, final read-only preflight, and
background launch only if every gate passes. It does not authorize weakening a
gate, reusing a failed segment, increasing the envelope, publishing weights,
or promoting a model.

## Objective and claim boundary

Train the first retained random-fresh corrected-v4 Generalist lineage whose
complete rollout rules/history and non-self-play opponent are both supplied by
the pinned strict Sanmill runtime, using a deterministic fixed-resource
curriculum and exact process-boundary resume.

Training graphs and outcomes are diagnostics. Completion does not prove
playing strength, MIF full conformance, or model promotion. Candidate strength
requires a later frozen held-out evaluation.

## Frozen technical configuration

- A2C, random weights, seed 42, `batch_games=1`, and default
  `update_every=64`;
- Sanmill referee and opponent at commit
  `a6623f88959f7453594df274fbe1f128af7ff55e`;
- `StrictFailurePolicy=true`, `StrictRefereeProfile=mif-stable-moving-v1`,
  single thread, disabled shuffling, fixed seed, fixed nodes, and no fallback;
- node ceilings `1,000,5,000,25,000,100,000,500,000`;
- stage games `500,500,500,1,000,2,500`;
- no score-based advancement and no lower-level random blending;
- 60% frozen-target games and 40% Sanmill-search games, with Sanmill referee
  on both strata;
- target refresh every 50 games and log/checkpoint cadence every 50 games;
- global temperature 0.90 to 0.20 by 80% of the 5,000-game schedule;
- `sim_ply_depth=5`, `max_ply=120`, minimal primary rollouts, and no branches;
- no recovery, resurrection, PPO, Sentinel, ValueNet, GapNet, S1A warm-start,
  S1B refresher, imitation mixing, or trainer-side opening forcing; and
- HumanDB frequencies/outcomes available while its unversioned historical
  Malom columns remain masked.

Resource transitions are curriculum exposure, not evidence that the previous
level was beaten. The supervisor does not stop on win rate, draw rate, or a
graph trend.

## Isolated local assets

Control directory:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sanmill_refereed/
managed-sanmill-v4-fresh-v1
```

Its immutable `plan.json`, `authorization.json`, controller ledger, segment
outputs, and recovery/quarantine directories remain ignored local run assets.

The fresh SpecialistDB is:

```text
data/specialist_db.sanmill_refereed_fresh_v1.long.sqlite
```

It must be newly created, empty, unbound, `quick_check=ok`, and
`sector-corrected-v1` before plan generation. It must not copy the smoke DB,
the completed local-GameAI DB, or any maintainer DB.

## Managed plan generation

From the clean published commit containing this document:

```powershell
.\.venv\Scripts\python.exe scripts\manage_generalist_run.py prepare `
  --control-dir learned_ai\checkpoints\scaffolded\s_gen_v2_sanmill_refereed\managed-sanmill-v4-fresh-v1 `
  --plan-id managed-sanmill-v4-fresh-v1 `
  --max-wall-hours 12 `
  --objective "fresh Sanmill-refereed corrected-v4 baseline with a deterministic fixed-resource curriculum" `
  --experiment-id dev-v4-sanmill-refereed-fresh-v1 `
  --max-games 5000 --segment-games 250 --max-ply 120 `
  --engine-profile sanmill-fixed-resource --self-play-ratio 0.60 `
  --sanmill-node-ladder 1000,5000,25000,100000,500000 `
  --sanmill-stage-games 500,500,500,1000,2500 `
  --specialist-db data\specialist_db.sanmill_refereed_fresh_v1.long.sqlite
```

The authorization must bind the resulting exact plan SHA-256, name
`product-owner` as authorizer, and record the approved game/time/truncation/
segment envelope plus delegated technical launch authority. No timeout or
automatic substitute may alter that product decision.

## Final launch gate

Before starting the supervisor:

1. `dev`, `origin/dev`, and the plan's Git commit must be identical and clean;
2. plan and authorization hashes, local path-config hash, output absence, and
   fresh SpecialistDB identity must verify;
3. the first segment's exact `--preflight long-run` command must return
   `ready_for_long_run`, exit zero, and report no errors or unresolved
   decisions;
4. the resolved resume-config must equal the managed plan pin; and
5. no older training or competing supervisor may own the intended resources.

If all checks pass, `run-authorized` may start in one hidden background
process. It may automatically move between its already authorized 250-game
segments only through verified exact resume. Any infrastructure, integrity,
rules, checkpoint, database, CUDA, or evidence-chain failure stops fail
closed for Agent diagnosis. Resource exhaustion requires a new owner decision.
