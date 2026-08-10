# Sanmill target-refresh/LR factorial diagnostic v1

Frozen design status: `designed_unlaunched_needs_publication`. This document
and its machine-readable contract authorize no training.

Machine-readable contract:
[`sanmill-target-refresh-lr-factorial-diagnostic-v1.json`](sanmill-target-refresh-lr-factorial-diagnostic-v1.json)

Plan identity:
`94f6381a40ab86401cb0e957677dd3a21dde01ed9ffd4c69b3fa252b21787e58`

This is a short two-seed mechanism diagnostic. It is not held-out validation,
playing-strength evidence, a trap-learning experiment, a promotion decision,
or authorization for retained or long training. All eight arms have zero
authorized segments.

## Product question

Which part of the coupled game-50 transition best explains the repeatable
decline against the frozen-model opponent:

1. refreshing the frozen opponent from the current learner;
2. changing the learning rate under the historical Sanmill-win rule; or
3. an interaction between the two?

The completed SpecialistDB read calibration cannot answer this because both
changes happened at the same boundary in every arm.

## Observed facts

The tracked
[transition diagnosis](../evidence/specialist-db-read-calibration-transition-diagnosis-2026-08-10.md)
binds the raw result identity and exact six-arm observations.

- Each arm logged `target_age=50` and learning rate `0.0001` at game 50.
- Each arm then logged `target_age=1` and learning rate `0.00005` at game 51.
- The learner's frozen-opponent score declined after that boundary in every
  arm, across three seeds and both SpecialistDB read projections.
- The learner had no Sanmill wins in the first window, so the historical
  adaptive rule used its minimum 0.5 multiplier.
- The learner samples actions at the scheduled temperature. The frozen model
  chooses a deterministic maximum-logit legal action. Refreshing can therefore
  change opponent difficulty even when the copied weights came from the same
  learner.

## Hypotheses

- Refresh hypothesis: copying the newer learner into a deterministic opponent
  makes the game-51 environment materially harder.
- Learning-rate hypothesis: reducing the rate at the same boundary prevents
  rapid adjustment to the changed environment.
- Interaction hypothesis: either change may be tolerable alone but harmful in
  combination.

## Supporting evidence

- The timing recurs across all six completed arms.
- Target age and learning rate expose both interventions directly in every
  game record.
- The first 50 games can be made identical within a new seed because neither
  factor acts until after the game-50 log boundary.
- A 2x2 design estimates both main effects and a difference-in-differences
  interaction without changing opponent work, rewards, data, or initialization.

## Counterevidence and limits

- The learner did not beat the 1,000-node Sanmill opponent before refresh.
  Weak policy quality may explain poor outcomes independently of either factor.
- Training outcomes are endogenous and cannot certify playing strength.
- A stronger deterministic target may provide useful pressure even if the
  short-run score falls.
- Fifty post-boundary games per arm are enough only for a directional mechanism
  check. A null result cannot prove that both factors are harmless long term.
- This RL trainer has no supervised validation-loss curve. Policy/value losses,
  entropy, and the fixed development policy-health audit serve different roles
  and must not be described as train/validation generalization evidence.

## Frozen 2x2 design

Seeds 64 and 65 are fresh. Each seed runs the following order:

| Order in seed | Frozen-target rule | Learning-rate rule | Condition |
| ---: | --- | --- | --- |
| 1 | refresh before game 51 | historical adaptive | refresh-adaptive |
| 2 | refresh before game 51 | fixed at `0.0001` | refresh-fixed |
| 3 | no refresh in the 5,000-game horizon | historical adaptive | no-refresh-adaptive |
| 4 | no refresh in the 5,000-game horizon | fixed at `0.0001` | no-refresh-fixed |

The no-refresh arms set the explicit refresh cadence to 5,001 games. This is
larger than the complete 5,000-game schedule, not merely larger than the
100-game controller ceiling, so an accidental continuation still cannot
silently refresh the target.

All four same-seed game ledgers must be byte-identical through game 50. At
game 51:

- refresh arms must report `target_age=1`;
- no-refresh arms must report `target_age=51`;
- fixed arms must report learning rate `0.0001`; and
- adaptive arms must report the exact frozen historical formula applied to
  the first 50 games' full-resource Sanmill win rate.

Any mismatch invalidates and stops the whole factorial. Only games 51-100 are
used for the factor contrasts.

## Factors held fixed

- Fresh A2C initialization; no checkpoint import or resume between arms.
- Same seed initialization and deterministic opponent/colour schedule within
  each four-arm block.
- Learning-rate base `0.0001`, gamma `0.99`, entropy coefficient `0.01`.
- Temperature starts at 0.90 on the unchanged 5,000-game schedule.
- 60% frozen-model and 40% Sanmill opponents.
- Sanmill fixed-resource ladder
  `1,000 / 5,000 / 25,000 / 100,000 / 500,000` nodes. Every arm ends before
  the first 500-game transition, so only 1,000-node search is used.
- One game at a time, update every 64 steps, simulated lookahead depth 5,
  minimal rollouts, no branches, maximum 120 complete logical plies.
- `malom-preserving-only` Mill shaping and zero policy auxiliary.
- `theoretical-only` SpecialistDB reads in all eight arms. This is a diagnostic
  isolation choice, not a retained-run read-mode selection. Each arm still
  writes to its own byte-identical empty `sector-corrected-v1` database.
- Sentinel, ValueNet, GapNet, imitation warm-start, imitation mixing, S1B,
  opening forcing, PPO, and recovery remain disabled.

## Resource envelope

Each arm contains exactly one 100-game segment and at most 0.25 active hour.
The full sequence is capped at 800 completed games and two active hours. Arms
run one process at a time. Any arm failure stops the sequence. There is no
automatic retry, extension, resume, continuation, promotion, publication, or
long training.

The post-boundary schedule contains 28 frozen-model games for seed 64 and 32
for seed 65. Colour and opponent counts are frozen in the machine-readable
contract and validated from the deterministic schedule before preparation.

## Required result evidence

The preregistered analyzer is
`learned_ai/evaluation/target_refresh_lr_factorial_result.py`; the one-shot
publisher is `scripts/report_target_refresh_lr_factorial_diagnostic.py`. Their
exact SHA-256 identities are in the machine-readable contract.

For every arm, the report must preserve:

- raw game and optimizer-update curves;
- explicit notice that supervised validation curves are unavailable;
- pre/post W/D/L split by opponent source, learner colour, combined source and
  colour, and termination reason;
- learning rate and target age at the boundary;
- temperature, entropy, chosen probability, policy/heuristic top-1, reward,
  Malom preserving/downgrade, and game-length observations;
- the fixed 29-state policy-health result;
- exact plan, authorization, preflight, manifest, checkpoint, database, log,
  lifecycle, MIF, ruleset, Malom, HumanDB, and Sanmill identities.

The primary statistic is the learner's score against the frozen-model opponent
in games 51-100. For each seed the analyzer computes:

- no-refresh minus refresh at both learning-rate settings;
- fixed minus adaptive at both refresh settings;
- the two factor main effects; and
- the interaction difference-in-differences.

A term is supported only when both fresh seeds have the same non-zero direction
and the absolute median contrast is at least 0.10. This threshold is frozen
before any new game is observed. Training W/D/L is diagnostic; the rule does
not convert it into a strength claim.

## Decision boundary

A supported refresh, learning-rate, or interaction term permits only the design
of a separately frozen successor probe that changes the indicated mechanism.
It does not select a retained setting and does not authorize held-out games or
long training. Mixed directions or a sub-threshold result remain inconclusive.

Held-out evaluation and long training remain blocked until this diagnostic is
completed, analyzed under the frozen rule, and followed by a new explicit
decision. This document itself authorizes no plan preparation, segment, retry,
resume, extension, promotion, publication, held-out run, or long training.
