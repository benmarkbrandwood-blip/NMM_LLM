# Sanmill target-refresh common-anchor diagnostic v1

Status: `designed_unlaunched_needs_publication`. The machine-readable source
of truth is
[`sanmill-target-refresh-common-anchor-diagnostic-v1.json`](sanmill-target-refresh-common-anchor-diagnostic-v1.json).
This document and its tooling do not authorize a run, held-out evaluation,
promotion, publication, or long training.

## Why this diagnostic exists

The completed two-seed target-refresh/LR factorial found a large and
repeatable score difference after game 50. That score was measured against
each arm's own training target, however, so the refresh treatment also changed
the opponent used as the denominator. The arms additionally ended with
different optimizer-update counts. Those are decisive confounders: the prior
result establishes a boundary mechanism signal, but it cannot say that target
refresh helped or harmed transferable learning.

This successor changes only the frozen-target refresh cadence. Learning rate
is fixed in every arm. Each same-seed pair is fresh and must be byte-identical
through game 50. At game 50 the trainer freezes a separate development-only
model anchor before the refresh arm refreshes its training opponent. The
measurement anchor and its lookahead advisor never refresh.

## Controlled comparison

Seeds 64 and 65 retain their observed game-50 optimizer counts: 18 and 16.
Every arm then performs exactly 16 additional A2C optimizer steps, ending at
absolute update counts 34 and 32 respectively. Warm-start, imitation mix,
S1B refresher and PPO are disabled, so one increment of `update_count` means
one A2C optimizer step. The analyzer also reports `batch_steps`; equal update
counts do not imply identical trajectory sample counts.

No-update measurements occur after update deltas 4, 8, 12 and 16. At every
checkpoint the current learner plays eight colour-balanced games against the
common model anchor and eight against a fixed 1,000-node Sanmill opponent.
Both use the strict Sanmill referee, fixed sampling temperature 0.20 and fresh
start positions. Measurement games do not enter the optimizer, training logs,
curriculum, target-refresh clock, HumanDB, or SpecialistDB.

The growing training SpecialistDB is deliberately disabled on the measurement
route. Training continues to use `theoretical-only`. This prevents arm-specific
database coverage accumulated after game 50 from changing the supposedly
common measurement features.

## Interpretation contract

Observed facts will include the raw training and update curves, exact update
counts, candidate and anchor checkpoint identities, per-class W/D/L,
termination reasons, policy observations, Malom preservation observations,
batch-step exposure and policy-health results. This online RL experiment has
no supervised train/validation curves, and the report must say so.

The primary contrast is `no-refresh minus refresh` against the fixed model
anchor. A mechanism signal requires the same non-zero direction in both seeds
for the mean of all four checkpoints, the same direction at the final
checkpoint, and an absolute median seed-mean score contrast of at least 0.10.
The fixed-node Sanmill stratum is separate corroboration or counterevidence;
it is not silently pooled into the primary gate.

Even a supported result selects no retained setting. It is development
evidence only. A held-out design, long run, retry, extension, promotion or
publication would require a later immutable decision and separate product
authorization.

## Resource and stop boundary

There are four arms in frozen order: seed 64 refresh/no-refresh, then seed 65
refresh/no-refresh. Each arm is one fresh process with a 150-training-game
safety ceiling, a 0.5-active-hour ceiling, 64 maximum no-update measurement
games, an isolated fresh `sector-corrected-v1` SpecialistDB and no exact
resume. The whole sequence is bounded by 600 training games, 256 measurement
games and two active hours. Any arm anomaly stops the sequence; there is no
automatic retry, continuation, held-out evaluation or long training.
