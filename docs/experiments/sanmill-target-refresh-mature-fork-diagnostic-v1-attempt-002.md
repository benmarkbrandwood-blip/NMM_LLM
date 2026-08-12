# Mature target-refresh fork diagnostic v1, attempt 002

Status: `designed_unlaunched_needs_publication`

Plan identity:
`442c170177b5a8b867b14db31e62b16219fc3ee65ae1fac804842e493c35089d`

The authoritative machine-readable contract is
[`sanmill-target-refresh-mature-fork-diagnostic-v1-attempt-002.json`](sanmill-target-refresh-mature-fork-diagnostic-v1-attempt-002.json).
This document does not authorize launch.

## Why this is attempt 002

The first source-only preparation produced readiness identity
`32df3a5beb1e5bb71c83ceca13647ad735d0c5c67c65b80a29b5201ff186534f`,
but final inspection proved that all six managed plans had
`policy_health=null`. This contradicted the frozen stop rule requiring every
arm to fail closed on a policy-health regression. No authorization file or
training segment existed. The complete first preparation and its six closed
SpecialistDB clones were moved intact to the ignored quarantine directory
`out/quarantine/target-refresh-mature-fork-diagnostic-v1-missing-policy-health-2026-08-12/`.

Attempt 002 uses new common-fork, database, control, authorization, ledger and
result paths. It cannot overwrite, authorize, resume or relabel attempt 001.
The corrected preparer passes `--policy-health-gate`, verifies the generated
gate byte-for-byte against the frozen corpus, audit implementation,
thresholds and device, and records the gate in readiness evidence.

## Scientific question

The completed game-50 direct cross-play found that retaining the stale target
was materially better than refreshing it early: across 288 no-update games,
no-refresh scored 178 wins, 12 draws and 98 losses, for a paired score effect
of `+0.2777778`. That observation does not show that a target should remain
stale indefinitely. This diagnostic asks whether refreshing once after the
learner has consumed 8,192 post-game-50 transitions is beneficial relative
to keeping the same stale target.

For each of seeds 67, 68 and 69, one untreated mature common fork is created
from the exact no-refresh transition-8,192 checkpoint. The pending pre-fork
queue is cleared and the transition-indexed temperature is normalized to
`0.8379808850090307`. Model, Adam state, random states, data state, counters,
curriculum, frozen target and every other non-allowlisted recovery field stay
equal. Two descriptor-only branches share an identical payload:

1. `refresh-mature` copies the mature learner into the frozen opponent once;
2. `stale-control` retains the original game-50 frozen opponent.

No further target refresh occurs in either arm.

## Frozen training exposure

Each of the six arms receives exactly 8,192 additional consumed transitions,
using exact 64-transition A2C batches, fixed learning rate `0.0001`, and a
transition-indexed temperature schedule from `0.8379808850090307` toward
`0.20` over the remaining 98,112-transition horizon. Other fixed settings
include:

- 60% frozen-policy and 40% Sanmill-search training opponents;
- 1,000 nodes for each Sanmill search call;
- maximum 120 logical plies;
- `malom-preserving-only` Mill reward;
- theoretical-only SpecialistDB reads;
- no policy auxiliary, imitation, PPO, recovery, Sentinel, ValueNet or
  GapNet;
- one fixed 64-state policy-health gate after the completed arm segment.

The order is seed 67 refresh/control, seed 68 refresh/control, then seed 69
refresh/control. Only one trainer process may run at a time. Any arm failure
stops the entire sequence and consumes the one-shot attempt.

## Frozen analysis

The result publisher must keep observation, hypothesis, supporting evidence,
counterevidence and next experiment separate. It records raw training-log
identities and fixed blocks of up to 50 games, then reports W/D/L and score by
opponent source, learner colour and termination reason. Training W/D/L is
context, not the selection statistic.

At 4,096 and 8,192 post-mature transitions, both policies are compared on the
same 64 placement/movement/flying states. Full action logits are evaluated at
temperatures 1.0 and 0.2, with top-1 agreement, Jensen-Shannon divergence,
total variation, Malom-preserving probability mass and rank movement.

At transition 8,192, the paired policies play 144 colour-swapped pairs: 288
CPU games over twelve audited phase-covered histories and four replicates.
The same-seed mature common-fork policy supplies shared lookahead features;
Sanmill is the strict portable referee only. Common colour-specific random
streams are reset for the two games in each pair. Rules draws and 120-ply
truncations remain separate.

The direct contrast is `refresh-mature minus stale-control`. Material support
requires an aggregate paired effect of at least `0.0833333`, the same effect
in at least two seeds, no opposite seed worse than `0.0416667`, and a
truncation rate no higher than 25%. Persistent policy divergence from 4,096
through 8,192 transitions is supporting mechanism evidence only. Neither
parameter divergence nor the training graph can substitute for the paired
direct outcome.

## Resource and claim boundary

The aggregate ceiling is 3,600 training games, 49,152 consumed transitions,
four active hours, 172,800,000 requested Sanmill search nodes and 288 later
no-update games. Each arm is capped at 600 games and 0.6 active hours.

Source, checkpoint, database, plan, runtime, rules, corpus or implementation
drift; non-finite state; pair contamination; label or SQLite sidecar drift;
Sanmill disagreement; policy-health failure; resource exhaustion; or an
incomplete ledger stops the sequence. There is no automatic retry, recovery,
extension, held-out evaluation, promotion, publication, retained run or long
training.

The final result is development mechanism evidence only. Even a material
`refresh-mature` result does not establish a repeated refresh cadence and
does not authorize the long run. A retained long-training plan and held-out
evaluation remain separate future decisions.
