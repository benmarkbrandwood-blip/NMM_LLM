# SpecialistDB training-read calibration v1

Frozen design status: `designed_unlaunched_needs_publication`. Operational
publication and preparation state is recorded by the external readiness
report; this immutable design label is not a live launch authorization.

Machine-readable contract:
[`specialist-db-training-read-calibration-v1.json`](specialist-db-training-read-calibration-v1.json)

Plan identity:
`36a1feb6bc9e403890f7c3b6b6f3444a97a9cd721272b760a2b25d0f8091459b`

This is a bounded three-seed mechanism calibration. It is not held-out
validation, playing-strength evidence, a trap-learning experiment, a promotion
decision, or authorization for a retained or long run. All six arms have zero
authorized segments.

## Product question

Does the trainer's historical empirical-first SpecialistDB projection cause a
repeatable change in learned policy weights, compared with using the same
writable databases but reading only trusted `sector-corrected-v1` Malom
labels?

This is the smallest training experiment that follows from the completed
5,000-game result and the no-update policy-mechanism audit. It deliberately
does not change Sanmill work, temperature, reward shaping, opponent mix,
opening forcing, auxiliary losses, or any network component.

## Observed facts

- The retained v3 run completed 5,000 games from one fresh seed. Its final 200
  games were 0 wins, 199 draws, and 1 loss. That late draw mass is an observed
  outcome, not proof of either good defence or failed trap creation.
- The retained run's SpecialistDB accumulated 164,431 positions. Of those,
  10,871 had at least three empirical samples, 35,197 had trusted Malom labels,
  and 2,666 had both.
- A candidate-blind 12-ply placement audit selected 100 states with usable
  empirical coverage before loading the final checkpoint.
- On those 100 states, suppressing empirical reads changed the final policy
  argmax in exactly three early placement states. Mean scheduled-temperature
  total variation was about 0.0174 and the maximum was about 0.1138.
- The full and theoretical-only projections chose a Malom-preserving action in
  all 100 audited states. No preserving-versus-downgrading boundary was
  crossed.
- The first phase-covered audit had zero SpecialistDB coverage. It remains
  useful negative coverage evidence but cannot support a no-effect claim.
- `origin/main` remains at
  `bc46b51e69724e12a8e5f17e3ff696b9f88456d9`. Its v2c recovery, repetition,
  network, and curriculum experiments are not selected for this one-factor
  calibration.

## Hypotheses

The primary hypothesis is that empirical-first SpecialistDB reads have a
small but repeatable causal effect on learned policy weights beyond trusted
theoretical labels alone. The effect may be helpful, harmful, or neutral; this
calibration measures whether it survives learning across seeds before asking
which direction is better.

Two competing explanations remain deliberately open:

1. the late draw-heavy behaviour may be reinforced by the cumulative empirical
   projection, which pools outcomes across model ages and opponent conditions;
   or
2. it may instead arise from temperature, frozen-target feedback, the changing
   node curriculum, limited Sanmill search at early levels, or ordinary policy
   learning.

The suggestion that stronger Mill play requires learning active traps is
plausible and worth separate work. It is not established by the retained run.
In particular, weak-node Sanmill games can produce uninformative or easily
repeated play, so this calibration must not be reported as a trap experiment.

## Supporting evidence

- The read-only audit changed only the SpecialistDB projection and used one
  fixed checkpoint, one byte-identical database snapshot, and a corpus frozen
  before candidate loading.
- The three changed argmaxes were localized to empirical reads: full and
  empirical-only agreed, while theoretical-only and all-disabled agreed.
- The new production read view exposes `full` and `theoretical-only` modes,
  leaves database writes unchanged, binds the choice in the run manifest, and
  logs per-rollout query and suppression counters.
- Per-worker counters prevent concurrent rollout telemetry from being mixed,
  although this calibration itself keeps `batch_games=1` because that is the
  proved exact-resume mode.

## Counterevidence and limits

- Three changed argmaxes out of 100 states are material under the frozen audit
  threshold but still a small local effect.
- All audited choices preserved Malom value. The audit did not demonstrate a
  direct rules-safety failure or explain win conversion.
- The coverage-positive corpus is placement-heavy and inspected. It is not an
  independent held-out evaluation.
- Separate writable databases will diverge after policy choices diverge. That
  divergence is downstream of the read intervention, but it means endpoint DB
  contents cannot be compared as though they were fixed covariates.
- Two hundred and fifty games per arm are sufficient for a mechanism
  calibration only. A null result is inconclusive about long-run strength.
- There is no supervised validation-loss curve. Policy loss, value loss and
  entropy are training curves; fixed-state diagnostics serve a different
  purpose and must remain labelled as development evidence.

## Frozen six-arm design

The order is fixed and arms run one process at a time:

| Order | Seed | Condition | SpecialistDB read projection |
| ---: | ---: | --- | --- |
| 1 | 61 | control | empirical-first `full` |
| 2 | 61 | treatment | `theoretical-only` |
| 3 | 62 | control | empirical-first `full` |
| 4 | 62 | treatment | `theoretical-only` |
| 5 | 63 | control | empirical-first `full` |
| 6 | 63 | treatment | `theoretical-only` |

Within each seed, both arms use the same fresh initialization and deterministic
game schedule. The only training-semantic difference is
`specialist_read_mode`. IDs, output paths, launch order, and separate database
paths are isolation metadata rather than learning factors.

Each arm starts with fresh random weights and its own byte-identical empty
`sector-corrected-v1` SpecialistDB. Both modes continue to record games and
trusted Malom labels. The treatment suppresses only empirical projections when
encoding policy features; it does not disable the database or its writes.

## Frozen training and resource contract

- A2C, learning rate `1e-4`, discount `0.99`, entropy coefficient `0.01`.
- One game at a time, update every 64 steps, no branches, simulated lookahead
  depth 5, maximum 120 complete logical plies.
- Temperature starts at 0.90 and follows the unchanged 5,000-game schedule
  toward 0.20.
- 60% frozen-target and 40% Sanmill opponents; target refresh every 50 games.
- Fixed Sanmill ladder
  `1,000 / 5,000 / 25,000 / 100,000 / 500,000` nodes. A 250-game arm remains
  entirely at the first, already observed 1,000-node level. This holds opponent
  work constant; it does not validate that level as a strength baseline.
- `malom-preserving-only` Mill shaping and zero policy auxiliary.
- Sentinel, ValueNet, GapNet, imitation warm-start, imitation mixing, S1B,
  opening forcing, PPO, recovery, and branching remain disabled.
- Exactly one 250-game segment per arm, no resume between conditions, at most
  1,500 completed games and three active hours total.
- Any arm failure stops the whole sequence. No automatic retry, extension,
  continuation, promotion, publication, or long training is allowed.

The deterministic schedule counts are:

| Seed | Frozen/Black | Frozen/White | Sanmill/Black | Sanmill/White |
| ---: | ---: | ---: | ---: | ---: |
| 61 | 73 | 85 | 42 | 50 |
| 62 | 90 | 60 | 51 | 49 |
| 63 | 83 | 76 | 53 | 38 |

## Required evidence

Every game record must expose the selected read mode plus query, row-present,
theoretical-available, empirical-available, projection-returned and
empirical-suppressed counts. The mechanism is not engaged unless:

- every full arm observes empirical availability and records zero suppression;
  and
- every theoretical-only arm observes empirical availability and suppresses
  exactly that many empirical projections.

Results must include raw training curves and complete-window summaries for all
three seeds. W/D/L must be split by opponent source and learner colour;
termination reasons must remain separate. Training W/D/L is diagnostic and is
not a selection metric.

Endpoint checkpoint comparisons use identical, fixed development corpora and
disable all SpecialistDB projections so the measured difference belongs to
learned weights rather than a live database read. Scratch reconstruction is
paired by seed. Report total variation, argmax changes, entropy, Malom
preserving mass and policy-health classes.

The frozen result implementation is
`learned_ai/evaluation/specialist_db_training_read_calibration_result.py`;
its immutable publisher is
`scripts/report_specialist_db_training_read_calibration.py`. The machine-
readable contract binds both file hashes and the result schema. The endpoint
route uses one same-seed reconstructed scratch network as the frozen-target
feature source for both arms, passes no SpecialistDB to the encoder, and then
compares the two learned policies on both fixed development corpora. This
prevents post-training database contents or arm-specific target features from
being mistaken for a policy-weight effect.

## Frozen decision boundary

A seed pair has a detectable learned-policy effect when its all-disabled
endpoint comparison has either at least three argmax changes or mean policy
total variation of at least 0.01. A reproducible effect requires at least two
of three seed pairs, complete intervention telemetry, finite updates and all
identity, referee, database, checkpoint and policy-health gates passing.

Passing this gate permits only the design of a separately frozen held-out
effectiveness comparison. It does not choose full or theoretical-only for a
retained run. A safety failure stops the sequence. A null or one-seed result is
inconclusive and must not be converted into a no-effect claim.

## Preparation and launch boundary

The contract and its source-only validator may be committed and published.
After publication, preparation may copy six isolated empty databases and
create six immutable managed plans and read-only preflight reports. It must not
create authorization files or start a trainer.

This document authorizes no segment, retry, resume, extension, promotion,
publication, retained run, or long training.
