# Sanmill target-refresh equal-transition diagnostic v1

Status: `designed_needs_implementation_no_launch_authority`

This successor is required only because the read-only common-anchor action
analysis classified the refresh and no-refresh policies as `near_identical`.
It must not be prepared or launched until its exact-transition implementation,
focused tests, immutable machine-readable contract, fresh database lineage,
managed plans, and preflights have been reviewed separately.

## Question

Does a single frozen-target refresh at a common game-50 boundary eventually
produce a material candidate-policy distribution change when both conditions
consume exactly the same number of learner transitions?

This is a mechanism question. It is not a strength comparison, curriculum
selection, held-out evaluation, promotion decision, or long-training launch.

## Why the previous horizon is insufficient

Attempt 003 matched 16 post-anchor optimizer updates, but each update consumed
all pending steps after a complete game. Its post-anchor consumed-step totals
therefore differed:

- seed 64: 1,161 refresh versus 1,241 no-refresh;
- seed 65: 1,348 refresh versus 1,214 no-refresh.

The fixed-state follow-up found substantial parameter separation but only
`1.97e-9` and `6.45e-9` all-phase mean Jensen-Shannon distance at the final
checkpoint. A longer run is justified, but only after the batch-size
confounder is removed.

## Frozen design

### Shared prefix and treatment fork

Use three fresh seeds: 64, 65, and 66. Each seed first runs one shared prefix
through game 50. The trainer must capture a fork checkpoint immediately before
the target-refresh decision. The checkpoint must contain the complete model,
optimizer, Python/NumPy/PyTorch/component RNG state, rolling histories,
pending transition queue, target state and age, and mutable data identity.

Two arms are then created from byte-identical copies of that fork state:

- `refresh-once`: replace the frozen target with the current candidate at the
  fork boundary and reset target age;
- `no-refresh`: retain the pre-boundary frozen target and its age.

No later target refresh is allowed in either arm. This isolates the one
game-50 intervention instead of confounding it with a different number of
later game-count-triggered refreshes.

The fork operation must be explicit and allowlisted. It may change only the
frozen target model and target-age fields in `refresh-once`, plus identities
that name the successor arm. It must not rebuild weights, optimizer state,
RNG state, histories, pending steps, data cursors, or reward state.

### Exact transition accounting

`update_every=64` remains fixed. The trainer must consume exactly the first 64
pending learner transitions per optimizer update and retain any overflow for
the next update. It must not clear the whole pending queue after an update.

Add a persisted `optimizer_consumed_transition_count` and a fail-closed
post-fork bound. Candidate checkpoints are captured after exactly:

- 1,024 transitions / 16 updates;
- 2,048 transitions / 32 updates;
- 4,096 transitions / 64 updates;
- 8,192 transitions / 128 updates.

The model at each boundary must have consumed exactly that many post-fork
transitions in both conditions. Full games may generate additional pending
transitions, but those transitions must remain unconsumed and cannot affect
the checkpointed model. Generated, consumed, and pending counts must be logged
separately.

There is no final undersized optimizer flush. The retained pending queue is
evidence for exact resume only and is excluded from the model comparison.

### Unchanged training factors

Apart from the one target-refresh intervention, retain the attempt-003
contract:

- A2C, fixed learning rate `1e-4`, entropy coefficient `0.01`;
- temperature `0.90` to `0.20` under the frozen 5,000-game schedule;
- `batch_games=1`, complete games, `max_ply=120`, no branches;
- 60% frozen-model and 40% strict Sanmill opponent schedule;
- fixed Sanmill resource stage and the existing node ladder;
- `sim_ply_depth=5`;
- trusted Malom with `malom-preserving-only` mill reward;
- fresh isolated `sector-corrected-v1` SpecialistDB clones and
  `theoretical-only` training reads;
- HumanDB frequencies/outcomes enabled and unversioned Malom fields masked;
- no PPO, imitation, warm-start, S1B, recovery, opening forcing, Sentinel,
  ValueNet, GapNet, or Malom policy auxiliary.

The shared prefix and both forks for a seed must use identical opponent
schedules through game 50. After the fork, game counts may differ and are
descriptive only; optimizer-consumed transition count is the comparison axis.

## Measurements

At each exact transition boundary, perform no-update inference on the existing
64 placement/movement/flying corpus. Use one frozen feature matrix per seed and
store all legal action logits. Report the same metrics as the completed
read-only analysis:

- top-1 agreement and margins;
- directional KL and Jensen-Shannon distance at temperatures 1.0 and 0.2;
- total variation;
- Malom-preserving probability mass and its change from the equal-action
  uniform reference;
- normalized rank displacement and discordant action-pair rate;
- placement, movement, flying, seed, and transition-boundary strata.

Do not run Sanmill or frozen-anchor outcome games during this diagnostic. The
previous measurement floor makes them uninformative at this stage. If the
policy distributions become materially different, a separate non-flooring
multi-start no-update outcome measurement must be designed and authorized.

## Predeclared decision rule

Reuse the already frozen distribution thresholds:

- `near_identical`: for every seed, maximum phase mean JS <= `5e-4`, maximum
  phase mean total variation <= `0.02`, and maximum phase mean absolute Malom
  preserving-mass delta <= `0.02`;
- `materially_diverged`: for at least two of three seeds, all-phase mean JS >=
  `0.005`, all-phase mean total variation >= `0.05`, or a phase mean absolute
  Malom preserving-mass delta >= `0.05`;
- otherwise: `inconclusive`.

Top-1 disagreement alone remains interpretive because the current policies
are almost uniform. A material result must first appear by 4,096 transitions
and remain directionally consistent at 8,192 transitions in at least two
seeds. If it appears only at the final boundary, the result is
`inconclusive_late_onset`, not automatic evidence for a longer run.

If all three seeds remain near-identical at 8,192 transitions, stop. Do not
extend the horizon automatically. The one-time target refresh has then failed
to show a broad policy-distribution effect under this training contract.

## Required implementation and regression gates

Before preparation, the following must pass:

1. Exact batch consumption: a 70-step queue performs one 64-step update and
   retains six byte-identical pending steps.
2. Multiple-update consumption: a 140-step queue performs two ordered 64-step
   updates and retains 12 steps.
3. No undersized final flush under a transition-bound diagnostic.
4. Checkpoint/resume preserves consumed and pending counts exactly.
5. A duplicate-control fork with no treatment produces byte-identical model,
   optimizer, RNG, logs, and checkpoint state after 512 consumed transitions.
6. The refresh fork differs from the no-refresh fork only in the allowlisted
   target state and age before post-fork training begins.
7. Fresh SpecialistDB copies share an identical closed template identity and
   never share a writable file.
8. Fixed-state feature and action identities are byte-identical within each
   paired checkpoint comparison.
9. Existing trainer, managed-run, checkpoint, common-anchor, Malom, DB-teacher,
   and label-provenance tests pass.

Any non-finite value, identity drift, transition overrun, missing pending
state, duplicate-control mismatch, database mutation, Sanmill error, or
checkpoint corruption stops the entire sequence. There is no automatic
retry, extension, resume, held-out evaluation, promotion, publication, or
long-training launch.

## Resource envelope for later planning

The scientific ceiling is six arms and 49,152 post-fork
optimizer-consumed transitions. A future managed contract may request at most
3,600 complete training games and six active wall hours as safety ceilings,
but those limits do not define successful completion. Reaching a resource
ceiling before all exact transition boundaries is a failed/incomplete run.

No resource grant or launch authorization is contained in this document.
