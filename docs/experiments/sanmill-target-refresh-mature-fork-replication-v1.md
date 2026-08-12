# Mature target-refresh fork replication v1

Status: `designed_unlaunched_needs_publication`

Plan identity:
`8071e4a011d7b2fb8adb2d84a42233182ed22a9de85afdebed0eff07e589299f`

The authoritative machine-readable contract is
[`sanmill-target-refresh-mature-fork-replication-v1.json`](sanmill-target-refresh-mature-fork-replication-v1.json).
This document does not authorize launch.

## Decision context

The completed seeds 67–69 mature-fork cohort compared one mature target
refresh with retaining the stale game-50 target. Its 288 no-update games
produced a paired `refresh-mature minus stale-control` score effect of
`-0.0763889`, just inside the preregistered material boundary of
`0.0833333`. Seed 67 supported the stale target, while seeds 68 and 69 were
near zero. The frozen result therefore classified the cohort as
`no_material_direct_effect` and selected no successor condition.

That result is not evidence for choosing seed 67 or the stale target by
itself. This plan uses the disjoint seeds 64–66 to test whether either
direction independently replicates and then applies one frozen pooled
six-seed gate. The gate can provide a target-cadence input to a separately
designed retained plan; it cannot authorize or select long training.

## Frozen sources

Seeds 64–66 already have untreated `no-refresh` checkpoints at exactly
8,192 post-game-50 optimizer-consumed transitions. Reusing those checkpoints
avoids retraining their prefixes and does not reuse any seed from the prior
mature cohort.

The source checkpoint file identities are:

- seed 64:
  `db3729776805380727153d20f33528ca34fdd5804147d1d2d8aca314f3e41eb9`;
- seed 65:
  `5804fd1bda0091acc7b7df506dcbaedbed208c6230c877819d4e955a99100cde`;
- seed 66:
  `014f841e9180f0c2f9b915c805550ea07e84714f8c352c3f5b5032290ee21614`.

All three SpecialistDB inputs have label version
`sector-corrected-v1`, pass `quick_check`, and match the asset identity in
their checkpoint. The seed-64 historical database has a zero-byte WAL and a
32,768-byte SHM sidecar. Both historical files remain untouched. Preparation
uses an ignored, byte-identical, closed main-file snapshot with SHA-256
`7a256312c00e63b321b6e96ead61ce8a36034ff10a9c2f5fdebb8a511d7faf19`.
The contract binds the original main file, both preserved sidecars and the
closed snapshot separately.

For each seed, preparation clears the pre-treatment pending transition queue
and normalizes temperature to `0.8379808850090307`. Model, Adam optimizer,
random states, data state, counters, curriculum, frozen target and all other
non-allowlisted recovery state remain equivalent. The two child treatments
are:

1. `refresh-mature`: copy the mature learner into the frozen opponent once;
2. `stale-control`: retain the original game-50 frozen opponent.

Neither arm refreshes the target again.

## Frozen exposure and order

Each of the six arms receives exactly 8,192 additional consumed transitions
in exact 64-transition A2C batches. The learning rate is fixed at `0.0001`;
temperature is indexed by post-fork transitions; the opponent mixture is 60%
frozen policy and 40% fixed 1,000-node Sanmill search. All components and
policy-health checks remain identical to the first mature cohort.

The immutable order is seed 64 refresh/control, seed 65 refresh/control, then
seed 66 refresh/control. Only one trainer may run at a time. Any arm failure
closes the one-shot sequence; there is no automatic retry, recovery, resume
or extension.

## Preregistered replication decision

The replication cohort first uses the original direct-effect rule. A material
direction requires an absolute aggregate paired effect of at least `1/12`,
support in at least two of three replication seeds, no opposite seed beyond
the original `1/24` guard and no more than 25% truncation.

A condition is selected as a cadence input only if all of these additional
cross-cohort gates pass in the same direction:

- replication-cohort aggregate effect is at least `1/12`;
- pooled six-seed aggregate effect is at least `1/12`;
- at least three of six seeds have an effect of at least `1/12`;
- no more than one seed has an effect of at most `-1/12`;
- pooled truncation is no more than 25%.

The final classes are
`replicated_material_mature_refresh_effect`,
`replicated_material_stale_target_effect`,
`no_replicated_material_effect`, and
`inconclusive_replication_truncation`. The decision record explicitly sets
`automatic_long_run_selection=false`.

## Resources, stops and claim boundary

The aggregate ceiling is 3,600 training games, 49,152 consumed transitions,
four active hours, 172,800,000 requested Sanmill node ceilings and 288
no-update measurement games. Each arm is capped at 600 games and 0.6 active
hours.

Source, prior-result, plan, runtime, rules, corpus or implementation identity
drift; cohort overlap; treatment contamination; non-finite state; database
or sidecar drift; Sanmill disagreement; policy-health failure; resource
exhaustion; or malformed cohort/pooled evidence stops the sequence.

The plan contains zero launch authorization. It permits no held-out
evaluation, promotion, publication, retained run or long training. Its only
claim is development replication evidence for one mature refresh and a
pooled six-seed cadence input.

