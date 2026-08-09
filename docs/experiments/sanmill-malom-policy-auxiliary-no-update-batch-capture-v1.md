# Sanmill Malom policy-auxiliary no-update batch capture v1

## Decision state

This experiment is `prepared_unlaunched`.  Its only purpose is to measure the
distribution of ordinary A2C policy-head and exact-WDL preserving-set gradients
on production-shaped rollout batches.  It does not select a coefficient or
normalization rule and does not authorize training.

The preceding four-arm calibration was inconclusive.  Its two recoverable
production batches showed that the applied auxiliary gradient was 0.69 times
the ordinary policy-head gradient in one batch and 26.7 times it in another.
The loss ratio did not reveal that variation.  A multi-seed batch distribution
is therefore required before choosing any normalization target, cap, or
low-gradient behavior.

## Frozen scope

- Three fresh random models use seeds 52, 53, and 54.  No candidate checkpoint
  is loaded.
- Each seed contributes 20 complete primary games: eight against the pinned
  1,000-node Sanmill route and twelve against its own fresh frozen target.
- Each opponent source is independently color-balanced.  One route per seed
  uses the production 12-ply deep feature path; its local index rotates across
  seeds so route depth is not confounded with one opponent/color cell.
- The rollout uses the production trainer, strict Sanmill referee, pinned
  read-only HumanDB, empty sector-corrected SpecialistDB snapshot, and corrected
  Malom tablebase identities inherited from the immutable parent route plan.
- Exact Malom action labels and the `malom-preserving-only` reward route are
  enabled.  The label-capture trigger does not enter an optimizer because no
  optimizer exists.
- Sentinel, legacy value/gap nets, imitation, opening forcing, recovery,
  confirmation/retry rollouts, branches, and persistence are disabled.

## Production batch boundary

The trainer's `update_every=64` behavior is a threshold, not a fixed slice.
It appends each complete game trajectory and, after that game, updates the
entire accumulated batch when it has at least 64 learner steps.  The batch can
therefore contain more than 64 steps.  At process end, a residual batch is
eligible only when it has at least the production minimum of eight steps.

This probe must reproduce that boundary exactly.  It must retroactively rescore
every complete trajectory before appending it, measure and clear a periodic
batch only after the threshold is reached, record an eligible final residual,
and preserve an ineligible residual as explicit excluded evidence.

## Read-only gradient measurement

For every eligible batch, the probe measures:

- policy, entropy, value, and exact-WDL auxiliary objective values and raw
  gradient norms;
- ordinary policy-plus-entropy gradient norm;
- auxiliary-to-ordinary policy-head cosine;
- exact label support by placement, movement, and flying phase; and
- diagnostic effective coefficients for target policy-head ratios 0.25, 0.5,
  and 1.0.

The target grid is descriptive.  It does not select a target.  A batch with no
informative action set, a non-finite value, or a denominator at or below
`1e-12` must be reported explicitly; it must not receive a fabricated scale.

The report also disaggregates support by seed, opponent source, learner color,
termination reason, and phase.  It records train-route data only: there is no
held-out validation curve, strength baseline, promotion result, or ablation
claim.

## Resource and safety boundary

The immutable maximum is 60 games, 7,200 logical plies, 24 Sanmill games,
1,440 search calls, 1.44 million requested Sanmill nodes, 33 measured batches,
and two active hours.  Any route, identity, label, finite-state, referee,
database, mutation, or resource failure stops the entire sequence.  There is
no automatic retry, extension, training, continuation, promotion, or
publication.

Preflight may verify identities and run an unscheduled two-ply no-search route
check, but execution still requires a separate explicit one-run authorization
bound to the readiness identity.
