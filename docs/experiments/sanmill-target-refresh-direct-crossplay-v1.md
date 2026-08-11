# Target-refresh direct cross-play v1

Status: `designed_unlaunched_needs_authorization`

This is the next bounded development diagnostic after the schedule-isolation
recovery. It performs no training. It asks whether the late policy separation
between `refresh-once` and `no-refresh` has a direct game consequence when the
two frozen policies play each other.

## Observation facts

- The completed schedule-isolation experiment compared three paired seeds at
  4,096 and 8,192 post-fork learner transitions.
- Its Sanmill-anchor outcome classifier found no material paired effect.
- Its full-action policy analysis found late separation at 8,192 transitions
  for seeds 67 and 68, but not seed 69 and not persistently at 4,096.
- The signed Malom-preserving probability-mass difference at 8,192 favoured
  `no-refresh` in every seed and phase.
- Those findings are development evidence. They do not establish strength or
  select a long-training condition.

## Hypothesis

The common 1,000-node Sanmill measurement opponent may have imposed a floor
that hid a real behavioural difference between the paired policies. Direct
cross-play can expose a relative consequence without changing either policy.

## Supporting evidence

At 8,192 transitions, seeds 67 and 68 crossed the preregistered Jensen-Shannon,
total-variation, and Malom-mass thresholds. The final signed Malom-mass
direction was consistent across all three seeds and all three phases.

## Counterevidence

The divergence was late rather than persistent. Seed 69 remained below every
material policy threshold. The earlier paired outcome measurement was below
its effect gate and had severe win-rate floor effects. Therefore this plan
does not assume that `no-refresh` is stronger.

## Frozen measurement

- Use only the 8,192-transition `refresh-once` and `no-refresh` checkpoints
  from seeds 67, 68, and 69.
- Reconstruct both policies with the same per-seed game-50 frozen anchor for
  lookahead features.
- Use the twelve audited replay histories: four placement, four movement, and
  four flying starts.
- Run four replicates per seed and start, with both colour assignments. This
  yields 144 pairs and 288 games.
- Use temperature `0.2` and a common random stream for each colour across a
  swapped pair.
- Use Sanmill only as the strict portable referee. It does not select moves.
- Cap play after the replay start at 120 complete logical plies. An unresolved
  cap is reported separately as a development truncation.
- Run on CPU and perform zero training games, optimizer updates, database
  writes, or checkpoint writes.

## Frozen interpretation

The paired contrast is the no-refresh score rate minus the refresh-once score
rate. A directional result requires an aggregate effect of at least 8.33
percentage points, support of that size in at least two seeds, no opposite
seed worse than 4.17 points, and no more than 25% max-ply truncations.

The only possible conclusions are:

- `material_no_refresh_direct_effect`;
- `material_refresh_once_direct_effect`;
- `no_material_direct_effect`; or
- `inconclusive_truncation`.

None automatically selects, promotes, publishes, resumes, or starts a model.
Even a material result is not held-out strength evidence.

## Execution boundary

The JSON plan is the authority. The implementation, all checkpoint records,
the source analysis, the replay corpus, the strict replay audit, HumanDB trust
policy, and corrected Malom manifest are content-bound. Readiness must be
generated from clean published `dev` and must then receive one explicit
product authorization before a one-shot launch. An anomaly fails closed and
does not authorize a retry.
