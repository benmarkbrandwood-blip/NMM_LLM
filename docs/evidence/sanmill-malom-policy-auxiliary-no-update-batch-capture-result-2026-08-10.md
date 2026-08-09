# Sanmill Malom policy-auxiliary no-update batch-capture result — 10 August 2026

## Decision

The authorized three-seed, 60-game no-update batch capture completed exactly
once.  Its result is accepted as gradient-scale diagnosis and does not select
a coefficient, normalization rule, candidate, or training lineage.

The immutable local result is
`out/diagnostics/sanmill-malom-policy-auxiliary-no-update-batch-capture-v1-20260809-001.json`:

- result identity:
  `b0dfd3415c55196c59e71cf67e45b00ab5844e9f62fbc9f3bdc31b09a694bd86`;
- file SHA-256:
  `2e310ecccf869f16b314093c6f50395e91019839b603d805ad3e0cab9d651fee`;
- readiness identity:
  `f8858720cafd994d9471af3312d9d428ffa4278b78883a66ec19ddb4954404f7`;
- plan identity:
  `a5c85ed13baecf3efed6780effdf590e97560a12e9ab197c5fc7bb4bf7341fab`;
- source commit:
  `9b4655872cebc62e9e643039d24f6dfcd7da565d`; and
- source tree: `66c054382ed462b1a0cb213ab6dd0b61b544d769`.

Independent recomputation matched the result identity, all 60 sample
identities, and all 19 batch identities.  The run started at
`2026-08-09T17:33:50.647676Z`, completed at
`2026-08-09T17:39:23.897286Z`, and used 330.3353 active wall seconds.  No
failure report exists and the one-run authorization is consumed.

## Observed facts

### Frozen route and resource use

Seeds 52, 53, and 54 each supplied 20 complete games: eight against the
pinned 1,000-node Sanmill route and twelve against that seed's fresh frozen
target.  Each opponent source was independently colour-balanced.  One route
per seed used the preregistered depth-12 feature path; all others used depth 5.

The run produced 2,965 logical plies and 1,473 learner steps.  Its 340 Sanmill
search calls requested 340,000 node ceilings and expanded 286,691 nodes.  It
therefore remained below the frozen ceilings of 7,200 logical plies, 1,440
search calls, 1.44 million requested node ceilings, 33 batches, and two active
hours.

Nineteen production-shaped batches were measured.  Batch sizes ranged from 54
to 114 learner steps, with median 77.  The whole-game threshold behavior was
preserved: trajectories were not cut at 64 steps.  Eighteen batches were
periodic and the one final-flush batch contained 54 steps.  No residual was
excluded.

The strict referee produced 29 `lose_fewer_than_three` terminals, 24
`lose_no_legal_moves` terminals, four threefold-repetition draws, and three
max-ply truncations.  Thus 57 of 60 games ended by a rules terminal rather than
the 120-ply diagnostic ceiling.

### No-update and mutation checks

Every learner and frozen-target state SHA-256 was unchanged before and after
its seed.  Every learner remained `requires_grad=false`, no `.grad` was
populated, and the learner initially and finally matched its frozen target.
No optimizer was constructed; optimizer steps, training updates, and
`backward()` calls were all zero.

HumanDB, the empty corrected SpecialistDB, source-evidence artifacts, tracked
source, and all three Sanmill installations were equal before and after the
run.  Neither SQLite input acquired a WAL or SHM sidecar.  HumanDB's
unversioned historical Malom columns remained disabled; they were not treated
as labels.

### Label support by phase

All 19 batches contained at least one informative exact-WDL preserving set.
The complete phase support was:

| Phase | Labelled steps | Informative steps | Informative rate |
| --- | ---: | ---: | ---: |
| Placement | 540 | 247 | 45.74% |
| Movement | 813 | 179 | 22.02% |
| Flying | 120 | 27 | 22.50% |
| Total | 1,473 | 453 | 30.75% |

This is training-route support.  It is not a held-out class-accuracy or
generalization metric.

### Gradient distribution

The unscaled auxiliary policy-head gradient was larger than the ordinary
policy-head gradient in every measured batch:

| Quantity | Minimum | Median | P90 nearest rank | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Ordinary policy-head gradient L2 | 0.0001638 | 0.0002224 | 0.0003206 | 0.0003251 |
| Raw auxiliary gradient L2 | 0.0002708 | 0.0005621 | 0.0008194 | 0.0009113 |
| Raw auxiliary / ordinary policy-head | 1.1605 | 2.3946 | — | 5.1985 |
| Auxiliary-to-policy-head cosine | -0.2428 | 0.1780 | 0.4661 | 0.4774 |

Fourteen cosines were positive and five were negative.  Negative alignment is
not automatically a failure: the exact-WDL signal may oppose an ordinary RL
policy gradient that favours a value-downgrading action.  This run did not
apply either direction, so it cannot determine whether that opposition is
beneficial during optimization.

The preregistered diagnostic target grid implied these detached per-batch
coefficients:

| Target auxiliary / ordinary policy-head norm | Minimum | Median | P90 nearest rank | Maximum |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.0481 | 0.1044 | 0.1849 | 0.2154 |
| 0.50 | 0.0962 | 0.2088 | 0.3697 | 0.4309 |
| 1.00 | 0.1924 | 0.4176 | 0.7395 | 0.8617 |

No target or coefficient was selected by the runner.

### Multiple-seed evidence

| Seed | Games | W/D/L | Learner steps | Batches | Informative steps | Raw ratio median | Cosine median | Target-0.25 coefficient median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 | 20 | 0/3/17 | 474 | 6 | 170 | 2.0402 | 0.2244 | 0.1267 |
| 53 | 20 | 6/4/10 | 613 | 8 | 174 | 2.1560 | 0.0030 | 0.1175 |
| 54 | 20 | 0/0/20 | 386 | 5 | 109 | 2.3946 | 0.4379 | 0.1044 |

The target-0.25 coefficient still ranged from 0.0481 to 0.2154 across
individual batches.  Seed 53 had nearly orthogonal median alignment, while
seed 54 had the highest median alignment but also the most negative individual
cosine.  A pooled median therefore does not describe every seed or batch.

The full W/D/L was 6 wins, 7 draws, and 47 losses.  The fresh learner lost all
24 Sanmill games; against frozen targets it scored 6 wins, 7 draws, and 23
losses.  These outcomes describe randomly initialized diagnostic routes and
are not a strength baseline, training curve, or promotion result.

## Hypothesis

A fixed raw auxiliary coefficient is a poor control of optimizer influence
because the raw auxiliary-to-policy-head gradient ratio changes materially by
batch and seed.  Detached per-batch normalization to a conservative fraction
of the ordinary policy-head norm may produce a more stable learning
intervention than another fixed coefficient.

The lowest preregistered target, 0.25, is the first candidate to test.  This is
a post-result experiment-design recommendation, not adoption into retained or
long training.

## Supporting evidence

- Every batch was informative and finite, so the measured spread is not an
  artifact of fabricated zero-support scales.
- A fixed coefficient of 0.10 would have applied approximately 0.116 to 0.520
  times the ordinary policy-head norm across these batches, despite being near
  the target-0.25 median.
- The preceding one-seed four-arm calibration was
  `inconclusive_recalibration_required`: no active coefficient crossed its
  preserving-mass threshold, while coefficient 0.30 crossed the frozen
  loss-scale limit.
- The earlier two persisted-batch audit measured applied ratios of 0.69 and
  26.7.  The current 19-batch distribution is narrower but independently
  confirms that loss magnitude and a fixed coefficient do not reliably bind
  policy-head gradient influence.

## Counterevidence and limits

- Nineteen batches across three fresh seeds are sufficient to design a bounded
  calibration, not to establish a universal gradient distribution.
- The run performed no update, so it has no train curve, validation curve,
  optimizer-state response, fixed-state probability change, learned
  per-phase downgrade rate, or strength result.
- The three seeds generated different trajectory lengths and phase mixtures.
  This is genuine route variation but also means that seed medians are not
  repeated measurements on identical batch contents.
- Only 27 informative flying steps were observed.  Flying-specific conclusions
  therefore remain especially weak.
- A negative cosine may be corrective or destabilizing.  Magnitude
  normalization alone cannot answer which interpretation is true.
- The previous four-arm experiment used seed 51 and evolving writable
  SpecialistDB copies.  It is mechanism context, not a pooled control sample
  for seeds 52–54.

## Next validation experiment

Do not start retained or long training from this result.  Implement an
explicit, deterministic, checkpoint-persisted normalization mode that scales
the preserving-set loss to target 0.25 of the ordinary policy-head gradient
norm.  It must:

1. use detached finite gradient norms from the same production batch;
2. fail closed below the existing `1e-12` denominator boundary;
3. apply a preregistered coefficient cap rather than allowing an inverse-norm
   explosion;
4. log raw norms, cosine, effective coefficient, cap status, and phase support;
5. preserve the existing fixed-coefficient mode for a controlled comparison;
   and
6. persist and exact-resume every normalization setting and relevant counter.

Prepare a fresh three-seed control-versus-normalized calibration with one
factor changed, identical schedules within seed, at most 600 total games and
two active hours.  Compare raw and complete-window train curves, fixed-state
preserving mass, per-phase and per-opponent downgrade rates, policy health,
finite updates, and seed-level effects.  Ordinary RL has no supervised
validation curve; the fixed development corpus must remain explicitly named
development evidence.  The preparation must stop at readiness and requires a
new explicit identity-bound launch authorization.
