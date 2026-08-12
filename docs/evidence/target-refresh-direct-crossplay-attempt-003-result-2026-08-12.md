# Target-refresh direct cross-play attempt 003 result

Status: `completed_validated_development_evidence`

The immutable attempt-003 plan completed once on 12 August 2026. The run
consumed 288 CPU no-update games, arranged as 144 colour-swapped pairs across
seeds 67, 68 and 69 and placement, movement and flying starts. It performed
no training, optimizer, database or checkpoint writes.

The machine-readable evidence is
[target-refresh-direct-crossplay-attempt-003-result-2026-08-12.json](target-refresh-direct-crossplay-attempt-003-result-2026-08-12.json).
Its evidence identity is
`9a5df62da2c605d580c3d519d8aa18d45e7e444639eb458dd93976f31fcb8d19`.

## Observed facts

- The no-refresh policy scored `178 W / 12 D / 98 L`, or `0.6388889`.
- The paired mean score effect, defined in advance as no-refresh minus
  refresh-once, was `+0.2777778`.
- Every seed pointed in the same direction: seed 67 `+0.5208333`, seed 68
  `+0.1979167`, and seed 69 `+0.1145833`.
- Every phase pointed in the same direction: placement `+0.2916667`, movement
  `+0.3958333`, and flying `+0.1458333`.
- Nine games reached the development truncation cap, a rate of `0.03125`,
  below the frozen `0.25` invalidation threshold.
- The frozen classifier therefore returned
  `material_no_refresh_direct_effect`.
- Independent publication re-read the canonical LF ledger, verified the
  complete plan/readiness/authorization/launch/result/completion identity
  chain, and reproduced every scientific field.

## Working hypothesis

Refreshing the frozen target at the original game-50 boundary disrupted the
learner more than leaving that early target unchanged over the tested 8,192
post-fork transitions. The evidence supports that narrow timing claim. It
does not identify the best later refresh cadence.

## Supporting evidence

- The aggregate effect is substantially larger than the preregistered
  `1/12` material-effect threshold.
- All three independent seeds support the same direction; none crosses the
  preregistered opposite-direction tolerance.
- Placement, movement and flying starts all support the same direction, so
  the result is not confined to one phase.
- Colour swapping and common per-colour random streams reduce first-player
  and sampling-stream confounding within each pair.
- Sanmill acted only as the strict portable referee, so differences came from
  the two frozen policies rather than an opponent-search schedule.

## Counterevidence and limits

- The no-refresh score was asymmetric by colour: `0.7777778` as Black and
  `0.5000000` as White. Pairing protects the primary contrast, but this is a
  reason not to reinterpret the raw score as general strength.
- Seed effects vary materially, from `+0.1145833` to `+0.5208333`.
- The games use a small development corpus and training-policy sampling at
  temperature `0.2`; they are not a held-out strength evaluation.
- A permanently stale frozen target would eventually cease to represent the
  improving learner and is not a defensible retained curriculum merely
  because it won this comparison.
- This run does not compare a slower cadence, a segment-boundary cadence, or
  a mature-policy refresh.

## Next discriminating experiment

Use each seed's mature no-refresh checkpoint at the 8,192-transition boundary
as a common fork. Refresh the frozen target exactly once from that mature
policy in the treatment arm and leave the existing target unchanged in the
control arm. Hold consumed post-fork transitions, update count, temperature
sequence, starts, sampling streams and fixed 1,000-node Sanmill work equal.
Compare policy distributions first, then paired development games across all
three seeds. This directly tests whether a later, segment-aligned refresh is
beneficial without selecting permanent no-refresh by default.

This evidence is development-mechanism evidence only. It is not held-out
strength, promotion, publication, or authorization for long training.
