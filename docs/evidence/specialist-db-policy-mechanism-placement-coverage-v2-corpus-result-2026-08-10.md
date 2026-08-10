# SpecialistDB Placement-Coverage Corpus v2 Result

## Outcome

Status: `coverage_gate_passed`

Corpus:
`specialist-db-policy-mechanism-placement-coverage-v2-corpus-2026-08-10.json`

Corpus SHA-256:
`5d0f1f548ebf7bc2d9522a345de5e375f2ab8fae7f37ef6caf976d92aebfa88b`

Evidence ID:
`4e3abbd8eb99ca9572a6524e83d71c91c058e0c331c5d2f0674b54e335c6558d`

Builder commit:
`aa41f1538e82c76a2a98bd23f5dc0a578cc1a599`

## Observed facts

- The builder loaded no checkpoint or model.
- All 64 frozen source histories replayed through 12 logical ply and yielded
  489 unique exact-FEN pre-move states after transposition deduplication.
- Across 8,691 legal successors, 688 had an empirical W/D/L distribution at
  the frozen three-sample support floor and 208 had a trusted theoretical
  label.
- Exactly 100 unique states had at least one empirical successor and were all
  retained. There was no ranking, quota, tie-break or cap.
- The frozen gate required at least 64 states and 500 empirical successor
  hits, so both requirements passed.
- The byte-identical local audit snapshot had SHA-256
  `82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe`
  before and after construction. It had no SQLite sidecars.

## Snapshot provenance

The original completed-run database had the same frozen SHA-256. An initial
tool attempt used a non-immutable SQLite `quick_check`, which created a zero-
byte WAL and an SHM sidecar on the original path before the guarded reader
stopped. No corpus output was produced by that attempt.

No sidecar was deleted. The successful run used a direct byte-for-byte copy of
the unchanged main database file after confirming that its WAL was zero bytes.
The builder was fixed first to use immutable read-only SQLite semantics, and a
focused regression test now verifies that `quick_check` creates no sidecars.

## Hypothesis and claim boundary

The coverage result supports the hypothesis that the v1 zero-difference result
was caused by corpus mismatch: a source-derived placement corpus exposes ample
empirical SpecialistDB coverage. It does not yet show that the final candidate
policy changes when those features are removed.

Because every selected state is in placement, the next result cannot be
generalized to movement or flying. The corpus is development evidence, not an
independent strength or promotion set.

## Next validation experiment

On the committed corpus identity above, load the fixed final retained-v3
checkpoint and execute the unchanged four-projection audit. Apply the v1
material-sensitivity thresholds without adjustment. Only a material result may
route to preparation of the paired three-seed single-factor calibration.
