# SpecialistDB Placement-Coverage Mechanism Audit v2

## Status and authority

Status: `frozen_unexecuted`

Audit ID: `specialist-db-policy-mechanism-placement-coverage-v2`

This follow-up is a read-only development diagnosis. It does not authorize
training, model updates, candidate-versus-baseline games, held-out evaluation,
promotion, publication, database rewriting or checkpoint generation.

## Reason for the follow-up

The v1 phase-covered audit inspected 1,583 legal successors but found no
usable SpecialistDB projection. Its zero policy difference is therefore a
coverage-negative result, not evidence that the database mechanism is
irrelevant.

This audit first constructs a coverage-positive corpus without loading any
candidate model. Only after that corpus and its identity are committed may the
final retained-v3 checkpoint be loaded for the same four-projection policy
comparison.

## Immutable source inputs

| Input | Identity |
| --- | --- |
| Complete-history source | `docs/experiments/sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.json` |
| Source SHA-256 | `3bcf9db2d003d10769b88767763eb7dfb950eecbff578b7c7ff7d1c208e19771` |
| Source composition | 22 Book, 21 HumanDB and 21 PerfectDB histories |
| Logical length | 12 logical ply per history |
| SpecialistDB | `data/specialist_db.sanmill_preserving_retained_v3.seed58.sqlite` |
| SpecialistDB SHA-256 | `82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe` |
| SpecialistDB label version | `sector-corrected-v1` |
| Empirical support floor | 3 samples |

The source histories were frozen independently of the retained-v3 candidate.
They are development evidence and are not a held-out promotion corpus.

## Candidate-blind corpus rule

1. Replay all 64 complete histories from a new board and fail closed on an
   illegal logical turn, step-order drift, flattened-action drift or final-FEN
   mismatch.
2. Consider the board immediately before each of the 12 logical turns.
3. Deduplicate states by exact NMM FEN. Preserve every source reference and
   stratum attached to a transposed state.
4. Query every complete legal-turn successor from the immutable SpecialistDB.
5. Retain every unique state having at least one empirical W/D/L distribution
   at the three-sample floor. Apply no ranking, tie-break, quota or cap.
6. Sort by minimum source logical ply and then exact FEN.

The builder must not accept, import or read a checkpoint or model. It must
record `candidate_loaded=false`, the source and database identities, complete
coverage counts, output identities, unchanged database SHA-256 and absence of
SQLite sidecars.

The coverage gate requires at least 64 selected unique states and at least 500
empirical successor hits. Failure stops the follow-up before any candidate is
loaded.

## Policy comparison

After the candidate-blind corpus is committed, re-run the v1 production-route
audit using the same final checkpoint, frozen target, HumanDB, corrected Malom,
lookahead settings, scheduled temperature and four SpecialistDB projections.

The primary contrast and material thresholds remain exactly those frozen in
v1: `full` versus `empirical_disabled`, with material sensitivity triggered by
at least three argmax changes, one critical Malom-preservation crossing or
mean scheduled-temperature total variation of at least 0.05.

## Claim boundary and routing

All available complete histories are placement prefixes. A positive result
therefore establishes placement-phase mechanism sensitivity only. It cannot be
generalized to movement or flying, and it is not a causal training or strength
result.

If the frozen material rule triggers, prepare but do not launch a paired,
three-seed, single-factor calibration of empirical SpecialistDB reads. If it
does not trigger, do not prepare that calibration solely from this mechanism
audit; return to the other registered hypotheses.
