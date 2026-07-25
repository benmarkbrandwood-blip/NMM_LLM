# Twelve-ply Perfect DB source audit

Status: `source-only-needs-decision`

This evidence audits a deterministic pool of twelve-logical-ply opening
histories produced exclusively from the corrected standard Malom Perfect
Database. It does not load a candidate model, play evaluation games, select a
final corpus, or assign a final Perfect DB quota.

## Frozen identities

- Generator commit:
  `0e31255d986799f01cb6814f55d86f8e6e4e52e8`
- Canonical evidence:
  `docs/evidence/sanmill-layered-perfect-source-audit-2026-07-25.json`
- Evidence byte length: `8,185,359`
- Evidence SHA-256:
  `c92393de709c88278972cab996ca1fbefac9130cec6bc8179f7abf4b568bab48`
- Audit identity:
  `fcac7e8e864e345669a497c600e4a901eff1f9203e3b83baf6895a89c2b0be56`
- Portable Perfect DB identity:
  `d6a85d25e87e28cc33e1408539585dadd83349da1cb63aa3f1a0f32307087508`
- `std.secval` SHA-256:
  `5078bf84505fe2845a4af7c36907efa2d66b2eb76f149ce12faa248117405b68`
- Fast sector-manifest SHA-256:
  `748ce88a9d1fae9fffb069fa5add0a09be7e890055044c299b755ac630d2e548`

The evidence contains no machine-specific database root. The local database is
resolved through `malom_db_path`.

## Query and selection contract

The audit fixed 128 route identifiers before inspecting the results. For each
route, it selected uniformly from the complete candidate array using a
SHA-256 rejection-sampled draw bound to the route, seed, logical ply, and
candidate-pool identity.

Every one of the twelve steps had to satisfy all of these conditions:

- query mode `strict_steps`;
- candidate order `full_turn_uci_lexicographic`;
- fallback `none`;
- one complete logical turn, including any mandatory removal;
- only candidates tied for the best Perfect DB outcome;
- a stable portable Perfect DB identity.

StrictSteps prefers faster wins and slower losses. Draw candidates with
different step values remain tied, so the audit records all raw step values
without incorrectly treating them as a draw tiebreak.

Two completely fresh Sanmill data-query processes generated the full payload.
Their canonical JSON bytes were identical.

## Results

| Measure | Result |
| --- | ---: |
| Fixed audit routes | 128 |
| Unique exact histories | 128 |
| Unique final FENs | 128 |
| Unique ring16 final orbits | 128 |
| Logical-ply selections | 1,536 |
| Selected draw / WDL 0 outcomes | 1,536 |
| Steps with multiple tied-best candidates | 1,450 |
| Steps with one best candidate | 86 |
| Candidate-pool size range | 1–24 |

The 128-route audit pool therefore demonstrates at least 128 distinct
exact-history, exact-position, and ring16-structure candidates under this
deterministic selection contract. It does not claim that 128 is the total
number available.

## Cross-source overlap

The Perfect DB routes were compared with all frozen named-Book records and all
83,002 unique genuine HumanDB twelve-ply histories.

| Prior source | Exact history | Final FEN | Ring16 orbit |
| --- | ---: | ---: | ---: |
| Book | 0 | 0 | 0 |
| HumanDB | 0 | 0 | 0 |

These zero overlaps make the audited Perfect DB pool independently useful for
the structural-diversity stratum. They do not make its histories human-common,
and they are not strength-promotion evidence.

## Decision boundary

The route pool is audit evidence, not the final corpus. The final 64-prefix
source counts remain unfrozen until the Book, HumanDB, and Perfect DB findings
are compared in the corpus decision brief. Placement prefixes also remain
separate from the existing movement/flying phase-coverage corpus.
