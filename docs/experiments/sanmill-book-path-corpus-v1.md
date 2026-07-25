# Sanmill Complete Book-Path Corpus v1

Status: infrastructure corpus contract and artifact frozen; evaluation use not
frozen.

This contract defines a reproducible inventory of every corrected Sanmill NMM
opening-book branch that can complete exactly eight logical plies from
`startpos`. It addresses the observed fact that some sequential book choices
reach `book_miss` before eight plies. It does not choose the later 75/25 source
mixture, candidate weighting, formal baseline, or evaluation launch.

## Source and replay boundary

The builder uses the pinned Sanmill data-query protocol and bundled corrected
book identity recorded by
[`sanmill-paired-prefix-infrastructure-v1.md`](sanmill-paired-prefix-infrastructure-v1.md).
Every node is replayed from `startpos` with its complete chronological Sanmill
action-token history. Sanmill remains authoritative for legality, mandatory
removal, logical counts, FEN, and history SHA-256.

The corpus must be host-path-free. It records the pinned Sanmill source and
binary identities plus the bundled book content identity, not the local
checkout path.

## Enumeration semantics

Enumeration starts with one empty prefix and expands every returned candidate
at each of eight logical depths.

- An `available` response contributes every complete candidate in Sanmill's
  stable source order.
- A `book_miss` is recorded as an incomplete leaf and is not expanded.
- A rules-terminal state before depth eight is recorded as an incomplete leaf.
- A protocol, identity, legality, count, FEN, or history error aborts the
  entire build.
- No book miss or terminal leaf invokes Perfect DB, HumanDB, search, or a
  random legal action.

Every candidate transition is independently replayed through
`history_summary`. Its output must be stable, non-pending, and exactly one
logical ply later. At depth eight, every retained path must have counts
`logical_ply_count=8` and `logical_plies_by_side=[4,4]`.

A movement or placement plus mandatory removal is one logical edge even when
its path adds two action tokens.

## Identity and deduplication

The v1 path key is the complete chronological action-token sequence. Paths are
sorted lexicographically by that sequence. Exact duplicate histories are an
error.

The builder does not deduplicate:

- final boards reached through different histories;
- rotational or reflection symmetries;
- paths that share a candidate group or named opening; or
- paths according to HumanDB or Perfect DB value.

Those transformations would change the sampling population and require a
separate pre-result policy.

Each path records:

- a content-derived path identity;
- all action tokens and eight logical step records;
- source candidate identity, stable index, source group, and rank per step;
- input and output FEN and history SHA-256 per step; and
- final FEN, action count, logical count, and per-side counts.

The corpus records a content identity over its complete ordered body. A loader
must recompute path and corpus identities and reject unknown or missing fields.

## Branch audit

For every input depth from zero through seven, the corpus records:

- number of input prefixes;
- available, book-miss, and terminal input counts;
- number of candidate edges returned;
- number of unique child histories; and
- number of compound primary-plus-removal edges.

The final summary records complete path count and incomplete-leaf totals.
These counts describe the pinned book graph only; they are not game-strength
or human-likeness statistics.

## Policy boundary

Building this corpus does not make it a formal opening distribution. A later
experiment must still freeze one explicit policy, for example:

- uniform selection over complete paths;
- a mathematically specified path weight derived from source ranks;
- stratification by named opening, symmetry, or final position; or
- rejection of this corpus in favour of an explicit per-ply source schedule.

The policy must be frozen before candidate results are viewed. HumanDB
frequencies may be analysed separately but are not used by this v1 builder.

## Acceptance evidence

The implementation must demonstrate:

- deterministic enumeration from two fresh data-query processes;
- exact equality of their canonical corpus bodies and identities;
- one-logical-ply handling of a compound action in a fixture;
- audited `book_miss` pruning without fallback;
- rejection of duplicate histories and source/history identity drift;
- strict load-time recomputation of every identity; and
- a local build against the pinned corrected book.

No candidate model is loaded and no game is played during this build.

## Frozen artifact evidence

Implementation commit
`024d1f891ac81e8cd0f2b7c25b22fbec28947d7d` added the fail-closed
enumerator, strict loader, exclusive builder, and regressions. From that clean
commit, the builder opened two fresh Sanmill data-query processes and obtained
identical canonical corpus bodies before publishing
[`sanmill-book-path-corpus-v1.json`](sanmill-book-path-corpus-v1.json).

The frozen identities are:

- corpus identity
  `3bc9bc05a66a1a53255444266388838489020667272fc2ffa7445e7cf44be985`;
- corpus file SHA-256
  `490537d892e4dc64b0b46331754bab448a3b3d99dad620131cb692916e540ceb`;
- bundled-book portable identity
  `58101aa9b7f58f30a0489c0d85a991ba78e3147d94a87e5456af3f2167f58eaf`;
- pinned Sanmill commit
  `db65eb3e73189d934d615d0f47519d395193c646`; and
- pinned Windows release binary SHA-256
  `cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc`.

The exhaustive depth audit is:

| Input logical ply | Input prefixes | Available | Book miss | Candidate edges | Compound edges |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 0 | 8 | 0 |
| 1 | 8 | 8 | 0 | 40 | 0 |
| 2 | 40 | 40 | 0 | 76 | 0 |
| 3 | 76 | 76 | 0 | 140 | 0 |
| 4 | 140 | 48 | 92 | 264 | 0 |
| 5 | 264 | 64 | 200 | 232 | 0 |
| 6 | 232 | 80 | 152 | 128 | 48 |
| 7 | 128 | 64 | 64 | 192 | 128 |

The result contains 192 complete exact histories and 508 audited
`book_miss` leaves, with no terminal leaf and no fallback. The 192 histories
have 192 distinct history SHA-256 values but only 84 distinct final FENs.
That difference is expected under the frozen path-equivalence rule and is
important for the later sampling decision: uniform-over-paths is not the same
distribution as uniform-over-final-positions.

The corpus-specific suite reports `6 passed`, including two fresh real
Sanmill processes. The combined book-corpus, data-query, and prefix suites
report `25 passed`. The prior clean Sanmill UCI/data-query/prefix regression
run remains `60 passed`; the 55-minute full repository suite was not repeated
for this artifact-only step.

This artifact closes the technical option to inventory complete book paths.
It does not select uniform path sampling, source-rank weighting, a 75/25
book/Perfect DB mixture, HumanDB participation, or any evaluation launch.
