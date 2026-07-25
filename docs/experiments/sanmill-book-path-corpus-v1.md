# Sanmill Complete Book-Path Corpus v1

Status: infrastructure corpus contract frozen; evaluation use not frozen.

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
