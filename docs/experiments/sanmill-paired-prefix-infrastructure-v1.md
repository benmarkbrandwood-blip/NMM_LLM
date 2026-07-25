# Sanmill Paired-Prefix Infrastructure v1

Status: infrastructure contract frozen; evaluation policy not frozen.

This document defines the caller-side contract for obtaining reproducible
opening prefixes from Sanmill. It does not authorise a candidate-versus-
baseline match, choose a formal baseline, or freeze the provisional source
mixture.

## Scope and ownership

Sanmill commit
`db65eb3e73189d934d615d0f47519d395193c646` and the release binary recorded by
the strict logical-turn bridge own:

- replay of the complete action history;
- legal action and mandatory-removal semantics;
- logical-ply, no-capture, and repetition history;
- terminal status and termination reason;
- opening-book, Perfect DB, and HumanDB candidate enumeration; and
- source and history identities.

NMM_LLM owns:

- the pair identifier, experiment identifier, and random seed;
- assignment of a candidate source to a pair;
- selection from Sanmill's returned candidate array;
- persistence of the selected prefix and its evidence; and
- replay of the identical prefix before both colour-swapped games.

The data-query process is separate from strict MTD(f) play. Opening-book,
Perfect DB, and HumanDB data must remain disabled inside later search.

## Logical prefix

The v1 infrastructure generates a prefix from `startpos` containing exactly
eight logical plies: four completed turns by White and four by Black.

A placement or movement that forms a Mill and its mandatory removal use two
Sanmill action tokens but remain one logical ply. A prefix is valid only when:

- it begins at `startpos` with `history_origin=game_start`;
- every selected candidate is a complete logical turn;
- no selected boundary has a pending removal;
- the final counts are eight logical plies and `[4, 4]` by side; and
- the final history and FEN are confirmed by `history_summary`.

Both games in a pair consume the same immutable action-token sequence. The
sampler must run once per pair, not once per game or colour assignment.

## Supported sources

### Corrected opening book

The bundled NMM opening book must have SHA-256
`cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5`,
109 Oracle positions, and 437 Oracle records. The query must return
`candidate_order=source_array`. Candidate-selection weighting is an explicit
experiment policy; source order alone does not silently choose a move.

### Perfect Database

The caller supplies the machine-local path through the ignored path registry.
Every query must report `query_mode=strict_steps`, `fallback=none`, and
`candidate_order=full_turn_uci_lexicographic`. Only candidates tied under
StrictSteps may be sampled. The database identity must be recorded without
committing its host-specific root path.

### HumanDB

Sanmill may expose stable human frequencies and outcomes from an explicit
schema-v2 database. This interface is available for audit and later policy
work. HumanDB is not automatically a third prefix source. Its use requires a
separate pre-result decision that records the database SHA-256 and considers
whether the same data influenced training.

## Policy separation

The sampler accepts an explicit source and candidate-selection policy. It has
no default 75/25 mixture and no implicit fallback chain.

The previously discussed 75% corrected-book and 25% Perfect DB assignment is
only a proposed infrastructure-smoke policy. Before it is used, the owning
experiment must freeze:

- source weights and the deterministic pair-assignment algorithm;
- candidate weighting for each source;
- the seed and pair-identifier namespace;
- required source identities; and
- whether a source miss aborts the whole smoke or invalidates only the
  affected predeclared pair.

HumanDB cannot be added by renormalising existing weights after results are
visible.

## Determinism and evidence

Every selection draw is derived from a versioned SHA-256 domain containing
the experiment identifier, pair identifier, base seed, source kind, logical
ply index, and draw purpose. Python's process-randomised `hash()` and mutable
global PRNG state are forbidden.

The immutable prefix record must contain:

- its schema and content identity;
- experiment, pair, seed, source, and candidate-policy identities;
- the pinned Sanmill source and binary identities;
- portable data-source identity;
- every selected candidate identity and stable index;
- the complete action-token history;
- per-step input and output history SHA-256 values;
- final FEN, logical counts, and final history SHA-256; and
- an explicit statement that the record is shared by games 0 and 1.

Absolute database and checkout paths are runtime inputs only. Durable records
name the ignored registry lookup key and content identities.

## Fail-closed rules

Prefix generation stops without substitution when any of the following
occurs:

- the process exits, times out, writes malformed JSON, or emits extra output;
- protocol version, request ID, operation, fields, or status differ;
- Sanmill returns `error`, a source miss, or a terminal state before completion;
- a source identity changes or violates its source-specific contract;
- Perfect DB reports any mode other than `strict_steps`;
- a candidate is malformed, incomplete, duplicated, or source-incompatible;
- the selected action-token sequence is illegal on replay;
- FEN, action count, logical count, side count, or history SHA-256 drifts;
- a removal remains pending at a training or evaluation boundary; or
- the final prefix does not contain exactly eight logical plies split `[4, 4]`.

There is no fallback from book to Perfect DB, HumanDB, search, a lower-ranked
candidate, or a random legal action. A policy that wants another source must
assign that source before querying the pair.

## Acceptance boundary

Infrastructure v1 is accepted only after focused tests demonstrate:

- strict parsing of available, miss, terminal, and error responses;
- rejection of unknown fields and source-mode drift;
- deterministic source assignment and candidate selection;
- one-logical-ply handling of primary-plus-removal turns;
- exact eight-ply and `[4, 4]` final counts;
- identical prefix reuse for both games of a pair;
- history/FEN/source identity mismatch rejection;
- byte-stable results from two fresh local data-query processes; and
- a book-only black-box prefix smoke against the pinned binary.

A Perfect DB or mixed-source smoke, formal node budget, candidate model, match
length, result interval, and launch remain separate decisions.

## Implementation evidence

The contract is implemented on `dev` in three independent commits:

- `fc7847f` freezes this infrastructure boundary;
- `a4e166e` adds the strict JSONL data-query client and parsers; and
- `d6ea9f5` adds deterministic source assignment, candidate sampling, and the
  paired-prefix record.

The focused Sanmill UCI, data-query, and prefix suites report `60 passed`.
The complete repository suite at
`d6ea9f5fad24744a00a0177e3ceae17db8f4678a` reports `1022 passed` and
`498 subtests passed` in 3306.78 seconds. This run did not load a candidate or
play a candidate-versus-baseline game.

Two fresh data-query processes produced byte-identical eight-logical-ply
book prefixes for the frozen diagnostic identity
`book-prefix-black-box-v1`, pair `pair-12`, seed 42, using the explicit
`source_declared` book policy. The final count was `[4, 4]` and the record
states that games 0 and 1 share the prefix.

The book is deliberately fail-closed but sparse. The same diagnostic namespace
for `pair-0` reached `book_miss` before selecting logical ply 5, so an arbitrary
pair identifier is not guaranteed to yield eight consecutive book moves.
There was no Perfect DB or random substitution. Before a 75/25 smoke can be
frozen, it must choose one of these pre-result policies:

- freeze an audited set of pair identifiers that have complete book paths;
- freeze and sample a complete book-path corpus directly; or
- redefine the proposal as an explicit per-ply source schedule.

Silently filling a book miss with Perfect DB is not an allowed interpretation.

A read-only query of the configured standard Perfect DB returned `available`,
24 tied initial candidates, `fully_available=true`, and the expected
`std.secval` SHA-256
`5078bf84505fe2845a4af7c36907efa2d66b2eb76f149ce12faa248117405b68`.
This was a source probe, not an eight-ply Perfect DB or mixed-source smoke.

The configured active HumanDB currently fails closed with
`database_not_immutable` because `human_db.sqlite-shm` is non-empty. No
sidecar was removed or altered. HumanDB remains outside the prefix policy, and
this local state must be resolved by the database owner before any later
HumanDB query smoke.

After the complete suite passed, the referenced Sanmill checkout acquired two
untracked interchange-format documents under `docs/standards/`. The strict
installation audit now rejects that dirty checkout. These files belong to the
other workspace and were not modified here. A new bridge or prefix smoke must
wait until the Sanmill workspace owner records or otherwise resolves them and
the pinned-scope audit is rerun.
