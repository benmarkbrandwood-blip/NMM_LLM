# Sanmill Layered Opening Prefix v2

Status: `needs_decision`

Decision date: 2026-07-25

This document freezes the source-only preparation contract for twelve-logical-
ply opening prefixes. It does not freeze a final corpus, a source ratio, an
evaluation baseline, or a match launch.

## Expert and product decisions

The opening prefix contains exactly twelve logical plies: six completed turns
by White and six by Black. A placement or movement that forms a Mill and its
mandatory removal still counts as one logical ply, even though Sanmill records
the primary action and removal as two action tokens.

The longer prefix was selected because twelve plies expose more of an opening's
strategic development than the earlier eight-ply diagnostic. The three source
strata are:

- `book`: author-curated representative opening variations;
- `human_db`: complete opening histories observed in the current PlayOK sample;
  and
- `perfect_db`: deterministic lines composed only of StrictSteps tied-best
  actions.

Every stratum remains separate in storage and reporting. A combined score may
be added only after the stratum results have been reported independently.
Placement-prefix coverage remains separate from the existing movement- and
flying-phase corpus.

The Mill expert considers one representative per author-defined Book variation
sufficient in principle, supports adding frequent HumanDB histories, and is
independently interested in Perfect DB lines. Those views guide the audits but
do not choose a representative when a source line expands into several complete
capture-resolved histories.

## Relationship to v1

The existing `nmm.sanmill-paired-prefix.v1` implementation and its evidence
remain an immutable eight-logical-ply diagnostic:

- `PREFIX_LOGICAL_PLIES` remains eight;
- final per-side counts remain `[4, 4]`;
- existing prefix and corpus identities remain unchanged; and
- the historical seven-Book-orbit and 57-Perfect-DB proposal is not a v2
  composition.

V2 uses new schema and identity domains. The logical-ply target is content-
identified, so a twelve-ply record cannot collide with an eight-ply record.
The required final counts are twelve and `[6, 6]`.

The seven Book `ring16` orbits reported by the v1 audit describe endpoints at
logical ply eight. They are not assumed to remain seven after four additional
plies.

## Source terminology

An author-defined Book variation and a structural `ring16` orbit are different
objects:

- a variation is a named source line with author-provided metadata;
- a complete history is one legal, capture-resolved action sequence;
- an exact endpoint is the final FEN for one complete history; and
- a `ring16` orbit groups endpoints under the audited structural symmetry.

A variation may expand to several complete histories when its notation omits a
mandatory removal. Several variations may reach the same endpoint or orbit.
No one-to-one correspondence may be claimed without audit evidence.

Within the Book stratum, durable evidence must distinguish at least:

- `oracle_query_book`, the state-indexed candidate graph exposed by Sanmill's
  `query_book` operation; and
- `named_book_variation`, the author-defined `openings` lines in the pinned
  asset.

They may share an asset but do not have interchangeable coverage semantics.

## V2 record boundary

Every accepted v2 prefix record must contain:

- schema version `nmm.layered-opening-prefix.v2`;
- the explicit target `logical_ply_count=12`;
- `logical_plies_by_side=[6, 6]`;
- one of the three source strata and an explicit source subtype;
- portable source and Sanmill identities;
- the complete ordered action-token history;
- twelve contiguous logical-turn records;
- per-turn actor, input and output FEN, history SHA-256, and action tokens;
- a stable source-history identifier and source-specific selection evidence;
- final FEN and `ring16` canonical FEN; and
- a content identity covering the schema, target length, source evidence,
  action history, final state, and all step evidence.

The record must begin at `startpos` with `history_origin=game_start`. No logical
boundary may have a pending removal. The final FEN, action-token count, logical
counts, side counts, and history SHA-256 must be confirmed by Sanmill replay.
Unknown fields, malformed histories, mismatched identities, and source-mode
drift fail closed.

The source-specific evidence is not interchangeable:

- Book records name the source variation or Oracle path and record every
  omitted-capture expansion decision.
- HumanDB records name an observed complete-history group and its frequency
  evidence.
- Perfect DB records preserve every StrictSteps query, WDL/step evidence,
  tied-best pool, and deterministic choice.

Absolute paths are machine-local inputs. Durable records use path-registry
keys and content identities.

## Book audit contract

The audit must separately enumerate or select by an explicit pre-result rule
all Book histories that reach logical ply twelve without leaving their declared
Book subtype. It records complete history, final FEN, variation identifier,
`ring16` orbit, and exact-history/FEN/orbit multiplicity.

An `oracle_query_book` miss at logical plies nine through twelve aborts that
path. It cannot switch to a named line, Perfect DB, HumanDB, search, or a random
legal move. A named variation that cannot be replayed as a complete legal
twelve-ply history is reported as unavailable or ambiguous, rather than being
silently repaired.

If pure Book coverage is insufficient, the evidence must first distinguish:

- a corpus containing only pure Book paths that reach twelve plies; and
- a separately proposed `book_seeded_perfect_db_continuation` source.

The second source is mixed and remains prohibited until a new product decision
explicitly authorises it.

## HumanDB audit contract

HumanDB prefixes must be grouped from complete twelve-ply histories that
actually occur in source game records. Independent per-position or per-ply
frequency queries cannot be chained into a claimed human history.

For each exact history, the audit records:

- occurrence and distinct-game counts;
- first- and second-player colour information;
- game-result distribution;
- final FEN and `ring16` orbit; and
- exact-history, FEN, and orbit overlap with the other strata.

Claims are limited to "most common in the current PlayOK sample." The audit
must publish the observed frequency distribution. It cannot lower a threshold,
invent a route, or merge histories after seeing that exact twelve-ply branches
are sparse.

The active HumanDB is not an immutable audit input while a non-empty SQLite
sidecar exists. Its sidecars must not be deleted. The database-owner workflow
must close writers or use SQLite's online backup mechanism to produce an
isolated snapshot. The snapshot evidence records SHA-256, schema, table row
counts, and `quick_check`. Human frequencies and empirical outcomes are usable;
unversioned historical Malom columns are not labels.

## Perfect DB audit contract

Perfect DB lines use deterministic twelve-logical-ply StrictSteps queries. At
every ply:

- only candidates tied for theoretical best may be selected;
- `fallback` must be `none`;
- the full tied-best candidate set, WDL/StrictSteps evidence, and multiplicity
  are recorded; and
- source and history identities must remain stable.

The same requests must produce byte-identical results in two fresh Sanmill data
query processes. Perfect DB provides theory-controlled quality and structural
diversity. It is not evidence of human popularity.

## Maintainer opening delivery

The temporary `maintainer_inbox/Openings` delivery is an independent input
candidate and requires its own provenance manifest. Byte-identical copies of
tracked assets are not new evidence. The fifteen additions in the delivered
`learned_openings.json` remain an application-generated `seed_source=learned`
candidate pool. They are not HumanDB frequencies, the corrected Sanmill Book,
or members of the formal v2 corpus.

## Decision gate

The final 64-prefix composition remains unset until the source audits report:

- the number of pure Book variations with complete twelve-ply histories;
- the number and concentration of repeated HumanDB histories;
- the number of non-duplicate Perfect DB structures;
- exact-history, endpoint-FEN, and `ring16` overlap among all strata; and
- legality, complete-history, and colour-balance checks.

If HumanDB supplies stable high-frequency lines, its evidence may justify
reducing the Perfect DB allocation. If exact HumanDB histories are too diffuse,
the audit must present that evidence and a separately named alternative. It
must not manufacture synthetic human lines.

Until that gate is resolved, the state remains `needs_decision`.

## Current Book evidence

The source-only
[twelve-ply Book audit](../evidence/sanmill-layered-book-source-audit-2026-07-25.md)
is complete. The `oracle_query_book` graph supplies zero pure twelve-ply
histories: all 1,472 paths entering the final ply report `book_miss`. The
author-defined named lines are a separate representation; 84 of 107 variations
produce complete legal twelve-ply prefixes, expanding to 112 capture-resolved
records and 110 unique exact histories and `ring16` endpoints.

These results do not freeze Book membership or authorise a Book-seeded Perfect
DB continuation. HumanDB and Perfect DB evidence remain outstanding.

## Current HumanDB evidence

The source-only
[twelve-ply HumanDB audit](../evidence/sanmill-layered-human-source-audit-2026-07-25.md)
is complete. SQLite online backup produced a 738,091,008-byte immutable
snapshot with `quick_check=ok`; the active WAL and non-empty SHM were retained.

The recursive current PlayOK sample contains 95,389 source files. After legal
replay, short-game exclusion, one incomplete mandatory-capture exclusion, and
deduplication by PlayOK session identifier, 92,939 distinct games contribute
83,002 exact twelve-ply histories. Of those histories, 77,828 are singletons
and 5,174 have at least two-game support. The top 64 histories cover only 1,165
games, or 1.254% of the eligible distinct-game sample.

This measured sparsity does not select a threshold or a HumanDB allocation. It
does establish that a small "most frequent" list cannot be described as
representing most human openings. Perfect DB evidence and cross-source overlap
remain outstanding.

## Explicit exclusions

This preparation must not:

- load a candidate model;
- play candidate-versus-baseline games;
- freeze a 64-prefix list or source ratio;
- modify or relabel v1 eight-ply evidence;
- substitute another source after a Book miss;
- delete active HumanDB WAL or SHM files;
- treat learned-opening additions as human frequency evidence; or
- describe a source-only audit as playing-strength or promotion evidence.
