# Twelve-Ply Book Source Audit

Date: 2026-07-25

Status: source-only evidence; final corpus remains `needs_decision`.

Machine-readable evidence:
[`sanmill-layered-book-source-audit-2026-07-25.json`](sanmill-layered-book-source-audit-2026-07-25.json)

## Scope

This audit evaluates two Book representations in the corrected Sanmill opening
asset without loading a candidate or playing a game:

- `oracle_query_book`: the state-indexed candidate graph exposed by
  `query_book`; and
- `named_book_variation`: the 107 author-defined `openings` lines.

No miss or incomplete line was continued through Perfect DB, HumanDB, search,
or a random action. `fallback=none` throughout.

The generator was NMM_LLM commit
`bcd925088670c824259ef055791eec8eb76cad46`. Two new Sanmill data-query
processes produced byte-identical payloads. The pinned Book identity is
`58101aa9b7f58f30a0489c0d85a991ba78e3147d94a87e5456af3f2167f58eaf`,
backed by asset SHA-256
`cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5`.

The JSON file is 1,386,833 bytes. Its file SHA-256 is
`31c01bf23a517961d90142b54951cd2cd95776ab02abe560d690ff0b1c430483`,
and its internal audit identity is
`b0d3bd526b7ab951c7dbd10552c77e8f6db5f1e69510829bdd642049bff72c48`.

## Oracle query graph

The exhaustive graph has no pure Book history that reaches twelve logical
plies. All 1,472 histories entering logical ply 12 return `book_miss`.

| Input logical ply | Input paths | Available | Book miss | Child paths |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 0 | 8 |
| 1 | 8 | 8 | 0 | 40 |
| 2 | 40 | 40 | 0 | 76 |
| 3 | 76 | 76 | 0 | 140 |
| 4 | 140 | 48 | 92 | 264 |
| 5 | 264 | 64 | 200 | 232 |
| 6 | 232 | 80 | 152 | 128 |
| 7 | 128 | 64 | 64 | 192 |
| 8 | 192 | 96 | 96 | 160 |
| 9 | 160 | 80 | 80 | 288 |
| 10 | 288 | 192 | 96 | 1,472 |
| 11 | 1,472 | 0 | 1,472 | 0 |

The 2,252 incomplete leaves are content-identified in the JSON evidence.
There were no terminal leaves and no fallback.

This means `oracle_query_book` currently contributes zero eligible pure
twelve-ply prefixes. It does not justify silently appending four Perfect DB
plies to an eight-ply Book path. Such a line would be a separately named
`book_seeded_perfect_db_continuation` source and remains unauthorised.

## Named variations

The named-line result is different because these lines are author-defined
trajectories rather than state-indexed Oracle continuations:

| Result | Variations |
|---|---:|
| Complete legal twelve-ply prefix | 84 |
| Source line shorter than twelve plies | 22 |
| Unreplayable before twelve plies | 1 |
| Total | 107 |

The one unreplayable variation is `novel-25964b79`; it has 18 source tokens but
has no legal continuation matching its token at logical ply 11. It was
reported, not repaired.

Omitted mandatory removals were expanded into complete legal histories before
Sanmill replay. The 84 complete variations therefore yield:

- 112 capture-resolved v2 prefix records;
- 110 unique exact action histories;
- 110 unique exact final FENs; and
- 110 unique final `ring16` orbits.

Two exact histories each occur under two variation identifiers:

- `book-40-658b81` and `book-41-827489`; and
- `book-19-5af166` and `book-47-502dd6`.

All other exact histories, final FENs, and final orbits have multiplicity one.
The JSON records every variation, source token, omitted-capture resolution,
complete action history, final FEN, final orbit, and v2 content identity.

For audit convenience only, each complete variation names the
lexicographically smallest capture-resolved action history as a deterministic
representative candidate. This does not freeze that representative into the
eventual 64-prefix corpus.

## Interpretation

The author-defined variation and `ring16` orbit counts happen to be highly
diverse at twelve plies, but they are not synonymous. One variation can expand
to several histories and orbits, and two variation identifiers can describe
the same exact history.

The old eight-ply result of seven final Book orbits remains valid historical
evidence for the Oracle-query corpus at its old endpoint. It is neither
relabeled nor extrapolated to twelve plies.

The Book stratum now has enough source evidence for the later composition
decision, but that decision still requires the genuine HumanDB frequency and
Perfect DB diversity audits.
