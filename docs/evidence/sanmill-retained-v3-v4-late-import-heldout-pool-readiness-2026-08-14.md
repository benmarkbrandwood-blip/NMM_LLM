# Retained-v3/v4 late-import held-out source-pool readiness

Status: `frozen_source_only_awaiting_precision_plan_and_authorization`

This is zero-candidate-game preparation for the next retained-v3/v4 decision.
It freezes a candidate-blind, training-disjoint source pool before either
candidate policy is loaded. It is not a held-out result, playing-strength or
equivalence evidence, a target-refresh causal comparison, or authority to run
an evaluation.

## Frozen identity

- artifact:
  `docs/experiments/sanmill-retained-v3-v4-late-import-heldout-pool-v1.json`;
- pool identity:
  `2eb04f542f88f8360f08f97e7657ca15646582a1532358dfeb04182ebad7d8f7`;
- ordered records identity:
  `4e5f9ecf7508a995b74af6a36bcf966c89d9141940770ebb21c3629446830a31`;
- artifact file SHA-256:
  `2cf7fb5506708b7c2e3b154bed8cbcfcc54b73899239475307f8840a7ac537da`;
- accepted starts: 361, with one source game and one unique `ring16` orbit
  per start;
- phase allocation: 153 placement, 152 movement and 56 flying.

The artifact contains complete logical-turn histories and strict-referee start
state. It deliberately omits player names and never uses the source winner or
result during selection.

## Why the sources are training-disjoint

The local import manifest contains 94,540 normalized PlayOK game IDs. The
active route HumanDB has 94,983 `processed_files` rows, but those rows normalize
to exactly 94,134 unique IDs, all contained in the import manifest. The set
difference is exactly 406 games. All 406 were imported in the narrow interval
`2026-07-20T10:41:27.255519` through
`2026-07-20T10:41:27.657548`, after the active HumanDB snapshot was built and
without rebuilding that database.

Both frozen retained route bundles bind the same active HumanDB identity
`8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31`.
They do not scan the raw `data/human_games` directory as an evaluation or
training position pool. The source set is also audited against the exact
candidate-owned SpecialistDB snapshots:

- retained-v3: `82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe`;
- retained-v4: `3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed`.

The late-import source-set identity is
`a3b710eea2435f82772f452e586f3ae5871e1f8f0ce3a0a14157aa97de0fbd71`.
The import-manifest file SHA-256 is
`90e1e12668b8f9e0ca93365ec499b1b328163f9d10961b89c741637e30327beb`.

## Fail-closed source and exposure audit

Every raw move was matched to exactly one legal complete NMM turn from the
recorded board-before-move state. Of 406 source games:

- 395 replayed completely and consistently;
- 11 contained an illegal or ambiguous raw move and were excluded;
- no source result or player metadata influenced acceptance;
- only board-before-move states at logical ply 12 or later were considered.

The 395 valid games supplied 12,371 candidate states. Exposure was checked
read-only under D4 against the active HumanDB and both SpecialistDB snapshots,
and under exact FEN plus the broader `ring16` orbit against eight previously
frozen experiment corpora. Rejection-hit counts are non-exclusive:

| Reason | State hits |
| --- | ---: |
| Active HumanDB D4 exposure | 5,651 |
| Retained-v3 SpecialistDB D4 exposure | 532 |
| Retained-v4 SpecialistDB D4 exposure | 539 |
| Prior corpus exact FEN | 7 |
| Prior corpus `ring16` orbit | 19 |

After these filters, 6,663 states from 361 independent source games remained.
The eligible-source phase coverage was 286 placement, 326 movement and 56
flying games. Selection reserved flying-capable games for flying, balanced the
remaining source games between placement and movement, used only frozen
SHA-256 ranks, and removed any cross-start `ring16` collision. No source was
lost to the final uniqueness rule.

## Strict current-referee replay

All 361 selected histories were replayed twice, each pass using a fresh pinned
Sanmill training-referee process per start. The 722 fresh-process observations
were byte-equal. Every selected start was strict nonterminal; zero starts were
excluded for an already-triggered threefold or fifty-move result.

Strict replay audit identity:
`36e607162607744a10bc40c70ed10f8a09067c3a314aaa0b16995004e4a2e14c`.

This is replay of existing human histories, not generation of a candidate
game. Candidate policies were never loaded and the candidate game count is
zero.

## Nested precision feasibility

The master order supports fixed prefixes without looking at any candidate
outcome:

| Conservative planning target | Starts | Candidate games | P / M / F | Source availability |
| ---: | ---: | ---: | ---: | --- |
| 3.0pp half-width | 64 | 256 | 22 / 21 / 21 | available |
| 2.0pp half-width | 142 | 568 | 48 / 47 / 47 | available |
| 1.5pp half-width | 253 | 1,012 | 99 / 98 / 56 | available |
| 1.0pp half-width | 568 | 2,272 | 153 / 152 / 56 available | **not available** |

Each future start would require four games: two candidates by two candidate
colours. The widths are conservative planning values derived from the larger
start-level score SD in the two completed visible corpora. They are not a
population variance guarantee. In particular, pool availability does not
select a decision framework or authorize the listed workload.

## Verification

- frozen pool validator: 361 histories replay locally with unique source and
  `ring16` identities;
- source-pool and phase-process web tests: 20 passed using a repository-local
  pytest base directory;
- expanded focused plus mandatory Malom, DB-teacher and label-provenance set:
  127 passed and 498 subtests passed;
- focused Ruff check: passed;
- direct second invocation validated the existing artifact and did not rebuild
  or overwrite it;
- the read-only web view now reports pool provenance, exposure support, strict
  replay support, phase mix and the infeasible 1pp row with help text.

The next product decision is still required: choose a fixed-width,
directional-minimum-effect or equivalence framework, then choose its bound.
Only after that choice may a subset, immutable evaluation plan, exact resource
envelope, readiness identity and separate product authorization be frozen.
