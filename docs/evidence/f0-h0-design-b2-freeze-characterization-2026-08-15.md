# F0-H0 Design B2 freeze and characterization evidence

Date: 2026-08-15

Status: `completed_freeze_and_nonfinal_characterization_screen_not_run`

## Verdict

The owner-selected Design B2 membership is frozen exactly as preregistered.
The expected counts reproduced without adjustment: 36,949 train, 887
selection, 386 confirmation, and 847 final-test sessions.  The selection,
confirmation, and final-test player sets contain 295, 160, and 322 keys and
have zero pairwise intersections.

This is not an F0-H0 screening verdict.  Independent support, modifiable-state
reach, concentration, and the product-effect upper bound were not computed.
No conclusion about feasibility, human behavior, model quality, or a later
gate follows from this record.

The binding objective addendum inherits the prior same-metric `ring16`
comparison: Design B measured 34.32%/53.60% decision-weighted overlap versus
37.55%/57.31% for the same-size random disjoint-game baseline.  State novelty
is therefore not an acceptance or contamination gate; game and player
membership own those boundaries.

Final-test remains sealed.  This round reports only its count and session-ID
identity.  It loaded zero final-test raw games, decisions, or derived
features.

## Frozen identities

| Artifact | Identity | File SHA-256 |
| --- | --- | --- |
| F0-D0 corpus | `4c54d55209543e70edaeb33cb1dea25d2707312c3781580ba326ae35882dea29` | manifest `0ab20955d551351ac25885b54d59a9f63fb6b2708e3292404d71dab2ff7dace6` |
| Freeze plan | `a4dc271d00a36394d4e5b61751f7536cf3e869cb90136fbe7bedd6016c6acb30` | `95aad0dc06fd46e026f09eaba59f3be9d4d3c5f520516a6f14f5b02e84b29bb3` |
| Official membership | `06c49903baf76ee7787af8333058e164cb54ea7a27035a1371747d6000d07b0b` | `06c3be92c87927d506dc36eb908aec3064220f4ead2ebb3b5ff3dfb7bf5032cb` |
| Characterization result | `183a39ab29ddfbec76a7188606b0a1297ffbdb845346a05753807f2c609b65e6` | `7ab7f68b29072e0de132525970b9cbcbbf68b58d07bda8ed36117e54c45da779` |

The four independently reproducible session-list identities are:

| Partition | Games | Session-list identity |
| --- | ---: | --- |
| train | 36,949 | `a73a87a731d1e09ad6552062cfb115fa3f7be424c0187710628da99373d3ca1b` |
| selection | 887 | `d34017d1d2325c60aff1b842c59fe9d39267364d499645e8b61a0ee0867e7715` |
| confirmation | 386 | `db4215ca75435a6f9bc13b24ad70a5d9f6e23639ac3e5ccefa839f9def53de45` |
| final-test | 847 | `2a7f6f19cee1ba97ceda6cd3070a96f70f0fef46414b67613e4b68bcad6ec6f8` |

All six session-set intersections are empty.  Of the 92,226 behavior games,
53,157 are outside the four selected definitions; their frozen session-list
identity is
`3a8ed5b7ac9f82a004b0037954c6e9f97b522deff02c04d0b731119a91e9befd`.
They were not reassigned to make the four partitions exhaustive.

## Method and ordering

Commit `ff76276` froze the plan, implementation hashes, access policy,
characterization definitions, cost sample, threshold, and fallback rule
before any official membership or benchmark result existed.  The membership
was then built solely from F0-D0 metadata.  It opened no raw game and stopped
unless the measured counts equalled 36,949/887/386/847 exactly.  Commit
`f30bc3c` fixed the resulting session lists before characterization or Malom
access.

The nonfinal pass verified every raw path, size, SHA-256, session ID, move
count, predecessor FEN, actor, turn, phase, legal move, and notation through
the existing strict replay.  It opened 38,222 train, selection, and
confirmation files and replayed 1,801,435 logical decisions.  It did not open
final-test.

## Nonfinal characterization

| Partition | Games | Players | Decisions | Placement | Movement | Flying | Strict outcomes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 36,949 | 2,216 | 1,742,416 | 658,641 | 1,025,077 | 58,698 | 15,135 |
| selection | 887 | 295 | 41,130 | 15,780 | 23,553 | 1,797 | 399 |
| confirmation | 386 | 160 | 17,889 | 6,864 | 10,463 | 562 | 160 |

Strict independently replayed outcomes remain on their separate F0-D0 basis:

| Partition | White win | Black win | Draw |
| --- | ---: | ---: | ---: |
| train | 3,214 | 3,786 | 8,135 |
| selection | 135 | 107 | 157 |
| confirmation | 33 | 40 | 87 |

Decision actor colors are 879,724 White and 862,692 Black in train; 20,786
White and 20,344 Black in selection; and 9,026 White and 8,863 Black in
confirmation.  Every source game contributes one White and one Black player
incidence.  The unique White/Black player-key counts are 1,892/1,763,
236/226, and 131/131 respectively.

| Partition | Date range | Length p50 | Length p95 | Maximum |
| --- | --- | ---: | ---: | ---: |
| train | 2025-12-11 to 2026-02-28 | 40 | 100 | 252 |
| selection | 2026-03-03 to 2026-03-31 | 39 | 100 | 199 |
| confirmation | 2026-04-03 to 2026-04-30 | 39 | 100 | 172 |

The machine manifest contains the complete nearest-rank player-game and game
length distributions, fixed length bins, color counts, and date bounds.
Final-test has no corresponding fields in that manifest.

## Final-test guard

`FrozenSplitAccess` is the supported content accessor for subsequent F0-H0
tools.  It checks a session ID before resolving raw content and raises
`FinalTestAccessError` for raw-game, decision, and derived-feature access.
Focused tests replace the underlying reader with a function that would fail
if called; the final-test exception occurs first.  A second test verifies that
neither decision loading nor a feature producer is invoked.  There is no
empty-data or neutral-value fallback.

This cannot prevent arbitrary new code from bypassing the repository API.
The binding experiment contract therefore also requires future F0-H0 tools to
use this accessor.  Bypassing it invalidates the final-test claim rather than
creating a valid result.

## Malom cost benchmark

The benchmark used 256 preregistered train+selection decision references,
identity
`b0663c48c97da72fc555dd65cab7ef375a1f16653b80e5294497af8d2c9035b9`.
It read 255 nonfinal raw games to reconstruct those exact states.  The only
accepted label metadata was `sector-corrected-v1`, with tracked dataset
content identity
`c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544`.
No oracle outcome or safe-set statistic was published.

| Pass | Queries | Legal actions | Seconds | Queries/s | Sector cache |
| --- | ---: | ---: | ---: | ---: | --- |
| cold first pass | 3,497 | 3,244 | 29.2633 | 119.50 | 0 to 79 |
| same-order warm repeat | 3,497 | 3,244 | 0.0800 | 43,709.99 | 79 to 79 |

The preregistered decision rule deliberately uses the upper 95% mean of
first-pass per-decision time, adds state construction, and multiplies by
1.25.  Under that fixed rule, full train+selection projects to 33,556,242
queries and 402,801.9 seconds, above the 7,200-second threshold.  Therefore
the frozen branch is `sample`: all 887 selection games plus 9,113 hash-ranked
train games.  Its 10,000-session identity is
`d43ee042514d9dea389849e943a5fb9d0f2d6218f6e226a980afc9354e9c8cd4`;
it contains 470,653 decisions and projects to 8,855,026 queries and
106,293.8 seconds under the same conservative rule.

The cold/warm ratio shows an important limitation.  The binding projection
charges one-time sector mapping latency as though it recurred across the
population.  A post-result, nonbinding cache-aware arithmetic check that
charges the 29.2633 seconds once and uses warm throughput yields about 4,462
seconds for full traversal and 1,199 seconds for the sample after the same
1.25 factor.  This check makes no new query and does not change the frozen
`sample` decision.  Switching back to full after seeing this result would
violate preregistration; any correction must be a new, separately frozen cost
contract before a screen.

## Claim and access boundaries

- Safety remains positional-only `A_pos`, never full-rule `A_allow`.
- Results are limited to the observed PlayOK-like source domain.  UI
  orientation, time control, exact source rules, and product population are
  not recoverable.
- The history filter is nonrandom: 1,751 excluded games contain 35 draws,
  while 92,789 retained games contain 26,157 draws.
- The 54,923 games without a verifiable terminal basis remain unverified.
- HumanDB was not read.  No database was written, rebuilt, or migrated.
- No game, search batch, model load, training update, or source-pool
  `2eb04f54` access occurred.
- The remaining 108 source-pool records are unchanged and unconsumed.

## Verification

The focused unit tests exercise deterministic B2 assignment, monotone cuts,
all three final-test denial paths, strict characterization bases, and the
frozen projection formula.  The evidence tests independently load and seal-
verify the plan, membership, and result, then assert every protected-access
counter and scope flag.

The exact commands and final results are recorded in the repository handoff.
