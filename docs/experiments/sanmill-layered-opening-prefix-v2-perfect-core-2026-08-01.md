# Twelve-ply layered-prefix Perfect DB core

Status: `all_source_membership_execution_and_corpus_frozen`

Decision date: 2026-08-01

The accepted 21-place Perfect DB stratum now has deterministic source-only
membership. The machine-readable record is
[stored beside this document](sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.json)
with membership identity
`94ffea816df83b94e91567abaf760418ad711220260902b8a71786620f2e2d36`.

The members are audit routes 000 through 020, with seeds 42 through 62. They
are taken in the frozen 128-route pool order after checking exact-history,
final-FEN, and D4/`ring16` duplication against the
[Book core](sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json)
and
[HumanDB core](sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json).
No route is skipped, consistent with the source audit's zero-overlap result.

## Frozen members

| Core ID | Audit route | Seed | History identity prefix | Tied/single-best steps |
| --- | --- | ---: | --- | ---: |
| `perfect-core-001` | `perfect-audit-route-000` | 42 | `f043572c2bf1` | 12 / 0 |
| `perfect-core-002` | `perfect-audit-route-001` | 43 | `3e20e69b18e8` | 11 / 1 |
| `perfect-core-003` | `perfect-audit-route-002` | 44 | `bcb0a404ab26` | 11 / 1 |
| `perfect-core-004` | `perfect-audit-route-003` | 45 | `7d5f956833dc` | 12 / 0 |
| `perfect-core-005` | `perfect-audit-route-004` | 46 | `32e8a16017fd` | 12 / 0 |
| `perfect-core-006` | `perfect-audit-route-005` | 47 | `3822de65f206` | 11 / 1 |
| `perfect-core-007` | `perfect-audit-route-006` | 48 | `35e1eb21429b` | 12 / 0 |
| `perfect-core-008` | `perfect-audit-route-007` | 49 | `ee5091cadcfd` | 12 / 0 |
| `perfect-core-009` | `perfect-audit-route-008` | 50 | `58d63263484b` | 10 / 2 |
| `perfect-core-010` | `perfect-audit-route-009` | 51 | `7420a819e131` | 12 / 0 |
| `perfect-core-011` | `perfect-audit-route-010` | 52 | `6e8ecc38af11` | 12 / 0 |
| `perfect-core-012` | `perfect-audit-route-011` | 53 | `e43940392fa9` | 11 / 1 |
| `perfect-core-013` | `perfect-audit-route-012` | 54 | `ba5347659710` | 11 / 1 |
| `perfect-core-014` | `perfect-audit-route-013` | 55 | `35034001b333` | 12 / 0 |
| `perfect-core-015` | `perfect-audit-route-014` | 56 | `e22e50cd1860` | 12 / 0 |
| `perfect-core-016` | `perfect-audit-route-015` | 57 | `c205b9994624` | 11 / 1 |
| `perfect-core-017` | `perfect-audit-route-016` | 58 | `5a1a79c61f1b` | 12 / 0 |
| `perfect-core-018` | `perfect-audit-route-017` | 59 | `74b211fdb77f` | 10 / 2 |
| `perfect-core-019` | `perfect-audit-route-018` | 60 | `70cfe0661000` | 12 / 0 |
| `perfect-core-020` | `perfect-audit-route-019` | 61 | `f7fe7674ed81` | 12 / 0 |
| `perfect-core-021` | `perfect-audit-route-020` | 62 | `d56e0ff55985` | 10 / 2 |

Across 252 selected logical plies, 240 have multiple tied-best candidates and
12 have one best candidate. Every selected candidate has database WDL 0 under
the frozen Standard StrictSteps query contract. This is theory-controlled
source evidence, not human-frequency evidence and not an observed evaluation
result.

Each member binds the complete source prefix identity and a canonical identity
for all twelve StrictSteps selection traces. The generator does not query or
regenerate Perfect DB routes; it selects only from the already frozen audit.

## Remaining boundary

All 64 source memberships are now frozen and structurally disjoint. A later
[HumanDB execution overlay](sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.md)
freezes complete strict-Sanmill records for the remaining 21 histories. The
combined source manifest is frozen in the
[source core decision](sanmill-layered-opening-prefix-v2-source-core-2026-08-01.md).
A subsequent
[executable-corpus decision](sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.md)
assembles and identity-freezes all 64 records. Neither source evidence nor
review images constitute an evaluation launch.

No candidate was loaded and no game was played. Evaluation and training remain
unauthorised.
