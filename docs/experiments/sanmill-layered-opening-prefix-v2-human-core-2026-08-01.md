# Twelve-ply layered-prefix HumanDB core

Status: `human_membership_frozen_perfect_pending`

Decision date: 2026-08-01

The accepted 21-place HumanDB stratum now has deterministic source-only
membership. The machine-readable record is
[stored beside this document](sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json)
with membership identity
`873cc67431205ab579be16dab1c5bdb6f3cccb25d2d39d56039327c0d02e5ebf`.

These are complete histories that occurred in the frozen current PlayOK
sample. They are not per-ply synthetic chains and are not claimed to represent
most human openings or all human players.

## Selection rule and result

The immutable ledger is already ordered by:

1. distinct-game count, descending;
2. occurrence count, descending; and
3. exact history identity, ascending.

Selection scans that order after the
[22-member Book core](sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json).
A history is skipped if its exact action sequence, final FEN, or D4/`ring16`
orbit has already been selected. Book has precedence over HumanDB, and an
earlier HumanDB member has precedence over a later structurally duplicate
history.

The 21 places are filled at ledger rank 31. Ten earlier histories are skipped;
all ten collide at `ring16`, five with Book and five with an earlier HumanDB
member. No exact-history or exact-FEN collision remains. The selected histories
have support from 16 to 27 distinct games.

| Core ID | Ledger rank | Distinct games | Occurrences | History identity prefix |
| --- | ---: | ---: | ---: | --- |
| `human-core-001` | 4 | 27 | 27 | `f18206eb6715` |
| `human-core-002` | 5 | 26 | 26 | `12206e28aa4d` |
| `human-core-003` | 6 | 25 | 26 | `976d0af49134` |
| `human-core-004` | 7 | 24 | 24 | `96211ae8b180` |
| `human-core-005` | 8 | 23 | 24 | `3ab9bcd99a5f` |
| `human-core-006` | 10 | 22 | 23 | `7770bf936291` |
| `human-core-007` | 11 | 22 | 23 | `aa850edb6264` |
| `human-core-008` | 13 | 21 | 23 | `901b31f8cda2` |
| `human-core-009` | 16 | 20 | 20 | `b074f7858192` |
| `human-core-010` | 17 | 19 | 20 | `09d2e94d49ee` |
| `human-core-011` | 18 | 19 | 19 | `784f27ad1b38` |
| `human-core-012` | 19 | 18 | 19 | `68c7fa779614` |
| `human-core-013` | 20 | 18 | 18 | `1316c55f7627` |
| `human-core-014` | 21 | 18 | 18 | `850d808c9131` |
| `human-core-015` | 23 | 18 | 18 | `b7fcf008a348` |
| `human-core-016` | 24 | 18 | 18 | `cd1aa5b424b9` |
| `human-core-017` | 25 | 17 | 18 | `46058c5b3f20` |
| `human-core-018` | 27 | 17 | 17 | `aa5bed1448c5` |
| `human-core-019` | 28 | 17 | 17 | `cfd412b49a3a` |
| `human-core-020` | 30 | 16 | 19 | `5f7559782f6a` |
| `human-core-021` | 31 | 16 | 17 | `4c7a09c199c4` |

The JSON preserves the complete 31-record selection window, including all ten
skips, so the membership can be rederived without the ignored local ledger.
It also binds the full ledger's SHA-256, byte length, schema, row count, and
path-registry key.

## Execution boundary

This decision freezes source membership, not an executable evaluation corpus.
The selected HumanDB histories still require complete per-step replay through
the pinned strict Sanmill bridge before their v2 prefix records can be frozen.
No random or alternate-source repair is permitted if replay fails.

No candidate was loaded and no game was played. Perfect DB membership, the
combined 64-member identity, review images, evaluation, and training remain
unfrozen and unauthorised.
