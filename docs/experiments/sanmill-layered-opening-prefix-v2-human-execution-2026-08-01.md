# Twelve-ply layered-prefix HumanDB execution records

Status: `human_execution_frozen_executable_corpus_pending`

Decision date: 2026-08-01

The 21 frozen HumanDB source members now have complete executable v2 prefix
records. The machine-readable evidence is
[stored beside this document](sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.json)
with execution identity
`1cf88ab8b3afb7c62112a0f2866eed9052587bbf2ef44dc57efa64c2749021d6`.

This overlay does not alter the accepted
[HumanDB membership](sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json)
or the combined
[source-core membership](sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json).
It supplies the strict per-step Sanmill evidence that was deliberately absent
from those source-only decisions.

## Replay contract

Every complete observed PlayOK history was replayed from `startpos` through
the separately pinned
[HumanDB replay runtime](sanmill-prefix12-human-replay-runtime-2026-08-01.json).
That contract requires Sanmill commit
`db65eb3e73189d934d615d0f47519d395193c646`, its exact tree and clean HEAD,
the frozen release-binary identity, strict protocol v1, and fail-closed
handling. No Book, HumanDB, Perfect DB, search, random, or model fallback can
repair a rejected history.

The complete operation was repeated in two fresh Sanmill data-query
processes. Their ordered request/response transcripts were exactly equal.
Each process handled 273 requests and 273 responses, for 546 transcript lines.
The canonical transcript identity is
`e61bef7940fb1dd9a6fffb67b98640825d72a0ebcfb105627fdaa871173c13fd`.

## Result

| Measure | Result |
| --- | ---: |
| Frozen HumanDB records | 21 |
| Complete logical plies | 252 |
| Compound Mill-and-removal turns | 13 |
| Action tokens per history | 12-13 |
| Unique prefix identities | 21 |
| Unique final Sanmill history identities | 21 |
| Fresh replay processes | 2 |

For every record, all twelve logical-turn boundaries are retained with their
input and output FENs, action tokens, history identity, side counts, and source
evidence. The final NMM FEN and D4/`ring16` canonical FEN exactly match the
frozen HumanDB and source-core membership records.

## Evidence boundary

This artifact freezes executable records for the HumanDB stratum only. The 22
Book and 21 Perfect DB records already exist in their frozen source audits,
but the combined executable 64-prefix corpus has not yet been assembled or
given its own identity. This replay loaded no candidate model, played no game,
and authorises neither evaluation nor training.
