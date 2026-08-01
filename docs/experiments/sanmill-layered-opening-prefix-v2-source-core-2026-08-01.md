# Twelve-ply layered-prefix source core

Status: `source_membership_frozen_execution_replay_pending`

Decision date: 2026-08-01

The three independently frozen strata are combined in one immutable
source-membership manifest:

| Stratum | Members | Membership identity |
| --- | ---: | --- |
| Book | 22 | `27563913709e56646f0646d74e6b4cd15c94b458250bd1c9dbb741e32239fa98` |
| HumanDB | 21 | `873cc67431205ab579be16dab1c5bdb6f3cccb25d2d39d56039327c0d02e5ebf` |
| Perfect DB | 21 | `94ffea816df83b94e91567abaf760418ad711220260902b8a71786620f2e2d36` |
| **Total** | **64** | source core below |

The machine-readable manifest is
[stored beside this document](sanmill-layered-opening-prefix-v2-source-core-2026-08-01.json).
Its source-membership identity is
`ed09e3952fa790b3cb044bd7a8fc06c9429d6675ece726abeef29d1c0ddf1608`.

It is bound to the accepted
[composition](sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.json)
and the separate
[Book](sanmill-layered-opening-prefix-v2-book-core-2026-08-01.json),
[HumanDB](sanmill-layered-opening-prefix-v2-human-core-2026-08-01.json), and
[Perfect DB](sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.json)
decisions.

## Invariants

Every member:

- begins from game start and represents exactly twelve complete logical plies;
- has final per-side logical counts `[6, 6]`;
- ends with White to move and no source-level pending-removal boundary; and
- remains identified by its source stratum and original member identity.

Across the complete source core there are 64 unique action histories, 64
unique final FENs, and 64 unique D4/`ring16` structures. Book is listed first,
then HumanDB, then Perfect DB. This ordering is presentation and identity
policy; later reports still separate the three strata.

## Execution boundary

Forty-three records—the 22 Book and 21 Perfect DB members—already reference
complete frozen v2 source-prefix records. The 21 HumanDB source members still
need complete per-step replay with the pinned strict Sanmill bridge. Therefore:

- source membership is frozen;
- the executable 64-prefix corpus is not frozen;
- no replay failure may be repaired with random, search, Book, HumanDB, or
  Perfect DB fallback; and
- no evaluation, training, publication, or promotion is authorised.

The deterministic
[review package](sanmill-layered-opening-prefix-v2-source-core-review-2026-08-01.md)
renders all 64 frozen final states as individual panels and six contact sheets.
Its manifest identity is
`db37224db6e400a32df9275e5e0665647541c4aa589b327b4317235e2eb27fba`.
Those images are membership-review material only, not replay or rules
acceptance evidence.
