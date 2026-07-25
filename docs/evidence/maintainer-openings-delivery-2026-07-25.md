# Maintainer Openings delivery — 25 July 2026

## Scope and provenance

This record inventories the `../maintainer_inbox/Openings` bundle received
from the `main` maintainer on 25 July 2026. The bundle was described only as an
Openings delivery; no claim of HumanDB frequency, corrected Sanmill Book
membership, or Perfect DB optimality accompanied it.

An exact retention copy is stored in the ignored archive:

```text
data/backups/maintainer_openings_20260725
```

The inbox originals were not deleted. Neither the inbox nor the archive is an
active training or evaluation input.

## File inventory

| File | Bytes | SHA-256 | Disposition |
| --- | ---: | --- | --- |
| `book_openings.json` | 15,883 | `48d774e401861d0acdc58574255bcc4ba6ad5817459240d70e18a5b8a0c8edd7` | Byte-identical to tracked `data/openings/book_openings.json`; not new evidence |
| `openings.json` | 15,883 | `48d774e401861d0acdc58574255bcc4ba6ad5817459240d70e18a5b8a0c8edd7` | Byte-identical to tracked `data/openings/openings.json`; not new evidence |
| `learned_openings.json` | 213,307 | `5227164d8a987c45070561cc5e7116011d190d86da755ce63e8fd1fc436e1374` | Independent learned-opening candidate pool; not merged |

Post-copy size and SHA-256 checks match the inbox files exactly.

## Learned-opening comparison

The tracked `data/openings/learned_openings.json` has 169 records, is 195,639
bytes, and has SHA-256
`e348cbd442bb221588bc96dd7ef0500ab8ca31aa1306b84cfec422d7d4ef1c8e`.
All 169 records are present and content-identical in the delivery. The
delivered file adds these 15 unique records:

```text
novel-073abf01
novel-1cc2201b
novel-1d507e0b
novel-3d9346d7
novel-47dbf374
novel-48729ce2
novel-5a8cfc2d
novel-5fec41ed
novel-6b4ed945
novel-76b83a7e
novel-a484ed88
novel-bd4ef2a3
novel-ce209172
novel-e8c1597f
novel-f376cf06
```

Every addition:

- contains an 18-ply `line_moves` sequence;
- declares `seed_source=learned`;
- has confidence `0.3`;
- has an empty `source_reference`;
- remains marked as needing an LLM-generated name; and
- declares the favoured side as unknown.

These fields describe application-generated candidates, not independently
observed human frequencies or theory labels.

## Use boundary

The fifteen additions are retained as a separate candidate pool. They are not:

- HumanDB histories or PlayOK frequency evidence;
- members of the corrected 109-position Sanmill opening-book asset;
- StrictSteps Perfect DB routes;
- part of the formal twelve-ply corpus; or
- evidence of candidate-model strength.

Using them later would require a new, explicitly named source contract and
legal replay/provenance audit. They must not be silently merged into any of the
current Book, HumanDB, or Perfect DB strata.
