# Maintainer Book Opening Plays review delivery — 26 July 2026

## Scope and provenance

After reviewing the expert-parent package, the `main` maintainer and
Mill-domain expert supplied `Book Opening Plays (2).docx`. This is a review
supplement to the earlier `Book Opening Plays.docx` delivery. It contains
semantic family notes, child-line annotations, and one subsequently confirmed
move-history correction.

An exact retention copy is stored beside the original in the ignored archive:

```text
data/backups/maintainer_book_opening_plays_20260726/Book Opening Plays (2).docx
```

The downloaded original was not deleted. Neither copy is an active training
or evaluation input.

## File identity and visual review

| Field | Value |
| --- | --- |
| Byte length | 3,434,996 |
| SHA-256 | `9ef34e0a984d63167a5db526e87e3849ec2752b05cf7a3ed27adfa932fcf9ad8` |
| Word tables | 8 |
| Embedded board images | 35 |
| Rendered pages | 15 |

The document was opened read-only in Microsoft Word, exported to PDF, rendered
to page images, and inspected on all fifteen pages. The installed LibreOffice
copy was not used for the final render because its headless launch reported a
local bootstrap error.

The first 35 non-empty move rows are unchanged from the original delivery.
All 35 embedded media names, byte lengths, and SHA-256 identities are also
unchanged. The new file adds seven review tables after the move-and-board
table. A stray `P01` label was also added in an otherwise empty row beneath
the move records; it is not a new move line.

## Confirmed P03 child correction

The review table says that P03 child 11 should have Black place at `d5`
instead of `d7`. The expert was asked whether this refers to source row 19 and
therefore to the following complete twelve-ply history:

```text
1.d6 d2
2.f4 b4
3.f6 f2
4.b6xf2 f2
5.c4 d5
6.e3 d1
```

He confirmed:

> Yes. Otherwise we have 2 games like that with black moving to d7.

The interpreted correction resolves uniquely and legally under the project
rules. It changes only source row 19. Source row 18 retains `d7`; the two rows
are no longer exact-history duplicates after the correction.

The original delivery, its transcription, and its source audit remain
immutable historical evidence. The correction is now applied through a
separately identified
[reviewed-source audit](sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.md);
it does not overwrite or relabel the original evidence.

## Semantic-review content

The added tables provide:

- provisional names or strategic descriptions for P01 through P12 and a
  P13-A/P13-B symmetry judgment;
- sixteen annotations for the continuations of the exact P03 eight-ply
  parent;
- two-child descriptions for P04 and P05;
- partial descriptions for P08 and P09; and
- an unnumbered `Knight moves` description associated with the later opening
  material.

The expert separately corrected an earlier rushed statement: P13-A and P13-B,
not P13 and P14, are the symmetry-related pair. Independent canonicalisation
confirms that P13-A and P13-B share one D4-normalised eight-ply board while
retaining different exact histories.

The review does not yet assign the requested `core`, `useful`, or
`optional/niche` coverage tier to every resulting family. It also does not
select a primary child for every multi-child exact parent. P14 has no
standalone row in the added parent table. These are remaining decision inputs,
not grounds for inferring an answer from layout or wording.

The extracted annotations, later chat clarifications, and four-point
completion state are recorded in the
[semantic review disposition](maintainer-book-opening-plays-semantic-review-2026-07-26.md).

## Use boundary

This delivery is expert semantic and correction evidence. It is not HumanDB
frequency evidence, Perfect DB value evidence, a final Book allocation, or
candidate-model evaluation evidence. No candidate model was loaded and no game
was played while preserving or reviewing it.
