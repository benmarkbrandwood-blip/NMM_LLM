# Maintainer Book Opening Plays delivery — 26 July 2026

## Scope and provenance

The `main` maintainer and Mill-domain expert supplied
`Book Opening Plays.docx` on 26 July 2026. He described its contents as
opening plays that, in his opinion, most established players know. The
document is a new expert-curated source candidate; it is not the corrected
Sanmill opening-book asset, HumanDB frequency evidence, or Perfect DB output.

An exact retention copy is stored in the ignored archive:

```text
data/backups/maintainer_book_opening_plays_20260726/Book Opening Plays.docx
```

The inbox original was not deleted. Neither copy is an active training or
evaluation input.

## File identity and document review

| Field | Value |
| --- | --- |
| Byte length | 3,432,474 |
| SHA-256 | `227584cde9d8c6278665a1b6decac6491d6b30b9b7add44a4b00200aec5e83c7` |
| Non-empty table rows | 35 |
| Embedded board images | 35 |
| LibreOffice-rendered pages | 12 |

Post-copy size and SHA-256 checks match the inbox file exactly.

The document was rendered to PDF and page images with an isolated
LibreOffice profile. All twelve pages were inspected. Every non-empty source
row has one move-list cell and one board image, and no row or image is clipped
or displaced.

The tracked
[`maintainer-book-opening-plays-source-2026-07-26.json`](maintainer-book-opening-plays-source-2026-07-26.json)
preserves each table row's text, embedded-image SHA-256, explicit
normalisation notes, and twelve-ply token candidates. Its transcription
identity is
`ccb673a7dd52e7614adb2994a20531ca12359cede03adb6f89b1a11291c6b581`.

## Explicit interpretation boundary

The transcription does not silently discard source ambiguity:

- row 1 expressly gives `c5` as a trap continuation and `d1` as the better
  continuation, so both are retained as separate candidate histories;
- row 11 contains only eleven typed tokens. Its embedded screenshot shows
  Black's final `c5`; the transcription records that token as a visual
  interpretation which still requires expert confirmation before final
  corpus membership;
- rows 18/19 and 14/20 are retained as separate source rows even though their
  typed histories duplicate one another; and
- row 23 mentions additional children and row 29 mentions other losing Black
  options, but only their explicitly supplied continuations are transcribed.

The original DOCX remains the authority for what was delivered. The
normalised JSON is a reviewable audit input, not a replacement or corrected
edition of that document.

## Expert claim boundary

The maintainer associated knowledge of these plays with example ratings of
1,400, 1,800, or 2,100 depending on the site or application. He also reported
roughly 90% wins against players without a plan and, for one human-play
application, 60% wins and 20% draws. He clarified that those app games often
included interruptions from other people.

These statements are retained as expert and practical-use context. They are
not a controlled sample, calibrated Elo study, causal estimate of the
openings' effect, candidate-model result, or basis for source weighting. With
draws scored as one half, the reported 60% wins and 20% draws imply a 70%
score rate, but that arithmetic does not make the observation a benchmark.

## Use boundary

The delivery is an independent `book`-stratum candidate subtype. Before any
line can enter a frozen corpus it must pass:

- project-rules replay to twelve complete logical plies;
- authoritative replay in two fresh pinned Sanmill query processes;
- exact-history, final-FEN, and `ring16` deduplication;
- overlap comparison with the corrected Sanmill Book, genuine HumanDB, and
  fixed Perfect DB audit pool; and
- the remaining product and expert review gate.

No candidate model was loaded and no game was played while archiving or
transcribing the delivery.
