# Twelve-Ply Expert Book Reviewed-Source Audit

Date: 2026-07-26

Status: source-only reviewed evidence; final corpus remains
`needs_decision`.

Machine-readable evidence:
[`sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json`](sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.json)

## Scope

This audit applies the one move-history correction that the `main` maintainer
and Mill-domain expert explicitly confirmed after delivering
`Book Opening Plays (2).docx`. It changes source row 19, logical ply 10 for
Black, from `d7` to `d5`.

The original document, transcription, audit, and review-image package remain
immutable historical evidence. This reviewed-source audit is a new lineage; it
does not overwrite or relabel them.

No other move token, label, normalization note, or source row changed. The
reviewed-source loader compares the complete v2 transcription with the
original v1 transcription and fails closed if it finds an undeclared change.

No candidate model was loaded, no game was played, and no fallback source was
available.

## Identities

| Evidence | Identity |
| --- | --- |
| Review DOCX | 3,434,996 bytes; SHA-256 `9ef34e0a984d63167a5db526e87e3849ec2752b05cf7a3ed27adfa932fcf9ad8` |
| Reviewed transcription | 27,402 bytes; SHA-256 `e4c9ed58e167091d4ccf2cb37c993b5143aac6afffd41793212359cebb9d084b` |
| Reviewed transcription identity | `03fc4195ab9cb68ac275ae70e7a680401a56257bfc4c98883f53aad931907e74` |
| Portable reviewed-source identity | `6e33d7af134ca88cc161a947590457c9f7af2d3a8a285f7d8a581ca2a0350142` |
| Original transcription identity | `de3f7a33e772501f8cd369fbfb540ab8d7dabf259c15c51087acf2b425436273` |
| NMM_LLM generator commit | `927c2a60f6af6f5ea5a3ba0badae02fa46f92d5e` |
| Pinned Sanmill interface commit | `db65eb3e73189d934d615d0f47519d395193c646` |

The canonical audit is 527,112 bytes, has file SHA-256
`5b5255dc384a065938aed6e39194a2bba2da398623f5d991d58a8522d3b803c5`,
and internal audit identity
`1f6f9ceb8df36150ea401145e16c88cc25550622c1ad85a1b54a067183b9978d`.

Two newly started pinned Sanmill data-query processes produced byte-identical
canonical audit bytes. All 36 variations replayed legally to twelve complete
logical plies and final side counts `[6, 6]`, with no pending removal.

## Effect of the confirmed correction

| Measure | Original audit | Reviewed audit | Change |
| --- | ---: | ---: | ---: |
| Source rows | 35 | 35 | 0 |
| Variation records | 36 | 36 | 0 |
| Unique exact histories | 34 | 35 | +1 |
| Unique final FENs | 33 | 34 | +1 |
| Unique final `ring16` orbits | 32 | 33 | +1 |
| Eight-ply exact parents | 15 | 15 | 0 |
| Eight-ply parent `ring16` orbits | 14 | 14 | 0 |

Rows 18 and 19 are no longer duplicates. Row 18 retains `d7`; row 19 now
contains:

```text
1.d6 d2
2.f4 b4
3.f6 f2
4.b6xf2 f2
5.c4 d5
6.e3 d1
```

Rows 14 and 20 remain the only exact-history duplicate pair. Rows 3, 14, and
20 still share one final FEN. Rows 3, 14, 16, and 20 still share one final
`ring16` orbit.

Because the corrected token occurs after logical ply eight, the parent
taxonomy and its 16-child P03 group are unchanged. Only the row-19 child,
endpoint, and downstream overlap evidence change.

## Cross-source overlap

Counts report raw reviewed-source records first and unique values second.

| Prior source | Exact history | Final FEN | `ring16` orbit |
| --- | ---: | ---: | ---: |
| Corrected Sanmill Book | 0 / 0 unique | 11 / 9 unique | 12 / 9 unique |
| Genuine HumanDB | 6 / 6 unique | 14 / 12 unique | 21 / 18 unique |
| Fixed Perfect DB audit pool | 0 / 0 unique | 0 / 0 unique | 0 / 0 unique |

The correction does not add an exact HumanDB occurrence. The same six expert
histories remain supported by 29 distinct games in the frozen current PlayOK
sample, with maximum support of nine games for one history.

The new row-19 endpoint increases structural overlap with the corrected
Sanmill Book and HumanDB, reinforcing the requirement to deduplicate final
FENs and `ring16` orbits across selected strata. It does not establish human
popularity or opening quality.

The Perfect DB result remains limited to the fixed 128-route audit pool. It is
not a claim about every possible theoretical route.

## Downstream consequence

All future expert-Book selection, overlap calculation, and regenerated review
assets must use this reviewed-source audit rather than the original audit.
The old review package remains useful evidence of what the expert saw before
confirming the correction, but its row-19 child panel is not a current
selection input.

The review supplement also contains partial family and child annotations.
Those annotations are semantic evidence, not part of this move-history audit.
They do not complete the requested parent-priority tiers, all primary-child
choices, or a standalone P14 classification. Those remaining judgments must
stay explicit rather than being inferred.

The overall source ratio, allocation between the two Book subtypes, and exact
64-prefix list remain unfrozen. This audit does not authorize candidate
evaluation, training, promotion, or release.
