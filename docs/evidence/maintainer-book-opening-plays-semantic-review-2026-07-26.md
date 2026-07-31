# Maintainer Expert Book Semantic Review Disposition

Initial evidence date: 2026-07-26

Latest expert follow-up: 2026-07-31

Status: second expert response recorded; a concrete shortlist and the final
corpus remain `needs_decision`.

## Evidence boundary

The `main` maintainer and Mill-domain expert supplied the seven semantic
tables in `Book Opening Plays (2).docx` after reviewing the original parent
and child package. He later answered the four-point follow-up through direct
messages relayed by the product owner on 31 July. The exact document delivery
identity and visual inspection are recorded in the
[review delivery evidence](maintainer-book-opening-plays-review-delivery-2026-07-26.md).

The same document supplied one move-history correction. That correction is
handled separately by the
[reviewed-source audit](sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.md):
row 19 now uses Black `d5` at logical ply 10, while row 18 retains `d7`.

This document disposes only the semantic part of the expert response. It does
not turn a family name into a strength label, infer missing priorities, select
a final child, or freeze a corpus member.

The summaries below normalize spelling and punctuation for readability while
preserving the strategic meaning of the supplied tables. Source identifiers
remain the review-only `Pxx` aliases and original source-row numbers.

## Four-point disposition

| Requested review point | State | Evidence received | Still needed |
| --- | --- | --- | --- |
| 1. Name or classify each human opening family | Answered | P14 is `Interrupted Knight`; the follow-up also refines P03, P04, P05, P09, and P13 names | Nothing further for P14 |
| 2. Correct structural-group versus human-family relationships | Partial | P03 receives child-level plan labels; children 001/006 are identified as the same plan at the shared parent, and 003/004 are outer/inner Parallel Mill Rush variants | Review or correct the concrete grouping derived from those labels; do not infer resolution of conflicting older/newer wording |
| 3. Assign `core`, `useful`, or `optional/niche` coverage priority | Unanswered | The block labelled `3)` describes plans and one primary child but supplies no coverage tiers | Review a concrete breadth-first shortlist instead of repeating the abstract tier request |
| 4. Select a primary child and strategically distinct additions | Partial | P03 child 001 is the primary Black response; technical evidence makes child 006 a same-endpoint transposition and child 012 its exact-history duplicate | Primary/additional choices for P04, P05, P08, P09, and P13-A, plus correction of proposed P03 additions |

The response therefore materially advances the review but does not close the
Book-selection gate.

## Parent-family evidence

| Parent | Expert name or description | Expert strategic note |
| --- | --- | --- |
| P01 | Black pillar fail | Parent of a Black L plan |
| P02 | Delayed Black L variant | No additional note |
| P03 | Z Mill umbrella | The twelve-ply children include closed/open Z, perpendicular/parallel Mill Rush, wrap, cardinal, Black L, and Black double-Mill plans |
| P04 | Mill Rush; delayed cardinal horizontal Mill | The expert says it is the same Mill Rush if Black's third piece does not interrupt it; no primary child is selected |
| P05 | Mill Rush; delayed cardinal vertical Mill | The expert supplies the vertical counterpart; no primary child is selected |
| P06 | Black rush intended to avoid White's mill rush | Children determine whether Black reaches the endgame successfully or White can win |
| P07 | Black L | White prepares a mill wrap that forces Black to place at `c4` |
| P08 | Battle lines | Children distinguish a White win from continued Black initiative toward a draw or win |
| P09 | Yin-Yang / Corner Cardinals; earlier note called it Star formation | White has two moves to begin forcing Black; otherwise Black can force White's moves and win through scan-ahead play |
| P10 | Black into White | Black should place at `f6`; White is advised to develop a Mill on line `a` or line `1` using `a4`, `d1`, and `a1` |
| P11 | Black from White | Similar response to P10; Black generally avoids line `6` |
| P12 | T start / Knight moves / Grandfather move | The supplied note says this start can develop into P12, P13, or P14 |
| P13-A / P13-B | Knight Attack; D4-symmetry pair | Keep placing in L patterns until a Mill can be formed; no primary child is selected |
| P14 | Interrupted Knight | Standalone family name supplied in the 31 July follow-up |

The expert corrected an earlier rushed message: the symmetry statement applies
to P13-A and P13-B, not to P13 and P14. Independent D4 canonicalization agrees
that P13-A and P13-B have the same normalized eight-ply board, while the audit
continues to preserve their different exact histories.

## 31 July follow-up evidence

The expert's first direct-message response supplied `Interrupted Knight` for
P14, called P03 a `Z Mill` umbrella, attached a short plan label to fifteen of
the sixteen numbered P03 child panels, and described P04, P05, P08, P09, and
P13. The numbering in that message omitted P03 child 004 and contained an
ambiguous note for child 006. The expert then clarified:

```text
P03 006 is the same as 001 at 8 ply.
004 is a variant of parallel mill rush. 003 can be outer, 004 can be inner.
```

The same response explicitly calls P03 child 001 the primary Black response.
It does not assign `core`, `useful`, or `optional/niche` tiers. The descriptions
for P04, P05, P08, P09, and P13 do not select one audited source row as primary.
The P08 wording mentions a trap and a better response, but it does not map them
unambiguously onto source rows 24 and 25; no row assignment is inferred.

The latest explicit clarification controls the outer/inner labels for P03
children 003 and 004. Other differences from the earlier supplement remain
visible rather than being silently treated as corrections. In particular, the
new message calls child 007 an outer wrap where the supplement called it an
inner wrap, and calls child 008 open where the supplement called it closed.

## P03 extended-family evidence

All sixteen records below share one exact eight-ply parent. The expert
explicitly cautioned that they should not automatically remain one human
family merely because their first eight plies are the same.

| Child | Source row | Reviewed plies 9-12 | Expert semantic evidence, including follow-up |
| ---: | ---: | --- | --- |
| 1 | 3 | `b2 c5 c4 e5` | Closed Z Mill, inner Mill wrap; explicitly selected as the primary Black response |
| 2 | 8 | `e5 d5 e4 g4` | Perpendicular Mill Rush; the supplement says White should not place at `d2` |
| 3 | 9 | `e5 d5 e4 e3` | Parallel Mill Rush, explicitly clarified as the outer variant |
| 4 | 10 | `e5 e4 d5 d7` | Parallel Mill Rush, explicitly clarified as the inner variant |
| 5 | 11 | `e5 e4 d5 c5` | Perpendicular Mill Rush, described earlier as a cousin of P05; no `d2` |
| 6 | 14 | `c4 e5 b2 c5` | The expert identifies it with child 001 at the shared parent; the audit proves a same-endpoint transposition, described below |
| 7 | 15 | `c4 a7 a4 g7` | Open Z Mill with `c4` interruption and a wrap; latest outer/earlier inner wording remains unresolved |
| 8 | 16 | `b2 a7 a4 g7` | Outer Mill wrap; latest open/earlier closed wording remains unresolved |
| 9 | 17 | `c4 a7 e3 g7` | Open Z Mill with `c4/e3` and an outer wrap |
| 10 | 18 | `c4 d7 e3 d1` | Open Z Mill with `c4/e3`, outer cardinal variant |
| 11 | 19 | `c4 d5 e3 d1` | Open Z Mill with `c4/e3`, inner cardinal variant; retains the confirmed row-19 correction |
| 12 | 20 | `c4 e5 b2 c5` | Closed Z Mill, inner wrap; exact-history duplicate of child 006 |
| 13 | 21 | `c4 d5 e3 e4` | Open Z Mill with `c4/e3`, inner-corner variant |
| 14 | 22 | `c4 d5 d3 e3` | Open Z Mill with `c4/d3`, Black L response |
| 15 | 23 | `c4 g1 d3 a1` | Open Z Mill with `c4/d3`, Black double-Mill response |
| 16 | 28 | `c4 e4 d3 d5` | Open Z Mill, inner-cardinal response; earlier note says White must use `b2` or `c3` |

These annotations demonstrate real strategic substructure and are sufficient
to construct a concrete partition proposal. They are not an explicit approval
to merge every child sharing one word. The proposed grouping must therefore be
shown back to the expert as a correction sheet rather than reported as his
final partition.

### Objective transposition disposition

The expert's uncertainty about which twelve-ply continuation separated child
001 from child 006 can be resolved from the frozen audit without another
domain judgment:

| Child | Source row | Plies 9-12 | Exact-history relationship | Endpoint |
| ---: | ---: | --- | --- | --- |
| 001 | 3 | `b2 c5 c4 e5` | Distinct from 006 | Shared final FEN and `ring16` orbit |
| 006 | 14 | `c4 e5 b2 c5` | Transposition of 001; exact duplicate of 012 | Shared final FEN and `ring16` orbit |
| 012 | 20 | `c4 e5 b2 c5` | Exact duplicate of 006 | Shared final FEN and `ring16` orbit |

Children 001 and 006 use the same two White placements and the same two Black
placements in a different order. Their exact-history hashes differ, while
their final NMM FEN and canonical `ring16` FEN are byte-identical. Child 012 is
an exact-history duplicate of 006. The expert-selected child 001 can therefore
serve as primary, while 006 and 012 remain provenance but are same-plan
redundant candidates for a diversity-oriented corpus.

## Other child evidence

- P01's original source labels `c3 a4 c4 d1` as the better continuation and
  `c3 a4 c4 c5` as the trap continuation.
- P04 is now described as a delayed cardinal horizontal Mill Rush. The earlier
  supplement distinguished its two children, but the follow-up does not select
  row 4 or row 5 as primary.
- P05 is now described as a delayed cardinal vertical Mill Rush. The earlier
  supplement distinguished its two children, but the follow-up does not select
  row 6 or row 7 as primary.
- P08 is labelled Battle lines. The first supplied note says there are two
  correct responses and describes a trap for less experienced Black players,
  while the follow-up mentions `W d3 / B e4`, a trap, and `e4 / g4` as a
  better response. That wording does not identify row 24 or row 25 reliably,
  so no primary child is selected.
- P09 is now called Yin-Yang or Corner Cardinals. Its forcing-plan explanation
  still does not select row 26 or row 27.
- P13 is now called Knight Attack, with repeated L-pattern placements until a
  Mill can be formed. No primary/additional/redundant choice was supplied for
  P13-A rows 32 and 33.

The P04 and P05 descriptions support treating their two children as
strategically different. They do not state which child is primary.

## Supplemental examples outside the current twelve-ply candidates

The expert separately supplied two longer examples described as a P05 child
and its inversion. Each contains fourteen logical plies and does not share the
audited P05 exact eight-ply parent. The expert says White should play `d3`
after the first and `d1` after the inverted example; otherwise Black can take
that point for a winning-trajectory move.

These are useful future family-expansion candidates. They are not members of
the current 36-record twelve-ply source audit and must not be silently inserted
into it.

## Current decision boundary

The source evidence now establishes legal histories, the reviewed row-19
correction, structural identities, cross-source overlap, and partial human
semantics. It does not establish:

- expert approval of the concrete human-family partition proposed from the
  P03 child labels;
- `core`, `useful`, or `optional/niche` tiers;
- a primary and any additional distinct child for P04, P05, P08, P09, and
  P13-A;
- the allocation between corrected Sanmill Book and expert-curated Book; or
- acceptance of the provisional `22 Book / 21 HumanDB / 21 Perfect DB`
  composition.

These are the currently identified remaining Book-semantic and product
decisions. No candidate model, game, final 64-prefix freeze, or evaluation
authority follows from this review.
