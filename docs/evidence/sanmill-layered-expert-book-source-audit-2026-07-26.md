# Twelve-Ply Expert Book Play Source Audit

Date: 2026-07-26

Status: source-only evidence; final corpus remains `needs_decision`.

Machine-readable evidence:
[`sanmill-layered-expert-book-source-audit-2026-07-26.json`](sanmill-layered-expert-book-source-audit-2026-07-26.json)

## Scope and identities

This audit covers the 35-row `Book Opening Plays.docx` delivery from the
`main` maintainer and Mill-domain expert. It treats the material as a distinct
Book subtype, `maintainer_expert_curated_play`; it does not relabel it as the
corrected Sanmill opening-book asset, HumanDB, or Perfect DB.

| Evidence | Identity |
| --- | --- |
| Source DOCX | 3,432,474 bytes; SHA-256 `227584cde9d8c6278665a1b6decac6491d6b30b9b7add44a4b00200aec5e83c7` |
| Tracked transcription | SHA-256 `16c83726eaa872db8ccc8195c12153215e74b0b3e14acbb4e187e7e2195a48d2`; internal identity `de3f7a33e772501f8cd369fbfb540ab8d7dabf259c15c51087acf2b425436273` |
| Portable source identity | `9c0c3ae58aaec67cb752899121cec465d19dd2bf332a869fddb0216e36a1463f` |
| NMM_LLM generator commit | `271209481196cb47ca0382ca0cff7f5d3a56d63b` |
| Pinned Sanmill interface commit | `db65eb3e73189d934d615d0f47519d395193c646` |
| Sanmill binary | 4,109,312 bytes; SHA-256 `cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc` |

The inspected Sanmill checkout was descendant
`aa6b0c99ee3fca13b0d34e6f929257959ed51414`. The installation verifier
confirmed that the pinned CLI, rules, build, bridge-document, and
opening-book scope had not changed.

The canonical audit is 521,638 bytes, has file SHA-256
`e6bdb4d66f868fde6cdf1928b25174d323da533c03f074cf1c717b428bbdf107`,
and internal audit identity
`29293523ba211775827bfd903ef8dd0a0220bcd958a46974e37809a55dd51ff4`.

No candidate model was loaded, no game was played, and no source fallback was
available.

## Replay and fresh-process result

All 35 source rows and 35 embedded image identities were retained. Row 1
contains two explicit final continuations, so the audit has 36 source
variation records. Row 11's final Black `c5` remains expressly marked as a
move read from its embedded screenshot rather than from the typed table cell.
At original resolution the move panel unambiguously shows `6. d5 c5`, White
to move, and six placed pieces per side, so no further expert confirmation is
required for that transcription.

Every variation:

- resolved to exactly one legal project-rules history;
- contained twelve complete logical plies with final side counts `[6, 6]`;
- preserved mandatory removal inside the corresponding logical ply;
- replayed through pinned Sanmill with matching intermediate history counts,
  final FEN, and eight-ply parent state; and
- ended without a pending removal.

Two newly started Sanmill data-query processes produced byte-identical
canonical audit bytes.

## Twelve-ply structure

| Measure | Raw records | Unique values |
| --- | ---: | ---: |
| Exact action history | 36 | 34 |
| Final NMM FEN | 36 | 33 |
| Final `ring16` orbit | 36 | 32 |

Rows 14/20 and 18/19 are exact-history duplicates. Rows 3, 14, and 20 share a
final FEN. At the `ring16` level rows 3, 14, 16, and 20 share one orbit, while
rows 18/19 share another. Source rows remain present as provenance, but these
duplicates cannot count as independent structures in a later corpus.

## Eight-ply parents and twelve-ply children

The audit also truncates each complete source history after eight logical
plies without modifying the immutable eight-ply v1 corpus:

| Parent measure | Unique values |
| --- | ---: |
| Exact action history | 15 |
| Exact NMM FEN | 15 |
| `ring16` orbit | 14 |

The 14 parent-orbit group sizes are:

```text
16, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1
```

The largest group contains rows 3, 8–11, 14–23, and 28: 16 of the 36
variation records share the same eight-ply parent and branch into different
twelve-ply children. This is useful repertoire detail, but it also confirms
the expert's concern that a diversity-oriented corpus may need more different
parents before adding several children from the same parent.

The grouping is structural evidence, not a claim that `ring16` orbits and
human opening-family names are identical.

## Cross-source overlap

Counts below report both raw source records and unique values so duplicated
expert rows cannot inflate overlap.

| Prior source | Exact history | Final FEN | `ring16` orbit |
| --- | ---: | ---: | ---: |
| Corrected Sanmill Book | 0 / 0 unique | 10 / 8 unique | 11 / 8 unique |
| Genuine HumanDB | 6 / 6 unique | 14 / 12 unique | 20 / 17 unique |
| Fixed Perfect DB audit pool | 0 / 0 unique | 0 / 0 unique | 0 / 0 unique |

The lack of exact-history overlap with the Sanmill Book means these are new
trajectories relative to its complete named lines. The endpoint overlap shows
that some trajectories still converge to already represented structures, so
the two Book subtypes cannot simply be concatenated.

The Perfect DB result is limited to the already frozen 128-route source-audit
pool; it is not a claim of no possible overlap with every theoretical line.

## Exact HumanDB support

Six expert histories occur exactly in the complete frozen 83,002-history
HumanDB ledger:

| Source row | Distinct games | White wins | Draws | Black wins |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 4 | 2 | 0 | 2 |
| 5 | 4 | 0 | 2 | 2 |
| 7 | 2 | 2 | 0 | 0 |
| 8 | 9 | 5 | 1 | 3 |
| 9 | 2 | 1 | 1 | 0 |
| 16 | 8 | 1 | 0 | 7 |

Together they have 29 distinct-game observations; the highest individual
support is nine games. This independently shows that some delivered routes
occur in the current PlayOK sample. It does not establish that all routes are
common, that the listed outcomes were caused by the opening, or that a
particular colour is favoured. The HumanDB result distribution must not be
turned into a quality label.

The maintainer's separate 60%-win/20%-draw observation came from uncontrolled
personal app play, often interrupted by other people. It remains practical
expert context rather than part of this measured ledger result.

## Decision consequence

The delivery adds a useful expert-curated Book candidate pool, but it does not
justify increasing the Book quota or freezing any member automatically.
A later Book selection rule should:

1. consider eight-ply parent coverage before taking multiple children;
2. preserve meaningful expert alternatives only after the parent coverage
   objective is met;
3. cover the corrected Sanmill Book's declared families separately;
4. remove exact-FEN and `ring16` duplicates across all three final strata; and
5. retain row 11's visual rather than typed evidence basis in any later
   provenance record.

The remaining Mill-domain review is defined in the
[expert parent-and-child review guide](../experiments/sanmill-layered-expert-book-parent-review.md).
It requests four explicit judgments:

1. name or describe the human opening family represented by each of the 14
   structural parent groups;
2. identify strategic family relationships that require combining different
   structural groups or splitting one normalised group;
3. classify the resulting parents as core, useful, or optional/niche for
   coverage; and
4. for each exact parent with multiple continuations, choose one primary
   twelve-ply child and identify any additional children whose plies 9–12
   express a genuinely different human strategy.

The fourth judgment is not a request to retain every different move sequence.
An additional child needs a distinct Mill, block, fork, chain, trap, escape,
regional-development, or opponent-response plan. Symmetric renderings,
transpositions, exact duplicates, and minor alternatives with the same plan do
not gain a separate place merely because their coordinates differ.

The guide lists every parent group, source row, eight-ply history, and
multi-child continuation. It embeds 15 full-size exact-parent panels and seven
same-exact-parent child comparison sheets, all bound to the frozen audit by a
hashed manifest. It also makes clear that `ring16` has already normalised
rotations, reflections, and the inner/outer-ring swap; the expert is being
asked for human strategic semantics that the structural audit cannot infer.

The overall 64-prefix composition, the allocation between the two Book
subtypes, and the exact selected members remain product and expert decisions.
