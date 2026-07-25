# Phase-Covered Corpus v1 — Review Record

Date: 23 July 2026

Status: **generated, mechanically audited, and reviewed once by the Mill
expert; not product-frozen or authorized for execution**.

Related:

- [corpus artifact](dev-v4-phase-covered-corpus-v1.json)
- [freeze-compatible FEN list](dev-v4-phase-covered-corpus-v1-start-positions.json)
- [next-evaluation decision brief](dev-v4-training-aligned-evaluation-v1-decision-brief.md)
- [completed Stage-0 result](../evidence/dev-v4-stage0-result-2026-07-23.md)

## Purpose

The completed Stage-0 corpus contained only placement starts. This draft
prepares controlled coverage of a decision that begins in each standard NMM
phase. It does not replace or extend the Stage-0 sample, and its observations
must never be pooled with the completed Stage-0 ledger.

The draft contains 64 exact and ring16-unique starts:

| Phase | Starts | Definition |
| --- | ---: | --- |
| Placement | 22 | At least one player still has an unplaced piece |
| Movement | 21 | All pieces placed and neither player has exactly three pieces |
| Flying | 21 | All pieces placed and the side to move has exactly three pieces |

The aggregate side-to-move count is White 32 / Black 32. Corrected Malom WDL
for the side to move is draw 24 / loss 22 / win 18. The flying source pool has
only three winning source steps, so the flying stratum deliberately uses
9 draws / 9 losses / 3 wins rather than pretending a balanced source
distribution exists.

## Source and projection boundary

The source is Sanmill's standard-NMM TGF legacy-oracle fixture at:

`crates/tgf-mill/testdata/legacy_oracle/0.json`

The generator resolves the checkout only through the ignored
`sanmill_checkout` path key. It reads the source at commit
`f8be034d10ba7b293f9c10d661b1cfd81e9be096`, Git blob
`a12eb55c32b5b6ccda0e1fbbde3761d384b453af`, and source SHA-256
`4b0ffc4e0d754e9cf2b3275726c366109173d303b1fa2dc0965c76d2db2b1f03`.

This fixture is evidence of deterministic, seeded, legal rules replay. It is
not a collection of expert games and is not a strength oracle. The selected
positions may therefore include unusual, strategically poor, or already
decided states. That variety may be useful for phase stress coverage, but it
requires human review before a formal claim.

The fixture contains eight trajectories and 1,142 action steps. Projection
found:

| Projection fact | Count |
| --- | ---: |
| Pending-removal intermediate states excluded | 60 |
| Stable action states | 1,082 |
| Duplicate ring16 states after first occurrence | 19 |
| Unique stable ring16 states | 1,063 |
| Placement pool | 133 |
| Movement pool | 789 |
| Side-to-move flying pool | 71 |
| Opponent-only flying states not used | 70 |

TGF removal is a separate action. A state with `action=r` cannot be represented
as a stable compact NMM FEN and is never used directly. Placement counters are
derived from the TGF reserve counts, not the action-step number.

All 71 direct flying starts have White to move. Ten selected flying positions
therefore apply the rules-preserving involution that swaps both piece colours
and the side to move. The source FEN and transform are recorded per entry; no
source step is selected twice.

## Data and label isolation

Every selected position was queried before selection against the exact assets
bound to the candidate training route:

| Resource | Bound identity | Selected exact matches |
| --- | --- | ---: |
| HumanDB | `8662e3331210893495aef38c0cb774bd387e508ac8b859261a78b43b74184d31` | 0 |
| Final SpecialistDB | `1203fc73cd7d0a06e2dd1ffaced5b031dff8bd704e22b34ba02182ff3865614d` | 0 |
| Corrected Malom manifest | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` | 64 labelled |

HumanDB historical Malom columns remain masked. The zero-overlap statement is
an exact-position lookup against these two snapshots; it is not a claim of
semantic independence from all possible training positions or symmetric
near-neighbours.

## Review images

The package contains 64 individual 720×840 PNGs and six 4×3 contact sheets.
Codex inspected all six full contact sheets and full-size representatives 001,
023, 044, 046, and 064. Numbering, phase boundaries, board coordinates, piece
colours, counts, side to move, color-transform labels, and sheet coverage were
consistent. No clipping, missing image, or rendering defect was observed.

Contact sheets:

- [001–012](assets/dev-v4-phase-covered-corpus-v1/contact-sheets/sheet-01.png)
- [013–024](assets/dev-v4-phase-covered-corpus-v1/contact-sheets/sheet-02.png)
- [025–036](assets/dev-v4-phase-covered-corpus-v1/contact-sheets/sheet-03.png)
- [037–048](assets/dev-v4-phase-covered-corpus-v1/contact-sheets/sheet-04.png)
- [049–060](assets/dev-v4-phase-covered-corpus-v1/contact-sheets/sheet-05.png)
- [061–064](assets/dev-v4-phase-covered-corpus-v1/contact-sheets/sheet-06.png)

This visual inspection verifies rendering, not Mill quality. The domain review
should flag any position that is unreachable under the intended history
semantics, tactically trivial in a way that would distort the sample, or too
unnatural to support the desired claim. Unusual appearance alone is not an
automatic exclusion because the source is intentionally a rules-replay stress
pool.

## Mill-expert first-pass review

The original maintainer and Mill-domain expert subsequently reviewed all 64
panels. He spent approximately 10–30 seconds on each and supplied the move he
would choose for every position. This is useful coverage evidence, but the
short review was not represented as exhaustive game-theoretic analysis.

He described the set as a useful spread overall, including some unlikely and
already poor or losing states. The explicitly qualified panels were:

| Panel | First-pass observation |
| --- | --- |
| 016 | unlikely; suggested `f6xd3` |
| 021 | unlikely; suggested `f6` |
| 022 | `c3xg4`; questioned whether the state is bad for Black |
| 023 | `f2-f4`; unlikely and bad for White |
| 027 | `d3-e3`; bad for White |
| 028 | unlikely; suggested `a4-b4` |
| 029 | unlikely; suggested `d1-g1` |
| 031 | poor prospects for Black; suggested `b6-d6` |
| 032 | poor prospects for Black; suggested `a4-a7` |
| 038 | bad for Black; suggested `e3-e4` |
| 047 | attributed the position to bad Black play; suggested `g1-d1` |
| 057 | questioned whether Black is already lost; suggested `d5-d1` and a Malom check |

No panel was explicitly rejected in that reply. The expert also recommended a
future tactical group where the side to move must choose among closing a Mill,
blocking an opponent's approaching Mill, removing a piece that enables a chain
Mill, or disabling the opponent's alternative Mill. His illustrative board
was expressly approximate rather than a ready-to-freeze critical position.

These comments do not prove natural HumanDB frequency, reachability under a
particular prior history, or balanced theoretical value. They do close the
pending request for an initial human look. Before freeze, the product owner
must choose one of three explicit dispositions: retain the stress-oriented
draft unchanged, replace specified outliers, or add a separately labelled
tactical stratum. Any exact-list change requires regenerated identities and a
new complete mechanical and visual audit.

## Reproduction and identities

The package was generated from the repository root with:

```powershell
.\.venv\Scripts\python.exe scripts\build_phase_covered_corpus.py
```

The command refuses to overwrite existing targets. A second isolated build
reproduced the corpus JSON identity, FEN-list hash, review-manifest identity,
and every one of the 71 PNG/manifest file hashes.

| Identity | SHA-256 |
| --- | --- |
| Canonical start-position list | `19e6d5f3083e7f023213c465e21d821c31a0ec58b40aa86116fd037694f5410a` |
| FEN-list file | `ef94caa9c0dc71e93f86c1d0948cc2b62e97a5a3bc833fc29696dc6264fc4f36` |
| Corpus identity | `cc5477b777cf38ea59c5f20af4bb7b6019b0d12b328ac139b2a8e7d9f2df0bc8` |
| Corpus file | `cf3c069cd1bb786236172eb28672bbed12886d771977c8c61e99501caa715d2e` |
| Review-manifest identity | `470d38a1e1ac311fb043b2ab31da20a2b413931947d63d33ddd24c5c0b20e4ab` |
| Review-manifest file | `c19680a1e01e0ca8b2ef164343ae070592b55d8478cdd53ebe9cf56769f6f8ce` |

## Freeze disposition

The current lifecycle value is
`domain_reviewed_draft_for_product_disposition_not_frozen`. No specification
references this list, no game has been run from it, and no launch is
authorized.

Before freeze, the product owner must choose the corpus disposition, baseline,
and fixed workload. Any exclusions or additions require regeneration, new
identities, and another complete visual and mechanical audit before an
immutable evaluation specification is created.
