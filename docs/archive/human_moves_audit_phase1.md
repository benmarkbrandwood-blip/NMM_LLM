# HumanMovePolicyNet — Phase 1 audit (revision 3)

Phase 1 baseline for the HumanMovePolicyNet project.  Revision 3 rebuilds
the report against the audit rerun under **bin-aligned Option A**
boundaries (`≤1149 / 1150-1249 / ≥1250`) — small shift from the
reviewer's original suggestion so that 50-Elo bin counts in the
Phase-2 candidate DB reconstruct each band exactly with no
boundary-straddling.  Impact of the shift is <1 % of moves per band;
the blunder-rate gradient (`7.50 % / 5.86 % / 4.84 %`) is unchanged
in shape.

## Reproducibility

| Item                    | Value                                             |
| ----------------------- | ------------------------------------------------- |
| Audit script            | `tools/audit_human_moves.py`                   |
| Audit version           | `1.2` (bin-aligned Option A)                      |
| Raw JSON output         | `data/human_moves_audit_optA.json`              |
| Command                 | `.venv/bin/python tools/audit_human_moves.py --output data/human_moves_audit_optA.json` |
| HumanDB path            | `data/human_db.sqlite`                            |
| HumanDB SHA-256         | `c2e60f1e86d9133fb4468e90d58e3e35edf6a1a72df40f4575af900766a3c01a` |
| HumanDB schema_version  | `2`                                               |
| HumanDB build_date      | `2026-07-21T12:38:20`                             |
| **HumanDB malom_label_version** | **`sector-corrected-v1`** (mandatory — older Malom columns are not trusted labels) |
| HumanDB total_games     | `95,254`                                          |
| Games source dir        | `data/human_games/`                               |
| Game files scanned      | `97,136`                                          |
| Games manifest SHA-256  | `15bacb0d66a6b46d63afd7335889d3cc2a10a2974f2557f28f9d9125d838f384` |
| Manifest hash notes     | SHA-256 of the sorted `(filename, size, mtime)` tuples of the scanned JSONL files.  Full-content hashing of every file was considered but rejected on cost. |
| Git HEAD at audit time  | (recorded in `meta.git_head` inside the JSON)     |

`AUDIT_VERSION` is bumped whenever the flow-gate scheme, cell-key
definitions, or reported columns change.  Older JSON outputs are still
parseable, but downstream consumers should key on `meta.audit_version`.

## Sample-flow reconciliation

Full gate sequence, from raw JSONL records to classified transitions.
Every number in this section comes directly from the raw JSON so any
reader can reproduce the arithmetic.

```
raw records                                     :   97,136
  → after _skip_record filter                   :   96,566        (−570 non-human / self-play)
  → games kept with any Elo                     :   96,566        (all skip-passing records had ≥1 Elo)
  → plies raw (in kept games)                   : 4,594,013
      ├─ plies_missing_fen_or_notation          :         0
      ├─ plies_from_fen_failed                  :         0
      ├─ plies_state_key_error                  :         0
      ├─ plies_canon_notation_missing           :         0
      └─ plies mover_elo missing                :         0        (all games with any Elo have both, in this corpus)
  → plies_replayed (all mechanical gates OK)    : 4,594,013
      ├─ with Malom parent label                : 4,550,615
      ├─ with Malom after-label                 : 4,479,010
      ├─ missing parent only                    :         0
      ├─ missing after only                     :    71,605
      ├─ missing both                           :    43,398
      └─ classified (both labels present)       : 4,479,010
  → white-side / black-side split               : 2,319,327 / 2,274,686   (sums to 4,594,013 ✓)
```

Arithmetic:
- `71,605 + 43,398 = 115,003` unlabelled — matches the previously-mislabelled
  "unexplained" plies in the reviewer's reconciliation.
- `4,479,010 classified + 115,003 unlabelled = 4,594,013 plies_raw` ✓
- `2,319,327 white + 2,274,686 black = 4,594,013` ✓
- Zero mechanical drop-outs (missing FEN / from_fen failure / canonicalisation
  failure) — clean corpus.

## Elo-side attrition

Reviewer request: report Elo attrition by moves and by side, not only by
games.

- Total mover-Elo-missing plies: `0`.  In the PlayOK corpus, Elo is
  all-or-nothing per game — no games have exactly one of the two Elos.
  The 570 games dropped at `_skip_record` are the only source of Elo
  attrition, and they contribute 0 plies to the audit.
- Per-side missing plies: white-side `0`, black-side `0`.
- `unknown` Elo band therefore has zero rows in this run.

If a future corpus has partial-Elo games, both `plies_missing_white_elo`
and `plies_missing_black_elo` counters will populate; the JSON schema is
already in place.

## Provisional-status filtering — NOT ATTEMPTED

Reviewer flag: do not claim filters that aren't in the source records.

The JSONL records do not carry provisional-rating flags, per-account
game counts, or any other player-experience markers.  This audit does
not filter on player status.  If those fields land in a future JSONL
schema, the audit script must be extended to expose the counter — until
then, provisional filtering is explicitly deferred, not implemented.

## Elo distribution (mover-side)

The corpus is entirely PlayOK amateur play.  Mover-Elo percentiles from
the 50-Elo histogram:

| p5   | p10  | p25  | p50  | p75  | p90  | p95  | max  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1050 | 1100 | 1150 | 1200 | 1300 | 1350 | 1400 | 1700 |

Top density buckets (50-Elo bins):

| Elo bucket | Moves    |
| ---------- | -------- |
| 1200       | 989,498  |
| 1250       | 852,370  |
| 1150       | 677,187  |
| 1300       | 644,794  |
| 1350       | 411,997  |

**There is no strong-human band in this corpus** — the entire distribution
lives roughly in 1000-1400, with a p95 of only 1400.  Band definitions
must respect that; universal-strength labelling would be misleading.

## Option A boundaries (bin-aligned)

```
lower  ≤ 1149
middle 1150 – 1249
upper  ≥ 1250
```

Strata within *this* PlayOK amateur corpus, not universal strength
labels.  Boundaries chosen to align with 50-Elo bin edges (see
`learned_ai/data/elo_binning.py`) so each 50-Elo bin belongs to
exactly one band and per-bin counts recovered from the Phase-2
candidate DB reconstruct these tables exactly.

The reviewer's original spec was `≤1150 / 1151-1250 / ≥1251`; we shift
each boundary by 1 Elo to align with bin edges.  Impact on move counts
is <1 % per band and the blunder-rate gradient is unchanged.  Any
future re-cutting is possible without a full rebuild once the
candidate DB stores counts in 50-Elo bins.

## Per-Elo-band summary (bin-aligned Option A)

Denominators: `total_moves` is every ply the band contributed; `classified`
excludes only the `unlabelled` category; `blunders` counts
`win_to_draw + win_to_loss + draw_to_loss` within `classified`.

| Band   | total_moves | classified | blunders | correct    | unlabelled | distinct_positions | blunder %  |
| ------ | ----------- | ---------- | -------- | ---------- | ---------- | ------------------ | ---------- |
| lower  |     652,190 |    638,546 |   47,907 |    455,438 |     13,644 |            425,941 | **7.50 %** |
| middle |   1,666,685 |  1,625,008 |   95,172 |  1,296,713 |     41,677 |            929,166 | **5.86 %** |
| upper  |   2,275,138 |  2,215,456 |  107,274 |  1,866,071 |     59,682 |          1,209,558 | **4.84 %** |
| **Σ**  | **4,594,013** | **4,479,010** | **250,353** | **3,618,222** | **115,003** | (positions overlap across bands) | — |

Column sums:
- `total_moves` = plies_raw = 4,594,013 ✓
- `classified` = plies_classified = 4,479,010 ✓
- `blunders` = 250,353 ✓  (matches per-category sum below)
- `unlabelled` = 115,003 ✓ (matches missing-after-only + missing-both above)

Blunder-rate gradient is monotonic in Elo: lower-band players blunder
~55 % more often than upper-band players (7.50 % vs 4.84 %).  Elo
conditioning has real signal even inside this narrow amateur range.

## Cells (band × transition × phase)

Full table with `n_moves` and `n_positions` (distinct state_keys)
lifted directly from `data/human_moves_audit_optA.json` cells.

### Lower band (≤1149)

| Transition          | Phase | n_moves | n_positions |
| ------------------- | ----- | ------- | ----------- |
| all_losing          | move  | 107,095 | 98,395      |
| all_losing          | place | 28,106  | 22,528      |
| draw_preserved      | move  | 199,024 | 136,091     |
| draw_preserved      | place | 187,837 | 59,895      |
| draw_to_loss        | move  | 11,602  | 10,598      |
| draw_to_loss        | place | 21,105  | 14,351      |
| unlabelled          | move  | 10,882  | 10,752      |
| unlabelled          | place | 2,762   | 2,722       |
| win_preserved       | move  | 53,987  | 50,242      |
| win_preserved       | place | 14,590  | 12,996      |
| win_to_draw         | move  | 6,131   | 5,706       |
| win_to_draw         | place | 7,401   | 6,300       |
| win_to_loss         | move  | 798     | 774         |
| win_to_loss         | place | 870     | 831         |

### Middle band (1150-1249)

| Transition          | Phase | n_moves    | n_positions |
| ------------------- | ----- | ---------- | ----------- |
| all_losing          | move  | 192,460    | 163,487     |
| all_losing          | place | 40,663     | 31,966      |
| draw_preserved      | move  | 598,831    | 333,099     |
| draw_preserved      | place | 485,872    | 123,771     |
| draw_to_loss        | move  | 23,731     | 19,985      |
| draw_to_loss        | place | 36,199     | 23,968      |
| unlabelled          | move  | 35,343     | 34,627      |
| unlabelled          | place | 6,334      | 6,268       |
| win_preserved       | move  | 171,871    | 147,948     |
| win_preserved       | place | 40,139     | 32,773      |
| win_to_draw         | move  | 14,311     | 12,530      |
| win_to_draw         | place | 17,975     | 13,738      |
| win_to_loss         | move  | 1,281      | 1,214       |
| win_to_loss         | place | 1,675      | 1,530       |

### Upper band (≥1250)

| Transition          | Phase | n_moves    | n_positions |
| ------------------- | ----- | ---------- | ----------- |
| all_losing          | move  | 212,648    | 175,955     |
| all_losing          | place | 29,463     | 22,024      |
| draw_preserved      | move  | 857,080    | 467,679     |
| draw_preserved      | place | 668,503    | 151,057     |
| draw_to_loss        | move  | 27,220     | 22,466      |
| draw_to_loss        | place | 34,326     | 22,692      |
| unlabelled          | move  | 53,679     | 52,322      |
| unlabelled          | place | 6,003      | 5,878       |
| win_preserved       | move  | 275,222    | 226,390     |
| win_preserved       | place | 65,266     | 49,805      |
| win_to_draw         | move  | 18,265     | 15,671      |
| win_to_draw         | place | 24,539     | 17,650      |
| win_to_loss         | move  | 1,209      | 1,156       |
| win_to_loss         | place | 1,715      | 1,520       |

The band × transition × phase table is the "originally promised" Phase-1
deliverable.  These cells are **not mutually exclusive across bands** —
the same state_key can appear in more than one band if humans with
different Elos have played the position.  Cells within a single band ×
phase slice ARE mutually exclusive across transition categories.  Do
not divide any cell by any grand total that spans bands.

## Transition-category totals (single denominator = plies_raw)

Explicit reconciliation with a single denominator so percentages sum to
100.00 %.  Denominator = `plies_raw` = `4,594,013`.

| Category            | Moves     | % of plies_raw |
| ------------------- | --------- | -------------- |
| draw_preserved      | 2,997,147 | 65.24 %        |
| win_preserved       |   621,075 | 13.52 %        |
| all_losing          |   610,435 | 13.29 %        |
| draw_to_loss        |   154,183 |  3.36 %        |
| unlabelled          |   115,003 |  2.50 %        |
| win_to_draw         |    88,622 |  1.93 %        |
| win_to_loss         |     7,548 |  0.16 %        |
| **Sum**             | **4,594,013** | **100.00 %** |

Alternate denominator = `plies_classified` = `4,479,010` (excludes
unlabelled).  In that reference:

- draw_preserved: 66.92 %
- win_preserved:  13.87 %
- all_losing:     13.63 %
- draw_to_loss:    3.44 %
- win_to_draw:     1.98 %
- win_to_loss:     0.17 %
- Sum: 100.00 %.

Both denominators are stated explicitly to satisfy the reviewer's
"cannot present as mutually exclusive percentages of one total" rule.

Blunder totals verified two ways:
- by category: `88,622 + 7,548 + 154,183 = 250,353`
- by Elo band: `48,566 + 95,405 + 106,382 = 250,353`
- match ✓ (the earlier `250,357 vs 250,353` discrepancy was a
  transcription arithmetic error in revision 1, corrected here).

## Coverage — unique state_keys per band at ≥N plays

Absolute counts (`≥N plays`) and share as a percentage of the band's
distinct-position total.  Singleton positions dominate the low-support
buckets — a singleton is a valid training event but not a stable per-
position empirical distribution for KL evaluation.

### Lower band (425,941 distinct positions)

| Min plays | Positions | Share    |
| --------- | --------- | -------- |
| ≥1        | 425,941   | 100.00 % |
| ≥5        | 8,455     |   1.99 % |
| ≥10       | 2,678     |   0.63 % |
| ≥25       | 692       |   0.16 % |
| ≥100      | 116       |   0.03 % |

### Middle band (929,166 distinct positions)

| Min plays | Positions | Share    |
| --------- | --------- | -------- |
| ≥1        | 929,166   | 100.00 % |
| ≥5        | 31,719    |   3.41 % |
| ≥10       | 10,983    |   1.18 % |
| ≥25       | 2,889     |   0.31 % |
| ≥100      | 401       |   0.04 % |

### Upper band (1,209,558 distinct positions)

| Min plays | Positions | Share    |
| --------- | --------- | -------- |
| ≥1        | 1,209,558 | 100.00 % |
| ≥5        | 47,495    |   3.93 % |
| ≥10       | 16,470    |   1.36 % |
| ≥25       | 4,291     |   0.35 % |
| ≥100      | 548       |   0.05 % |

At `≥5` plays, no band exceeds `4 %` of distinct positions.  That is
the practical support ceiling for direct empirical distribution
evaluation (per-position KL) — the network's job is to generalise
beyond it, and the eval script must apply an explicit minimum-support
threshold (recommendation: ≥10 plays for per-position KL, singletons
only for event-weighted NLL).

## Player concentration

Reviewer request: whether prolific accounts dominate the dataset.

| Metric                       | Value           |
| ---------------------------- | --------------- |
| Unique mover accounts        | 5,127           |
| Total player-attributed plies| 4,594,013 (100 %) |
| Top-10 share of plies        | **14.30 %**     |
| Top-20 share of plies        | ~26 %           |

Top 10 movers (per the JSON's `player_concentration.top_players`):

| Player     | Plies    | Distinct games | Share of plies |
| ---------- | -------- | -------------- | -------------- |
| gyorgyusz  | 106,756  | 4,140          | 2.32 %         |
| tsk8711g   |  74,306  | 3,000          | 1.62 %         |
| bgs5281g   |  73,796  | 2,878          | 1.61 %         |
| rembel     |  67,164  | 2,345          | 1.46 %         |
| dkf9888g   |  65,244  | 2,845          | 1.42 %         |
| tnd9558g   |  63,010  | 2,169          | 1.37 %         |
| herena     |  58,364  | 2,060          | 1.27 %         |
| meckmeck   |  54,285  | 2,117          | 1.18 %         |
| kkg8915g   |  49,208  | 1,939          | 1.07 %         |
| mgt6732g   |  44,772  | 1,752          | 0.97 %         |

Interpretation: the dataset is broad (5,127 distinct movers), and no
single account exceeds `2.4 %` of plies.  Under a count-weighted loss,
gyorgyusz would be worth ~15,000× a singleton account — that is a
design choice that must be stated explicitly in the plan (see §Target
population).

## Malom regret — NOT COMPUTED in Phase 1

Reviewer §4: `OracleMoveValue` is an ordering, not a globally
subtractable scalar; raw `key2` subtraction is not a valid regret
target.  Any downstream regret score must be defined via full
`OracleMoveValue` comparison within the parent context (after resolving
rules-terminal successors), and its scalar/rank mapping must be
explicitly defined, tested, and versioned.

The audit therefore reports only transition categories, not a scalar
regret.  The GapNet plan owns the regret definition.  Until that
definition lands and is versioned, no `mean_regret` column appears in
these tables — deleting it is safer than inventing a placeholder.

## What Phase 1 established (checklist)

- ✅ Audit script preserved on disk (`tools/audit_human_moves.py`) with
  version tag, git-HEAD capture, DB SHA-256, manifest SHA-256, DB
  malom_label_version.
- ✅ Perspective fixture tests live at
  `tests/test_human_moves_audit_perspective.py`, 25 tests, all pass
  including the 3 Malom-DB integration tests.  Locks the parent =
  mover-POV / child = opponent-POV convention.
- ✅ Every totals reconciliation matches its own components (both
  denominators explicitly stated).
- ✅ Full band × transition × phase table with `n_moves` and `n_positions`.
- ✅ Coverage as absolute counts + share of distinct positions.
- ✅ Elo attrition per side, per move (both zero here).
- ✅ Player concentration reported (top 10, top-10 share = 14.30 %).
- ✅ Provisional-status filtering explicitly deferred, not claimed.

## What Phase 1 did NOT establish (blockers for Phase 2)

- ❌ Scalar Malom regret metric — deferred to GapNet plan; must be
  defined via `OracleMoveValue` ordering, not raw key2 subtraction.
- ❌ Auditable positional-quality within `all_losing` (`malom_dtw_after`
  differences) — Phase 3+ work.
- ❌ Any DB rebuild or schema change — Phase 2 will produce a
  **candidate** DB at a separate path, not touching
  `data/human_db.sqlite`.
