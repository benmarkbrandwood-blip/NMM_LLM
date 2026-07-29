# HumanBlunderNet — Phase 1 audit (revision 2)

Phase 1 baseline for the HumanBlunderNet project.  This revision rebuilds
the report against `data/human_blunder_audit_optA.json` (the audit rerun
under Option A boundaries with expanded sample-flow instrumentation) and
addresses every reviewer comment on the first revision.

## Reproducibility

| Item                    | Value                                             |
| ----------------------- | ------------------------------------------------- |
| Audit script            | `tools/audit_human_blunders.py`                   |
| Audit version           | `1.1`                                             |
| Raw JSON output         | `data/human_blunder_audit_optA.json`              |
| Command                 | `.venv/bin/python tools/audit_human_blunders.py --output data/human_blunder_audit_optA.json` |
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

## Option A boundaries

```
lower  ≤ 1150
middle 1151-1250
upper  1251+
```

Strata within *this* PlayOK amateur corpus, not universal strength
labels.  Any future re-cutting is possible without a full rebuild once
the Phase-2 candidate DB stores counts in 50-Elo bins (see plan).

## Per-Elo-band summary (Option A)

Denominators: `total_moves` is every ply the band contributed; `classified`
excludes only the `unlabelled` category; `blunders` counts
`win_to_draw + win_to_loss + draw_to_loss` within `classified`.

| Band   | total_moves | classified | blunders | correct    | all_losing | unlabelled | distinct_positions | blunder %  |
| ------ | ----------- | ---------- | -------- | ---------- | ---------- | ---------- | ------------------ | ---------- |
| lower  |     662,018 |    648,141 |   48,566 |    462,619 |    136,956 |     13,877 |            431,500 | **7.49 %** |
| middle |   1,674,617 |  1,632,642 |   95,405 |  1,304,002 |    233,235 |     41,975 |            932,750 | **5.84 %** |
| upper  |   2,257,378 |  2,198,227 |  106,382 |  1,851,601 |    240,244 |     59,151 |          1,201,793 | **4.84 %** |
| **Σ**  | **4,594,013** | **4,479,010** | **250,353** | **3,618,222** | **610,435** | **115,003** | (positions overlap across bands) | — |

Column sums:
- `total_moves` = plies_raw = 4,594,013 ✓
- `classified` = plies_classified = 4,479,010 ✓
- `blunders` = 250,353 ✓  (matches per-category sum below)
- `unlabelled` = 115,003 ✓ (matches missing-after-only + missing-both above)

Blunder-rate gradient is monotonic in Elo: lower-band players blunder
~55 % more often than upper-band players (7.49 % vs 4.84 %).  Elo
conditioning has real signal even inside this narrow amateur range.

## Cells (band × transition × phase)

Full table with `n_moves` and `n_positions` (distinct state_keys)
lifted directly from `data/human_blunder_audit_optA.json` cells.

### Lower band (≤1150)

| Transition          | Phase | n_moves | n_positions |
| ------------------- | ----- | ------- | ----------- |
| all_losing          | move  | 108,556 | 99,636      |
| all_losing          | place | 28,400  | 22,759      |
| draw_preserved      | move  | 202,189 | 137,853     |
| draw_preserved      | place | 190,697 | 60,621      |
| draw_to_loss        | move  | 11,762  | 10,742      |
| draw_to_loss        | place | 21,368  | 14,507      |
| unlabelled          | move  | 11,046  | 10,916      |
| unlabelled          | place | 2,831   | 2,789       |
| win_preserved       | move  | 54,917  | 51,048      |
| win_preserved       | place | 14,816  | 13,182      |
| win_to_draw         | move  | 6,237   | 5,794       |
| win_to_draw         | place | 7,518   | 6,381       |
| win_to_loss         | move  | 805     | 781         |
| win_to_loss         | place | 876     | 837         |

### Middle band (1151-1250)

| Transition          | Phase | n_moves    | n_positions |
| ------------------- | ----- | ---------- | ----------- |
| all_losing          | move  | 192,578    | 163,624     |
| all_losing          | place | 40,657     | 31,959      |
| draw_preserved      | move  | 602,504    | 334,764     |
| draw_preserved      | place | 488,279    | 124,243     |
| draw_to_loss        | move  | 23,798     | 20,040      |
| draw_to_loss        | place | 36,243     | 24,009      |
| unlabelled          | move  | 35,629     | 34,904      |
| unlabelled          | place | 6,346      | 6,281       |
| win_preserved       | move  | 172,884    | 148,747     |
| win_preserved       | place | 40,335     | 32,918      |
| win_to_draw         | move  | 14,344     | 12,565      |
| win_to_draw         | place | 18,046     | 13,783      |
| win_to_loss         | move  | 1,286      | 1,219       |
| win_to_loss         | place | 1,688      | 1,540       |

### Upper band (1251+)

| Transition          | Phase | n_moves    | n_positions |
| ------------------- | ----- | ---------- | ----------- |
| all_losing          | move  | 211,069    | 174,715     |
| all_losing          | place | 29,175     | 21,822      |
| draw_preserved      | move  | 850,242    | 464,906     |
| draw_preserved      | place | 663,236    | 150,118     |
| draw_to_loss        | move  | 26,993     | 22,286      |
| draw_to_loss        | place | 34,019     | 22,514      |
| unlabelled          | move  | 53,229     | 51,892      |
| unlabelled          | place | 5,922      | 5,801       |
| win_preserved       | move  | 273,279    | 224,970     |
| win_preserved       | place | 64,844     | 49,516      |
| win_to_draw         | move  | 18,126     | 15,559      |
| win_to_draw         | place | 24,351     | 17,527      |
| win_to_loss         | move  | 1,197      | 1,146       |
| win_to_loss         | place | 1,696      | 1,506       |

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

### Lower band (431,500 distinct positions)

| Min plays | Positions | Share  |
| --------- | --------- | ------ |
| ≥1        | 431,500   | 100.00 % |
| ≥5        | 8,668     |   2.01 % |
| ≥10       | 2,737     |   0.63 % |
| ≥25       | 701       |   0.16 % |
| ≥100      | 118       |   0.03 % |

### Middle band (932,750 distinct positions)

| Min plays | Positions | Share  |
| --------- | --------- | ------ |
| ≥1        | 932,750   | 100.00 % |
| ≥5        | 31,903    |   3.42 % |
| ≥10       | 11,038    |   1.18 % |
| ≥25       | 2,911     |   0.31 % |
| ≥100      | 406       |   0.04 % |

### Upper band (1,201,793 distinct positions)

| Min plays | Positions | Share  |
| --------- | --------- | ------ |
| ≥1        | 1,201,793 | 100.00 % |
| ≥5        | 47,021    |   3.91 % |
| ≥10       | 16,307    |   1.36 % |
| ≥25       | 4,234     |   0.35 % |
| ≥100      | 544       |   0.05 % |

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

- ✅ Audit script preserved on disk (`tools/audit_human_blunders.py`) with
  version tag, git-HEAD capture, DB SHA-256, manifest SHA-256, DB
  malom_label_version.
- ✅ Perspective fixture tests live at
  `tests/test_human_blunder_perspective.py`, 25 tests, all pass
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
