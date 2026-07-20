# Endgame Training Feasibility and Maintainer Questions

## Status and Scope

This note records a read-only feasibility diagnosis made on 20 July 2026. It
does not define a training experiment, approve a smoke, or authorise a long
run. Its purpose is to separate a useful maintainer observation from facts
that are reproducible from the delivered repository and local assets.

The observation supplied for review was:

> Half my irritation Im sure is not experimenting with the ai. Its at 10/20 -
> strong beginning and early mid game but not so great at end game. Still,
> stringest one yet. Im itching to start another ai model and fix a few bits on
> the game here and there.

This is a valuable diagnostic lead, but it is not yet strength evidence. The
exact checkpoint, route, opponent, positions, colours, time controls, and game
records behind the observation have not been supplied.

## Facts Reproduced Locally

### The delivered checkpoint is not the quoted 10/20 checkpoint

The delivered historical Generalist `s_gen_v2/best.pt` reports:

| Field | Value |
| --- | --- |
| `stage` | `s_gen_v2` |
| `game_count` | `11450` |
| `difficulty` | `6` |
| `best_win_rate` | `0.225` |
| move/value feature dimensions | `134` / `80` |

The same directory contains `best1.pt` through `best6.pt`, but no checkpoint
identified as difficulty 10. These assets also pre-date the corrected Malom
decoder and persisted-label migration. They remain useful legacy comparison
inputs, but they cannot establish the state or strength of the model described
in the quotation.

### The current Generalist can advance without an endgame-specific pass

At evidence commit `ef1daeafe8201ad760989c1b7ab75fd792d4a0f6`, the
Generalist trainer:

- defines a 20-level opponent curriculum;
- starts primary games from `BoardState.new_game()`;
- records an endgame branch bucket below twelve total on-board pieces, but has
  branch rollouts disabled by the default `--max-branches-per-game 0`;
- sets `MALOM_REWARD = 0.0`, so a queried coarse Malom move result contributes
  no direct per-move reward;
- uses the corrected Malom adapter for lookahead termination and diagnostic or
  SpecialistDB work, rather than as a complete all-candidate policy target;
- advances difficulty from aggregate full-difficulty heuristic-opponent
  outcomes, without a separate endgame conversion or theoretical-regret gate.

These facts make the reported phase profile plausible: a model can win enough
complete games through strong early play to advance while retaining a weaker
endgame. They do not prove that this happened in the unavailable 10/20 model.

### Corrected Generalist inputs are available, but the long run is undecided

A read-only long-run preflight at the evidence commit confirmed:

- the configured Malom directory was available with `std.secval` and 498
  sector files;
- HumanDB passed SQLite quick-check, with historical unversioned Malom columns
  masked and empirical human data retained;
- the active SpecialistDB passed SQLite quick-check, carried
  `sector-corrected-v1`, and had zero rows in its three data tables;
- the intended corrected Generalist output directory did not exist;
- Sentinel, ValueNet, and GapNet were explicitly disabled.

The preflight verdict was `needs_decision`, because the long-run update
algorithm, opponent schedule, budget, cadence, and stop choices are not frozen.
The reviewed command was:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --preflight long-run `
  --paths-config data\training_paths.local.json `
  --no-sentinel `
  --no-value-net `
  --no-gap-net `
  --temp-start 0.90 `
  --seed 42 `
  --max-games 5000 `
  --batch-games 1 `
  --max-branches-per-game 0 `
  --no-s1a-warmstart
```

The mandatory Malom/provenance tests plus focused preflight, run-contract,
manifest, launch, and checkpoint-envelope tests reported `155 passed, 498
subtests passed` at that clean evidence commit. Later local commits and
uncommitted work require a fresh verification; this result must not be
silently carried forward to a different tree.

## Provisional Audit of the Internal Endgame WDL Files

The local `data/endgame` directory contains fourteen `.wdl` files totalling
1,131,620,677 bytes plus `fullgame.bin`. The following read-only checks were
performed with deterministic seed `20260720`.

### Inventory coverage

The build schedule in `tools/build_endgame_db.py` enumerates both colour-count
orders for totals from six through eleven. Against that schedule:

- `7v3` is missing among tables through total piece count ten;
- all six total-piece-count-eleven tables are absent: `3v8`, `4v7`, `5v6`,
  `6v5`, `7v4`, and `8v3`.

This matters because the endgame trainer classifies positions below twelve
total pieces as endgame positions.

### Query and raw-entry samples

Twenty randomly generated settled boards were queried for each loaded table,
ten with each side to move. Of 280 board queries:

| Result | Count |
| --- | ---: |
| Internal WDL and Malom outcomes matched | 200 |
| Internal WDL and Malom outcomes disagreed | 0 |
| Internal WDL returned unknown | 80 |
| Malom returned unknown | 0 |

All twenty board samples returned internal unknown for each of `4v6`, `5v4`,
`5v5`, and `6v4`.

A second sampler read 20,000 deterministic random packed entries from every
table. The estimated unknown proportions in those four files were:

| Table | Sampled unknown entries |
| --- | ---: |
| `4v6` | 19,931 / 20,000 (99.655%) |
| `5v4` | 20,000 / 20,000 (100%) |
| `5v5` | 20,000 / 20,000 (100%) |
| `6v4` | 19,343 / 20,000 (96.715%) |

The other ten table samples contained no unknown packed entries. Their 200
random board queries agreed with the corrected Malom WDL projection.

These are coverage diagnostics, not an independent proof of correctness or
corruption. The generated boards were not filtered through a reachability
proof, the raw-entry sample is statistical, and the ad hoc sampler has not been
promoted to a reviewed repository tool. The results are nevertheless strong
enough to prohibit treating all fourteen files as complete authoritative
training labels without a reproducible full audit. A corrected experiment can
instead use the complete Malom value/comparator and explicitly disable the
internal WDL reward until the local files are validated or rebuilt.

## Smallest Useful Next Experiment

The shortest evidence-producing loop is not an immediate long run:

1. Obtain or identify the exact 10/20 checkpoint and its training log.
2. Freeze a colour-swapped evaluation corpus stratified by opening, midgame,
   3v3, 4v3, 4v4, 5v4, 6v4, capture-to-fly, and W/D/L starting value.
3. Replay the quoted model, the delivered legacy Generalist, and the first
   corrected baseline with the same route, time budget, and random streams.
4. Report complete Malom move regret, W-to-D/L and D-to-L downgrades, W
   conversion, rules draws, outcome, and latency separately by phase.
5. Only if the endgame deficit reproduces, define a separate fresh
   `endgame-corrected-v1` experiment with isolated output and database paths.

An endgame specialist may be useful as a diagnostic candidate, but the
existing legacy endgame trainer must not be launched unchanged. A candidate
should use corrected all-candidate Malom values, validated endgame starts, and
an explicit decision about whether its learned result is later distilled into
a unified Phase-aware Generalist. It must also prove that an endgame gain does
not purchase an opening or full-game regression.

## Questions for the Original Maintainer

1. Does `10/20` mean the Generalist curriculum's difficulty level 10, or a
   separate strength score?
2. Which exact checkpoint, commit, branch, training output directory, and log
   correspond to that observation? Can a checksum or copy be supplied?
3. Which runtime route selected moves: Generalist, Overseer, specialist router,
   or `GameAI`? Which Sentinel, ValueNet, GapNet, Malom, and internal WDL inputs
   were enabled?
4. Was the endgame assessment based on human games, heuristic opponents,
   self-play, or particular recorded positions? Can representative full FENs,
   colours, moves, and time controls be supplied?
5. Was the 10/20 model trained before or after the sector-corrected Malom
   decoder and atomic-capture changes?
6. Are `endgame_4_6.wdl`, `endgame_5_4.wdl`, `endgame_5_5.wdl`, and
   `endgame_6_4.wdl` expected to be partially built? Was `endgame_7_3.wdl`
   intentionally omitted or was the build interrupted?
7. What generator revision, completion log, hash manifest, and rule variant
   produced the fourteen `.wdl` files and `fullgame.bin`?

Answers to these questions can change the experiment choice, but not the
current evidence boundary: no supplied historical checkpoint or internal WDL
file should be relabelled as corrected merely from narrative provenance.
