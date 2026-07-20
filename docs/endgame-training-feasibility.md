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

The maintainer later corrected “10/20” to “9/20” and supplied two checkpoints,
training/update logs, a plot, and a browser screenshot. The owner confirms that
all of them were produced by continued `main` training. They narrow the model
identity but still do not supply the exact source commit, frozen launch
contract, route manifest, opponent records, phase-labelled positions, colours,
or fixed work controls needed for a strength claim.

## Facts Reproduced Locally

### The author checkpoint supports level 9, not `dev` lineage

The repository's older `s_gen_v2/best.pt` is a game-11,450, difficulty-6
checkpoint. The newly supplied `../notes/best (copy).pt` is distinct: it reports
stage `s_gen_v2`, game 17,400, difficulty 9, temperature `0.5956225`, and SHA-256
`335462EC3A503E316EAAEF63A7669F1A725FC488A2C27E29B39EFD0021B804D6`.
That metadata supports the maintainer's correction to level 9.

It remains a legacy weights-only `main` artefact. It has no optimiser, RNG,
data identity, run contract, or complete trainer state, and its exact source
commit is unknown. The `/home/.../dev/...` path embedded in its metadata is a
directory label, not Git branch evidence. The file is suitable only for an
explicitly labelled replay or legacy comparison and must never initialise or
resume the fresh `dev` experiment. Full hashes and log findings are in
[`docs/evidence/author-main-generalist-audit-2026-07-20.md`](evidence/author-main-generalist-audit-2026-07-20.md).

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
endgame. They do not prove that this happened in the supplied level-9 model.

### The author logs show concentration and instability, not a phase result

The 10,547-row `main` training log shows `policy_top1_rate` rising from about
0.42 in its first 500 rows to 0.84 in its last 500, while entropy falls from
about 1.55 to 0.34. This supports the narrow observation that sampling became
concentrated on the policy argmax. It does not show that those moves are better:
heuristic-top-1 agreement also rises, no row identifies policy-versus-heuristic
disagreement positions, and every row is labelled `phase_bucket=main`.

The log also has 768 duplicated game numbers, six counter regressions, and a
mid-file opponent-schedule change. It is not one frozen experiment. The update
log's finite policy loss has median about `9.88e7` and maximum about `1.71e29`,
which is incompatible with treating the curve as uncomplicated improvement.
The recorded `--ppo --temp-start 1.1` launch and a log beginning near `0.9`
also show that the author run was affected by trainer behaviour already changed
on `dev`.

The inspected PPO path stores old log probabilities from temperature-scaled
logits and recomputes new probabilities from unscaled logits. Because the exact
author source commit is absent, this is a serious mechanism risk rather than a
complete causal diagnosis of the loss curve. It is enough to keep PPO out of
the fresh baseline pending a focused ratio test and fix.

### Browser Generalist is not the training route

The supplied manual-game screenshot shows the Generalist checkbox selected,
but it does not prove which learned inputs affected the move. The browser builds
the Generalist with globally loaded Sentinel, ValueNet, GapNet, HumanDB, and
SpecialistDB objects; the visible checkboxes are not a component manifest. The
trainer passes Malom as the Generalist lookahead's `endgame_db`, while the
browser loader omits that argument and `GeneralistAgent.score_moves()` encodes
with `db=None` even after `set_db()` is called.

This train/inference mismatch is a plausible contributor to differing endgame
behaviour, but not a diagnosis. A replay must freeze the exact route, feature
inputs, checkpoint hash, work per move, and phase-labelled starts before
comparing models or deciding to enable a separate endgame database.

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

The shortest evidence-producing loop is not an immediate long run or enabling
the unverified internal endgame files:

1. Record the author checkpoint/log hashes already supplied and, if possible,
   obtain the exact `main` commit and launch configuration. Do not make that a
   `dev` dependency.
2. Complete the first baseline's launch controls: explicit imitation-mix
   disablement and an observable fixed-work search budget. Keep it on A2C.
3. Freeze a colour-swapped evaluation corpus stratified by opening, midgame,
   3v3, 4v3, 4v4, 5v4, 6v4, capture-to-fly, and W/D/L starting value.
4. Replay the author level-9 model, the older legacy Generalist, and the first
   corrected baseline with the same explicit route, components, work budget,
   and random streams.
5. Report complete Malom move regret, W-to-D/L and D-to-L downgrades, W
   conversion, rules draws, outcome, and latency separately by phase.
6. Only if the endgame deficit reproduces, define a separate fresh
   `endgame-corrected-v1` experiment with isolated output and database paths.

An endgame specialist may be useful as a diagnostic candidate, but the
existing legacy endgame trainer must not be launched unchanged. A candidate
should use corrected all-candidate Malom values, validated endgame starts, and
an explicit decision about whether its learned result is later distilled into
a unified Phase-aware Generalist. It must also prove that an endgame gain does
not purchase an opening or full-game regression.

## Remaining Questions for the Original Maintainer

1. Which exact `main` commit and launch configuration produced the supplied
   level-9 checkpoint and the appended logs?
2. Which runtime route selected moves: Generalist, Overseer, specialist router,
   or `GameAI`? Which Sentinel, ValueNet, GapNet, Malom, and internal WDL inputs
   were enabled?
3. Was the endgame assessment based on human games, heuristic opponents,
   self-play, or particular recorded positions? Can representative full FENs,
   colours, moves, and completed search work be supplied?
4. Was the level-9 model trained before or after the sector-corrected Malom
   decoder and atomic-capture changes?
5. Are `endgame_4_6.wdl`, `endgame_5_4.wdl`, `endgame_5_5.wdl`, and
   `endgame_6_4.wdl` expected to be partially built? Was `endgame_7_3.wdl`
   intentionally omitted or was the build interrupted?
6. What generator revision, completion log, hash manifest, and rule variant
   produced the fourteen `.wdl` files and `fullgame.bin`?

Answers to these questions can change the experiment choice, but not the
current evidence boundary: no supplied historical checkpoint or internal WDL
file should be relabelled as corrected merely from narrative provenance.
