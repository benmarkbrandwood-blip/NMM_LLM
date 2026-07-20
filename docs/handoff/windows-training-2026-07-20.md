# Windows Training Handover - 20 July 2026

## Executive Summary

The repository is now usable on the Windows 11 host, the downloaded databases
and existing model artefacts are in their intended locations, and the focused
Malom/provenance suite is green. A long corrected training run has not been
started.

The auto-resume and temperature-schedule defects have now been fixed and tested
on local `dev`. The owner has defined the first experiment as a fresh,
Malom-corrected v4-style baseline with legacy Sentinel, ValueNet, and GapNet
disabled. Its bounded one-game smoke passed with a checkpoint observation; a
later `dev` commit corrected the inaccurate checkpoint report and records a
conservative explicit-resume policy. A long monitored run has not started and
remains gated on frozen launch choices plus a newly authorised final smoke and
preflight.

Read
[`docs/local-training-layout.md`](../local-training-layout.md) for the relative
storage map and machine-local lookup keys, and
[`docs/v5-specialist-plan.md`](../v5-specialist-plan.md) for the broader design.
Machine-specific absolute values are intentionally kept only in the ignored
`data/training_paths.local.json`. Path names shown in committed documents are
relative to the repository root; Markdown link targets are relative to their
containing files so that they render correctly.
The v5 plan is a target and audit document; statements in it that describe the
decoder as currently broken have been superseded by the commits listed below.

## Repository and Workspace Boundary

- Repository: the Git repository containing this document
- Branch: `dev`
- Remote: `origin`, using
  `git@github.com:benmarkbrandwood-blip/NMM_LLM.git`
- Intended execution host: Windows 11, without a WSL requirement
- Parent directory: data container only; it must not become a Git repository

The current Codex task is already open at the repository root, as confirmed by
`git rev-parse --show-toplevel`. Future tasks should use the same workspace
boundary and begin by reading the repository's [`AGENTS.md`](../../AGENTS.md)
and this file. Consult
[`docs/local-training-layout.md`](../local-training-layout.md) when the
storage relation or machine-local configuration key is needed.

## Git Synchronisation Completed

The earlier rewritten-but-patch-equivalent divergence has been resolved. Before
the update, `5880316` was patch-equivalent to remote `9e46334`, `5a17738` was
patch-equivalent to remote `643a5e7`, and local `06598c9` was the additional
PyO3/Python 3.13 compatibility change.

On 20 July 2026, the owner explicitly authorised local `dev` to replace the
remote branch with `--force-with-lease`. The lease was pinned to remote tip
`643a5e766768239bac030d32afc8915f5f90a570`, and the update completed
successfully. Immediately before the documentation commit containing this
handover, both `dev` and `origin/dev` pointed to:

```text
06598c9dabeabdd613070d3bbc8634bc2f2b3977
```

`git rev-list --left-right --count dev...origin/dev` returned `0 0`. The
handover commit `8751da4` was subsequently pushed and is now the recorded
`origin/dev` tip. Local `dev` then added the independently tested auto-resume
and temperature commits `5eadb4e` and `006715b`, the component-disable commit
`24be10b`, the experiment-definition and smoke-evidence commits `80f4a1f` and
`53d86d1`, and the follow-up maintenance commits through `9c7dceb`. Inspect the
live graph rather than relying on this snapshot. The completed
force-with-lease approval is not standing permission for a future push or
history rewrite; obtain fresh authorisation when such an operation becomes
necessary.

## Environment State

The current local environment was checked as follows:

| Component | State |
| --- | --- |
| Python virtual environment | `.venv`, Python 3.13.1 |
| PyTorch | Importable |
| Native `nmm_core` extension | Importable |
| ChromaDB | 1.5.9, importable |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported memory |
| NVIDIA driver | 610.74 |

`python -m pip check` reports no broken installed requirements. Modules such as
`sentence_transformers`, `faiss`, and `sklearn` are not installed, but they are
not declared by the repository's two requirements files and did not cause the
current test-collection failures. Do not call them missing project dependencies
without first defining a feature that requires them.

Commit `06598c9` records successful `cargo check --locked`, editable
installation of the CPython 3.13 extension, and fifteen native parity tests.
The focused Python verification was re-run during this handover:

```text
102 passed, 498 subtests passed
```

The command was:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_malom_db.py `
  tests/test_sentinel_db_teacher.py `
  tests/test_malom_label_provenance.py -q
```

`scripts/train_s_gen_v2.py --help` also completes successfully. A fresh full
collection found 705 tests and stopped on four repository-interface errors:

- `tests/test_legal_moves.py` imports the absent
  `learned_ai.models.action_encoder` module;
- `tests/test_sentinel_feature_builder.py` imports the absent
  `learned_ai.models.state_encoder` module;
- `tests/test_sentinel_labels.py` imports the absent historical symbol
  `DEFAULT_BACKWARD_DECAY`;
- `tests/test_sentinel_model.py` imports the absent historical symbol
  `SentinelOutput`.

These are missing/stale internal interfaces, not third-party dependency errors.
The complete suite is therefore not a clean project baseline. Do not hide the
four collection errors, but do not confuse them with the focused Malom result
either.

Follow-up maintenance on local `dev` recalibrated the two stale GameAI tactical
fixtures against a legal terminal-mill position. It also replaced tests that
depended on an untracked `data/games` corpus with deterministic JSONL fixtures;
the current Sentinel and TrajectoryDB loader tests therefore execute rather
than skip when that local directory is absent. The four unrelated collection
errors above remain unresolved and continue to bound any full-suite claim. The
combined Generalist, GameAI, Sentinel-dataset, and TrajectoryDB verification
reported `58 passed`; the mandatory Malom/provenance rerun again reported
`102 passed, 498 subtests passed`. A fresh collection-only check still stopped
on exactly the same four interface errors listed above.

## Data and Model State

The Google Drive delivery referenced by the original handover has been moved
out of the import staging directory and into the intended repository-local or
external locations. The staging directory is now empty. Its host-specific
location and the relative destination map are recorded in the
[`docs/local-training-layout.md`](../local-training-layout.md) path list.

Available assets include:

- the 738,091,008-byte HumanDB and 95,389 human-game `.jsonl` files;
- fourteen endgame WDL tables and `fullgame.bin`;
- the complete external Malom directory, with 512 files totalling
  83,582,223,577 bytes;
- Sentinel `best.pt`;
- historical opening, midgame, endgame, and generalist checkpoints;
- value-net and gap-net artefacts.

The assets are present, but they are not all equally trustworthy:

- HumanDB human frequencies, outcomes, and counts remain useful.
- HumanDB's unversioned historical Malom columns are masked by current readers.
- `data/specialist_db.sector_corrected.sqlite` is trusted and deliberately
  empty. It is the active SpecialistDB for a corrected run.
- Both legacy SpecialistDB deliveries are isolated in the ignored backup
  directory and must remain read-only.
- Historical checkpoints and nets pre-date the corrected decoder/provenance
  migration. Retain them as exploratory baselines; do not claim that they were
  trained from corrected labels.
- The original maintainer describes the endgame tables and `fullgame.bin` as
  outputs of their backwards solver. That is a provenance statement, not an
  independent correctness check. A follow-up read-only inventory and sampling
  audit found missing table coverage plus concentrated unknown entries in four
  loaded tables; see
  [`docs/endgame-training-feasibility.md`](../endgame-training-feasibility.md).
  That diagnostic is not a full differential proof. Record hashes and complete
  the reviewed validation before using those files as authoritative labels or
  acceptance evidence.
- `gap_net_path` is deliberately blank in the local path configuration even
  though the files exist. Do not enable it until its label provenance is
  reviewed.

## Completed Correctness Work

The following commits on `dev` form the relevant correction chain:

| Commit | Result |
| --- | --- |
| `44a0fd3` | Corrects sector-adjusted Malom value decoding |
| `98ff63a` | Makes Mill formation plus capture an atomic Malom move query |
| `803eee8` | Resolves rules-terminal states before tablebase lookup |
| `216a77f` | Compares moves with complete oracle values rather than incomplete child fields |
| `8da033e` | Rejects impossible positive move-quality deltas |
| `7cf7725` | Ignores recursively imported game data and SQLite training data |
| `5880316` | Versions persisted Malom labels and gates every direct consumer |
| `5a17738` | Covers suffixed SpecialistDB SQLite files in `.gitignore` |
| `06598c9` | Updates PyO3 to build `nmm_core` under Python 3.13 |

The decoder and capture semantics were also checked against real Malom files:
961 sampled positions matched the corrected reference projection. This
external comparison supports the result, but the project tests and this
repository's rule semantics remain the primary acceptance evidence.

## Persisted-label Behaviour

Current code uses `malom_label_version=sector-corrected-v1` as the trust gate.
It has the following intended behaviour:

- a new or unlabelled SpecialistDB may adopt the current version;
- a labelled but unversioned SpecialistDB is treated as legacy;
- empirical game statistics may still be read from legacy data, but legacy
  Malom priors are ignored;
- new Malom labels cannot be appended to a legacy labelled database;
- HumanDB readers preserve human statistics whilst masking legacy WDL/DTW;
- HumanDB builders refuse to mix corrected labels into a legacy labelled DB;
- direct gap-dataset and trajectory-label consumers require current metadata.

The active HumanDB has 1,560,069 labelled position rows and 1,691,422 labelled
move rows but no label-version key, so its Malom fields are intentionally
untrusted. The active corrected SpecialistDB has zero positions, winning lines,
or preferred plays.

The 20 July author update added 406 valid human-game JSONL files. Their content
matches `human_games_94559.zip`, and the import manifest grew from 94,134 to
94,540 entries. Four added records have an empty `moves` list and were retained
unchanged from the source package. `data/human_db.sqlite` was not rebuilt, so
its 94,429-game inventory still represents the earlier corpus.
The source ZIP is archived outside Git at
`../human_database/human_games_94559.zip`; its SHA-256 is
`45523234085518031A09725A2DBCAB395E55026787E420A04C37EBA10A0E4D07`.
Do not run the current builder's `--update` mode blindly: all 94,983 existing
`processed_files.file_path` values use the author's `/home/...` absolute path,
so Windows paths would be treated as new files and their statistics would be
added again. Migrate those keys or perform a controlled rebuild before adding
the 406 games to HumanDB.

The accompanying 268,521,472-byte SpecialistDB passed `integrity_check` and
contains 1,954,437 positions with 339,904 labels, but it has no `meta` table and
therefore no trusted label version. It is quarantined as
`data/backups/drive_import_20260720/specialist_db.sqlite.legacy-author-update-20260720`
with SHA-256
`5C6A4EA1ACFB90BF05248580A07DAE7CF4645C09E5A4A69E2EC89EA9EE41811B`.
The active corrected database was not replaced and retains SHA-256
`CB4153A14752357587890EB5F8B655AB04AF8242E43BE1C80D4847A11D101A94`.

The downloaded `build_endgame_db.py` and `build_fullgame_db.py` are byte-for-byte
identical to the repository copies. The downloaded `build_human_db_sha.py` is
an older version that lacks the repository's Malom label-provenance guard, so
it was not copied over `tools/build_human_db_sha.py`.

## Source-note Evidence Boundary

The machine-local `Notes.md` and its screenshots are historical operator
observations, not a specification, test result, or source of authoritative
labels. Path and asset claims in that note were checked independently before
being recorded here. Preferences such as "the generalist is the way to go",
reported difficulty levels, proposed specialist grading changes, expected
Sentinel improvement, and possible trap training remain hypotheses until a
reproducible experiment supports them.

The screenshots also pre-date the corrected Malom decoder, so their Malom
arrows cannot be used as oracle evidence. They do preserve useful diagnostic
leads:

- in one recorded position the policy/Overseer assigned `100%` to `f2` while
  the displayed Sentinel score was `54%`; displayed alternatives included
  `d3` at `92%` and `d1` at `82%`;
- two other `100%` selections coincided with the highest displayed Sentinel
  score, and another position showed a distributed policy, so the screenshots
  do not establish universal policy collapse or universal disagreement;
- the aggregate dashboard shows large policy/value-loss spikes. Its green
  vertical markers are difficulty advances generated by
  `tools/plot_specialist_training.py`, not recovery events.

The note's report that the midgame specialist and generalist reached level 7
and approached level 8 is therefore historical context only. The suggestion
that opening and endgame specialists need different grading is an experiment
proposal, not a diagnosed cause. Before acting on either claim, replay recorded
FENs with a pinned checkpoint and log policy entropy, top-one mass, Sentinel
rank, legal-move coverage, and corrected oracle values; evaluate strength only
with frozen, colour-swapped matches and intervals.

## Generalist Trainer Corrections

### Auto-resume follows the configured output directory

The machine-specific configuration sends new output to:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected
```

Commit `5eadb4e` changes `_choose_resume_path()` so `--auto-resume-best` reads
`best.pt` from the resolved `args.out_dir`; it no longer falls back to the
historical fixed directory. Regression tests cover explicit-resume precedence,
the configured output path, and isolation from the old directory. The fresh
baseline still intentionally omits both `--resume` and `--auto-resume-best`.

### The CLI temperature schedule controls the loop

Commit `006715b` passes `--temp-start` into the schedule for both fresh and
resumed game counts. Temperature reaches the fixed `0.20` endpoint after 80 per
cent of `--max-games`. Recovery no longer resets temperature: it still restores
the selected weights and applies the existing draw-penalty grace, but
exploration stays on the global schedule. Focused tests cover a custom start,
ordinary decay, endpoint clamping, and the unchanged default schedule.

Commit `fe0b1f1` additionally makes `--temp-start` reject zero, negative, and
non-finite values during argument parsing, before training resources are
opened. Focused tests cover valid decimal and exponential forms plus zero,
negative, `NaN`, infinities, and non-numeric input.

### Final checkpoint reporting matches repository state

Commit `bf9472c` always reports the final `latest.pt` path and reports
`best.pt` only when that file actually exists. The best snapshot is optional:
it is created only at a logging checkpoint after at least 10 heuristic games
when the current win rate strictly improves on the prior best at that
difficulty. Regression tests cover both reporting outcomes and all sides of
that gate.

## First Dev Experiment Decision

The owner selected `dev-v4-malom-corrected-fresh-v1`: a fresh-initialised,
Malom-corrected v4-style Generalist baseline. It does not load the author's
continuing `main` checkpoint, does not use automatic resume, starts with an
empty `sector-corrected-v1` SpecialistDB, and explicitly disables the legacy
Sentinel, ValueNet, and GapNet. The trainer exposes `--no-sentinel`,
`--no-value-net`, and `--no-gap-net` so this choice overrides machine-local
configured paths rather than depending on missing files.

The complete definition, preflight evidence, claim boundary, isolated smoke
command, and result are in
[`docs/experiments/dev-v4-malom-corrected-baseline.md`](../experiments/dev-v4-malom-corrected-baseline.md).

The smoke ran from clean commit
`80f4a1fe525d98706b1b0913083f2c2067f8bf66`, completed one 33-ply game on CUDA,
and exited successfully. It started from scratch, disabled all three legacy
learned inputs, loaded Malom and HumanDB, wrote a trusted disposable
SpecialistDB, and left the active empty baseline DB unchanged. This is
integration evidence only, not strength evidence.

The generated `latest.pt` is readable, but the final console message named
`best.pt` even though no such file was produced by the one-game run. This does
not invalidate the historical smoke. Commit `bf9472c` fixes the message; a
one-game run is now explicitly reported as having no best checkpoint.

The experiment now treats `latest.pt` as a weight-continuation snapshot and
`best.pt` as optional model-selection evidence. Initial launch remains fresh
and omits all resume flags. After an interruption, automatic continuation is
forbidden: inspect the latest snapshot and use a separately recorded explicit
`--resume` only if weight continuation is acceptable. Current checkpoints do
not restore optimiser, rolling-history, difficulty-local, target-age, or RNG
state, so they are not evidence of exact trainer-state recovery.

## Live Malom and Legacy-model Boundary

The old note says `specialist_router.py` was a temporary containment against a
broken Malom decoder. In current code, the specialist and generalist router
score paths still call the feature encoder with `db=None`, while separate Web
and `GameAI` paths can attach and query the now-corrected Malom implementation.
The blanket historical instruction to keep Malom out of all inference has
therefore been superseded, but the active path remains important evidence.

Any smoke or release check must record which route made the decision and test
that route with corrected atomic-capture, terminal-state, perspective, and
full-value semantics. The existing Sentinel, value-net, gap-net, specialist,
and generalist checkpoints all pre-date the correction. They may be used only
as explicitly labelled legacy inputs or ablations; loading one does not make it
a corrected model. Whether Sentinel training improves after corrected labels
is still untested.

## Mixed-opponent Handover Copy

The uncommitted mixed-opponent edit from the previous maintainer was preserved
outside the repository as `train_s_gen_v2_handoff_unfinished.py`. Its exact
repository-relative location is recorded under the reference-only `notes`
entry in the
[`docs/local-training-layout.md`](../local-training-layout.md) path list.

The tracked `scripts/train_s_gen_v2.py` was restored afterwards. Do not replace
the tracked script with the archived copy. The current tracked schedule already
supports a configurable frozen self-play ratio and gives 15 per cent of
heuristic games a randomly lower difficulty. It does not implement the full
requested schedule of fixed higher/lower proportions, deliberate blunders, or
value/gap/Sentinel opponent blends.

The archived comments propose a 10/20/10/10/50 per-game schedule and describe
the blended branch as 10 per cent ValueNet, 30 per cent GapNet, and 20 per cent
Sentinel. The code does not establish those claimed inner blend weights:

- it supplies a ValueNet without changing the default zero
  `value_net_blend`;
- it attaches Sentinel in the default advisory mode rather than a 20 per cent
  move-selection override;
- it leaves GapNet on the existing phase-specific defaults rather than a
  uniform 30 per cent blend;
- its blunder branch uses a 25 per cent per-move probability inside selected
  games; that exact event distribution must be documented and tested rather
  than inferred from the prose request.

Those comments express intent, not completed behaviour. The draft also lets
most experimental opponent types affect level-advancement history, which would
confound grading unless each stratum is logged and advancement is defined
against a stable opponent.

That experimental schedule is not required to establish the first corrected
baseline. If revisited, audit each opponent type, sampling probability,
determinism, diagnostics, and failure fallback, then implement and test it as a
new change rather than recovering the interrupted edit wholesale.

## Monitoring and Resource Notes

`scripts/train_s_gen_v2.py` uses a `ThreadPoolExecutor` when `--batch-games` is
greater than one. Game simulation remains substantially CPU-bound, and the
original operator observed that excessive parallelism slowed iteration. Treat
that as a benchmark lead rather than a fixed worker recommendation: record the
worker count, games/hour, CPU and RAM use, GPU utilisation, search settings,
and output/database contention before selecting long-run concurrency. Keep the
first integration smoke at `--batch-games 1`.

The existing monitor can be started from the repository root with:

```powershell
.\.venv\Scripts\python.exe tools\plot_specialist_training.py
```

It refreshes every 20 minutes by default and visualises existing logs; it is a
health monitor, not strength or correctness evidence. Before a long run,
record the log path, refresh interval, checkpoint cadence, stop criteria, and
who or what will inspect stalled games, non-finite losses, recovery loops, and
database growth.

## Deferred and Conditional Work from the Original Notes

- Direct "learn traps" training is not implemented. The v5 plan defines fixed
  trap scenarios for stress testing and diagnosis, which is not evidence that
  a trap curriculum is necessary or effective.
- The v5 teacher/HumanPolicy signal, human-evaluation power, rule/oracle
  semantics, and implementation complexity require the independent reviews
  specified by that plan before their optional branches are opened. They are
  not prerequisites for the minimal corrected v4-style baseline.
- Puzzle repair, Windows/Linux installers, hosting, a book link, and additional
  languages are product backlog ideas. They are outside this training handover
  and carry no implementation commitment.
- Starting a separate Sanmill-trained AI is not an accepted next action. The
  pinned Sanmill checkout is a reference and possible differential-test input
  under the boundary recorded in the local-layout document.

## Recommended Next Actions

The workspace/root check, graph inspection, trainer fixes, focused tests,
102-test Malom/provenance rerun, first-experiment component decision, bounded
smoke, and checkpoint-policy correction are complete. Proceed in this order:

1. Freeze the long-run update algorithm, opponent schedule, temperature start,
   game budget, seed, concurrency, checkpoint cadence, monitor interval, and
   stop criteria.
2. Run the repository training-readiness workflow and a newly authorised
   bounded smoke on the intended launch commit with new disposable paths.
3. Re-run the clean-worktree, path, active-DB, test, and component preflight
   immediately before any long run. Do not push, smoke, or start that run
   without the corresponding explicit approval.

The executed isolated smoke command was:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --paths-config data\training_paths.local.json `
  --out-dir learned_ai\checkpoints\smoke\s_gen_v2_v4_malom_corrected_fresh_v1 `
  --specialist-db data\specialist_db.smoke.v4_malom_corrected_fresh_v1.sqlite `
  --no-sentinel `
  --no-value-net `
  --no-gap-net `
  --temp-start 0.90 `
  --seed 42 `
  --max-games 1 `
  --batch-games 1 `
  --max-ply 40 `
  --sim-ply-depth 2 `
  --minimal-rollouts `
  --no-s1a-warmstart
```

The command intentionally omitted `--resume`, `--auto-resume-best`, and `--ppo`.
It exited successfully in approximately 24.4 seconds. Its output and database
remain ignored and separate from the intended long-run paths. The experiment
document records their verified contents and the checkpoint observation.

The original handover's 50,000-game PPO command should not be launched
unchanged. PPO and the more complex opponent mixture are optional experiments
under the v5 plan, not prerequisites for a corrected baseline.

## Recorded and Remaining Owner Decisions

The following choices are recorded for the first `dev` experiment:

- start from random model weights, not a historical checkpoint;
- use the corrected v4-style Generalist path, not claim the staged v5 baseline;
- exclude legacy Sentinel, ValueNet, and GapNet from the first run.

The checkpoint roles and interruption policy are recorded above. The remaining
decisions before a long run are its update algorithm, opponent schedule,
temperature start, game budget, seed, concurrency, checkpoint/monitor cadence,
and stop criteria. The local endgame/fullgame files also remain exploratory
unless separately validated and promoted.

Until the remaining launch choices are recorded, safe work consists of local
inspection, tests, and launch-contract review. It does not include an
additional smoke, a long training job, a push, or a history rewrite.

## Reference Material

- [`docs/endgame-training-feasibility.md`](../endgame-training-feasibility.md):
  read-only analysis of the reported 10/20 phase profile, the delivered
  checkpoint boundary, provisional local WDL coverage evidence, and questions
  for the original maintainer.
- [`docs/v5-specialist-plan.md`](../v5-specialist-plan.md): target
  architecture, evidence boundaries, and staged acceptance plan.
- [`docs/malom-fix.md`](../malom-fix.md): decoder investigation and correction
  background.
- [`docs/specialist-db-fix.md`](../specialist-db-fix.md): legacy SpecialistDB
  contamination background.
- Machine-local Sanmill checkout: independent TGF rules, search, and Perfect
  DB reference, with an existing NMM_LLM coordinate/HumanDB codec. See the
  Sanmill entry in the
  [`docs/local-training-layout.md`](../local-training-layout.md) path index;
  use only at a recorded commit and within the documented integration boundary.
- Machine-local `Notes.md` and screenshots: historical maintainer observations,
  not authoritative facts or acceptance evidence. See the reference-only
  `notes` entry in the
  [`docs/local-training-layout.md`](../local-training-layout.md) path list and
  apply the evidence boundary above.
- Machine-local `train_s_gen_v2_handoff_unfinished.py`: preserved, unfinished
  mixed-opponent draft; see the same local-layout entry and treat it as
  reference-only.
