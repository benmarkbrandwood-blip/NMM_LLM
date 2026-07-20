# Windows Training Handover - 20 July 2026

## Executive Summary

The repository is now usable on the Windows 11 host, the downloaded databases
and existing model artefacts are in their intended locations, and the focused
Malom/provenance suite is green. A long corrected training run has not been
started.

The next safe objective is not simply to repeat the old training command. It is
to preserve the corrected data boundary, fix or explicitly avoid the
generalist trainer's hard-coded auto-resume path, correct and test its
temperature schedule, choose whether the next run starts from scratch or
accepts a documented legacy warm start, run a bounded smoke test, and only then
start a monitored run.

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
documentation commit will make local `dev` ahead until that new commit is
separately pushed. The completed force-with-lease approval is not standing
permission for a future push or history rewrite; inspect the graph again and
obtain fresh authorisation when such an operation becomes necessary.

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
collection found 696 tests and stopped on four repository-interface errors:

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

## Data and Model State

The Google Drive delivery referenced by the original handover has been moved
out of the import staging directory and into the intended repository-local or
external locations. The staging directory is now empty. Its host-specific
location and the relative destination map are recorded in the
[`docs/local-training-layout.md`](../local-training-layout.md) path list.

Available assets include:

- the 738,091,008-byte HumanDB and 94,983 human-game `.jsonl` files;
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
- The old SpecialistDB is isolated in the ignored backup directory and must
  remain read-only.
- Historical checkpoints and nets pre-date the corrected decoder/provenance
  migration. Retain them as exploratory baselines; do not claim that they were
  trained from corrected labels.
- The original maintainer describes the endgame tables and `fullgame.bin` as
  outputs of their backwards solver. That is a provenance statement, not an
  independent correctness check. Record hashes and run reader/differential
  validation before using those files as authoritative labels or acceptance
  evidence.
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

## Known Generalist Trainer Risks

### Auto-resume ignores the configured output directory

The machine-specific configuration sends new output to:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected
```

However, `_choose_resume_path()` in `scripts/train_s_gen_v2.py` currently
handles `--auto-resume-best` by hard-coding:

```text
learned_ai/checkpoints/scaffolded/s_gen_v2/best.pt
```

It does not use `args.out_dir`. Consequently, running the original handover
command with `--auto-resume-best` would silently load the historical pre-fix
generalist even though the output directory is named `sector_corrected`.

Before a long run, either:

1. fix auto-resume so it resolves `best.pt` inside the configured output
   directory and add a focused regression test; or
2. omit `--auto-resume-best` and provide an explicit `--resume` path only after
   the checkpoint-lineage decision has been recorded.

For a genuinely fresh corrected run, omit both options. This issue should be
fixed as a separate commit before relying on automatic continuation.

### Temperature options and recovery reheat do not control the loop

The original note proposes increasing temperature after repeated recovery and
uses `--temp-start 1.1`. The current script first assigns
`temperature = args.temp_start`, but the start of every training-loop iteration
replaces it with `_compute_temperature(game_count, args.max_games)`. That
function uses the module constants `TEMP_START=0.90` and `TEMP_END=0.20`, not
the command-line value. Recovery later assigns `temperature = TEMP_START`, but
the next loop iteration replaces that assignment before another rollout.

Consequently, the CLI start value and recovery assignment do not implement the
requested rollout schedule. Before a long run, define the intended annealing
and decide whether recovery reheat is desirable at all. Implement the approved
state as checkpoint-resumable behaviour, or remove the ineffective assignment,
and add focused tests covering the CLI value, ordinary decay, approved recovery
behaviour, and resumption. Until then, do not describe recovery as increasing
exploration.

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

Proceed in this order:

1. Confirm that the current or newly opened Codex task remains at the
   repository root with `git rev-parse --show-toplevel`.
2. Inspect `git status` and the local/remote graph. The earlier divergence is
   resolved; do not repeat the force-with-lease operation or push the
   documentation commit without explicit approval.
3. Fix the `--auto-resume-best`/configured-output mismatch in one focused
   commit, with a regression test.
4. Fix and test the CLI temperature schedule in a separate focused commit.
   Decide explicitly whether recovery should reheat; implement the approved
   behaviour or remove the ineffective reset. Do not combine this work with
   mixed-opponent experimentation.
5. Add or run focused checks for every Malom-enabled inference route intended
   for the next baseline, then re-run the 102-test Malom/provenance suite.
6. Record an explicit lineage and component choice: fresh corrected model or
   documented legacy warm start, and which legacy Sentinel/value-net inputs are
   enabled as inputs or ablations. The safer model-lineage default is fresh.
7. Run one bounded smoke game using separate smoke output and SpecialistDB
   paths. Confirm that the selected Malom, HumanDB, legacy model inputs, GPU,
   log writer, checkpoint writer, and versioned SpecialistDB initialise, and
   record the exact decision route exercised.
8. Inspect the smoke log and database metadata. If an endgame/fullgame asset is
   enabled, validate its manifest and reader separately. Only then choose
   long-run concurrency, monitoring intervals, and stop criteria.

A suitable fresh smoke command, after the auto-resume review, is:

```powershell
.\.venv\Scripts\python.exe scripts\train_s_gen_v2.py `
  --out-dir learned_ai\checkpoints\smoke\s_gen_v2 `
  --specialist-db data\specialist_db.smoke.sqlite `
  --max-games 1 `
  --batch-games 1 `
  --max-ply 40 `
  --sim-ply-depth 2 `
  --minimal-rollouts `
  --no-s1a-warmstart `
  --no-s1b-refresher
```

Do not add `--auto-resume-best` to this command. This is an integration smoke
test, not strength evidence or a test of recovery reheating. Keep its output
separate from the intended long-run directory.

The original handover's 50,000-game PPO command should not be launched
unchanged. PPO and the more complex opponent mixture are optional experiments
under the v5 plan, not prerequisites for a corrected baseline.

## Open Decisions for the Owner

The next task should seek explicit decisions on these points when they become
actionable:

1. Must the corrected generalist start from scratch, or may a historical
   checkpoint be used as a clearly labelled warm start?
2. Is the immediate target a bounded continuation of the existing v4
   generalist, or the stricter staged v5 reference baseline?
3. Should the first baseline load the legacy Sentinel and value net as explicit
   inputs/ablations, or exclude them until corrected retraining is available?
4. Should mixed-opponent work be revisited only after the corrected baseline is
   measured?
5. Will the local endgame/fullgame files remain exploratory, or be promoted to
   an authoritative role only after independent validation?

Until those choices are recorded, safe work consists of local inspection,
tests, the two focused trainer fixes, and a separate bounded smoke run. It does
not include a long training job, a push, or a history rewrite.

## Reference Material

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
