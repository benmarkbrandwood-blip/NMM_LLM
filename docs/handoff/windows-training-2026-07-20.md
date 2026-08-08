# Windows Training Handover — 20 July 2026 (updated 8 August 2026)

## Executive Summary

The repository is usable on the Windows 11 host and the downloaded databases
and existing model artifacts are in their intended locations. The focused
Malom/provenance and current trainer-contract suites are green. The 7 August
complete run collected and executed 1,235 tests: 1,227 passed and eight
machine-local Sanmill bridge tests failed closed because the moving checkout
changed protected paths relative to their historical strict-v2 pin. The
historical binary bytes also remain unavailable. This is not a clean all-pass
claim; see the
[current complete-test baseline](../evidence/current-complete-test-baseline-2026-08-07.md).
The authorized corrected-v4 managed plan
`managed-v4-baseline-v1` completed 5,000 games in 20 verified segments on
21 July 2026 (UTC). Its completion is lineage and infrastructure evidence, not
playing-strength or promotion evidence. No further training run is authorized.

The separately authorized rules-corrected successor smoke
`successor-rules-v2-smoke-001` completed on 7 August from clean, published
`dev` commit `5cb44b1`. It used fresh weights, the isolated empty corrected DB,
the final MIF and rules identities, and explicit disable controls. Two counted
games produced one finite 14-step Adam update and a verified version-2
`latest.pt`; the lifecycle chain and post-run database audit pass. This is
infrastructure evidence only. The smoke authorization is consumed and no long
run is authorized. See the
[successor smoke result](../evidence/successor-training-smoke-result-2026-08-07.md).

On 8 August the product owner selected a separate fresh
`dev-v4-sanmill-refereed-fresh-v1` lineage after confirming that Sanmill must
participate in training, not merely in later evaluation. The new route starts
from random weights, keeps the completed local-GameAI run immutable, uses
Sanmill as the authoritative complete-history referee for every primary
rollout, and replaces local `GameAI` with fixed-node `go logical` search on
the non-frozen opponent stratum. Its exact ignored runtime is pinned to
Sanmill commit `a6623f88959f7453594df274fbe1f128af7ff55e`, tree
`17b9b0fd51ee8dac54c0454a6935978a47d19e0c`, binary SHA-256
`5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`,
and strict-referee semantic digest
`sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94`.
The initial contract disables retries, branches, observation-based recovery,
and the uncalibrated local-GameAI advancement gate. Process-level tests cover
cross-process search determinism, full logical turns with compulsory removal,
illegal-action rejection, and an eight-ply learner/opponent rollout whose
entire history is checked by Sanmill. This is implementation evidence only.
The separately authorized `smoke-001` then failed closed during its first
game: Sanmill correctly emitted a game-over FEN with raw action `?`, but the
NMM_LLM mirror called a projector limited to placing and moving states. No
counted game, optimiser update, checkpoint, or training log was produced.
Commit `4e734e4a3105b1a590fbb11ab13c3197cb6a9fce` repairs that terminal-only
projection boundary and adds the exact 43-logical-ply regression. The failed
output and its subsequently diagnostic-written SpecialistDB are quarantined;
the first one-run authorization is consumed. The separately authorized
`smoke-002` then ran from clean published commit `894360d`, after its final
preflight again returned `ready_for_smoke`. It completed both scheduled games,
one finite 47-step A2C update, a verified version-2 `latest.pt`, and a valid
completed lifecycle chain. Its isolated corrected database now contains 94
positions and one winning line and is retained as completed smoke evidence,
not a fresh input. The second one-run authorization is consumed. No retry or
long run is authorized. The node ladder, representative throughput envelope,
and later advancement rule remain unfrozen.
See the
[fresh Sanmill experiment contract](../experiments/dev-v4-sanmill-refereed-fresh-v1.md).
The terminal repair verification reports 182 trainer, launch, checkpoint,
resume, bridge, and referee tests passed with six documented historical
moving-checkout tests deselected, plus 103 Malom and label-provenance tests
with 498 parameterized subtests passed. Ruff and `git diff --check` also pass
for the changed scope. The
[failure record](../evidence/sanmill-refereed-fresh-v1-smoke-001-failure-2026-08-08.md)
owns the raw identities and diagnostic-side-effect boundary. A new full
`tests/` run collected 1,246 tests and reached approximately 16% with no
failure before the 15-minute command limit, so this work does not replace or
upgrade the separately recorded 7 August complete-suite baseline.
The
[smoke-002 result](../evidence/sanmill-refereed-fresh-v1-smoke-002-result-2026-08-08.md)
owns the successful launch identities and explicitly excludes its two games
from node-ladder, strength, and retained-run throughput claims.

The scoped Stage-0 evaluation `dev-v4-formal-paired-eval-v1` completed on
23 July 2026 with protocol decision **`accepted`**. Expert review had rejected
the 64-position corpus and synthetic
one-endpoint-per-named-line alternative and established that
`policy-argmax-v1` zeroes a lookahead feature block used during training. The
draw-lifecycle and partial-ledger restart defects found in the paired runner
are repaired; deterministic start reuse is now rejected; and new specifications
bind the clean Git commit, runtime, device, route, components, and feature
contract. The owner reviewed all 107 generated candidates, requested removal
of original review position 101, and accepted the remaining 106. The resulting
run was a 106-start, placement-only Stage-0 training-signal diagnostic against
scratch initialization. Its replacement corpus and PNG package were
regenerated and audited. A clean read-only audit reverified the corpus,
bundles, isolated
targets, runtime identity, and in-memory specification; 28 focused readiness
tests pass. On 23 July 2026, the product owner explicitly authorized the exact
CPU freeze and 212-game run. Independent recomputation verified 193 candidate
wins, 8 draws, 11 losses, pair-score-difference mean `0.8584905660377359`,
and interval `[0.7972174156720373, 0.9197637164034345]`. This is not a formal
strength or promotion result, and it does not authorize training, a rerun,
promotion, or publication.

Post-Stage-0 preparation is now complete up to new product choices. A strict
training-route bundle preserves the final policy, its six-game-old frozen
target, HumanDB continuation, final SpecialistDB counterfactual features,
corrected Malom early termination, and the historical rollout evaluator. A
real read-only load verified all four artifact identities. A separate
64-position draft covers placement/movement/side-to-move flying as 22/21/21,
is White/Black balanced 32/32, has no exact HumanDB or final SpecialistDB
matches, and includes 64 individual PNGs plus six inspected contact sheets.
The source is seeded legal TGF rules replay, not expert play. The corpus remains
unfrozen and unapproved; no post-Stage-0 candidate game has been run. The Mill
expert has now completed a quick first pass over all 64 panels, supplied a move
for each, identified several unlikely or poor states, judged the spread useful
overall, and proposed additional tactical Mill-choice positions. A product
freeze decision remains; the review is not a blanket acceptance or rejection.

Sanmill bridge v2 also passed on 25 July. It uses the pinned versioned strict
error, logical-turn, and `statejson` interfaces instead of the historical
assertion build. Two fresh processes reproduced the same 57-turn rule-terminal
game, and rule, action, history, aggregate-budget, and local performance probes
passed. This was infrastructure evidence only: no candidate was loaded and
formal candidate-versus-baseline evaluation remains stopped.

The maintainer's `main` history through `67af016` and the 21/22 July staged
upload were integrated and audited without activating their databases or
checkpoints. Later `main` history through fetched tip `bc46b51e` was reviewed
commit by commit on 7 August but was not merged or cherry-picked. The active
decision record is
[`main-integration-audit-2026-08-07.md`](../evidence/main-integration-audit-2026-08-07.md).
The rebuilt HumanDB has current label metadata and matched 30 deterministic
Malom probes; the rebuilt SpecialistDB has current metadata and zero Malom
labels but retains 2.1 million empirical positions. Seven updated checkpoints
remain weights-only maintainer-`main` artifacts with unknown corrected-data
lineage. The older v2a trainer fork is preserved but quarantined on `dev`, and
the imported in-place SpecialistDB clearing tool has been made non-destructive.

Read
[`docs/local-training-layout.md`](../local-training-layout.md) for the relative
storage map and machine-local lookup keys, and
[`docs/v5-specialist-plan.md`](../v5-specialist-plan.md) for the modular v5
design and its owning subdocuments.
Machine-specific absolute values are intentionally kept only in the ignored
`data/training_paths.local.json`. Path names shown in committed documents are
relative to the repository root; Markdown link targets are relative to their
containing files so that they render correctly.
The dated monolithic v5 snapshot is historical. The current v5 entry point
records that the coarse sector-corrected decoder is already repaired while the
complete comparator, rule-history, proof, and release questions remain open.

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
`53d86d1`, and the follow-up maintenance commits through `9c7dceb`. Later local
infrastructure commits through `59a4cf9` add exact-resume hardening, bounded
segments, checkpoint migration and validation, self-describing evaluation
bundles, paired promotion evidence, and the first author-asset refresh. Inspect
the live graph rather than relying on that intermediate snapshot. Later local
commits through `4893fb6` add fail-closed pure-RL controls,
deterministic fixed-node heuristic work with actual-node evidence, and
product-authorized managed training supervision. Inspect the live local and
remote graph before making synchronization claims. The completed
force-with-lease approval is not standing permission for a future push or
history rewrite; obtain fresh authorisation when such an operation becomes
necessary.

At the 21 July formal-evaluation review, local `dev` was at
`bc92d3346c8da55b6cdf1d56b20b7cab10317c75`, one commit ahead of
`origin/dev`, with modified and untracked experiment documents and draft
artifacts. That is not a clean reproducible evaluation freeze point. Recheck
the live graph and working tree before relying on this snapshot.

## 22 July Main Integration and Upload Audit

The maintainer's active `main` tip was initially `b9a13ce` and advanced during
verification to `67af016`. Its history was not compared to `dev` by a blind
tip diff: commit-graph inspection showed that `9d09851` was a one-parent
import close to older `dev` commit `0ad5991`, followed by the
maintainer's plans, assets, and v2a work. Merge commit `8717f1c` records the
integration. All seventeen snapshot conflicts retained the newer `dev` side;
the non-conflicting maintainer artifacts were preserved for audit.

Final merge commit `4593034` imports `67af016`'s v2a per-difficulty
best-rate persistence while retaining the quarantine. It fixes one legacy
best-save threshold across restarts but does not provide the complete `dev`
exact-resume state and therefore does not authorize running v2a.

Two independent safety commits follow that merge:

- `f7c5b19` makes SpecialistDB label clearing an explicit, source-hash-bound
  copy migration and adds three regression tests;
- `76f3ff3` quarantines the older main-lineage v2a runtime entry point and
  removes its unsafe smoke/resume examples while retaining the source for
  reviewed feature porting.

The rebuilt database candidates are intact under the ignored archive
`data/backups/maintainer_upload_20260721`. The former sibling `Mills`
directory was renamed `maintainer_inbox` on 25 July and left empty for future
incoming deliveries. The HumanDB sidecar hash matches, both post-move SQLite
quick checks pass, and 30 sampled HumanDB labels match the current corrected
Malom adapter for W/D/L and DTW. This supports the archived candidate but does
not replace the active HumanDB or change the completed baseline. The archived
SpecialistDB's retained empirical history also makes it a different
experiment input from the fresh baseline DB.

The imported retraining plan remains a proposal. Current code and the v5 design
resolve its apparent model-contract ambiguity locally: Sentinel stays DB-free
with oracle slots masked, the proposed next-move ranker is a separate
HumanPolicy path rather than ValueNet v2, and GapNet retains its implemented
current-position quality-gap target. The imported checkpoints still lack
corrected-data lineage, but no maintainer reconstruction is needed unless a
future experiment proposes to adopt them.
See
[`docs/evidence/main-integration-audit-2026-07-22.md`](../evidence/main-integration-audit-2026-07-22.md)
for exact hashes, counts, conflict policy, and question boundaries. The prior
draft message to the maintainer can wait; a shorter evidence-based question set
should be sent only after this integration audit is complete.

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
now-resolved test-collection failures. Do not call them missing project
dependencies without first defining a feature that requires them.

Commit `06598c9` records successful `cargo check --locked`, editable
installation of the CPython 3.13 extension, and fifteen native parity tests.
The extension was rebuilt after the fixed-node API change; an end-to-end probe
used exactly 25,000 requested nodes twice and selected the same move both
times. The full Rust unit suite reported `24 passed`.
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

`scripts/train_s_gen_v2.py --help` also completes successfully. Follow-up
maintenance on `dev` recalibrated two stale GameAI tactical fixtures against a
legal terminal-mill position and replaced tests that depended on an untracked
`data/games` corpus with deterministic JSONL fixtures. The current Sentinel and
TrajectoryDB loader tests therefore execute rather than skip when that local
directory is absent.

The four historical collection errors were stale tests rather than missing
active production interfaces. Commits `08507e0`, `c12b935`, `cc07a81`, and
`af17232` align, respectively, the legal-move, Sentinel feature-builder,
Sentinel label, and Sentinel model tests with the current contracts. A fresh
collection at `af17232` reports `925 tests collected`; the updated files and
their adjacent contract chains report `67 passed`. The mandatory
Malom/provenance rerun again reports `102 passed, 498 subtests passed`.

The first complete run made possible by those collection fixes reports:

```text
47 failed, 878 passed, 498 subtests passed in 3023.67s (0:50:23)
```

That result remains the first credible failing baseline. The failures were then
resolved against their owning contracts rather than skipped, weakened, or made
dependent on network downloads:

- `9e3bda7` gives Stage 3/6 a deterministic local embedding, explicit Chroma
  client lifetime, and a declared Chroma version with public close support;
- `3a57564` derives scaffolded lookahead width from the loaded model width and
  rejects incompatible explicit advisors before tensor execution;
- `39cf56e` preserves a training sample when ValueNet receives a singleton or
  very small dataset;
- `becfe17` and `c56f03a` assert the B-40 and SE-10 feature contributions
  directly instead of assuming no later tactical term also applies;
- `cee5e45` preserves the objective forced-dead-block label on the default V2
  path while continuing to suppress V1 score explanations;
- `901909a` replaces the B-78 pseudo-quiet fixture with an actually legal quiet
  full-game-DB move;
- `f13e9e9` gives the exact B-66 move regressions deterministic node budgets;
- `799a944` checks the B-22 defensive postcondition over all opponent replies;
- `c4c3454` isolates the documented contested-mill late-game contribution; and
- `08e8c33` compares WDL distributions by turn bit, the colour-symmetry
  invariant appropriate for side-to-move table values.

After the Chroma, scaffolded-width, and ValueNet changes, the intermediate
complete run had only the eight independently diagnosed tactical/endgame
failures: `8 failed, 919 passed, 498 subtests passed`. The affected non-solver
files then reported `97 passed`, and the complete 3v3 builder file reported
`13 passed` in 22 minutes 44 seconds.

A final clean-working-tree run at code HEAD `08e8c33` is the current complete
Python baseline:

```text
927 passed, 498 subtests passed in 3098.55s (0:51:38)
```

There are no remaining collection or runtime failures in that suite. This does
not change the separate experiment freeze, provenance, or training-authorization
gates.

This author-bundle review reran the current trainer contract, preflight,
checkpoint-envelope, exact-resume, launch, temperature, data-contract, and
paired-evaluation tests at code HEAD `59a4cf9`: `113 passed`. The mandatory
Malom/provenance group again reported `102 passed, 498 subtests passed`.
The earlier `tests/test_scaffolded_policy.py` result of `22 passed, 3 failed`
identified the real feature-width mismatch fixed by `3a57564`; it is historical,
not a current interface failure. The absence of a regression for PPO old/new
log-probability temperature consistency remains a separate coverage observation,
not a failing test.

## Data and Model State

Files from the Google Drive deliveries are no longer held in an ambiguous
sibling staging directory. Reviewed assets are in their repository-local or
external destinations, while the inactive 21 July database candidates are in
`data/backups/maintainer_upload_20260721`. The former `Mills` directory is now
the empty `maintainer_inbox` for future deliveries. Its role and the relative
destination map are recorded in the
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
- `data/specialist_db.sector_corrected.sqlite` is trusted completed-run state.
  It began empty, but the 5,000-game managed baseline populated it; do not
  describe or reuse it as an empty input for another fresh experiment.
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
untrusted. The active corrected SpecialistDB began empty. After the completed
managed run, a 22 July read-only audit found SHA-256
`1203FC73CD7D0A06E2DD1FFACED5B031DFF8BD704E22B34BA02182FF3865614D`,
SQLite `quick_check=ok`, 132,182 positions, 41,904 current-version Malom
labels, 916 winning lines, no preferred plays, and lineage root
`managed-v4-baseline-v1-segment-0001`.

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
The active corrected database was not replaced by that author update. The
recorded pre-run SHA-256
`CB4153A14752357587890EB5F8B655AB04AF8242E43BE1C80D4847A11D101A94`
was subsequently superseded by legitimate managed-run writes; its current
identity and counts are recorded above.

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

### Newly supplied author-`main` Generalist evidence

The owner confirms that the newly supplied Generalist checkpoints, JSONL logs,
plot, and browser screenshot all came from the maintainer's continuing `main`
training. They are not `dev` artefacts even though a legacy checkpoint embeds a
host directory containing the word `dev`. Exact hashes and the read-only audit
are recorded in
[`docs/evidence/author-main-generalist-audit-2026-07-20.md`](../evidence/author-main-generalist-audit-2026-07-20.md).

The delivered `best (copy).pt` is a finite, legacy weights-only `s_gen_v2`
checkpoint at game 17,400 and difficulty 9. That supports the maintainer's
correction from “10/20” to “9/20”, but its exact source commit and full launch
contract remain unknown. It has no optimiser, RNG, data identity, or complete
trainer state and must never initialise or resume the fresh `dev` experiment.

The accompanying log supports a narrower version of the maintainer's policy
observation. Across its first and last 500 rows, `policy_top1_rate` rises from
about 0.42 to 0.84 while entropy falls from about 1.55 to 0.34. However,
`heuristic_top1_rate` also rises, from about 0.30 to 0.51. These fields measure
whether the sampled move equals each argmax; they do not measure strength or
isolate positions where policy and heuristic disagree. The 10,547-row file
also contains duplicate game numbers, six counter regressions, and a mid-log
opponent-schedule change, so it is an appended operational history rather than
one frozen experiment.

The 1,190-row update log raises a separate stop condition for PPO reuse. Its
policy loss has median about `9.88e7`, reaches about `1.71e29`, and ends around
`7.80e21`, while value loss remains ordinary and all values remain finite. The
inspected trainer family records old PPO log probabilities from
temperature-scaled logits but recomputes new log probabilities without that
temperature. The missing exact `main` commit prevents attributing every spike
to that mismatch, but PPO remains quarantined for the first `dev` baseline
until a deterministic ratio test and reviewed fix exist.

The latest browser screenshot proves only that the Generalist checkbox was
selected during one manual game. It does not freeze the actual feature inputs,
opponent, position, colours, or work budget. The author log has only
`phase_bucket=main`, so the reported strong opening and weak endgame profile
still requires a phase-stratified replay before it can guide architecture.

The author-update SpecialistDB does contain 27 promoted preferred plays, which
supports the narrow “favourite plays” statement. It still lacks a `meta` table
and a trusted Malom label version, so it remains quarantined and read-only. The
maintainer also explicitly said the internal endgame files had not been
checked, consistent with keeping them disabled as authoritative inputs.

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

The historical smoke's `latest.pt` is a pre-envelope weights-continuation
snapshot; `best.pt` remains optional model-selection evidence. Subsequent
infrastructure now emits a version-2 checkpoint envelope and has proved bounded
exact-resume parity for model, optimiser, scheduler/scaler, counters, rolling
histories, curriculum, target state, component RNGs, data cursor, log state,
and SpecialistDB identity. Initial launch still uses explicit `fresh` mode.
Unscoped automatic resume remains forbidden. Within one separately authorized
immutable managed plan, the supervisor may start a new isolated segment only
from the verified `latest.pt` of the immediately preceding completed segment,
using explicit `exact-resume`. Legacy checkpoints, including every
author-`main` file, remain weights-only and cannot satisfy that gate.

## Managed Run and Stage-0 Evaluation Completion

The separately authorized managed plan `managed-v4-baseline-v1` later
completed `completed_games=5000` and `completed_segments=20`. Its frozen
training commit is `9ee3543195255456b2b3832f8371a8f64d25a6af`, and its plan
SHA-256 is
`3f696e60c508a972dc42c79f630e90ad20e870001190321a13f0c3a12a4251c1`.
The final candidate source is
`managed_v4_baseline_v1/segments/segment-0020/latest.pt`. The candidate and
architecture-matched scratch-init evaluation bundles have both passed CPU
verification.

The paired-runner prerequisites identified by expert review are repaired.
Repetition and 50-move transitions now stop on `engine.finished` and retain the
engine's draw reason. In-progress evidence is fsynced to `<output>.partial`;
same-spec ordered hash-valid prefixes resume only missing games, malformed
prefixes fail closed, and complete evidence is recomputed before atomic final
publication. The specification also rejects duplicate starts and any pair
count above the unique corpus size. New freeze records bind a clean Git commit,
selected CPU/CUDA device, platform, PyTorch, float32, the policy route,
disabled components, and zeroed lookahead features; execution fails closed on
runtime drift. Legacy unbound specifications remain readable and recomputable
but cannot create new game evidence. `python -m pytest
tests/test_paired_evaluation.py -q` reports `15 passed`.

The first formal paired-evaluation proposal was narrowed because:

- pure argmax plus modulo start selection makes repeated starts exact copies,
  invalidating the old 64-start / 256-pair nominal sample size;
- 49 of 107 named lines have 2–42 legal endpoints because removal choices are
  omitted, one line fails replay, and one successful endpoint is terminal;
- 110 raw Sanmill Oracle keys contain 108 stable placement keys that project
  to 107 unique playable NMM positions; the other two are pending removals and
  are retained only as successor provenance;
- the proposed `policy-argmax-v1` route zeroes the 72-feature lookahead block
  supplied during training.

The completed Stage-0 diagnostic used 106 owner-accepted unique stable
Oracle-projected starts, one colour-swapped pair per start, for 212 games
against the verified scratch-init control. Sanmill documents the Oracle as
independently engine-derived, but 28 of 106 selected positions overlap
named-line trajectories and all positions are early placement. It is neither
demonstrated held out nor training-disjoint. Stage 0 therefore tests only
whether a training signal is
visible under a placement-only feature ablation; it is not a strength or
promotion gate.

The generated freeze-compatible list has canonical `start_positions_sha256`
`04bc5782ab79ebeba34d0ff91bcd40fe05e823d539b16ba234b5eedcd123bb9d`.
The review artifact's pre-freeze status was
`owner_review_complete_not_frozen`; the selected list is now frozen by spec
identity `26f80c14d70320aa025c85319791c625e821babb2e542095aeb4711d4c11d48b`.
It links 106 individual PNGs and nine contact sheets. Automated replay found 438 legal
source recommendations and one illegal `c3` recommendation. The associated
source candidate is the owner-excluded original position 101 and remains only
as provenance. Codex inspected every regenerated contact sheet, the
post-exclusion boundary images, and the excluded panel. The combined
corpus/evaluation focused suite reports `28 passed`; owner review and read-only
readiness verification are complete.

The controlling records are:

- [evaluation contract](../experiments/dev-v4-formal-paired-eval-v1.md)
- [expert decision record](../experiments/dev-v4-formal-paired-eval-v1-decision-brief.md)
- [rejected corpus and generated replacement review](../experiments/dev-v4-formal-paired-eval-v1-corpus-review.md)
- [Stage-0 readiness evidence](../evidence/dev-v4-stage0-readiness-2026-07-22.md)
- [Stage-0 result evidence](../evidence/dev-v4-stage0-result-2026-07-23.md)

The authorized run is complete and its one-run authorization is consumed. Its
`accepted` decision means only that a training signal is visible against
random initialization under the placement-only zero-lookahead ablation. Any
route-aligned or phase-covered v2 must be separately preregistered and
authorized; observations may not be pooled as one prespecified sample.

The next-evaluation preparation records are:

- [training-aligned product decision brief](../experiments/dev-v4-training-aligned-evaluation-v1-decision-brief.md)
- [phase-corpus review record](../experiments/dev-v4-phase-covered-corpus-v1-review.md)
- [phase-corpus artifact](../experiments/dev-v4-phase-covered-corpus-v1.json)
- [complete Sanmill book-path contract](../experiments/sanmill-book-path-corpus-v1.md)
- [complete Sanmill book-path artifact](../experiments/sanmill-book-path-corpus-v1.json)
- [Sanmill prefix-diversity audit](../evidence/sanmill-prefix-diversity-audit-2026-07-25.md)
- [twelve-ply layered-prefix contract](../experiments/sanmill-layered-opening-prefix-v2.md)
- [twelve-ply corpus decision brief](../experiments/sanmill-layered-opening-prefix-v2-decision-brief.md)

## Strict Sanmill Bridge Validation

The product owner deferred the current in-repository `GameAI` as the formal
baseline because its compact lifecycle does not preserve authoritative
repetition and no-capture history. The assertion-build bridge v1 passed on
23 July and remains immutable historical evidence. Sanmill then supplied
explicit `StrictFailurePolicy`, `go logical nodes N`, and `statejson`
interfaces. The current bridge v2 is pinned to Sanmill commit
`db65eb3e73189d934d615d0f47519d395193c646`, tree
`b8fa6c0119c2dec4443efc59deab8b7d835e0c88`, and ordinary Windows release
binary SHA-256
`cac2ec6fe45a9d798a89c6b8a5f52c767aa1c885a1156a96269b44ebf81976cc`.
It did not load the candidate or run candidate-versus-baseline games.

The v2 bridge uses one thread, MTD(f), IDS, shuffling off, seed 42, fixed node
ceilings, and no wall-clock limit. `StrictFailurePolicy=true` makes rejected
histories and search failures versioned hard errors. The logical-turn path
never enters Perfect DB, patch/trap, depth-4, or random recovery. HumanDB,
Perfect DB, patches, traps, lazy AI, and `FocusOnBlockingPaths` remain inactive
in bridge search. Normal smoke turns send no positive explicit depth, so
Sanmill's non-developer `DrawOnHumanExperience` phase-depth policy remains
active.

`statejson` now supplies the authoritative FEN, complete action and logical-ply
counts, no-capture and repetition counters, legal actions, terminal reason,
standard-rule identity, and history SHA-256. `go logical nodes N` returns an
ordinary action or a Mill-forming action plus its required removal under one
aggregate ceiling. The command does not mutate engine state: NMM_LLM replays
the returned tokens and requires the resulting FEN, counts, history identity,
and outcome to match.

The first full v2 invocation stopped before writing evidence because the
adapter treated a closed terminal snapshot that retained `action=remove` as an
ongoing pending-removal contradiction. A deterministic turn-57 reproduction
proved that the completed removal had already changed the phase to
`game_over`, cleared pending removal and legal actions, and produced the
authoritative `loseFewerThanThree` result. Commit `70de75b` permits only that
terminal combination; the ongoing-state check remains strict. The focused
suite then reported 41 passed, readiness was repeated, and the recorded smoke
passed.

Two fresh processes produced the same 57 complete logical turns, 65 UCI action
tokens, eight removals, and final White win after timing was excluded. Their
semantic identity is
`ae51a16b726e7227f499f054310fed5fbd4b158d8f1b998a4d8cb65d1f7c27bc`.
Black-box probes passed for the 100-ply no-capture draw, threefold repetition,
fewer-than-three loss, compound Mill/removal, capture reset, and
`DrawOnHumanExperience` opening depth.

At a 500,000-node ceiling, the representative movement sample used 500,000
nodes in about 59.7 ms and flying used 500,000 in about 36.2 ms. Placement
completed depth 3 after 1,080 nodes and about 0.15 ms. The explicit depth-8
compound Mill probe used 11,776 nodes in about 13.8 ms. These are single-host
observations, not a frozen formal workload or latency guarantee.

The v2 evidence identity is
`b8e31cb621e95ecdf5708145c3c4c3ba43b0fbae863bd93460db1beba96cd188`.
It is bound to NMM_LLM source commit
`70de75bb8247ec6795b69045ac53558161e6c045`, the exact pinned Sanmill source
and binary, rule identity, strict contract, and corrected opening-book asset.
The complete repository suite was not rerun for this bridge-only update; the
41-test focused result does not replace the prior full-suite baseline.

On 7 August, Sanmill local commit
`a6623f88959f7453594df274fbe1f128af7ff55e` added an opt-in
`mif-stable-moving-v1` strict-referee profile. NMM_LLM-side source inspection
and a fresh black-box process confirmed origin-counted occurrences at logical
plies 0/4/8 and fail-closed rejection after the action-8 terminal draw. The
default live profile is unchanged. The implementation was not yet pushed when
reviewed, so formal referee adoption still requires remote publication, a
clean pinned release build, and a repeated bridge audit. See the
[strict-referee parity record](../evidence/sanmill-strict-referee-parity-2026-08-07.md).

Current and historical bridge records are:

- [v2 human-readable result](../evidence/sanmill-strict-uci-bridge-smoke-v2-2026-07-25.md)
- [v2 machine-readable result](../evidence/sanmill-strict-uci-bridge-smoke-v2-2026-07-25.json)
- [v2 contract](../experiments/sanmill-strict-uci-bridge-smoke-v2.md)
- [historical v1 result](../evidence/sanmill-strict-uci-bridge-smoke-2026-07-23.md)
- [historical v1 contract](../experiments/sanmill-strict-uci-bridge-smoke-v1.md)

The opening-book data defect is closed. Sanmill commit
`69d379a1a4e23395a45706df60f63282da20e85f` removed the occupied-`c3`
recommendation and added authoritative whole-asset legality tests. Commit
`6f080c5a6d15919bf0a45fa5528c45d4487a2b8f` removed a duplicate `c5`
recommendation that otherwise altered rank-biased selection weight. The final
asset SHA-256 is
`cdc4768bc461c22177634985a4cc1d92452774e2992515b937fed8812eb076f5`;
all 109 entries and 437 unique recommendations replay legally, with zero
duplicates. These corrections and the later strict/data-query/logical-turn
interfaces are now present on Sanmill `master`.

The missing-provider-interface and NMM_LLM paired-prefix implementation
blockers are closed. Commits `a4e166e` and `d6ea9f5` provide a strict JSONL
client and a deterministic, source-policy-explicit sampler. They verify every
source identity, FEN, history SHA-256, action count, logical count, and stable
primary-plus-removal boundary. One prefix is recorded for both games in a
colour-swapped pair. The focused Sanmill bridge/query/prefix suites report
`60 passed`; the complete repository suite at `d6ea9f5` reports
`1022 passed` and `498 subtests passed` in 3306.78 seconds.

The provisional infrastructure design still proposes 75% corrected-book
prefixes and 25% StrictSteps Perfect DB tied-best prefixes, but the code has no
default mixture. Prefixes cover exactly eight logical player moves in total:
four by each side, or four full rounds, not eight rounds. A Mill-forming move
plus its required removal is one logical move even though it uses two UCI
tokens. Both games replay the same frozen-seed prefix before strict MTD(f)
resumes. No optional database is enabled inside later MTD(f) search.

The corrected book is sparse. A fixed `pair-12` diagnostic generated the same
eight-ply book prefix in two fresh processes, but `pair-0` reached
`book_miss` before its sixth logical move. No fallback was used. Before a
75/25 smoke, use either an eligible book pair-ID set, a frozen complete-path
corpus, or an explicit per-ply source schedule. Do not interpret a book miss
as permission to switch sources at runtime.

The complete-path option is now implemented as inventory-only infrastructure.
Contract commit `8edb148` and implementation commit `024d1f8` produced a
host-path-free artifact from two fresh Sanmill processes. Both enumerations
contained the same 192 complete eight-logical-ply histories, 508 pruned
`book_miss` leaves, and zero fallback or pre-depth terminal leaves. The
artifact corpus identity is
`3bc9bc05a66a1a53255444266388838489020667272fc2ffa7445e7cf44be985`;
its file SHA-256 is
`490537d892e4dc64b0b46331754bab448a3b3d99dad620131cb692916e540ceb`.
The 192 histories end in only 84 distinct FENs. Therefore the inventory does
not silently define uniform path, final-position, or source-rank weighting,
nor does it freeze the proposed book/Perfect DB proportions.

The later source-only diversity audit projects those 84 FENs through the
book's own `ring16` symmetry and finds only seven endpoint orbits. In contrast,
two fresh processes generated the same first 64 fixed-seed StrictSteps
prefixes, comprising 64 unique exact FENs and 64 unique `ring16` orbits with
zero book-orbit overlap. Audit identity
`a7bc734ad3f85d2ae3ab75c901467da7b1835932fefa9aadd6067e1f4a982990`
records this pre-result evidence. The 75/25 proposal is therefore not
recommended as a diversity-first policy. The non-frozen technical
recommendation for 64 opening prefixes is one representative from each of the
seven book orbits plus 57 orbit-unique StrictSteps prefixes. Book-style
exposure would be a different named objective.

The configured Perfect DB returned 24 StrictSteps-tied initial candidates and
reported complete standard-sector coverage. The active HumanDB currently
fails closed with `database_not_immutable` because its SQLite `-shm` sidecar
is non-empty; no sidecar was changed. Whether HumanDB becomes evidence only or
a third source remains unfrozen.

The preceding paragraphs are immutable eight-ply v1 history. They do not
define the current opening-prefix design. On 25 July the product owner and
Mill expert selected a separate twelve-logical-ply v2 contract with Book,
HumanDB, and Perfect DB as independent strata. The eight-ply 7/57 proposal
remains historical evidence and must not be relabelled as a twelve-ply corpus.

The v2 source-only audits are now complete:

- the Oracle Book query graph supplies zero pure twelve-ply routes, while 84
  of 107 named variations supply 112 capture-resolved records and 110 unique
  histories/orbits;
- an online SQLite backup preserved the active HumanDB sidecars and bound a
  92,939-game PlayOK sample containing 83,002 exact histories, of which 5,174
  have at least two-game support; and
- two fresh Sanmill processes generated byte-identical evidence for 128
  StrictSteps routes, all unique by exact history, final FEN, and ring16 orbit,
  with zero overlap against Book or HumanDB.

The maintainer Openings delivery is also archived by identity. Its two Book
files duplicate tracked assets; its 15 learned additions remain an independent
candidate pool, not a formal stratum.

The later 35-row `Book Opening Plays.docx` delivery is separately archived and
audited as `maintainer_expert_curated_play`, a Book subtype rather than a
fourth reporting stratum. Its 36 explicit records all replay legally in two
fresh pinned Sanmill processes and reduce to 34 histories, 33 final FENs, 32
final ring16 orbits, and 14 eight-ply parent orbits. The expert-confirmed
row-19 correction has since been replayed through a separate reviewed-source
lineage, producing 35 histories, 34 final FENs, and 33 final ring16 orbits
without changing the 14 eight-ply parent orbits. One parent supplies 16
children. Six exact histories occur in 29 distinct frozen PlayOK games; no
exact history matches the Sanmill named lines, although nine unique final
FENs and orbits overlap. Row 11's final `c5` is screenshot-derived and remains
explicitly identified as visual evidence, but the original-resolution move
panel is unambiguous and it is not an expert-confirmation blocker.

The product owner accepted the near-balanced
`22 Book / 21 HumanDB / 21 Perfect DB` counts on 1 August; the
[composition decision](../experiments/sanmill-layered-opening-prefix-v2-composition-decision-2026-08-01.md)
freezes the ratio while leaving membership pending. The expert review
supplement provides substantial family and
child semantics, establishes that P03 needs extended-family splitting, and
identifies P13-A/P13-B as a symmetry pair. It does not provide a standalone
P14 classification, complete priority tiers, a final P03 partition, or every
primary-child choice. A 31 July direct-message follow-up now closes the P14
name as `Interrupted Knight`, labels the P03 children, distinguishes the outer
and inner Parallel Mill Rush variants, and selects P03 child 001 as the primary
Black response. The frozen audit independently proves that children 001/006
are different-history same-endpoint transpositions and that child 012 exactly
duplicates 006. On 1 August the expert clarified the selection rule: retain
every unique placement pattern, do not spend another place on a different
route to the same placement, and let the project arrange representatives. The
[semantic disposition](../evidence/maintainer-book-opening-plays-semantic-review-2026-07-26.md)
and
[short follow-up](../experiments/sanmill-layered-expert-book-review-follow-up.md)
separate that expert statement from project D4 normalization. The historical
[shortlist](../experiments/sanmill-layered-expert-book-shortlist-proposal-2026-07-31.md)
records the correction sheet he reviewed. The subsequent
[coverage decision](../experiments/sanmill-layered-expert-book-coverage-decision-2026-08-01.md)
freezes 33 D4-unique representatives as the complete Expert Book catalogue and
diagnostic membership while preserving all 36 raw records as provenance. Its
execution contract is not frozen. Status is
`executable_64_prefix_corpus_frozen_evaluation_not_authorized` for the
balanced 64-prefix core, and
the subsequent
[Book core decision](../experiments/sanmill-layered-opening-prefix-v2-book-core-2026-08-01.md)
freezes 15 expert-curated and seven Sanmill named-line members. The 22 records
cover all 14 expert parent orbits and all seven Sanmill declared families and
are unique by exact history, final FEN, and `ring16` orbit. The subsequent
[HumanDB core decision](../experiments/sanmill-layered-opening-prefix-v2-human-core-2026-08-01.md)
freezes 21 genuine PlayOK histories. Frequency-ordered selection reaches
ledger rank 31 after ten `ring16` skips; selected support ranges from 16 to 27
distinct games. The subsequent
[Perfect DB core decision](../experiments/sanmill-layered-opening-prefix-v2-perfect-core-2026-08-01.md)
freezes audit routes 000 through 020. All 64 source members are now unique by
exact history, final FEN, and `ring16`. The
[source core decision](../experiments/sanmill-layered-opening-prefix-v2-source-core-2026-08-01.md)
freezes their combined identity. The deterministic
[review package](../experiments/sanmill-layered-opening-prefix-v2-source-core-review-2026-08-01.md)
contains all 64 individual panels and six visually inspected contact sheets;
its manifest identity is
`db37224db6e400a32df9275e5e0665647541c4aa589b327b4317235e2eb27fba`.
The later
[HumanDB execution overlay](../experiments/sanmill-layered-opening-prefix-v2-human-execution-2026-08-01.md)
freezes all 21 missing per-step records. The review images remain visual
membership material rather than execution authority.

The two untracked interchange-format drafts that temporarily dirtied the
referenced Sanmill checkout were removed by its workspace owner. The pinned
installation audit and 60 Sanmill UCI/data-query/prefix regressions then passed
again. No file was imported from the other workspace. The active HumanDB
sidecar remains a separate unresolved local-data condition.

The reference Sanmill checkout has since advanced through protected CLI and
rules paths and its release binary no longer matches the historical strict-v2
identity. The old binary bytes are unavailable. A clean isolated checkout at
the same pinned commit and tree was therefore built and frozen under the
separate
[prefix12 replay runtime decision](../experiments/sanmill-prefix12-human-replay-runtime-2026-08-01.md).
It is resolved through `sanmill_prefix12_checkout` and has release-binary
SHA-256
`6502f7a2180769666c1ba6c801288a5ba079920e2bd6c1121f0e8b0c27e11e53`.
This new identity does not replace the historical smoke evidence and is
authorised only for strict source-history replay.

That runtime replay is now complete. Two fresh processes produced exactly
equal ordered transcripts for all 21 HumanDB histories: 273 requests and 273
responses per process, covering 252 logical plies and 13 compound
Mill-and-removal turns. The transcript identity is
`e61bef7940fb1dd9a6fffb67b98640825d72a0ebcfb105627fdaa871173c13fd`;
the frozen HumanDB execution identity is
`1cf88ab8b3afb7c62112a0f2866eed9052587bbf2ef44dc57efa64c2749021d6`.
All 64 source-member execution records now exist. The subsequent
[executable-corpus decision](../experiments/sanmill-layered-opening-prefix-v2-executable-corpus-2026-08-01.md)
assembles them under identity
`417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`.
It preserves the 43-record historical and 21-record exact-HEAD Sanmill binary
identities separately. Evaluation and training remain unauthorised.
The five focused corpus checks pass. The wider layered-prefix run reports 81
passes after explicitly deselecting the known historical Perfect DB local
regeneration check; running that check against the advanced moving checkout
correctly fails closed on protected-path drift.

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

The newly supplied browser evidence exposes a second route mismatch. The
trainer constructs its Generalist lookahead with the configured Malom database
as `endgame_db`, but `load_generalist()` does not pass an endgame database and
`GeneralistAgent.score_moves()` still calls the encoder with `db=None`. Although
the browser calls `set_db()` after loading Malom, that score path never consumes
the stored reference. Conversely, the browser constructs the Generalist with
globally loaded Sentinel, ValueNet, GapNet, HumanDB, and SpecialistDB objects;
unchecked UI boxes are not an auditable component-disable contract for those
features.

The candidate side can now be reconstructed as the separately named
`s-gen-v2-training-aligned-v1` route. The route bundle contains both final and
frozen-target weights and binds component flags, HumanDB, final SpecialistDB,
Malom, and fixed depth. Required resources are opened read-only and failures
propagate. The route deliberately retains the trainer's historical
empty-square comparison bug; silently correcting it would define another
experiment. A new paired protocol and competent baseline are not yet frozen,
so this preparation is not execution authority. Every future evaluation must
still emit checkpoint, route, component, data, Malom, and fixed-work
identities.

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

## MIF 1.0 Independent Interoperability Adapter

The NMM_LLM adapter is locked to MIF commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978` and implements all seven
`MIF-INTEROP/1` operations: capabilities, MFEN/MPK canonicalization, finite
rules execution, checkpoint-verifying replay, full-state transform, complete
logical-turn projection, and the `legal-actions-v1` harness projection. It is
intentionally independent of the MIF Python reference runner; the latter is
used only as a separate black-box comparison process.

The adapter's honest runtime claim is narrower than every possible
`mif-finite-rules-v3` variant. It advertises the two frozen candidate corpus
rulesets, accepts patches inside their implemented semantic subset, and fails
closed outside that subset rather than approximating unsupported capture
mechanisms, semantic-state extensions, mill effects, or stalemate policies
with the legacy NMM_LLM board engine. The following candidate records are
historical inputs to the later immutable Suite release; none is a `full` or
conversion claim.

Candidate-3 gameplay is implemented by NMM_LLM commits `748dae2`, `feb4646`,
and `121b663`. Candidate-4 changes only the locked source and corpus identities
in NMM_LLM commit `bbbde2ee4bf1ba0e45e259baa595a29cb85895b9`; it does not alter
the independent state machine. The existing implementation matched all 58
candidate-4 cases before the pin update, including the three new
asymmetric-reserve origin cases. At the clean pin commit, 55 focused tests and
Ruff pass. The generator accepts the clean MIF checkout and all seven frozen
hashes, while MIFCAP publishes the 17-case smoke and 58-case deterministic
corpus identities without claiming a Suite. The exact pin, hashes, scope, and
host-local command-array generator are in
[`docs/interop/mif-1.0-independent-adapter.md`](../interop/mif-1.0-independent-adapter.md).

The raw
[candidate-4 report](../evidence/mif-interop-candidate-4-nmm-reference-report-2026-08-06.json)
has SHA-256
`89dfcd97c914764aa95bcb5e6b6ecdb23686591037dbf8c5493fe8b3dfbc142f`
and records 58/58 equality between the published MIF reference and NMM_LLM at
clean commit `bbbde2ee4bf1ba0e45e259baa595a29cb85895b9`.

Sanmill subsequently published candidate-4 commit
`e6d639d41f079b15ca697268d0c2c21dad5c2bc3` and the tracked three-party
report `interop/evidence/mif-interop-candidate-4-three-project-report-2026-08-06.json`.
That report has SHA-256
`895c04cd69fc00e50bdcd349b150293e52fcc4150c63321d8c9771015f70aaaf`
and records 58/58 with cases digest
`sha256:d11317a090300f8a47f77afed647bdbd236dcdb1996c0147a81c874fa39dfd82`
and config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
An independent rerun using MIF `7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`,
Sanmill `e6d639d41f079b15ca697268d0c2c21dad5c2bc3`, and NMM_LLM
`11bebd14e0d538a41a4b43aebfe57ee74c2a2601` reproduced the same report
hash.

Sanmill closed the M3 evidence-chain gap at evidence commit
`9431b95f151502f415f096c7d96ca944e5d578de`. Its companion manifest binds the
three published commits, all seven candidate-4 inputs, and the 58/58 report
identity above. M3 is closed for those exact candidate identities.

Candidate-4 M4 uses MIF launch commit
`40718e80d36ec9c060fc17997568d637a74e6d9f` over the unchanged wire commit
`7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`. The fixed launch SHA-256 is
`560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e`;
the reference-baseline SHA-256 is
`29d198dbcf8221fa0235af6a72db9d6a82646b45fc653c584071821a9a4bb61b`.
The prescribed pre-fix run at NMM_LLM
`e2ab05d29885af9a16a9aa5d5f62b1517cf6d91b` reproduced 10/10 seeded runs
and 3/5 mutations at config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.

Tested NMM_LLM implementation commit
`6c1538082fc551203d827782d137a5799c810535` aligns only the two failing
diagnostic shapes; it does not change gameplay or replay semantics. The
[two-party raw report](../evidence/mif-interop-candidate-4-m4-reference-nmm-report-2026-08-07.json)
has SHA-256
`2bc434699902a1c468b604797d4456ee0c968817b057ec4dc8254a623a1ba64c`,
records 10/10 seeded runs and 5/5 mutations, and has config digest
`sha256:c6eb5edc21773c017e7a2d5d9050b38cb08450658a286e64a395f1edc6b7074e`.
The adjacent
[companion manifest](../evidence/mif-interop-candidate-4-m4-reference-nmm-evidence-manifest-2026-08-07.json)
binds the exact MIF and NMM_LLM commits plus Sanmill implementation
`ae9a1d8a16261478631a3a7583cbf35c7b6e0df5`, evidence
`9431b95f151502f415f096c7d96ca944e5d578de`, and its two-party report hash.
The final three-party preflight records 10/10 and 5/5 with config digest
`sha256:4184d56c696b2e5031d95cc18918757af9f91fa5634dded579ecfae2ef3cf70f`.
This closes NMM_LLM's Candidate-4 M4 differential evidence only. No result
claims MIF Suite 1.0 conformance. Earlier candidate reports remain historical
evidence for their recorded identities.

An additional 1,138-test repository run was attempted without skips, but the
15-minute command limit stopped it at roughly 15% with no failure reported.
It is not a full-suite pass and must not be presented as one.

The later Suite-finalization gate supersedes the pre-Suite capability status
without changing the wire implementation. MIF commit
`3ee7e57c7d4c7208be91f62914f344a587fb0f70` fixes Suite raw SHA-256
`088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4` and
JCS SHA-256
`81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`.
NMM_LLM implementation commit
`a7e7dbd5461cc2d8d8c0a09317d6091598202214` publishes the exact Suite pin
and marks only the six required classes and two Suite rulesets as tested. It
continues to declare no `full` class and `conversion=none`.

The final Reference/NMM_LLM evidence records 58/58 deterministic cases, 10/10
seeded differential runs, 5/5 mutation families, and zero unexplained
differences. The capability, deterministic-report, and differential-report raw
SHA-256 values are respectively
`cd661b1156bf7269f976e050446d01797c9959482f1e1843e21ae3ea7f70dcce`,
`3463f438531fd52847df44fa4186dcba13ed22c7c570a0cc216d9a7eaa797665`,
and `4c86725bfcd1759433374938c8d8eb2a1dacfa6ea3723592eff759162fce8da6`.
The
[Suite evidence manifest](../evidence/mif-suite-1.0-nmm-adapter-evidence-2026-08-07.json)
and [interop record](../interop/mif-1.0-independent-adapter.md) own the complete
identity and scope details. A current three-adapter preflight also passed
58/58, 10/10, and 5/5.

The 66 focused MIF tests and static checks pass. A four-shard run exercised all
1,179 repository tests: 1,170 passed initially, the sole Windows Chroma cleanup
failure passed alone, and eight machine-local Sanmill tests remained
fail-closed because the historical strict-v2 binary bytes are unavailable.
This is not a clean full-repository pass and must not be represented as one.
The Suite claim is only `exact-for-tested-domain`; it is not `full`
conformance and makes no conversion claim.

A later clean single-process run at published `dev` commit `f06d457` collected
and executed all 1,235 current tests without skips. It reported `1,227 passed`,
`8 failed`, and `498 subtests passed` in 2,876.61 seconds. A focused rerun
proved that all eight failures stop at the same protected-source-path check
before any Sanmill query, search, replay, or gameplay assertion. No NMM_LLM
trainer, MIF, Malom, checkpoint, exact-resume, or managed-run test failed. The
full evidence and claim boundary are in the
[7 August complete-test baseline](../evidence/current-complete-test-baseline-2026-08-07.md).

MIF Suite 1.0 is now immutably published as tag `mif-suite-1.0` at release
commit `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`. Suite JCS SHA-256 remains
`81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`;
final evidence SHA-256 is
`2c23983281858386bc66e3adfce52f365c712d9e63a31c53f6a68bd6b2de08e1`;
release-manifest SHA-256 is
`dde89416bf5251cdc445ebdb9b92a899f58ec3930d1d8077ae26f1cb1a084499`.
Training manifests now bind those identities, ruleset
`nmm-training-core@2`, semantic digest
`52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`,
and an independent experiment digest. The remaining long-run gates are
experiment-specific rather than MIF publication gates.

## Recommended Next Actions

The rules-corrected successor smoke is complete and passed. Preserve its
manifest, lifecycle ledger, logs, checkpoint, database, and recorded hashes as
one disposable evidence bundle. Do not exact-resume it or use its populated DB
as a fresh long-run input. Before proposing a long run, execute the current
complete test suite, create a separate empty corrected DB and output root, and
freeze a new immutable plan with an evidence-backed wall-time envelope. The
two-game smoke is not a representative throughput benchmark, and no long-run
authorization exists.

The complete-suite gate has now been executed and recorded. The current
[successor decision brief](../experiments/dev-v4-rules-corrected-successor-v2-decision-brief.md)
recommends a fresh 5,000-game objective, `max_ply=120`, 250-game segments, and
a 12-active-hour limit. Historical segment timing and partial per-game logs
support that conservative bound; the two-game smoke remains excluded as a
throughput benchmark. The objective and resource envelope still require an
explicit product decision, followed by exact plan publication and a separate
launch authorization.

The workspace/root check, graph inspection, earlier trainer fixes, focused
tests, mandatory Malom/provenance rerun, first-experiment component decision,
bounded smoke, managed-plan hardening, and the 5,000-game managed run are
complete. The owner-reviewed 106-position package is committed, its focused
checks pass, and the complete Python baseline is also clean. Do not launch more
training merely because the managed run ended. Proceed in this order:

1. Preserve the completed plan, ledgers, segment checkpoints, candidate bundle,
   and scratch-init bundle under their recorded identities.
2. Keep both rebuilt database candidates archived and inactive, and keep every
   imported checkpoint out of the `dev` resume lineage. Ask for additional
   checkpoint lineage only if a future experiment proposes to adopt one; use
   the locally resolved Sentinel, ValueNet/HumanPolicy, and GapNet boundaries
   recorded in the retraining plan.
3. Preserve the completed owner review: original review position 101 is
   excluded, the other 106 are accepted, and the withdrawn concern about 83 is
   not a corpus defect.
4. Preserve the completed Stage-0 spec and final ledger together under their
   recorded hashes; the one-run authorization is consumed.
5. Preserve the `accepted` result as ablation-only training-signal evidence.
   Do not rerun it or treat acceptance as promotion evidence.
6. Preserve both bridge generations, treating the v2 logical-turn result as
   current and v1 as historical. Keep `GameAI` deferred as formal referee. The
   60-turn ceiling was smoke-only, and no candidate-versus-baseline authority
   was consumed.
7. Keep the Sanmill book, data-query, strict-error, logical-turn, and state
   commits pinned by identity even though they are now on Sanmill `master`.
   Do not silently float to later CLI or rule changes.
8. Preserve the eight-ply implementation and 7/57 evidence as v1 history.
   For current work, use the twelve-ply v2 contract and completed Sanmill Book,
   expert Book, HumanDB, and Perfect DB audits. The HumanDB immutable snapshot
   was created without deleting active sidecars. Preserve the accepted
   `22 Book / 21 HumanDB / 21 Perfect DB` balanced-core split. Its 64 source
   members, review manifest, source-member execution records, and combined
   executable corpus are now frozen. Preserve executable-corpus identity
   `417d74ebe01734c43e48531cab81ba742bc89e455f1c834ea7e31006b886f8b9`.
   The expert gate is closed: preserve the frozen
   [33-pattern coverage catalogue](../experiments/sanmill-layered-expert-book-coverage-decision-2026-08-01.md)
   and keep the separately frozen balanced-core subset without changing that
   catalogue. Row 11's screenshot-derived `c5` is
   source-verified and
   retains a visual provenance marker. There is no runtime source fallback.
9. Record the Mill expert's completed first-pass review of all 64 panels. He
   supplied a plausible move for each, marked several unlikely or poor states,
   described the overall spread as useful, and suggested adding positions
   where closing a Mill competes with blocking or enabling a chain Mill. This
   refers to the separate phase-coverage draft, not the already frozen
   22/21/21 twelve-ply opening corpus. It is domain feedback, not an automatic
   phase-corpus freeze. Any later tactical stratum must remain separately
   identified.
10. After the prefix-policy and corpus decisions close, freeze the formal
   fixed-node ceiling, history-bearing start representation, accepted starts,
   pair count, rules-compliant termination contract, and interval rule. Then
   implement and audit the formal runner and request launch separately. Do not
   pool any result with Stage 0.

The previously executed isolated smoke command was:

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
remain ignored and separate from the intended long-run paths. It is historical
evidence, not a current launch command: the hardened CLI now requires
`--launch` and `--run-id`, and the reviewed command must state its start mode
explicitly. It also predates an explicit imitation-mix disable control. The
experiment document records its verified contents and the checkpoint
observation.

The original handover's 50,000-game PPO command should not be launched
unchanged. PPO and the more complex opponent mixture are optional experiments
under the v5 plan, not prerequisites for a corrected baseline.

## Recorded and Remaining Owner Decisions

The following choices are recorded for the first `dev` experiment:

- start from random model weights, not a historical checkpoint;
- use the corrected v4-style Generalist path, not claim the staged v5 baseline;
- exclude legacy Sentinel, ValueNet, and GapNet from the first run.

The following choices are recorded for the next formal-evaluation design:

- defer the current `GameAI` and validate pinned Sanmill as rule/history owner;
- use one thread, shuffling off, a fixed seed, and fixed node ceilings rather
  than a wall-clock search limit;
- retain Sanmill's normal `DrawOnHumanExperience` opening-depth behavior;
- retain the corrected Sanmill opening book as one future opening source, but
  keep book play off until a deterministic interface and paired policy pass;
- keep HumanDB, patches, and traps off; keep the perfect database out of MTD(f)
  search while allowing a separately audited tied-best prefix sampler;
- treat the 75% book / 25% StrictSteps perfect-database mix and eight logical
  player moves as a historical provisional smoke design, not a frozen formal
  decision; the later eight-ply 7/57 diversity proposal is also historical;
- use twelve logical plies and separately report Book, genuine HumanDB, and
  StrictSteps Perfect DB strata for the current design; the 22/21/21
  composition, 64 source-member identities, HumanDB strict replays, and
  executable corpus are frozen, while evaluation remains unauthorised;
- preserve all 33 D4-unique expert-curated Book placement patterns in a
  separately reported diagnostic catalogue while retaining all 36 source
  records as provenance; its execution protocol is not yet frozen;
- use 60 complete turns only as a bridge/performance smoke ceiling; it is not a
  rules draw or a formal match-length decision; and
- do not run candidate-versus-baseline games until a later immutable contract
  and explicit launch authorization exist.

The product owner delegated routine technical choices for the authorized
managed baseline to the Agent. The resulting immutable plan used A2C, no
imitation warm-start or mixing, 50/50 frozen/heuristic opponents, 500,000
native nodes per heuristic move, full depth-5 rollout, temperature `0.90` to
`0.20`, 5,000 games, seed 42, single-game batching, and 250-game exact-resume
segments. That plan and its authorization are complete historical contracts;
they are not authority for another run. Its `max_ply=60` cap was truncation,
not a rules draw, and must not be copied into a successor plan without a new
decision. The active ruleset has a 100-movement-logical-ply no-progress draw,
so a full-game cap must exceed the placement phase plus that window if the
rule is to be observable.

The product owner should be asked only about the objective, total game or
wall-time envelope, launch, later resource expansion, and publication or
promotion. Technical failures remain Agent diagnosis. The local
endgame/fullgame files also remain exploratory unless separately validated and
promoted.

The managed plan and its Stage-0 evaluation are complete. The evaluation's
one-run authorization is consumed. Safe work now includes inspection,
documentation, preservation, and implementation after the product choices in
the next-evaluation decision brief. It does not include an additional
evaluation, smoke or long training job, promotion/publication, or history
rewrite without the applicable separate authorization.

## Reference Material

- [`docs/endgame-training-feasibility.md`](../endgame-training-feasibility.md):
  read-only analysis of the corrected 9/20 phase observation, supplied
  author-`main` bundle, Generalist runtime route, provisional local WDL
  coverage evidence, and remaining questions for the original maintainer.
- [`docs/evidence/author-main-generalist-audit-2026-07-20.md`](../evidence/author-main-generalist-audit-2026-07-20.md):
  hashes and reproducible diagnostic findings for the newly supplied
  author-`main` checkpoints, logs, screenshots, and related database claims.
- [`docs/evidence/main-integration-audit-2026-07-22.md`](../evidence/main-integration-audit-2026-07-22.md):
  commit-graph-aware `main` integration, staged rebuilt-database validation,
  updated checkpoint identities, v2a boundary, and remaining maintainer
  confirmations.
- [`docs/retrain_v2_plan.md`](../retrain_v2_plan.md): maintainer proposal for
  Sentinel, ValueNet, and GapNet v2 work; useful design input but not a frozen
  or authorized run contract.
- [`docs/v5-specialist-plan.md`](../v5-specialist-plan.md): modular v5 entry
  point, evidence boundary, feasibility-first route, and links to the owning
  oracle, training, human-data, runtime, release, and governance
  specifications.
- [`docs/managed-training-operations.md`](../managed-training-operations.md):
  durable Agent/product authority boundary, managed contracts, commands,
  status model, and stop policy.
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
