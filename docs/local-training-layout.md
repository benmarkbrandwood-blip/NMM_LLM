# Local Windows Training Layout

## Purpose

This document records the intended storage boundary for the Windows 11
training machine. It separates source control, machine-specific configuration,
large databases, imported source material, and recoverable staging data.

All committed paths in this document are relative to the repository root (the
directory containing `AGENTS.md`) unless the text names a configuration key
instead. Machine-specific absolute values, including cross-volume paths, belong
only in the ignored `data/training_paths.local.json`; query that file locally
instead of copying its values into committed documentation.

The primary Codex workspace and sole Git root is:

```text
.
```

The current Codex task is already open in the directory containing `AGENTS.md`.
Future tasks should use the same root. Do not open its parent (`..`) as the
primary workspace for repository work, and do not create an outer Git
repository merely to describe its children.

## Sibling Directories

| Path | Role | Version-control rule |
| --- | --- | --- |
| `.` | Application, training code, tests, plans, and selected shared model artefacts | The only Git repository |
| `../NMM_DB` | External Malom tablebase | Never add to Git |
| `../human_database` | Human-game source archives and database-building source material | Never add to Git |
| `../opening_book` | Source opening-book material | Keep outside Git unless deliberately imported as reviewed source data |
| `../notes` | Original handover (`Notes.md`), its images, the unfinished archived trainer copy, and the 20 July author-`main` diagnostic bundle | Historical reference only; independently verify claims, never use these files as `dev` resume/input evidence, and do not execute or merge the draft blindly |
| `../maintainer_inbox` | Temporary incoming directory for files received from the `main` maintainer | Keep new deliveries in dated subdirectories until their identity and purpose are recorded; never use inbox contents directly as active training inputs |
| `../.cargo-target` | Optional external Rust build cache | Currently empty; not project source |

The empty `../.git` and `../.agents` directories are Codex workspace
placeholders, not an initialised repository. A real `git init` directory would
contain files such as `HEAD`, `config`, and `objects`. They may reappear whilst
an older Codex task still uses the parent as its workspace root. The current
task is already rooted correctly; after all older tasks are closed, the empty
placeholders may be removed if desired.

## External Reference Checkout Index

External checkouts are read-only reference inputs. They are not additional
workspace roots, runtime dependencies, or sources of authoritative labels.

### Sanmill

- Local lookup: read `sanmill_checkout` from the ignored
  `data/training_paths.local.json`. This is a documentation/reference lookup
  key, not a trainer input.
- Observed revision on 22 July 2026: branch `next`, commit
  `ab9cccb9da65c0d784b982f532e7d1cedc8bea19`, two commits ahead of
  `origin/next`.
- Observed worktree state: `play_area.dart` and its human-database statistics
  test have unrelated local modifications. The NMM opening-book asset used
  below is tracked, clean, and byte-identical to the reviewed blob at commit
  `6a64010aed7ea4193502ea17c242f68e09fe576a`. The Oracle corpus builder reads
  that pinned Git blob rather than requiring this reference checkout to remain
  at the historical commit. Preserve all unrelated changes and do not alter
  this checkout from an NMM_LLM task.
- Licence: AGPL-3.0-or-later. NMM_LLM is also AGPL-3.0, but copied code must
  still retain source attribution and licence provenance.
- API stability: the relevant crates are version `0.1.0`. At the observed
  revision, `docs/FRAMEWORK_API.md` still illustrates a 256-byte opaque
  payload while `tgf-core` defines 320 bytes. Compile adapters against the
  pinned commit and assert boundary sizes instead of relying on the prose
  example.

Useful paths relative to the Sanmill checkout root are:

- `crates/tgf-mill/src/human_db_codec.rs`: already defines NMM_LLM's exact
  24-point order and converts one combined move-plus-capture turn into TGF's
  staged base and removal actions;
- `src/ui/flutter_app/assets/opening_books/nmm/opening_book.json`: pinned
  ring16 named-line and Oracle source for the Stage-0 evaluation-corpus review;
- `crates/tgf-mill/src/rules/` and
  `crates/tgf-mill/testdata/legacy_oracle/`: independent rule, history, and
  regression references;
- `crates/perfect-db/src/database.rs`, `wdl_plane.rs`, and `mill.rs`: Rust
  tablebase loading, sector correction, symmetry handling, and public move
  queries;
- `crates/perfect-db/csrc/perfect_wrappers.h` and `perfect_player.cpp`: the
  complete legacy value comparator, perspective conversion, and move-value
  behaviour;
- `crates/tgf-cli`: a headless UCI-like process surface suitable for bounded
  differential and opponent experiments.

The preferred first integration is a test-only process or small pinned Rust
adapter, not a replacement for `native/nmm_core`. Compare settled NMM_LLM turns:
a move that forms a Mill and its following TGF removal must be combined before
state, terminal, or Malom comparisons. A bare NMM_LLM FEN omits repetition and
no-progress history, so the adapter protocol must carry those counters and
signatures explicitly. TGF's high-level `PerfectOutcome` and
`PerfectMoveOrdering` collapse the ultra-strong ordering among draws, so v5
oracle work must retain `DatabaseEval` raw and sector fields and use the full
verified comparator. Record the Sanmill commit in every differential report or
generated evidence set. Project rules and independently tested NMM_LLM
semantics remain authoritative.

Fixed-node heuristic training follows the Sanmill-aligned quiescence and
candidate-set principles in
[`docs/fixed-node-heuristic-search.md`](fixed-node-heuristic-search.md).

### Mill Interchange Format

- Local lookup: read `mif_checkout` from the ignored
  `data/training_paths.local.json`. It is used only by interoperability tools
  and is not a trainer input or runtime dependency.
- Immutable release identity: tag `mif-suite-1.0`, release commit
  `a0a0f21cff5d6fbde045cd1482e416b92e0dc45a`, Suite JCS SHA-256
  `81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`,
  final evidence SHA-256
  `2c23983281858386bc66e3adfce52f365c712d9e63a31c53f6a68bd6b2de08e1`,
  and release-manifest SHA-256
  `dde89416bf5251cdc445ebdb9b92a899f58ec3930d1d8077ae26f1cb1a084499`.
  Every new training run persists these values as evidence identity; MIF is
  not imported as trainer gameplay code.
- Active trainer rules identity: `data/rulesets/nmm-training-core@2.json`,
  semantic digest
  `52f6ad24a0b95f68c1a7fd6b35b52550abce48c36d1686d155e497cdcad31f6a`
  and document digest
  `1dfdf5777f36866a53a942c1addd21857d3b72eede8ea2bf4fe1beedfbe878f2`.
  The rollout cap remains experiment metadata and is not part of this MRS.
- Frozen wire-semantics input: commit
  `7e45d5a3fa970a535ed6a8a8ff5981aba4b9c978`. The current adapter-finalization
  evidence was generated at the clean Suite-candidate commit
  `3ee7e57c7d4c7208be91f62914f344a587fb0f70`, Suite JCS SHA-256
  `81a5feabc281bfc4f830addabc2c6846d1f191bbbcf04e548f04b35dd358ae6f`,
  and Suite raw SHA-256
  `088ca33234289b06d9276aa4c430758222aa85d61621dee7bef4bfc6dcc069a4`.
  These are historical evidence inputs wrapped by the immutable release
  above. Do not run comparison cases against a floating MIF checkout and then
  attribute the result to any pin. The command generator rejects worktree
  changes and verifies the wire artefacts, Suite, differential launch, release
  manifest, and licence identities recorded in the interoperability document.
- Candidate-4 M4 harness input: commit
  `40718e80d36ec9c060fc17997568d637a74e6d9f`, with launch SHA-256
  `560ef369fde248bd96d3468a4336442db1d970ede04f488821509e69925fd48e`
  and reference-baseline SHA-256
  `29d198dbcf8221fa0235af6a72db9d6a82646b45fc653c584071821a9a4bb61b`.
  This later checkout adds the black-box differential harness and does not
  replace or silently alter the frozen adapter wire identity above.
- NMM_LLM independently implements the wire semantics under
  `learned_ai/interop/mif_v1/`; the MIF Python reference runner may be launched
  only as a separate black-box comparison process. Its gameplay code is not a
  library dependency and must not be copied into this repository.
- The durable contract, source hashes, scope limits and machine-local
  three-party command generator are documented in
  [`docs/interop/mif-1.0-independent-adapter.md`](interop/mif-1.0-independent-adapter.md).

## Repository-local Data Inventory

The base inventory was measured on 20 July 2026. Rows explicitly dated 21 or
22 July supersede that start-of-run snapshot.

| Asset | Current location and state |
| --- | --- |
| HumanDB | `data/human_db.sqlite`, 738,091,008 bytes; 94,429 games, 2,152,889 positions, and 2,516,356 move rows |
| Twelve-ply HumanDB audit snapshot | Ignored point-in-time SQLite online backup under `data/backups/prefix12_human_db_20260725`; 738,091,008 bytes, SHA-256 `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`, `quick_check=ok`; resolve the database, source manifest, and complete-history ledger through the local path registry |
| Archived rebuilt HumanDB candidate | `data/backups/maintainer_upload_20260721/human_db.sqlite`, 745,385,984 bytes; versioned candidate only, not active |
| Archived maintainer Openings delivery | Exact ignored retention copy under `data/backups/maintainer_openings_20260725`; the two Book files duplicate tracked assets, while the 184-record learned file contains 15 unmerged `seed_source=learned` candidates; see the delivery evidence |
| Archived maintainer Book Opening Plays deliveries | Exact ignored DOCX retention copies under `data/backups/maintainer_book_opening_plays_20260726`; the original is 3,432,474 bytes with SHA-256 `227584cde9d8c6278665a1b6decac6491d6b30b9b7add44a4b00200aec5e83c7`, while the 15-page expert-review supplement is 3,434,996 bytes with SHA-256 `9ef34e0a984d63167a5db526e87e3849ec2752b05cf7a3ed27adfa932fcf9ad8`; the original tracked transcription remains immutable and the confirmed row-19 correction is applied only through the separately identified reviewed-source audit |
| Pinned Sanmill prefix-replay runtime | Isolated clean checkout resolved through `sanmill_prefix12_checkout`; commit `db65eb3e73189d934d615d0f47519d395193c646`, release binary SHA-256 `6502f7a2180769666c1ba6c801288a5ba079920e2bd6c1121f0e8b0c27e11e53`; source-only HumanDB replay runtime, not the moving reference checkout or historical smoke-v2 binary |
| Pinned Sanmill training runtime | Ignored isolated clean checkout under `data/runtimes`, resolved through `sanmill_training_checkout`; commit `a6623f88959f7453594df274fbe1f128af7ff55e`, tree `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`, release binary SHA-256 `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` and size 5,641,216 bytes; authoritative referee and fixed-node opponent only for the fresh Sanmill lineage |
| Human game files | `data/human_games`, 95,389 `.jsonl` files plus import metadata; the 20 July author update added 406 files and raised `imported.json` from 94,134 to 94,540 entries |
| Human game source archive | `../human_database/human_games_94559.zip`, 121,796,279 bytes; SHA-256 `45523234085518031A09725A2DBCAB395E55026787E420A04C37EBA10A0E4D07` |
| Corrected SpecialistDB | `data/specialist_db.sector_corrected.sqlite`; after the completed managed run it is 17,268,736 bytes with 132,182 positions, 41,904 Malom labels, 916 winning lines, no preferred plays, and current metadata |
| Historical mill-bonus SpecialistDB template | Ignored main file at `data/specialist_db.mill_bonus_ablation_v1.template.sqlite`; 45,056 bytes, SHA-256 `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`, `quick_check=ok`, `malom_label_version=sector-corrected-v1`, zero positions, winning lines and preferred plays. As observed on 13 August it has an empty WAL and a 32,768-byte SHM sidecar. Preserve those sidecars; do not use this path as a new closed-source claim. |
| No-refresh retained-v4 attempt-002 SpecialistDB snapshot | Ignored closed snapshot at `data/specialist_db.no_refresh_retained_v4.attempt_002.template.sqlite`; copied byte-for-byte from the closed, unlaunched attempt-001 runtime DB. It is 45,056 bytes with SHA-256 `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`, `quick_check=ok`, `malom_label_version=sector-corrected-v1`, zero positions, winning lines and preferred plays, and no WAL, SHM or journal sidecar. Its one runtime copy was consumed by failed attempt 002; preserve this snapshot as source evidence. |
| No-refresh retained-v4 attempt-002 failed runtime DB | Ignored consumed database at `data/specialist_db.sanmill_no_refresh_retained_v4.seed70.attempt_002.sqlite`. The failed first rollout changed it to 73,728 bytes, SHA-256 `5acb8251fe601f2708082632520ec9d462ca3140563ec0cdc4b2e8001f0f5a0c`, with 188 positions, one winning line, zero preferred plays and no sidecars. Preserve it with the failed control directory; never use it as a template, resume input or successor database. |
| No-refresh retained-v4 attempt-003 SpecialistDB snapshot | Ignored closed snapshot at `data/specialist_db.no_refresh_retained_v4.attempt_003.template.sqlite`; copied byte-for-byte from the still-empty attempt-002 template, not from its failed runtime DB. It remains 45,056 bytes with SHA-256 `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d`, `quick_check=ok`, `malom_label_version=sector-corrected-v1`, zero positions, winning lines and preferred plays, and no WAL, SHM or journal sidecar. Its one copy into the attempt-003 writable path was consumed by the completed run; preserve the snapshot as input evidence. |
| No-refresh retained-v4 attempt-003 completed runtime DB | Ignored consumed database at `data/specialist_db.sanmill_no_refresh_retained_v4.seed70.attempt_003.sqlite`; 32,600,064-byte main file with SHA-256 `3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed`, `quick_check=ok`, `malom_label_version=sector-corrected-v1`, 242,006 positions, 4,185 winning lines and zero preferred plays. A post-completion reader created a zero-byte WAL and 32,768-byte SHM at 15:44 local time without changing the main-file identity. Preserve the main file and sidecars with the completed control directory; do not delete the sidecars, reuse the DB as a fresh input or relabel it sidecar-free. |
| Retained-v3/v4 diagnostic SpecialistDB snapshots | The v3 input is the existing ignored `data/specialist_db.sanmill_preserving_retained_v3.seed58.audit_snapshot.sqlite`: 22,188,032 bytes, SHA-256 `82d7fbcd897be2493ee40b40a44aa7cd941c95ff538b4f9bf21e2977cd4a8abe`, `quick_check=ok`, sidecar-free and byte-identical to the original main file whose empty WAL and 32,768-byte SHM remain preserved. The v4 input is the ignored `learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-passivity-diagnostic-v1/v4-specialist-db-snapshot.sqlite`: 32,600,064 bytes, SHA-256 `3d69d1acb007dbd26a48ae1c6acec4bb29f905ffedd21c816ad1771a6cf942ed`, `quick_check=ok`, sidecar-free and copied once from the completed attempt-003 main file after confirming its WAL is empty. Both strict route probes preserved absent WAL, SHM and journal files. Use these paths read-only only for plan `035c68f8`; do not treat them as new training lineage or templates. |
| Retained-v3/v4 passivity diagnostic route bundles and result | Ignored CPU-verified bundles under `learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-passivity-diagnostic-v1/v3-route-bundle` and `v4-route-bundle`, exported from the exact final `latest.pt` plus each segment-0020 `run-manifest.json`. Bundle identities are `b6d7ecf62ea9aeba893eff51e794d9307c444f361f54c9e1e832ac5b5d7bc5a0` and `817d2e36fbd0b614c5c48737ee987f684b99eb6ff697591618123ec7307a2d0f`. The same ignored root now contains completed spec identity `d47c94e2`, a 256-row `games.jsonl` with SHA-256 `c064f29d77cedd42a9ef405ec44dbbda045b47be31092e952568cecb5d49b562`, result identity `d250f03d`, completion identity `fe1f243c`, and identity-bound safe-progress and oracle-order zero-game reports. Preserve the entire namespace and both bundles read-only; the corpus is reused development evidence, the authorization is consumed, and none of these files is a new training lineage or held-out template. |
| Retained-v3/v4 phase-process source corpus | Tracked source-only artifact `docs/experiments/sanmill-retained-v3-v4-phase-process-corpus-v1.json`, corpus identity `3be3d76c34511e0f78d0f5bfe4a338c415c393306a955538bb85823e9d62c080`, file SHA-256 `8353ff3e52465bf99f7cf468a9cbcb4681a673ac2cebcdae00c253df8a22670b`. It contains 39 candidate-blind, strict-current-referee-nonterminal phase histories after excluding the earlier 12-start development corpus and three pre-start threefold terminals. It has zero exact/`ring16` overlap with the completed diagnostic openings and zero D4 start-state hits in HumanDB or either candidate SpecialistDB. It authorizes no game. The successor-owned route/database snapshots are listed below; the plan-`035c68f8` paths above remain plan-owned. |
| Retained-v3/v4 phase-process successor inputs | Ignored read-only copies under `learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-phase-process-generalization-v1/inputs`, snapshot identity `b35ecc061e53a35e227c69ff886a7c6534e707bd124abdbe13acbbf9647f48ac`, canonical manifest SHA-256 `cda9456e0234a9532ddfb1b90e3a78bb6a35ef788c0eddfca607e9f33cb1942a`. The v3/v4 bundle identities remain `b6d7ecf6...` / `817d2e36...`; route file-list identities are `97c6413a...` / `f701206d...`; sidecar-free SpecialistDB SHA-256 values remain `82d7fbcd...` / `3d69d1ac...`. These are successor-owned evaluation inputs, not new training lineages. Preserve the completed plan namespace separately; no game or launch authority follows from these copies. |
| Retained-v3/v4 phase-process frozen plan | Tracked canonical plan `docs/experiments/sanmill-retained-v3-v4-phase-process-generalization-v1.json`, plan identity `4c85ff3362927db9b63014e0c91022a5d169d19efa4aa85b3a643febd0ce3256`, file SHA-256 `09245e5f66af3d18ba2818d1dfac70b4c7eec8d63c9388d501b32846dfccf9d3`, implementation commit `5a318a063b561b12bafe5e72e44ff6fdc9426f1e`, plan commit `117a5be8086af04ba0b311f44a23cdc9804a7284`, stable source readiness `0ff79e398233c7ed9fcdec4cc5cd406837330140a3c1cec720e11eaa274ae365`. It binds 39 starts / 156 games / two active hours, a relative 108-ply start-clustered process endpoint, and zero-new-game mechanism reanalysis. It supplied no authority by itself; direct authorization was later consumed by the completed run below. |
| Retained-v3/v4 phase-process completed result | Ignored runtime under `learned_ai/checkpoints/evaluation/sanmill-retained-v3-v4-phase-process-generalization-v1`: authorization identity `ceedd13ae9abcce8b8f9e5103488057114408e887e5bd946b95041fc781faebb`, spec `bb349a96df3e8445d3687c7c24dc474fe595d63aa890085ed6c6b2a94574fe72`, launch `827820192f3e694d374b94c918f6410404437410715587dcd788d951bb5e4dc3`, 156-row ledger SHA-256 `45506e5cedf5ab9bdcba9dd687349869b639fb8bd46fd8990cbaf4bb79ef3211`, result `6007af186b9a7ce908416f4578ebc31c0c19fc27733c32ed44751bb39cc3c812`, mechanism result `afcdff218b8dfd47bb17c3f0438a9e3fd9e1298547a9c205e1949c5e26c97562`, and completion `48ac2ad4c6abc79b69c7de597ad46a5197949b4dcdee1f962e621d1be2fc57c8`. All games reached strict rules terminals, zero hit the safety cap, and the primary process decision is `inconclusive`. Preserve the complete namespace read-only; do not rerun, resume, extend, relabel held out, or treat it as strength, refresh-causal or equivalence evidence. |
| Archived rebuilt SpecialistDB candidate | `data/backups/maintainer_upload_20260721/specialist_db.sqlite`, 290,820,096 bytes; versioned empirical-history candidate only, not active |
| Legacy SpecialistDB snapshots | Two ignored, read-only snapshots under `data/backups/drive_import_20260720`; neither is an active training database |
| Endgame databases | `data/endgame`, fourteen `.wdl` files plus `fullgame.bin` at 571,683,560 bytes |
| Malom tablebase | `../NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted`; 512 files and 83,582,223,577 bytes |
| Sentinel | `learned_ai/sentinel/checkpoints/best.pt` |
| Generalist checkpoints | `learned_ai/checkpoints/scaffolded/s_gen_v2/best.pt` and `best1.pt` through `best6.pt`; `main` updated `best.pt` and `best6.pt`, but both remain maintainer-`main` weights-only history |
| Author `main` diagnostics | `../notes/best (copy).pt`, `best6.pt`, `train_log.jsonl`, and `update_log.jsonl`; reference-only and not part of the `dev` checkpoint lineage |
| Specialist checkpoints | Opening: two; midgame: four; endgame: two, all under `learned_ai/checkpoints/scaffolded` |
| Value nets | `data/value_net.npz` and the tracked human, phase, and trajectory variants |
| Gap-net artefacts | `data/gap_net.npz` and `data/gap_net_training.npz`; present but disabled in the local training path configuration pending provenance review |

The checkpoint and net files listed above exist. The important limitation is
their lineage: they pre-date the sector-decoder and persisted-label migration,
so they are exploratory baselines rather than evidence of a corrected training
run.

### Author `main` diagnostic bundle

The owner confirms that the four newly supplied Generalist files under
`../notes` came from the maintainer's continuing `main` training, not `dev`.
Their inventory is:

| Asset | SHA-256 | Legacy metadata or state |
| --- | --- | --- |
| `../notes/best (copy).pt` | `335462EC3A503E316EAAEF63A7669F1A725FC488A2C27E29B39EFD0021B804D6` | `s_gen_v2`, game `17400`, difficulty `9`; weights-only |
| `../notes/best6.pt` | `0E024D4402160BEFA4A7DDEDB56735FCA8CC9D924FC069A550F1761E643CA93D` | `s_gen_v2`, game `12750`, difficulty `6`; weights-only |
| `../notes/train_log.jsonl` | `FE332D39E9CA92552EF79A493C175208DAD4168ADA9E484EDEF1A630C58B250B` | 10,547 valid JSON rows spanning appended/restarted histories |
| `../notes/update_log.jsonl` | `41D68B45FFB31F7B6207AC7FD58EF906F5FAE626C56A971933473F4CC25FE03B` | 1,190 valid JSON rows; diagnostic only |

The checkpoints have finite tensors but no optimiser, RNG, run contract, or
complete trainer state. The embedded `/home/.../dev/...` source path is a
directory name and does not override the owner-confirmed `main` lineage. See
[`docs/evidence/author-main-generalist-audit-2026-07-20.md`](evidence/author-main-generalist-audit-2026-07-20.md)
for the plot, log, configuration, and runtime-route audit.

## Persisted-label Trust Boundary

The imported HumanDB contains historical Malom values in 1,560,069 position
rows and 1,691,422 move rows, but it has no `malom_label_version` metadata.
Those columns were produced before the sector correction and are therefore
untrusted. Current readers mask those fields while retaining human move
frequency, result, and game-count statistics.

The corrected SpecialistDB began empty and was the database used by the
completed 5,000-game managed baseline. On 22 July its SHA-256 is
`1203FC73CD7D0A06E2DD1FFACED5B031DFF8BD704E22B34BA02182FF3865614D`;
SQLite `quick_check` passes; metadata includes
`malom_label_version=sector-corrected-v1` and lineage root
`managed-v4-baseline-v1-segment-0001`; it now contains 132,182 positions,
41,904 Malom labels, 916 winning lines, and no preferred plays. It is trusted
completed-run state, not an empty input for another fresh experiment.

### Archived 21 July rebuilt databases

The maintainer's uploaded candidates were moved on 25 July to the ignored
archive `data/backups/maintainer_upload_20260721`. They have not replaced
either active database. Read-only inspection on 22 July, followed by
post-move hash and SQLite checks on 25 July, found:

- HumanDB SHA-256
  `F0B20D33AEFCBAB9AEDC8537F12FA2E53F7865B0387E2175AFD0EA32D1B90E42`;
  the supplied sidecar matches, SQLite `quick_check` passes, metadata is
  `sector-corrected-v1`, all 2,167,498 position rows have Malom WDL, and
  2,472,054 of 2,533,886 move rows have successor Malom WDL;
- SpecialistDB SHA-256
  `DF269D692E43815B88373F54B5AB1287022BC6736ECC8A5B95C7FB8A97FCD629`;
  SQLite `quick_check` passes, metadata is `sector-corrected-v1`, it has
  2,112,951 empirical positions, 60,117 winning lines, 30 preferred plays,
  and zero persisted Malom labels.

Thirty deterministic HumanDB probes, comprising five position and five
successor-move rows from each W/D/L class, matched both W/D/L and DTW when
queried through the current corrected Malom adapter. This supports the sampled
labels and metadata; it does not activate the file or prove every row.

The archived SpecialistDB retains the maintainer's empirical self-play history.
It is therefore not interchangeable with the empty corrected baseline DB. A
new experiment must explicitly choose one lineage. See
[`docs/evidence/main-integration-audit-2026-07-22.md`](evidence/main-integration-audit-2026-07-22.md)
for the original audit and
[`docs/evidence/maintainer-upload-relocation-2026-07-25.md`](evidence/maintainer-upload-relocation-2026-07-25.md)
for the relocation, file purposes, and small-file disposition.

The 25 July Openings delivery is retained separately under
`data/backups/maintainer_openings_20260725`. Its two Book files are exact
duplicates of tracked assets. Its learned-opening file contains 15 additional
low-confidence application-generated candidates and has not been merged into
an active source. See
[`docs/evidence/maintainer-openings-delivery-2026-07-25.md`](evidence/maintainer-openings-delivery-2026-07-25.md).

The 26 July expert-curated Book Opening Plays deliveries are retained
separately under `data/backups/maintainer_book_opening_plays_20260726`. The
exact DOCX files are ignored. The original tracked source transcription
preserves every move row and embedded-image identity while keeping visual
interpretations explicit. The later review supplement preserves semantic
tables and a confirmed row-19 correction without overwriting the original
evidence. Neither file is the Sanmill Book asset or an active input. See the
[original delivery record](evidence/maintainer-book-opening-plays-delivery-2026-07-26.md)
and the
[review delivery record](evidence/maintainer-book-opening-plays-review-delivery-2026-07-26.md).
The correction's current replay, expert semantic disposition, and frozen
33-pattern coverage catalogue are recorded separately in the
[reviewed-source audit](evidence/sanmill-layered-expert-book-reviewed-source-audit-2026-07-26.md),
[semantic disposition](evidence/maintainer-book-opening-plays-semantic-review-2026-07-26.md),
and the
[coverage decision](experiments/sanmill-layered-expert-book-coverage-decision-2026-08-01.md).

The original legacy SpecialistDB is isolated at:

```text
data\backups\drive_import_20260720\specialist_db.sqlite.legacy-pre-sector-fix
```

Its SHA-256 is:

```text
3DDD7172457E846602CBB026CEA3EB1F9E024B0D828F28EFA323105004DAE48F
```

The author's 20 July update is separately isolated at:

```text
data\backups\drive_import_20260720\specialist_db.sqlite.legacy-author-update-20260720
```

It is 268,521,472 bytes and has SHA-256:

```text
5C6A4EA1ACFB90BF05248580A07DAE7CF4645C09E5A4A69E2EC89EA9EE41811B
```

Its SQLite integrity check passes and it contains 1,954,437 positions, of which
339,904 have a Malom label, plus 54,456 winning lines and 27 preferred plays.
All 27 preferred-play rows are marked promoted. It has no `meta` table, so the
labels are unversioned and must be treated as legacy. The file is retained only
as a read-only empirical/audit snapshot; it did not replace the active
corrected database.

Do not open either legacy snapshot in write mode, copy it back to the active
database path, or add corrected labels to it.

The 406 new human-game files were imported without rebuilding the active
`data/human_db.sqlite`. That active HumanDB therefore still describes the
earlier corpus. The archived rebuilt candidate is the separately versioned
file described above; moving it into the active role remains an explicit
future decision. The active database's
94,983 `processed_files.file_path` keys use the author's `/home/...` absolute
paths, while the current builder compares Windows absolute paths. A blind
`--update` would therefore treat the existing corpus as new and double-count
it; migrate the processed-file keys or perform a controlled rebuild first.

## Machine-specific Configuration

`data/training_paths.local.json` is ignored by Git and is the path registry for
this machine. Query its actual values with:

```powershell
Get-Content data/training_paths.local.json
```

Do not paste machine-specific absolute values from that output into tracked
documents. The intended logical mapping is:

| Key | Repository-relative target or purpose |
| --- | --- |
| `generalist_output_dir` | `learned_ai/checkpoints/scaffolded/s_gen_v2_sector_corrected` |
| `sentinel_checkpoint` | `learned_ai/sentinel/checkpoints/best.pt` |
| `value_net_path` | `data/value_net.npz` |
| `gap_net_path` | Disabled pending provenance review |
| `human_db_path` | `data/human_db.sqlite` |
| `human_db_prefix12_snapshot_path` | Ignored online-backup snapshot for the twelve-ply source audit |
| `human_db_route_probe_snapshot_path` | Ignored, closed HumanDB snapshot for the no-update Sanmill route probe |
| `human_db_prefix12_source_manifest_path` | Ignored content manifest for the recursive PlayOK JSONL sample |
| `human_db_prefix12_history_ledger_path` | Ignored complete 83,002-history frequency and overlap ledger |
| `human_games_imported_manifest_path` | `data/human_games/imported.json` |
| `specialist_db_path` | `data/specialist_db.sector_corrected.sqlite` |
| `specialist_db_route_probe_snapshot_path` | Ignored, empty `sector-corrected-v1` SpecialistDB snapshot for the no-update route probe |
| `malom_db_path` | `../NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted` |
| `sanmill_checkout` | Cross-volume reference checkout; read the actual value from the ignored registry |
| `sanmill_training_checkout` | Ignored exact source/runtime pin for Sanmill-refereed training; normally under `data/runtimes` |
| `mif_checkout` | Frozen MIF source and black-box interoperability-harness checkout; read the actual value from the ignored registry |

The generalist trainer consumes the ordinary model/data keys above and, only
when `--referee-engine sanmill` is selected, the
`sanmill_training_checkout` runtime key. `sanmill_checkout` and `mif_checkout`
are only local reference-path indexes
for documentation and differential-test tooling. The archived candidates
deliberately have no configuration keys: find their relative paths in the
inventory and relocation record, then create a separately reviewed experiment
contract before use.

The trainer resolves configuration in this order:

1. explicit command-line argument;
2. matching `NMM_*` environment variable;
3. `training_paths.local.json`, overlaid on shared settings;
4. repository default.

The tracked `data/settings.json` still contains the previous maintainer's WSL
Malom path. That does not affect `train_s_gen_v2.py` when the local overlay is
present. Tools that read only `settings.json` may still need their own portable
path work; do not replace shared settings merely to make one training command
work.

## Data-handling Rules

- Put future maintainer deliveries under
  `../maintainer_inbox/<delivery-date-or-bundle-name>/`. Record the sender,
  delivery date, purpose, size, and checksum before moving them elsewhere.
  Never point training or evaluation directly at an inbox file.
- Keep the 21 July candidate databases in
  `data/backups/maintainer_upload_20260721` until a separately reviewed
  activation or retirement decision records their lineage. Do not copy them
  over active databases merely because their metadata audit passed.
- Let `.gitignore` protect databases, recursive human-game records, endgame
  tables, local paths, generated checkpoints, and backup snapshots.
- Before replacing a large database, record its size and checksum and retain a
  recoverable copy in the ignored backup directory.
- Do not alternate between Windows and WSL within one run. Windows is the
  current chosen environment; WSL is optional, not a correctness or
  performance requirement.
- Use the local path configuration instead of editing code to switch between
  machines.
