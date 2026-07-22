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
| `../Mills` | Temporary import staging directory | Currently empty; safe to reuse as staging |
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

## Repository-local Data Inventory

This inventory was measured on 20 July 2026.

| Asset | Current location and state |
| --- | --- |
| HumanDB | `data/human_db.sqlite`, 738,091,008 bytes; 94,429 games, 2,152,889 positions, and 2,516,356 move rows |
| Human game files | `data/human_games`, 95,389 `.jsonl` files plus import metadata; the 20 July author update added 406 files and raised `imported.json` from 94,134 to 94,540 entries |
| Human game source archive | `../human_database/human_games_94559.zip`, 121,796,279 bytes; SHA-256 `45523234085518031A09725A2DBCAB395E55026787E420A04C37EBA10A0E4D07` |
| Corrected SpecialistDB | `data/specialist_db.sector_corrected.sqlite`; metadata is `sector-corrected-v1`; all three data tables are currently empty |
| Legacy SpecialistDB snapshots | Two ignored, read-only snapshots under `data/backups/drive_import_20260720`; neither is an active training database |
| Endgame databases | `data/endgame`, fourteen `.wdl` files plus `fullgame.bin` at 571,683,560 bytes |
| Malom tablebase | `../NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted`; 512 files and 83,582,223,577 bytes |
| Sentinel | `learned_ai/sentinel/checkpoints/best.pt` |
| Generalist checkpoints | `learned_ai/checkpoints/scaffolded/s_gen_v2/best.pt` and `best1.pt` through `best6.pt` |
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

The clean SpecialistDB is intentionally empty. It is safe for a corrected run
because it carries `malom_label_version=sector-corrected-v1`; the trainer can
add empirical game statistics and freshly decoded Malom labels without mixing
them with legacy labels.

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

The 406 new human-game files were imported without rebuilding
`data/human_db.sqlite`. HumanDB therefore still describes the earlier corpus
until a separately reviewed incremental or full rebuild is performed. Its
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
| `specialist_db_path` | `data/specialist_db.sector_corrected.sqlite` |
| `malom_db_path` | `../NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted` |
| `sanmill_checkout` | Cross-volume reference checkout; read the actual value from the ignored registry |

The generalist trainer consumes the seven training keys above.
`sanmill_checkout` is only a local reference-path index for documentation and
differential-test tooling.

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

- Keep the Google Drive import in its canonical destinations above. The
  `Mills` staging directory has already been emptied.
- Let `.gitignore` protect databases, recursive human-game records, endgame
  tables, local paths, generated checkpoints, and backup snapshots.
- Before replacing a large database, record its size and checksum and retain a
  recoverable copy in the ignored backup directory.
- Do not alternate between Windows and WSL within one run. Windows is the
  current chosen environment; WSL is optional, not a correctness or
  performance requirement.
- Use the local path configuration instead of editing code to switch between
  machines.
