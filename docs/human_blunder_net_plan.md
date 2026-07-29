# HumanBlunderNet — design plan (revision 3)

Goal: model the **unfiltered** human move distribution — including losing
moves — so downstream code (GapNet v2, opponent modelling, humanlike-play
mode) can reason about which mistakes real players are likely to make.
Malom remains the sole source of objective move quality; HumanBlunderNet
only estimates likelihood.

The plan is intentionally staged: audit → semantic tests → candidate
schema rebuild → training → evaluation.  Each phase gates the next; no
code proceeds beyond the audit until every reviewer-flagged blocker is
closed.

## Revision history

- **rev 1** — initial plan.
- **rev 2** — first reviewer feedback: granular Elo-bin storage,
  retain all-losing positions, plain count-weighted CE, candidate-DB
  pattern.
- **rev 3** — second reviewer feedback (this document): correct
  HumanPrefNet description, softmax over all legal moves, mover-POV
  feature contract, deferred/versioned regret, remove WSL absolute
  path, single shared builder implementation, explicit target-
  population definition, provisional filtering only if data exists.

## Contrast with HumanPrefNet — corrected

Reviewer flag: revision 2 stated HumanPrefNet "drops L-after moves as
blunders" — that is the reverse of what the code actually does.
`tools/train_human_pref_net.py::_per_state_filter` inspects
`malom_wdl_after` (opponent-POV) and keeps `L`-after records (opponent
losing ⇒ human found a winning move).  If no `L` exists, it falls back
to `D`-after.  It drops `W`-after (opponent winning ⇒ human played into
a losing continuation).  Tests locking this direction are in
`tests/test_human_blunder_perspective.py::TestHumanPrefFilterDirection`.

|                          | HumanPrefNet (existing)                                                     | HumanBlunderNet (new)                                                                            |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| What it models           | Filtered "competent" human style                                            | Unfiltered human choice behaviour, including losing moves                                        |
| Data filter              | Prefers `L`-after (winning moves); falls back to `D`-after; drops `W`-after | No filter — every recorded move retained, including all-losing positions                         |
| Loss                     | Bradley-Terry pairwise BCE (order only)                                     | **Ordinary count-weighted cross-entropy over observed human move events**                        |
| Softmax denominator      | Pairwise (chosen vs sampled other) — inference relies on ranking only       | **Every legal move at the position** — observed moves get their counts, unobserved get target 0  |
| Output                   | Uncalibrated per-move scalar                                                | Calibrated per-move probability; `Σ p(m over legal) = 1`                                         |
| Feature perspective      | Successor board, successor's side-to-move (opponent POV)                    | Successor board, **explicitly** encoded from the original mover's POV via colour argument        |
| Elo conditioning         | None                                                                        | Elo band as one-hot input; band membership defined at training time from 50-Elo storage bins     |
| Downstream consumer      | Humanlike-play opponent, GapNet aux                                         | GapNet v2 expected-regret formula (defined + versioned in the GapNet plan)                       |

Same 79-input board feature vector shape.  HumanBlunderNet adds a 3-way
Elo one-hot → 82-float input.

## Output form

Distribution `p(m | position, elo_configuration)` over legal moves.
Not a scalar `P(mistake)` — the downstream GapNet formula needs the full
distribution.  `P(mistake)` is derivable at inference as
`Σ p(m) · [is_blunder(m)]`, so no separate model.

## Target population — explicit statement (reviewer §7)

**Count-weighted training models the distribution of move *events*.**
A player who contributes 100,000 plies has 100,000× the influence of a
player who contributes 1 ply.  The Phase-1 audit shows the top mover
contributes `~2.32 %` of all plies; the top-10 movers together
contribute `14.30 %`.

That may be exactly the desired target — "a randomly selected PlayOK
move event" — or it may not.  Alternatives that require Phase-2
schema changes (per-player accumulation columns):

- Per-player uniform reweighting → target = "a randomly selected
  PlayOK player, uniformly."
- Per-player capped weighting → the weight-per-player is capped at some
  quantile.

**This plan commits to modelling move events** (option 1) unless the
user overrides.  Rationale: (a) move events are the natural unit for
the GapNet expected-regret formula which is per-position, not per-
player; (b) the Phase-1 top-10 share of 14.30 % is high but not
extreme; and (c) reweighting adds a hyperparameter that couples with the
Elo band structure in ways that are hard to audit.

The choice is recorded here so any change is a deliberate revision.

## Elo strategy — sparse 50-Elo bin storage, Option A as training-time config

Reviewer §6: hard-coded band columns cannot be reallocated once counts
are aggregated.  The active DB already loses this information (v2 has
no Elo columns at all).  The candidate DB (Phase 2) will store move
counts in **50-Elo bins**, and configurations like A are applied at
training time as a function that sums the bins.

**Option A** (this plan's default first training configuration;
strata within the PlayOK amateur corpus, not universal strength
labels; **boundaries bin-aligned** so 50-Elo bins never straddle two
bands):

| Band   | Cut-off        | Audited share of moves |
| ------ | -------------- | ---------------------- |
| lower  | ≤ 1149         | 14.2 %                 |
| middle | 1150 – 1249    | 36.3 %                 |
| upper  | ≥ 1250         | 49.5 %                 |

The reviewer's original spec used `≤1150 / 1151-1250 / ≥1251` — we
shift each boundary by 1 Elo so that no 50-Elo bucket
(`int(elo) // 50 * 50`) straddles two bands.  Impact on move counts is
<1 % per band and the blunder-rate gradient
(`7.50 % / 5.86 % / 4.84 %`) is unchanged.

Rerunning the audit with different cut-offs is a script-only change; no
DB rebuild.  Options B / C etc. can be tried without invalidating any
stored data.

**Reviewer §6 sub-decision**: sparse bin table (candidate schema §2.2)
is preferred over a JSON blob column, for query composability.  If size
proves problematic, a JSON-blob variant is an acceptable fallback but
must document the boundary-versioning limitation.

## Phase 1 — audit (COMPLETE)

Deliverables landed and reviewed:

- `tools/audit_human_blunders.py` (audit version 1.1).
- `data/human_blunder_audit_optA.json` — machine-readable output.
- `data/human_blunder_audit.json` — original-boundary snapshot for
  reviewer's reconciliation trace.
- `docs/human_blunder_audit_phase1.md` — reconciled report, revision 2.
- `tests/test_human_blunder_perspective.py` — 25 tests including 3
  Malom-DB integration tests; all pass.
- Full sample-flow reconciliation, per-cell `n_moves` and
  `n_positions`, coverage counts + shares, per-side Elo attrition
  (both zero here), player concentration (top-10 share 14.30 %).

Findings that inform the plan:

- Monotonic blunder-rate gradient across Option A bands
  (`lower 7.49 %`, `middle 5.84 %`, `upper 4.84 %`) — Elo conditioning
  has real signal.
- No strong-human data (`p95 = 1400`, `max = 1700`) — "lower/middle/
  upper" phrasing throughout.
- No provisional-status filtering claimed — the source records do not
  carry those flags.
- Malom-perspective convention: parent = mover-POV, child =
  opponent-POV, flip via `{W: L, L: W, D: D}`.  Locked with tests.

## Phase 2 — Malom-perspective tests + candidate DB rebuild

Two sub-phases; §2.1 blocks §2.2 which blocks Phase 3.  **Nothing in
Phase 2 has been implemented yet beyond §2.1.**

### 2.1 Malom perspective tests — DONE (reviewer §3)

- Unit tests for `_FLIP` involution and every `(pre, after)` case in
  `_classify_transition`.
- Integration test: `MalomDB.query(parent).outcome` returns mover-POV;
  `MalomDB.query(child).outcome` returns opponent-POV; the flip is
  consistent.
- Semantic assertion: a Malom-labelled `W` parent has at least one
  legal move classified as `win_preserved`.
- HumanPrefNet's `_per_state_filter` covered as well, so the plan's
  contrast row above is code-verified.

All 25 tests pass on the current codebase.  Any change to the audit
script's flip logic will be caught by the fixture suite.

### 2.2 Candidate DB rebuild — NOT STARTED (reviewer §12)

**Do not touch `data/human_db.sqlite`.**  The current builders do not
enforce a clean schema-version migration and rely on host-path-
dependent `processed_files` identity, which carries a double-count
risk on `--update`.

Instead:

- **One shared implementation.**  Reviewer §12 flagged that
  `tools/build_human_db.py` and `tools/build_human_db_sha.py` have
  already drifted.  Phase 2 collapses the shared build pipeline into a
  library module (probably `tools/_human_db_build.py`), with the two
  scripts becoming thin wrappers that differ only in whether the
  SHA-256 sidecar file is emitted.  A regression test asserts the two
  entry points produce byte-identical DB output on a small fixture.
- **Candidate output path.**  New candidate DB at
  `data/human_db_candidate.sqlite`.  Existing default output paths
  remain the safe active path.  Activation (rename candidate over
  active) is a **separate later decision** by the user, with backups
  first.
- **Logical source identity instead of host paths.**  Reviewer §12: use
  the JSONL file's content SHA-256 (or a stable session_id from the
  record) as the processed_files primary key, not the absolute file
  path.  This lets a re-run on a different machine reconcile without
  double-counting.
- **Malom path is caller-supplied**, resolved from
  `data/training_paths.local.json` or the CLI, **never a hard-coded
  absolute path in the script or plan**.  Revision 2 embedded a WSL
  path — that has been removed.  The command shown below reads Malom
  from the local config.
- **Schema version 3** on the candidate; active DB stays at v2.
  Candidate additions:
  - `moves` unchanged in shape; retains existing all-Elo
    `wins/losses/draws/total` rollups as game-outcome semantics
    (see §Schema semantics).
  - New table `moves_elo_bins` with primary key
    `(state_key, notation, elo_bin)` and single column `total`
    (game-outcome-independent count of mover events).  Sparse: rows
    only exist where the bin has ≥1 play.  Per reviewer §13, only
    `total` is required to estimate `p(move | position, band)`.
  - Optional `game_wins, game_losses, game_draws` columns in
    `moves_elo_bins` are **deferred** — the plan will not store them
    unless a specific downstream analysis requests them.
  - New table `positions_elo_bins` with primary key
    `(state_key, elo_bin)` and column `total_games`.
- **Provenance in `meta`:**
  - `schema_version = 3`
  - `elo_bin_size = 50`
  - `feature_canonicalisation_version = <the version of make_board_state_key at build time>`
  - `malom_label_version = <inherited or refreshed>`
  - `source_manifest_sha256 = <sha of (session_id or content-sha, size) tuples>`
  - `builder_git_commit = <HEAD sha>`
  - `built_at = <ISO timestamp>`
- **Candidate validation script** `tools/validate_human_db_candidate.py`:
  - SQLite `PRAGMA quick_check` returns `ok`.
  - Row-count reconciliation: `sum(positions_elo_bins.total_games)`
    over bins equals `positions.total_games` per state_key.
  - Move-count reconciliation: `sum(moves_elo_bins.total)` over bins
    equals `moves.total` per (state_key, notation).
  - Fixed semantic probes for hand-picked positions (Malom
    perspective + expected Elo-bin totals).
  - Emits a signed report file with the candidate DB's SHA-256,
    schema version, source manifest hash, and validation outcomes.
- **Rebuild command.**  Malom path is auto-resolved (priority order:
  `--malom-db` CLI arg → `NMM_MALOM_DB` env var → `malom_db_path` in
  `data/training_paths.local.json` → empty).  The `--candidate-out`
  flag is required whenever `--output` would resolve to the active
  HumanDB path (the fail-closed guard is enforced in code).
  ```
  .venv/bin/python tools/build_human_db_sha.py \
      --rebuild \
      --games-dir data/human_games \
      --candidate-out data/human_db_candidate.sqlite
  ```
  Add `--no-malom` for a fast pre-Malom smoke pass, or `--malom-db /path`
  to override the config-resolved path.
- **Activation** is a subsequent, deliberate action — never automatic:
  ```
  cp data/human_db.sqlite data/human_db.sqlite.pre-v3.bak
  mv data/human_db_candidate.sqlite data/human_db.sqlite
  ```

## Phase 3 — HumanBlunderNet training — NOT STARTED

`tools/train_human_blunder_net.py`, structurally similar to
`tools/train_human_pref_net.py` but differing on the axes below.

### 3.1 Every legal move participates in the softmax (reviewer §9)

**Critical fix from revision 2.**  For every `(position, band)` in the
training loader, the sample must include every legal move at the
parent, not just moves observed in HumanDB.

- Observed moves: target count = `sum(moves_elo_bins.total for bin in band)`.
- Unobserved legal moves: target count = `0`, and their logits
  participate in the softmax denominator.
- Loss:
  ```
  L = − Σ (position, band, move) count_band(position, move) · log p_pred(move | position, band)
  ```
  This is equivalent to training on individual observed move events.

Without unobserved moves in the denominator, the model is not
penalised for assigning probability mass to moves it never saw — and
inference softmax always covers every legal move.

Optional probability smoothing (e.g. add-ε to unobserved) is deferred;
the baseline choice set must match inference regardless.

### 3.2 Input features and mover-POV contract (reviewer §7, §10)

- Board features are always encoded from the **original mover's**
  perspective.  Concretely:
  ```
  successor = board.apply_move(move)
  feats = board_to_features(successor, original_mover_colour)
  ```
  where `original_mover_colour` is the colour of the player who made
  the move — not `successor.turn` (which is the opponent).
- Reviewer §10: the plan's revision 2 wrote `board_to_features(chosen_board)`
  as a single-argument call, which is not the current interface.
  The correct call is `board_to_features(board, colour)`.  Any Phase-3
  code that gets this wrong will be caught by a mandatory colour-swap
  perspective test on the training loader.
- One-hot Elo band `[lower, middle, upper]` set from the training
  configuration (Option A by default); `unknown` band moves are
  excluded from training.
- Feature dim: 79 board + 3 Elo one-hot = 82.

### 3.3 Sample generation

Per `(state_key, band)`, sample = `(parent_board, mover_colour, band,
legal_moves, per-move-counts)`.  Sourced from the candidate DB's
`moves_elo_bins` table joined with a per-position replay to recover
`parent_board` and enumerate `legal_moves`.

**All-losing positions retained** (reviewer §5): humans still choose
between losing continuations.  GapNet may explicitly mask them if it
uses coarse W/D/L regret, but the behaviour model does not discard
them.

### 3.4 Loss (reviewer §6, §8)

**Ordinary count-weighted cross-entropy over observed move events.**
No focal loss.  No inverse-frequency reweighting.  Any reweighting
breaks the calibration guarantee needed by the GapNet expected-regret
formula.

Rare severe blunders (`win_to_loss` at 0.16 %) are reported as an
evaluation stratum (§Phase 4) and, if separate detection is desired
later, handled through an auxiliary head or an entirely separate
model — never by perturbing the primary probability head.

### 3.5 Model

MLP 82 → 128 → 64 → 32 → 1 (same shape as HumanPrefNet, +3 Elo dims).
Softmax over legal-successor scores at inference.

Saved as `.npz` sibling to `data/human_pref_net.npz`.  Default output:
`data/human_blunder_net.npz`.

**Model provenance (reviewer §14)** — the `.npz` records:

- `feature_version`
- `elo_bin_size`
- `elo_band_config_name` (e.g. `"option_a_1150_1250"`)
- `source_db_sha256`
- `split_version`
- `training_objective` (e.g. `"count_weighted_ce"`)
- `git_commit`
- `layer_count` (existing convention)

Filename alone is not sufficient provenance.

Inference wrapper `ai/human_blunder_advisor.py` mirrors
`ai/human_pref_advisor.py` but adds an `elo_band: str` parameter.

### 3.6 Split (reviewer §8)

- Reuse `learned_ai.data.human_db_split.in_val_bucket` as the primary
  position-level held-out split — same one HumanPrefNet uses, for
  cross-model comparability.
- Bump / commit `split_version = "1"` in the split module; the value
  is recorded in the model `.npz`.
- **Game-held-out diagnostic** in addition to position-held-out.  Reserve
  5 % of `session_id`s (deterministic hash) as a game-held-out set.
- **Assert no canonical state appears in both train and val** (per
  reviewer's "same canonical state must not appear in both training
  and validation" rule).  Programmatic check in the training loader.

## Phase 4 — evaluation — NOT STARTED

`tools/eval_human_blunder_net.py` reports, on both position- and
game-held-out sets:

- **Primary:** event-weighted negative log-likelihood.
- Top-1 / top-3 / top-5 move-accuracy vs the most-frequent human move.
- Probability calibration (reliability diagram bin counts + ECE).
- Per-Elo-band breakdown of every metric.
- Per-phase (`place`/`move`) breakdown.
- Per-Malom-regret-severity stratum (only once the GapNet plan
  provides a versioned regret mapping; until then, per-transition-
  category is used).
- Per-position KL only for positions with `≥ min_support` plays
  per band (min_support default = 10; singletons excluded from
  per-position distribution comparison but retained in event-
  weighted NLL).
- Empirical HumanDB comparison where support permits.

Baselines to report alongside:

- Uniform over legal moves.
- Empirical HumanDB frequencies (`moves_elo_bins.total`) where
  support is sufficient.
- HumanPrefNet softmax at the same temperature (for style comparison).

## Regret — deferred to GapNet plan (reviewer §11)

Malom's complete move values are **ordered objects, not a globally
subtractable scalar**.  Raw `key2` subtraction is not a valid regret
target.

Any downstream regret used in the GapNet expected-regret formula must
be defined via `OracleMoveValue` comparison within the parent context
(after resolving rules-terminal successors) and its scalar/rank
mapping explicitly specified, tested, and versioned.

The GapNet plan must choose and version one of the following (or
another tested mapping):

- probability of choosing a non-oracle-optimal move
- expected ordinal rank loss (`rank_of_chosen_move − rank_of_best_move`)
- frozen W/D/L utility (`W=1, D=0, L=-1`) plus separately defined
  within-class distance (e.g. `dtw` differential)
- another tested utility mapping

A proposed `malom.query_regret(board, move)` API does not currently
exist.  When implemented, it must **fail closed**: missing parent or
successor probes return an explicit "unavailable" — a missing Malom
value must never silently become `regret = 0`.

## Schema semantics — disambiguate from Malom (reviewer §13)

The existing `wins/losses/draws/total` on `moves` and `positions` are
**final human game outcomes for the mover** — distinct from Malom's
W/D/L labels.  This has caused reader confusion.

Candidate DB v3 additions:

- Inline SQL comments on each `wins/losses/draws` column making the
  semantics explicit.
- The new `moves_elo_bins` stores only `total` per bin (mover events);
  no game-outcome subcounts unless a specific downstream analysis
  requests them (deferred).
- If mover-event totals are ever augmented with per-band game-outcome
  columns, they must be named `game_wins_band`, `game_losses_band`,
  `game_draws_band` to avoid Malom-W/D/L confusion.
- Schema description file `data/human_db_schema_v3.md` alongside the
  candidate DB.  States whether position-level band totals count
  "occurrences of the side-to-move player" (yes, they do — one
  increment per mover ply at that state_key).

## Blockers before Phase 2.2 (DB rebuild) proceeds

- ✅ Perspective fixture tests pass (2.1).
- ⏳ User authorises the candidate DB rebuild against a candidate path.
- ⏳ Shared builder library extraction agreed (reviewer §12) so the two
  entry points can no longer drift.
- ⏳ Sparse-table vs JSON-blob decision confirmed (default: sparse).
- ⏳ Target population — event-weighted default accepted, or a per-
  player scheme requested.

Phase 3 (network training) has additional blockers on top of these:

- ⏳ Regret mapping defined and versioned in the GapNet plan (only if
  Phase-4 eval needs regret-stratified metrics — otherwise Phase 3 can
  proceed).
- ⏳ `split_version` bumped and committed.

## Non-goals

- Rebuilding or retraining GapNet.
- Retiring HumanPrefNet — it remains a separately-purposed
  "competent filtered human style" model.
- Retraining any AI or shipping any inference change until Phase 4
  evaluation shows a clean improvement on held-out metrics.
- Filtering on provisional-rating status — the source records do not
  carry those flags.  Not implemented, not claimed.
