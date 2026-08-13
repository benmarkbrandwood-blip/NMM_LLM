# HumanMovePolicyNet — design plan (revision 3)

Goal: model the **unfiltered** human move distribution — including losing
moves — so downstream code (GapNet v2, opponent modelling, humanlike-play
mode) can reason about which mistakes real players are likely to make.
Malom remains the sole source of objective move quality; HumanMovePolicyNet
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
- **rev 3** — second reviewer feedback: correct
  HumanPrefNet description, softmax over all legal moves, mover-POV
  feature contract, deferred/versioned regret, remove WSL absolute
  path, single shared builder implementation, explicit target-
  population definition, provisional filtering only if data exists.
- **rev 4** (2026-07-30, this document) — Phase 2 and Phase 3 landed
  end-to-end.  Candidate DB rebuild ran under the shared builder
  library with Malom auto-resolution; Phase 3 pipeline shipped as
  three commits (extractor + trainer, advisor, eval) with 23 tests
  green.  Marks Phase 2 §2.2 and Phase 3 sections COMPLETE; Phase 4
  eval script implemented.  Phase 5 (GapNet v2 consumption) remains
  out of scope.
- **rev 5** (2026-08-11 / -12) — Phase 5 renamed and repurposed as
  the **v3 teacher retrain** driven by GapNet v3 Decision 6B (see
  `docs/gap_net_v3_stage_e_rebuild_checklist.md`).  Codex found that
  the v2 candidate was trained under `three_way_split(state_key)`,
  which does not provide session-level out-of-fold guarantees for the
  P_h teacher signal used by GapNet.  Phase 5 adds a session-isolated
  retrain pipeline that shares one frozen session ledger, source
  manifest, and split seed with the GapNet Stage D extractor.  The v2
  candidate `data/human_move_policy_net_v2_candidate.npz` is retained
  untouched as an exploratory comparison — the v3 teacher lands under
  `data/human_move_policy_net_v3_teacher_candidate.npz`.  Tooling
  (session ledger builder, extractor `--session-ledger` flag, trainer
  safety guard) landed in commits `6d61d40`, `ec567b2`, and `9efe0ba`;
  the retrain run itself is gated by a readiness checkpoint and has
  not yet been executed.  The original "Phase 5 (GapNet v2 consumption)"
  is dropped as GapNet v2 is superseded by v3.

## Contrast with HumanPrefNet — corrected

Reviewer flag: revision 2 stated HumanPrefNet "drops L-after moves as
blunders" — that is the reverse of what the code actually does.
`tools/train_human_pref_net.py::_per_state_filter` inspects
`malom_wdl_after` (opponent-POV) and keeps `L`-after records (opponent
losing ⇒ human found a winning move).  If no `L` exists, it falls back
to `D`-after.  It drops `W`-after (opponent winning ⇒ human played into
a losing continuation).  Tests locking this direction are in
`tests/test_human_moves_audit_perspective.py::TestHumanPrefFilterDirection`.

|                          | HumanPrefNet (existing)                                                     | HumanMovePolicyNet (new)                                                                            |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| What it models           | Filtered "competent" human style                                            | Unfiltered human choice behaviour, including losing moves                                        |
| Data filter              | Prefers `L`-after (winning moves); falls back to `D`-after; drops `W`-after | No filter — every recorded move retained, including all-losing positions                         |
| Loss                     | Bradley-Terry pairwise BCE (order only)                                     | **Ordinary count-weighted cross-entropy over observed human move events**                        |
| Softmax denominator      | Pairwise (chosen vs sampled other) — inference relies on ranking only       | **Every legal move at the position** — observed moves get their counts, unobserved get target 0  |
| Output                   | Uncalibrated per-move scalar                                                | Calibrated per-move probability; `Σ p(m over legal) = 1`                                         |
| Feature perspective      | Successor board, successor's side-to-move (opponent POV)                    | Successor board, **explicitly** encoded from the original mover's POV via colour argument        |
| Elo conditioning         | None                                                                        | Elo band as one-hot input; band membership defined at training time from 50-Elo storage bins     |
| Downstream consumer      | Humanlike-play opponent, GapNet aux                                         | GapNet v2 expected-regret formula (defined + versioned in the GapNet plan)                       |

Same 79-input board feature vector shape.  HumanMovePolicyNet adds a 3-way
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

- `tools/audit_human_moves.py` (audit version 1.1).
- `data/human_moves_audit_optA.json` — machine-readable output.
- `data/human_moves_audit.json` — original-boundary snapshot for
  reviewer's reconciliation trace.
- `docs/human_moves_audit_phase1.md` — reconciled report, revision 2.
- `tests/test_human_moves_audit_perspective.py` — 25 tests including 3
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

### 2.2 Candidate DB rebuild — COMPLETE (2026-07-30)

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

**Validation result (2026-07-30).**  `tools/validate_human_db_candidate.py`
ran against `data/human_db_candidate.sqlite` and returned `ok: True`:

- `schema_version: 3`
- `quick_check: ok`
- `candidate_sha256: df71395f51b9497b5d0916fbcc5a6a8456d082be79c280ecb354d5ba73f87d4f`
- `missing_tables: []`
- `missing_meta_rows: []`
- `positions_elo_bins` reconciliation mismatches: 0
- `moves_elo_bins` reconciliation mismatches: 0
- Semantic probe (starting position `malom_wdl`): `D` ✓

Report on disk: `data/human_db_candidate.sqlite.validation.json`.
**Candidate DB is NOT yet activated** — activation over
`data/human_db.sqlite` is a separate later decision.

## Phase 3 — HumanMovePolicyNet training — COMPLETE (2026-07-30)

**Training result (2026-07-30).**  `tools/train_human_move_policy_net.py`
ran to convergence on `data/human_move_policy_dataset/` (2.2 M state keys,
2.56 M samples from candidate DB SHA `df71395f`):

- Stopped at epoch 22 / 200 (patience=6, grad_clip=1.0, lr=3e-4, dropout=0.2)
- Best val event NLL: **1.5953**
- Elapsed: ~11.6 h
- Output: `data/human_move_policy_net.npz` (trainer_git_commit `5c9ca45`)

Eval report: `data/human_move_policy_eval.json`.

Landed as three commits over 2026-07-29 / 2026-07-30:

- **Commit C** `397d828` — extractor + trainer (`tools/extract_human_move_policy_dataset.py`,
  `tools/train_human_move_policy_net.py`).  Two-stage design per advisor guidance: extract
  once (successor feature bank dedup'd by state_key + sample records + provenance),
  iterate on hyperparameters cheaply.  8 extractor tests + 3 trainer tests, all green.
- **Commit D** `16571a6` — inference wrapper (`ai/human_move_policy_advisor.py`).
  Pure-numpy MLP forward pass mirroring `ai/human_pref_advisor.py`; adds an
  `elo_band` parameter to `rank()` / `probs()`.  6 tests including the band-
  conditioning assertion.
- **Commit E** `780d3a2` — held-out eval (`tools/eval_human_move_policy_net.py`).
  Event-weighted NLL, top-1/3/5, ECE calibration, per-band + per-phase +
  per-Malom-transition strata, empirical KL against HumanDB frequencies at
  `--min-support` support.  6 tests.

Total: 6 new production files, 4 new test files, 23 tests, all green.  End-to-end
pipeline documented in `script_commands.md` under the HumanMovePolicyNet sections.

Design axes from the plan below are implemented as originally specified; the
critical fixes flagged by the reviewer (softmax over EVERY legal move, mover-POV
board features via the two-argument call, count-weighted CE without focal /
inverse weighting) are locked in code and asserted by tests.

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
`data/human_move_policy_net.npz`.

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

Inference wrapper `ai/human_move_policy_advisor.py` mirrors
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

## Phase 4 — evaluation — COMPLETE (2026-07-30)

Landed in Commit E `780d3a2` as `tools/eval_human_move_policy_net.py`.  Every
metric listed below is reported on the position-held-out slice
(`sample_is_val == True`).  Game-held-out diagnostic (§3.6) is deferred
to a follow-up commit — this ships the position-level split that
HumanPrefNet + ValueNet already share.

**Eval results (2026-07-30)** — from `data/human_move_policy_eval.json`,
512 805 val samples, 1 042 147 events:

| Stratum | NLL | Top-1 | Top-3 | Top-5 | ECE |
| --- | --- | --- | --- | --- | --- |
| Overall | 1.595 | 45.6 % | 78.5 % | 88.2 % | 0.178 |
| band=lower | 1.619 | 45.2 % | 77.7 % | 87.8 % | 0.180 |
| band=middle | 1.579 | 46.3 % | 78.8 % | 88.9 % | 0.180 |
| band=upper | 1.600 | 45.2 % | 78.5 % | 87.8 % | 0.176 |
| phase=place | 1.915 | 39.4 % | 71.6 % | 82.2 % | 0.209 |
| phase=move | 1.341 | 50.5 % | 84.0 % | 92.9 % | 0.171 |
| trans=win_preserved | 1.633 | 45.0 % | 74.5 % | 86.4 % | 0.188 |
| trans=win_to_draw | 2.044 | 26.9 % | 61.5 % | 78.3 % | 0.130 |
| trans=win_to_loss | 2.542 | 16.7 % | 46.4 % | 67.1 % | 0.080 |
| trans=draw_preserved | 1.590 | 46.4 % | 80.0 % | 88.7 % | 0.187 |
| trans=draw_to_loss | 2.247 | 21.0 % | 55.5 % | 73.0 % | 0.104 |
| trans=all_losing | 1.317 | 51.7 % | 83.9 % | 92.8 % | 0.168 |
| trans=unlabelled | 1.875 | 37.3 % | 68.1 % | 83.3 % | 0.177 |

Empirical KL (≥10-support positions, n=6 157): **mean KL 0.557**.

Notes: Blunder transitions (`win_to_loss`, `draw_to_loss`, `win_to_draw`)
are harder to predict (lower top-1, higher NLL), as expected — these are
low-frequency minority decisions that the model's count-weighted CE
objective down-weights relative to their importance.  ECE on `win_to_loss`
is low (0.080) because the model is less confident and more dispersed,
which is actually better-calibrated than the majority class.

Report structure below is the design spec that Commit E implements:


`tools/eval_human_move_policy_net.py` reports, on both position- and
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

## Phase 4b — extended eval + v2 split — COMPLETE (2026-08-01)

Extended eval script (`tools/eval_human_move_policy_net.py`, rewritten commit
`de2e932`) and v2 split rework landed 2026-07-31; re-extraction + re-train +
full Phase 4b eval completed 2026-08-01.  Results in
`data/gap_v3_prerequisite_eval.json` (git commit 7b9d6bb).

**v2 dataset:** 5/15/80 three-way split (`three_way_split()` by state_key
hash), 2 050 608 train / 384 837 val / 127 968 test samples; 2 209 726
state_keys; extraction 515 s.

**v2 model:** 26 epochs, best val event NLL **1.5816** (v1: 1.5953), elapsed
65 150 s (~18.1 h).  Temperature scaling: T\* = 1.0 (no improvement from
scaling).

**Eval results (2026-08-01, v2 model)** — 384 837 val samples, 785 550 events:

| Stratum | NLL | Brier | Top-1 | Top-3 | Top-5 | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 1.582 | 0.616 | 45.5 % | 79.5 % | 89.0 % | 0.175 |
| band=lower | 1.613 | 0.620 | 44.8 % | 78.3 % | 88.0 % | 0.175 |
| band=middle | 1.567 | 0.609 | 46.1 % | 79.6 % | 89.3 % | 0.177 |
| band=upper | 1.583 | 0.619 | 45.3 % | 79.8 % | 89.0 % | 0.174 |
| phase=place | 1.883 | 0.644 | 39.3 % | 73.8 % | 84.0 % | 0.201 |
| phase=move | 1.342 | 0.607 | 50.5 % | 84.0 % | 92.9 % | 0.171 |
| trans=win_preserved | 1.630 | 0.703 | 45.1 % | 74.6 % | 86.5 % | 0.189 |
| trans=win_to_draw | 2.037 | 0.855 | 26.7 % | 61.8 % | 79.0 % | 0.133 |
| trans=win_to_loss | 2.541 | 0.968 | 15.3 % | 46.1 % | 67.8 % | 0.084 |
| trans=draw_preserved | 1.572 | 0.670 | 46.3 % | 81.3 % | 89.8 % | 0.182 |
| trans=draw_to_loss | 2.222 | 0.912 | 21.4 % | 56.3 % | 73.8 % | 0.106 |
| trans=all_losing | 1.322 | 0.629 | 51.5 % | 83.8 % | 92.7 % | 0.169 |
| trans=unlabelled | 1.871 | 0.750 | 37.5 % | 68.3 % | 83.1 % | 0.179 |
| lmc_2-5 | 0.900 | 0.473 | 61.4 % | 95.7 % | 100 % | 0.211 |
| lmc_6-10 | 1.298 | 0.592 | 51.3 % | 85.3 % | 95.3 % | 0.217 |
| lmc_11-20 | 1.960 | 0.728 | 37.7 % | 64.7 % | 78.1 % | 0.199 |
| lmc_21+ | 2.089 | 0.848 | 34.0 % | 74.7 % | 83.1 % | 0.182 |
| game_val_only | 1.571 | 0.675 | 44.9 % | 76.5 % | 88.3 % | 0.173 |

**Baselines (overall):**

| Baseline | NLL | Brier | Top-1 | Top-3 | Top-5 | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Uniform | 2.347 | 0.808 | 11.7 % | 31.2 % | 48.4 % | 0.020 |
| Empirical (≥10 support, n=4 673 positions) | 1.320 | ~0 | 55.8 % | 85.1 % | 93.3 % | 0.235 |

Empirical KL (≥10-support, n=4 673 positions): **mean KL 0.552** (v1: 0.557).
Abstention: **0 samples** (0 encoding failures). OOD coverage: 100 % of val
positions appear in session index.

**Notes:**

- NLL improvement over uniform: 30–33 % relative per band — model is
  substantially better than chance at every Elo tier.
- Brier score newly reported; values 0.47–0.62 depending on lmc, consistent
  with a high-entropy multi-class task.
- lmc-stratum gradient clear: NLL 0.90 (lmc 2-5) → 2.09 (lmc 21+), confirming
  that positions with more legal moves are harder to predict.
- game_val_only stratum (42 607 samples, positions seen only in game-val games)
  tracks closely with overall val — no evidence of game-level leakage from the
  position-level split.
- ECE 0.175 per band.  The §16 gate threshold of 0.05 is unreachable in
  practice: the uniform model itself achieves 0.020 because predicting uniform
  is trivially calibrated regardless of human choice patterns.  The empirical
  baseline ECE is 0.235 — worse than the model.  Gate threshold requires
  revision before Stage B is declared fully closed.

**Stage B gate check (§16 of gap_net_v3_plan.md):**

- ✅ NLL gate: 30–33 % relative improvement per band (threshold: ≥ 20 %).
- ❌ ECE gate: 0.174–0.177 per band after T\*=1.0 scaling (threshold: ≤ 0.05).
  Gate threshold must be revised — see note above.
- ✅ Abstention / OOD: 0 abstentions.

## Phase 5 — v3 teacher retrain (session-isolated, for GapNet v3)

**Motivation.**  GapNet v3 (Decision 6B in
`docs/gap_net_v3_stage_e_rebuild_checklist.md`) uses `P_h(m | state, band)`
from this network as its **teacher signal** when computing training targets
for `G_v(state, band)`.  If the teacher was itself trained on the same
state_keys that GapNet later evaluates against, the teacher signal on those
"held-out" GapNet val/test positions is not out-of-fold — it was seen
during training.  Codex flagged this in review of GapNet commit `728ddad`:

> the HumanMovePolicyNet used to generate P_h must not have trained on the
> held-out GapNet sessions or states.

The v2 candidate was trained under `three_way_split(state_key)` (§3.6 above),
which does not provide session-level cleanliness.  Phase 5 addresses this by
retraining the teacher with strict single-tier session isolation.

**The v2 candidate is not replaced.**  `data/human_move_policy_net_v2_candidate.npz`
stays intact as an exploratory-comparison baseline.  The v3 teacher lands
under the separately-named `data/human_move_policy_net_v3_teacher_candidate.npz`.
Both `.npz`s can be loaded by the same `ai/human_move_policy_advisor.py`.

### 5.1 Frozen shared session ledger (Batch 3a — LANDED, commit `6d61d40`)

The session ledger is the single source of truth for session→split assignments,
shared with the GapNet Stage D extractor.  Neither the teacher retrain nor the
GapNet extractor may compute its own split independently.

`tools/build_gap_v3_session_ledger.py` produces
`data/gap_v3_session_ledger.json` recording:

- Per-file SHA-256, size, mtime, and game count for every JSONL under
  `data/human_games/`.
- Per-session `{session_id, session_hash=sha256(session_id), split, source_file}`,
  where `split = game_level_split(session_id)` (unchanged from
  `learned_ai.data.human_db_split`).
- `files_manifest_sha256` — a single hash of the sorted
  `(rel_path, sha256, size_bytes)` triples that captures source identity.
- Deterministic iteration and first-occurrence rule for duplicate session_ids.

12 regression assertions land in `tests/test_gap_v3_session_ledger.py`.

**Cost.**  Full-scan estimate ~7.6 h due to double file read (SHA + JSONL).
Smoke run on 20 files passes in 5.6 s.  Single-pass optimisation deferred until
the full ledger run is authorised.

### 5.2 Session-isolated dataset extraction (Batch 3b — LANDED, commit `ec567b2`)

`tools/extract_human_move_policy_dataset.py --session-ledger PATH --session-index PATH`
switches the split scheme from `three_way_split(state_key)` to strict single-tier
session isolation:

```
mask == 0b001 → include in TRAIN (state_key reached only by train-tier sessions)
mask == 0b010 → include in VAL   (state_key reached only by val-tier sessions)
mask == 0b100 → include in TEST  (state_key reached only by test-tier sessions)
any other mask → DROP (mixed-tier or uncovered)
```

Mixed-tier and uncovered state_keys are counted in
`session_split_disposition` per (`strict_train`, `strict_val`, `strict_test`,
`mixed_tier`, `uncovered`) and recorded in provenance.

Safety guards:

- Refuses to overwrite the v2 dataset directory `data/human_move_policy_dataset/`
  when `--session-ledger` is set; suggests
  `data/human_move_policy_dataset_v3_session/` instead.
- Verifies the session_index's referenced `metadata.npz` SHA-256 matches the
  currently on-disk file — refuses on mismatch to catch stale indexes.
- Verifies `game_split_mask` length agrees with `state_keys` length.

10 regression assertions land in `tests/test_gap_v3_hmpn_session_split.py`
(strict rule per mask value, no-train-leakage invariant, disposition sum,
SHA/length mismatch rejection, guard behaviours).

### 5.3 Session-aware trainer (Batch 3b — LANDED, commit `9efe0ba`)

`tools/train_human_move_policy_net.py` is unchanged in loss / optimiser /
architecture (still §3.5's 82 → 128 → 64 → 32 → 1 MLP with count-weighted
cross-entropy).  Two safety additions only:

- `_peek_dataset_provenance(dataset_dir)`: reads `metadata.npz` provenance
  without loading the memmap so `main()` can inspect `split_scheme` before
  starting training.
- `_guard_output_path(output, dataset_provenance)`: refuses to overwrite
  `data/human_move_policy_net_v2_candidate.npz` when the dataset was
  extracted with the session-ledger scheme.  Suggests the v3 filename.
  Any non-v2 output path is allowed — the guard only protects the v2
  filename.
- Session-ledger identity fields surfaced at the top level of the trained
  `.npz`'s provenance:  `dataset_split_scheme`, `session_ledger_sha256`,
  `session_ledger_files_manifest_sha256`, `session_ledger_version`,
  `session_index_sha256`.

9 regression assertions land in `tests/test_gap_v3_hmpn_trainer_ledger_guard.py`.

### 5.4 End-to-end pipeline (not yet run)

```
# Step 1 — build the frozen session ledger (~7.6 h, single-pass optimisation
# deferred until authorised).  Shared with GapNet Stage D extractor.
.venv/bin/python tools/build_gap_v3_session_ledger.py \\
    --games-dir data/human_games \\
    --output    data/gap_v3_session_ledger.json

# Step 2 — extract the session-isolated teacher training dataset.
.venv/bin/python tools/extract_human_move_policy_dataset.py \\
    --db             data/human_db_candidate.sqlite \\
    --session-ledger data/gap_v3_session_ledger.json \\
    --session-index  data/human_move_policy_session_index.npz \\
    --output-dir     data/human_move_policy_dataset_v3_session

# Readiness checkpoint — user reviews train / val / test state, event,
# band, phase counts before the ~18 h training run is authorised.

# Step 3 — train the v3 teacher candidate.
.venv/bin/python tools/train_human_move_policy_net.py \\
    --dataset-dir data/human_move_policy_dataset_v3_session \\
    --output      data/human_move_policy_net_v3_teacher_candidate.npz
```

### 5.5 Phase 5 gates (must pass before v3 teacher is trusted by GapNet)

Draft — thresholds to be reviewed with the user before the retrain runs.

| Gate | Threshold |
|------|-----------|
| Session-isolation invariant | Zero state_keys in the training pool have `game_split_mask & 0b110 != 0` (asserted at extract time, provenance records `session_split_disposition`). |
| Training pool size | `n_state_keys_train` ≥ target-TBD relative to v2 candidate (checklist coverage-floor concept applies here too — quantify before authorising the run). |
| Event-weighted NLL | Within a documented tolerance of the v2 candidate's held-out NLL; explicit target TBD.  Retraining on a smaller pool may lose 1–3 % NLL and still be the right trade for session-clean OOB. |
| ECE (calibration) | ≤ v2 candidate's ECE + a documented tolerance; explicit target TBD.  Same reasoning as NLL. |
| Provenance chain | `.npz` records `dataset_split_scheme = session_ledger_strict_single_tier`, `session_ledger_sha256`, `session_ledger_files_manifest_sha256` matching the frozen ledger; asserted by a future provenance regression test. |
| Compatibility | Loads under existing `ai/human_move_policy_advisor.py` without code changes (same architecture). |

### 5.6 Rollback

- Reversion base: any commit up to and including `2f36d69` predates Batch 3.
- The v2 candidate is not overwritten regardless of Phase 5 outcome.
- **v2 candidate is NOT a valid fallback for Stage D.**  Codex review 2026-08-12
  reminded us that Decision 6A (reuse v2 candidate) is off the table because
  the v2 candidate was trained under `three_way_split(state_key)`, which
  provides neither session-level nor state-key-level cleanliness against
  GapNet's held-out slices.  Consequently:
    * `data/human_move_policy_net_v2_candidate.npz` is retained solely as an
      **exploratory comparison** — any provenance record referring to it must
      label it as such.
    * If the v3 teacher fails a Phase 5 gate, GapNet Stage D **must delay**
      until a re-authored Phase 5 iteration produces a clean teacher.  Falling
      back to v2 would silently reintroduce the leakage the whole rebuild was
      designed to eliminate.
    * Any Stage D or Stage E promotion evidence generated using v2 as the
      teacher is inadmissible for the checklist's promotion gates.

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

## Blockers before Phase 2.2 (DB rebuild) — ALL CLOSED

- ✅ Perspective fixture tests pass (2.1).
- ✅ User authorised the candidate DB rebuild against a candidate path.
- ✅ Shared builder library extracted to `tools/_human_db_build.py` (reviewer §12).
- ✅ Sparse-table scheme confirmed (default: sparse).
- ✅ Target population — event-weighted default accepted.
- ✅ Candidate DB built, validated (`ok: True`), SHA-256 recorded above.

Phase 3 blockers (network training) — all closed:

- ✅ Regret mapping defined and versioned in `docs/gap_net_v3_plan.md`.
- ✅ `split_version` committed via `learned_ai/data/human_db_split.py`.

## Non-goals

- Rebuilding or retraining GapNet.
- Retiring HumanPrefNet — it remains a separately-purposed
  "competent filtered human style" model.
- Retraining any AI or shipping any inference change until Phase 4
  evaluation shows a clean improvement on held-out metrics.
- Filtering on provisional-rating status — the source records do not
  carry those flags.  Not implemented, not claimed.
