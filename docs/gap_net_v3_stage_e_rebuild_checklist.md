# GapNet v3 Stage D+E rebuild — checklist

Compiled 2026-08-06 after Codex review of commit `d1df6de`; revised same day with
factual corrections from the user.  Tracks progress and reversion points for the
corrective work required by the plan/implementation contract gaps.  Any deviation
from what is checked here must be discussed before implementing.

## Locked decisions (2026-08-06)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Split | **1B** — rebuild from JSONL with session-level split *before* aggregation (plan §8.4). |
| 2 | Band conditioning | **2A** — 82 features = 79 board + 3-way Elo-band one-hot.  Architecture: 82 → 128 → 64 → 32 → 3. |
| 3 | Baseline reference | **3A** — extractor emits `targets_uniform.f32.bin`; empirical G_v is the *reference* on high-support rows.  Global training mean kept as a separate "mean-predictor" baseline. |
| 4 | Component D | **4A** — dropped for regret_v1.  Three heads only.  D returns under a future regret version once specified and tested. |
| 5 | D4 augmentation | **5A** — CLI flag `--d4-augmentation {on,off}`, default off.  Choice recorded in provenance.  D4-on model trained separately as controlled ablation. |
| 6 | HMPN leakage | **6B** — retrain HumanMovePolicyNet on training-session data only.  Retain v2 candidate untouched as exploratory comparison; new teacher gets a separate name.  Teacher and GapNet share one frozen session ledger, source manifest, and split seed. |

### Correction on Decision 6

Codex's earlier framing of 6A assumed the current v2 policy net was already trained on a
session-level split (aligning it with a future GapNet session split).  That assumption is
**wrong**: `data/human_move_policy_net_v2_candidate.npz` was trained via
`three_way_split(state_key)` — a state-key hash split.  `game_split_mask` in
`data/human_move_policy_session_index.npz` was only ever an evaluation diagnostic; it did
not gate training.  Therefore reusing the current teacher would leave both session-level
and state-key-level leakage, not just state-key-level.  6A is off the table.

Option 6C (5-fold OOF) is not required for the first clean candidate.  Note also that a
future 6C variant must define folds **over sessions**, not state_keys — a state-key fold
OOF would not provide strict session-level OOF.

## Artefacts (do NOT delete without approval)

| Artefact | Status |
|----------|--------|
| `data/gap_net_v3_dataset/` | Existing state-key-split dataset.  Rename to `data/gap_net_v3_dataset_state_key_split/` before running the new extractor.  Keep as pipeline evidence / smoke-test data. |
| `data/gap_net_v3_candidate.npz` | Not yet built. |
| `data/human_move_policy_net_v2_candidate.npz` | Current P_h teacher.  Retained untouched as exploratory comparison — Decision 6B rebuild produces a **separately named** artefact (`data/human_move_policy_net_v3_teacher_candidate.npz` proposed). |
| `data/human_move_policy_session_index.npz` | 2,209,726-entry `game_split_mask` + `player_split_mask`.  Reference for both the frozen session ledger and the extractor rewrite. |
| `d1df6de` | Last commit before rebuild.  Reversion base. |

## Rebuild plan

### Step 0 — Frozen shared session ledger — ✅ TOOL LANDED (Batch 3a, commit 6d61d40)

**Must land before Stage D extractor rewrite AND HumanMovePolicyNet retrain.**

- [x] Freeze the JSONL source manifest: `tools/build_gap_v3_session_ledger.py` records
      per-file SHA-256, size, mtime, and `n_games` for every JSONL under `--games-dir`.
      Output: `data/gap_v3_session_ledger.json`.
- [x] Split seed: uses `learned_ai.data.human_db_split.game_level_split()` unchanged,
      recorded in ledger provenance as `split_function` + `split_manifest_version`.
- [x] Per-session records include `session_hash = sha256(session_id)` for owning-tier
      tie-break, and `split = game_level_split(session_id)`.
- [x] Deterministic scan order + first-occurrence rule for duplicate session_ids.
- [x] `files_manifest_sha256` captures source identity (single hash of the sorted
      `(rel_path, sha256, size_bytes)` triples).
- [x] Ledger consumers (Batch 3b, Batch 4) verify this hash and refuse mismatches.
- [ ] **Full ledger run not yet executed.**  Smoke run passed on 20 files (5.6 s);
      full-scan cost ~7.6 h due to double file read (SHA + JSONL) — single-pass
      optimisation deferred until we're ready to run the full ledger.

### Stage D redo — extraction rewrite — ✅ TOOLING DONE, RUN PENDING (Batch 4, commits fdd5a97 + 93cc7da + Codex 2026-08-15 fixes)

- [x] Rewrite `tools/extract_gap_v3_dataset_v2.py`:
  - [x] Read `data/human_games/*.jsonl` under the frozen manifest via the session ledger's `files` manifest.
  - [x] Use `session_id → split_tier` from the frozen ledger — no recomputation.
  - [x] **Multi-tier state assignment rule** implemented per Decision (owning-tier rule):
    1. For each state_key, enumerate the set of sessions that reached it.
    2. Compute the **smallest SHA-256 session-hash** across those sessions — that session's split_tier becomes the state_key's **owning tier**.
    3. Aggregate move counts using events **from the owning tier only**.
    4. **Discard** events for that state from all other tiers.
    5. **Never** combine counts across tiers before or during tier assignment.
    6. Record, per (band, phase), the counts of `states_discarded_other_tier_by_band_phase`, `events_discarded_other_tier_by_band_phase`, and `states_kept_by_band_phase` in provenance.
  - [x] Compute empirical `P_h` from per-(state_key, band) owning-tier counts; gate by frozen `min_support` (default 25).  Denominator is legal-only (Codex P1 2026-08-14: illegal notations are dropped and reported separately as `empirical_illegal_events_seen`).
  - [x] Compute model `P_h` via `HumanMovePolicyAdvisor.probs_strict()` from the v3 teacher (no silent uniform fallback; Codex P1 2026-08-14 + P2 2026-08-15 regression test).
  - [x] Compute uniform `P_h = 1/n_legal` per position.
  - [x] **A/B/C target discipline (fail-closed):**
    - [x] For each emitted row, **all three** components are finite in `targets.f32.bin` AND `targets_uniform.f32.bin`.
    - [x] If any component of R_v is unavailable for any legal successor at the parent, **abstain the entire row** — write to `abstained.jsonl` with reason and do not emit.
    - [x] `targets_empirical.f32.bin` is the **only** target file where NaN is permitted — NaN entries there indicate empirical support was below the frozen `min_support` threshold.
  - [x] Emit `metadata.npz` with `state_keys`, `band_idx`, `split`, `phase`, `mover_color`, `n_legal`, `ph_source`, `owning_session_min_hash`, and provenance (as `provenance` field with `production_ready` + `non_ready_reasons`).
  - [x] Emit provenance (embedded in metadata.npz): session-ledger SHA + files_manifest_sha256, teacher SHA + verified lineage, min_support value, per-(band, phase) discarded/kept counts, `git_commit`, emitted-per-(band, phase).
  - [x] Emit `abstained.jsonl` per abstention reason (bad_board, no_legal_moves, n_legal_lt2, parent_no_malom, not_emittable including `teacher_probs_failed:*` and `non_finite_target`).
  - [x] Output directory: `data/gap_net_v3_dataset_v2/` (no-clobber guard refuses existing outputs unless `force=True`).
  - [x] **Coverage floor:** if final emission falls below the configured floor, extractor sets `gate_status="halt_coverage_floor"`, marks `production_ready=False` with reason `coverage_floor not met`, and reports per-(band, phase) counts.  Stage E trainer refuses to consume non-ready datasets unless `--allow-non-production-dataset` is passed, and any such run propagates a `promotion_eligible=false` taint into the saved NPZ (Codex 2026-08-15 P1).
- [ ] **Full extraction run not yet executed.**  Awaiting authorisation + HMPN v3 teacher.

### Stage D-a — HumanMovePolicyNet retrain (Decision 6B) — 🟡 TOOLING DONE, RUN PENDING (Batch 3b, commits ec567b2 + 9efe0ba)

- [x] Rebuild the teacher's training dataset using the frozen session ledger:
  - [x] `tools/extract_human_move_policy_dataset.py --session-ledger PATH --session-index PATH`
        switches split scheme to strict single-tier session isolation.
  - [x] State_keys appearing in val/test-tier sessions are dropped from the training pool
        (mask must equal exactly `0b001`/`0b010`/`0b100`; mixed and uncovered dropped).
  - [x] Guard: refuses to overwrite the v2 dataset at `data/human_move_policy_dataset/`
        when `--session-ledger` is set.
  - [x] Provenance records `session_ledger_sha256`, `session_ledger_files_manifest_sha256`,
        `session_index_sha256`, `session_split_disposition`.
- [x] Trainer safety guard (`tools/train_human_move_policy_net.py`):
  - [x] `_peek_dataset_provenance` inspects `split_scheme` before starting training.
  - [x] `_guard_output_path` refuses to overwrite `data/human_move_policy_net_v2_candidate.npz`
        when the dataset was extracted with the session-ledger scheme.
  - [x] Surfaces `dataset_split_scheme`, `session_ledger_sha256`,
        `session_ledger_files_manifest_sha256`, `session_ledger_version`,
        `session_index_sha256` at the top level of the trained `.npz`'s provenance.
- [ ] **Readiness checkpoint before launching the ~18 h teacher training run:**
      report train / val / test **state, event, band, phase counts** to the user.  This is a
      gate, not authorisation — the user reviews the counts before green-lighting the run.
- [ ] Train under `data/human_move_policy_net_v3_teacher_candidate.npz` (guard enforces
      this; v2 candidate stays untouched).
- [ ] Record the frozen session ledger SHA + JSONL manifest SHA + split seed in the new
      teacher's `.npz` provenance (auto-surfaced from dataset provenance chain).

### Stage E redo — training rewrite — ✅ DONE (commits 728ddad + 0e0224b)

- [x] Rewrite `tools/train_gap_net_v3.py`:
  - [x] 79 board features + 3 band one-hot → 82 input.
  - [x] 82 → 128 → 64 → 32 → 3 MLP.
  - [x] Load all three targets files (`targets`, `targets_uniform`, `targets_empirical`).
  - [x] `--d4-augmentation {on,off}` flag, default off.  Board block permuted only; band one-hot invariant.
  - [x] Per-band val metrics reported for every component.
  - [x] Gate semantics (per Decision 3):
    - [x] On high-support rows (empirical present): compare candidate / teacher / uniform against empirical *as reference*.
    - [x] Report teacher-fidelity MSE separately for model-only rows — do NOT label as empirical validation.
    - [x] Mean-predictor baseline (train target mean) reported separately.
  - [x] Fail-closed finite-target validation on load.
  - [x] Provenance records D4 flag, per-band gate results, all source SHAs.

### Tests

Landed to date — 62/62 assertions pass locally:

Stage E trainer coverage (commits 728ddad + 0e0224b):
- [x] `tests/test_gap_v3_band_onehot.py` — band one-hot at feature dims 79–81 (4).
- [x] `tests/test_gap_v3_three_head.py` — 3 output heads, save/load roundtrip (3).
- [x] `tests/test_gap_v3_gate_formulas.py` — hand-computed gate MSEs (5).
- [x] `tests/test_gap_v3_fail_closed.py` — model + uniform targets strict-finite; only empirical may hold NaN (8; expanded from 6 after Codex Blocker 1 fix).
- [x] `tests/test_gap_v3_d4_flag.py` — D4 augmentation permutes only board block (5).
- [x] `tests/test_gap_v3_no_zero_default.py` — missing Malom never becomes 0 (6).

Session ledger + HMPN retrain coverage (Batch 3a + 3b):
- [x] `tests/test_gap_v3_session_ledger.py` — ledger determinism, session-hash correctness, manifest hash sensitivity, duplicate handling (12).
- [x] `tests/test_gap_v3_hmpn_session_split.py` — strict single-tier rule, no-train-leakage invariant, disposition counts, `_load_state_key_masks` SHA/length checks, `_guard_output_dir` on default v2 dataset path (10).
- [x] `tests/test_gap_v3_hmpn_trainer_ledger_guard.py` — `_peek_dataset_provenance`, `_guard_output_path` blocks/allows across all (split_scheme × output_path) combinations (9).

Pending — write against synthetic data (no full training required):

- [ ] `tests/test_gap_v3_provenance.py` — required provenance fields present under a synthetic save.
- [ ] `tests/test_gap_v3_provenance_roundtrip.py` — save/load preserves every provenance field.

Landed — Batch 4 session-isolation + owning-tier + Codex 2026-08-14/15 hardening coverage:

- [x] `tests/test_gap_v3_session_split_isolation.py` — end-to-end: no val/test session events leak into GapNet Stage D train aggregation (5).
- [x] `tests/test_gap_v3_owning_tier_rule.py` — multi-tier state_key assignment picks smallest-hash session's tier and discards other-tier events (5).
- [x] `tests/test_gap_v3_extractor_v2_hardening.py` — P1-B/P1-A ledger consumption, no-clobber, sha drift, missing/orphan file handling (11).
- [x] `tests/test_gap_v3_extractor_v2_p1_followup.py` — Codex 2026-08-14 P1×5+P2×1 + 2026-08-15 P1×2+P2×2 (probs_strict abstain seam, taint save/load round trip, between-pass file deletion, manifest-driven iteration) (31).

Pending — depend on a D4-on trained model:

- [ ] `tests/test_gap_v3_symmetry_invariance.py` — inference invariance under D4 to within 1e-3.

### Promotion-gate freeze (before running Stage E) — 🟡 STRUCTURE LANDED, NUMBERS PENDING USER REVIEW (Batch 5)

- [x] Update `docs/gap_net_v3_plan.md` §16 wording to match Decision 3A reference-based framing.
- [x] Add per-band thresholds — **structure landed** (Gate 1 `X_A[b, c]` ≥ 30 % vs uniform; Gate 2 `X_B[b, c]` ≤ 20 % vs teacher, both per-(band × component)).  **Numeric values are initial drafts**; user reviews before Stage E run.
- [x] Explicitly note teacher-fidelity is not empirical validation (bolded under "Separately reported" in the Stage E cell).

## Current position (2026-08-15)

Batch progress:
- ✅ Batch 3a — session ledger builder + 12 tests (commit `6d61d40`).
- ✅ Batch 3b — HMPN extractor + trainer session-ledger flags + guards + 19 tests (commits `ec567b2` + `9efe0ba`).
- ⏳ Batch 3c — HMPN plan doc (`docs/human_move_policy_net_plan.md`) amendment describing v3 teacher retrain pipeline.  Not yet started.
- ✅ Batch 4 — Stage D GapNet extractor rewrite consuming the session ledger + session-isolation/owning-tier tests + Codex 2026-08-14 P1-B/P1-A hardening + Codex 2026-08-14 P1×5+P2×1 follow-up + Codex 2026-08-15 P1×2+P2×2 follow-up + Codex 2026-08-15 3rd-pass frozen-threshold registry / Stage F wording (commits `fdd5a97` → `93cc7da` → `da596ef` → `affd2cd` → this commit).  Tooling done; full extraction run still pending authorisation and HMPN v3 teacher.
- ⏳ Batch 5 — Promotion-gate freeze (per-band thresholds in `docs/gap_net_v3_plan.md` §16).
- ⏸️ Batch 6 — Results (blocked on runs; runs blocked on readiness checkpoints and user authorisation).

Runs pending user authorisation:
- Full session-ledger run (`build_gap_v3_session_ledger.py`, ~7.6 h estimated; may want single-pass optimisation first).
- HMPN v3 teacher retrain (~18 h) — readiness checkpoint required before launch.
- GapNet v3 Stage D re-extraction (Batch 4 must land first) — coverage floor gate.
- GapNet v3 Stage E training run — promotion gate wording must be frozen first (Batch 5).

## Progress log

- 2026-08-06 — Codex review of `d1df6de`; user locked Decisions 1–5.
- 2026-08-06 — Checklist created; Decision 6 open.
- 2026-08-06 — `tools/train_gap_net_v3.py` rewritten for new spec (commit `728ddad`).  Five focused tests drafted, passing.
- 2026-08-06 — User corrected 6A premise (HMPN v2 trained via `three_way_split(state_key)`, not sessions); Decision 6 locked to 6B.  Multi-tier rule tightened; A/B/C target discipline corrected (unavailable → abstain row, not NaN); coverage floor + readiness checkpoint added; no-zero-default regression added.  Checklist committed as `2f36d69`.
- 2026-08-06 — Batch 3a session ledger builder + 12 tests committed as `6d61d40`.  Smoke run on 20 JSONL files passed in 5.6 s.
- 2026-08-11 — Codex review of `728ddad` found two blockers:
    - Blocker 1: `_load_split` accepted NaN in `targets.f32.bin` and `targets_uniform.f32.bin`.  Fixed to require strict-finite (only `targets_empirical` may hold NaN).  Fail-closed tests inverted + expanded 6 → 8 assertions.  Commit `0e0224b`.
    - Blocker 2: `docs/gap_net_v3_plan.md` still specified 79→128→64→32→4 / four heads and marked old state-key-split Stage D as ✅ PASSED.  Reconciled with a Corrections preamble + inline SUPERSEDED markers on §11.1 and §16 Stage D/E rows.  Commit `2a152af`.
- 2026-08-11 — Windows locale portability fix: `ai/mills_llm.py` reads `phase_strategy.md` with explicit `encoding="utf-8"`.  Predates our commits, flagged by Codex as non-blocking.  Commit `4853296`.
- 2026-08-11 — Batch 3b extractor: `--session-ledger` flag, strict single-tier rule, `_load_state_key_masks`, `_apply_session_ledger_split`, `_guard_output_dir` + 10 tests.  Commit `ec567b2`.
- 2026-08-11 — Batch 3b trainer: `_peek_dataset_provenance`, `_guard_output_path`, session-ledger identity surfaced at top-level provenance + 9 tests.  Commit `9efe0ba`.
- 2026-08-11 — Checklist updated with current-position section + progress log entries.
- 2026-08-12 — Batch 4 Stage D extractor (v2) + Stage E gate hardening: owning-tier rule, session-isolation, per-band gate thresholds, executable gate formulas (commits pre-`fdd5a97`).
- 2026-08-13 — Codex review of Batch 4 hardened: `_verify_ledger_complete` P1-B fail-closed helper, Windows fsync via `os.open(O_RDWR)`, gate false-positive fix.  Commit `bb8fca9`.
- 2026-08-14 — Batch 4 v2 extractor with hardened ledger consumption (commit `fdd5a97`) + six Codex 2026-08-14 findings (P1×5 + P2×1: probs_strict, legal-only empirical denominator, teacher lineage, production_ready flag, per-(band, phase) discard counts, emitted_by_band_phase counter).  Commit `93cc7da`.
- 2026-08-15 — Codex review of `93cc7da` (BLOCK): four findings addressed —
    - P1 between-pass file-deletion detection: `_iter_jsonl_events` refactored to be manifest-driven (frozen-manifest iteration; missing-on-disk → fatal).  Regression test `test_iter_events_detects_between_pass_deletion`.
    - P1 taint propagation into saved model NPZ: `_verify_dataset_production_ready` now returns structured taint dict; `main()` captures it and writes `dataset_production_ready`, `dataset_non_ready_reasons`, `non_production_override`, `promotion_eligible` into provenance; model label degrades to `gap_net_v3_candidate_nonproduction` when tainted.  Save/load round-trip regressions added.
    - P2 executable regression for `probs_strict` abstain path: extracted `_compute_teacher_ph` helper; test `test_compute_teacher_ph_abstains_on_degenerate_teacher` proves (a) probs() would return uniform, (b) probs_strict() raises, (c) extractor seam returns `teacher_probs_failed:` reason.
    - P2 checklist reconciliation: this file updated to reflect landed Batch 4 status.
- 2026-08-15 — Codex 2nd-pass review of `da596ef` (BLOCK): four more findings addressed —
    - P1 promotion_eligible tied to dataset readiness ALONE (silent for gate FAIL / all-SKIP).  Fix: extracted `_build_saved_provenance` helper — the SINGLE construction point for provenance + model label.  `promotion_eligible` now requires **dataset_eligibility AND stage_e_gate_verdict == "PASS" AND gate_thresholds_frozen_id** (new CLI `--gate-thresholds-frozen-id STRING`; empty/None → not promotable).  Model label degrades to `gap_net_v3_candidate_nonpromotion` when any condition fails; `promotion_ineligible_reasons` enumerates why.  Seven builder unit tests cover all promotion states.
    - P1 dataset provenance parsed by truthiness.  Fix: new `_validate_dataset_provenance_contract` enforces strict contract — `production_ready` must be exact JSON boolean (`is True`/`is False`, not `bool()`); `non_ready_reasons` must be a list; `gate_status` must be a non-empty string; invariants A–D check ready↔reasons↔gate consistency.  Malformed provenance is a hard stop and CANNOT be bypassed by `--allow-non-production-dataset`.  Ten strict-contract regression tests including the specific Codex cases: `"true"` string, `1` int, missing fields, ready-with-reasons, ready-with-halt-gate, non-ready-with-empty-reasons, string-instead-of-list, and override-does-not-bypass.
    - P2 round-trip tests recreate provenance manually.  Fix: builder is the single seam; two round-trip tests now use the builder directly.  Bounded integration test `test_main_integration_records_all_promotion_fields` invokes `trainer.main()` end-to-end on a synthetic 60-row dataset via `sys.argv` monkeypatching (1 epoch, ~1s) and asserts every promotion field survives from builder through `_save` to the reloaded NPZ.  This catches main-path wiring drift Codex flagged as uncaught.
    - Evidence-correction note: Codex 2026-08-15T08:57+08 recorded 184 passed + 9 skipped for `tests/test_gap_v3*.py` at `da596ef` (9 tests require a full-dataset fixture unavailable in Codex CI).  Local pass count differs due to fixture availability.
- 2026-08-15 — Codex 3rd-pass review of `affd2cd` (BLOCK): three narrow items addressed —
    - P1 identity-bound frozen thresholds: introduced `configs/stage_e_thresholds.json` registry with one initial `frozen=false` draft entry.  `--gate-thresholds-frozen-id` now MUST resolve to a registry entry with `frozen=true`; the entry's canonical sha256 + full body are recorded in provenance.  Trainer additionally range-validates x_a ∈ [0, 1], x_b ∈ [0, 1], min_high_support > 0, min_denominator > 0 finite — applied to both the registry entry and the runtime CLI values.  CLI values must equal the frozen entry EXACTLY or the run fails.  Negative regression added: a 2×-worse candidate with permissive thresholds (e.g. x_a=-2.0, x_b=1.0) is rejected before the gate math runs.
    - P1 Stage E ≠ promotion: builder now records `stage_e_passed` and `stage_f_eligible` separately; `promotion_eligible` is ALWAYS False from this trainer.  Model label is `gap_net_v3_stage_e_candidate` (when Stage F eligible) or `gap_net_v3_stage_e_candidate_ineligible` — the historical `gap_net_v3_candidate` name is now reserved for the Stage F artifact.  Promotion requires Stage F single-use artifact + remaining plan gates + a separate live-promotion decision, per plan §13/§16.
    - P2 evidence + checklist prose: this file corrected — stale reference to `test_main_integration_produces_valid_promotable_npz` fixed to actual name `test_main_integration_records_all_promotion_fields`; "forthcoming taint/deletion/probs_strict fixes" wording superseded by the landed commit chain fdd5a97 → 93cc7da → da596ef → affd2cd → this commit.  Codex 2026-08-15T12:23+08 recorded 202 passed + 9 skipped at `affd2cd` (fixture-gated skips); this commit's new registry / range / Stage-F tests add further coverage.

## Reversion

- Any step reverts by `git checkout d1df6de -- <path>` for that path.
- Dataset rename is reversible (`mv data/gap_net_v3_dataset_state_key_split/ data/gap_net_v3_dataset/`).
- Existing HumanMovePolicyNet v2 candidate is preserved unchanged regardless of Decision 6B outcome.
- Existing state-key-split GapNet dataset is preserved as exploratory pipeline evidence.
