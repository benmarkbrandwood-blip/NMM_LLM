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

### Step 0 — Frozen shared session ledger

**Must land before Stage D extractor rewrite AND HumanMovePolicyNet retrain.**

- [ ] Freeze the JSONL source manifest: enumerate every game file in `data/human_games/`
      with its SHA-256, mtime and byte size.  Record as `data/gap_v3_session_ledger.json`.
- [ ] Frozen split seed: adopt `b"gap_v3_session_split_v1"` (or user-approved value).
- [ ] Compute per-session `game_level_split(session_id)` under that seed.  Record
      `session_id → split_tier` and per-session SHA-256 hash for the tie-break rule.
- [ ] This ledger is the single source of truth for both the teacher retrain (Decision 6B)
      and the GapNet Stage D extractor.  Neither may compute its own split independently.

### Stage D redo — extraction rewrite

- [ ] Rewrite `tools/extract_gap_v3_dataset.py`:
  - [ ] Read `data/human_games/*.jsonl` directly under the frozen manifest (Step 0).
  - [ ] Use `session_id → split_tier` from the frozen ledger — do NOT recompute.
  - [ ] **Multi-tier state assignment rule** (explicit, no shortcuts):
    1. For each state_key, enumerate the set of sessions that reached it.
    2. Compute the **smallest SHA-256 session-hash** across those sessions — that session's split_tier becomes the state_key's **owning tier**.
    3. Aggregate move counts using events **from the owning tier only**.
    4. **Discard** events for that state from all other tiers.
    5. **Never** combine counts across tiers before or during tier assignment.
    6. Record, per (band, phase), the counts of `states_discarded_other_tier` and `events_discarded_other_tier` in `provenance.json`.
  - [ ] Compute empirical `P_h` from per-(state_key, band) owning-tier counts; gate by frozen `min_support` (default 25).
  - [ ] Compute model `P_h` via `HumanMovePolicyAdvisor` from the new teacher (Decision 6B).
  - [ ] Compute uniform `P_h = 1/n_legal` per position.
  - [ ] **A/B/C target discipline (fail-closed):**
    - [ ] For each emitted row, **all three** components (class_downgrade, wdl_utility_loss, ordinal_rank_loss) must be finite in `targets.f32.bin` AND `targets_uniform.f32.bin`.
    - [ ] If any component of R_v is unavailable for any legal successor at the parent, **abstain the entire row** — write to `abstained.jsonl` with reason and do not emit.  This replaces the earlier (wrong) draft that would have written NaN into required targets.
    - [ ] `targets_empirical.f32.bin` is the **only** target file where NaN is permitted — NaN entries there indicate empirical support was below the frozen `min_support` threshold, not R_v unavailability.
  - [ ] Emit `metadata.npz` with `state_keys`, `band_idx`, `split`, `phase`, `mover_color`, `n_legal`, `ph_source`, `owning_session_min_hash`, `provenance`.
  - [ ] Emit `provenance.json`: session-ledger SHA + seed, JSONL manifest hash, HumanMovePolicyNet SHA + version, min_support value, per-(band, phase) discarded counts, `git_commit`.
  - [ ] Emit `abstained.jsonl` per abstention reason (including all-legal-successor R_v unavailability).
  - [ ] Output directory: `data/gap_net_v3_dataset_v2/`.
  - [ ] **Coverage floor:** if final emission falls below X% of the state-key-split dataset's row count (X to be frozen with user before run), **halt and report** actual per-(band, phase) counts.  Do NOT fall back to mixed aggregates silently.

### Stage D-a — HumanMovePolicyNet retrain (Decision 6B)

- [ ] Rebuild the teacher's training dataset using the frozen session ledger:
  - [ ] Include only positions reached by **train-tier sessions** under the ledger.
  - [ ] State_keys appearing in val/test-tier sessions are dropped from the teacher training pool.  Keep an eval subset for the teacher's own diagnostic gates.
- [ ] **Readiness checkpoint before launching the ~18 h teacher training run:**
      report train / val / test **state, event, band, phase counts** to the user.  This is a
      gate, not authorisation — the user reviews the counts before green-lighting the run.
- [ ] Train under an explicitly separate output name (proposed: `data/human_move_policy_net_v3_teacher_candidate.npz`).  Do not overwrite `data/human_move_policy_net_v2_candidate.npz`.
- [ ] Record the frozen session ledger SHA + JSONL manifest SHA + split seed in the new teacher's `.npz` provenance.

### Stage E redo — training rewrite

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

Existing (prepared, unstaged, 23 assertions passing locally):

- [x] `tests/test_gap_v3_band_onehot.py` — band one-hot at feature dims 79–81.
- [x] `tests/test_gap_v3_three_head.py` — 3 output heads, save/load roundtrip against tiny synthetic model.
- [x] `tests/test_gap_v3_gate_formulas.py` — hand-computed gate MSEs.
- [x] `tests/test_gap_v3_fail_closed.py` — `_load_split` refuses non-finite targets.
- [x] `tests/test_gap_v3_d4_flag.py` — D4 augmentation permutes only board block.
- [x] `tests/test_gap_v3_no_zero_default.py` — missing Malom never becomes 0 (per plan §14 no-zero-default contract).

Pending — write against synthetic data (no full training required, per user 2026-08-06):

- [ ] `tests/test_gap_v3_provenance.py` — required provenance fields present under a synthetic save.
- [ ] `tests/test_gap_v3_provenance_roundtrip.py` — save/load preserves every provenance field.

Pending — depend on extractor rewrite:

- [ ] `tests/test_gap_v3_session_split_isolation.py` — no val/test session events leak into train aggregation.
- [ ] `tests/test_gap_v3_owning_tier_rule.py` — multi-tier state assignment picks smallest-hash session's tier and discards other-tier events.

Pending — depend on a D4-on trained model:

- [ ] `tests/test_gap_v3_symmetry_invariance.py` — inference invariance under D4 to within 1e-3.

### Promotion-gate freeze (before running Stage E)

- [ ] Update `docs/gap_net_v3_plan.md` §16 wording to match Decision 3A reference-based framing.
- [ ] Add per-band thresholds.
- [ ] Explicitly note teacher-fidelity is not empirical validation.

## Progress log

- 2026-08-06 — Codex review of `d1df6de`; user locked Decisions 1–5.
- 2026-08-06 — Checklist created; Decision 6 open.
- 2026-08-06 — `tools/train_gap_net_v3.py` rewritten for new spec (unstaged).
- 2026-08-06 — Five focused tests drafted, passing (unstaged).
- 2026-08-06 — User corrected 6A premise (HMPN v2 trained via `three_way_split(state_key)`, not sessions); Decision 6 locked to 6B.  Multi-tier rule tightened; A/B/C target discipline corrected (unavailable → abstain row, not NaN); coverage floor + readiness checkpoint added; no-zero-default regression added to tests batch.

## Reversion

- Any step reverts by `git checkout d1df6de -- <path>` for that path.
- Dataset rename is reversible (`mv data/gap_net_v3_dataset_state_key_split/ data/gap_net_v3_dataset/`).
- Existing HumanMovePolicyNet v2 candidate is preserved unchanged regardless of Decision 6B outcome.
- Existing state-key-split GapNet dataset is preserved as exploratory pipeline evidence.
