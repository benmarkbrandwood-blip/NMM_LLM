# Malom DB — Issues and Fix Plan

This document catalogues the known problems with how the Malom perfect-play database
is decoded and used throughout training, the specialist database, and the game.

Status (2026-07-20): Steps 1 and 3 are implemented and verified against Sanmill
plus the local 498-sector database. Steps 2, 4, and 5 remain pending; these
changes did not write to SpecialistDB or resume training.

---

## Issue 1 — Core decoder bug: sector value never applied (CRITICAL)

**File:** `ai/malom_db.py` — `decode_entry()`, lines 563–571

**What should happen (per Sanmill reference `database.rs`):**
```
absolute_key1 = raw_key1 + sector_value
W only when absolute_key1 == virt_win  (+299)
L only when absolute_key1 == virt_loss (-299)
everything else → D
```

**What the code does:**
```python
if key1 == virt_win:   return "W"
if key1 == virt_loss:  return "L"
# Non-virtual values may appear in intermediate sectors; map by sign
if key1 > 0:  return "W"
return "L"
```

`decode_entry` receives the raw 12-bit `key1` straight from the sector file. It never adds
the sector value (`secval`) to get the absolute key, so:
- It only returns "D" for the literal cases `key1 == 0` (exact zero) or `key1 == Count`
- Every other non-virtual value is classified as W or L by sign of the raw field
- This misclassifies ultra-strong draw grades (which have non-zero raw key1) as W or L

**Scale of the problem:** `std.secval` contains 498 sector entries; **491 of 498 have
non-zero sector values**. The decoder is wrong in the overwhelming majority of sectors.

**Why secvals are loaded but unused:** `MalomDB._load_secval()` correctly parses
`std.secval` and stores sector values in `self._secvals`. But `decode_entry` is a
standalone function — it has no access to `self._secvals` and no `sector_value` argument.
The fix requires passing the correct `sector_value` for the queried sector into `decode_entry`.

---

## Issue 2 — Cascading damage to SpecialistDB malom labels

**File:** `learned_ai/data/specialist_db.py` — `label_position_malom()`

During training, after each game the training scripts call `specialist_db.label_position_malom()`
for the 10 most consequential positions, writing the Malom W/D/L label to
`positions.malom_label`. This label is retained forever via COALESCE (won't be
overwritten by subsequent insertions).

Because every label written so far came from the broken decoder, all `malom_label`
entries in `data/specialist_db.sqlite` are potentially incorrect.

At inference, `scaffolded_encoder.py` reads these labels to populate the
counterfactual feature slots `[40:57]`. The model has been trained with wrong signals
in those slots and is now evaluated with wrong signals in those slots — consistently
wrong, but wrong.

**Consequence:** The SpecialistDB must be treated as Malom-label-contaminated.
Once the decoder is fixed, all positions need relabelling.

---

## Issue 3 — LookaheadAdvisor trajectory termination uses broken WDL

**File:** `learned_ai/models/lookahead_advisor.py`, lines 219–231

During training feature generation, `LookaheadAdvisor` uses `_endgame_db.query(b)` to
terminate lookahead trajectories early when it hits a "known" position. The result is
written directly into the 134-float feature matrix for the position.

A position that is actually a draw but reads as W or L from the broken decoder causes
the trajectory to terminate with the wrong value and fill the remaining ply slots with
that value. This has been happening for all ~12 000 generalist games and the specialist
training runs.

The generalist checkpoint and all specialist checkpoints have been trained on these
corrupted features.

---

## Issue 4 — `malom_win_move_rate = 1.0` is not a meaningful signal

**File:** `scripts/train_s_gen_v2.py`, lines 1021–1023

```python
malom_win_move_rate = _safe_mean(
    [1.0 for d in sd if d.malom_chosen_dtm is not None and d.malom_chosen_dtm >= 0] +
    [0.0 for d in sd if d.malom_chosen_dtm is not None and d.malom_chosen_dtm < 0]
)
```

`malom_chosen_dtm` stores the float returned by `ExternalSolvedDB.query_move_quality()`.
Despite the legacy field name, this is a WDL delta: `0` means the move preserves the
exact root value, while `-1` or `-2` means it downgrades that value. A positive delta
is impossible under exact minimax semantics and is now rejected as an adapter
contradiction. The `>= 0` check is therefore equivalent to asking whether the chosen
move preserved the Malom value; unavailable or inconsistent probes remain `None`.

Because the underlying WDL labels are wrong, a `1.0` rate here only means the model is
consistently choosing moves that the broken decoder agrees with — not that the moves are
theoretically sound. The training plot Row 0 (`malom_win_move_rate`) should not be
treated as a correctness signal until the decoder is fixed.

---

## Issue 5 — Mill + capture atomic action semantics

**File:** `learned_ai/sentinel/db_teacher.py` — `query_move_quality()`, line 165

```python
after_board = board.apply_move(move)
after = self._lookup(after_board)
```

When a move closes a Mill, the resulting board is in "capture pending" state — a piece
must still be removed. Malom does not represent this intermediate state; it only stores
positions where the board is at a complete, actionable game state.

`query_move_quality` queries Malom on the post-move board before the capture is resolved.
This will either return `None` (sector not found) or look up the wrong position entirely.

`ExternalSolvedDB._enumerate_legal_moves()` correctly expands Mill moves into one entry
per legal capture target (lines 196–207), but `query_move_quality` does not use this
expansion — it applies the move dict as-is. The fix: when the applied move closes a
Mill, the query should be skipped or the move dict must already include `capture`.

---

## Issue 6 — `MALOM_REWARD = 0.0` is a symptom not a fix

**File:** `scripts/train_s_gen_v2.py` (and open/mid/end variants)

```python
MALOM_REWARD = 0.0   # zeroed: per-step Malom reward incentivised safe draws over wins
```

The reward was zeroed because Malom was steering training away from wins. But the root
cause is likely the broken decoder: positions that are theoretically draws were being
read as losses, so taking "safe" moves was being penalised as bad. Zeroing the reward
removed the symptom but leaves the training loop without any Malom signal at all.

---

## Issue 7 — Malom used at inference through SpecialistDB

The v4 design intends Malom NOT to be queried directly at inference (both
`SpecialistRouter.score_moves` and `GeneralistAgent.score_moves` now pass `db=None`).
However, SpecialistDB's `malom_label` field is used at inference to populate the
counterfactual slots in the 134-float feature vector. This is indirect Malom usage at
inference — and it's carrying corrupted labels.

---

## Summary of affected components

| Component | Problem | Severity |
|---|---|---|
| `ai/malom_db.py` — `decode_entry` | Raw key1 compared without sector offset; 491/498 sectors wrong | Critical |
| `data/specialist_db.sqlite` — `malom_label` | All labels written from broken decoder; permanently stored | High |
| `learned_ai/models/lookahead_advisor.py` | Trajectory termination using broken WDL; corrupts training features | High |
| Training feature matrices (all checkpoints) | Trained on corrupted lookahead features | High |
| `malom_win_move_rate` metric | Measures agreement with broken labels; not a correctness signal | Medium |
| `ExternalSolvedDB.query_move_quality` | Does not handle Mill + capture atomicity | Medium |
| `MALOM_REWARD = 0.0` | Workaround that removes Malom signal entirely | Low (but blocks recovery) |

---

## Proposed fix sequence

### Step 1 — Fix `decode_entry` (completed 2026-07-20)

`decode_entry` needs `sector_value` for the queried sector passed in as an argument.
The call sites in `MalomDB.query()` already have `sector = (qW, qB, qWF, qBF)` and
`self._secvals[sector]` available — pass it through.

Correct logic:
```python
absolute_key1 = raw_key1 + sector_value
if absolute_key1 == virt_win:   return "W"
if absolute_key1 == virt_loss:  return "L"
return "D"   # every other legal value is a draw grade
```

Write a comparison test against the Sanmill reference for a sample of known positions
before using the fixed decoder for any labels or training.

### Step 2 — Rebuild SpecialistDB Malom labels

Once the decoder is verified correct, clear all `malom_label` fields in
`positions` and re-run a labelling pass over the stored positions. Alternatively,
drop and rebuild the DB from scratch — the position hashes are deterministic so
game re-play isn't needed; just re-label existing rows.

### Step 3 — Fix Mill atomicity in `query_move_quality` (completed 2026-07-20)

`query_move_quality` now validates the input against the rules engine's complete
atomic-action list before either Malom lookup. A Mill close without its mandatory
capture, a spurious capture, or any other illegal action returns `None`. A valid
action is applied once with its capture before the successor position is queried.

### Step 4 — Reinstate `MALOM_REWARD`

Once the decoder is correct, evaluate whether re-enabling a small per-step Malom
reward is appropriate. With correct signals, a W-preserving move should no longer
be penalised by the decoder noise that caused the original zeroing.

### Step 5 — Retrain or continue from checkpoint

The existing checkpoints were trained with corrupted lookahead features. Options:
- **Continue training** from the current checkpoint with fixed decoder and relabelled DB.
  The model will gradually learn to reinterpret the corrected feature signals.
- **Retrain from scratch** for a clean baseline. Likely only necessary if the
  corruption is judged severe enough that the current weights are actively misleading.

The generalist is at difficulty 6 with reasonable draw rates — continuing training
with the fix applied is probably sufficient unless ablation shows the corrupted
features were load-bearing.
