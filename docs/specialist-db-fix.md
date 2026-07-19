# SpecialistDB Fix Plan

Companion document to `docs/malom-fix.md`. That doc establishes that the Malom
decoder (`ai/malom_db.py`) has been writing wrong W/D/L labels into
`data/specialist_db.sqlite` since the DB was first created. This document covers
what to do about the DB itself.

---

## What is contaminated

Every row in the `positions` table that has a non-NULL `malom_label` was written
from the broken decoder. The `COALESCE` rule (`ON CONFLICT DO UPDATE SET
malom_label = COALESCE(positions.malom_label, excluded.malom_label)`) means the
first (wrong) label is permanent — subsequent insertions never overwrite it.

These labels are read at inference to populate the counterfactual feature slots
`[40:57]` of the 134-float per-move feature vector. The model has been trained
with these slots carrying wrong values and is evaluated with the same wrong values —
so the corruption is at least self-consistent. But once the decoder is fixed, the
trained weights and the DB labels will diverge.

The `wins`, `draws`, `losses` columns are NOT contaminated — those are empirical
counts from actual self-play outcomes and are independent of Malom decoding.

The `winning_lines` and `preferred_plays` tables are also not directly contaminated
(they don't store Malom labels), though their win-rate stats are based on self-play
outcomes which may themselves have been influenced by wrong Malom rewards during training.

---

## Options

### Option A — Clear all malom_labels, relabel after decoder fix

Fastest path. SQL:

```sql
UPDATE positions SET malom_label = NULL;
```

Then run a one-off labelling pass after the decoder is corrected. The training
scripts already call `specialist_db.label_position_malom()` after each game; a
standalone script could iterate over all rows and query the corrected decoder.

**Pros:** Preserves all empirical `wins/draws/losses` data (12 000+ games worth).
All canonical position hashes and winning_lines entries remain intact.

**Cons:** Until the relabelling pass runs, the counterfactual slots are blank (0.5)
at inference — same as when Malom is not present. This is safe but removes the DB
signal temporarily.

### Option B — Drop and rebuild the DB from scratch

Delete `data/specialist_db.sqlite` and let training rebuild it from zero.
New training games (with fixed decoder) will produce correct labels from the start.

**Pros:** Completely clean slate; no risk of stale data interfering.

**Cons:** Loses all 12 000+ games' worth of empirical position statistics.
The model was already trained on those statistics; losing them doesn't make the
checkpoint worse, but new training starts without the warm DB.

### Option C — Keep the DB as-is and retrain with fixed decoder

Don't touch the DB now. When the decoder is fixed:
1. The wrong `malom_label` values remain but training will now see correct Malom
   signals via the per-game labelling calls.
2. `COALESCE` still protects existing labels from being overwritten — so positions
   already in the DB keep their wrong labels indefinitely.

**Pros:** Zero disruption to ongoing training.

**Cons:** Wrong labels accumulate forever. The counterfactual feature slots will
carry a mix of correct and incorrect values for an indefinite period. Not recommended.

---

## Recommended approach: Option A

1. **Back up the DB first:**
   ```bash
   cp data/specialist_db.sqlite data/specialist_db.sqlite.pre-malom-fix
   ```

2. **Clear all Malom labels:**
   ```bash
   sqlite3 data/specialist_db.sqlite "UPDATE positions SET malom_label = NULL;"
   ```

3. **Apply the decoder fix** (see `docs/malom-fix.md` Step 1).

4. **Run a relabelling pass** — a small standalone script that:
   - Iterates over all rows in `positions` where `malom_label IS NULL`
   - Decodes the `pos_hash` back to a board state (the canonical hash is stored)
   - Queries the corrected `MalomDB.query()` for each position
   - Writes the result via `label_position_malom()`

   Note: the canonical D4 board hash does not store the side-to-move or piece
   counts explicitly — the relabelling pass will need to store enough state to
   reconstruct a `BoardState`. The simplest approach is to re-run a short training
   pass (even 100 games) with the fixed decoder; this will naturally relabel the
   positions that get touched during play. Full relabelling of all 300k+ position
   rows requires a dedicated tool.

5. **Update `label_position_malom()` in `specialist_db.py`** to assert that a new
   correct label matches any existing label rather than silently skipping on
   conflict (follow the v5 recommendation: disagreement should be flagged, not silenced).

---

## COALESCE behaviour after the fix

After clearing labels (Option A) and restarting training with the fixed decoder,
the first correct label written for each position will be retained by COALESCE.
Subsequent insertions won't overwrite it — which is the desired behaviour once the
label is known to be correct.

If there is any uncertainty about a label (e.g. the decoder is still under test),
add an `allow_overwrite=True` flag to `label_position_malom()` that uses
`DO UPDATE SET malom_label = excluded.malom_label` unconditionally, so that
corrections can propagate.

---

## Impact on existing checkpoints

The specialist and generalist checkpoints trained so far have learned a mapping
from 134-float features (which include the wrong counterfactual slots) to move
probabilities. After the DB is cleaned and the decoder fixed:

- At **inference**: the counterfactual slots will temporarily read 0.5 (unknown)
  until relabelling is complete. The model will behave as if Malom is absent —
  same as before SpecialistDB was built.
- At **training**: the corrected decoder starts writing correct labels. New games
  will see correct signals in those slots. The model will gradually learn the
  corrected relationship.
- A **full retrain from scratch** is only necessary if the wrong labels caused
  pathological weight configurations. Given the low weight assigned to the
  counterfactual slots relative to other features, continued training from the
  current checkpoint is likely sufficient.

---

## Files to change

| File | Change |
|---|---|
| `ai/malom_db.py` | Fix `decode_entry` — apply sector value before W/L classification (see malom-fix.md) |
| `learned_ai/data/specialist_db.py` | `label_position_malom()` — add `allow_overwrite` flag; add assertion that correct labels match on conflict |
| `data/specialist_db.sqlite` | Clear `malom_label` column before first relabelling pass |
| `scripts/relabel_specialist_db.py` | New script: iterate positions, query corrected decoder, write labels (write after decoder fix is verified) |
