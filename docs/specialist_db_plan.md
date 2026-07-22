# Specialist DB — Malom Label Fix Plan

## Current state

```
data/specialist_db.sqlite
  total positions  : 2,112,817
  malom_label rows : 364,262   (W: 209,451 / D: 41,548 / L: 113,263)
  malom_label_version (meta) : None   ← no version tag written
```

The 364,262 Malom labels were written by the **old buggy decoder** (sector offset not
applied before W/D/L classification — many wins and losses were reported as draws).
The new `SpecialistDB` code checks `meta.malom_label_version == "sector-corrected-v1"`
on open; because the tag is missing (`None`), `malom_labels_trusted = False` and
`require_trusted_malom_labels()` will block any training run that tries to consume or
append Malom labels.

The self-play WDL counts (`wins`, `draws`, `losses` columns) are **unaffected** — they
are empirical game results with no Malom involvement.  Only `malom_label` is corrupt.

## Why we cannot re-label in-place

`pos_hash` is SHA-1 of the D4-canonical FEN string.  SHA-1 is not reversible, and the
original board objects are not stored in the DB, so we cannot reconstruct each position
to re-query Malom.  Re-labelling would require replaying every training game from the
original trajectory logs — impractical given volume.

## Fix strategy: clear labels, stamp version, re-accumulate

Clearing all `malom_label` values leaves the DB in a state equivalent to a self-play-only
database that has never seen Malom.  The new `SpecialistDB` constructor auto-adopts
`sector-corrected-v1` when the label count is zero, so the DB becomes immediately trusted.
New correct labels accumulate naturally during the next training run via
`label_position_malom()` (called by `db_teacher.py`).

## Migration steps

### 1. Back up
```bash
cp data/specialist_db.sqlite "data/specialist_db.sqlite.pre-malom-fix-$(date +%Y%m%d).bak"
```

### 2. Run migration script (see below)
```bash
.venv/bin/python scripts/fix_specialist_db_malom_labels.py
```

### 3. Verify
```bash
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('data/specialist_db.sqlite')
print('positions total :', conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0])
print('malom labels    :', conn.execute('SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL').fetchone()[0])
print('version tag     :', conn.execute('SELECT value FROM meta WHERE key=\"malom_label_version\"').fetchone())
conn.close()
"
```

Expected output:
```
positions total : 2,112,817   (unchanged)
malom labels    : 0
version tag     : ('sector-corrected-v1',)
```

## Migration script to create: scripts/fix_specialist_db_malom_labels.py

```python
"""Clear corrupt pre-sector-corrected Malom labels from specialist_db.sqlite.

Self-play WDL counts are preserved.  New correct labels will accumulate
during the next training run using the fixed decoder.
"""
import sqlite3
from pathlib import Path

from learned_ai.data.malom_label_provenance import (
    CURRENT_MALOM_LABEL_VERSION,
    write_current_malom_label_version,
)

DB_PATH = Path("data/specialist_db.sqlite")

conn = sqlite3.connect(str(DB_PATH))
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

version_row = conn.execute(
    "SELECT value FROM meta WHERE key='malom_label_version'"
).fetchone()
existing_version = version_row[0] if version_row else None

if existing_version == CURRENT_MALOM_LABEL_VERSION:
    print("DB already has sector-corrected-v1 labels — nothing to do.")
    conn.close()
    raise SystemExit(0)

before = conn.execute(
    "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
).fetchone()[0]
print(f"Clearing {before:,} Malom labels (version={existing_version!r}) ...")

with conn:
    conn.execute("UPDATE positions SET malom_label = NULL WHERE malom_label IS NOT NULL")
    # Remove stale version tag if present, then write current
    conn.execute("DELETE FROM meta WHERE key='malom_label_version'")
    write_current_malom_label_version(conn)

after = conn.execute(
    "SELECT COUNT(*) FROM positions WHERE malom_label IS NOT NULL"
).fetchone()[0]
new_version = conn.execute(
    "SELECT value FROM meta WHERE key='malom_label_version'"
).fetchone()[0]

print(f"Done.  Labels remaining: {after}  |  version tag: {new_version!r}")
conn.close()
```

## After migration

- Training can resume immediately; `require_trusted_malom_labels()` will pass.
- `db_teacher.py` (now using the corrected Malom decoder) will re-populate
  `malom_label` on newly encountered positions over time.
- The Win/Draw/Loss ratio will likely shift compared to the old labels: the old
  decoder over-reported draws, so expect more W and L labels going forward.

## Notes

- The `data/human_db.sqlite` is a separate database and has its own Malom label
  columns (`positions.malom_wdl` and `moves.malom_wdl_after`).  Check its status
  separately using `malom_label_provenance.human_db_has_malom_labels()` before
  any human-db annotation run.
- The `data/specialist_db.sqlite.pre-fix-backup` that exists in the repo already
  predates this plan and was taken before a different fix; use the new timestamped
  backup from step 1 as the restore point.
