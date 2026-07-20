"""Clear corrupt pre-sector-corrected Malom labels from specialist_db.sqlite.

Self-play WDL counts are preserved. New correct labels will accumulate
during the next training run using the fixed decoder.
"""
import sys
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

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
