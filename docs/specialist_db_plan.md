# SpecialistDB Malom-Label Migration

## Status

This document was imported from the maintainer's `main` branch on 22 July
2026. Its original in-place command is superseded. Repository policy forbids
modifying or relabelling the isolated legacy SpecialistDB, and the active
corrected baseline database must not be replaced by an implicit migration.

The maintainer has also supplied a rebuilt candidate in the sibling `Mills`
staging directory. A read-only audit found:

| Property | Observed value |
| --- | --- |
| SHA-256 | `DF269D692E43815B88373F54B5AB1287022BC6736ECC8A5B95C7FB8A97FCD629` |
| SQLite `quick_check` | `ok` |
| Positions | 2,112,951 |
| Winning lines | 60,117 |
| Preferred plays | 30 |
| Persisted Malom labels | 0 |
| Malom label version | `sector-corrected-v1` |

That file is an inactive candidate. It preserves a large empirical self-play
history and is therefore not equivalent to the empty SpecialistDB used by the
completed fresh `dev` baseline. Activating it requires a separately frozen
experiment decision; it must not silently replace the current database.

## Contamination Boundary

The legacy database's non-null `malom_label` values were written before the
sector-offset decoder correction. Its empirical `wins`, `draws`, `losses`,
`winning_lines`, and `preferred_plays` do not become Malom labels and may be
preserved for a separately defined experiment.

The `pos_hash` key is a SHA-1 digest of a canonical state and is not reversible.
Consequently, the old label rows cannot be reconstructed and re-queried in
place. A migration can only clear them and allow newly encountered positions
to acquire corrected labels later.

## Safe Copy Migration

The migration utility never edits its source. It requires all three of:

- an explicit source path;
- a new, non-existing output path;
- the expected SHA-256 identity of the source.

Example with deliberately unspecified machine-local values:

```powershell
.\.venv\Scripts\python.exe scripts\fix_specialist_db_malom_labels.py `
  --source <reviewed-legacy-copy> `
  --output <new-corrected-copy> `
  --expected-sha256 <reviewed-source-sha256>
```

The utility opens the source read-only, checks its identity and SQLite
integrity, copies it through SQLite's backup API, clears Malom labels only in
the new output, writes `malom_label_version=sector-corrected-v1`, and verifies
that position counts and database integrity remain intact. It refuses:

- identical source and output paths;
- an existing output;
- a source hash mismatch;
- a source already carrying current label provenance;
- an unknown non-empty label-version value.

The resulting database has trusted provenance for labels written after the
migration, but initially has no Malom labels. Its retained self-play statistics
must still be identified as legacy empirical data in any run manifest.

## Non-Negotiable Boundaries

- Never run a write operation against either legacy snapshot recorded in
  `docs/local-training-layout.md`.
- Never copy a migrated or downloaded database over the active corrected
  baseline path without a new experiment contract and recoverable backup.
- Never describe retained empirical rows as retrained after the decoder fix.
- Verify the exact database hash, label version, label count, and intended role
  during training preflight.
