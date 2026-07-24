# Maintainer Upload Relocation — 25 July 2026

## Scope

This record documents an administrative relocation of files received from the
`main` maintainer. It does not activate a database, authorize training or
evaluation, change a frozen experiment, or alter checkpoint lineage.

The incoming directory formerly named `../Mills` was renamed
`../maintainer_inbox` and left empty. Future deliveries belong in dated or
otherwise uniquely named subdirectories there until they have been audited.
The two large database candidates from the 21 July delivery were moved to the
ignored archive:

```text
data/backups/maintainer_upload_20260721
```

The active `human_db_path` and `specialist_db_path` values were not changed.
No candidate-path keys were added to the local training registry, because that
strict registry accepts only supported runtime keys. The relative archive
paths in this record are the lookup index; using either candidate requires a
separate reviewed experiment contract.

## Archived Files and Purpose

| File | Identity | Purpose and status |
| --- | --- | --- |
| `human_db.sqlite` | 745,385,984 bytes; SHA-256 `F0B20D33AEFCBAB9AEDC8537F12FA2E53F7865B0387E2175AFD0EA32D1B90E42` | Maintainer-rebuilt HumanDB candidate containing human frequencies and outcomes plus `sector-corrected-v1` Malom metadata and labels. It is retained for a future explicitly contracted HumanDB query, evaluation, or training experiment; it is not the active HumanDB. |
| `human_db.sqlite.sha256` | Supplied checksum sidecar; its recorded digest matches the archived database | Delivery evidence only. It is not a database or runtime input. |
| `specialist_db.sqlite` | 290,820,096 bytes; SHA-256 `DF269D692E43815B88373F54B5AB1287022BC6736ECC8A5B95C7FB8A97FCD629` | Maintainer-rebuilt SpecialistDB candidate with 2,112,951 empirical positions, 60,117 winning lines, 30 preferred plays, and no persisted Malom labels. It preserves historical self-play and is not equivalent to either a fresh empty database or the completed `dev` baseline database. |

After the move, both databases produced `ok` from read-only immutable SQLite
`PRAGMA quick_check`. Their SHA-256 values still match the pre-move audit, and
the HumanDB sidecar still matches `human_db.sqlite`. The archive is excluded
from Git by the existing `data/backups/` rule.

## Small-file Disposition

The former incoming directory also held three small files. Keeping exact
duplicates beside the repository would create an ambiguous second source of
truth, so they were handled as follows:

- `retrain_v2_plan.md` was removed from the incoming directory after exact Git
  blob `5d9f9d646d69c1394632386c1c0a446c87366e51` was verified at commit
  `0920dd80510c1ea3df807c7654777f952b11268c` as
  `docs/retrain_v2_plan.md`.
- `train_s_gen_v2a.py` was removed after exact Git blob
  `60ab7ac1b42a8c3291ba19c6eeaba561b355f250` was verified at commit
  `9d09851f2b93e3cf3faf7736ec4d5c7bcbc9f4ab` as
  `scripts/train_s_gen_v2a.py`. Repository history remains the review source;
  the runtime entry point remains governed by the current quarantine and
  safety documents.
- `sanmill-api-plan.md` was deliberately discarded after the owner confirmed
  that it was obsolete. Future Sanmill integration questions should use the
  checkout indexed by `sanmill_checkout` and its current
  `docs/FRAMEWORK_API.md`, `docs/HUMAN_DATABASE.md`, and
  `docs/OPENING_BOOK.md`.

## Future Incoming-file Workflow

1. Place each new delivery in
   `../maintainer_inbox/<delivery-date-or-bundle-name>/`.
2. Record sender, delivery date, stated purpose, file size, and checksum before
   renaming, editing, or using a file.
3. Compare small source or document files with Git by content identity and
   history before creating another tracked copy.
4. Move retained large artefacts to an ignored, uniquely named archive only
   after recording their purpose and provenance.
5. Never configure training or evaluation directly against an inbox path.
   Activation requires a separate reviewed experiment decision.
