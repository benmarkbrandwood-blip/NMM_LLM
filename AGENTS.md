# Repository Working Agreement

## Start Here

The directory containing this file is the only Git repository and must be the
primary Codex workspace. Its parent directory is a data container, not a
repository. Do not initialise an outer repository there.

Before changing code or Git history:

1. Run `git rev-parse --show-toplevel` and confirm that it returns this
   repository.
2. Read the current handover in
   [`docs/handoff/windows-training-2026-07-20.md`](docs/handoff/windows-training-2026-07-20.md).
3. Read [`docs/local-training-layout.md`](docs/local-training-layout.md)
   before moving, regenerating, or deleting any training data.
4. Inspect `git status --short --branch` and the local/remote commit graph.

## Git Safety

- Do not use a blind `git pull` to resolve rewritten but patch-equivalent
  commits. Establish the commit graph and patch equivalence first.
- Do not push, force-push, merge, rebase, or rewrite history unless the user
  explicitly authorises that operation.
- Preserve unrelated user changes and ignored local training artefacts.
- Keep one independently justified fix per commit.
- Write commit subjects and bodies in English. Unless the user asks for a
  different format, wrap commit-message body lines at 72 ASCII characters.

## Local Paths and Large Data

- Machine-specific paths belong in `data/training_paths.local.json`. This
  file is ignored and must not be committed.
- Keep the Malom tablebase and source archives outside the repository.
- Keep imported SQLite databases, recursive game records, endgame tables,
  generated checkpoints, and backup snapshots ignored unless the repository
  already contains an explicit tracked exception.
- Never overwrite or relabel the isolated legacy SpecialistDB. Its location
  and checksum are recorded in the local-layout document.

## Malom and Training Safety

- Treat project rules and the repository's independently tested semantics as
  authoritative. Sanmill is a useful reference implementation, not a reason
  to bypass analysis of this codebase.
- Treat machine-local handover notes and screenshots as historical context,
  not as authoritative facts or acceptance evidence. Verify their claims
  against current code, configuration, artefacts, and tests before acting.
- Persisted Malom labels are trusted only when their metadata version is
  `sector-corrected-v1`.
- The imported HumanDB's human frequencies and outcomes are usable, but its
  unversioned historical Malom columns are not training labels.
- Historical specialist, generalist, Sentinel, value-net, and gap-net
  artefacts must retain their recorded provenance. Do not silently describe
  them as retrained after the decoder correction.
- Do not start a long training run until its checkpoint lineage, output
  directory, database paths, and smoke-test result have been recorded.
- In particular, review the known `--auto-resume-best` path issue documented
  in the handover before starting the generalist trainer.

## Proportionate Verification

For Malom, DB-teacher, or label-provenance changes, run at least:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_malom_db.py `
  tests/test_sentinel_db_teacher.py `
  tests/test_malom_label_provenance.py -q
```

Also run focused tests for every modified subsystem. Do not present the full
test suite as a clean baseline until its known missing/stale internal-interface
collection failures have been resolved.
