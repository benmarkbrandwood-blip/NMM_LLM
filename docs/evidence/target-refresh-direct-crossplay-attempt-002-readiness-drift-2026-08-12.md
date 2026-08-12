# Target-refresh direct cross-play attempt-002 readiness drift

## Status

Attempt-002 is superseded before authorization or launch. No game, policy
sample, ledger, result, optimizer update, database write or checkpoint write
occurred.

Its persisted readiness identity was
`5f416731f16502461104b382fda431a673cbe40df3c54dc85f9166c78c9ef71f`.
A fresh build from the same clean published commit and the same immutable plan
produced
`3b17bd7d6d8c982717b01c7effef3f72053bef624996c394244f04baaacf2943`.
The launch gate would therefore have failed closed even if the former identity
had been authorized.

## Deterministic reproduction

The complete semantic comparison found exactly one changing fact, represented
twice as the preflight before/after observation:

- `human_db.sqlite-shm` persisted mtime:
  `1786453342597313700` ns;
- freshly observed mtime: `1786493064901697000` ns.

The before and after value inside each individual preflight were equal. The
HumanDB main-file identity, schema probe, Malom identity, checkpoints, strict
referee, source commit and schedule did not differ.

## Cause and correction boundary

The direct cross-play opens HumanDB with SQLite `immutable=1`, so only the
main database file belongs to its runtime view. A different SQLite reader or
the database owner may update shared-memory sidecar metadata without changing
that immutable main-file snapshot. Binding the absolute `-shm` mtime into the
readiness identity therefore made a valid preflight expire for an unrelated
external event.

The correction excludes WAL/SHM metadata from readiness identity and from the
source-drift decision. It does not make source integrity permissive:

- the immutable HumanDB main-file identity and `quick_check` remain required;
- main-file size/mtime drift remains fail closed;
- the Malom manifest and `std.secval` hash remain required;
- Malom file size/mtime drift remains fail closed;
- the runner still records volatile sidecar observations as telemetry; and
- HumanDB continues to open through the immutable main-file route.

Focused tests require sidecar-only changes to preserve the stable observation
identity and require HumanDB-main or Malom changes to alter it. A fresh
attempt must bind the corrected implementation and use a new isolated output
directory. Attempt-002 must not be overwritten or launched.

This is execution-integrity evidence only. It is not training, held-out
evaluation, strength, promotion, publication or long-run authorization.
