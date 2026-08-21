# Product D9/D10 Classical-First Default — 21 August 2026

## Status

The product owner directly authorized the D9 and D10 default-route change in
the source task. Implementation commit
`21e4a97e55d2ec34ca504ed171917e3ad00f2aaf` changes the human-versus-AI
default from specialist-first to classical-first, followed by the existing
single final `ProductPositionalSafetyGate`.

This is a product source change on `dev`. It is not a deployment, release,
training action, checkpoint change, model promotion, or model alias change.

## Decision basis

The product-owner decision follows the frozen held-out comparison:

- plan identity
  `bae8e6ad8d23ba42f6ac68e5a3b8dcb8e9d53a98670e53637486f97989d1b0e1`;
- result identity
  `89d24d3abc99811ee644a00e99d300c5721c8d1410a7c1458370ff15ce80f28d`;
- independent recomputation identity
  `7d4685acc5b5c8c6e01bd79f864d4455476e72a8dde14ca1f96fe105b0e1baf0`.

On each difficulty, classical-first minus specialist-first was
`+14.5833pp`, with a start-clustered 95% interval of
`[+11.4834pp, +17.6832pp]`. Both routes used the same final position-only
`A_pos` gate. The product owner, not the measurement harness, authorized this
separate source change.

## Source behavior

Before this change, `_ai_turn` enabled `SpecialistRouter` automatically when
the selected difficulty was at least 9. After this change:

1. D9 and D10 use the existing classical coordinator result by default.
2. That result reaches the same `_finalize_product_ai_move` choke used before.
3. The choke invokes the existing `ProductPositionalSafetyGate` exactly once.
4. `SpecialistRouter` remains available only through the explicit
   `use_overseer_player` product control.
5. The separate explicit generalist route and its precedence contract remain
   unchanged.
6. D1 through D8 retain their classical default and low-difficulty safety-gate
   bypass.

The three specialist checkpoints, their aliases, router implementation, and
specialist tests are unchanged. The focused verification includes their
tracked checkpoint-identity test. No checkpoint was loaded for gameplay or
modified by this task.

The difficulty selector now describes D9 and D10 as classical AI with
positional safety instead of incorrectly labelling their default as
specialist AI. `/api/overseer_status` exposes a stable product-route contract,
specialist availability, and the last actual decision source. The per-turn
log now records the actual source, difficulty, and final safety status.

## Safety and fallback boundary

The safety algorithm and trust contract did not change:

- only Malom metadata version `sector-corrected-v1` is trusted;
- the safe set is position-only `A_pos`;
- this is not history-aware `A_allow`;
- the strict game rules remain authoritative for terminal outcomes.

When Malom is unavailable or a query fails, the existing gate returns the
already computed classical move so the game remains playable. The server log
and status endpoint explicitly report that play as unfiltered. The product
must not present this fallback as an enabled or successful safety filter.

## Red-green evidence

The minimum route test was written and executed before changing the product
implementation:

```text
.\.venv\Scripts\python.exe -m pytest `
  tests/test_specialist_positional_safety.py::test_d9_d10_default_route_does_not_enable_specialist_override `
  -q --basetemp tmp/pytest-product-route-red
```

It failed one test because `_spec_by_diff` still enabled the automatic
specialist override. After the implementation change, the focused source and
runtime-contract suite passed:

```text
.\.venv\Scripts\python.exe -m pytest `
  tests/test_specialist_positional_safety.py `
  tests/test_product_positional_safety.py `
  tests/test_product_malom_runtime.py `
  tests/test_sanmill_product_route_heldout.py `
  tests/test_specialist_db_checkpoint_identity.py `
  -q --basetemp tmp/pytest-product-route-focused
```

Result: 38 tests passed. These tests prove the default D9/D10 route does not
invoke the specialist override, the explicit specialist and generalist paths
remain present, all product sources converge on the same final choke, D1-D8
remain outside the high-difficulty gate, status and logs expose the source,
and the specialist checkpoint identities remain available.

The repository-required Malom, DB-teacher, and label-provenance command was:

```text
.\.venv\Scripts\python.exe -m pytest `
  tests/test_malom_db.py `
  tests/test_sentinel_db_teacher.py `
  tests/test_malom_label_provenance.py `
  -q --basetemp tmp/pytest-product-route-malom
```

Result: 103 tests and 498 subtests passed. Pytest emitted only the known host
warning that its optional `.pytest_cache` could not be written; the isolated
`--basetemp` work completed normally.

Task-scope Ruff and syntax checks passed:

```text
ruff check tests/test_specialist_positional_safety.py
ruff check --select E9,F63,F7,F82 web/app.py
.\.venv\Scripts\python.exe -m py_compile web/app.py `
  tests/test_specialist_positional_safety.py
git diff --check
```

The bytecode cache was directed to `tmp/` because the tracked test directory
is not writable for `__pycache__`. A complete `ruff check web/app.py` reports
56 inherited findings. The parent version reports the same 56 findings, and
none is on a line changed by this task. They were not hidden, weakened, or
expanded into an unrelated cleanup.

## Access and change audit

This task started no game, evaluation, training, fitting, or tuning process.
It wrote no database, loaded no protected segment, read no remaining source
pool record, and changed no model, checkpoint, alias, training data, route
measurement, frozen plan, authorization, ledger, or result.

Tracked product changes are limited to `web/app.py`, the difficulty text in
`web/templates/index.html`, and focused tests. Existing untracked `tmp/`
content was preserved.

## Claim boundary

The held-out basis contains 54 placement-origin and 54 movement-origin starts,
with no flying-origin start. Games entered the flying phase, but this is not
independent flying-origin coverage.

Neither the held-out result nor this source change is an overall-strength,
human-opponent, causal-mechanism, equivalence, or history-aware safety claim.
The change does not authorize deployment, release, external publication,
training, promotion, or deletion of the specialist research route.
