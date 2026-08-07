# Current Complete Python Baseline — 7 August 2026

## Result

The complete collected Python suite was executed at published `dev` commit
`f06d457c91091a1ae21463d04ad2928649e3ffb2`. The repository was on `dev`,
`dev == origin/dev`, and the tracked worktree was clean before collection and
execution.

Collection reported:

```text
1235 tests collected in 10.68s
```

The unfiltered execution reported:

```text
8 failed, 1227 passed, 498 subtests passed in 2876.61s (0:47:56)
```

This is a complete execution with no skipped test in the collected set. It is
not a clean all-pass claim.

## Command

The suite used the repository virtual environment, disabled only pytest's
cache provider, and used a fresh isolated temporary root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  -p no:cacheprovider `
  --basetemp=<fresh-system-temporary-directory> `
  --durations=20
```

Disabling the cache provider does not change test selection or assertions. No
test was deleted, skipped, deselected, retried inside the acceptance run, or
made dependent on a substitute asset.

## Failure Classification

All eight failures are machine-local historical Sanmill bridge tests:

- `test_local_perfect_routes_are_byte_stable_across_fresh_processes`;
- `test_local_complete_book_corpus_is_stable_across_fresh_processes`;
- `test_local_book_query_is_byte_stable_and_stream_remains_aligned`;
- `test_local_book_prefix_is_byte_stable_across_fresh_processes`;
- `test_local_pinned_sanmill_contract_and_book_gate`;
- `test_local_strict_position_error_is_immediate_and_stream_stays_aligned`;
- `test_local_logical_turn_is_reproducible_and_includes_mill_removal`; and
- `test_local_terminal_state_returns_zero_node_logical_result`.

A separate focused reproduction ran the eight exact node IDs and reported:

```text
8 failed in 1.78s
```

Every failure stopped at the same installation gate with
`Sanmill checkout changed pinned bridge source paths`. The moving Sanmill
checkout now contains many changes under the protected strict-UCI, rules,
Perfect DB, and related paths compared with the historical strict-v2 pin. The
tests correctly refused to reinterpret that checkout or its current binary as
the recorded historical dependency. None reached a query, search, replay, or
gameplay assertion.

The failure is therefore one known external identity condition repeated by
eight local integration tests, not eight independent NMM_LLM rule or search
regressions. It must not be hidden by adding skips or weakening the protected-
path check. Restoring a runnable historical test would require the exact
historical source and binary identity or a separately versioned replacement
contract.

## Slow-test Context

The longest duration was 1,346.13 seconds in setup for the complete 3v3
endgame builder property test. The Sentinel integration game took 379.54
seconds. Several fixed-search tactical tests took 30–60 seconds each. These
durations explain most of the complete-suite wall time but are not Generalist
training-throughput measurements.

## Readiness Meaning

All collected NMM_LLM unit, Malom/provenance, MIF, Generalist trainer,
checkpoint, exact-resume, managed-run, and successor-contract tests completed
without a reported failure. The eight external bridge failures do not block a
self-contained NMM_LLM Generalist experiment whose own rules and MIF
identities are frozen. They do block claiming that the floating local Sanmill
checkout satisfies the historical strict-v2 bridge contract.

Sanmill local commit
`a6623f88959f7453594df274fbe1f128af7ff55e` separately closes the newly
identified origin-counted referee-semantic difference, but it intentionally
does not make the current checkout byte-equivalent to the historical bridge.
Formal use of the new referee profile still requires remote publication, a
new clean pinned release binary, and a new versioned NMM_LLM bridge audit.

The complete-test gate is therefore adequate for preparing a new
self-contained successor plan, while the overall long-run verdict remains
`needs_decision` until the objective and resource envelope are frozen and a
separate product authorization is recorded.
