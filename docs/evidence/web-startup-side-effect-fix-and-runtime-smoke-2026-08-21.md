# Web startup side-effect fix and runtime smoke -- 21 August 2026

## Outcome

The product startup defect is fixed and the implementation is published at
`dev == origin/dev == 718c7ef8baa5882b5b23264302170d0487498957`.
Web startup now opens HumanDB as an immutable read-only query snapshot and
loads the optional N-gram opponent model only from an existing valid cache.
When the cache is absent, N-gram use is visibly disabled and no raw game
corpus is enumerated or parsed.

The one authorized loopback runtime smoke was consumed and **failed closed**.
The service started in about three seconds and each of the three permitted GET
requests returned HTTP 200 exactly once.  A PowerShell response-capture bug
then discarded the response bodies, so the required runtime assertions about
the JSON fields and rendered controls cannot be proved.  No request was
repeated and the service was not restarted.  This record must not be described
as a passed runtime smoke.

The negative smoke result does not invalidate the independently tested startup
fix.  It means only that this once-only run did not produce complete runtime
acceptance evidence.

## Authority and immutable baseline

The product owner directly authorized this bounded source fix, one ordinary
`dev -> origin/dev` fast-forward publication of each verified commit, and one
loopback smoke.  The smoke allowed only GET `/api/ping`, GET
`/api/overseer_status`, and GET `/`; it prohibited games, WebSockets, sessions,
move inference, Sanmill, training, model changes, and database writes.

Work began from the expected clean tracked baseline:

- branch and source: `dev == origin/dev == ddf12571588ff51ecb806bafed73e237ef5f4bd9`;
- tracked worktree: clean;
- preserved unrelated ignored/untracked content: `tmp/`;
- previous failed smoke evidence:
  `docs/evidence/product-classical-first-runtime-smoke-2026-08-21.md`.

The prior failure was correctly attributed to product import behavior, not to
the smoke operator.  With `data/ngram_model.json` absent, `web.app` called
`NGramOpponentModel.load_from_games` and recursively scanned the raw human
corpus.  HumanDB was also opened in writable WAL mode, which changed the SHM
sidecar mtime.

## Red evidence

The new focused test file was written before the implementation.  The first
host-default pytest invocation produced the expected code failures but also
four unrelated setup errors because the host pytest temporary root was not
accessible.  It was rerun against a repository-local ignored base directory:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_web_startup_resources.py -q `
  --basetemp tmp\pytest-web-startup-red-20260821 `
  -p no:cacheprovider
```

Result: `8 failed`.  The failures independently demonstrated that the old
source still called `load_from_games`, directly opened `HumanDB`, did not
publish startup-resource state, and lacked the side-effect-bounded loader.
The behavioral fixtures used only isolated temporary files and fakes.  They
did not enumerate or read the real raw game corpus.

## Implementation

Implementation commit:
`718c7ef8baa5882b5b23264302170d0487498957`.

The change is intentionally limited to startup resource handling:

1. `web/startup_resources.py` adds a cache-only N-gram loader.  A missing,
   empty, or invalid cache disables the optional model, records the reason,
   logs that raw scanning is disabled, and never falls back to
   `load_from_games`.
2. The same module opens Web HumanDB with `read_only=True, immutable=True`.
   A runtime query view delegates reads but makes attempted dynamic HumanDB
   additions visible and does not forward them to the immutable database.
   Normal completed-game record persistence remains separate.
3. `web/app.py` no longer contains a Web startup or status path that enumerates
   `data/human_games`.  Historical TrajectoryDB scanning is not used as an
   import fallback.  The offline N-gram builder remains available in
   `ai/ngram_opponent_model.py` and was not invoked or changed.
4. `/api/overseer_status` now includes `ngram_opponent_model` and
   `human_db_runtime`.  The former includes the cache-only mode and disabled
   reason; the latter includes explicit read-only and immutable flags.
5. D1--D8, D9/D10 classical-first routing, explicit specialist/generalist
   routes, `ProductPositionalSafetyGate`, `A_pos`, and the
   `sector-corrected-v1` trust contract were not changed.

One process-only metadata deviation is preserved rather than rewritten: the
implementation commit body contains a literal `\n` between its two sentences.
The author and committer are both `Calcitem <calcitem@outlook.com>`.  Amend,
rebase, or other history rewriting was forbidden, so the published commit was
not rewritten merely to repair message formatting.

## Green verification

Focused and affected-subsystem command:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_web_startup_resources.py `
  tests\test_ngram_opponent_model.py `
  tests\test_specialist_positional_safety.py `
  tests\test_product_positional_safety.py `
  tests\test_product_malom_runtime.py -q -rs `
  --basetemp tmp\pytest-web-startup-focused-rs-20260821 `
  -p no:cacheprovider
```

Result: `57 passed, 1 skipped`.  The skip is the existing isolated-process
native-runtime test at `tests/test_product_positional_safety.py:334`; it is
explicitly marked as requiring a separate process and was not weakened.

Mandatory provenance command:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_malom_db.py `
  tests\test_sentinel_db_teacher.py `
  tests\test_malom_label_provenance.py -q `
  --basetemp tmp\pytest-web-startup-malom-20260821 `
  -p no:cacheprovider
```

Result: `103 passed, 498 subtests passed`.

The combined run produced `160 passed, 1 skipped, 498 subtests passed`.
The nine startup-resource tests passed, including a real temporary SQLite
fixture proving that immutable HumanDB access did not create WAL/SHM files or
change the database bytes or nanosecond mtime.

Additional checks:

- `ruff check web\startup_resources.py tests\test_web_startup_resources.py`:
  passed;
- parent `web/app.py` Ruff inventory: 56 existing findings;
- modified `web/app.py` Ruff inventory: 53 findings, with no new finding;
- `py_compile` for the two changed modules and new test, using an ignored local
  pycache prefix: passed;
- `git diff --check`: passed.

The implementation commit was fetched, checked as a fast-forward descendant,
and published normally from `dev` to `origin/dev` before the smoke.

## One-time runtime smoke

### Pre-snapshot

The run used exact published commit `718c7ef8...`, loopback port `58797`, and
one Uvicorn process launched directly from `.venv`.  No relevant process or
listener existed before launch.  `data/ngram_model.json` was absent.

Selected frozen identities before launch were:

| Resource | SHA-256 |
| --- | --- |
| HumanDB main | `d8e22da38273f7c26eb76803ae91fc3fae711f508383ffbe3096c2946912b440` |
| HumanDB WAL | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| HumanDB SHM | `fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb` |
| Malom manifest file | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| specialist open | `d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701` |
| specialist mid | `a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2` |
| specialist end | `5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8` |
| generalist | `494cec3f78d3b8f8f05d61d30a7c620d796cc386d5e355ca3fdaa5e3d16a792f` |

HumanDB main/WAL/SHM high-resolution mtimes before launch were respectively
`639199818960000000`, `639222004749625022`, and
`639228980568546120` UTC ticks.

### Startup observations

The service process was PID `54336`.  It logged server start at
`2026-08-21 17:10:34 +08:00` and completed application startup at about
`17:10:37`, rather than scanning the raw corpus for ten minutes.

The append-only server-log range was lines 7--27.  It explicitly recorded:

- HumanDB immutable read-only load: 2,152,889 positions and 94,429 games;
- TrajectoryDB historical scan skipped;
- N-gram disabled because the cache file is missing, with raw game scanning
  disabled;
- the three specialists and explicit generalist loaded read-only;
- Malom selected from `local-registry:malom_db_path` with the trusted manifest;
- the position-only specialist `A_pos` filter enabled with
  `sector-corrected-v1`.

The aggregate HumanDB itself has no trusted historical Malom label version;
those old label columns were visibly disabled.  This is separate from the
product positional-safety Malom runtime, which passed its tracked
`sector-corrected-v1` manifest and inventory gate.

The existing optional Malom hash prewarm logged a non-fatal missing
`ai.malom_puzzle_search` module.  This did not disable the validated product
Malom runtime and was not changed in this task.

### HTTP and fail-closed reason

Uvicorn access evidence shows exactly three requests, each once:

- GET `/api/ping`: HTTP 200;
- GET `/api/overseer_status`: HTTP 200;
- GET `/`: HTTP 200.

The PowerShell collector used `$home` for the third response.  PowerShell
variable names are case-insensitive, so that name collided with the read-only
automatic `$HOME` variable.  Although all three requests completed, the final
structured output expression then failed and none of the response bodies was
durably captured.

The requests were not repeated.  Consequently this run does **not** prove the
runtime JSON values for route, safety, N-gram, HumanDB, or
`last_decision_source`, nor does it prove the rendered checkbox state from the
actual HTTP response.  Their source contracts and focused tests passed, but
those are not substituted for the missing runtime response evidence.

### Shutdown and post-snapshot

Ctrl+C released port `58797`, but PID `54336` remained alive without a
listener and did not complete normal process teardown.  It was force-stopped
as the exact smoke-owned PID.  The transcript does not establish completion
within the separate 30-second shutdown-confirmation allowance, so that bound
is not claimed.  Final audit found no relevant process and no listener on
8000, 8080, or 58797.

HumanDB main, WAL, and SHM were byte-, size-, existence-, and mtime-identical
after shutdown.  The three UTC tick values above were unchanged.  Settings,
the local path registry, Malom manifest, four checkpoints, N-gram absence, and
the two implementation source files were also identical.

The only snapshotted file change was the permitted server log:

- size: 480 -> 3,070 bytes;
- lines: 6 -> 27;
- SHA-256: `0e5e597d626717c5fe0d3e6d1158a8539dc64fe99f7424f4e0d74b2c99598b3a`
  -> `7fc8664d08a2e7dacf754ff8a8cc773aa3ba6ccbff58fd6aeef2750454a380fb`.

No game, WebSocket, session, move inference, Sanmill process, browser, Ollama,
training, fitting, database write, checkpoint change, alias change, or raw
human-corpus scan occurred.

## Decision and boundary

The source change is complete, verified, and published.  The one-time runtime
smoke classification is `failed_closed_response_capture`; it is not runtime
acceptance.  The once-only authorization is consumed, so no retry or restart
is permitted under this task.

This work supplies no gameplay, strength, human-opponent, deployment, release,
history-aware safety, or `A_allow` claim.  The product safety contract remains
position-only `A_pos` under the exact validated `sector-corrected-v1` runtime.
