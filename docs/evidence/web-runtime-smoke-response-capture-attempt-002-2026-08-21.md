# Web runtime smoke response-capture attempt 002 -- 21 August 2026

## Decision

The one newly authorized loopback service attempt is
`failed_closed_shutdown_gate`.

The three actual HTTP response bodies were durably captured and independently
recomputed.  They prove the requested product runtime contract for this exact
process: N-gram was cache-only and visibly disabled because its cache was
absent; HumanDB was available through an immutable read-only handle; D9/D10
reported `classical-first`; explicit Specialist remained opt-in; the final
choke was `ProductPositionalSafetyGate`; and the enabled safety contract was
position-only `A_pos` with `sector-corrected-v1`.  The rendered home page also
contained both classical-plus-positional-safety labels and an unchecked
explicit Specialist control.

The overall smoke still failed closed.  Ctrl+C did not make PID `76456` exit
naturally.  It was force-stopped at 29.395254 seconds after the stop request,
and disappearance of both PID and listener was observed at 30.384039 seconds.
That exceeded the separate 30-second shutdown-confirmation limit by
0.384039 seconds.  The service was not restarted and no request was repeated.

This result preserves, and does not rewrite, the preceding
`failed_closed_response_capture` evidence.  It also does not change product
code or authorize another smoke.

## Authority and immutable source

The product owner directly authorized one new zero-game loopback smoke only
to correct response variable/persistence handling and to make shutdown timing
exact.  The authorization prohibited product-code or test changes, any fourth
HTTP request, a second Uvicorn attempt, games, WebSockets, sessions, inference,
Sanmill, training, database writes, deployment, and release.

Before consuming the attempt, `git fetch origin dev` completed and the source
state was:

- branch: `dev`;
- HEAD: `f4d08283a90fcef99fee42c97c9e1f48b5eda0e6`;
- `origin/dev`: the same commit;
- `origin/dev` was an ancestor of HEAD;
- tracked worktree: clean;
- pre-existing untracked state: `tmp/` only; and
- product implementation commit
  `718c7ef8baa5882b5b23264302170d0487498957` was an ancestor of HEAD.

The prior evidence already records `160 passed, 1 skipped, 498 subtests
passed`, task-scope Ruff, `py_compile`, and `git diff --check`.  As required
for this no-code-change attempt, those tests were not rerun.

## Collector preflight

All collector work was confined to the untracked directory
`tmp/web-runtime-smoke-20260821-attempt-002/`.  Before Uvicorn or any network
request, PowerShell's parser inspected four temporary scripts.  The first
offline pass rejected invalid line continuation in the snapshot and shutdown
helpers.  Those temporary scripts were corrected and parsed again; this was
before the one service attempt and consumed no smoke resource.

The final offline preflight established:

- all four scripts parsed without an error;
- no variable named `$home` was present;
- `$pingResponse`, `$statusResponse`, and `$rootResponse` were present;
- the three body paths and acceptance path were distinct and writable;
- no Uvicorn process had started and the request count was zero; and
- an in-memory/file fixture exercised JSON and HTML parsing successfully.

The fixture incorrectly modeled ping as `{"status":"ok"}`.  The real endpoint
returned `{"ok":true}`.  That collector defect is discussed below; it did not
lose or alter any real response body and does not permit a repeated request.

The temporary artifact manifest contains 30 files and has file SHA-256
`f2c74f86fd2d30a965a41c124bf82fb2c736c700d7fb090cdc8b34cd782efd94`.
No file under `tmp/` is committed.

## Pre-snapshot

A loopback socket selected unused port `60999` and immediately released it.
No listener existed on 8000, 8080, or 60999 before launch.  No Web product or
Sanmill process existed.  Pre-existing Ollama PID `20896` and unrelated
Python PID `91168` were recorded and remained identical after the smoke; this
task did not start either process.

Selected identities were:

| Resource | Size | SHA-256 |
| --- | ---: | --- |
| HumanDB main | 738,091,008 | `d8e22da38273f7c26eb76803ae91fc3fae711f508383ffbe3096c2946912b440` |
| HumanDB WAL | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| HumanDB SHM | 32,768 | `fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb` |
| Malom manifest | 69,298 | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| specialist open | 523,389 | `d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701` |
| specialist mid | 523,389 | `a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2` |
| specialist end | 523,389 | `5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8` |
| generalist | 523,389 | `494cec3f78d3b8f8f05d61d30a7c620d796cc386d5e355ca3fdaa5e3d16a792f` |

HumanDB main, WAL, and SHM pre-run UTC mtime ticks were respectively
`639199818960000000`, `639222004749625022`, and
`639228980568546120`.  `data/ngram_model.json` was absent.  Settings, the
machine-local path registry, Web startup sources, Malom resolver, and product
safety source were also hashed before launch.

## Exact runtime commands

The relevant top-level commands were:

```powershell
git fetch origin dev

& 'tmp/web-runtime-smoke-20260821-attempt-002/capture_snapshot.ps1' `
  -OutputPath 'tmp/web-runtime-smoke-20260821-attempt-002/pre-snapshot.json' `
  -Port 60999

.\.venv\Scripts\python.exe -m uvicorn web.app:app `
  --host 127.0.0.1 --port 60999 --log-level info

& 'tmp/web-runtime-smoke-20260821-attempt-002/collect_smoke_responses.ps1' `
  -BaseUrl 'http://127.0.0.1:60999' `
  -OutputDirectory `
    'tmp/web-runtime-smoke-20260821-attempt-002/formal' `
  -ValidatorPath (Resolve-Path `
    'tmp/web-runtime-smoke-20260821-attempt-002/validate_collected_responses.ps1').Path

& 'tmp/web-runtime-smoke-20260821-attempt-002/monitor_shutdown.ps1' `
  -ProcessId 76456 -Port 60999 `
  -StopSignalRequestedUtc '2026-08-21T09:33:52.0983287+00:00' `
  -OutputPath 'tmp/web-runtime-smoke-20260821-attempt-002/shutdown.json'

& 'tmp/web-runtime-smoke-20260821-attempt-002/capture_snapshot.ps1' `
  -OutputPath 'tmp/web-runtime-smoke-20260821-attempt-002/post-snapshot.json' `
  -Port 60999
```

The service was launched directly from the current `.venv`.  `run_nmm.bat`,
a browser, Ollama launch, and Sanmill were not used.

## Response capture and runtime acceptance

Uvicorn reported server PID `76456`.  The process start was
`2026-08-21T09:32:34.8337048Z`; the listener was observed ready at
`2026-08-21T09:33:09.6946271Z`.  Only after listener readiness, the collector
made these requests in this order, exactly once each:

| Request | Start UTC | End UTC | ms | HTTP | Bytes | Body SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `/api/ping` | `09:33:22.8283409` | `09:33:22.9004602` | 72.1193 | 200 | 11 | `4062edaf750fb8074e7e83e0c9028c94e32468a8b6f1614774328ef045150f93` |
| `/api/overseer_status` | `09:33:22.9271561` | `09:33:22.9316439` | 4.4878 | 200 | 4,210 | `99e2a0c3f3ff5da100cf06f5c55f5421f9d4d9bf67abba3d5dcb4df6de72039b` |
| `/` | `09:33:22.9404727` | `09:33:22.9501116` | 9.6389 | 200 | 39,434 | `124d41d476ce32c9dda85a387a7c16c05579428c8b078bc455ac480e9e5b56ed` |

Each body and its metadata were written before the next request was made.
The real ping body is `{"ok":true}`.  The original collector acceptance file
remains failed because its fixture expected a nonexistent `status` property;
it was not overwritten or presented as passed.  A separate read-only audit of
the three already-persisted bodies treats HTTP 200 plus `ok=true` as the
endpoint's successful response.  It made no HTTP request.

The persisted status body reports:

- `ngram_opponent_model.mode=cache-only`, `available=false`,
  `enabled=false`, reason `cache file is missing`, and
  `raw_corpus_scan_attempted=false`;
- `human_db_runtime.available=true`, `read_only=true`, `immutable=true`,
  2,152,889 positions, and 94,429 games;
- `product_route.default_difficulty_9_10=classical-first`;
- `specialist_override=explicit-use_overseer_player-only`;
- `generalist_override=explicit-use_generalist_player-only`;
- `final_choke=ProductPositionalSafetyGate`;
- `safe_set=A_pos`, `positional_only=true`, and `history_aware=false`;
- `specialist_available=true` and `last_decision_source=null`;
- Malom validation `passed`, selected source
  `local-registry:malom_db_path`, label `sector-corrected-v1`, manifest file
  SHA-256 `f4c52b00...`, 512 components, and 83,582,223,577 bytes; and
- product positional safety `configured=true`, `enabled=true`, mode `A_pos`,
  label `sector-corrected-v1`, and all decision/request counters zero.

The status also exposed the stale shared-settings Malom candidate as rejected
because its path did not exist.  The selected local-registry candidate passed
manifest, inventory, adapter, and oracle validation.

The persisted home page contains both:

- `9 — Classical AI + positional safety (30 s)`; and
- `10 — Classical AI + positional safety (60 s)`.

It also contains `id="chk-overseer-player"`; the actual input tag has no
`checked` attribute.  Thus the explicit Specialist control exists and is not
selected by default.

These values establish the runtime HTTP contract for this exact process only.
They do not turn the overall shutdown-failed smoke into a pass.

## Shutdown timing

The stop signal was requested at `2026-08-21T09:33:52.0983287Z`, after
77.264624 seconds of service-process life and well below the 570-second
active limit.  Ctrl+C did not produce a natural exit.  The bounded monitor
requested exact-PID force termination at
`2026-08-21T09:34:21.4935825Z`, 29.395254 seconds after the stop request.

PID and port disappearance were both observed at
`2026-08-21T09:34:22.4823674Z`, 30.384039 seconds after the stop request and
0.384039 seconds beyond the confirmation limit.  The process lifetime through
confirmed exit was 107.648663 seconds.  The unified process session ended with
exit code 1 after forced termination.  It is not described as a normal
shutdown.

Post-audit found no PID `76456`, no listener on 60999, and no listener on 8000
or 8080.  No smoke-owned child process remained.  The set of unrelated
recorded processes was identical before and after.

## Read-only and side-effect audit

HumanDB main, WAL, and SHM matched pre-run existence, size, content SHA-256,
and high-resolution UTC mtime ticks exactly.  The Malom manifest, local path
registry, settings, all four checkpoints, and all four implementation/source
files also matched in existence, size, hash, and mtime.  The N-gram cache was
absent both before and after.

The allowed `data/logs/server.log` append was:

- lines: 27 -> 48, 21 appended;
- bytes: 3,070 -> 5,660, 2,590 appended;
- SHA-256: `7fc8664d08a2e7dacf754ff8a8cc773aa3ba6ccbff58fd6aeef2750454a380fb`
  -> `07f2f315222c00f9e1f0d9b1cdb0458b22dc71608019cb933c4461e0fb1ee1d3`;
  and
- appended-range SHA-256:
  `2743bbf7d79045cf25c296491ea00847bdb4498aba5f8bf9a89e00962d2854d9`.

The 21 lines record immutable HumanDB, skipped historical TrajectoryDB scan,
cache-missing N-gram disablement, read-only model loads, the selected Malom
runtime, and enabled `A_pos`.  The existing non-fatal missing
`ai.malom_puzzle_search` prewarm remains visible and did not disable product
Malom.

There was no game, WebSocket, session, `new_game`, move inference, Sanmill,
training, fitting, model/checkpoint/alias change, database write, N-gram build,
or raw human-corpus read.  In addition to the request contract, the persisted
product-safety counters remained zero and `last_decision` remained null.

## Boundary

The runtime response portion is accepted only for exact published source
`f4d08283`, implementation ancestor `718c7ef8`, this host's validated Malom
runtime, and this zero-move process.  The smoke as a whole failed the mandatory
shutdown gate.  No additional service attempt, product change, deployment,
release, game, training, or evaluation is authorized by this result.
