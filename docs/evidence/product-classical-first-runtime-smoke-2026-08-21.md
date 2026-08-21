# D9/D10 Classical-First Local Runtime Smoke -- 21 August 2026

## Outcome

The bounded local runtime smoke did **not** pass.  It failed closed before
any HTTP request or product-route observation because `web.app:app` did not
finish import/startup and did not bind its loopback port within the ten-minute
window.

The process was started at `2026-08-21T08:34:14.3698916Z`.  A stop was
requested at `2026-08-21T08:44:16.8638302Z`, and both smoke-owned Python
processes were absent by `2026-08-21T08:44:18.0434655Z`.  Thus the process was
active for about `603.67` seconds through confirmed exit.  This exceeded the
600-second bound by about `3.67` seconds, including shutdown confirmation.  It
must not be described as a successful or in-budget smoke.

No implementation fix was made.  The observed startup bottleneck is outside
the delegated permission to repair a defect specifically in the new default
route, status, or UI.  There was no second attempt.

## Frozen repository state

Before startup:

- repository root: `I:/Mill_Training/NMM_LLM`;
- branch: `dev`;
- local HEAD: `6b48b73e19bc23d4369c683db81a86700cd47ce9`;
- `origin/dev`: the same commit;
- tracked worktree: clean;
- only the pre-existing ignored/untracked `tmp/` namespace was visible.

The product source inspected for this smoke was:

- `web/app.py` SHA-256
  `8f8bf57b9b78303a052acf1e882f3cd15537a95a3b8e78102ee7b8e94000a0d5`;
- `ai/ngram_opponent_model.py` SHA-256
  `91ddcd8a814b670f98b7e1e6eefde581ec0b6f65e67eef5acee341968b7f41b8`.

## Startup command and process boundary

`run_nmm.bat` was not used.  The smoke selected unused loopback port `50256`
and launched the current repository environment with the equivalent of:

```powershell
Start-Process `
  -FilePath .\.venv\Scripts\python.exe `
  -ArgumentList @(
    '-m', 'uvicorn', 'web.app:app',
    '--host', '127.0.0.1', '--port', '50256',
    '--log-level', 'info'
  ) `
  -WorkingDirectory I:\Mill_Training\NMM_LLM `
  -RedirectStandardOutput `
    tmp\product-route-runtime-smoke-20260821-001\uvicorn.stdout.log `
  -RedirectStandardError `
    tmp\product-route-runtime-smoke-20260821-001\uvicorn.stderr.log `
  -WindowStyle Hidden -PassThru
```

Windows created a virtual-environment launcher process, PID `70312`, and its
base-interpreter child, PID `47440`.  No process used port 8000 or 8080 before
startup.  A pre-existing Ollama process, PID `20896`, and unrelated Python
process, PID `91168`, were observed but were not started, contacted, or
stopped by this smoke.

At the final pre-stop observation the base interpreter had accumulated
`53.140625` CPU seconds, `714,289,152` bytes of working set, and 230 handles.
It still had no listener.  Only PIDs `47440` and `70312` were terminated.
After shutdown, neither PID existed and port `50256` had no listener.  The
pre-existing Python process remained.

The other material shell commands were:

```powershell
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD
git rev-parse origin/dev

Get-NetTCPConnection -State Listen |
  Where-Object { $_.LocalPort -in 8000, 8080 }
Get-Process -Name python,pythonw,uvicorn,ollama

Get-FileHash -Algorithm SHA256 -LiteralPath <frozen-resource-path>
Get-ChildItem data\logs -Filter 'server.log*' -File

Get-NetTCPConnection -State Listen -LocalPort 50256
Get-Process -Id 70312,47440

Stop-Process -Id 47440
Stop-Process -Id 70312
Get-NetTCPConnection -State Listen -LocalPort 50256
Get-Process -Id 70312,47440
```

An initial attempt to obtain process command lines with
`Get-CimInstance Win32_Process` returned `Access denied`.  It occurred before
startup and was replaced by `Get-Process` plus listener ownership checks; it
did not alter the service configuration or the product process.

## HTTP and product behavior audit

The polling command checked for a listening socket before issuing requests.
Because the socket never appeared, none of the permitted requests was sent:

- `GET /api/ping`: not sent;
- `GET /api/overseer_status`: not sent;
- `GET /`: not sent.

There was no WebSocket connection, session creation, `new_game`, move request,
move inference, Sanmill process, game, or browser launch.  The empty Uvicorn
stdout/access log independently supports the absence of HTTP traffic.

Consequently, this run did **not** establish any of the requested runtime
facts.  In particular, it did not observe an enabled product positional safety
gate, a selected Malom source, a null pre-move `last_decision_source`, or the
rendered homepage controls.  Those facts remain statically represented in
the frozen source, but static source is not substituted for the failed
runtime check:

- `web/app.py:383-389` declares D9/D10 `classical-first`, explicit-only
  specialist override, `ProductPositionalSafetyGate`, `A_pos`,
  `positional_only=true`, and `history_aware=false`;
- `web/app.py:1101` exposes `last_decision_source`;
- `web/templates/index.html:332-333` labels D9/D10 as
  `Classical AI + positional safety`;
- `web/templates/index.html:438-440` retains an unchecked Specialist AI
  checkbox.

## Startup evidence and diagnosis

The application log appended only these three lines for this attempt:

```text
2026-08-21 16:34:14,964 INFO === Server started ===
2026-08-21 16:34:17,056 INFO HumanDB: 2152889 positions, 94429 games — skipping TrajectoryDB file scan.
2026-08-21 16:34:17,058 INFO TrajectoryDB: file scan skipped (HumanDB active).
```

The captured stderr also reported that unversioned historical Malom columns
in HumanDB were disabled, as required, and repeated the HumanDB and
TrajectoryDB messages.  It contained no exception.  Uvicorn never emitted
its normal server-start or access-log records.

The highest-confidence diagnosis is the synchronous N-gram corpus build that
immediately follows those log records:

- `data/ngram_model.json` is absent;
- `web/app.py:157-165` therefore calls `load_from_games` for `data/games` and
  then `data/human_games` before later model, Malom, route, or Uvicorn startup;
- `ai/ngram_opponent_model.py:68` recursively opens and parses every JSONL
  file;
- the local `data/human_games` directory contains 94,529 files, including
  94,527 JSONL files, totalling 751,334,698 bytes;
- no `NGramOpponentModel: built` completion record appeared;
- CPU time and working set continued to rise until termination.

This is a code-and-log-supported diagnosis, not a captured Python stack trace,
so it is recorded as a high-confidence inference rather than an independently
proven exact stack location.  It also explains why specialist checkpoint and
Malom runtime initialization were never reached.

## Read-only identity and write audit

The following primary resources had identical SHA-256, byte size, and mtime
before and after the attempt:

| Resource | SHA-256 |
| --- | --- |
| opening specialist | `d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701` |
| movement specialist | `a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2` |
| endgame specialist | `5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8` |
| generalist checkpoint | `494cec3f78d3b8f8f05d61d30a7c620d796cc386d5e355ca3fdaa5e3d16a792f` |
| Sentinel checkpoint | `b5433076765500d6520640335c7bdd5c9a67738e6e3b3bb1647e28087ae0bcfc` |
| HumanDB main file | `d8e22da38273f7c26eb76803ae91fc3fae711f508383ffbe3096c2946912b440` |
| solved endgame file | `573b65250c5f660aa40038433c1b8577b4c13e9f9af4a422a6fd732f01279c77` |
| Malom manifest file | `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| local path registry | `89cb9a14ab8706609f2b53bb2458d7f41ed182b5f5d0a27b3776bb88c6085798` |

The three phase ValueNet files and GapNet also retained their previously
recorded hashes and mtimes.  `data/specialist_db.sqlite` remained absent.

One filesystem side effect must not be hidden: opening HumanDB changed the
mtime of `data/human_db.sqlite-shm` from
`2026-08-18T07:35:46.0631501Z` to
`2026-08-21T08:34:16.8546120Z`.  Its size after shutdown was 32,768 bytes and
its SHA-256 was
`fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb`.
The main SQLite file retained its exact identity, and the zero-byte WAL kept
its original mtime and empty-file hash.  Because the pre-start SHM hash was
not captured, this record does not claim that the SHM content was unchanged.

No autosave, game-count file, runtime quarantine ledger, product game JSONL,
checkpoint, alias, training datum, or database-main-file change was created.
The only intended log change was `data/logs/server.log`, which grew from 240
to 480 bytes.  Its SHA-256 changed from
`68e076a9ee3eb97d0129113a3480b0bc3efbe28c96e3873e986000b7848aedf7`
to
`0e5e597d626717c5fe0d3e6d1158a8539dc64fe99f7424f4e0d74b2c99598b3a`.
The ignored smoke stderr is 289 bytes with SHA-256
`8815c17206f12f9aa2d42bfeebea6a30c087889781428f489f875d65bedf788d`;
stdout is empty.

## Decision boundary

The static D9/D10 classical-first implementation remains independently tested
and published at `6b48b73`, but this local runtime smoke is
`failed_closed_startup_timeout`.  It supplies no runtime acceptance evidence
for the route, Malom activation, status endpoint, or homepage.  It also
records an unrelated startup-cost issue and a HumanDB SHM side effect for a
future owner decision.  No game, inference, evaluation, training, deployment,
release, or product-route modification was authorized or performed.
