# Sanmill Frozen-Target Index 31 Reproduction v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

The first failing black-side frozen-target control is technically ready for
exactly one no-update reproduction after the terminal-turn mirror fix. This
is a diagnostic replay, not training, strength evidence, or authority for a
second run.

## Failure and fix boundary

The consumed continuation run completed immutable parent indices 7 through
30 and failed closed at index 31. Its failure ledger has SHA-256
`52ec411dc9f6b83b4b3d0a5bd4ef6a256db2e2feedbcbe16b49d1a84fbd5fbda`
and report identity
`514b9c094d44154d42d169a0978e48310f0e28ab5f87ed431f4bfb140eb3256d`.
The local and projected boards had identical pieces and placement counts;
their structured terminal results both named White as winner because Black
had fewer than three pieces. The only difference was the compact FEN turn:
the local atomic board retained `B`, while Sanmill declared
`side_to_move=null` and its game-over FEN retained `W`.

Commit `bb4fe56e1af4488df0d6a8338e0ff1114f5f9e6c` therefore treats only the
turn field as undefined after a structured Sanmill terminal. It continues to
compare every board point and both placement counts, checks the structured
winner against the local rules result, and retains exact turn comparison for
every non-terminal state. A terminal board-position mismatch still fails
closed in focused regression coverage.

## Frozen identity and bounds

| Field | Value |
| --- | --- |
| Published plan source | `4dd834ecb245f0c3aa1f931ef48b79b360b8db49`; clean `dev == origin/dev` |
| Parent identity / raw SHA-256 | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` / `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` |
| Reproduction identity / raw SHA-256 | `8aa3d733d20fa58fa105eee992cab6f000507e22ffe59a97767681425d7259ec` / `65fb386c42548fbde49ef066a5de70c05f7a3dfc38371dafdeb7a49419818e71` |
| Selected entry | parent index 31; `frozen-target-normal-0-B`; seed `5768742839362539388` |
| Work ceiling | one complete game; 120 logical plies; zero Sanmill search calls and node requests |
| Model | fresh learner and frozen target digest `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer or gradients |
| HumanDB | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`; `quick_check=ok`; no sidecars; historical Malom labels disabled |
| SpecialistDB | `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`; empty `sector-corrected-v1`; no sidecars |
| Malom | 512 components; manifest `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The published preflight returned
`ready_for_authorized_continuation_probe`, kept
`launch_authorized=false`, passed the unscheduled two-ply no-search route
check, and found no source, model, runtime, rules, or database drift.

## Verification

```text
38 focused referee/probe/continuation/diagnostic tests passed
103 mandatory Malom/DB-teacher/provenance tests passed
498 parameterized subtests passed
Ruff passed
git diff --check passed
```

The first run of the new acceptance regression failed for the captured reason
before the adapter fix. This is not a full-repository test-suite claim.

## Exact command

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route_continuation.py `
  --launch continuation `
  --plan docs\experiments\sanmill-no-update-frozen-target-index31-reproduction-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-frozen-target-index31-reproduction-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-frozen-target-index31-reproduction-v1-20260808-001.json
```

Delegated technical authority applies only after this readiness record is
published and a final matching preflight passes. Completion, fail-closed
output, or operator termination consumes the one run. There is no automatic
retry and no training launch follows from either outcome.
