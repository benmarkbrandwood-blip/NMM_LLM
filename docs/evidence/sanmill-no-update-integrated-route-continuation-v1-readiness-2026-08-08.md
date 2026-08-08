# Sanmill No-Update Route Continuation v1 Readiness — 8 August 2026

## Verdict

`ready_for_probe`

The remaining immutable parent range is technically ready for one no-update,
fail-closed continuation. It starts after seven individually completed games,
stops at the first error, and publishes either all 29 samples or an atomic
failure ledger with the exact completed prefix. It is not training, strength,
or node-ladder evidence.

## Frozen identity and bounds

| Field | Value |
| --- | --- |
| Published source | `e55671592aa70679590dce241dd8f84b960ee1d2`; clean `dev == origin/dev` |
| Parent identity / raw SHA-256 | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` / `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` |
| Continuation identity / raw SHA-256 | `807fcae96ee03634d5abb61b9982fcfaf364b07ab1c139142ad8cc1cffdadb08` / `b10ba116f43468567f053ef965d72371cc570ea999bbd00234c5aa284ad6ad75` |
| Parent range | indices 7 through 35, end exclusive 36 |
| Games | 29 total: 23 Sanmill search opponents and 6 frozen-target controls |
| Work ceiling | 3,480 logical plies; 1,380 search calls; 226,500,000 requested node ceilings |
| Model | fresh learner and target digest `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer, gradients, checkpoint, or rollout persistence |
| HumanDB | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`; `quick_check=ok`; no sidecars |
| SpecialistDB | `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`; empty corrected source; no sidecars |
| Malom | 512 components; manifest `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The published preflight returned `ready_for_authorized_continuation_probe`,
kept `launch_authorized=false`, passed the unscheduled two-ply no-search route
check, and found no input identity drift. The completed and `.failure.json`
destinations below were absent.

## Verification

```text
35 focused continuation/diagnostic/probe/referee tests passed
103 mandatory Malom/DB-teacher/provenance tests passed
498 parameterized subtests passed
Ruff passed
git diff --check passed
```

The new code contains no gameplay implementation. It derives a contiguous
slice from the content-addressed parent and invokes the existing `run_probe`.
This is not a full-repository test-suite claim.

## Exact command

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route_continuation.py `
  --launch continuation `
  --plan docs\experiments\sanmill-no-update-integrated-route-continuation-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-continuation-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-continuation-v1-20260808-001.json
```

The owner's delegated technical authority applies only after this readiness
record is published and a final matching preflight passes. Completion,
fail-closed output, or operator termination consumes the one run. There is no
automatic retry and no training launch follows from either outcome.
