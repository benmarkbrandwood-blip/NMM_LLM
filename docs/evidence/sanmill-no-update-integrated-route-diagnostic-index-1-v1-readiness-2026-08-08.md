# Sanmill Route Diagnostic Index 1 v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

This verdict is limited to one no-update diagnostic game selecting parent
schedule index 1. It is not authority for a retry, another schedule entry, a
multi-game probe, a training smoke, or long training.

## Readiness gates

| Gate | Observed | Result |
| --- | --- | --- |
| Repository | root `.`; branch `dev`; source `25e46bfb9c4b50ca148125543bc5d4afc78a5b39`; clean and published | pass |
| Parent plan | identity `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb`; raw SHA-256 `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` | pass |
| Diagnostic plan | identity `298700a21e11fcff1f8789c2d6fb166af03c461f31cdc94e3adb2ed1675ffc2f`; raw SHA-256 `68e22209e4ec903af2532339943d3575f7dd930ee021e8abafece883dd26e75c` | pass |
| Selection | exact parent index 1; game `game:a4679c60cd80c02ea260887a978df8534fa4eba9a60a0dc5ade3072228d77430`; Black learner; seed `2623237247545163822`; normal depth-5 route; 1,000-node Sanmill | pass |
| Bound | one game; 120 logical plies; 60 searches; 60,000 requested node ceilings; no retry | pass |
| Model | fresh learner and target digest `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer; gradients disabled | pass |
| HumanDB | SHA-256 `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`; `quick_check=ok`; no sidecars | pass |
| SpecialistDB | SHA-256 `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`; empty; `sector-corrected-v1`; `quick_check=ok`; no sidecars | pass |
| Malom | 512 components; manifest SHA-256 `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` | pass |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary SHA-256 `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`; strict identity `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` | pass |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` | pass |
| Route check | two logical plies; zero opponent searches; no mismatch | pass |
| Output isolation | proposed completed and failure destinations are absent | pass |

The published-source preflight returned
`ready_for_authorized_minimal_diagnostic` and `launch_authorized=false`, as
designed. The repository owner delegated technical decisions and bounded
diagnostic launches while retaining the product/resource gate for long
training. This record applies that delegation only after it is itself
published and the same final preflight passes again.

The HumanDB warning is expected. Its unversioned historical Malom columns are
disabled, while its human frequency data remains available; corrected oracle
information comes from the separately verified Malom input.

## Verification

```text
30 focused Sanmill diagnostic/probe/referee tests passed
103 mandatory Malom/DB-teacher/provenance tests passed
498 parameterized subtests passed
Ruff passed
git diff --check passed
```

This is not a full-repository test-suite claim.

## Exact commands

Final read-only preflight:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --preflight `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-1-v1.json `
  --paths-config data\training_paths.local.json
```

Single permitted diagnostic invocation:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-1-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-index-1-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-index-1-v1-20260808-001.json
```

Any completed result, fail-closed result, or operator termination consumes
this plan's one-run authority. There is no automatic retry or next-index
escalation.
