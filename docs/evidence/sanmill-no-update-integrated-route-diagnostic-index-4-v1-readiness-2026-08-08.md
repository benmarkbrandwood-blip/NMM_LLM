# Sanmill Route Diagnostic Index 4 v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

The published-source preflight passed for the first deep-route schedule entry.
This verdict covers one no-update game only and is not training authority.

| Field | Value |
| --- | --- |
| Published source | `158fe1bed99004a6ed4d07344bb8961e3c3840e3`; clean `dev == origin/dev` |
| Plan identity / raw SHA-256 | `6e3ee428bd6e8cead86b736d957912b35c12529c928aad72ade795e0db19f2ed` / `9d131a72c1962181f3b8e9ddd4783693f944069aef3b1b6044457a97e63027de` |
| Selected role | `sanmill-1000-deep-0-W`; parent index 4 |
| Game / seed | `game:23617ce9e274d505ef0577f8b9447638d4244556a4f409a67ae20dcfcb66eafc` / `2549456208036943109` |
| Route | `sim_ply_depth=12`; 1,000-node Sanmill |
| Work ceiling | one game; 120 logical plies; 60 searches; 60,000 requested node ceilings |
| Model | fresh learner and target `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer or gradients |
| Data and Malom | same immutable HumanDB, empty corrected SpecialistDB, and 512-component corrected Malom identities used by indices 0–3 |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

Preflight returned `ready_for_authorized_minimal_diagnostic`, retained
`launch_authorized=false`, and passed the no-search route check. Both proposed
output paths were absent. Verification remains the passing 30-test focused
Sanmill baseline, mandatory 103 tests plus 498 subtests, Ruff, and
`git diff --check`; this is not a full-suite claim.

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-4-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-index-4-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-index-4-v1-20260808-001.json
```

Delegated technical authority applies after publication and a final identical
preflight. Any completion, fail-closed result, or termination consumes it;
there is no retry or automatic move to index 5.
