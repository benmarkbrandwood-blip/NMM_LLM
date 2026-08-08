# Sanmill Route Diagnostic Index 3 v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

The published-source preflight passed for exactly one no-update game selecting
parent schedule index 3. This is not retry, multi-game probe, training-smoke,
or long-training authority.

| Field | Value |
| --- | --- |
| Published source | `928fe92d6b6b41ab677747aba84941aac04389a0`; clean `dev == origin/dev` |
| Plan identity / raw SHA-256 | `6f9b92a64e60aa3e5fa140dd82161c52ef05b074c6e2797449d9b885f458238a` / `4ddea695d0396c4ea4d47b58843d90029d11268463a9acf07b579794b5d1a2c9` |
| Selected role | `sanmill-1000-normal-1-B`; parent index 3 |
| Game / seed | `game:6fb7d3b49480668e2dcc47798a69af6289a61222c5edb3cfcf1e8da91190f0fa` / `8050135631487067790` |
| Work ceiling | one game; 120 logical plies; 60 searches; 60,000 requested node ceilings |
| Model | fresh learner and target `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer or gradients |
| Data | HumanDB `97be7152...b6f93c`; empty corrected SpecialistDB `b4d522d2...86b7f`; both `quick_check=ok`, no sidecars |
| Malom | 512 components; manifest `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The preflight returned `ready_for_authorized_minimal_diagnostic`, preserved
`launch_authorized=false`, passed the two-ply no-search route check, and found
fresh result and failure paths. The HumanDB Malom-label warning is expected.

The relevant verification baseline remains 30 focused Sanmill tests, 103
mandatory tests and 498 parameterized subtests, Ruff, and `git diff --check`,
all passing. This is not a full-suite claim.

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-3-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-index-3-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-index-3-v1-20260808-001.json
```

Delegated technical authority applies only after this record is published and
the final preflight remains identical. Any termination consumes the one run.
