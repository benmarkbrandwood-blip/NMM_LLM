# Sanmill Route Diagnostic Index 6 v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

The first 5,000-node parent entry is ready for exactly one no-update game.

| Field | Value |
| --- | --- |
| Published source | `6fae3879942695400a74e9579318f451a762d286`; clean `dev == origin/dev` |
| Plan identity / raw SHA-256 | `5af3a6fcc24bfbba9b4d424a181f5d9db6835c27e4e3c95c7533295a2ee3ff18` / `c474d0e8345c1f17bf3a29360cdd1a87e67050ce56772c1deb9443360fa31400` |
| Selected role | `sanmill-5000-normal-0-W`; parent index 6; seed `701517017423886323` |
| Work ceiling | one game; 120 plies; 60 searches; 300,000 requested node ceilings |
| Model and inputs | same fresh no-optimizer model and immutable verified data/rules/runtime identities used by the passing index 0–5 diagnostics |
| Referee | `mif-stable-moving-v1`; `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The published preflight returned `ready_for_authorized_minimal_diagnostic`,
passed the no-search route check, and found fresh completed and failure paths.
The focused 30-test baseline, mandatory 103 tests plus 498 subtests, Ruff, and
`git diff --check` remain passing. This is not a full-suite or training claim.

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-6-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-index-6-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-index-6-v1-20260808-001.json
```

Delegated technical authority applies after publication and a final matching
preflight. Any termination consumes the one run; no retry or training follows
automatically.
