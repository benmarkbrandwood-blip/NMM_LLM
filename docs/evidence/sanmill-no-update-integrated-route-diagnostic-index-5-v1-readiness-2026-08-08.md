# Sanmill Route Diagnostic Index 5 v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

The final 1,000-node deep-route entry is ready for one no-update diagnostic
game. This does not authorize a retry, higher-node probe, or training.

| Field | Value |
| --- | --- |
| Published source | `5482369b6d71708e477842c296d65f4320fe1374`; clean `dev == origin/dev` |
| Plan identity / raw SHA-256 | `6c285de356556e2577062f8bc2d64c904b81d34d4e6fe71b589f74dda7d3cce4` / `5d8e189227b0b1cd8b3027e37d4be7c75eb53d68ad617d66aae62a591f5c0ac9` |
| Selected role | `sanmill-1000-deep-0-B`; parent index 5 |
| Game / seed | `game:1a3984c50f3aded3fa23fa54429fd0c13ae20a13c1c3c7019edaeb6be9ab8086` / `1889687500557573843` |
| Route / ceiling | simulation depth 12; one game; 120 plies; 60 searches; 60,000 requested node ceilings |
| Model | fresh learner and target `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer or gradients |
| Immutable inputs | same verified HumanDB, empty corrected SpecialistDB, 512-component Malom, rules, and strict Sanmill runtime used for indices 0–4 |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The published preflight returned `ready_for_authorized_minimal_diagnostic`,
passed the two-ply no-search route check, and found both proposed output paths
absent. The focused 30-test Sanmill baseline, mandatory 103 tests plus 498
subtests, Ruff, and `git diff --check` remain passing; this is not a full-suite
claim.

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-5-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-index-5-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-index-5-v1-20260808-001.json
```

Delegated technical authority applies only after this evidence is published
and the final preflight passes unchanged. Any termination consumes the run.
