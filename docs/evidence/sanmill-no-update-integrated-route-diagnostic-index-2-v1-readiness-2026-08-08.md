# Sanmill Route Diagnostic Index 2 v1 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

This verdict covers exactly one no-update game for parent schedule index 2.
It does not authorize a retry, another index, a training smoke, or long
training.

## Frozen gate

| Field | Value |
| --- | --- |
| Published source | `5af7347886cd023285c3dd5e13fc8c558bddc8d3`; clean `dev == origin/dev` |
| Parent plan identity | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` |
| Diagnostic plan identity | `cad483e133887c4b710f9fe8ccfdd44876987be650f7306d1d0b18a11b184b48` |
| Diagnostic plan raw SHA-256 | `bbd407496204ef9af2f51367351ce4cdf4896ed05eedd9419f155b89be25ea09` |
| Selected role | `sanmill-1000-normal-1-W`; parent index 2 |
| Game / seed | `game:7bfae064c9b1a2c787410bb94ebd25a95977a2c79cb5d2d4a446cb06fdf06481` / `8933699534328013511` |
| Work ceiling | one game; 120 logical plies; 60 searches; 60,000 requested node ceilings |
| Model | fresh learner and target digest `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer; gradients disabled |
| HumanDB | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`; `quick_check=ok`; no sidecars |
| SpecialistDB | `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`; empty corrected source; no sidecars |
| Malom | 512 components; manifest `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee | `mif-stable-moving-v1`; semantic digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |

The published-source preflight returned
`ready_for_authorized_minimal_diagnostic`, kept `launch_authorized=false`, and
passed the two-ply no-search route check. The proposed result and failure paths
were absent. The HumanDB warning is expected because only its unversioned
historical Malom columns are disabled.

Verification remains the immediately preceding diagnostic-infrastructure
baseline: 30 focused Sanmill tests passed; the mandatory group reported 103
passed and 498 parameterized subtests; Ruff and `git diff --check` passed.
This is not a full-suite claim.

## Exact command

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-index-2-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-index-2-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-index-2-v1-20260808-001.json
```

The repository owner's delegated technical authority applies only after this
record is published and a final identical preflight passes. Completion,
fail-closed output, or operator termination consumes the one-run authority.
