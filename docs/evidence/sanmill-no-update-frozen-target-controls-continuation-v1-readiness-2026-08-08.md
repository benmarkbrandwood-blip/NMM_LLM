# Sanmill Frozen-Target Controls Continuation v1 Readiness — 8 August 2026

## Verdict

`ready_for_probe`

The four remaining immutable frozen-target controls are technically ready for
one no-update, fail-closed continuation. They close parent indices 32 through
35 after the exact index-31 reproduction passed. This is not training,
strength evidence, or authority for a retry.

## Frozen identity and bounds

| Field | Value |
| --- | --- |
| Published plan source | `e65e7e3be191d6578018876c2104106259e43f2d`; clean `dev == origin/dev` |
| Parent identity / raw SHA-256 | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` / `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` |
| Continuation identity / raw SHA-256 | `1a554be7d9c23c9b4f4ed5f932fc6318185c7b9cd274b655196449cc27598de6` / `d953c0098355b028074d3770c223e4c4c04d2e463d540aa8658fef6a7ab83796` |
| Parent range | indices 32 through 35, end exclusive 36 |
| Roles | normal-1 White/Black and deep-0 White/Black frozen-target controls |
| Work ceiling | four complete games; 480 logical plies; zero Sanmill search calls and node requests |
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
5 focused continuation-contract tests passed for this plan
103 mandatory Malom/DB-teacher/provenance tests passed
498 parameterized subtests passed
Ruff passed
git diff --check passed
```

This is not a full-repository test-suite claim.

## Exact command

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route_continuation.py `
  --launch continuation `
  --plan docs\experiments\sanmill-no-update-frozen-target-controls-continuation-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-frozen-target-controls-continuation-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-frozen-target-controls-continuation-v1-20260808-001.json
```

Delegated technical authority applies only after this readiness record is
published and a final matching preflight passes. Completion, fail-closed
output, or operator termination consumes the one run. There is no automatic
retry and no training launch follows from either outcome.
