# Sanmill No-Update Integrated-Route Minimal Diagnostic v1 Readiness — 8 August 2026

## Verdict

`needs_decision`

The one-entry diagnostic implementation is technically ready for publication,
but it is not ready to execute from the current repository state. The tested
implementation commit `bdd6ed1adf67962e2604861f650f525fcf4b3d6d` is clean
and one commit ahead of `origin/dev`. The production preflight correctly
rejects that unpublished state. No diagnostic game, probe game, training
smoke, or long run is authorized by this record.

## Readiness gates

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Repository | root `.`; branch `dev`; tracked tree clean | sole repository root and clean `dev` | pass |
| Source | `bdd6ed1adf67962e2604861f650f525fcf4b3d6d`; tree `322df17a322638aa36a38e537a3abb051dc7aebb`; one commit ahead of `origin/dev` | exact clean source and `dev == origin/dev` | pending publication |
| Parent plan | identity `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb`; raw SHA-256 `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` | immutable failed-probe plan | pass |
| Diagnostic plan | identity `5554489e3278dca88cc4f816e97ced1bdf17e7a89b0e4c02991c808d7087e4b0`; raw SHA-256 `d445f95249146a54fb5cebc3e09e6dee542fbb9a7c6ea3bc313b723ff7bce04e` | tracked, content-addressed plan | pass |
| Selection | exact parent `scheduled_index=0`, original game identity, White learner, normal depth-5 route, 1,000-node Sanmill | one unchanged parent entry | pass |
| Bound | one game; 120 logical plies; 60 searches; 60,000 requested node ceilings | no automatic expansion or retry | pass |
| Model | fresh learner and frozen target both `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`; no optimizer; gradients disabled | original no-update model contract | pass |
| HumanDB | SHA-256 `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c`; `quick_check=ok`; no sidecars | immutable frequency source | pass |
| SpecialistDB | SHA-256 `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f`; empty; `sector-corrected-v1`; `quick_check=ok`; no sidecars | immutable empty corrected source | pass |
| Malom | 512 components; manifest SHA-256 `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747` | corrected pinned tablebase | pass |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; tree `17b9b0fd51ee8dac54c0454a6935978a47d19e0c`; binary SHA-256 `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`; strict identity `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` | clean pinned runtime with strict no-fallback referee | pass |
| Route check | two logical plies, zero opponent searches, Sanmill count two, `max-ply-truncation` | unscheduled read-only route only | pass |
| Output isolation | completed and `.failure.json` paths both absent; no `tgf` process remains | fresh no-overwrite destinations and no process leak | pass |
| Launch | preflight returned `launch_authorized=false` | separate explicit one-run authority | not authorized |

The HumanDB warning about a missing historical Malom-label version is expected
for this source: its human frequencies remain enabled, while its unversioned
historical Malom columns remain disabled. The diagnostic obtains corrected
oracle information from the separately verified Malom input.

## Verification

The implementation and its adjacent contracts report:

```text
74 passed
103 passed, 498 subtests passed
Ruff: passed
git diff --check: passed
```

The 74-test group covers the diagnostic and parent probe, no-update controls,
Sanmill training referee, update exclusion, checkpoint v2, exact resume,
temperature scheduling, and node calibration. The 103-test group is the
mandatory Malom, DB-teacher, and label-provenance set from `AGENTS.md`. This is
not a new full-repository test-suite claim.

The local audit invoked `preflight_probe_diagnostic(...,
require_published=False)` only to inspect the committed but unpublished source.
It returned `ready_for_authorized_minimal_diagnostic` and
`launch_authorized=false`. Repeating it with `require_published=True` stopped
before runtime construction with:

```text
SanmillRouteProbeError: probe source commit must already be published
```

## Reviewed commands

After the implementation and this readiness record are published, the final
read-only preflight command is:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --preflight `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-v1.json `
  --paths-config data\training_paths.local.json
```

The frozen execution command remains recorded in the owning
[experiment contract](../experiments/sanmill-no-update-integrated-route-diagnostic-v1.md).
It was not run and is not authorized by this evidence.

## Next gates

1. Obtain explicit authority for an ordinary push of the implementation and
   readiness commits.
2. From the resulting clean `dev == origin/dev`, repeat the exact read-only
   preflight above and require
   `status=ready_for_authorized_minimal_diagnostic` with
   `launch_authorized=false`.
3. Ask separately whether the exact one-game command may be run once.

Publication and a passing final preflight do not imply step 3. A failure,
success, interruption, or operator termination would consume any later
one-run authority; no retry or next-index escalation is automatic.
