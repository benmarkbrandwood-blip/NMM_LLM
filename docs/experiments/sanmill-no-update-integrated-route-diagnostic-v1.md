# Sanmill No-Update Integrated-Route Minimal Diagnostic v1

## Status and authority

Status: `local_preflight_passed_needs_publication_and_final_preflight`

This contract prepares one diagnostic game. It does not authorize that game,
another 36-game probe, a training smoke, or a long run. Silence, elapsed time,
or a successful preflight cannot supply execution authority.

The parent
[36-game probe](sanmill-no-update-integrated-route-probe-v1.md) failed closed
with `Sanmill and NMM board mirrors diverged`. Its old runner did not retain a
partial ledger, so the failing schedule index is unknown. This diagnostic
therefore does not claim that index zero caused the historical failure.

## Minimal falsifiable question

The machine-readable
[diagnostic plan](sanmill-no-update-integrated-route-diagnostic-v1.json)
selects only parent schedule index zero. It asks:

> With the original fresh model, per-game seed, learner colour, normal
> lookahead route, 1,000-node Sanmill opponent, 120-logical-ply cap, immutable
> data, and pinned referee, does the first parent schedule entry reproduce the
> authoritative board-mirror mismatch?

Index zero is the smallest exact prefix of the historical schedule. It keeps
the original game identity and avoids inventing a replacement seed or easier
position. A failure will atomically preserve the two board projections,
portable Sanmill state, complete action history, terminal fields, applied
move/search record, and the exact schedule entry. A completion establishes
only that index zero did not reproduce at the frozen source and inputs. It
does not identify which later entry failed or prove that schedule entries are
interchangeable.

## Frozen identity and work ceiling

| Field | Value |
| --- | --- |
| Parent plan identity | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` |
| Parent raw SHA-256 | `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` |
| Diagnostic plan identity | `5554489e3278dca88cc4f816e97ced1bdf17e7a89b0e4c02991c808d7087e4b0` |
| Diagnostic raw SHA-256 | `d445f95249146a54fb5cebc3e09e6dee542fbb9a7c6ea3bc313b723ff7bce04e` |
| Selected parent index | `0` |
| Game identity | `game:cfaf6392cb40a776a61370d4fc0608b5501c612b93fb531efe3d21a1940c33bf` |
| Learner / opponent | White / Sanmill |
| Route | normal, `sim_ply_depth=5` |
| Sanmill ceiling | 1,000 nodes per opponent turn |
| Maximum games | 1 |
| Maximum logical plies | 120 |
| Maximum search calls | 60 |
| Maximum requested node ceilings | 60,000 |

Every model, data, ruleset, MIF, Sanmill source, binary, strict-referee,
component-disable, and no-update identity is inherited from and verified
against the immutable parent plan. The implementation derives a one-entry
execution view and calls the same `execute_probe_schedule` production route;
it does not copy or replace gameplay, projection, replay, search, feature,
reward, or rule logic.

## Evidence behavior

- A successful game is atomically written only after the single selected
  entry completes and all model, database, source, and runtime invariants pass.
- A failure is atomically written to the distinct no-overwrite
  `<output-stem>.failure.json`; the completed-result path remains absent.
- No optimizer, backward call, checkpoint write, rollout persistence, branch,
  retry, recovery, or database write is permitted.
- No automatic escalation to another index, a longer prefix, the full probe,
  or training is permitted.
- Either outcome remains diagnostic evidence only, not throughput or playing
  strength evidence.

## Read-only preflight command

After the implementation and readiness commits are published on `dev`, the
following command is the final read-only gate. It runs only the existing
unscheduled two-ply, zero-search route check and does not consume the selected
game:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --preflight `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-v1.json `
  --paths-config data\training_paths.local.json
```

The required result is
`status=ready_for_authorized_minimal_diagnostic` with
`launch_authorized=false`. A dirty tree, unpublished `dev`, changed plan,
changed database, changed Sanmill runtime, missing CUDA, or a failed route
check is a stop condition.

## Frozen execution command — not authorized

The following command is recorded for review only. It must not be run without
a new explicit one-run authorization after the final published preflight:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_sanmill_integrated_route.py `
  --launch diagnostic `
  --plan docs\experiments\sanmill-no-update-integrated-route-diagnostic-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-diagnostic-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-diagnostic-v1-20260808-001.json
```

The output and its derived failure path must both be absent before launch.
The command is single-use if later authorized: success, failure, interruption,
or operator termination consumes that authority and does not permit a retry.

## Next gate

Publish the implementation and readiness evidence, then repeat the exact
read-only preflight from clean `dev == origin/dev`. Execution remains a
separate owner decision. No result from this diagnostic may automatically
select another schedule index or authorize the original 36-game probe.
