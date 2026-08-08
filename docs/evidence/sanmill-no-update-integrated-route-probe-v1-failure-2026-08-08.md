# Sanmill No-Update Integrated-Route Probe v1 Failure — 8 August 2026

## Verdict

`fatal_stop`

The separately authorized 36-game no-update probe failed closed and exited
with code 1. Its one-run authority is consumed. It was not retried, no
completed result was published, and this record is not authority for another
probe, diagnostic replay, training smoke, or long run.

## Frozen execution identity

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-probe-v1-20260808-001` |
| NMM_LLM commit | `98dcf23129a22e93298f62d04085a3fb4c2e9d9d`; clean; `dev == origin/dev` |
| NMM_LLM tree | `0b5a16ba63d45da5c663ed0bd88902d4dd3adc03` |
| Plan identity | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` |
| Plan raw SHA-256 | `b422ea44fad5fa4181294a4bc3d6e77a48692b43672ccc2ed6976d87f1aabe36` |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill tree | `17b9b0fd51ee8dac54c0454a6935978a47d19e0c` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Referee semantic digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |
| HumanDB SHA-256 | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c` |
| SpecialistDB SHA-256 | `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f` |

The final preflight at this exact published source returned
`ready_for_authorized_probe`. It verified the immutable plan, CUDA device,
fresh model, absent optimizer, pinned runtime and rules, corrected Malom,
closed database snapshots, and an unscheduled two-ply no-search route. The
owner then authorized exactly the command below once.

## Exact command

```powershell
.\.venv\Scripts\python.exe scripts\probe_sanmill_integrated_route.py `
  --launch probe `
  --plan docs\experiments\sanmill-no-update-integrated-route-probe-v1.json `
  --paths-config data\training_paths.local.json `
  --run-id sanmill-no-update-integrated-route-probe-v1-20260808-001 `
  --output out\diagnostics\sanmill-no-update-integrated-route-probe-v1-20260808-001.json
```

No argument, schedule entry, identity, or output path was changed.

## Failure

The production rollout raised:

```text
learned_ai.evaluation.sanmill_uci.SanmillBridgeError:
Sanmill and NMM board mirrors diverged
```

The failing call chain was:

```text
execute_probe_schedule
  -> _rollout
  -> SanmillTrainingOpponent.choose_move
  -> SanmillTrainingGame.search_and_apply
  -> SanmillTrainingGame.apply_nmm_move
  -> assert_current_board(board.apply_move(normalised))
```

This establishes that a Sanmill search result had passed search/replay checks
and was being compared with the NMM mirror after applying the same normalized
complete logical turn. It does not establish which state field differed or
why. The exception currently omits both projected FENs, terminal fields,
history identity, logical-ply count, actions, and schedule identity.

The runner retains completed samples only in memory and atomically publishes
only after all 36 games pass. Consequently, the failed process left no valid
partial ledger from which to determine the number of completed games or the
failing schedule index. Elapsed time is not sufficient evidence for either.
This record therefore does not guess that the first schedule entry failed.

## Post-failure audit

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Completed output | absent | no completed report after failure | pass |
| Sanmill processes | zero remaining | context-managed shutdown | pass |
| Git | clean published `98dcf23` | no source drift | pass |
| HumanDB | frozen SHA-256; no sidecars | immutable read source | pass |
| SpecialistDB | frozen SHA-256; no sidecars | immutable empty corrected source | pass |
| Post-failure preflight | `ready_for_authorized_probe` | inputs and basic route remain intact | pass |
| Full scheduled result | unavailable | 36 valid ordered samples | fatal stop |

The post-failure preflight reconstructed the same deterministic fresh model
digest and again observed no optimizer and `requires_grad=false`. Because the
failed process exited before its end-of-run model checks, that reconstruction
is not falsely presented as a direct after-digest of the terminated process.
No update-capable object existed in the route, but a complete result cannot be
claimed.

## Diagnosis boundary and next gate

The immediate cause is an observed authority/mirror mismatch. The root cause
is not yet known. In particular, this evidence cannot distinguish a board
occupancy, side-to-move, placement-counter, phase-transition, compound-turn,
or rules-terminal projection difference.

Before requesting any replay, the implementation should:

1. attach host-path-free structured context to the existing fail-closed
   mirror error, including both compact board projections, Sanmill FEN,
   complete actions, terminal fields, history SHA-256, logical-ply count, and
   schedule identity;
2. atomically quarantine a failure manifest that includes all already
   completed sample identities and the failing schedule entry without ever
   labelling it a completed report; and
3. add focused regressions proving that diagnostics do not weaken comparison,
   change gameplay, or expose a random fallback.

Only after those changes are independently reviewed and published may a new,
separately authorized minimal diagnostic reproduction be proposed. A second
36-game probe must not be the first diagnostic step.
