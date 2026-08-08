# Sanmill Frozen-Target Controls Continuation v1 Result — 8 August 2026

## Verdict

`remaining_controls_passed; no_update_route_complete`

All four remaining immutable frozen-target controls completed without a
Sanmill mirror mismatch. Together with the individually completed indices
0–6, the quarantined valid prefix 7–30, and the passing exact reproduction at
index 31, every entry in the frozen 36-game no-update schedule now has a
completed sample. This closes only the inference/referee route; it does not
validate training updates or authorize a long run.

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-frozen-target-controls-continuation-v1-20260808-001` |
| Published source | `518aa5c8ad7f53c729d31e7ba1c398c998066d45`; clean `dev == origin/dev` |
| Plan identity / raw SHA-256 | `1a554be7d9c23c9b4f4ed5f932fc6318185c7b9cd274b655196449cc27598de6` / `d953c0098355b028074d3770c223e4c4c04d2e463d540aa8658fef6a7ab83796` |
| Report identity | `4ffff8dfcbde93bfd48d42c476c52a8567de37f1215e21464e852ece50c25883` |
| Raw result SHA-256 / size | `4882a9e0d8b22f603d12f08859d9936b7eed1a10f178bd9e7eb3399e5fd7ce39` / 168,072 bytes |
| Parent range | indices 32, 33, 34, and 35 in exact order |
| Work observed | four complete games; 244 logical plies; zero opponent search calls |
| Outcomes | two learner wins and two learner losses; all ended by fewer than three pieces |
| Timing | 35.138942 total process wall seconds |
| Integrity | completed result present; failure result absent; no remaining `tgf` process |

## Per-game evidence

| Index | Role | Seed | Plies | Result | Final history SHA-256 |
| ---: | --- | ---: | ---: | --- | --- |
| 32 | `frozen-target-normal-1-W` | `4498028818635829754` | 62 | loss | `bedcdb41fe5b95520e30ca73032d849de1d84cab0176e68d4597d06d94f856c2` |
| 33 | `frozen-target-normal-1-B` | `7039821642329624856` | 40 | win | `4826b78107b7f6f216b6f166910da8b0b1aecbe73fd675e022104e7aeba38213` |
| 34 | `frozen-target-deep-0-W` | `5098805285117675881` | 58 | loss | `829063bc09f94bd0c298452cb320df9eeb2132e6f9620eea11a6cb2a0fbaec8e` |
| 35 | `frozen-target-deep-0-B` | `2266579693117331260` | 84 | win | `fd08f3397e1a935f09b71b240b276e14b3b38568143f1661061325f2d84b4a64` |

The result validator independently recomputed the report identity and
verified the exact four-entry schedule. Learner and frozen-target weights
remained byte-identical at
`15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00`.
There were zero backward calls, optimizer constructions, checkpoint writes,
or rollout persistence. HumanDB, SpecialistDB, Malom, and ruleset records were
identical before and after the run.

The two wins and two losses are deterministic route observations, not a
strength estimate. The sample was chosen to cover roles and routing, not to
estimate performance.

## Next gate

The next artifact must be a separate update-capable Sanmill-refereed training
smoke. It must prove at least one real optimizer/backward update, checkpoint
creation and loadability, exact source/data/rules/experiment identities, and
fail-closed behavior under the same referee. Its resource bound and launch
command must be frozen before execution. Long training remains unauthorized
until that smoke passes and a successor long-run readiness record is reviewed.
