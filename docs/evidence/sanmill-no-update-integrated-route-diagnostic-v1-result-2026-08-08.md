# Sanmill No-Update Integrated-Route Diagnostic v1 Result — 8 August 2026

## Verdict

`index_zero_did_not_reproduce; historical_failure_still_unlocated`

The bounded diagnostic completed parent schedule index 0 without a Sanmill/NMM
mirror mismatch. This excludes only that exact game under the frozen source,
model, data, rules, runtime, seed, colour, search budget, and route-depth
identities below. It does not establish which entry failed in the earlier
36-game probe, prove that the defect disappeared, measure strength or
throughput, or authorize training.

## Execution identity

| Field | Value |
| --- | --- |
| Completed run ID | `sanmill-no-update-integrated-route-diagnostic-v1-20260808-002` |
| NMM_LLM commit | `90d4ecd3f72d8f6b72d88576d00972083991eb71`; clean; `dev == origin/dev` |
| NMM_LLM tree | `9247bcb92043c4be4bf77f097f88dba40c504f61` |
| Parent plan identity | `7aa079dcf59556d35f3fbd58072b6baa8c9f798e087cc336e29c8309552e3cfb` |
| Diagnostic plan identity | `5554489e3278dca88cc4f816e97ced1bdf17e7a89b0e4c02991c808d7087e4b0` |
| Diagnostic plan raw SHA-256 | `d445f95249146a54fb5cebc3e09e6dee542fbb9a7c6ea3bc313b723ff7bce04e` |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Sanmill runtime identity | `705eabcc3ff7a878071737b7dde19f22a94ac5c32aab177812667267cadde5ea` |
| Referee semantic digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |
| HumanDB SHA-256 | `97be7152573815180df6950b6150c667b1e5c2c8b1b21748b3ed9cf020b6f93c` |
| SpecialistDB SHA-256 | `b4d522d23720ab86013bbadd6b3414fba1e205ab988f3949c6ed417dca486b7f` |
| Report identity | `c7f963cf6bd075e11fac81a3f7fcb10f81280272d639581058fe9ef49c263f87` |
| Raw result SHA-256 | `da736e15f7862a30ba3cf2057e01c33559c5c7ecc419fecf649b0765722d4ef6` |
| Raw result size | 30,788 bytes |

The ignored raw report remains at
`out/diagnostics/sanmill-no-update-integrated-route-diagnostic-v1-20260808-002.json`.
Independent recomputation after the run removed its `report_identity` member,
canonicalized the remaining object, and reproduced the stored report identity.

## Launch handling

The final published-source preflight returned
`ready_for_authorized_minimal_diagnostic`. It bounded the work to one complete
game, 120 logical plies, 60 Sanmill searches, and 60,000 requested node
ceilings, with no optimizer or model update.

An initial invocation using run ID suffix `-001` was terminated by the outer
command harness after 2.867 seconds because the harness timeout was set too
low. Post-termination inspection found no result, no failure manifest, and no
remaining TGF process. That invocation is not represented as a game result.
After this explicit inspection, the delegated technical launch authority was
used for one replacement invocation with suffix `-002` and an adequate outer
timeout. The replacement authority is consumed; this plan must not be run
again.

## Completed observation

The replacement run started at `2026-08-08T08:10:10.193024Z` and completed at
`2026-08-08T08:10:29.060657Z`. The measured game itself took 15.987730 seconds.

| Field | Result |
| --- | --- |
| Parent scheduled index | 0 |
| Role | `sanmill-1000-normal-0-W` |
| Learner colour | White |
| Torch seed | `5741917532047058806` |
| Node ceiling | 1,000 |
| Route depth | normal; `sim_ply_depth=5` |
| Logical plies | 26 |
| Sanmill search calls | 13 |
| Sanmill reported nodes | 11,024 |
| Compound turns | 4 |
| Termination | `lose_no_legal_moves`; Black won |
| Mirror comparison | no mismatch |
| Backward calls / checkpoint writes | 0 / 0 |
| Learner digest before/after | identical: `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00` |
| Frozen digest before/after | identical: `15106b6e15f419c60a526fddd3851be8733b042074c791885b42252a69c1af00` |
| HumanDB / SpecialistDB before/after | unchanged; both `quick_check=ok`; no sidecars |

The HumanDB warning during launch is expected: its unversioned historical
Malom columns remain disabled while its human frequency data remains eligible.
It is not a missing required component or an implicit fallback.

## Next gate

The historical failure index remains unknown because the original runner did
not publish a partial schedule ledger. The elapsed time of that failed run is
not sufficient to name an entry. The next diagnostic should preserve the same
parent plan and execution route while selecting parent schedule index 1 as a
new one-game, no-update diagnostic. The diagnostic-plan validator currently
hard-codes index 0, so it must first be generalized with a focused failing
test to validate an explicitly selected parent entry by its immutable schedule
identity. This is diagnostic infrastructure only; gameplay, referee,
projection, search, and model-update semantics must remain unchanged.
