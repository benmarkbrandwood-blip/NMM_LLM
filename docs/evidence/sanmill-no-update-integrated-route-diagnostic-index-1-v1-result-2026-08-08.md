# Sanmill Route Diagnostic Index 1 v1 Result — 8 August 2026

## Verdict

`index_one_did_not_reproduce; historical_failure_still_unlocated`

Parent schedule index 1 completed without a Sanmill/NMM mirror mismatch. With
the separately recorded index-0 result, both learner colours in the first
1,000-node normal-route repetition have now completed under the frozen fresh
model. This does not exclude later seeds, the deep route, higher node budgets,
or the frozen-target route, and it is not training or strength evidence.

## Execution identity

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-diagnostic-index-1-v1-20260808-001` |
| NMM_LLM commit | `6574acba40c36478f8672068a1eb3b5fec4ddabb`; clean; `dev == origin/dev` |
| Diagnostic plan identity | `298700a21e11fcff1f8789c2d6fb166af03c461f31cdc94e3adb2ed1675ffc2f` |
| Diagnostic plan raw SHA-256 | `68e22209e4ec903af2532339943d3575f7dd930ee021e8abafece883dd26e75c` |
| Sanmill commit | `a6623f88959f7453594df274fbe1f128af7ff55e` |
| Sanmill binary SHA-256 | `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619` |
| Referee semantic digest | `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` |
| Report identity | `aa8ed3c5ba58a5e1fcde48085323ac4b0193b7be2f916653a60811999b6b8de6` |
| Raw result SHA-256 | `c891537de11b42be6f1f2c3b4d559f7a2bb84b5c511a57b137503e0b5f0372b5` |
| Raw result size | 28,804 bytes |

The ignored raw report remains at
`out/diagnostics/sanmill-no-update-integrated-route-diagnostic-index-1-v1-20260808-001.json`.
Removing `report_identity` and canonicalizing the remaining object reproduced
the stored report identity exactly.

## Completed observation

| Field | Result |
| --- | --- |
| Parent scheduled index | 1 |
| Role | `sanmill-1000-normal-0-B` |
| Learner colour | Black |
| Torch seed | `2623237247545163822` |
| Node ceiling / route | 1,000 / normal depth 5 |
| Logical plies | 23 |
| Sanmill search calls / reported nodes | 12 / 9,996 |
| Compound turns | 2 |
| Measured game wall time | 12.752959 seconds |
| Termination | `lose_no_legal_moves`; learner result `loss` |
| Mirror comparison | no mismatch |
| Backward calls / checkpoint writes | 0 / 0 |
| Learner and target digests | unchanged before/after |
| HumanDB and SpecialistDB | unchanged; `quick_check=ok`; no sidecars |

The one-run authority is consumed. The next bounded step is a distinct
diagnostic selecting immutable parent schedule index 2. It must retain the
same fail-closed diagnostics and no-update boundaries; passing index 1 does
not authorize reusing this plan.
