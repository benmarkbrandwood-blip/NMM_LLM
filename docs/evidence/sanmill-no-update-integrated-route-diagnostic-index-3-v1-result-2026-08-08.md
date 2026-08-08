# Sanmill Route Diagnostic Index 3 v1 Result — 8 August 2026

## Verdict

`index_three_did_not_reproduce; normal_1000_group_clear`

Parent schedule index 3 completed without a mirror mismatch. Together with
indices 0–2, every entry in the 1,000-node normal-route group has now passed
individually. The historical failure remains unlocated; the deep route and all
higher node groups remain untested by this sequence.

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-diagnostic-index-3-v1-20260808-001` |
| Published source | `e9fd8e555d7b36c3b078972c439801e6e1223d01` |
| Plan identity / raw SHA-256 | `6f9b92a64e60aa3e5fa140dd82161c52ef05b074c6e2797449d9b885f458238a` / `4ddea695d0396c4ea4d47b58843d90029d11268463a9acf07b579794b5d1a2c9` |
| Report identity | `a26af04950900388fcef8a62700884d18f111ce6e5bb054646ed358861637c1b` |
| Raw result SHA-256 / size | `2ba9352e9536ff1dd8063a453cf44b6b669f49d92fe973b003a75c5adb8aa1fc` / 31,734 bytes |
| Selected role | `sanmill-1000-normal-1-B`; seed `8050135631487067790` |
| Logical plies | 31 |
| Sanmill searches / reported nodes | 16 / 14,061 |
| Compound turns | 5 |
| Measured wall time | 13.007150 seconds |
| Termination | `lose_no_legal_moves`; learner result `loss` |
| Integrity | no mirror mismatch; no backward call or checkpoint; model and data identities unchanged |

The ignored raw report remains under `out/diagnostics/`. Independent
canonical recomputation reproduced its report identity. The one-run authority
is consumed. The next diagnostic must use a new plan for parent index 4, the
first deep-route entry; this record is not training authority.
