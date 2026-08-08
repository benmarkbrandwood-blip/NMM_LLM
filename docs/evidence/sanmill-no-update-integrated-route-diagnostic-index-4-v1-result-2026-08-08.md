# Sanmill Route Diagnostic Index 4 v1 Result — 8 August 2026

## Verdict

`index_four_did_not_reproduce; first_deep_route_passed`

The first 1,000-node deep-route entry completed without a Sanmill/NMM mirror
mismatch. Indices 0–4 are now excluded under their exact frozen seeds and
roles. The historical failure remains unlocated.

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-diagnostic-index-4-v1-20260808-001` |
| Published source | `dc6ac1b8e6cee77dd95bcd31b469dfd24a465bef` |
| Plan identity / raw SHA-256 | `6e3ee428bd6e8cead86b736d957912b35c12529c928aad72ade795e0db19f2ed` / `9d131a72c1962181f3b8e9ddd4783693f944069aef3b1b6044457a97e63027de` |
| Report identity | `f7b11d1bdf5ee9ba862767ab8b4f1ad82274f7ea42de8ae64eb9c1b2730191f8` |
| Raw result SHA-256 / size | `d02cba21c51af388bbaeb08f16debde9c7cd8c4ca96a79a9e1d3cecd1d37ae3e` / 40,476 bytes |
| Selected role | `sanmill-1000-deep-0-W`; seed `2549456208036943109`; simulation depth 12 |
| Logical plies | 48 |
| Sanmill searches / reported nodes | 24 / 21,387 |
| Compound turns | 6 |
| Measured wall time | 9.531240 seconds |
| Termination | `lose_no_legal_moves`; learner result `loss` |
| Integrity | no mismatch; model and both databases unchanged; no update or checkpoint |

Independent canonical recomputation reproduced the stored report identity.
The raw report remains ignored under `out/diagnostics/`. The one-run authority
is consumed. Index 5 requires a new frozen plan and is the remaining
1,000-node deep-route colour counterpart. This is not training authority.
