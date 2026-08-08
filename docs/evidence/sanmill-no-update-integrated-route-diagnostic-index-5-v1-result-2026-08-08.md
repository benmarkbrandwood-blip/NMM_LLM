# Sanmill Route Diagnostic Index 5 v1 Result — 8 August 2026

## Verdict

`index_five_did_not_reproduce; complete_1000_node_layer_clear`

Parent schedule index 5 completed without a mirror mismatch. All six
1,000-node entries now pass under their exact parent identities: four normal
and two deep, balanced by learner colour. The historical failure therefore
was not reproduced in that complete layer on the current diagnostic source.

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-diagnostic-index-5-v1-20260808-001` |
| Published source | `3d5d3435aec9e3d22830c49d0a0b9b0046193eaa` |
| Plan identity / raw SHA-256 | `6c285de356556e2577062f8bc2d64c904b81d34d4e6fe71b589f74dda7d3cce4` / `5d8e189227b0b1cd8b3027e37d4be7c75eb53d68ad617d66aae62a591f5c0ac9` |
| Report identity | `a3032e96b646d63f9ceb1b110c2820c5128a1bf8bdf0a5e06c74d31cb88cdbe0` |
| Raw result SHA-256 / size | `45cb39d6f51986d0b8b94fd21a87ebbcc385af0005b237b26b601d3be6f6bc0d` / 30,604 bytes |
| Selected role | `sanmill-1000-deep-0-B`; seed `1889687500557573843`; simulation depth 12 |
| Logical plies | 29 |
| Sanmill searches / reported nodes | 15 / 13,062 |
| Compound turns | 4 |
| Measured wall time | 14.312177 seconds |
| Termination | `lose_no_legal_moves`; learner result `loss` |
| Integrity | no mismatch; model and data identities unchanged; zero update or checkpoint work |

Independent recomputation reproduced the report identity. The one-run
authority is consumed. The next exact parent entry is index 6, the first
5,000-node normal-route game. Passing the 1,000-node layer is not strength or
training evidence and does not prove the old failure was fixed.
