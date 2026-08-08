# Sanmill Route Diagnostic Index 2 v1 Result — 8 August 2026

## Verdict

`index_two_did_not_reproduce; historical_failure_still_unlocated`

Parent schedule index 2 completed without a Sanmill/NMM mirror mismatch. The
first three immutable schedule entries have now passed individually, but this
does not exclude any later seed, deep route, node ceiling, or opponent kind.
It is no-update diagnostic evidence only.

## Frozen result

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-diagnostic-index-2-v1-20260808-001` |
| Published source | `7996df522f62b63f215f3ba08a7690dfe9a671f8` |
| Plan identity / raw SHA-256 | `cad483e133887c4b710f9fe8ccfdd44876987be650f7306d1d0b18a11b184b48` / `bbd407496204ef9af2f51367351ce4cdf4896ed05eedd9419f155b89be25ea09` |
| Report identity | `48bae4d166fe1beb151af58c68e0851e3471177d1bf5a4f061162a0af2226690` |
| Raw result SHA-256 / size | `c586c6d4cc3d25c3e223c6d25995323952df7c56312c2ed68032bbc8dbc690ca` / 31,855 bytes |
| Selected role | `sanmill-1000-normal-1-W`; seed `8933699534328013511` |
| Logical plies | 30 |
| Sanmill search calls / reported nodes | 15 / 13,051 |
| Compound turns | 4 |
| Measured wall time | 14.821290 seconds |
| Termination | `lose_no_legal_moves`; learner result `loss` |
| Mirror comparison | no mismatch |
| Model mutation | zero backward calls and checkpoint writes; learner and target digests unchanged |
| Data mutation | HumanDB and SpecialistDB identities unchanged; `quick_check=ok`; no sidecars |

The ignored raw report remains at
`out/diagnostics/sanmill-no-update-integrated-route-diagnostic-index-2-v1-20260808-001.json`.
Independent canonical recomputation reproduced its stored report identity.
The one-run authority is consumed. A new plan is required for index 3; this
result is not retry or training authority.
