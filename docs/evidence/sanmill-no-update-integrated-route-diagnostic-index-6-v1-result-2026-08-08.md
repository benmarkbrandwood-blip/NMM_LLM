# Sanmill Route Diagnostic Index 6 v1 Result — 8 August 2026

## Verdict

`index_six_did_not_reproduce; continuation_probe_needed`

The first 5,000-node entry completed without a mirror mismatch. Exact parent
indices 0–6 are now excluded on the current diagnostic source. Continuing with
one plan per entry would add process noise without strengthening isolation, so
the next step is a separate contiguous-range contract that starts at index 7,
stops and quarantines on the first failure, and preserves original indices.

| Field | Value |
| --- | --- |
| Run ID | `sanmill-no-update-integrated-route-diagnostic-index-6-v1-20260808-001` |
| Published source | `29488cb730a0521cc49b1bd27f95165fd9a04a27` |
| Plan identity / raw SHA-256 | `5af3a6fcc24bfbba9b4d424a181f5d9db6835c27e4e3c95c7533295a2ee3ff18` / `c474d0e8345c1f17bf3a29360cdd1a87e67050ce56772c1deb9443360fa31400` |
| Report identity | `8f94d942b2495a5d92d921acbec25657c9448225d0aa0ecf3462576f73087580` |
| Raw result SHA-256 / size | `64a45382b3a56f5fb4dce3b90cadb34ce8ea49192cf65e0656c73563e0c09f70` / 31,592 bytes |
| Selected role | `sanmill-5000-normal-0-W`; seed `701517017423886323` |
| Logical plies | 26 |
| Sanmill searches / reported nodes | 13 / 48,022 |
| Compound turns | 4 |
| Measured wall time | 8.773455 seconds |
| Termination | `lose_no_legal_moves`; learner result `loss` |
| Integrity | no mismatch; no update/checkpoint; model and data identities unchanged |

Independent recomputation reproduced the stored report identity. The one-run
authority is consumed. This is no-update diagnostic evidence, not proof that
the old failure was fixed and not training authority.
