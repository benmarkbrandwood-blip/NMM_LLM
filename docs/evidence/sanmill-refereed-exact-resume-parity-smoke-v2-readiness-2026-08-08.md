# Sanmill Exact-Resume Parity Smoke v2 Readiness — 8 August 2026

## Verdict

`ready_for_smoke`

V2 is ready for the staged bounded execution defined by its frozen contract.
The continuous and first-segment commands may each run once after final
matching preflights on this record's published commit. Segment 0002 remains
conditional on a passing exact-resume preflight against the newly produced
segment-0001 checkpoint, with no intervening Git change.

This verdict does not authorize retained training or reuse of any smoke
checkpoint. Any failure consumes the affected invocation and stops v2.

## Gate summary

| Gate | Observed | Expected | Result |
| --- | --- | --- | --- |
| Published plan source | `f9e93a14cdc0edcd4aad69c5b599e20093cae8bf`; clean `dev == origin/dev` | Clean published source | Pass |
| V2 contract | SHA-256 `8bc7d5df0601b554ad0ef98fae0130e7ce3a6f331f3f222f41052fdd02326da0` | New isolated retry contract | Pass |
| V1 isolation | V1 outputs and databases exist only under distinct ignored names; v2 commands reference none | No checkpoint, DB, or output reuse | Pass |
| V2 outputs | Continuous, segment-0001, and segment-0002 paths absent | New isolated outputs | Pass |
| V2 SpecialistDB inputs | Two distinct 45,056-byte files; empty 0/0/0 rows; unbound; `quick_check=ok`; `sector-corrected-v1`; identical initial content SHA-256 `5a5d8eb1df4184b1ed3581258ab2490f6b1320c7f9fd8a5322affeaf2cad540d` | Empty trusted DB per route | Pass |
| Lineage | Both routes start fresh at seed 42; only segment 0002 may exact-resume v2 segment 0001 | No historical import | Pass |
| Sanmill | commit `a6623f88959f7453594df274fbe1f128af7ff55e`; binary SHA-256 `5fbf3cba4d5994fd92029713c355f0ab016683fe71cc066eca65ac515c124619`; strict-referee digest `sha256:1b2b88cf1f6a6904696d45e2707bd55559ac47e6991edd99a95a8d6cac0b1a94` | Exact clean strict runtime | Pass |
| Data/rules | Corrected Malom identity `f4c52b00...`; HumanDB Malom labels masked; rules semantic digest `sha256:52f6ad24...` | Frozen trusted inputs | Pass |
| Training | A2C, update every 8 steps, batch 1, depth 5, 120-ply ceiling, finite fixed-node Sanmill, all optional legacy components disabled | Exact v2 contract | Pass |
| Work | Four games, at most 480 logical plies, two 1,000-node Sanmill-opponent games and two frozen-target games | Bounded infrastructure smoke | Pass |
| Regression | Runtime-shaped exact resume in a fresh lineage now passes; weights-only import into a fresh experiment ID remains blocked | Narrow policy fix | Pass |
| Verification | 67 focused tests; 103 mandatory provenance tests and 498 subtests; focused Ruff; diff check | Proportionate green evidence | Pass |
| Fresh preflights | Both returned `ready_for_smoke`, `errors=[]`, `unresolved_decisions=[]` | Passing initial gates | Pass |
| Exact-resume preflight | Requires v2 segment-0001 checkpoint | Must pass later on the same source | Conditional |

## Initial preflight identities

The uninterrupted route produced config SHA-256
`92ded33f05221238cf5c3f2ea0f860808895859dba4bd70f4a9fdc0164d49f2c`,
resume-config SHA-256
`35988c4224c8a2ee9f7e65cb2efa859d8de63cd9fd5b39ea2e5315130c9fc307`,
and experiment digest
`sha256:02f415190a9b81af00f0880af062b39ef95a919f3b1f3c8188dbb3040788782f`.

Segment 0001 produced config SHA-256
`a8b3ca753239b6c58594f7bf62ff06f5ae41bdf12b70eca99702f46728ddc9a0`,
resume-config SHA-256
`c1e9ac93d821decb2e31ac0947edbf1f87f6cca61390bd6ec9e89c39980945d7`,
and experiment digest
`sha256:6dbe821a971cdbb4040c1f675b47d85d0c94b58327e42f791293ab2ea21e314d`.

The different database paths intentionally produce different route identities;
the selected database has no gameplay effect. Segment 0002 must reproduce the
segment-0001 resume-config and experiment identities exactly.

## Required execution order

The exact commands are frozen in
[the v2 contract](../experiments/sanmill-refereed-exact-resume-parity-smoke-v2.md).

1. Publish this readiness record.
2. Rerun both fresh preflights on the resulting clean commit.
3. Launch the uninterrupted reference once and segment 0001 once.
4. Check periodic updates, lifecycle, checkpoint integrity, and both databases.
5. Without changing Git, run the frozen segment-0002 preflight.
6. Launch segment 0002 once only if the verdict is `ready_for_smoke` with no
   errors or unresolved decisions.
7. Run the frozen semantic verifier and preserve success or failure evidence.

No retry, path substitution, checkpoint substitution, or silent parameter
change is permitted within this authority.
