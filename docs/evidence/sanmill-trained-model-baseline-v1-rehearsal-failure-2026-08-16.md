# Trained-Model Baseline Rehearsal Failure (2026-08-16)

## Outcome

The required non-evidence rehearsal failed closed before its first complete
game. No formal measurement marker was created, no formal game was started,
and no candidate score or outcome is available.

The frozen protocol identity is
`35a27d27eefefbfe9c3d93a2f23615ffcd3bfe0fab1404cfd9269c1dadd2107f`.
The direct authorization identity is
`3f30c558ec5710fa96392ec92af4fb3da300c5d04ca0a925de1ee9518fd5a54f`.
That authorization remains unconsumed by its marker rule, but its
implementation binding no longer matches the repaired source. It cannot be
used for a later execution.

The machine-readable failure identity is
`76f7256d03c7265295f2d4f00b9a8ae492c137deaa0d034a6b530b1734a2a615`.

## Failure

The first live case entered the retained-v4 free arm. The positional oracle
inventory completed 22 Malom queries. The retained-v4 lookahead then
completed one additional Malom query and its counting proxy called
`ResourceLedger.add_malom()` without the required `count` argument. Python
raised `TypeError` before a move was returned.

This was an evaluation-tool defect. It was not a model result, a Malom label
failure, a Sanmill search result, or a strict-referee outcome.

The failed attempt consumed exactly 23 Malom read-only queries, zero engine
single-step searches, and zero complete games. The exception occurred before
the first per-game resource checkpoint, so exact active time was not durably
recorded. The complete command wall time, 8.057233 seconds, is charged as a
conservative active-time bound.

The failed namespace is
`out/evaluation/sanmill-trained-model-baseline-v1-rehearsal-20260816-001`.
It contains only the `NON-EVIDENCE.json` marker and the pre-game resource
baseline. Both files must remain byte-for-byte unchanged and the namespace
must never be reused.

## Reproduction and repair

A focused regression first reproduced the exact failure:

`test_counting_malom_proxy_records_each_completed_query`

The smallest repair passes an explicit count of one after each completed
proxy query. The test then passed, followed by all 13 task-focused tests and
task-scope Ruff.

The authorized implementation file had SHA-256
`71cc35dd6ad2992b4019e9eee85e1c043054486454f587e1aed4919e6c44460c`.
The repaired file has SHA-256
`f0a5fb20b5b5e3684adc87cae8d7bd985d94b9a5a0984e3e41c67c1bb1eb4df2`.
This deliberate difference is why the old authorization cannot be used.

## Disposition

There will be no automatic retry, continuation, or formal execution. A new
direct product-owner authorization must bind the repaired implementation and
a fresh rehearsal namespace before another rehearsal may begin. The new
authorization must also state how the failed attempt's 23 Malom queries and
8.057233-second conservative active-time charge relate to its resource
envelope.

Protected selection, confirmation, final-test, and research-confirmation
content remained unopened. The remaining 108 records in source pool
`2eb04f54` were neither read nor consumed. No database was written, no model
was fitted or updated, and no checkpoint or alias was changed.

This failure yields no playing-strength, training-value, promotion,
deployment, publication, or release conclusion.

