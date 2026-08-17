# Production Specialist Malom Positional Safety

Date: 2026-08-18

Status: implemented and verified on `dev`; not a deployment claim

## Outcome

The difficulty 9/10 specialist override now scores every legal move first and
then selects the model's highest-scored move inside the Malom W/D/L-preserving
set `A_pos`. The original argmax, selected argmax, parent tier, safe-set size,
query count, elapsed time, and intervention flag are retained in diagnostics.

The filter is positional-only. It does not include repetition history or the
no-progress clock and is not `A_allow` or a full-rule safety proof. It accepts
only the tracked `sector-corrected-v1` Malom manifest and verifies the exact
component inventory and file sizes before enabling the product route.

This change is justified by 732 observed position-level self-downgrades in
9,360 active-specialist turns. The separate 24.0157 percentage-point score
difference from the lightweight measurement remains an internal descriptive
result whose frozen primary decision was `inconclusive_precision`. It is not
a user-benefit promise or the basis for this change.

## Training-to-serving feature investigation

The three checkpoint payloads all declare `move_feat_dim=134`. Their hashes
remain:

- opening: `d020e1442676e16cdced6c91dac958817c3a22a283cc293d6e19930a87703701`;
- movement: `a587ab995224a1d43c99fd2f42e4bff9c060ac6da55edcddb43a39fc07ef26d2`;
- endgame: `5de51a1afd5794374d4394cce2950957a23f02504b5c5952a062d91414b94be8`.

The 134 inputs are 62 base features plus 72 lookahead features. The encoder's
`db` argument does query Malom, but the result remains in `db_moves` for reward
or diagnostics and is explicitly not fed into the feature rows. Training and
serving both pass `db=None` at that direct encoder boundary. There is therefore
no evidence that a dedicated Malom feature dimension is present during
training and forced to zero during serving.

There is a narrower lookahead risk:

| Route | Current trainer construction | Current product construction |
| --- | --- | --- |
| opening | no early-exit DB | no early-exit DB |
| movement | `endgame_db=db` | no early-exit DB |
| endgame | Malom, else WDL DB | no early-exit DB |

The product loads `SpecialistRouter` before Malom and later calls `set_db`.
That setter does not update the already-created lookahead advisors. Thus the
current training and service code paths can differ for movement and endgame.
The old weights-only checkpoints do not preserve the original command, data,
or Malom availability, so it cannot be proved that these exact artifacts were
trained with the early-exit path. The supported conclusion is a code-backed
train/serve skew risk, not a proven artifact-level mismatch.

This change does not alter the 134 model inputs or attempt to repair that
uncertain skew. It adds a final independent safety constraint. Propagating
Malom into the lookahead would change the model's input distribution and needs
separate provenance and compatibility evidence.

## Product code and delivery boundary

The active route in the verified tree is:

1. `web/app.py` activates the specialist override for difficulty at least 9;
2. `learned_ai/agents/specialist_router.py` routes by phase and scores all legal
   moves;
3. `learned_ai/agents/positional_safety.py` performs the final `A_pos` filter.

At inspection time `dev` and `origin/dev` were at `dc745fc` before this change.
The repository default branch is `origin/main`, but its tracked tree does not
contain the three active checkpoint files. A clean `origin/main` checkout
therefore cannot reproduce the measured active-specialist path. The current
`dev` tree is the only repository-bounded, self-contained path containing the
app code, exact checkpoints, and lightweight evidence together.

No NMM Web service process was running on this host during inspection. The
repository cannot prove which branch an external production host deploys.
This delivery therefore updates and verifies `dev`; it is not represented as
already deployed elsewhere. A separate main-branch integration would be
needed if `origin/main` is the external release source.

## Failure behavior and observability

The new behavior is deliberately asymmetric:

- startup Malom absence, an untrusted manifest, or inventory mismatch leaves
  the filter disabled and logs an error;
- the status endpoint reports `playable=false`, the exact disabled reason,
  label and manifest identities, counters, the last error, and the last
  successful decision;
- the product UI starts with the specialist-player checkbox disabled and only
  enables it when the status endpoint reports the `A_pos` filter active;
- a required parent or successor query failure, mixed context, incomplete
  legal inventory, invalid score vector, or empty safe probability mass fails
  closed and returns no specialist override;
- the already-computed classical coordinator move is then used, with both a
  persistent server-log event and console output.

This keeps a user game alive without silently playing the unfiltered specialist
argmax. The classical fallback itself is not claimed to be position-safe.

Before this change, a raw router failure could result from a missing preferred
and fallback model, an empty encoding/legal inventory, or any caught encoder,
lookahead, or inference exception. Exceptions were logged by the child logger,
but a plain `None` result was only printed to stderr and there was no persistent
counter or denominator. No `server.log*` files containing such events were
present on this host. The historical production fallback rate therefore cannot
be recovered. The 29.1339-percent lightweight score applies only to the
successfully scored, fail-closed evaluation route and cannot be weighted to
unknown real-user fallback traffic.

The new in-process counters distinguish unavailable-filter requests, Malom
runtime failures, model-score failures, and successful interventions. They do
not retroactively estimate the old rate.

## Generalist scope decision

The optional `GeneralistAgent` path remains unchanged. It is not the active
difficulty 9/10 specialist route, and the repository already records that
placing retained-v4 weights in that product wrapper does not reproduce the
frozen training-aligned route. Extending this filter there would combine a
safety change with an unresolved route-identity problem. It should be handled
only after that optional product path has its own validated contract.

## Real-event regression evidence

The tracked fixture
`tests/fixtures/specialist_positional_downgrades_v1.json` samples the first
recorded D-to-L event in placement, movement, and flying from the immutable
lightweight candidate ledger. The ledger SHA-256 is
`4d5292bc8748832f01a79541cd0babef007c750b8af92900f33bc98ba83825c2`.
Each source row hash, game identity, ply, original move, and transition is
checked against the local ledger when it is present.

The live `sector-corrected-v1` tablebase independently reproduced each frozen
safe set. A read-only end-to-end load of the exact active product components
then reproduced the recorded raw argmax and applied the new filter:

| Phase | Recorded/raw downgrade | Filtered model argmax | In frozen `A_pos` |
| --- | --- | --- | --- |
| placement | `place d1` | `place d7` | yes |
| movement | `d2-d3` | `d5-c5` | yes |
| flying | `g7-d2` | `d3-c5` | yes |

All three calls recorded an intervention and no runtime or scoring failure.
This was state inference only: no complete game, search batch, training, or
database write was performed.

The source-level bypass regression also parses the product AI-turn function
and fails if it calls the unfiltered specialist scorer. The UI regression
fails if the player checkbox is not disabled by default or is not gated by
the backend `playable` state.

## Latency

Structural verification of the 512-file, 83.58 GB Malom inventory took
26.29 ms. The final filter makes one parent query plus at most one successor
query per legal move. On the three real fixtures it made 13, 6, and 55 queries.

The first query for previously unseen Malom piece-count sectors must construct
the process-local combinatorial hash state. In the full product component load,
the observed cold filter times were 2,050 ms for placement and 2,754 ms for
movement; the flying fixture used already initialized small sectors and took
6.38 ms. Repeated warm calls took approximately 0.11 to 0.86 ms across the
three fixtures.

The worst observed cold addition is 9.2 percent of the existing 30-second
difficulty-9 search and 4.6 percent of the 60-second difficulty-10 search.
That is acceptable for the current high-difficulty interaction envelope, but
it is not negligible and remains visible in every decision record. No claim is
made for lower-difficulty or generalist latency.

## Evidence discipline

### Observed facts

- The free active-specialist arm recorded 732 W/D/L downgrades in 9,360 turns.
- The three checkpoint widths are 134 and their hashes are unchanged.
- Direct Malom query output is not a model feature; lookahead early-exit can
  differ between current training and serving construction.
- The three sampled raw model argmaxes exactly reproduce the recorded unsafe
  moves, and the filter replaces each with a live-Malom `A_pos` move.
- Historical fallback frequency is not measurable from available logs.

### Interpretation

The final filter removes one directly observed mechanism: a successfully
scored specialist move cannot lower its position-only Malom W/D/L tier. It
does not prove stronger play, full-rule safety, product effect, or a correction
of the uncertain lookahead feature skew.

### Counterevidence and limits

- `A_pos` ignores repetition and the no-progress clock.
- The lightweight score contrast missed its frozen precision target.
- The fixture has three representative events, not all 732 events.
- An external deployed branch or host was not observable.
- Cold sector initialization adds measurable latency.

### Next validation

No new game experiment is required for this code change. Operational follow-up
should inspect the new status counters and log records after ordinary product
use. Any future claim about user score, latency distribution, or full-rule
safety requires a separately authorized and preregistered measurement.

## Verification

The focused and required provenance suites completed with:

```text
138 passed, 498 subtests passed
```

They included the new 12-test specialist safety file, runtime quarantine,
checkpoint loading, strict route dependencies, Malom DB, DB-teacher, and Malom
label-provenance tests. Task-scope Ruff passed for the new filter, router,
DB-teacher, and tests. The legacy `web/app.py` file has pre-existing broad Ruff
findings, so it was checked with the repository's critical `E9,F63,F7,F82`
set, which passed. Python compilation passed using a repository-local pycache
because the default test `__pycache__` directory is not writable. `git diff
--check` passed.

No checkpoint, alias, database, protected data segment, or frozen experiment
record was modified. The reserved source pool `2eb04f54` was not read or
consumed.
