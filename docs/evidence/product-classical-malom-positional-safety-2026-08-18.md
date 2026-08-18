# Product Classical Malom Positional Safety -- 18 August 2026

## Outcome

The `dev` Web product now resolves Malom from portable, validated inputs and
routes every adversarial AI move through one final positional safety choke.
Default classical play is constrained only at difficulties 9 and 10. Explicit
specialist and generalist routes are constrained at every difficulty because
choosing either route already opts out of the ordinary difficulty ladder.

This is a product code change, not a game experiment. No complete game,
training, weight update, checkpoint change, model-alias change, or database
write was performed. No protected data partition or source-pool record was
opened.

The gate implements `A_pos`: it preserves the mover's Malom W/D/L tier for the
pure board position. Malom does not carry repetition or no-progress history,
so this is not `A_allow` and is not full-rule safety.

## Direct reason for the change

The frozen classical-search ledger contains 117 observed position-level
self-downgrades: 61 at difficulty 9 and 56 at difficulty 10. All occurred in
placement or movement. The flying route, which was dominated by the product's
solved endgame lookup, had zero observed downgrade.

The change is justified by those observed defects and by the code path that
previously accepted a coordinator or database-bypass result without a final
Malom check. It is not justified by a forecast score gain. No user-facing gain
is predicted or promised.

## Malom resolution contract

`ai/malom_runtime.py` evaluates candidates in this order:

1. `NMM_MALOM_DB` environment override;
2. ignored `data/training_paths.local.json` key `malom_db_path`;
3. legacy shared `data/settings.json` key `malom_db_path`;
4. Sentinel `external_db_path`.

Every configured candidate is checked independently, even after a
higher-priority candidate succeeds. A candidate is accepted only when:

- the directory exists;
- the tracked manifest is readable and declares `malom_tablebase`,
  `theoretical_wdl`, the `malom_oracle` consumer, and
  `sector-corrected-v1`;
- the complete component-name inventory and every component size match;
- the strict read-only adapter opens and exposes the complete oracle surface.

The first passing candidate is selected. Later passing candidates are closed
and reported as validated but not selected. Every empty or rejected candidate
and its reason remain visible in runtime status.

The local check selected `local-registry:malom_db_path`. It verified:

- manifest identity
  `f4c52b00f00d25131a28743218a601bb34f60172970620de608c80e93ce28747`;
- content identity
  `c414fe88778f8d1d95cd3015532b43cad59f09e8398d8e46c42188b6829f3544`;
- 512 components totalling 83,582,223,577 bytes;
- the tracked shared WSL-style candidate was rejected because its path does
  not exist on this host.

Full component hashes were not recomputed at product startup; their frozen
values are bound by the manifest, while startup checks exact names and sizes.
This matches the requested startup validation without reading 83.58 GB.

The Tools page now reads and writes the Malom path only through ignored
`data/training_paths.local.json`. It no longer writes that machine-specific
value into tracked `data/settings.json`.

## Final move choke

The final helper in `web/app.py` is called by both the human-versus-AI turn and
AI-versus-AI loop immediately before the strict game engine applies the move.
It receives the original source and choice, and returns both the final move and
a structured decision record.

For specialist and generalist routes, every legal move is scored first. The
final choke preserves the model's ordering and selects its highest-scored move
inside `A_pos`. This also covers the generalist path without asserting that its
product route is equivalent to any frozen training route.

The classical coordinator has no public, source-independent root-whitelist
entry that also covers its book and database bypasses. Putting a filter inside
search would therefore miss some observed sources. When the original classic
choice is unsafe, the implementation instead performs deterministic
fixed-depth-2 root re-ranking over `A_pos` only. If that secondary re-ranking
fails after Malom has established `A_pos`, the canonical first safe move is
used and the failure is logged. An unsafe original move is never restored after
a successful Malom partition.

Lower difficulties retain their existing unfiltered behavior so perfect
position-level preservation does not collapse the difficulty ladder. The
policy scope is reported as
`difficulty-9-10-and-explicit-learned-routes`.

## Failure semantics and observability

Classic search is the last playable fallback, so unavailable Malom cannot
reject the user's game. Startup unavailability or a runtime query failure keeps
the original classical result, but records and logs that the move was
unfiltered. A specialist or generalist choice falls back to the already
computed classical result in the same condition; it does not play an
unverified learned argmax.

The following surfaces expose the selected source, all candidate outcomes,
manifest and content identities, validation result, counters, last error,
original move, final move, W/D/L tiers, query count, and elapsed time:

- `/api/sentinel_status`;
- `/api/overseer_status`;
- `/api/tool_status`;
- each WebSocket `ai_move` record.

The UI distinguishes enabled `A_pos` from visible unfiltered classic fallback.
It does not display an unavailable filter as active.

## Regression evidence and latency

The tracked fixture samples the first recorded W-to-D placement and movement
events from the 117-event ledger. Its source ledger SHA-256 is
`b27903f745bda6bcf4200308b96a978e5acb1d1c4f2230737c597e26f9a11701`.

The exact frozen `origin/main` runtime at the recorded 13,887,000-node budget
reproduced both unsafe moves. The new gate, using live
`sector-corrected-v1` Malom data, replaced both with moves in the fixture's
independently recorded `A_pos` set. The fixture covers placement and movement;
the measured flying route had no downgrade to reproduce.

The live two-case probe took 2.82 seconds in total. The cold movement sector
accounted for a maximum observed single decision of 2.653 seconds; a repeated
warm pass stayed below the one-second guard. The cold cost came primarily from
the first Malom sector access rather than safe-set root re-ranking. It is
recorded per move and is bounded inside the existing high-difficulty thinking
envelope. This is an engineering latency observation, not a game result.

## Verification

Focused product-safety tests:

```text
27 passed
```

They include invalid path and manifest cases, exact inventory rejection,
startup unavailability, runtime query failure, low-difficulty bypass, explicit
generalist coverage, restricted-root enforcement, deterministic safe fallback,
source-level final-choke coverage, status visibility, frozen-runtime
reproduction, and live-Malom fixture checks.

Required Malom and provenance group:

```text
103 passed, 498 subtests passed
```

The new resolver, safety module, and focused tests pass Ruff. The two modified
legacy entry files retain their pre-existing repository Ruff findings; the
task-scoped check ignores only those already-established categories and adds
no new Ruff finding. Python compilation of all four modified Python modules
passes.

## Claim boundary

The change prevents a class of directly observed position-level W/D/L
self-downgrades when validated Malom is available. It does not prove a score
increase, human benefit, full-rule safety, model superiority, product value,
refresh causality, or equivalence. It authorizes no training, promotion,
deployment, publication, or release.
