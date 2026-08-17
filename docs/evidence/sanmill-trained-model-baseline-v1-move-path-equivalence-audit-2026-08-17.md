# Trained-model baseline move-path equivalence audit

Date: 2026-08-17

Decision: `successful_turn_state_transition_equivalent`

This audit was completed before any attempt-003 tooling change.  It compares
the direct experiment path

```text
SanmillUciSession.search_logical_turn
-> semantic validation
-> SanmillTrainingGame.apply_nmm_move
```

with `SanmillTrainingGame.search_and_apply`.  It also checks the exact path
used by the already published safe-guidance attempt-002 result.

## Frozen source inspected

- repository commit:
  `3fdd797c964861e9cf4487cdafb0cc4fa8ad684b`;
- pinned Sanmill commit:
  `a6623f88959f7453594df274fbe1f128af7ff55e`;
- `learned_ai/training/sanmill_referee.py`:
  `1ff9c0de59e2815e6c027f1391360172086b1da07642c4448ad72cc87a1d00f3`;
- `learned_ai/evaluation/sanmill_uci.py`:
  `f8e4e3984391d299e1a466860be52388bf39bfdd36d0d746dd8deb8e8e617441`;
- `learned_ai/evaluation/sanmill_safe_guidance_gameplay.py`:
  `8f781370f6bf6d1c77e55fc2849de875cfab54e378f2a5a57b78e9bd364efd48`;
- `learned_ai/evaluation/sanmill_trained_model_baseline.py`:
  `6b68cde86a9836c7d8f7d37fd3528a1814180ec8fbbcd47412be09e6c7267572`;
- pinned `logical_turn.rs`:
  `b7bcf1ba0693790e03fd3934c4e42edd34d7c2f4cd163c9948e023471615ad9e`;
- pinned UCI `mod.rs`:
  `a8372375ce703e74b14ffd683b972722595709dbc7c43a0e291610c64120fb16`.

The pinned Sanmill worktree was clean.

## Line-by-line call comparison

`search_and_apply` performs the following operations in
`sanmill_referee.py:515-527`:

1. assert the local/Sanmill board mirror;
2. reject an already terminal root;
3. call `session.search_logical_turn(node_budget, depth=depth)` inside the
   optional `sanmill_opponent_search` timing scope;
4. require `status == "ok"` and a non-null `model_action`;
5. call `apply_nmm_move(board, result.model_action,
   search_result=result)`.

The safe-guidance attempt-002 path in
`sanmill_safe_guidance_gameplay.py:953-991` and the trained-model baseline
path in `sanmill_trained_model_baseline.py:1327-1337` both:

1. call the same `session.search_logical_turn` method with the same frozen
   100,000-node budget and no depth override;
2. validate the returned typed result without changing it;
3. retain `result.model_action` unchanged;
4. call the same `apply_nmm_move(board, move, search_result=result)`.

The extra experiment operations are observational or fail-closed only:
resource reservation/accounting, Malom labeling, and
`_checked_search_result`.  `_checked_search_result` reads typed fields and a
portable serialization; it does not mutate the board, referee, search result,
or Sanmill process.

## Strict-rule state transition

Both paths enter the same `apply_nmm_move` implementation at
`sanmill_referee.py:434-506` with the same three inputs.  That implementation
is the only transition boundary and performs the following sequence:

1. validate the pre-move board mirror and reject a terminal state;
2. normalize and validate the atomic move against all local legal moves;
3. expand the move to one placement/movement token and, when applicable, one
   removal token;
4. validate the search result's full-turn tokens and atomic model action;
5. append those exact tokens to the single in-memory complete history;
6. issue `position startpos moves <complete history>`;
7. read a new authoritative `statejson` record;
8. verify action count, logical-ply count, per-color count, history identity,
   resulting FEN, terminal bit, winner, and terminal reason;
9. update `_state` and recheck the post-move local/Sanmill board mirror.

Consequently the following semantics are identical on every successful
turn:

- **No-progress clock.**  Both paths replay the same complete action history,
  and both read `no_capture_count` from the same resulting `statejson`.
- **Repetition.**  Both paths replay the same history from the same counted
  start origin under `mif-stable-moving-v1`; the repetition stack, current
  count, history length, threefold terminal bit, and history SHA-256 therefore
  have the same inputs.
- **Capture.**  A mill plus compulsory removal is kept as one logical turn but
  two action tokens in both paths.  The removal token is part of the history
  used for clock and repetition reconstruction.
- **Flying.**  Flying uses the same atomic `from`/`to` representation, the
  same local legal-move check, the same UCI token expansion, and the same full
  replay.
- **Engine state.**  `go logical` does not commit a move.  The pinned UCI
  handler builds a local `ParsedPosition` from the current state and a clone
  of its history, while `run_logical_go` builds local game, workbench, and
  searcher values and returns only a serialized response.  The handler does
  not assign the calculated child back to its live position.  Both Python
  paths subsequently synchronize the engine in the same way by replaying the
  full history through `position_startpos`.
- **Caches.**  Both call the same `go logical` implementation, which uses and
  clears the same search transposition-table route.  Neither Python wrapper
  adds a distinct engine-cache update.

## Differences outside successful transition semantics

The two APIs are not byte-for-byte equivalent in their outer failure and
instrumentation behavior:

- `search_and_apply` performs a local mirror assertion before starting the
  search; the manual path performs the same assertion at the start of
  `apply_nmm_move`, after the search has returned.
- `search_and_apply` can emit the optional `sanmill_opponent_search` timing
  observation.  The experiment paths construct the game without that timing
  observer and instead maintain their own resource ledger.
- the experiment paths apply stricter typed-result, resource, and Malom
  checks before committing the returned move.

These differences can change where an invalid turn fails and whether a bad
search has already consumed resources.  They cannot change the strict-rule
state of a successfully committed turn.  This audit therefore does not claim
total exception-trace equivalence; it establishes state-transition and
terminal-adjudication equivalence for every persisted successful turn.

## Effect on the published safe-guidance attempt-002 result

The published safe-guidance attempt-002 formal run used the same manual
`search_logical_turn` plus `apply_nmm_move` composition inspected here.  Each
persisted turn records the authoritative post-replay `no_capture_count`,
repetition count and history length, terminal bit, and terminal reason.  The
formal ledger contains 1,524 strict-rule terminals and no incomplete game.

The observation that guidance produced more threefold draws and fewer
fifty-move draws therefore does **not** require correction.  Its counts remain
109/305 for random-safe, 292/146 for full guidance, and 307/129 for geometry
guidance (threefold/fifty-move).  The observation is still descriptive and
limited to the exact frozen runtime and start pool; the present audit adds no
new product, strength, or causal claim.

No existing frozen record was modified or reinterpreted.
