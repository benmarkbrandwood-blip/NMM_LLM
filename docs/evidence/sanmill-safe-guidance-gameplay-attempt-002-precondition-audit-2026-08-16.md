# Safe-guidance gameplay attempt-002 precondition audit

Date: 2026-08-16

Status: frozen before the non-evidence rehearsal

## Scope

This record implements only the four preconditions attached to the product
owner's direct authorization for attempt-002.  It does not create a gameplay
measurement marker and does not contain an experimental result.

The frozen gameplay protocol remains
`1d368c336db5f49493a2abf3c9e7d507c013d9fed3d14cd928ee988575969cc6`.
The frozen start pool remains
`385a376dd82953c23c232f34e3dd5a84e5887b978c60627657eccfa6821eb6e9`,
with membership identity
`cb84ed8180b103d7c25d56a5051fb2476047788505ed0cb9f437c39c9048fb15`.

The new attempt specification is
`1610b74f48bd0b65c164b184e154376979fce274f0c58cbbc28c8322c3cc80e4`.
It preserves both failed attempt-001 output trees byte for byte and does not
reuse either namespace.

## Attempt-001 preservation and sunk cost

The following records remain immutable:

- authorization identity `806e7b674c96ca3f5dd98067a09b6c76bda3db2cca12c75d92ba3cc5f7b495e2`;
- execution failure identity
  `7aa4f771c3871cfd48c6935167c5f8c1ad40534281bcf2069c2ad3d36a2dc55f`;
- first failed-preflight output tree identity
  `0f5e115f00f6f426f508b04a141039a881dc5224a25a8f567801a0c3b49caa37`;
- consumed execution output tree identity
  `a402bd038242d5ee8516ffbe6ddae577d0a705b29bb643753121bcb14e919f11`.

The sealed exact attempt-001 cost is 72 Sanmill single-step searches and
12,638 Malom read-only queries.  It is disclosed as sunk cost and is not
included in the new attempt-002 envelope.  The failed first game's additional
in-memory resource increment was not recoverable.  It remains unknown and is
not estimated or relabelled.

## Failed first-start visibility audit

The failed start was
`00092c974cabf05874f066b8948e791f9fdc82d84a65759da1ba78f212a643b0`.

The consumed output namespace contains only its authorization binding and
measurement-started marker.  It contains no game ledger, progress record,
completion marker, result manifest, move record, score, winner, or terminal
outcome.  The persisted failure record and captured exception contain the
start identity, schedule metadata, exception type, and code location, but no
move or result.

The strict terminal state nevertheless existed in the exited evaluator's
process memory before the packaging exception.  The available records cannot
prove that no observer could have seen that process state.  The audit therefore
fails closed: the whole start is excluded from attempt-002.

The original 1,530-game schedule is first reconstructed unchanged.  Only the
six rows for that start are then removed.  All other ordinals, game IDs, arm
assignments, colors, and random-safe seeds remain unchanged.  The formal
attempt therefore contains 254 starts and 1,524 games.  Its conservative
95-percent half-width remains 0.014758, below the unchanged 0.015 precision
limit.

## Cross-component contract audit

The execution path was audited from the frozen start through final analysis.
The former `winner` defect was one instance of a broader risk: several
components deliberately use typed runtime objects but expose differently
shaped portable records.  The repaired path now checks each boundary before
its value can enter a durable record.

| Boundary | Required contract | Fail-closed check |
| --- | --- | --- |
| Frozen state to strict replay | One legal NMM move per logical action sequence; exact FEN, action history, logical ply, and optional strict-history identity | `replay_start` and `_matching_move` reject any mismatch |
| NMM legal moves to Malom inventory | Exactly the same move multiset; valid W/D/L parent and successor values | `_checked_oracle_inventory` compares normalized move multisets and labels |
| Frozen policy to `A_pos` | Selected move is from the best positional W/D/L tier and the safe set is nonempty | `FrozenSafePolicy.choose` constructs and checks the tier-preserving set |
| UCI search to NMM move | Typed `UciLogicalTurnResult`, requested node budget, one logical ply, matching semantic record, matching action sequence, and internally additive node counts | `_checked_search_result` rejects a malformed result before application |
| Applied move to strict referee | NMM legality, action equality, board parity, history counters, and strict-referee identity | `SanmillTrainingGame.apply_nmm_move` plus `_checked_position_state` |
| Strict terminal state to result | Typed direct winner and reason agree with the nested portable `outcome` | `_strict_terminal_outcome` and `_finalize_game_record` |
| Result to durable game record | Top-level winner, candidate score, nested outcome, turn count, and termination class agree | `validate_game_record` |
| Inducement to compact analysis | Primary event fields exist at generation; 1k/100k/500k decomposition is complete before persistence or analysis | staged `validate_game_record(..., require_decomposition=True)` |
| Per-game resource use to journal | Monotone exact snapshots, chained identities, one row per completed game | `append_resource_checkpoint` writes and fsyncs before the game record |
| Journal to recovery | Complete newline, canonical row hash, chain predecessor, exact baseline, and exact increments | `load_resource_checkpoints` rejects partial or inconsistent prefixes |
| Original schedule to reduced attempt | Only the named failed start is removed; every surviving game identity is preserved | `select_schedule_excluding_starts` |
| Complete records to primary analysis | Exactly six arm/color cells for each of the exact 254 surviving start IDs | `analyze_games(..., expected_start_ids=...)` |

Focused negative tests prove that the repaired checks have discrimination:

- a genuinely different canary move is rejected;
- an old top-level-only portable terminal shape is rejected;
- a UCI result carrying the wrong node budget is rejected;
- disagreement between the top-level and nested winner is rejected;
- an inducement without its frozen budget decomposition is rejected before
  analysis; and
- a subprocess that exits via `os._exit(73)` after two completed games leaves
  both fsynced resource checkpoints exactly recoverable.

## Resource durability order

For every completed rehearsal or formal game, the writer order is fixed:

1. obtain the post-game cumulative resource snapshot;
2. append and fsync the chained resource checkpoint;
3. append and fsync the chained full game record; and
4. atomically replace the small progress record.

This order makes the resource journal authoritative even if a process exits
between completion and game serialization.  Automatic recovery or resumption
is not authorized; recoverability is evidence, not permission to continue.

## Rehearsal gate

The frozen rehearsal uses four complete games from two starts outside the
formal pool.  Two games exercise the live random-safe and full-guided paths.
One independently recorded decisive continuation exercises strict result
packaging.  One constructed threefold continuation exercises a rules-draw
path.  The output is written to a fresh namespace explicitly labelled
non-evidence.

The rehearsal must produce all four strict terminal records, at least one draw,
at least one decisive result, at least one rules draw, a recoverable four-row
resource journal, a completion record, and an analysis record.  Failure of any
condition stops attempt-002 before a new authorization record or formal
measurement marker is created.

## Claim boundary

This audit and the coming rehearsal are technical evidence only.  They do not
measure the frozen experiment, do not establish a score effect or conversion
rate, and do not support a human-trap, product, promotion, deployment,
publication, training, or release claim.
