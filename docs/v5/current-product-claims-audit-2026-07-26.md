# Current Product Rule-Claims Audit

Status: governing correction of current public rule-history wording until E0
accepts one authoritative rules path.

## Scope

This audit concerns the repository's current Web/`GameEngine` product, not the
pinned Sanmill strict bridge and not a future v5 formal runtime. The bridge is
accepted infrastructure evidence at its recorded revision; it has not replaced
the production game loop.

The reviewed public surfaces were:

- the feature claims in `README.md`;
- Web templates and static sources;
- current game-loop comments, terminal reasons, and serialised state.

No additional threefold-repetition or no-capture claim was found in the
current Web templates or static sources.

## Verified Current Behaviour

The board and rules modules implement the core placement, movement, flying,
Mill, removal, mobility, and material-terminal mechanics used by the product.
The following history-dependent behaviour is narrower than the former README
claim:

1. `GameEngine` recognises only one fixed two-player back-and-forth pattern
   over the latest eight half-moves. It does not count general occurrences of
   an equivalent decision state.
2. `_post_placement_moves` increments on every half-move after placement and
   is not reset after a capture. It is therefore not a correct implementation
   of a no-capture counter.
3. `BoardState.to_fen_string()` serialises board occupancy, side to move, and
   placed-piece counts only. It cannot restore repetition multiplicity,
   no-progress state, claims, or terminal history.

The former README statements “Full Nine Men's Morris rules” and “Draw by
threefold repetition, 50-move rule” therefore exceeded current implementation
evidence.

## Permitted Wording

Current public material may state that the product supports:

- core board play: placement, movement, flying, Mills, removals, and ordinary
  material/mobility terminal outcomes;
- a compatibility detector for one fixed eight-half-move oscillation pattern;
- an automatic draw after 100 post-placement half-moves under the current
  compatibility counter; and
- mutual draw agreement.

It must not call the compatibility mechanisms complete threefold repetition,
a correct no-capture/50-move rule, complete standard adjudication, or an
authoritative full-history referee.

## Closure

Restoring stronger wording requires E0 acceptance of:

- the signed rules variant and normative draw semantics;
- a complete history-bearing state and replay format;
- general repetition-equivalence counting;
- the correct no-progress increment and capture/reset transitions;
- terminal precedence and claim semantics;
- independent boundary/property tests and differential evidence; and
- use of that accepted path by the actual product host.

Until then, this is a claims correction rather than a claim that the missing
semantics have been implemented.
