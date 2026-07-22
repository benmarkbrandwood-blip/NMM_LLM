# Sanmill Strict UCI Bridge Smoke v1

Date: 23 July 2026

Status: **authorized for bridge validation only; not a formal evaluation
specification and not authority for candidate-versus-baseline games**.

Related:

- [next-evaluation decision brief](dev-v4-training-aligned-evaluation-v1-decision-brief.md)
- [local path registry contract](../local-training-layout.md)
- [completed Stage-0 result](../evidence/dev-v4-stage0-result-2026-07-23.md)

## Decision and purpose

The in-repository `GameAI` is deferred as the formal playing-strength
baseline. Its compact position and current game lifecycle do not yet preserve
the complete no-capture and repetition history needed for an authoritative
referee.

The next bounded task is to validate a strict subprocess bridge to one exact
Sanmill revision. Sanmill owns the standard-NMM rule state, complete action
history, legal-action enumeration, and terminal adjudication. NMM_LLM mirrors
stable positions only to detect coordinate, move-codec, and ordinary legality
divergence. A passing mirror does not transfer rule authority back to the
current NMM_LLM `GameEngine`.

The pinned source revision for this smoke is:

```text
da922965946ab87b3b3f9eed5b170f3e01d6c473
```

The checkout is found through the ignored `sanmill_checkout` entry in
`data/training_paths.local.json`. No host-specific checkout path is committed.
The audit must reject a different `HEAD`, any tracked or untracked source-tree
change, an absent binary, or a binary without the required fail-closed build
marker.

## Strict engine contract

The bridge must start `target/release/tgf[.exe] mill uci`, complete the UCI
handshake, verify every required option is advertised, and apply these values
before search:

| Area | Required value |
| --- | --- |
| Search | `Algorithm=2` (MTD(f)); explicit `go depth 64 nodes N` |
| Concurrency | `Threads=1`; `UseLazySmp=false` |
| Randomness | `Shuffling=false`; `SearchShuffleSeed=42` |
| Clock | `MoveTimeMs=0`; no wall-clock search command |
| Optional data | perfect database and patch/trap behavior disabled |
| Non-rule draw heuristic | `DrawOnHumanExperience=false` |
| Standard pieces | 9 pieces; flying at 3; no diagonal lines |
| Variant captures | all non-standard capture variants disabled |
| Standard draws | `NMoveRule=100`; `EndgameNMoveRule=100`; threefold enabled |

All other variant values that distinguish standard Nine Men's Morris from a
Sanmill variant are also sent explicitly. Environment variables beginning
with `TGF_` are removed from the child environment so an ambient eval-weight,
database, patch, trap, SMP, or tracing setting cannot alter the run.

`nodes N` is the work ceiling. The explicit high depth prevents Sanmill's
phase/skill depth table from silently turning a requested node budget into a
much smaller depth-limited search. The bridge records both the ceiling and the
reported node count. The node count for a later formal evaluation remains a
product decision; this smoke must not freeze it.

## Random-fallback prohibition

At the pinned revision, Sanmill's optimized UCI path normally tries a depth-4
search and then `random_search` if the main search returns no move for an
ongoing position. `Shuffling=false` does not disable that failure fallback.

For this smoke the binary must therefore be built with optimization and Rust
debug assertions enabled:

```powershell
cargo --config profile.release.debug-assertions=true build --release -p tgf-cli
```

The assertion is immediately before the fallback chain. The bridge verifies
that the assertion text is present in the compiled binary. If the main search
fails, the worker assertion prevents the fallback from running; absence of a
valid `bestmove` then becomes a timeout/process failure. The bridge never
substitutes a random or locally selected move. An illegal, malformed, absent,
or unexplained `bestmove none` is a hard failure.

## Rule and codec checks

The validation report must include all of the following:

1. the pinned commit, clean-tree result, binary SHA-256, build command, UCI
   identity, advertised required options, and exact option contract;
2. focused Sanmill rule tests for the 100-ply no-capture rule, capture reset,
   full-history threefold repetition, standard legal actions, and immediate
   terminal import;
3. black-box UCI probes showing no-capture draw, threefold draw, capture
   reset, and fewer-than-three terminal behavior;
4. two fresh-process replays with identical options, seed, node budget, and
   initial history, whose moves, stable FENs, terminal result, and reported
   nodes are byte-for-byte equal after timing fields are excluded;
5. a Sanmill-versus-Sanmill bridge smoke of at most 60 complete player turns.
   A mill-forming turn includes its staged `x<square>` removal; both UCI
   searches and their node counts are recorded;
6. at every stable state, equality between Sanmill's primary legal-action set
   and the base move set projected from NMM_LLM's atomic legal moves, plus
   equality of the removal set while a Sanmill removal is pending; and
7. fixed-node timing samples for representative placement, movement, and
   flying positions. They inform, but do not select, the later formal budget.

Sanmill remains authoritative when a historical rule outcome is involved.
The compact NMM_LLM FEN contains no repetition window and no no-capture
counter. Consequently, a future formal start cannot be represented only by
that FEN if its prior history is meant to count. It must carry a replayable
Sanmill action prefix, or explicitly freeze reset-at-start semantics after
domain review.

## Stop conditions and claim boundary

Stop and publish a failed or blocked report on any source/binary identity
drift, unsupported option, process failure, protocol timeout, random-fallback
marker failure, illegal move, rule-probe failure, nondeterministic semantic
replay, legal-set divergence, malformed evidence, or resource anomaly.

The 60-turn ceiling is only an infrastructure and performance bound. Reaching
it is not a rules draw and says nothing about playing strength. This task must
not load the candidate bundle, run candidate-versus-Sanmill games, compute a
promotion interval, freeze the 64-position review corpus, or authorize a
formal evaluation.
